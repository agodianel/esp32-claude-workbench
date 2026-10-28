"""Local search over Espressif datasheets and Technical Reference Manuals.

Default backend is SQLite FTS5 (standard library, BM25 ranking). Indexing PDFs
needs ``pypdf`` (``pip install esp32-workbench[docs]``). An optional semantic
backend (``[semantic]`` extra) adds sentence-transformer embeddings in FAISS; chunk
text always lives in SQLite, so nothing is unpickled at search time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS = 40
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class DocHit:
    """One search result."""

    source: str
    page: int
    url: str
    target: str
    text: str
    score: float


def cache_dir() -> Path:
    """Return the index cache directory (``$ESP32WB_CACHE`` or the XDG cache dir)."""
    if env := os.environ.get("ESP32WB_CACHE"):
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "esp32-workbench"


def db_path() -> Path:
    return cache_dir() / "docs.sqlite"


def load_sources() -> list[dict]:
    """Return the built-in document source list."""
    data = resources.files("esp32_workbench") / "data" / "docs_sources.json"
    return json.loads(data.read_text(encoding="utf-8"))["sources"]


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (and create if needed) the docs database."""
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
        "text, source UNINDEXED, page UNINDEXED, url UNINDEXED, target UNINDEXED, "
        "tokenize='porter unicode61')"
    )
    return conn


def add_document(
    conn: sqlite3.Connection, pages: list[str], source: str, url: str, target: str = ""
) -> int:
    """Index the text of one document (one string per page). Returns chunks added."""
    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    rows = [
        (chunk, source, page_num, url, target)
        for page_num, page_text in enumerate(pages, start=1)
        for chunk in chunk_text(page_text)
    ]
    conn.executemany(
        "INSERT INTO chunks (text, source, page, url, target) VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    return len(rows)


def extract_pdf_pages(path: Path) -> list[str]:
    """Extract text per page from a PDF (requires ``pypdf``)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Indexing PDFs needs pypdf: pip install 'esp32-workbench[docs]'"
        ) from exc
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it is already cached."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "esp32-workbench"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URLs
        dest.write_bytes(response.read())
    return dest


def build_index(
    extra_files: list[Path] | None = None,
    builtin: bool = True,
    semantic: bool = False,
    log=print,
) -> int:
    """Download the built-in sources (optionally) plus local PDFs and index them."""
    conn = connect()
    total = 0
    jobs: list[tuple[str, str, str, Path | None]] = []
    if builtin:
        for src in load_sources():
            jobs.append((src["name"], src["url"], ",".join(src.get("targets", [])), None))
    for f in extra_files or []:
        jobs.append((f.name, f.resolve().as_uri(), "", f))

    for name, url, target, local in jobs:
        try:
            pdf = local or download(url, cache_dir() / "pdfs" / url.rsplit("/", 1)[-1])
            added = add_document(conn, extract_pdf_pages(pdf), name, url, target)
            total += added
            log(f"indexed {added:5d} chunks  {name}")
        except Exception as exc:  # keep going: one bad source must not stop the rest
            log(f"skipped {name}: {exc}")

    if semantic:
        build_semantic_index(conn, log)
    conn.close()
    return total


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    terms = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(f'"{t}"' for t in terms)


def search_fts(
    query: str, top_k: int = 5, target: str | None = None, conn: sqlite3.Connection | None = None
) -> list[DocHit]:
    """BM25 keyword search."""
    fts = _fts_query(query)
    if not fts:
        return []
    own = conn is None
    conn = conn or connect()
    sql = "SELECT source, page, url, target, text, bm25(chunks) FROM chunks WHERE chunks MATCH ?"
    params: list = [fts]
    if target:
        sql += " AND (target = '' OR instr(target, ?) > 0)"
        params.append(target)
    sql += " ORDER BY bm25(chunks) LIMIT ?"
    params.append(top_k)
    rows = conn.execute(sql, params).fetchall()
    if own:
        conn.close()
    return [DocHit(r[0], r[1], r[2], r[3], r[4], -r[5]) for r in rows]


def build_semantic_index(conn: sqlite3.Connection, log=print) -> None:  # pragma: no cover
    """Embed every chunk and write a FAISS index keyed by SQLite rowid."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = conn.execute("SELECT rowid, text FROM chunks").fetchall()
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([r[1] for r in rows], normalize_embeddings=True)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(np.asarray(vectors, dtype="float32"), np.array([r[0] for r in rows]))
    faiss.write_index(index, str(cache_dir() / "docs.faiss"))
    log(f"semantic index: {len(rows)} vectors")


def search_semantic(query: str, top_k: int = 5) -> list[DocHit]:  # pragma: no cover
    """Embedding search (needs the ``[semantic]`` extra and ``docs index --semantic``)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(cache_dir() / "docs.faiss"))
    vector = SentenceTransformer(EMBEDDING_MODEL).encode([query], normalize_embeddings=True)
    scores, ids = index.search(vector, top_k)
    conn = connect()
    hits = []
    for score, rowid in zip(scores[0], ids[0]):
        if rowid == -1:
            continue
        row = conn.execute(
            "SELECT source, page, url, target, text FROM chunks WHERE rowid = ?", (int(rowid),)
        ).fetchone()
        if row:
            hits.append(DocHit(*row, float(score)))
    conn.close()
    return hits


def semantic_available() -> bool:
    if not (cache_dir() / "docs.faiss").is_file():
        return False
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def search(
    query: str, top_k: int = 5, target: str | None = None, mode: str = "auto"
) -> list[DocHit]:
    """Search the local docs index. ``mode`` is ``auto``, ``fts`` or ``semantic``."""
    if mode == "semantic" or (mode == "auto" and semantic_available()):
        return search_semantic(query, top_k)
    return search_fts(query, top_k, target)


def index_exists() -> bool:
    if not db_path().is_file():
        return False
    conn = sqlite3.connect(db_path())
    try:
        return conn.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


NOT_INDEXED = (
    "ESP32 docs index not found. Build it once with:\n"
    "  pip install 'esp32-workbench[docs]' && esp32wb docs index\n"
    "Until then, consult https://docs.espressif.com/projects/esp-idf/en/stable/ directly."
)


def format_hits(query: str, hits: list[DocHit]) -> str:
    """Format search hits as markdown with citations."""
    if not hits:
        return f'No matches for "{query}".'
    lines = [f'# ESP32 Docs: "{query}"', ""]
    for i, hit in enumerate(hits, start=1):
        lines.append(f"## [{i}] {hit.source} — page {hit.page}")
        lines.append(f"Source: {hit.url}#page={hit.page}")
        lines.append("")
        lines.append(hit.text)
        lines.append("")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``esp32wb docs`` subcommands."""
    sub = parser.add_subparsers(dest="docs_command", required=True)
    idx = sub.add_parser("index", help="Download and index datasheets/TRMs")
    idx.add_argument("--file", type=Path, nargs="*", default=[], help="Extra local PDFs to index")
    idx.add_argument("--no-builtin", action="store_true", help="Only index --file PDFs")
    idx.add_argument("--semantic", action="store_true", help="Also build embeddings ([semantic])")
    find = sub.add_parser("search", help="Search the index")
    find.add_argument("query")
    find.add_argument("--top-k", type=int, default=5)
    find.add_argument("--target", help="Prefer documents for this target")
    find.add_argument("--mode", choices=["auto", "fts", "semantic"], default="auto")
    find.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Run a docs subcommand and return the exit code."""
    if args.docs_command == "index":
        total = build_index(args.file, not args.no_builtin, args.semantic)
        print(f"Indexed {total} chunks into {db_path()}")
        return 0 if total else 1

    if not index_exists():
        print(NOT_INDEXED)
        return 1
    hits = search(args.query, args.top_k, args.target, args.mode)
    if args.json:
        print(json.dumps([asdict(h) for h in hits], indent=2))
    else:
        print(format_hits(args.query, hits))
    return 0


def main() -> None:
    """Standalone entry point."""
    parser = argparse.ArgumentParser(prog="esp32wb docs", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))
