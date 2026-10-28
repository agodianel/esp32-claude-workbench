"""Tests for the local FTS5 docs index."""

from pathlib import Path

import pytest

from esp32_workbench import docs


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ESP32WB_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_chunk_text_overlaps():
    words = [f"w{i}" for i in range(700)]
    chunks = docs.chunk_text(" ".join(words), size=300, overlap=50)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w250"


def test_index_and_search():
    conn = docs.connect()
    docs.add_document(
        conn,
        [
            "ADC2 is also used by Wi-Fi so readings can fail " * 3,
            "I2C master clock stretching " * 5,
        ],
        "ESP32-S3 TRM",
        "https://example.invalid/s3.pdf",
        "esp32s3",
    )
    hits = docs.search_fts("ADC2 Wi-Fi", conn=conn)
    assert hits and hits[0].page == 1 and hits[0].source == "ESP32-S3 TRM"
    assert docs.search_fts("ADC2", target="esp32c3", conn=conn) == []
    assert docs.search_fts("ADC2", target="esp32s3", conn=conn)
    conn.close()
    assert docs.index_exists()
    assert "page=1" in docs.format_hits("ADC2", docs.search("ADC2", mode="fts"))


def test_reindexing_replaces_source():
    conn = docs.connect()
    docs.add_document(conn, ["first version text about uart " * 4], "Doc", "u")
    docs.add_document(conn, ["second version text about spi " * 4], "Doc", "u")
    assert not docs.search_fts("uart", conn=conn)
    assert docs.search_fts("spi", conn=conn)


def test_query_with_fts_syntax_is_safe():
    conn = docs.connect()
    docs.add_document(conn, ["gpio matrix routing details here " * 4], "Doc", "u")
    assert docs.search_fts('gpio" OR (NEAR', conn=conn)
    assert docs.search_fts("!!!", conn=conn) == []


def test_not_indexed():
    assert not docs.index_exists()


def test_builtin_sources_are_https_pdfs():
    sources = docs.load_sources()
    assert sources and all(s["url"].startswith("https://www.espressif.com/") for s in sources)


def test_build_index_with_local_file(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sensor.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        docs, "extract_pdf_pages", lambda p: ["bme280 humidity sensor register map " * 4]
    )
    lines = []
    assert docs.build_index([pdf], builtin=False, log=lines.append) > 0
    assert docs.search_fts("bme280")[0].source == "sensor.pdf"
