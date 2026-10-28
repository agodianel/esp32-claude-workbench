"""Zero-dependency MCP server (stdio transport) exposing the workbench tools.

Implements the subset of the Model Context Protocol that tool servers need:
``initialize``, ``ping``, ``tools/list`` and ``tools/call`` over newline-delimited
JSON-RPC 2.0 on stdin/stdout. Running it needs only ``python3`` — no ``pip install``
— so the same server works from the Claude Code, Codex, Cursor, Copilot, Gemini CLI
and Antigravity plugin/config formats.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from esp32_workbench import (
    __version__,
    buildlog,
    chips,
    crash,
    docs,
    doctor,
    logs,
    mission,
    partitions,
    pins,
    sdkconfig,
    stacks,
)
from esp32_workbench.contract import generate_contract

SERVER_NAME = "esp32-workbench"
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "ESP32 firmware tools. Call chip_info or pin_audit before writing GPIO code, "
    "search_esp_docs before peripheral/ADC/Wi-Fi/BLE code, project_doctor for a full "
    "project health check, and crash_decode / build_log_triage when debugging. "
    "Pass absolute paths."
)


@dataclass
class Tool:
    """An MCP tool backed by a Python function returning markdown text."""

    name: str
    description: str
    schema: dict
    func: Callable[..., str]


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


STR = {"type": "string"}
TARGET = {"type": "string", "enum": chips.list_targets(), "description": "ESP-IDF target"}


def _read_text(text: str | None, path: str | None) -> str:
    if text:
        return text
    if path:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    raise ValueError("Provide either 'text' or 'path'")


def _pin_audit(path: str, target: str | None = None) -> str:
    return pins.scan(Path(path), target)["report"]


def _sdkconfig_review(path: str) -> str:
    return sdkconfig.review(Path(path))["report"]


def _partition_check(
    csv_path: str, sdkconfig_path: str | None = None, app_bin: str | None = None
) -> str:
    return partitions.check_file(
        Path(csv_path),
        Path(sdkconfig_path) if sdkconfig_path else None,
        Path(app_bin) if app_bin else None,
    )["report"]


def _stack_check(path: str) -> str:
    return stacks.format_report(*stacks.scan_directory(Path(path)))


def _log_summary(text: str | None = None, path: str | None = None) -> str:
    return logs.format_summary(logs.summarize_logs(_read_text(text, path)))


def _crash_decode(
    text: str | None = None,
    path: str | None = None,
    elf: str | None = None,
    target: str | None = None,
) -> str:
    report = crash.parse_panic(_read_text(text, path))
    if elf:
        crash.decode_frames(report, Path(elf), target)
    return crash.format_report(report)


def _build_log_triage(text: str | None = None, path: str | None = None) -> str:
    return buildlog.format_report(buildlog.triage(_read_text(text, path)))


def _chip_info(target: str) -> str:
    from esp32_workbench.cli import format_chip

    return format_chip(chips.load_chip(target))


def _search_docs(query: str, top_k: int = 5, target: str | None = None) -> str:
    if not docs.index_exists():
        return docs.NOT_INDEXED
    return docs.format_hits(query, docs.search(query, int(top_k), target))


def _create_mission(
    feature_name: str,
    description: str,
    project_root: str,
    board: str = "",
    target: str = "",
) -> str:
    path = mission.create_mission(feature_name, description, board, target, Path(project_root))
    return f"Mission created: {path}\n\n{path.read_text(encoding='utf-8')}"


def _validate_mission(path: str) -> str:
    result = mission.validate_mission(Path(path).read_text(encoding="utf-8"))
    return json.dumps(result, indent=2)


def _contract(
    summary: str,
    non_goals: list[str] | None = None,
    files: list[str] | None = None,
    risks: list[str] | None = None,
    tests: list[str] | None = None,
    rollback: str = "Revert the commit.",
) -> str:
    return generate_contract(summary, non_goals, files, risks, tests, rollback)


def _doctor(path: str, target: str | None = None) -> str:
    return doctor.run_doctor(Path(path), target)["report"]


LIST = {"type": "array", "items": STR}

TOOLS: list[Tool] = [
    Tool(
        "project_doctor",
        "Run all static checks (GPIO pins, task stacks, sdkconfig, partition table) on an "
        "ESP-IDF project directory and return one combined report.",
        _obj({"path": STR, "target": TARGET}, ["path"]),
        _doctor,
    ),
    Tool(
        "pin_audit",
        "Audit GPIO usage in a C/H file or project directory for the target chip: flash/PSRAM "
        "pins, strapping pins, input-only misuse, USB/UART0 pins, ADC2+Wi-Fi conflicts, double "
        "assignment. Target is auto-detected from sdkconfig when omitted.",
        _obj({"path": STR, "target": TARGET}, ["path"]),
        _pin_audit,
    ),
    Tool(
        "chip_info",
        "Return verified pin facts for an ESP32-family chip (strapping, flash, PSRAM, USB, "
        "JTAG, UART0, ADC1/ADC2 channel maps) with Espressif source links.",
        _obj({"target": TARGET}, ["target"]),
        _chip_info,
    ),
    Tool(
        "sdkconfig_review",
        "Review an sdkconfig / sdkconfig.defaults for security, stability, performance and "
        "production-readiness problems. Run before any release.",
        _obj({"path": STR}, ["path"]),
        _sdkconfig_review,
    ),
    Tool(
        "partition_check",
        "Validate a partitions.csv: alignment, overlaps, flash size, OTA slots/otadata, NVS, "
        "coredump, and app binary fit.",
        _obj({"csv_path": STR, "sdkconfig_path": STR, "app_bin": STR}, ["csv_path"]),
        _partition_check,
    ),
    Tool(
        "stack_check",
        "Find xTaskCreate/xTaskCreatePinnedToCore calls and flag undersized task stacks.",
        _obj({"path": STR}, ["path"]),
        _stack_check,
    ),
    Tool(
        "crash_decode",
        "Decode an ESP32 panic (Guru Meditation, abort, assert, stack overflow, heap "
        "corruption) from text or a log file. With an ELF and the ESP-IDF toolchain on PATH, "
        "resolves addresses to functions and lines.",
        _obj({"text": STR, "path": STR, "elf": STR, "target": TARGET}, []),
        _crash_decode,
    ),
    Tool(
        "log_summary",
        "Summarize an ESP32 serial log: error/warning counts, reset reason, crash detection, "
        "busiest components.",
        _obj({"text": STR, "path": STR}, []),
        _log_summary,
    ),
    Tool(
        "build_log_triage",
        "Classify ESP-IDF build/flash errors (missing headers, linker errors, IRAM/DRAM "
        "overflow, partition too small, target mismatch, serial port problems) with fixes.",
        _obj({"text": STR, "path": STR}, []),
        _build_log_triage,
    ),
    Tool(
        "search_esp_docs",
        "Search locally indexed Espressif datasheets and Technical Reference Manuals. Use "
        "before writing peripheral, ADC, timer, Wi-Fi or BLE code; cite the returned pages.",
        _obj({"query": STR, "top_k": {"type": "integer"}, "target": TARGET}, ["query"]),
        _search_docs,
    ),
    Tool(
        "create_mission",
        "Create a mission file (persistent task state) in <project_root>/missions/.",
        _obj(
            {
                "feature_name": STR,
                "description": STR,
                "project_root": STR,
                "board": STR,
                "target": STR,
            },
            ["feature_name", "description", "project_root"],
        ),
        _create_mission,
    ),
    Tool(
        "validate_mission",
        "Check a mission file has all required sections.",
        _obj({"path": STR}, ["path"]),
        _validate_mission,
    ),
    Tool(
        "generate_contract",
        "Generate an implementation contract (scope, risks, tests, rollback). Files use "
        "'path|ACTION|description' format.",
        _obj(
            {
                "summary": STR,
                "non_goals": LIST,
                "files": LIST,
                "risks": LIST,
                "tests": LIST,
                "rollback": STR,
            },
            ["summary"],
        ),
        _contract,
    ),
]

TOOL_INDEX = {t.name: t for t in TOOLS}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _result(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle(message: dict) -> dict | None:
    """Handle one JSON-RPC message and return the response (None for notifications)."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if msg_id is None:
        return None  # notifications (e.g. notifications/initialized) need no reply

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return _result(
            msg_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": INSTRUCTIONS,
            },
        )
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(
            msg_id,
            {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.schema}
                    for t in TOOLS
                ]
            },
        )
    if method == "tools/call":
        tool = TOOL_INDEX.get(params.get("name", ""))
        if tool is None:
            return _error(msg_id, -32602, f"Unknown tool: {params.get('name')}")
        arguments = {k: v for k, v in (params.get("arguments") or {}).items() if v is not None}
        try:
            text, is_error = tool.func(**arguments), False
        except Exception as exc:  # report tool failures to the model, not as protocol errors
            text, is_error = f"{type(exc).__name__}: {exc}", True
            traceback.print_exc(file=sys.stderr)
        return _result(msg_id, {"content": [{"type": "text", "text": text}], "isError": is_error})

    return _error(msg_id, -32601, f"Method not found: {method}")


def serve(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """Serve newline-delimited JSON-RPC until stdin closes."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            responses = [_error(None, -32700, "Parse error")]
        else:
            batch = message if isinstance(message, list) else [message]
            responses = [r for m in batch if isinstance(m, dict) and (r := handle(m)) is not None]
        for response in responses:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> None:
    """Entry point for ``esp32wb-mcp``."""
    serve()


if __name__ == "__main__":
    main()
