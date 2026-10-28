"""Parse and summarize ESP32 serial log output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LogEntry:
    """A parsed ESP32 log line."""

    level: str
    tag: str
    message: str
    raw: str


@dataclass
class LogSummary:
    """Summary of analyzed ESP32 log output."""

    total_lines: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    debug_count: int = 0
    verbose_count: int = 0
    unparsed_count: int = 0
    errors: list[LogEntry] = field(default_factory=list)
    warnings: list[LogEntry] = field(default_factory=list)
    reset_reason: str | None = None
    crash_detected: bool = False
    crash_type: str | None = None
    tags: Counter = field(default_factory=Counter)
    boot_successful: bool = False


# Pattern for ESP-IDF log format: X (timestamp) tag: message
LOG_PATTERN = re.compile(r"^([EWIDV])\s+\((\d+)\)\s+([^:]+):\s+(.*)$")

# Pattern for reset reason
RESET_PATTERN = re.compile(r"rst:(0x[0-9a-fA-F]+)\s+\(([^)]+)\)")

# Pattern for Guru Meditation
GURU_PATTERN = re.compile(r"Guru Meditation Error.*panic'ed\s+\(([^)]+)\)")

# Reset reason codes differ between chips, but the ROM always prints the reason name
# next to the code (e.g. "rst:0xc (RTC_SW_CPU_RST)"). Interpret the name, not the code.
RESET_NAME_MEANINGS: list[tuple[str, str]] = [
    ("BROWN", "brownout — power supply dipped below threshold"),
    ("POWERON", "normal power-on"),
    ("DEEPSLEEP", "wake from deep sleep"),
    # A ROM-reported timer-group WDT reset is the hardware stage-2 reset: the system hung so
    # hard the panic handler never ran. Task/interrupt WDT *messages* earlier in the log say why.
    ("TG0WDT", "timer group 0 watchdog hard reset — system hung; check earlier WDT messages"),
    ("TG1WDT", "timer group 1 watchdog hard reset — system hung; check earlier WDT messages"),
    ("RTCWDT", "RTC watchdog — system hang or boot failure"),
    ("SUPER_WDT", "super watchdog"),
    ("USB", "reset from USB (e.g. flashing or USB-JTAG)"),
    ("JTAG", "reset from JTAG"),
    ("EFUSE", "eFuse error"),
    ("GLITCH", "power glitch detected"),
    ("SW", "software reset (esp_restart or panic handler)"),
]


def explain_reset(code: str, name: str) -> str:
    """Return a human-readable reset reason for a ROM ``rst:`` line."""
    upper = name.upper()
    for needle, meaning in RESET_NAME_MEANINGS:
        if needle in upper:
            return f"{name} ({meaning})"
    return f"{name} ({code})"


def parse_log_line(line: str) -> LogEntry | None:
    """Parse a single ESP32 log line.

    Args:
        line: Raw log line string.

    Returns:
        LogEntry if parseable, None otherwise.
    """
    match = LOG_PATTERN.match(line.strip())
    if match:
        return LogEntry(
            level=match.group(1),
            tag=match.group(3).strip(),
            message=match.group(4).strip(),
            raw=line.strip(),
        )
    return None


def summarize_logs(content: str) -> LogSummary:
    """Analyze ESP32 log content and produce a summary.

    Args:
        content: Raw serial log text.

    Returns:
        LogSummary with counts, errors, warnings, crash info.
    """
    summary = LogSummary()

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        summary.total_lines += 1

        # Check for reset reason
        reset_match = RESET_PATTERN.search(stripped)
        if reset_match:
            summary.reset_reason = explain_reset(reset_match.group(1).lower(), reset_match.group(2))

        # Check for Guru Meditation
        guru_match = GURU_PATTERN.search(stripped)
        if guru_match:
            summary.crash_detected = True
            summary.crash_type = guru_match.group(1)

        # Check for boot success indicators
        if "app_main" in stripped.lower() or "application started" in stripped.lower():
            summary.boot_successful = True

        # Parse structured log line
        entry = parse_log_line(stripped)
        if entry:
            summary.tags[entry.tag] += 1

            if entry.level == "E":
                summary.error_count += 1
                summary.errors.append(entry)
            elif entry.level == "W":
                summary.warning_count += 1
                summary.warnings.append(entry)
            elif entry.level == "I":
                summary.info_count += 1
            elif entry.level == "D":
                summary.debug_count += 1
            elif entry.level == "V":
                summary.verbose_count += 1
        else:
            summary.unparsed_count += 1

    return summary


def format_summary(summary: LogSummary) -> str:
    """Format a LogSummary into a readable markdown report.

    Args:
        summary: Analyzed log summary.

    Returns:
        Formatted markdown string.
    """
    lines: list[str] = []
    lines.append("# Log Summary Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append(f"- **Total lines**: {summary.total_lines}")
    lines.append(f"- **Errors**: {summary.error_count}")
    lines.append(f"- **Warnings**: {summary.warning_count}")
    lines.append(f"- **Info**: {summary.info_count}")
    lines.append(f"- **Debug**: {summary.debug_count}")
    lines.append(f"- **Verbose**: {summary.verbose_count}")
    lines.append(f"- **Unparsed**: {summary.unparsed_count}")
    lines.append("")

    # Reset reason
    if summary.reset_reason:
        lines.append("## Reset Reason")
        lines.append(f"- {summary.reset_reason}")
        lines.append("")

    # Crash info
    if summary.crash_detected:
        lines.append("## ⚠️ Crash Detected")
        lines.append(f"- **Type**: Guru Meditation — {summary.crash_type}")
        lines.append("")

    # Boot status
    lines.append("## Boot Status")
    status = "✅ Successful" if summary.boot_successful else "❓ Unknown (no boot indicator found)"
    lines.append(f"- {status}")
    lines.append("")

    # Errors
    if summary.errors:
        lines.append("## Errors")
        for entry in summary.errors[:20]:  # Cap at 20
            lines.append(f"- `[{entry.tag}]` {entry.message}")
        if len(summary.errors) > 20:
            lines.append(f"- ... and {len(summary.errors) - 20} more errors")
        lines.append("")

    # Warnings
    if summary.warnings:
        lines.append("## Warnings")
        for entry in summary.warnings[:10]:  # Cap at 10
            lines.append(f"- `[{entry.tag}]` {entry.message}")
        if len(summary.warnings) > 10:
            lines.append(f"- ... and {len(summary.warnings) - 10} more warnings")
        lines.append("")

    # Top tags
    if summary.tags:
        lines.append("## Active Components (by log count)")
        for tag, count in summary.tags.most_common(10):
            lines.append(f"- `{tag}`: {count} messages")
        lines.append("")

    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (shared by ``summarize-logs`` and ``esp32wb logs``)."""
    parser.add_argument("path", help="Serial log file, or - for stdin")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Summarize a serial log for parsed CLI arguments and return the exit code."""
    if args.path == "-":
        content = sys.stdin.read()
    else:
        filepath = Path(args.path)
        if not filepath.is_file():
            print(f"ERROR: File not found: {filepath}")
            return 1
        content = filepath.read_text(encoding="utf-8", errors="replace")

    summary = summarize_logs(content)
    if args.json:
        data = asdict(summary)
        data["tags"] = dict(summary.tags.most_common())
        print(json.dumps(data, indent=2))
    else:
        print(format_summary(summary))
    return 0


def main() -> None:
    """CLI entry point for summarize-logs command."""
    parser = argparse.ArgumentParser(prog="summarize-logs", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
