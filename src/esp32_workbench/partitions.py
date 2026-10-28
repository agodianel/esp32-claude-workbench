"""Check an ESP-IDF partition table CSV for layout and OTA-readiness problems.

Rules follow the ESP-IDF partition table guide:
https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/partition-tables.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

APP_ALIGN = 0x10000
DATA_ALIGN = 0x1000
DEFAULT_TABLE_OFFSET = 0x8000
TABLE_SIZE = 0x1000
OTADATA_SIZE = 0x2000
NVS_MIN_SIZE = 0x3000
NAME_MAX = 15  # 16 bytes including the null terminator
APP_USAGE_WARN = 0.90

SIZE_PATTERN = re.compile(r"^(0x[0-9a-fA-F]+|\d+)\s*([KkMm]?)$")
FLASH_SIZE_PATTERN = re.compile(r'^CONFIG_ESPTOOLPY_FLASHSIZE="(\d+)MB"', re.MULTILINE)
TABLE_OFFSET_PATTERN = re.compile(r"^CONFIG_PARTITION_TABLE_OFFSET=(0x[0-9a-fA-F]+)", re.MULTILINE)


@dataclass
class Partition:
    """One row of a partition table."""

    name: str
    type: str
    subtype: str
    offset: int
    size: int
    flags: str
    line: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class PartitionIssue:
    """A detected partition table problem."""

    severity: str  # ERROR, WARNING, INFO
    partition: str
    message: str


def parse_size(text: str) -> int:
    """Parse a partition size/offset (decimal, hex, optional K/M suffix)."""
    match = SIZE_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Invalid size or offset: '{text}'")
    value = int(match.group(1), 0)
    suffix = match.group(2).upper()
    return value * {"": 1, "K": 1024, "M": 1024 * 1024}[suffix]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_csv(content: str, table_offset: int = DEFAULT_TABLE_OFFSET) -> list[Partition]:
    """Parse partition CSV content, computing blank offsets like ``gen_esp32part.py``."""
    partitions: list[Partition] = []
    next_offset = table_offset + TABLE_SIZE

    for line_num, raw in enumerate(content.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        cols += [""] * (6 - len(cols))
        name, ptype, subtype, offset_text, size_text, flags = cols[:6]
        if not size_text:
            raise ValueError(f"Line {line_num}: missing size for partition '{name}'")

        align = APP_ALIGN if ptype == "app" else DATA_ALIGN
        offset = parse_size(offset_text) if offset_text else _align_up(next_offset, align)
        size = parse_size(size_text)
        partitions.append(Partition(name, ptype, subtype, offset, size, flags, line_num))
        next_offset = offset + size

    return partitions


def check_partitions(
    partitions: list[Partition],
    flash_size: int | None = None,
    table_offset: int = DEFAULT_TABLE_OFFSET,
    app_size: int | None = None,
    coredump_enabled: bool = False,
) -> list[PartitionIssue]:
    """Validate a parsed partition table.

    Args:
        partitions: Parsed partitions.
        flash_size: Flash size in bytes, if known.
        table_offset: Partition table offset (CONFIG_PARTITION_TABLE_OFFSET).
        app_size: Size of the built application binary in bytes, if known.
        coredump_enabled: Whether core dump to flash is configured.

    Returns:
        List of issues, errors first.
    """
    issues: list[PartitionIssue] = []

    def add(severity: str, name: str, message: str) -> None:
        issues.append(PartitionIssue(severity, name, message))

    seen: set[str] = set()
    for p in partitions:
        if p.name in seen:
            add("ERROR", p.name, "Duplicate partition name")
        seen.add(p.name)
        if len(p.name) > NAME_MAX:
            add("ERROR", p.name, f"Name is longer than {NAME_MAX} characters")
        if p.type == "app" and p.offset % APP_ALIGN:
            add(
                "ERROR",
                p.name,
                f"App partition offset {p.offset:#x} is not 64 KB (0x10000) aligned",
            )
        elif p.offset % DATA_ALIGN:
            add("ERROR", p.name, f"Offset {p.offset:#x} is not 4 KB (0x1000) aligned")
        if p.offset < table_offset + TABLE_SIZE:
            add("ERROR", p.name, f"Offset {p.offset:#x} overlaps the bootloader/partition table")
        if flash_size is not None and p.end > flash_size:
            add(
                "ERROR",
                p.name,
                f"Ends at {p.end:#x}, beyond the {flash_size // (1024 * 1024)} MB flash",
            )

    ordered = sorted(partitions, key=lambda p: p.offset)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.offset < prev.end:
            add("ERROR", cur.name, f"Overlaps '{prev.name}' (ends at {prev.end:#x})")

    apps = [p for p in partitions if p.type == "app"]
    ota_apps = [p for p in apps if p.subtype.startswith("ota_")]
    otadata = [p for p in partitions if p.type == "data" and p.subtype == "ota"]
    nvs = [p for p in partitions if p.type == "data" and p.subtype == "nvs"]

    if not apps:
        add("ERROR", "-", "No app partition — nothing to boot")
    if ota_apps and not otadata:
        add("ERROR", "-", "OTA app slots exist but there is no otadata (data, ota) partition")
    if otadata and otadata[0].size != OTADATA_SIZE:
        add("WARNING", otadata[0].name, f"otadata should be {OTADATA_SIZE:#x} bytes")
    if len(ota_apps) == 1:
        add("WARNING", ota_apps[0].name, "Only one OTA slot — an update cannot fall back")
    if len({p.size for p in ota_apps}) > 1:
        add("WARNING", "-", "OTA slots differ in size — an image that fits one may not fit another")
    if not nvs:
        add("WARNING", "-", "No NVS partition — Wi-Fi, BLE and PHY calibration need NVS")
    for p in nvs:
        if p.size < NVS_MIN_SIZE and "readonly" not in p.flags:
            add("WARNING", p.name, f"NVS is smaller than the recommended {NVS_MIN_SIZE:#x} bytes")
    if coredump_enabled and not any(p.subtype == "coredump" for p in partitions):
        add("ERROR", "-", "Core dump to flash is enabled but there is no coredump partition")

    if app_size is not None and apps:
        smallest = min(apps, key=lambda p: p.size)
        usage = app_size / smallest.size
        if usage > 1:
            add(
                "ERROR",
                smallest.name,
                f"App binary ({app_size:#x}) does not fit ({smallest.size:#x})",
            )
        elif usage > APP_USAGE_WARN:
            add("WARNING", smallest.name, f"App binary uses {usage:.0%} of the smallest app slot")

    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    return sorted(issues, key=lambda i: order[i.severity])


def read_sdkconfig_hints(sdkconfig: Path) -> dict:
    """Pull flash size, table offset and core dump setting from an sdkconfig."""
    hints: dict = {}
    if not sdkconfig.is_file():
        return hints
    text = sdkconfig.read_text(encoding="utf-8", errors="ignore")
    if match := FLASH_SIZE_PATTERN.search(text):
        hints["flash_size"] = int(match.group(1)) * 1024 * 1024
    if match := TABLE_OFFSET_PATTERN.search(text):
        hints["table_offset"] = int(match.group(1), 16)
    hints["coredump_enabled"] = "CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH=y" in text
    return hints


def format_report(partitions: list[Partition], issues: list[PartitionIssue]) -> str:
    """Format a partition check as markdown."""
    lines = ["# Partition Table Report", "", "## Layout", ""]
    lines.append("| Name | Type | SubType | Offset | Size | End |")
    lines.append("|------|------|---------|--------|------|-----|")
    for p in sorted(partitions, key=lambda p: p.offset):
        lines.append(
            f"| {p.name} | {p.type} | {p.subtype} | {p.offset:#x} | "
            f"{p.size:#x} ({p.size // 1024} KB) | {p.end:#x} |"
        )
    lines.append("")
    if issues:
        lines.append("## Issues")
        lines.append("")
        icons = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
        for issue in issues:
            lines.append(
                f"- {icons[issue.severity]} **{issue.severity}** `{issue.partition}`: {issue.message}"
            )
    else:
        lines.append("## ✅ No Issues Found")
    lines.append("")
    return "\n".join(lines)


def check_file(
    csv_path: Path,
    sdkconfig: Path | None = None,
    app_bin: Path | None = None,
    flash_size: int | None = None,
) -> dict:
    """Check a partition CSV and return a JSON-serialisable result."""
    hints = read_sdkconfig_hints(sdkconfig) if sdkconfig else {}
    table_offset = hints.get("table_offset", DEFAULT_TABLE_OFFSET)
    partitions = parse_csv(csv_path.read_text(encoding="utf-8"), table_offset)
    issues = check_partitions(
        partitions,
        flash_size=flash_size or hints.get("flash_size"),
        table_offset=table_offset,
        app_size=app_bin.stat().st_size if app_bin and app_bin.is_file() else None,
        coredump_enabled=hints.get("coredump_enabled", False),
    )
    return {
        "partitions": [asdict(p) for p in partitions],
        "issues": [asdict(i) for i in issues],
        "errors": sum(1 for i in issues if i.severity == "ERROR"),
        "report": format_report(partitions, issues),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for ``esp32wb partitions``."""
    parser.add_argument("csv", type=Path, help="partitions.csv")
    parser.add_argument("--sdkconfig", type=Path, help="sdkconfig for flash size / table offset")
    parser.add_argument("--bin", type=Path, help="Built app binary to check against slot sizes")
    parser.add_argument("--flash-size", type=parse_size, help="Flash size, e.g. 4M")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Check a partition table for parsed CLI arguments and return the exit code."""
    if not args.csv.is_file():
        print(f"ERROR: File not found: {args.csv}")
        return 1
    try:
        result = check_file(args.csv, args.sdkconfig, args.bin, args.flash_size)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        print(result["report"])
    return 1 if result["errors"] else 0


def main() -> None:
    """Standalone entry point."""
    parser = argparse.ArgumentParser(prog="esp32wb partitions", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))
