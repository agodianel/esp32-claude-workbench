"""Scan ESP32 C/H source files to extract GPIO pin assignments and detect conflicts.

Rules come from the chip database (``chips.py``), so the same scan is correct for
ESP32, ESP32-S3, ESP32-C3, ESP32-C6 and the other supported targets. It flags:
- Pins that do not exist on the target or are wired to SPI flash / PSRAM
- Strapping, UART0 console, USB-Serial-JTAG and JTAG pins
- Double-assigned pins (two peripherals on the same GPIO)
- Input-only pins used as outputs
- ADC2 usage in Wi-Fi projects (on targets where that conflicts)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from esp32_workbench.chips import Chip, list_targets, resolve_chip


@dataclass
class PinAssignment:
    """A detected GPIO pin assignment in source code."""

    gpio: int
    purpose: str
    file: str
    line: int
    raw: str


@dataclass
class PinIssue:
    """A detected issue with pin usage."""

    severity: str  # ERROR, WARNING, INFO
    gpio: int
    message: str
    file: str
    line: int


SKIP_DIRS = {"build", ".venv", "managed_components", ".git", "node_modules"}

# Patterns to detect GPIO usage
PATTERNS = [
    # #define PIN_NAME GPIO_NUM_XX or #define PIN_NAME XX
    re.compile(r"#define\s+(\w+)\s+GPIO_NUM_(\d+)", re.MULTILINE),
    re.compile(
        r"#define\s+(\w+(?:_PIN|_GPIO|_SDA|_SCL|_MOSI|_MISO|_CS|_CLK|_RST|_DC|_LED|_BTN|_INT))\s+(\d+)",
        re.MULTILINE,
    ),
    # gpio_config_t with pin_bit_mask
    re.compile(r"\.pin_bit_mask\s*=\s*\(\s*1ULL\s*<<\s*(\d+)\s*\)", re.MULTILINE),
    # gpio_set_direction(GPIO_NUM_XX, ...)
    re.compile(r"gpio_set_direction\s*\(\s*GPIO_NUM_(\d+)", re.MULTILINE),
    # gpio_set_level(GPIO_NUM_XX, ...)
    re.compile(r"gpio_set_level\s*\(\s*GPIO_NUM_(\d+)", re.MULTILINE),
    # i2c_config .sda_io_num = XX
    re.compile(r"\.(sda_io_num|scl_io_num)\s*=\s*(?:GPIO_NUM_)?(\d+)", re.MULTILINE),
    # spi_bus_config_t fields
    re.compile(r"\.(mosi_io_num|miso_io_num|sclk_io_num)\s*=\s*(?:GPIO_NUM_)?(\d+)", re.MULTILINE),
    # UART tx_io_num / rx_io_num
    re.compile(
        r"\.(tx_io_num|rx_io_num|rts_io_num|cts_io_num)\s*=\s*(?:GPIO_NUM_)?(\d+)", re.MULTILINE
    ),
    # LEDC output_io_num
    re.compile(r"\.output_io_num\s*=\s*(?:GPIO_NUM_)?(\d+)", re.MULTILINE),
]

# Legacy ADC2 channel macros; the GPIO comes from the chip's ADC2 table
ADC2_CHANNEL_PATTERN = re.compile(r"ADC2_CHANNEL_(\d+)")

# Wi-Fi usage indicators
WIFI_INDICATORS = [
    re.compile(r"#include\s*[\"<]esp_wifi\.h[\">]"),
    re.compile(r"esp_wifi_init"),
    re.compile(r"WIFI_MODE_STA|WIFI_MODE_AP|WIFI_MODE_APSTA"),
]

OUTPUT_KEYWORDS = ("led", "mosi", "sda", "scl", "tx", "cs", "rst", "dc", "clk", "output")

COMMENT_PATTERN = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def strip_comments(content: str) -> str:
    """Blank out C comments while keeping line numbers stable."""
    return COMMENT_PATTERN.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), content)


def iter_sources(path: Path):
    """Yield C/H files under ``path`` (or ``path`` itself), skipping build output."""
    if path.is_file():
        yield path
        return
    for filepath in sorted(path.rglob("*.[ch]")):
        if not SKIP_DIRS.intersection(filepath.parts):
            yield filepath


def scan_file(filepath: Path, chip: Chip | None = None) -> list[PinAssignment]:
    """Scan a single C/H file for GPIO pin assignments.

    Args:
        filepath: Path to the source file.
        chip: Target chip, used to resolve ADC2 channel macros to GPIOs.

    Returns:
        List of detected PinAssignment objects.
    """
    assignments: list[PinAssignment] = []
    try:
        original = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return assignments

    content = strip_comments(original)
    lines = original.splitlines()

    def add(gpio: int, purpose: str, pos: int) -> None:
        line_num = content[:pos].count("\n") + 1
        raw_line = lines[line_num - 1].strip() if line_num <= len(lines) else ""
        assignments.append(
            PinAssignment(
                gpio=gpio, purpose=purpose, file=str(filepath), line=line_num, raw=raw_line
            )
        )

    for pattern in PATTERNS:
        for match in pattern.finditer(content):
            groups = match.groups()
            if len(groups) == 2:
                purpose, gpio_text = groups
            else:
                purpose, gpio_text = "gpio_usage", groups[0]
            gpio = int(gpio_text)
            if 0 <= gpio <= 63:
                add(gpio, purpose, match.start())

    if chip is not None:
        channel_map = chip.adc2_channel_to_gpio
        for match in ADC2_CHANNEL_PATTERN.finditer(content):
            gpio = channel_map.get(int(match.group(1)))
            if gpio is not None:
                add(gpio, f"adc2_channel_{match.group(1)}", match.start())

    return assignments


def detect_wifi_usage(path: Path) -> bool:
    """Check if the project uses Wi-Fi.

    Args:
        path: Project root directory or a single source file.

    Returns:
        True if Wi-Fi usage is detected.
    """
    for filepath in iter_sources(path):
        try:
            content = strip_comments(filepath.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if any(pattern.search(content) for pattern in WIFI_INDICATORS):
            return True
    return False


def analyze_pins(
    assignments: list[PinAssignment],
    uses_wifi: bool = False,
    chip: Chip | None = None,
) -> list[PinIssue]:
    """Analyze pin assignments for issues.

    Args:
        assignments: List of detected pin assignments.
        uses_wifi: Whether the project uses Wi-Fi.
        chip: Target chip. Defaults to the classic ESP32.

    Returns:
        List of PinIssue objects, errors first.
    """
    chip = chip or resolve_chip(None)
    issues: list[PinIssue] = []
    pin_users: dict[int, list[PinAssignment]] = defaultdict(list)

    for a in assignments:
        pin_users[a.gpio].append(a)

    def issue(severity: str, gpio: int, message: str, where: PinAssignment) -> None:
        issues.append(PinIssue(severity, gpio, message, where.file, where.line))

    for gpio, users in pin_users.items():
        first = users[0]

        if chip.gpios and gpio not in chip.gpios:
            issue("ERROR", gpio, f"GPIO {gpio} does not exist on {chip.name}", first)
            continue

        if gpio in chip.flash_spi:
            issue(
                "ERROR",
                gpio,
                f"GPIO {gpio} is connected to internal SPI flash on {chip.name} — NEVER use this pin",
                first,
            )

        if gpio in chip.psram_pins:
            condition = f" ({chip.psram_condition})" if chip.psram_condition else ""
            issue(
                "WARNING",
                gpio,
                f"GPIO {gpio} is used by PSRAM on some {chip.name} modules{condition}",
                first,
            )

        if gpio in chip.conditional:
            issue("WARNING", gpio, f"GPIO {gpio} on {chip.name}: {chip.conditional[gpio]}", first)

        if gpio in chip.strapping:
            issue(
                "WARNING",
                gpio,
                f"GPIO {gpio} is a boot strapping pin on {chip.name} ({chip.strapping[gpio]}) "
                "— may affect boot if externally driven",
                first,
            )

        if gpio in (chip.uart0_tx, chip.uart0_rx):
            direction = "TX" if gpio == chip.uart0_tx else "RX"
            issue(
                "WARNING",
                gpio,
                f"GPIO {gpio} is default UART0 {direction} on {chip.name} — using it will disable serial output",
                first,
            )

        if gpio in chip.usb_jtag:
            issue(
                "WARNING",
                gpio,
                f"GPIO {gpio} is a USB D-/D+ pin on {chip.name} — using it disables the USB "
                "console/flashing interface",
                first,
            )

        if gpio in chip.input_only:
            for user in users:
                if any(kw in user.purpose.lower() for kw in OUTPUT_KEYWORDS):
                    issue(
                        "ERROR",
                        gpio,
                        f"GPIO {gpio} is input-only but assigned to output function '{user.purpose}'",
                        user,
                    )

        policy = chip.adc2_wifi_policy
        if uses_wifi and gpio in chip.adc2 and policy:
            for user in users:
                if "adc" not in user.purpose.lower():
                    continue
                if policy == "forbidden":
                    issue(
                        "ERROR",
                        gpio,
                        f"GPIO {gpio} is ADC2 — cannot be used while Wi-Fi is active "
                        f"on {chip.name}. Use an ADC1 pin.",
                        user,
                    )
                else:
                    issue(
                        "WARNING",
                        gpio,
                        f"GPIO {gpio} is ADC2, which Wi-Fi also uses on {chip.name} — reads can "
                        "fail while Wi-Fi holds ADC2. Handle read errors or use an ADC1 pin.",
                        user,
                    )

        unique_purposes = {u.purpose for u in users}
        if len(unique_purposes) > 1:
            purpose_list = ", ".join(sorted(unique_purposes))
            issue(
                "ERROR",
                gpio,
                f"GPIO {gpio} is assigned to multiple functions: {purpose_list}",
                first,
            )

    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    return sorted(issues, key=lambda i: (order[i.severity], i.gpio))


def format_report(
    assignments: list[PinAssignment],
    issues: list[PinIssue],
    uses_wifi: bool,
    chip: Chip | None = None,
) -> str:
    """Format pin scan results as a markdown report.

    Args:
        assignments: All detected pin assignments.
        issues: All detected issues.
        uses_wifi: Whether Wi-Fi is used.
        chip: Target chip the rules were applied for.

    Returns:
        Formatted markdown string.
    """
    chip = chip or resolve_chip(None)
    lines: list[str] = []
    lines.append("# Pin Scan Report")
    lines.append("")

    error_count = sum(1 for i in issues if i.severity == "ERROR")
    warning_count = sum(1 for i in issues if i.severity == "WARNING")
    lines.append("## Summary")
    lines.append(f"- **Target**: {chip.name} (`{chip.target}`)")
    lines.append(f"- **Pins detected**: {len({a.gpio for a in assignments})}")
    lines.append(f"- **Assignments**: {len(assignments)}")
    lines.append(f"- **Errors**: {error_count}")
    lines.append(f"- **Warnings**: {warning_count}")
    lines.append(f"- **Wi-Fi detected**: {'Yes' if uses_wifi else 'No'}")
    lines.append("")

    pin_map: dict[int, list[PinAssignment]] = defaultdict(list)
    for a in assignments:
        pin_map[a.gpio].append(a)

    if pin_map:
        lines.append("## Pin Allocation Table")
        lines.append("")
        lines.append("| GPIO | Purpose | File | Line |")
        lines.append("|------|---------|------|------|")
        for gpio in sorted(pin_map):
            for a in pin_map[gpio]:
                lines.append(f"| {gpio} | `{a.purpose}` | {Path(a.file).name} | {a.line} |")
        lines.append("")

    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            icon = "❌" if issue.severity == "ERROR" else "⚠️"
            lines.append(f"- {icon} **{issue.severity}** GPIO {issue.gpio}: {issue.message}")
            lines.append(f"  - File: `{Path(issue.file).name}:{issue.line}`")
        lines.append("")
    else:
        lines.append("## ✅ No Issues Found")
        lines.append("")

    return "\n".join(lines)


def scan_directory(
    directory: Path, target: str | None = None
) -> tuple[list[PinAssignment], list[PinIssue], bool]:
    """Scan a C/H file or every C/H file in a directory tree.

    Args:
        directory: Root directory (or single file) to scan.
        target: ESP-IDF target. Auto-detected from sdkconfig when omitted.

    Returns:
        Tuple of (assignments, issues, uses_wifi).
    """
    chip = resolve_chip(target, directory)
    all_assignments: list[PinAssignment] = []
    for filepath in iter_sources(directory):
        all_assignments.extend(scan_file(filepath, chip))

    uses_wifi = detect_wifi_usage(directory)
    issues = analyze_pins(all_assignments, uses_wifi, chip)
    return all_assignments, issues, uses_wifi


def scan(path: Path, target: str | None = None) -> dict:
    """Scan ``path`` and return a JSON-serialisable result."""
    chip = resolve_chip(target, path)
    assignments, issues, uses_wifi = scan_directory(path, chip.target)
    return {
        "target": chip.target,
        "chip": chip.name,
        "uses_wifi": uses_wifi,
        "assignments": [asdict(a) for a in assignments],
        "issues": [asdict(i) for i in issues],
        "errors": sum(1 for i in issues if i.severity == "ERROR"),
        "warnings": sum(1 for i in issues if i.severity == "WARNING"),
        "report": format_report(assignments, issues, uses_wifi, chip),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (shared by ``scan-pins`` and ``esp32wb pins``)."""
    parser.add_argument("path", type=Path, help="Project directory or C/H file")
    parser.add_argument(
        "--target",
        choices=list_targets(),
        help="ESP-IDF target (default: CONFIG_IDF_TARGET from sdkconfig, else esp32)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Execute a pin scan for parsed CLI arguments and return the exit code."""
    if not args.path.exists():
        print(f"ERROR: Path not found: {args.path}")
        return 1
    result = scan(args.path, args.target)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        print(result["report"])
    return 1 if result["errors"] else 0


def main() -> None:
    """CLI entry point for pin scanner."""
    parser = argparse.ArgumentParser(prog="scan-pins", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
