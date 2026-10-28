"""Triage ESP-IDF build, flash and monitor output into known failure classes with fixes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Header → component that provides it (for REQUIRES / PRIV_REQUIRES hints).
# ESP-IDF v5.3+ split the driver component into esp_driver_*; the legacy
# "driver" component still re-exports them, so both are mentioned.
HEADER_COMPONENTS: dict[str, str] = {
    "esp_wifi.h": "esp_wifi",
    "esp_netif.h": "esp_netif",
    "esp_event.h": "esp_event",
    "nvs_flash.h": "nvs_flash",
    "nvs.h": "nvs_flash",
    "esp_http_client.h": "esp_http_client",
    "esp_http_server.h": "esp_http_server",
    "esp_https_ota.h": "esp_https_ota",
    "mqtt_client.h": "mqtt",
    "esp_timer.h": "esp_timer",
    "esp_adc/adc_oneshot.h": "esp_adc",
    "esp_adc/adc_continuous.h": "esp_adc",
    "driver/gpio.h": "esp_driver_gpio (or driver)",
    "driver/i2c_master.h": "esp_driver_i2c (or driver)",
    "driver/spi_master.h": "esp_driver_spi (or driver)",
    "driver/uart.h": "esp_driver_uart (or driver)",
    "driver/ledc.h": "esp_driver_ledc (or driver)",
    "driver/rmt_tx.h": "esp_driver_rmt (or driver)",
    "driver/gptimer.h": "esp_driver_gptimer (or driver)",
    "esp_lcd_panel_ops.h": "esp_lcd",
    "esp_littlefs.h": "joltwallet/littlefs (component manager)",
    "esp_spiffs.h": "spiffs",
    "esp_vfs_fat.h": "fatfs",
    "cJSON.h": "json (or espressif/cjson from the component manager)",
    "esp_partition.h": "esp_partition",
    "esp_ota_ops.h": "app_update",
    "esp_bt.h": "bt",
    "esp_crt_bundle.h": "mbedtls",
}


@dataclass
class BuildFinding:
    """A recognised failure in build/flash output."""

    category: str
    severity: str  # ERROR, WARNING
    line: int
    excerpt: str
    fix: str


@dataclass
class _Rule:
    category: str
    severity: str
    pattern: re.Pattern
    fix: str


RULES: list[_Rule] = [
    _Rule(
        "missing_header",
        "ERROR",
        re.compile(r"fatal error: ([\w/.\-]+\.h): No such file or directory"),
        "Header not found. Add the providing component to REQUIRES/PRIV_REQUIRES in this "
        "component's CMakeLists.txt{component_hint}, or add the dependency with "
        "`idf.py add-dependency`.",
    ),
    _Rule(
        "undefined_reference",
        "ERROR",
        re.compile(r"undefined reference to [`']([^'`]+)'"),
        "Linker cannot find `{0}`. Check the source file is in SRCS, the providing component is "
        "in REQUIRES, the function is not `static`, and any Kconfig option that enables it is set.",
    ),
    _Rule(
        "iram_overflow",
        "ERROR",
        re.compile(r"region [`']iram\S*' overflowed by (\d+) bytes"),
        "IRAM is full by {0} bytes. Remove IRAM_ATTR from non-critical functions, disable "
        "CONFIG_*_IN_IRAM options you don't need, or enable CONFIG_COMPILER_OPTIMIZATION_SIZE.",
    ),
    _Rule(
        "dram_overflow",
        "ERROR",
        re.compile(r"region [`']dram\S*' overflowed by (\d+) bytes"),
        "DRAM (static data) is full by {0} bytes. Move large static buffers to the heap or PSRAM "
        "(EXT_RAM_BSS_ATTR), and reduce static Wi-Fi/LWIP buffer counts.",
    ),
    _Rule(
        "app_partition_too_small",
        "ERROR",
        re.compile(r"app partition is too small for binary"),
        "The app no longer fits its partition. Use a larger partition table (e.g. "
        "CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE or a custom partitions.csv), or shrink the "
        "binary with CONFIG_COMPILER_OPTIMIZATION_SIZE. Run `esp32wb partitions` to check.",
    ),
    _Rule(
        "target_mismatch",
        "ERROR",
        re.compile(
            r"Target '?(\w+)'? specified .* is not consistent with target '?(\w+)'?"
            r"|This chip is (ESP32[-\w]*), not (ESP32[-\w]*)"
        ),
        "Build target and chip disagree. Run `idf.py set-target <chip>` (this resets "
        "sdkconfig) and rebuild.",
    ),
    _Rule(
        "component_not_found",
        "ERROR",
        re.compile(r"Failed to resolve component '([\w\-/]+)'|Cannot find component"),
        "Component `{0}` is unknown. Check the name in REQUIRES, that the component directory "
        "has a CMakeLists.txt with idf_component_register(), or add it with "
        "`idf.py add-dependency`.",
    ),
    _Rule(
        "dependency_solver",
        "ERROR",
        re.compile(r"Cannot solve dependencies|Because no versions of .* match"),
        "The component manager could not satisfy idf_component.yml. Relax version ranges or "
        "check that the dependency supports your ESP-IDF version and target.",
    ),
    _Rule(
        "implicit_declaration",
        "ERROR",
        re.compile(r"implicit declaration of function '(\w+)'"),
        "`{0}` is used without a prototype. Include its header; if the header is included, the "
        "API may have been removed or renamed in your ESP-IDF version.",
    ),
    _Rule(
        "undeclared_config",
        "ERROR",
        re.compile(r"'(CONFIG_\w+)' undeclared"),
        "`{0}` is not defined. Enable it in menuconfig / sdkconfig.defaults, or guard the code "
        "with `#ifdef {0}`. Delete sdkconfig and rebuild if defaults changed.",
    ),
    _Rule(
        "undeclared_identifier",
        "ERROR",
        re.compile(r"'(\w+)' undeclared"),
        "`{0}` is undeclared: missing include, wrong target, or an API changed between ESP-IDF "
        "versions.",
    ),
    _Rule(
        "unknown_type",
        "ERROR",
        re.compile(r"unknown type name '(\w+)'"),
        "Type `{0}` is unknown: missing include or a type removed in your ESP-IDF version.",
    ),
    _Rule(
        "werror",
        "ERROR",
        re.compile(r"error: .*\[-Werror=([\w\-]+)\]"),
        "A warning (-W{0}) is treated as an error. Fix the warning; do not disable -Werror globally.",
    ),
    _Rule(
        "legacy_driver",
        "WARNING",
        re.compile(r"legacy (\w+) driver is deprecated|The legacy (\w+) driver is deprecated"),
        "A legacy driver is in use. Migrate to the new driver API (see the ESP-IDF v5 migration "
        "guide). Mixing legacy and new drivers for the same peripheral aborts at runtime.",
    ),
    _Rule(
        "idf_env",
        "ERROR",
        re.compile(
            r"idf\.py: command not found|IDF_PATH .* not set|python virtual environment .* not found",
            re.IGNORECASE,
        ),
        "ESP-IDF environment is not active. Run `. $IDF_PATH/export.sh` (or install.sh first).",
    ),
    _Rule(
        "serial_permission",
        "ERROR",
        re.compile(r"Permission denied: '(/dev/\w+)'|could not open port '?(/dev/\S+?)'?:"),
        "No access to the serial port. On Linux add your user to the dialout (or uucp) group and "
        "log in again; check no other monitor holds the port.",
    ),
    _Rule(
        "no_serial_data",
        "ERROR",
        re.compile(r"Failed to connect to (ESP32[-\w]*): No serial data received"),
        "The chip did not enter download mode. Hold BOOT, tap EN/RESET, release BOOT; check "
        "the cable carries data and GPIO strapping pins are not held by external circuitry.",
    ),
]


def triage(text: str) -> list[BuildFinding]:
    """Scan build/flash output and return recognised findings (deduplicated)."""
    findings: list[BuildFinding] = []
    seen: set[tuple[str, str]] = set()

    for line_num, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            match = rule.pattern.search(line)
            if not match:
                continue
            groups = [g for g in match.groups() if g] or [""]
            key = (rule.category, groups[0])
            if key in seen:
                break
            seen.add(key)

            component_hint = ""
            if rule.category == "missing_header":
                header = groups[0]
                provider = HEADER_COMPONENTS.get(header)
                component_hint = f" (`{header}` is provided by `{provider}`)" if provider else ""
            fix = rule.fix.replace("{component_hint}", component_hint).format(*groups)
            findings.append(BuildFinding(rule.category, rule.severity, line_num, line.strip(), fix))
            break  # first matching rule wins for a line

    order = {"ERROR": 0, "WARNING": 1}
    return sorted(findings, key=lambda f: (order[f.severity], f.line))


def format_report(findings: list[BuildFinding]) -> str:
    """Format build triage findings as markdown."""
    if not findings:
        return (
            "# Build Triage\n\nNo known failure pattern recognised. Look for the first `error:` "
            "line in the output — later errors are often consequences of it.\n"
        )
    lines = ["# Build Triage", ""]
    for f in findings:
        icon = "❌" if f.severity == "ERROR" else "⚠️"
        lines.append(f"## {icon} {f.category} (line {f.line})")
        lines.append("")
        lines.append(f"```\n{f.excerpt}\n```")
        lines.append(f"**Fix**: {f.fix}")
        lines.append("")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for ``esp32wb build-log``."""
    parser.add_argument("path", help="Build/flash output file, or - for stdin")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Triage build output for parsed CLI arguments and return the exit code."""
    if args.path == "-":
        text = sys.stdin.read()
    else:
        source = Path(args.path)
        if not source.is_file():
            print(f"ERROR: File not found: {source}")
            return 1
        text = source.read_text(encoding="utf-8", errors="replace")

    findings = triage(text)
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        print(format_report(findings))
    return 0
