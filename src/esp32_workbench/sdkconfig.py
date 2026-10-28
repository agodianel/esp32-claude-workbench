"""Analyze ESP-IDF sdkconfig files for common misconfigurations.

Parses sdkconfig and sdkconfig.defaults to flag:
- Security issues (disabled TLS verification, default passwords)
- Performance issues (wrong clock speed, suboptimal settings)
- Stability issues (watchdog disabled, small stack sizes)
- Compatibility issues (wrong flash size, missing features for peripherals used)
- Production readiness (debug logging in release, assertions enabled)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ConfigIssue:
    """A detected sdkconfig issue."""

    severity: str  # ERROR, WARNING, INFO
    category: str  # security, performance, stability, compatibility, production
    key: str
    message: str
    recommendation: str


# Rules: (key_pattern, expected_value_or_None, severity, category, message, recommendation)
# If expected_value is None, we check for presence. If it's a string, we check the value.
RULES: list[dict] = [
    # === Security ===
    {
        "key": "CONFIG_ESP_TLS_SKIP_SERVER_CERT_VERIFY",
        "bad_value": "y",
        "severity": "ERROR",
        "category": "security",
        "message": "TLS server certificate verification is DISABLED — connections are vulnerable to MITM attacks",
        "recommendation": "Set CONFIG_ESP_TLS_SKIP_SERVER_CERT_VERIFY=n and provide proper CA certificates",
    },
    {
        "key": "CONFIG_MBEDTLS_CERTIFICATE_BUNDLE",
        "bad_value": None,  # bad if missing
        "severity": "WARNING",
        "category": "security",
        "message": "Mozilla certificate bundle not enabled — HTTPS connections may fail",
        "recommendation": "Enable CONFIG_MBEDTLS_CERTIFICATE_BUNDLE=y for standard TLS connections",
    },
    {
        "key": "CONFIG_ESP_WIFI_PASSWORD",
        "bad_value_pattern": r"^(password|12345678|admin|default)$",
        "severity": "ERROR",
        "category": "security",
        "message": "Wi-Fi password appears to be a default/weak password",
        "recommendation": "Use a strong Wi-Fi password or environment variables",
    },
    # === Stability ===
    {
        "key": "CONFIG_ESP_TASK_WDT_EN",
        "bad_value": None,  # bad if missing (disabled)
        "severity": "WARNING",
        "category": "stability",
        "message": "Task watchdog timer may be disabled — hangs will go undetected",
        "recommendation": "Enable CONFIG_ESP_TASK_WDT_EN=y to detect task hangs",
    },
    {
        "key": "CONFIG_ESP_TASK_WDT_TIMEOUT_S",
        "bad_value_range": (0, 2),
        "severity": "WARNING",
        "category": "stability",
        "message": "Watchdog timeout is very short — may trigger false positives",
        "recommendation": "Set CONFIG_ESP_TASK_WDT_TIMEOUT_S to 5-15 seconds",
    },
    {
        "key": "CONFIG_FREERTOS_HZ",
        "bad_value_range": (0, 99),
        "severity": "WARNING",
        "category": "performance",
        "message": "FreeRTOS tick rate is below 100 Hz — timer resolution will be coarse",
        "recommendation": "Set CONFIG_FREERTOS_HZ=1000 for 1ms tick resolution",
    },
    {
        "key": "CONFIG_ESP_MAIN_TASK_STACK_SIZE",
        "bad_value_range": (0, 3583),
        "severity": "WARNING",
        "category": "stability",
        "message": "Main task stack size is small — may cause stack overflow with Wi-Fi/BLE",
        "recommendation": "Set CONFIG_ESP_MAIN_TASK_STACK_SIZE to at least 4096 (8192 for TLS)",
    },
    # === Performance ===
    {
        "key": "CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ",
        "bad_value": "80",
        "severity": "INFO",
        "category": "performance",
        "message": "CPU is running at 80 MHz — this is low for Wi-Fi/BLE workloads",
        "recommendation": "Set CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=160 or 240 for better performance",
    },
    # === Compatibility ===
    {
        "key": "CONFIG_ESPTOOLPY_FLASHSIZE",
        "bad_value": '"2MB"',
        "severity": "WARNING",
        "category": "compatibility",
        "message": "Flash size is 2MB — OTA updates need at least 4MB for two app partitions",
        "recommendation": "Use 4MB or larger flash if OTA is needed",
    },
    # === Production ===
    {
        "key": "CONFIG_LOG_DEFAULT_LEVEL_DEBUG",
        "bad_value": "y",
        "severity": "INFO",
        "category": "production",
        "message": "Default log level is DEBUG — verbose logging in production wastes CPU and flash",
        "recommendation": "Use CONFIG_LOG_DEFAULT_LEVEL_WARN or CONFIG_LOG_DEFAULT_LEVEL_ERROR for production",
    },
    {
        "key": "CONFIG_LOG_DEFAULT_LEVEL_VERBOSE",
        "bad_value": "y",
        "severity": "WARNING",
        "category": "production",
        "message": "Default log level is VERBOSE — extremely noisy, impacts performance",
        "recommendation": "Reduce to INFO or WARN for production builds",
    },
    {
        "key": "CONFIG_COMPILER_OPTIMIZATION_DEBUG",
        "bad_value": "y",
        "severity": "INFO",
        "category": "production",
        "message": "Compiler optimization is set to Debug (-Og) — not optimal for production",
        "recommendation": "Use CONFIG_COMPILER_OPTIMIZATION_SIZE=y or PERF=y for production",
    },
    {
        "key": "CONFIG_HEAP_POISONING_COMPREHENSIVE",
        "bad_value": "y",
        "severity": "INFO",
        "category": "production",
        "message": "Comprehensive heap poisoning is enabled — significant performance overhead",
        "recommendation": "Disable in production builds (keep for development/debugging)",
    },
    {
        "key": "CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG",
        "bad_value": "y",
        "severity": "INFO",
        "category": "production",
        "message": "Compiler optimization is set to Debug (legacy ESP-IDF key) — not optimal for production",
        "recommendation": "Use CONFIG_COMPILER_OPTIMIZATION_SIZE=y or PERF=y for production",
    },
    {
        "key": "CONFIG_ESP_SYSTEM_PANIC_SILENT_REBOOT",
        "bad_value": "y",
        "severity": "WARNING",
        "category": "production",
        "message": "Panic handler reboots silently — crash causes from the field will be lost",
        "recommendation": "Use CONFIG_ESP_SYSTEM_PANIC_PRINT_REBOOT=y and enable core dumps",
    },
    {
        "key": "CONFIG_BOOTLOADER_LOG_LEVEL",
        "bad_value_range": (4, 5),
        "severity": "INFO",
        "category": "production",
        "message": "Bootloader log level is Debug/Verbose — slows boot and leaks layout details",
        "recommendation": "Use CONFIG_BOOTLOADER_LOG_LEVEL_INFO or lower for production",
    },
    {
        "key": "CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH",
        "bad_value": None,
        "severity": "INFO",
        "category": "stability",
        "message": "Core dump to flash is not enabled — post-mortem crash analysis will be difficult",
        "recommendation": "Enable CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH=y (needs a coredump partition)",
    },
]


def parse_sdkconfig(content: str) -> dict[str, str]:
    """Parse sdkconfig content into key-value pairs.

    Args:
        content: Raw sdkconfig file content.

    Returns:
        Dictionary of config key to value.
    """
    config: dict[str, str] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            # Check for "# CONFIG_X is not set" pattern
            not_set_match = re.match(r"^#\s+(CONFIG_\w+)\s+is not set$", line)
            if not_set_match:
                config[not_set_match.group(1)] = "__NOT_SET__"
            continue

        match = re.match(r"^(CONFIG_\w+)=(.*)$", line)
        if match:
            config[match.group(1)] = match.group(2)

    return config


def analyze_config(config: dict[str, str]) -> list[ConfigIssue]:
    """Analyze parsed sdkconfig for issues.

    Args:
        config: Parsed config dictionary.

    Returns:
        List of ConfigIssue objects.
    """
    issues: list[ConfigIssue] = []

    for rule in RULES:
        key = rule["key"]

        if "bad_value" in rule:
            bad_val = rule["bad_value"]
            if bad_val is None:
                # Bad if key is missing or not set
                if key not in config or config[key] == "__NOT_SET__":
                    issues.append(
                        ConfigIssue(
                            severity=rule["severity"],
                            category=rule["category"],
                            key=key,
                            message=rule["message"],
                            recommendation=rule["recommendation"],
                        )
                    )
            else:
                # Bad if key equals bad_value
                if key in config and config[key] == bad_val:
                    issues.append(
                        ConfigIssue(
                            severity=rule["severity"],
                            category=rule["category"],
                            key=key,
                            message=rule["message"],
                            recommendation=rule["recommendation"],
                        )
                    )

        if "bad_value_pattern" in rule:
            if key in config:
                val = config[key].strip('"')
                if re.match(rule["bad_value_pattern"], val, re.IGNORECASE):
                    issues.append(
                        ConfigIssue(
                            severity=rule["severity"],
                            category=rule["category"],
                            key=key,
                            message=rule["message"],
                            recommendation=rule["recommendation"],
                        )
                    )

        if "bad_value_range" in rule:
            if key in config and config[key] != "__NOT_SET__":
                try:
                    val = int(config[key])
                    low, high = rule["bad_value_range"]
                    if low <= val <= high:
                        issues.append(
                            ConfigIssue(
                                severity=rule["severity"],
                                category=rule["category"],
                                key=key,
                                message=f"{rule['message']} (current: {val})",
                                recommendation=rule["recommendation"],
                            )
                        )
                except ValueError:
                    pass

    # Cross-check: OTA requires sufficient flash and two app partitions
    flash_size = config.get("CONFIG_ESPTOOLPY_FLASHSIZE", "")
    has_ota = any("OTA" in k and config.get(k) == "y" for k in config)
    if has_ota and flash_size == '"2MB"':
        issues.append(
            ConfigIssue(
                severity="ERROR",
                category="compatibility",
                key="CONFIG_ESPTOOLPY_FLASHSIZE",
                message="OTA is enabled but flash size is only 2MB — not enough for two app partitions",
                recommendation="Use at least 4MB flash for OTA support",
            )
        )

    # Cross-check: BLE + Wi-Fi coexistence
    has_bt = config.get("CONFIG_BT_ENABLED") == "y"
    has_wifi = config.get("CONFIG_ESP_WIFI_ENABLED") == "y" or any(
        "WIFI" in k and config.get(k) == "y" for k in config
    )
    if has_bt and has_wifi:
        coex = config.get("CONFIG_SW_COEXIST_ENABLE", "__NOT_SET__")
        if coex != "y":
            issues.append(
                ConfigIssue(
                    severity="WARNING",
                    category="compatibility",
                    key="CONFIG_SW_COEXIST_ENABLE",
                    message="Both Wi-Fi and Bluetooth are enabled but software coexistence is not — may cause RF conflicts",
                    recommendation="Enable CONFIG_SW_COEXIST_ENABLE=y for Wi-Fi/BT coexistence",
                )
            )

    return sorted(
        issues,
        key=lambda i: (
            {"ERROR": 0, "WARNING": 1, "INFO": 2}[i.severity],
            i.category,
        ),
    )


def format_report(issues: list[ConfigIssue], config: dict[str, str]) -> str:
    """Format analysis results as markdown.

    Args:
        issues: Detected issues.
        config: Parsed config for summary stats.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("# sdkconfig Analysis Report")
    lines.append("")

    error_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARNING")
    info_count = sum(1 for i in issues if i.severity == "INFO")

    lines.append("## Summary")
    lines.append(
        f"- **Config entries**: {len([k for k, v in config.items() if v != '__NOT_SET__'])}"
    )
    lines.append(f"- **Errors**: {error_count}")
    lines.append(f"- **Warnings**: {warn_count}")
    lines.append(f"- **Info**: {info_count}")
    lines.append("")

    if not issues:
        lines.append("## ✅ No Issues Found")
        lines.append("")
        return "\n".join(lines)

    # Group by category
    by_category: dict[str, list[ConfigIssue]] = {}
    for issue in issues:
        by_category.setdefault(issue.category, []).append(issue)

    category_titles = {
        "security": "🔒 Security",
        "stability": "🛡️ Stability",
        "performance": "⚡ Performance",
        "compatibility": "🔧 Compatibility",
        "production": "📦 Production Readiness",
    }

    for cat, cat_issues in by_category.items():
        title = category_titles.get(cat, cat.title())
        lines.append(f"## {title}")
        lines.append("")
        for issue in cat_issues:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[issue.severity]
            lines.append(f"### {icon} {issue.key}")
            lines.append(f"**{issue.severity}**: {issue.message}")
            lines.append("")
            lines.append(f"**Fix**: {issue.recommendation}")
            lines.append("")

    return "\n".join(lines)


def analyze_file(filepath: Path) -> tuple[list[ConfigIssue], dict[str, str]]:
    """Analyze an sdkconfig file.

    Args:
        filepath: Path to sdkconfig or sdkconfig.defaults.

    Returns:
        Tuple of (issues, parsed_config).
    """
    content = filepath.read_text(encoding="utf-8")
    config = parse_sdkconfig(content)
    issues = analyze_config(config)
    return issues, config


def review(filepath: Path) -> dict:
    """Analyze an sdkconfig file and return a JSON-serialisable result."""
    issues, config = analyze_file(filepath)
    return {
        "file": str(filepath),
        "target": config.get("CONFIG_IDF_TARGET", "").strip('"') or None,
        "issues": [asdict(i) for i in issues],
        "errors": sum(1 for i in issues if i.severity == "ERROR"),
        "warnings": sum(1 for i in issues if i.severity == "WARNING"),
        "report": format_report(issues, config),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (shared by ``analyze-sdkconfig`` and ``esp32wb sdkconfig``)."""
    parser.add_argument("path", type=Path, help="sdkconfig or sdkconfig.defaults file")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Execute an sdkconfig review for parsed CLI arguments and return the exit code."""
    if not args.path.is_file():
        print(f"ERROR: File not found: {args.path}")
        return 1
    result = review(args.path)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        print(result["report"])
    return 1 if result["errors"] else 0


def main() -> None:
    """CLI entry point for sdkconfig analyzer."""
    parser = argparse.ArgumentParser(prog="analyze-sdkconfig", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
