"""Scan ESP32 source files for FreeRTOS task creation and flag undersized stacks.

Detects xTaskCreate and xTaskCreatePinnedToCore calls, extracts task name,
stack size, and priority, then flags potential stack overflow risks based on
known stack requirements for common ESP-IDF operations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from esp32_workbench.pins import SKIP_DIRS


@dataclass
class TaskInfo:
    """Parsed FreeRTOS task creation info."""

    name: str
    stack_size: int | None  # None if couldn't parse (macro or expression)
    stack_expr: str  # Raw expression for stack size
    priority: str
    core: str | None  # None for xTaskCreate, 0/1/tskNO_AFFINITY for pinned
    function: str
    file: str
    line: int


@dataclass
class StackIssue:
    """A detected stack sizing issue."""

    severity: str  # ERROR, WARNING, INFO
    task_name: str
    message: str
    file: str
    line: int


# Minimum recommended stack sizes by task type
STACK_THRESHOLDS = {
    "minimal": 1024,
    "simple": 2048,
    "networking": 4096,
    "tls_https": 8192,
    "json_heavy": 8192,
}

# Keywords that indicate a task needs more stack
HEAVY_KEYWORDS = {
    "wifi": 4096,
    "http": 8192,
    "https": 8192,
    "tls": 8192,
    "ssl": 8192,
    "mqtt": 4096,
    "ble": 4096,
    "bt": 4096,
    "bluetooth": 4096,
    "ota": 8192,
    "json": 4096,
    "cjson": 4096,
    "websocket": 8192,
    "tcp": 4096,
    "udp": 3072,
    "dns": 3072,
    "sntp": 3072,
    "mdns": 4096,
    "display": 4096,
    "gui": 8192,
    "lvgl": 8192,
    "camera": 8192,
}

# Pattern for xTaskCreate(function, "name", stack, param, priority, handle)
TASK_CREATE_PATTERN = re.compile(
    r"xTaskCreate\s*\(\s*"
    r"(\w+)\s*,\s*"  # function
    r'"([^"]*?)"\s*,\s*'  # task name (quoted string)
    r"([^,]+?)\s*,\s*"  # stack size
    r"[^,]+?\s*,\s*"  # parameter
    r"([^,]+?)\s*,\s*"  # priority
    r"[^)]+?\)",  # handle
    re.DOTALL,
)

# Pattern for xTaskCreatePinnedToCore(func, "name", stack, param, priority, handle, core)
TASK_CREATE_PINNED_PATTERN = re.compile(
    r"xTaskCreatePinnedToCore\s*\(\s*"
    r"(\w+)\s*,\s*"  # function
    r'"([^"]*?)"\s*,\s*'  # task name
    r"([^,]+?)\s*,\s*"  # stack size
    r"[^,]+?\s*,\s*"  # parameter
    r"([^,]+?)\s*,\s*"  # priority
    r"[^,]+?\s*,\s*"  # handle
    r"([^)]+?)\)",  # core
    re.DOTALL,
)


def try_parse_int(expr: str) -> int | None:
    """Try to parse a stack size expression as an integer.

    Handles simple expressions like '4096', '4 * 1024', 'configMINIMAL_STACK_SIZE + 1024'.

    Args:
        expr: Raw stack size expression.

    Returns:
        Integer value or None if unparseable.
    """
    expr = expr.strip()

    # Direct integer
    try:
        return int(expr)
    except ValueError:
        pass

    # Simple multiplication: X * Y
    mult_match = re.match(r"(\d+)\s*\*\s*(\d+)", expr)
    if mult_match:
        return int(mult_match.group(1)) * int(mult_match.group(2))

    return None


def scan_file(filepath: Path) -> list[TaskInfo]:
    """Scan a C file for task creation calls.

    Args:
        filepath: Path to source file.

    Returns:
        List of TaskInfo objects.
    """
    tasks: list[TaskInfo] = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return tasks

    # Scan for xTaskCreatePinnedToCore first (more specific)
    for match in TASK_CREATE_PINNED_PATTERN.finditer(content):
        pos = match.start()
        line_num = content[:pos].count("\n") + 1

        stack_expr = match.group(3).strip()
        tasks.append(
            TaskInfo(
                function=match.group(1).strip(),
                name=match.group(2).strip(),
                stack_size=try_parse_int(stack_expr),
                stack_expr=stack_expr,
                priority=match.group(4).strip(),
                core=match.group(5).strip(),
                file=str(filepath),
                line=line_num,
            )
        )

    # Scan for xTaskCreate
    for match in TASK_CREATE_PATTERN.finditer(content):
        pos = match.start()
        line_num = content[:pos].count("\n") + 1

        # Skip if this is actually xTaskCreatePinnedToCore
        if "PinnedToCore" in content[pos : pos + 30]:
            continue

        stack_expr = match.group(3).strip()
        tasks.append(
            TaskInfo(
                function=match.group(1).strip(),
                name=match.group(2).strip(),
                stack_size=try_parse_int(stack_expr),
                stack_expr=stack_expr,
                priority=match.group(4).strip(),
                core=None,
                file=str(filepath),
                line=line_num,
            )
        )

    return tasks


def analyze_tasks(tasks: list[TaskInfo]) -> list[StackIssue]:
    """Analyze task stack sizes for potential issues.

    Args:
        tasks: List of detected tasks.

    Returns:
        List of StackIssue objects.
    """
    issues: list[StackIssue] = []

    for task in tasks:
        if task.stack_size is None:
            issues.append(
                StackIssue(
                    severity="INFO",
                    task_name=task.name,
                    message=f"Stack size is a macro/expression '{task.stack_expr}' — cannot verify automatically",
                    file=task.file,
                    line=task.line,
                )
            )
            continue

        # Check absolute minimum
        if task.stack_size < STACK_THRESHOLDS["minimal"]:
            issues.append(
                StackIssue(
                    severity="ERROR",
                    task_name=task.name,
                    message=f"Stack size {task.stack_size} bytes is below absolute minimum ({STACK_THRESHOLDS['minimal']})",
                    file=task.file,
                    line=task.line,
                )
            )
            continue

        # Check against task name/function for known heavy operations
        combined = f"{task.name} {task.function}".lower()
        for keyword, min_stack in HEAVY_KEYWORDS.items():
            if keyword in combined and task.stack_size < min_stack:
                issues.append(
                    StackIssue(
                        severity="WARNING",
                        task_name=task.name,
                        message=(
                            f"Task appears to use {keyword.upper()} "
                            f"(stack: {task.stack_size}) — "
                            f"recommended minimum: {min_stack} bytes"
                        ),
                        file=task.file,
                        line=task.line,
                    )
                )
                break  # Only report most critical keyword

        # General check for small stacks
        if task.stack_size < STACK_THRESHOLDS["simple"]:
            issues.append(
                StackIssue(
                    severity="WARNING",
                    task_name=task.name,
                    message=f"Stack size {task.stack_size} bytes is small — minimum recommended is {STACK_THRESHOLDS['simple']}",
                    file=task.file,
                    line=task.line,
                )
            )

    return sorted(
        issues, key=lambda i: 0 if i.severity == "ERROR" else 1 if i.severity == "WARNING" else 2
    )


def format_report(tasks: list[TaskInfo], issues: list[StackIssue]) -> str:
    """Format task stack analysis as markdown.

    Args:
        tasks: All detected tasks.
        issues: Detected issues.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("# Task Stack Analysis Report")
    lines.append("")

    error_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARNING")

    lines.append("## Summary")
    lines.append(f"- **Tasks found**: {len(tasks)}")
    lines.append(f"- **Errors**: {error_count}")
    lines.append(f"- **Warnings**: {warn_count}")
    lines.append("")

    # Task table
    if tasks:
        lines.append("## Task Inventory")
        lines.append("")
        lines.append("| Task Name | Function | Stack | Priority | Core | File |")
        lines.append("|-----------|----------|-------|----------|------|------|")
        for t in tasks:
            stack_str = str(t.stack_size) if t.stack_size else f"`{t.stack_expr}`"
            core_str = str(t.core) if t.core is not None else "any"
            short_file = Path(t.file).name
            lines.append(
                f"| {t.name} | `{t.function}` | {stack_str} | {t.priority} | {core_str} | {short_file}:{t.line} |"
            )
        lines.append("")

    # Stack sizing guide
    lines.append("## Stack Sizing Guide")
    lines.append("")
    lines.append("| Task Type | Minimum Stack |")
    lines.append("|-----------|--------------|")
    lines.append("| Simple processing | 2,048 bytes |")
    lines.append("| String/JSON operations | 4,096 bytes |")
    lines.append("| Wi-Fi / MQTT / TCP | 4,096 bytes |")
    lines.append("| TLS / HTTPS / OTA | 8,192 bytes |")
    lines.append("| GUI / LVGL / Camera | 8,192+ bytes |")
    lines.append("")

    # Issues
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[issue.severity]
            short_file = Path(issue.file).name
            lines.append(f"- {icon} **{issue.task_name}**: {issue.message}")
            lines.append(f"  - `{short_file}:{issue.line}`")
        lines.append("")
    else:
        lines.append("## ✅ No Issues Found")
        lines.append("")

    return "\n".join(lines)


def scan_directory(directory: Path) -> tuple[list[TaskInfo], list[StackIssue]]:
    """Scan all C files in a directory for task creations.

    Args:
        directory: Root directory to scan.

    Returns:
        Tuple of (tasks, issues).
    """
    all_tasks: list[TaskInfo] = []

    files = [directory] if directory.is_file() else sorted(directory.rglob("*.c"))
    for filepath in files:
        if SKIP_DIRS.intersection(filepath.parts):
            continue
        all_tasks.extend(scan_file(filepath))

    issues = analyze_tasks(all_tasks)
    return all_tasks, issues


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (shared by ``check-stacks`` and ``esp32wb stacks``)."""
    parser.add_argument("path", type=Path, help="Project directory or C file")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Check task stacks for parsed CLI arguments and return the exit code."""
    if not args.path.exists():
        print(f"ERROR: Path not found: {args.path}")
        return 1
    tasks, issues = scan_directory(args.path)
    errors = sum(1 for i in issues if i.severity == "ERROR")
    if args.json:
        payload = {"tasks": [asdict(t) for t in tasks], "issues": [asdict(i) for i in issues]}
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(tasks, issues))
    return 1 if errors else 0


def main() -> None:
    """CLI entry point for task stack checker."""
    parser = argparse.ArgumentParser(prog="check-stacks", description=__doc__.splitlines()[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
