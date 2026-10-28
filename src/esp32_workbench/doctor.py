"""Run every applicable static check on an ESP-IDF project in one pass."""

from __future__ import annotations

from pathlib import Path

from esp32_workbench import partitions, pins, sdkconfig, stacks
from esp32_workbench.chips import detect_target


def find_partition_csv(project: Path, config_text: str) -> Path | None:
    """Find the custom partition CSV named in sdkconfig, or a partitions.csv at the root."""
    for line in config_text.splitlines():
        if line.startswith("CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="):
            candidate = project / line.split("=", 1)[1].strip().strip('"')
            if candidate.is_file():
                return candidate
    default = project / "partitions.csv"
    return default if default.is_file() else None


def run_doctor(project: Path, target: str | None = None) -> dict:
    """Run pins, stacks, sdkconfig and partition checks.

    Returns:
        Dict with per-check results, total ``errors`` and a combined markdown ``report``.
    """
    project = project.resolve()
    checks: dict[str, dict] = {}
    sections: list[str] = [f"# ESP32 Project Doctor: {project.name}", ""]

    detected = target or detect_target(project)
    sections.append(f"- **Target**: {detected or 'unknown (defaulting to esp32)'}")
    sections.append("")

    pin_result = pins.scan(project, target)
    checks["pins"] = {"errors": pin_result["errors"], "warnings": pin_result["warnings"]}
    sections.append(pin_result["report"])

    tasks, stack_issues = stacks.scan_directory(project)
    checks["stacks"] = {
        "tasks": len(tasks),
        "errors": sum(1 for i in stack_issues if i.severity == "ERROR"),
    }
    sections.append(stacks.format_report(tasks, stack_issues))

    config_file = next(
        (project / n for n in ("sdkconfig", "sdkconfig.defaults") if (project / n).is_file()),
        None,
    )
    config_text = ""
    if config_file:
        config_text = config_file.read_text(encoding="utf-8", errors="ignore")
        review = sdkconfig.review(config_file)
        checks["sdkconfig"] = {"file": config_file.name, "errors": review["errors"]}
        sections.append(review["report"])
    else:
        sections.append("# sdkconfig\n\nNo sdkconfig or sdkconfig.defaults found — skipped.\n")

    csv_file = find_partition_csv(project, config_text)
    if csv_file:
        try:
            part = partitions.check_file(csv_file, config_file)
            checks["partitions"] = {"file": csv_file.name, "errors": part["errors"]}
            sections.append(part["report"])
        except ValueError as exc:
            checks["partitions"] = {"file": csv_file.name, "errors": 1}
            sections.append(f"# Partition Table Report\n\n❌ Could not parse: {exc}\n")

    total = sum(c.get("errors", 0) for c in checks.values())
    sections.insert(2, f"- **Total errors**: {total}")
    return {"target": detected, "checks": checks, "errors": total, "report": "\n".join(sections)}
