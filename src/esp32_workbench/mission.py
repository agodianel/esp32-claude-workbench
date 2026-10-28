"""Create and validate mission markdown files (persistent task state for agents)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Required top-level sections in a mission file
REQUIRED_SECTIONS: list[str] = [
    "Goal",
    "Board / Target",
    "Constraints",
    "Files in Scope",
    "Acceptance Criteria",
    "Test Plan",
    "Known Risks",
    "Current Status",
    "Next Step",
]

# Optional but recommended sections
RECOMMENDED_SECTIONS: list[str] = [
    "Design Notes",
    "History",
]


def extract_sections(content: str) -> list[str]:
    """Extract all H2 (##) section titles from markdown content.

    Args:
        content: Raw markdown text.

    Returns:
        List of section titles (without the ## prefix).
    """
    sections: list[str] = []
    for line in content.splitlines():
        match = re.match(r"^##\s+(.+)$", line.strip())
        if match:
            sections.append(match.group(1).strip())
    return sections


def validate_mission(content: str) -> dict[str, list[str]]:
    """Validate a mission file's content for required sections.

    Args:
        content: Raw markdown text of the mission file.

    Returns:
        Dict with 'missing_required', 'missing_recommended', and 'found' lists.
    """
    found_sections = extract_sections(content)
    found_lower = [s.lower() for s in found_sections]

    missing_required: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section.lower() not in found_lower:
            missing_required.append(section)

    missing_recommended: list[str] = []
    for section in RECOMMENDED_SECTIONS:
        if section.lower() not in found_lower:
            missing_recommended.append(section)

    return {
        "found": found_sections,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
    }


def slugify(text: str) -> str:
    """Turn a feature name into a filename-safe kebab-case slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "mission"


def render_mission(
    feature_name: str,
    description: str,
    board: str = "",
    target: str = "",
    today: date | None = None,
) -> str:
    """Render a new mission document that passes :func:`validate_mission`."""
    day = (today or date.today()).isoformat()
    return f"""# Mission: {feature_name}

## Goal

{description}

## Board / Target

- **Chip**: {target or "[esp32 / esp32s3 / esp32c3 / ...]"}
- **Board**: {board or "[DevKitC / custom / ...]"}
- **ESP-IDF version**: [e.g. v5.3]

## Constraints

- [Hardware constraints — available pins, peripherals, power budget]
- [Software constraints — RTOS priorities, memory budget, latency]

## Files in Scope

| File | Action | Notes |
|------|--------|-------|
| `path/to/file` | CREATE / MODIFY / DELETE | [Brief description] |

## Design Notes

[Architecture decisions, trade-offs, alternatives considered.]

## Acceptance Criteria

- [ ] [Specific, testable criterion]
- [ ] All existing tests pass.
- [ ] New tests cover the change.
- [ ] Pin audit passes (if GPIO changes).

## Test Plan

- [ ] [Test case]
- [ ] No-hardware tests pass.
- [ ] Hardware tests pass (if applicable, `@pytest.mark.hardware`).

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| [Description] | LOW/MED/HIGH | [How to handle] |

## Current Status

- **State**: NOT STARTED
- **Last updated**: {day}
- **Completed items**: none
- **Blocking issues**: none

## Next Step

Write the implementation contract.

## History

| Date | Action | Notes |
|------|--------|-------|
| {day} | Created | Initial mission |
"""


def create_mission(
    feature_name: str,
    description: str,
    board: str = "",
    target: str = "",
    root: Path | None = None,
    today: date | None = None,
) -> Path:
    """Write a new mission file to ``<root>/missions/`` and return its path.

    ``root`` defaults to the current working directory, i.e. the firmware project
    the agent is working in — never the workbench installation itself.
    """
    day = today or date.today()
    missions_dir = (root or Path.cwd()) / "missions"
    missions_dir.mkdir(parents=True, exist_ok=True)
    path = missions_dir / f"{day.isoformat()}-{slugify(feature_name)}.md"
    if path.exists():
        raise FileExistsError(f"Mission already exists: {path}")
    path.write_text(render_mission(feature_name, description, board, target, day), encoding="utf-8")
    return path


def validate_file(filepath: Path) -> bool:
    """Validate a mission file at the given path.

    Args:
        filepath: Path to the mission markdown file.

    Returns:
        True if all required sections are present, False otherwise.
    """
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return False

    content = filepath.read_text(encoding="utf-8")
    result = validate_mission(content)

    print(f"\nValidating: {filepath.name}")
    print(f"  Found sections: {len(result['found'])}")

    if result["missing_required"]:
        print("  ❌ Missing required sections:")
        for section in result["missing_required"]:
            print(f"     - {section}")
    else:
        print("  ✅ All required sections present")

    if result["missing_recommended"]:
        print("  ⚠️  Missing recommended sections:")
        for section in result["missing_recommended"]:
            print(f"     - {section}")

    return len(result["missing_required"]) == 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``esp32wb mission`` subcommands."""
    sub = parser.add_subparsers(dest="mission_command", required=True)

    new = sub.add_parser("new", help="Create a mission file in <root>/missions/")
    new.add_argument("feature_name", help="Short feature name, e.g. 'Wi-Fi reconnect'")
    new.add_argument("--description", "-d", default="[What needs to be accomplished.]")
    new.add_argument("--board", default="")
    new.add_argument("--target", default="")
    new.add_argument("--root", type=Path, default=None, help="Project root (default: cwd)")

    check = sub.add_parser("validate", help="Validate mission files")
    check.add_argument("files", nargs="+", type=Path)
    check.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Run a mission subcommand and return the exit code."""
    if args.mission_command == "new":
        try:
            path = create_mission(
                args.feature_name, args.description, args.board, args.target, args.root
            )
        except FileExistsError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"Mission created: {path}")
        return 0

    if args.json:
        results = {}
        for filepath in args.files:
            if filepath.is_file():
                results[str(filepath)] = validate_mission(filepath.read_text(encoding="utf-8"))
            else:
                results[str(filepath)] = {"error": "file not found"}
        print(json.dumps(results, indent=2))
        ok = all(not r.get("error") and not r["missing_required"] for r in results.values())
        return 0 if ok else 1

    all_valid = all([validate_file(f) for f in args.files])
    if all_valid:
        print("\n✅ All mission files valid.")
    return 0 if all_valid else 1


def main() -> None:
    """CLI entry point for validate-mission command."""
    if len(sys.argv) < 2:
        print("Usage: validate-mission <mission-file.md> [...]")
        print("       validate-mission missions/*.md")
        sys.exit(1)

    all_valid = all([validate_file(Path(arg)) for arg in sys.argv[1:]])
    if not all_valid:
        sys.exit(1)
    print("\n✅ All mission files valid.")


if __name__ == "__main__":
    main()
