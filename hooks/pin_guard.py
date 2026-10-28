#!/usr/bin/env python3
"""Claude Code PostToolUse hook: audit GPIO usage right after a C/H file is edited.

Reads the hook payload on stdin. If the edited file is C/H source and the pin audit
finds ERRORs for the project's target chip, prints them to stderr and exits 2 so
Claude sees them and can fix the code. Warnings and clean files exit 0 silently.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esp32_workbench.pins import scan  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    path = Path(file_path)
    if path.suffix not in (".c", ".h") or not path.is_file():
        return 0

    try:
        result = scan(path)
    except Exception:  # a broken hook must never block editing
        return 0
    errors = [i for i in result["issues"] if i["severity"] == "ERROR"]
    if not errors:
        return 0

    print(
        f"esp32-workbench pin audit ({result['chip']}) found problems in {path.name}:",
        file=sys.stderr,
    )
    for issue in errors:
        print(f"  line {issue['line']}: {issue['message']}", file=sys.stderr)
    print("Fix these or explain why they are safe on this board.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
