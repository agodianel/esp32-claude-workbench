"""Generate an implementation contract from a template with provided parameters."""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


def generate_contract(
    summary: str,
    non_goals: list[str] | None = None,
    files: list[str] | None = None,
    risks: list[str] | None = None,
    tests: list[str] | None = None,
    rollback: str = "Revert the commit.",
) -> str:
    """Generate an implementation contract markdown string.

    Args:
        summary: One-paragraph change description.
        non_goals: List of non-goals.
        files: List of affected files in 'path|action|description' format.
        risks: List of risk descriptions.
        tests: List of test case descriptions.
        rollback: Rollback strategy text.

    Returns:
        Formatted markdown string for the implementation contract.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    non_goals_md = ""
    if non_goals:
        non_goals_md = "\n".join(f"- {ng}" for ng in non_goals)
    else:
        non_goals_md = "- [To be defined]"

    files_md = ""
    if files:
        files_md = "| File | Action | Description |\n|------|--------|-------------|\n"
        for f in files:
            parts = f.split("|")
            if len(parts) == 3:
                files_md += f"| `{parts[0].strip()}` | {parts[1].strip()} | {parts[2].strip()} |\n"
            else:
                files_md += f"| `{f.strip()}` | MODIFY | [To be described] |\n"
    else:
        files_md = (
            "| File | Action | Description |\n"
            "|------|--------|-------------|\n"
            "| [path] | [action] | [description] |"
        )

    risks_md = ""
    if risks:
        risks_md = "\n".join(f"- {r}" for r in risks)
    else:
        risks_md = "- [To be assessed]"

    tests_md = ""
    if tests:
        tests_md = "\n".join(f"- [ ] {t}" for t in tests)
    else:
        tests_md = "- [ ] [To be defined]"

    contract = textwrap.dedent(f"""\
        # Implementation Contract

        **Date**: {date}

        ## Change Summary

        {summary}

        ## Non-Goals

        {non_goals_md}

        ## Affected Files

        {files_md}

        ## Risk Assessment

        {risks_md}

        ## Test Plan

        {tests_md}

        ## Rollback Strategy

        {rollback}

        ## Acceptance Criteria

        - [ ] All tests pass.
        - [ ] No new warnings.
        - [ ] PR prepared with evidence.

        ## Approval

        - [ ] Contract reviewed
        - [ ] Contract accepted — proceed with implementation
    """)

    return contract


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (shared by ``generate-contract`` and ``esp32wb contract``)."""
    parser.add_argument(
        "summary",
        help="One-paragraph description of the change.",
    )
    parser.add_argument(
        "--non-goals",
        nargs="*",
        help="Things this change does NOT do.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Affected files in 'path|action|description' format.",
    )
    parser.add_argument(
        "--risks",
        nargs="*",
        help="Risk descriptions.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        help="Test case descriptions.",
    )
    parser.add_argument(
        "--rollback",
        default="Revert the commit.",
        help="Rollback strategy.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Prints to stdout if not specified.",
    )


def run(args: argparse.Namespace) -> int:
    """Generate a contract for parsed CLI arguments and return the exit code."""
    contract = generate_contract(
        summary=args.summary,
        non_goals=args.non_goals,
        files=args.files,
        risks=args.risks,
        tests=args.tests,
        rollback=args.rollback,
    )

    if args.output:
        Path(args.output).write_text(contract, encoding="utf-8")
        print(f"Contract written to {args.output}")
    else:
        print(contract)
    return 0


def main() -> None:
    """CLI entry point for generate-contract command."""
    parser = argparse.ArgumentParser(
        prog="generate-contract",
        description="Generate an implementation contract for ESP32 firmware changes.",
    )
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
