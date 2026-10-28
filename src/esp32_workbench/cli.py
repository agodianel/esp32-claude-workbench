"""``esp32wb`` — unified command line for the ESP32 Workbench."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from esp32_workbench import (
    __version__,
    buildlog,
    contract,
    crash,
    docs,
    doctor,
    installer,
    logs,
    mission,
    partitions,
    pins,
    sdkconfig,
    stacks,
)
from esp32_workbench.chips import Chip, list_targets, load_chip


def format_chip(chip: Chip) -> str:
    """Render a chip's pin facts as markdown."""

    def pins_text(values) -> str:
        return ", ".join(str(v) for v in sorted(values)) or "—"

    lines = [f"# {chip.name} (`{chip.target}`)", ""]
    lines.append(f"- **Architecture**: {chip.arch} (`{chip.toolchain_prefix}-*`)")
    lines.append(f"- **GPIOs**: {pins_text(chip.gpios)}")
    lines.append(f"- **Flash (never use)**: {pins_text(chip.flash_spi)}")
    if chip.psram_pins:
        lines.append(f"- **PSRAM**: {pins_text(chip.psram_pins)} — {chip.psram_condition}")
    lines.append(f"- **Input-only**: {pins_text(chip.input_only)}")
    lines.append(f"- **USB D-/D+**: {pins_text(chip.usb_jtag)}")
    lines.append(f"- **JTAG**: {pins_text(chip.jtag)}")
    lines.append(f"- **UART0 console**: TX {chip.uart0_tx}, RX {chip.uart0_rx}")
    lines.append(
        f"- **ADC1**: {', '.join(f'GPIO{g}=CH{c}' for g, c in sorted(chip.adc1.items())) or '—'}"
    )
    lines.append(
        f"- **ADC2**: {', '.join(f'GPIO{g}=CH{c}' for g, c in sorted(chip.adc2.items())) or '—'}"
    )
    if chip.adc2_wifi_policy:
        lines.append(f"- **ADC2 + Wi-Fi**: {chip.adc2_wifi_policy}")
    lines.append("")
    lines.append("## Strapping pins")
    for gpio, text in sorted(chip.strapping.items()):
        lines.append(f"- **GPIO {gpio}**: {text}")
    if chip.notes:
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in chip.notes)
    lines.append("")
    lines.append("## Sources")
    lines.extend(f"- {url}" for url in chip.sources)
    return "\n".join(lines) + "\n"


def _chip(args: argparse.Namespace) -> int:
    if not args.target:
        for target in list_targets():
            print(f"{target:10s} {load_chip(target).name}")
        return 0
    chip = load_chip(args.target)
    if args.json:
        data = asdict(chip)
        data["adc2_wifi_policy"] = chip.adc2_wifi_policy
        print(json.dumps(data, indent=2, default=sorted))
    else:
        print(format_chip(chip))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    result = doctor.run_doctor(args.path, args.target)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        print(result["report"])
    return 1 if result["errors"] else 0


def _init(args: argparse.Namespace) -> int:
    """Drop agent rules, mission templates and agent integrations into a firmware project."""
    project = args.path.resolve()
    project.mkdir(parents=True, exist_ok=True)
    template_root = resources.files("esp32_workbench") / "data" / "project"
    for item in ("AGENTS.md", "CLAUDE.md"):
        dest = project / item
        if dest.exists() and not args.force:
            print(f"skip   {dest} (exists; use --force to overwrite)")
            continue
        dest.write_text((template_root / item).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"wrote  {dest}")
    (project / "missions").mkdir(exist_ok=True)

    if args.agents:
        agents = list(installer.AGENTS) if args.agents == ["all"] else args.agents
        installer.print_log(installer.install_skills(agents, "project", project))
        for agent in agents:
            print(f"mcp    {installer.write_mcp_config(agent, 'project', project)}")
    return 0


def _install_skills(args: argparse.Namespace) -> int:
    agents = list(installer.AGENTS) if "all" in args.agent else args.agent
    lines = installer.install_skills(
        agents, args.scope, args.project, symlink=args.symlink, force=args.force
    )
    installer.print_log(lines)
    if args.with_mcp:
        for agent in agents:
            print(f"mcp    {installer.write_mcp_config(agent, args.scope, args.project)}")
    return 0


def _mcp_config(args: argparse.Namespace) -> int:
    command = args.command.split() if args.command else None
    if args.write:
        path = installer.write_mcp_config(args.agent, args.scope, args.project, command)
        print(f"Updated {path}")
    else:
        print(installer.render_mcp_config(args.agent, args.scope, command), end="")
    return 0


def _mcp_serve(_: argparse.Namespace) -> int:
    from esp32_workbench.mcp_server import serve

    serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``esp32wb`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="esp32wb",
        description="Multi-chip ESP32 firmware toolkit for AI coding agents.",
    )
    parser.add_argument("--version", action="version", version=f"esp32wb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, module, help_text: str):
        p = sub.add_parser(name, help=help_text, description=help_text)
        module.add_arguments(p)
        p.set_defaults(handler=module.run)
        return p

    p = sub.add_parser("doctor", help="Run every static check on a project")
    p.add_argument("path", type=Path, nargs="?", default=Path("."))
    p.add_argument("--target", choices=list_targets())
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_doctor)

    add("pins", pins, "Audit GPIO usage for the target chip")
    add("stacks", stacks, "Flag undersized FreeRTOS task stacks")
    add("sdkconfig", sdkconfig, "Review sdkconfig for production readiness")
    add("partitions", partitions, "Validate a partition table CSV")
    add("crash", crash, "Decode a panic / Guru Meditation dump")
    add("logs", logs, "Summarize a serial log")
    add("build-log", buildlog, "Triage ESP-IDF build/flash errors")
    add("docs", docs, "Index and search Espressif datasheets/TRMs")
    add("mission", mission, "Create or validate mission files")
    add("contract", contract, "Generate an implementation contract")

    p = sub.add_parser("chip", help="Show verified pin facts for a target (or list targets)")
    p.add_argument("target", nargs="?", choices=list_targets())
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_chip)

    agent_choices = [*installer.AGENTS, "all"]

    p = sub.add_parser("init", help="Set up a firmware project for AI agents")
    p.add_argument("path", type=Path, nargs="?", default=Path("."))
    p.add_argument("--agents", nargs="*", choices=agent_choices, help="Also install skills + MCP")
    p.add_argument("--force", action="store_true", help="Overwrite AGENTS.md / CLAUDE.md")
    p.set_defaults(handler=_init)

    p = sub.add_parser("install-skills", help="Install skills for one or more agents")
    p.add_argument("--agent", nargs="+", choices=agent_choices, required=True)
    p.add_argument("--scope", choices=["project", "user"], default="project")
    p.add_argument("--project", type=Path, default=None, help="Project root (default: cwd)")
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copying")
    p.add_argument("--force", action="store_true", help="Replace same-named skills not ours")
    p.add_argument("--with-mcp", action="store_true", help="Also register the MCP server")
    p.set_defaults(handler=_install_skills)

    p = sub.add_parser("mcp-config", help="Print (or write) MCP server config for an agent")
    p.add_argument("agent", choices=list(installer.AGENTS))
    p.add_argument("--scope", choices=["project", "user"], default="project")
    p.add_argument("--project", type=Path, default=None)
    p.add_argument("--command", help="Server command (default: esp32wb-mcp or uvx)")
    p.add_argument("--write", action="store_true", help="Merge into the agent's config file")
    p.set_defaults(handler=_mcp_config)

    p = sub.add_parser("mcp", help="Run the MCP server on stdio")
    p.set_defaults(handler=_mcp_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``esp32wb``."""
    args = build_parser().parse_args(argv)
    try:
        code = args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
