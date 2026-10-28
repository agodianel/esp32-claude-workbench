"""Install the workbench skills and MCP server config into coding agents.

Paths and config formats come from each vendor's docs (checked 2026-10-28):
- Claude Code: https://code.claude.com/docs/en/skills.md
- Codex: https://learn.chatgpt.com/docs/build-skills, https://learn.chatgpt.com/docs/extend/mcp
- Cursor: https://cursor.com/docs/skills.md, https://cursor.com/docs/mcp.md
- GitHub Copilot: https://docs.github.com/en/copilot/concepts/agents/about-agent-skills
- Gemini CLI: https://geminicli.com/docs/cli/skills.md, https://geminicli.com/docs/tools/mcp-server.md
- Antigravity: https://antigravity.google/docs/skills.md, https://antigravity.google/docs/mcp.md
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

SERVER_NAME = "esp32-workbench"
SKILL_MARKER = "source: esp32-workbench"
REPO_URL = "https://github.com/agodianel/esp32-claude-workbench"


@dataclass(frozen=True)
class AgentTarget:
    """Where one agent looks for skills and MCP servers.

    MCP entries are ``(path, format)``; format is one of ``json-mcpServers``,
    ``json-servers`` (VS Code), ``json-copilot-cli`` or ``toml-codex``.
    """

    name: str
    project_skills: str
    user_skills: tuple[str, ...]
    project_mcp: tuple[str, str]
    user_mcp: tuple[str, str]


AGENTS: dict[str, AgentTarget] = {
    "claude": AgentTarget(
        "Claude Code",
        ".claude/skills",
        ("~/.claude/skills",),
        (".mcp.json", "json-mcpServers"),
        ("~/.claude.json", "json-mcpServers"),
    ),
    "codex": AgentTarget(
        "OpenAI Codex",
        ".agents/skills",
        ("~/.agents/skills",),
        (".codex/config.toml", "toml-codex"),
        ("~/.codex/config.toml", "toml-codex"),
    ),
    "cursor": AgentTarget(
        "Cursor",
        ".agents/skills",
        ("~/.agents/skills",),
        (".cursor/mcp.json", "json-mcpServers"),
        ("~/.cursor/mcp.json", "json-mcpServers"),
    ),
    "copilot": AgentTarget(
        "GitHub Copilot",
        ".agents/skills",
        ("~/.agents/skills",),
        (".vscode/mcp.json", "json-servers"),
        ("~/.copilot/mcp-config.json", "json-copilot-cli"),
    ),
    "gemini": AgentTarget(
        "Gemini CLI",
        ".agents/skills",
        ("~/.agents/skills",),
        (".gemini/settings.json", "json-mcpServers"),
        ("~/.gemini/settings.json", "json-mcpServers"),
    ),
    "antigravity": AgentTarget(
        "Google Antigravity",
        ".agents/skills",
        ("~/.gemini/config/skills", "~/.gemini/antigravity-cli/skills"),
        (".agents/mcp_config.json", "json-mcpServers"),
        ("~/.gemini/config/mcp_config.json", "json-mcpServers"),
    ),
}


def skills_source() -> Path:
    """Locate the bundled ``skills/`` directory (source checkout or installed wheel)."""
    if env := os.environ.get("ESP32WB_SKILLS_DIR"):
        return Path(env)
    packaged = Path(str(resources.files("esp32_workbench") / "skills"))
    if packaged.is_dir() and any(packaged.glob("*/SKILL.md")):
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "skills"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("Bundled skills not found; set ESP32WB_SKILLS_DIR")


def list_skills(source: Path | None = None) -> list[Path]:
    """Return every skill directory in the bundle."""
    source = source or skills_source()
    return sorted(p.parent for p in source.glob("*/SKILL.md"))


def _expand(path: str, project: Path) -> Path:
    return Path(path).expanduser() if path.startswith("~") else project / path


def _is_ours(skill_dir: Path) -> bool:
    skill_md = skill_dir / "SKILL.md"
    return skill_md.is_file() and SKILL_MARKER in skill_md.read_text(encoding="utf-8")


def install_skills(
    agents: list[str],
    scope: str = "project",
    project: Path | None = None,
    symlink: bool = False,
    force: bool = False,
    source: Path | None = None,
) -> list[str]:
    """Copy (or symlink) every bundled skill into each agent's skills directory.

    Agents that share a directory (e.g. ``.agents/skills``) are only written once.
    Existing skill folders are replaced only if they were installed by the workbench,
    unless ``force`` is set.

    Returns:
        Human-readable log lines.
    """
    project = (project or Path.cwd()).resolve()
    skills = list_skills(source)
    destinations: dict[Path, list[str]] = {}
    for agent in agents:
        target = AGENTS[agent]
        paths = (target.project_skills,) if scope == "project" else target.user_skills
        for p in paths:
            destinations.setdefault(_expand(p, project), []).append(target.name)

    log: list[str] = []
    for dest_root, names in destinations.items():
        dest_root.mkdir(parents=True, exist_ok=True)
        for skill in skills:
            dest = dest_root / skill.name
            if dest.exists() or dest.is_symlink():
                if not force and not dest.is_symlink() and not _is_ours(dest):
                    log.append(f"skip   {dest} (exists and was not installed by esp32-workbench)")
                    continue
                if dest.is_symlink() or dest.is_file():
                    dest.unlink()
                else:
                    shutil.rmtree(dest)
            if symlink:
                dest.symlink_to(skill.resolve(), target_is_directory=True)
            else:
                shutil.copytree(skill, dest)
        log.append(f"ok     {len(skills)} skills -> {dest_root}  ({', '.join(names)})")
    return log


def default_server_command() -> list[str]:
    """Command an agent should run to start the MCP server."""
    installed = shutil.which("esp32wb-mcp")
    # Under `uvx` the binary lives in uv's disposable cache (.../uv/archive-v0/...), which
    # can be garbage-collected; writing that path into a config would break later.
    if installed and "archive-v" not in installed:
        return [installed]
    return ["uvx", "--from", f"git+{REPO_URL}", "esp32wb-mcp"]


def _mcp_location(agent: str, scope: str) -> tuple[str, str]:
    target = AGENTS[agent]
    return target.project_mcp if scope == "project" else target.user_mcp


def server_entry(fmt: str, command: list[str]) -> dict:
    """Build one MCP server entry in the agent's dialect."""
    entry: dict = {"command": command[0], "args": command[1:]}
    if fmt == "json-servers":
        entry = {"type": "stdio", **entry}
    if fmt == "json-copilot-cli":
        entry = {"type": "local", **entry, "tools": ["*"]}
    return entry


def render_mcp_config(agent: str, scope: str = "project", command: list[str] | None = None) -> str:
    """Return the config snippet that registers the MCP server for ``agent``."""
    command = command or default_server_command()
    _, fmt = _mcp_location(agent, scope)
    if fmt == "toml-codex":
        args = ", ".join(json.dumps(a) for a in command[1:])
        return f"[mcp_servers.{SERVER_NAME}]\ncommand = {json.dumps(command[0])}\nargs = [{args}]\n"
    key = "servers" if fmt == "json-servers" else "mcpServers"
    return json.dumps({key: {SERVER_NAME: server_entry(fmt, command)}}, indent=2) + "\n"


def write_mcp_config(
    agent: str,
    scope: str = "project",
    project: Path | None = None,
    command: list[str] | None = None,
) -> Path:
    """Merge the MCP server entry into the agent's config file and return its path."""
    project = (project or Path.cwd()).resolve()
    rel, fmt = _mcp_location(agent, scope)
    path = _expand(rel, project)
    command = command or default_server_command()
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "toml-codex":
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        # The section runs until the next line that starts a new [table].
        section = re.compile(
            rf"^\[mcp_servers\.{re.escape(SERVER_NAME)}\]\n(?:(?!\[).*\n?)*", re.MULTILINE
        )
        snippet = render_mcp_config(agent, scope, command)
        updated = (
            section.sub(snippet, existing)
            if section.search(existing)
            else (existing + ("\n" if existing and not existing.endswith("\n") else "") + snippet)
        )
        path.write_text(updated, encoding="utf-8")
        return path

    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    key = "servers" if fmt == "json-servers" else "mcpServers"
    data.setdefault(key, {})[SERVER_NAME] = server_entry(fmt, command)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def print_log(lines: list[str]) -> None:
    for line in lines:
        print(line, file=sys.stdout)
