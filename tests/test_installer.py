"""Tests for skill installation and MCP config generation."""

import json
from pathlib import Path

import pytest

from esp32_workbench import installer


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    src = tmp_path / "bundle"
    for name in ("esp32-pin-audit", "esp32-crash-review"):
        (src / name).mkdir(parents=True)
        (src / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: x\nmetadata:\n  {installer.SKILL_MARKER}\n---\n"
        )
    return src


def test_project_install_dedupes_shared_dirs(tmp_path: Path, bundle: Path):
    project = tmp_path / "proj"
    log = installer.install_skills(list(installer.AGENTS), "project", project, source=bundle)
    assert (project / ".claude/skills/esp32-pin-audit/SKILL.md").is_file()
    assert (project / ".agents/skills/esp32-crash-review/SKILL.md").is_file()
    assert len([line for line in log if line.startswith("ok")]) == 2  # .claude + .agents


def test_user_scope_antigravity_paths(tmp_path: Path, bundle: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    installer.install_skills(["antigravity"], "user", tmp_path, source=bundle)
    assert (tmp_path / "home/.gemini/config/skills/esp32-pin-audit/SKILL.md").is_file()
    assert (tmp_path / "home/.gemini/antigravity-cli/skills/esp32-pin-audit/SKILL.md").is_file()


def test_does_not_clobber_foreign_skill(tmp_path: Path, bundle: Path):
    foreign = tmp_path / ".agents/skills/esp32-pin-audit"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: esp32-pin-audit\ndescription: mine\n---\n")
    log = installer.install_skills(["codex"], "project", tmp_path, source=bundle)
    assert any(line.startswith("skip") for line in log)
    assert "mine" in (foreign / "SKILL.md").read_text()
    installer.install_skills(["codex"], "project", tmp_path, force=True, source=bundle)
    assert installer.SKILL_MARKER in (foreign / "SKILL.md").read_text()


def test_reinstall_updates_own_skills_and_symlinks(tmp_path: Path, bundle: Path):
    installer.install_skills(["claude"], "project", tmp_path, source=bundle)
    installer.install_skills(["claude"], "project", tmp_path, symlink=True, source=bundle)
    assert (tmp_path / ".claude/skills/esp32-pin-audit").is_symlink()


def test_render_formats():
    cmd = ["esp32wb-mcp"]
    assert "[mcp_servers.esp32-workbench]" in installer.render_mcp_config("codex", command=cmd)
    vscode = json.loads(installer.render_mcp_config("copilot", "project", cmd))
    assert vscode["servers"]["esp32-workbench"]["type"] == "stdio"
    cli = json.loads(installer.render_mcp_config("copilot", "user", cmd))
    assert cli["mcpServers"]["esp32-workbench"] == {
        "type": "local",
        "command": "esp32wb-mcp",
        "args": [],
        "tools": ["*"],
    }
    gemini = json.loads(
        installer.render_mcp_config("gemini", command=["uvx", "--from", "x", "esp32wb-mcp"])
    )
    assert gemini["mcpServers"]["esp32-workbench"]["args"] == ["--from", "x", "esp32wb-mcp"]


def test_write_json_config_merges(tmp_path: Path):
    cfg = tmp_path / ".cursor/mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    installer.write_mcp_config("cursor", "project", tmp_path, ["esp32wb-mcp"])
    data = json.loads(cfg.read_text())
    assert set(data["mcpServers"]) == {"other", "esp32-workbench"}


def test_write_codex_toml_is_idempotent(tmp_path: Path):
    cfg = tmp_path / ".codex/config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "gpt"\n\n[mcp_servers.other]\ncommand = "x"\n')
    installer.write_mcp_config("codex", "project", tmp_path, ["a"])
    installer.write_mcp_config("codex", "project", tmp_path, ["b"])
    text = cfg.read_text()
    assert text.count("[mcp_servers.esp32-workbench]") == 1
    assert 'command = "b"' in text and "[mcp_servers.other]" in text


def test_bundled_skills_found():
    assert installer.list_skills()


def test_server_command_avoids_uv_cache(monkeypatch):
    monkeypatch.setattr(
        installer.shutil, "which", lambda _: "/h/.cache/uv/archive-v0/x/bin/esp32wb-mcp"
    )
    assert installer.default_server_command()[0] == "uvx"
    monkeypatch.setattr(installer.shutil, "which", lambda _: "/usr/local/bin/esp32wb-mcp")
    assert installer.default_server_command() == ["/usr/local/bin/esp32wb-mcp"]
