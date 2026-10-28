"""Validate every bundled skill against the Agent Skills spec (agentskills.io/specification).

Spec-compliant skills load in Claude Code, Codex, Cursor, Copilot, Gemini CLI and Antigravity.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SKILLS = sorted(p.parent for p in (ROOT / "skills").glob("*/SKILL.md"))
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
LINK_RE = re.compile(r"\]\(([^)#]+)\)")


def frontmatter(skill: Path) -> dict[str, str]:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if match := re.match(r"^([a-z_-]+):\s*(.*)$", line):
            fields[match.group(1)] = match.group(2).strip().strip('"')
    fields["_raw"] = block
    return fields


def test_expected_skills_present():
    names = {s.name for s in SKILLS}
    expected = {
        "esp32-repo-scout",
        "esp32-feature-contract",
        "esp32-pin-audit",
        "esp32-arch-review",
        "esp32-test-plan",
        "esp32-log-triage",
        "esp32-crash-review",
        "esp32-build-fix",
        "esp32-release-check",
        "esp32-peripheral-bringup",
        "esp32-pr-prepare",
    }
    assert expected <= names


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_name_matches_folder(skill: Path):
    name = frontmatter(skill)["name"]
    assert name == skill.name
    assert NAME_RE.match(name) and len(name) <= 64


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_description(skill: Path):
    description = frontmatter(skill)["description"]
    assert 50 <= len(description) <= 1024
    assert "use when" in description.lower(), "description should say when to use the skill"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_ownership_marker_and_license(skill: Path):
    fields = frontmatter(skill)
    assert "source: esp32-workbench" in fields["_raw"]
    assert fields.get("license") == "MIT"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_no_claude_only_fields(skill: Path):
    raw = frontmatter(skill)["_raw"]
    for field in ("allowed-tools:", "disable-model-invocation:", "context:", "hooks:"):
        assert field not in raw, f"{field} is not portable across agents"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_body_size(skill: Path):
    assert len((skill / "SKILL.md").read_text(encoding="utf-8").splitlines()) <= 500


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_relative_links_resolve_inside_skill(skill: Path):
    for md in skill.rglob("*.md"):
        for link in LINK_RE.findall(md.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target = (md.parent / link).resolve()
            assert target.exists(), f"{md.relative_to(ROOT)} -> {link} is broken"
            assert skill.resolve() in target.parents or target == skill.resolve(), (
                f"{md.relative_to(ROOT)} -> {link} leaves the skill folder"
            )


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_codex_metadata(skill: Path):
    meta = skill / "agents" / "openai.yaml"
    assert meta.is_file()
    text = meta.read_text(encoding="utf-8")
    assert "display_name:" in text and "short_description:" in text


def test_plugin_manifests_agree():
    claude = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    portable = json.loads((ROOT / "plugin.json").read_text())
    gemini = json.loads((ROOT / "gemini-extension.json").read_text())
    import esp32_workbench

    assert claude["name"] == portable["name"] == gemini["name"] == "esp32-workbench"
    assert (
        claude["version"] == portable["version"] == gemini["version"] == esp32_workbench.__version__
    )
