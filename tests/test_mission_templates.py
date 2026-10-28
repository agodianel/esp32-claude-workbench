"""Tests for mission template validation."""

from pathlib import Path

from esp32_workbench.mission import REQUIRED_SECTIONS, extract_sections, validate_mission

# Path to the project root
ROOT = Path(__file__).parent.parent


class TestExtractSections:
    """Test the section extraction from markdown."""

    def test_extracts_h2_sections(self):
        content = "# Title\n## Section One\ntext\n## Section Two\ntext"
        sections = extract_sections(content)
        assert sections == ["Section One", "Section Two"]

    def test_ignores_h1_and_h3(self):
        content = "# Title\n## Keep This\n### Ignore This\ntext"
        sections = extract_sections(content)
        assert sections == ["Keep This"]

    def test_empty_content(self):
        sections = extract_sections("")
        assert sections == []

    def test_no_sections(self):
        content = "Just plain text\nwithout any headers"
        sections = extract_sections(content)
        assert sections == []


class TestValidateMission:
    """Test mission file validation."""

    def test_valid_mission_has_all_sections(self):
        # Build content with all required sections
        lines = ["# Test Mission"]
        for section in REQUIRED_SECTIONS:
            lines.append(f"\n## {section}\nContent here.")
        content = "\n".join(lines)

        result = validate_mission(content)
        assert result["missing_required"] == []

    def test_missing_section_detected(self):
        content = "# Mission\n## Goal\nDo something."
        result = validate_mission(content)
        assert len(result["missing_required"]) > 0
        assert "Goal" not in result["missing_required"]

    def test_case_insensitive_matching(self):
        lines = ["# Test Mission"]
        for section in REQUIRED_SECTIONS:
            lines.append(f"\n## {section.lower()}\nContent here.")
        content = "\n".join(lines)

        result = validate_mission(content)
        assert result["missing_required"] == []

    def test_found_sections_are_listed(self):
        content = "# Mission\n## Goal\ntext\n## Constraints\ntext"
        result = validate_mission(content)
        assert "Goal" in result["found"]
        assert "Constraints" in result["found"]


class TestMissionTemplateFile:
    """Test that the actual mission template file is valid."""

    def test_mission_template_has_all_required_sections(self):
        template_path = (
            ROOT / "skills" / "esp32-feature-contract" / "assets" / "mission_template.md"
        )
        assert template_path.exists(), f"Mission template not found at {template_path}"

        content = template_path.read_text(encoding="utf-8")
        result = validate_mission(content)
        assert result["missing_required"] == [], (
            f"Mission template missing required sections: {result['missing_required']}"
        )

    def test_example_mission_has_all_required_sections(self):
        example_path = ROOT / "missions" / "examples" / "wifi-reconnect-mission.md"
        assert example_path.exists(), f"Example mission not found at {example_path}"

        content = example_path.read_text(encoding="utf-8")
        result = validate_mission(content)
        assert result["missing_required"] == [], (
            f"Example mission missing required sections: {result['missing_required']}"
        )
