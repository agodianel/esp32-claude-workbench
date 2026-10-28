"""Tests for playbook file integrity and structure.

Playbooks live in the ``references/`` folder of the skill that uses them.
"""

from pathlib import Path

import pytest

# Path to the project root
ROOT = Path(__file__).parent.parent

# Expected playbook files and the skill that owns each one
PLAYBOOK_SKILLS = {
    "build_failure.md": "esp32-build-fix",
    "guru_meditation.md": "esp32-crash-review",
    "wifi_debug.md": "esp32-log-triage",
    "i2c_bringup.md": "esp32-peripheral-bringup",
    "watchdog_reset.md": "esp32-crash-review",
    "memory_review.md": "esp32-arch-review",
    "ota_update.md": "esp32-release-check",
    "ble_debug.md": "esp32-log-triage",
    "spi_bringup.md": "esp32-peripheral-bringup",
}
EXPECTED_PLAYBOOKS = list(PLAYBOOK_SKILLS)


def playbook(filename: str) -> Path:
    """Return the path of a playbook inside its skill's references folder."""
    return ROOT / "skills" / PLAYBOOK_SKILLS[filename] / "references" / filename


# Required sections in every playbook
REQUIRED_PLAYBOOK_SECTIONS = [
    "Trigger",
    "Symptoms",
    "Triage Steps",
    "Resolution Checklist",
    "Prevention",
]


def extract_h2_sections(content: str) -> list[str]:
    """Extract H2 section titles from markdown."""
    import re

    return [match.group(1).strip() for match in re.finditer(r"^##\s+(.+)$", content, re.MULTILINE)]


class TestPlaybookFilesExist:
    """Test that all expected playbook files exist."""

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_exists(self, filename: str):
        playbook_path = playbook(filename)
        assert playbook_path.exists(), f"Playbook not found: {playbook_path}"

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_is_not_empty(self, filename: str):
        playbook_path = playbook(filename)
        content = playbook_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100, f"Playbook {filename} appears to be empty or too short"


class TestPlaybookStructure:
    """Test that playbooks have required structure."""

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_has_title(self, filename: str):
        playbook_path = playbook(filename)
        content = playbook_path.read_text(encoding="utf-8")
        assert content.strip().startswith("# "), (
            f"Playbook {filename} should start with an H1 title"
        )

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_has_required_sections(self, filename: str):
        playbook_path = playbook(filename)
        content = playbook_path.read_text(encoding="utf-8")
        found_sections = extract_h2_sections(content)
        found_lower = [s.lower() for s in found_sections]

        for required in REQUIRED_PLAYBOOK_SECTIONS:
            assert required.lower() in found_lower, (
                f"Playbook {filename} missing required section: '{required}'. "
                f"Found sections: {found_sections}"
            )


class TestPlaybookContent:
    """Test playbook content quality."""

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_has_checklist(self, filename: str):
        """Playbooks should have at least one checklist item."""
        playbook_path = playbook(filename)
        content = playbook_path.read_text(encoding="utf-8")
        assert "- [ ]" in content, f"Playbook {filename} should contain at least one checklist item"

    @pytest.mark.parametrize("filename", EXPECTED_PLAYBOOKS)
    def test_playbook_minimum_length(self, filename: str):
        """Playbooks should have substantial content."""
        playbook_path = playbook(filename)
        content = playbook_path.read_text(encoding="utf-8")
        lines = [line for line in content.splitlines() if line.strip()]
        assert len(lines) >= 20, (
            f"Playbook {filename} has only {len(lines)} non-empty lines, expected >= 20"
        )
