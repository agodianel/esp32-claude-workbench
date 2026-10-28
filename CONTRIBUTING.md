# Contributing to ESP32 Claude Workbench

Thank you for your interest in contributing! This guide will help you get started.

## What You Can Contribute

### New Skills
Skills follow the open [Agent Skills spec](https://agentskills.io/specification) so they work
in Claude Code, Codex, Cursor, Copilot, Gemini CLI and Antigravity:
1. Create `skills/esp32-your-skill/` — the folder name must equal the `name` field
   (lowercase letters, digits, hyphens).
2. Add `SKILL.md` with frontmatter `name`, `description` (what it does + "Use when ..." with
   trigger words, ≤ 1024 chars), `license: MIT` and `metadata: {source: esp32-workbench}`.
3. Add `agents/openai.yaml` (Codex display metadata).
4. Put long material in `references/` and link it relatively; never link to another skill.
5. `pytest tests/test_skills_spec.py` must pass.

### New Playbooks
Playbooks live in the `references/` folder of the skill that uses them:
1. Follow the existing structure (Trigger, Symptoms, Triage Steps, Resolution Checklist, Prevention).
2. Label chip-specific facts with the chip, or point to `esp32wb chip <target>`.
3. Include real log signatures and error patterns.

### Chip Data
Chip facts live in `src/esp32_workbench/data/chips/<target>.json`. Every value must come from
an Espressif primary source listed in `sources`; use `null` when a fact cannot be confirmed.

### New Project Templates
Add a reusable ESP-IDF template under `templates/`:
1. Include a minimal working project with `CMakeLists.txt`, `main/`, and `sdkconfig.defaults`.
2. Add a `README.md` explaining the template's purpose.
3. Follow ESP-IDF best practices.

### Tooling Improvements
Improve or add tools under `src/esp32_workbench/` (one module per tool, wired into `cli.py`
and `mcp_server.py`):
1. Standard library only, Python 3.9-compatible (the plugin launchers use stock `python3`).
2. Follow PEP 8 and use type hints.
2. Add corresponding tests in `tests/`.
3. Ensure the tool works without ESP32 hardware.

### Bug Fixes and Documentation
Always welcome! Please include test evidence when applicable.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/agodianel/esp32-claude-workbench.git
cd esp32-claude-workbench

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install for development
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes following the coding standards in `CLAUDE.md`.
3. Add or update tests as needed.
4. Run the full test suite: `pytest tests/ -v`
5. Update documentation if applicable.
6. Submit a PR with:
   - Clear description of the change.
   - Test evidence (command output or screenshots).
   - Any risk notes for embedded-specific changes.

## Code Style

- **Python**: PEP 8, type hints, `pathlib.Path` for files.
- **C/C++**: ESP-IDF coding style, `snake_case`, static for file-scoped functions.
- **Markdown**: ATX-style headers, consistent formatting, no trailing whitespace.

## Testing

All contributions must include tests where applicable:
- Repository tooling → `pytest` tests in `tests/`.
- New templates → validation tests for structure.
- New playbooks → integrity tests for required sections.
- New skills → `tests/test_skills_spec.py` validates the Agent Skills spec.

Run the full suite before submitting:
```bash
pytest tests/ -v --tb=short
```

## Questions?

Open an issue with the `question` label if you need help or clarification.
