# AGENTS.md — contributing to ESP32 Workbench

Instructions for AI coding agents working **on this repository**. (The rules for agents
working on ESP32 firmware ship separately — `esp32wb init` writes them into a firmware
project from `src/esp32_workbench/data/project/`.)

## Layout

| Path | What |
|------|------|
| `src/esp32_workbench/` | Python package: one module per tool, `cli.py` (`esp32wb`), `mcp_server.py` |
| `src/esp32_workbench/data/chips/*.json` | Per-chip GPIO facts with Espressif source URLs |
| `skills/<name>/` | Agent Skills (spec: agentskills.io) — shipped to every agent |
| `bin/` | No-install launchers (`esp32wb`, `esp32wb-mcp`) used by plugins |
| `hooks/` | Claude Code plugin hooks |
| `.claude-plugin/`, `plugin.json`, `mcp.json`, `.agents/plugins/`, `gemini-extension.json` | Plugin manifests per agent |
| `templates/` | ESP-IDF example projects |
| `missions/` | Task state for work on this repo |

## Rules

1. **Core stays standard-library only and Python 3.9-compatible.** Optional features go
   behind extras (`[docs]`, `[semantic]`). No `zip(strict=)`, `match`, or other 3.10+ syntax.
2. **Never invent hardware facts.** Chip data needs a primary-source URL in `sources`;
   unknown values are `null`.
3. **Every tool gets three front-ends**: a module with `add_arguments()` / `run()`, a
   subcommand in `cli.py`, and an MCP tool in `mcp_server.py`. All support `--json`.
4. **Skills must pass `tests/test_skills_spec.py`**: `name` equals the folder name, kebab-case,
   description ≤ 1024 chars, metadata marker `source: esp32-workbench`, relative links only.
5. Work through a mission file + implementation contract for non-trivial changes.

## Commands

```bash
uv sync --extra dev
uv run pytest                      # full suite
uv run ruff check . && uv run ruff format --check .
uv run esp32wb doctor templates/wifi-station
claude plugin validate .           # Claude Code plugin + marketplace
```
