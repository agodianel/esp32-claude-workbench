# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-28

### Breaking
- Skills renamed to kebab-case and moved to `skills/` to satisfy the open Agent Skills spec
  (`esp32_pin_audit` → `esp32-pin-audit`, …). `hardware_check` merged into `esp32-pin-audit`.
- `mcp/` folder replaced by the `esp32_workbench` package; start the server with `esp32wb-mcp`
  or install the plugin. `tools.*` imports are now `esp32_workbench.*`.
- `playbooks/` moved into each skill's `references/`; `boards/` into `esp32-pin-audit`.
- `docs_sources.yaml` / `scripts/index_esp_docs.py` replaced by `esp32wb docs index`.

### Added
- **Multi-chip support**: verified pin databases for ESP32, S2, S3, C2, C3, C5, C6, H2, P4 with
  Espressif source links; target auto-detected from `CONFIG_IDF_TARGET`.
- **Cross-agent distribution**: Claude Code plugin + marketplace (with a GPIO guard hook),
  Agent Plugins manifest (Codex, Copilot CLI, Cursor), Codex marketplace, Gemini CLI extension,
  Antigravity plugin, and `esp32wb install-skills` / `esp32wb init` for every agent.
- **`esp32wb` CLI** with `--json` on every command: `doctor`, `pins`, `chip`, `crash`,
  `build-log`, `partitions`, `sdkconfig`, `stacks`, `logs`, `docs`, `mission`, `contract`.
- **Crash decoder** for Xtensa and RISC-V panics, with `addr2line` resolution and cause classification.
- **Partition table checker** (alignment, overlaps, flash size, OTA, NVS, coredump, binary fit).
- **Build-log triage** with header → component hints and fixes.
- **Zero-dependency MCP server** (13 tools), tested against the official MCP SDK client.
- Two new skills: `esp32-build-fix`, `esp32-release-check`.

### Fixed
- Pin rules no longer assume the classic ESP32 (e.g. S3 GPIO 38 output was a false error,
  S3 flash pins 27–32 were missed). ADC2 + Wi-Fi is an error only where Espressif says it is
  unusable (classic ESP32); on S2/S3 it is a warning (arbitrated).
- `create_mission` output now passes `validate-mission`, and missions are written to the
  firmware project instead of the workbench checkout.
- Inconsistent rule sets between the MCP tools and the CLI tools (merged into one core).
- Commented-out code no longer produces pin findings.
- Reset reasons are read from the ROM's printed name, so they are correct on every chip.
- Docs search no longer unpickles index files and no longer needs torch to run.

## [0.1.1] - 2026-03-27

### Added
- **MCP Server backend (`mcp/mcp_server.py`)**: A local `FastMCP` implementation that exposes four development tools directory to Claude Code.
- **RAG for ESP32 Documentation (`scripts/index_esp_docs.py`)**: A local vector indexing and retrieval tool using FAISS, PyMuPDF, and SentenceTransformers to allow semantic offline search over ESP32 TRMs and Datasheets. 
- **Tool: `search_docs.py`**: Connects Claude to the local FAISS semantic index for querying hardware constraints directly.
- **Tool: `pin_audit.py`**: MCP tool designed to audit C/H files for dangerous GPIO combinations (Input-Only output mappings, ADC2 conflicts, Boot strapping rules).
- **Tool: `sdkconfig_check.py`**: MCP tool checking ESP-IDF `sdkconfig` files for required production-readiness flags like Watchdog timers, Panic behaviors, and buffer allocations.
- **Tool: `mission_generator.py`**: Automates generating state-tracked markdown documents for new firmware tasks via MCP.
- **Hardware Limitations Playbook**: `.claude/skills/hardware_check/SKILL.md` skill, `playbooks/hardware_limitations.md`, and default board layouts under `boards/`.
- Tests for MCP tools achieving >90% code coverage.
