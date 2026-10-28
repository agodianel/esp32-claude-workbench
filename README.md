<div align="center">

# ⚡ ESP32 Workbench

![ESP32 Workbench Architecture](assets/banner_flow.png)

**Hardware-aware skills and tools for AI-assisted ESP32 firmware development.**
Works with Claude Code, OpenAI Codex, Cursor, GitHub Copilot, Gemini CLI and Google Antigravity.

[![CI](https://github.com/agodianel/esp32-claude-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/agodianel/esp32-claude-workbench/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-spec%20compliant-purple.svg)](https://agentskills.io/specification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Stop prompting. Start engineering.*

[Install](#-install) · [Skills](#-skills) · [Tools](#-tools) · [Chips](#-supported-chips) · [Workflow](#-workflow) · [Upgrading from 0.1](#-upgrading-from-01)

</div>

---

## 🎯 What it does

AI agents write plausible firmware that breaks on real hardware: a flash pin used as chip
select, ADC2 read while Wi-Fi is on, a 2 KB stack for a TLS task, a GPIO that only exists on a
different chip. The workbench gives your agent **deterministic tools and verified chip data**
so it checks instead of guessing, plus a **disciplined workflow** (mission → contract → test
plan → evidence → PR) so changes stay reviewable.

- **11 skills** that follow the open [Agent Skills](https://agentskills.io) standard, so one set works in every major coding agent.
- **9 chips**: ESP32, S2, S3, C2, C3, C5, C6, H2 and P4. Pin facts come from Espressif docs, with a source link for each.
- **13 MCP tools and an `esp32wb` CLI**: pin audit, crash decoder, build-log triage, partition and sdkconfig review, docs search and more.
- **Zero dependencies.** The core is standard-library Python, and the MCP server runs on plain `python3` with no `pip install`.
- **Guard hook for Claude Code**: every edit to a `.c`/`.h` file is pin-audited, and errors go straight back to the agent.

---

## 📦 Install

Pick your agent. Every route installs the same skills; the plugin routes also register the
MCP server.

### Claude Code: plugin (skills + MCP + hook)

```text
/plugin marketplace add agodianel/esp32-claude-workbench
/plugin install esp32-workbench@esp32-workbench
```

On Claude Code 2.1.275 or later this is one step:
`/plugin install esp32-workbench --marketplace agodianel/esp32-claude-workbench`

Skills are then available as `/esp32-workbench:<skill>`, for example `/esp32-workbench:esp32-crash-review`.

### OpenAI Codex: plugin

```bash
codex plugin marketplace add agodianel/esp32-claude-workbench
```

Then open `/plugins` in the Codex CLI and install **ESP32 Workbench**. Invoke skills with `$esp32-pin-audit` or let Codex pick them.

### Gemini CLI: extension

```bash
gemini extensions install https://github.com/agodianel/esp32-claude-workbench
```

### Google Antigravity: plugin

```bash
agy plugin install https://github.com/agodianel/esp32-claude-workbench
```

### Cursor, GitHub Copilot, or any Agent Skills client

```bash
# Skills only, using an installer you probably already have
npx skills add agodianel/esp32-claude-workbench
gh skill install agodianel/esp32-claude-workbench --agent github-copilot   # or cursor, codex, gemini-cli, antigravity

# Skills + MCP config for any combination of agents, in one step
uvx --from git+https://github.com/agodianel/esp32-claude-workbench \
  esp32wb install-skills --agent cursor copilot --with-mcp
```

### Set up a firmware project in one command

From your ESP-IDF project root:

```bash
uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb init . --agents all
```

This writes `AGENTS.md` (firmware rules every agent reads) and `CLAUDE.md` (which imports
it), creates `missions/`, installs the skills into `.claude/skills/` and `.agents/skills/`,
and registers the MCP server for each agent.

<details>
<summary>Where things go, per agent</summary>

| Agent | Project skills | User skills | MCP config written by `--with-mcp` |
|-------|---------------|-------------|------------------------------------|
| Claude Code | `.claude/skills/` | `~/.claude/skills/` | `.mcp.json` |
| Codex | `.agents/skills/` | `~/.agents/skills/` | `.codex/config.toml` / `~/.codex/config.toml` |
| Cursor | `.agents/skills/` | `~/.agents/skills/` | `.cursor/mcp.json` |
| GitHub Copilot | `.agents/skills/` | `~/.agents/skills/` | `.vscode/mcp.json` / `~/.copilot/mcp-config.json` |
| Gemini CLI | `.agents/skills/` | `~/.agents/skills/` | `.gemini/settings.json` |
| Antigravity | `.agents/skills/` | `~/.gemini/config/skills/`, `~/.gemini/antigravity-cli/skills/` | `.agents/mcp_config.json` |

Print a config snippet without writing anything: `esp32wb mcp-config codex`.
</details>

---

## 🛠️ Skills

| Skill | Use it when… |
|-------|--------------|
| `esp32-repo-scout` | Starting on an unfamiliar ESP-IDF project |
| `esp32-feature-contract` | Before any change: creates the mission and implementation contract |
| `esp32-pin-audit` | Adding or changing GPIO, I2C, SPI, UART, ADC or LEDC pins; choosing pins for a board |
| `esp32-peripheral-bringup` | Bringing up an I2C/SPI/UART/ADC device that doesn't respond |
| `esp32-arch-review` | Reviewing RTOS safety, memory, error handling before merge |
| `esp32-test-plan` | Planning host-side, Unity and on-target tests |
| `esp32-build-fix` | `idf.py build` or flashing fails |
| `esp32-crash-review` | Guru Meditation, abort, assert, stack overflow, watchdog reset |
| `esp32-log-triage` | Reading a serial log: boot loops, Wi-Fi/BLE drops, resets |
| `esp32-release-check` | Pre-release: sdkconfig, partitions, OTA, security settings |
| `esp32-pr-prepare` | Writing the PR with test evidence, risks and rollback |

Each skill tells the agent which tool to run, and includes debugging playbooks in its
`references/` folder.

---

## 🔧 Tools

The same tools are available as an MCP server (`esp32wb-mcp`) and a CLI (`esp32wb`). Every
command accepts `--json`.

| CLI | MCP tool | What it does |
|-----|----------|--------------|
| `esp32wb doctor [path]` | `project_doctor` | Runs every check below on a project and gives one report |
| `esp32wb pins <path>` | `pin_audit` | Flags flash/PSRAM, strapping, USB, UART0 and input-only pin misuse, ADC2+Wi-Fi use and double assignments, for the project's chip |
| `esp32wb chip <target>` | `chip_info` | Verified pin facts for a chip, with Espressif source links |
| `esp32wb crash <log> --elf app.elf` | `crash_decode` | Decodes Xtensa and RISC-V panics, runs `addr2line` and classifies the cause (NULL dereference, stack overflow, heap corruption…) |
| `esp32wb build-log <log>` | `build_log_triage` | Missing headers and the component to add, linker errors, IRAM/DRAM overflow, partition too small, target mismatch, serial-port problems |
| `esp32wb partitions <csv>` | `partition_check` | Alignment, overlaps, flash size, OTA slots and otadata, NVS, coredump, whether the app binary fits |
| `esp32wb sdkconfig <file>` | `sdkconfig_review` | Security, stability and production-readiness review |
| `esp32wb stacks <path>` | `stack_check` | Undersized FreeRTOS task stacks |
| `esp32wb logs <log>` | `log_summary` | Serial log summary with reset reason |
| `esp32wb docs search "<q>"` | `search_esp_docs` | Local search over Espressif TRMs and datasheets, with page citations |
| `esp32wb mission new/validate` | `create_mission` / `validate_mission` | Persistent task state |
| `esp32wb contract "<summary>"` | `generate_contract` | Implementation contract |

```bash
idf.py build 2>&1 | esp32wb build-log -          # pipe straight from the build
esp32wb crash monitor.log --elf build/app.elf     # decode with the right toolchain for the chip
```

### Docs search

Searching needs no dependencies; building the index needs `pypdf`:

```bash
pip install 'esp32-workbench[docs]'
esp32wb docs index                              # downloads the ESP32/S3/C3/C6 TRMs + datasheets
esp32wb docs index --file my_sensor.pdf --no-builtin   # add your own datasheets
```

Search uses SQLite full-text search (BM25). For embedding-based search, install
`esp32-workbench[semantic]` and run `esp32wb docs index --semantic`.

---

## 🧩 Supported chips

`esp32` · `esp32s2` · `esp32s3` · `esp32c2` · `esp32c3` · `esp32c5` · `esp32c6` · `esp32h2` · `esp32p4`

The target is read from `CONFIG_IDF_TARGET` in `sdkconfig` / `sdkconfig.defaults`, or set it
with `--target`. Each chip file in [`src/esp32_workbench/data/chips/`](src/esp32_workbench/data/chips/)
lists the Espressif pages its facts came from. Facts that couldn't be confirmed are left
empty rather than guessed, and the notes record where Espressif's own sources disagree.

---

## 🔄 Workflow

```text
1. Issue: "Wi-Fi drops after 30 minutes on the S3 board"
2. esp32-repo-scout        → components, tasks, pins, target
3. esp32-feature-contract  → missions/2026-10-28-wifi-stability.md + contract (scope, risks, rollback)
4. Review and accept the contract
5. esp32-test-plan         → host / Unity / on-target cases
6. Implement               → pin guard hook audits every C/H edit (Claude Code)
7. esp32wb doctor .        → pins, stacks, sdkconfig, partitions
8. esp32-arch-review       → RTOS safety, memory, error paths
9. esp32-pr-prepare        → PR with test evidence, risk notes, rollback
10. Mission → DONE
```

---

## 🧪 Development

```bash
git clone https://github.com/agodianel/esp32-claude-workbench.git
cd esp32-claude-workbench
uv sync --extra dev
uv run pytest                 # 270+ tests, no hardware needed
uv run ruff check . && uv run ruff format --check .
claude plugin validate .      # Claude Code plugin + marketplace
```

| Layer | What | Hardware? |
|-------|------|-----------|
| Repository | Tools, chip data, skills spec, manifests, MCP protocol (including interop with the official MCP SDK) | No |
| Host-side C | Pure logic compiled natively | No |
| ESP-IDF Unity | Components in isolation | Yes |
| Target | `pytest-embedded`, marked `@pytest.mark.hardware` | Yes |

Repository layout: [`AGENTS.md`](AGENTS.md). Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Project templates

[`templates/`](templates/) holds ready-to-build ESP-IDF projects (`esp-idf-basic`,
`wifi-station`, `mqtt-node`). CI runs `esp32wb doctor` on each one.

---

## ⬆️ Upgrading from 0.1

| 0.1 | 0.2 |
|-----|-----|
| `.claude/skills/esp32_pin_audit` etc. | `skills/esp32-pin-audit` (kebab-case, as the open spec requires) |
| `hardware_check` skill | merged into `esp32-pin-audit` |
| `playbooks/*.md` | `skills/<skill>/references/*.md` |
| `python mcp/mcp_server.py` + `mcp/requirements.txt` | `esp32wb-mcp` (no dependencies) or the plugin |
| `python scripts/index_esp_docs.py` | `esp32wb docs index` |
| `tools.*` Python package | `esp32_workbench.*` |
| `scan-pins`, `analyze-sdkconfig`, … | still work; `esp32wb <command>` is the new front door |

If you registered the old server with `claude mcp add esp32-workbench -- python …/mcp/mcp_server.py`,
remove it (`claude mcp remove esp32-workbench`) and install the plugin instead.

---

## Star History

<a href="https://www.star-history.com/?repos=agodianel%2Fesp32-claude-workbench&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=agodianel/esp32-claude-workbench&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=agodianel/esp32-claude-workbench&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=agodianel/esp32-claude-workbench&type=date&legend=top-left" />
 </picture>
</a>

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">

**ESP32 Workbench** — *Structure over magic. Testing over claims. Workflows over prompts.*

</div>
