---
name: esp32-feature-contract
description: Creates a mission file and an implementation contract (scope, affected files, risks, tests, rollback) that must be accepted before any ESP32 firmware code is changed. Use when the user asks to "add a feature", "implement", "fix a bug", "refactor", "write a contract", "start a mission", "plan this change", "/contract", or "resume the mission" in an ESP-IDF, ESP32, ESP32-S3, ESP32-C3 or ESP32-C6 project, and before editing any firmware source.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Feature Contract

No firmware edit without an accepted contract. The workflow is:

**mission → contract → test plan → implement → test evidence → PR**

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.

| Need | MCP tool | CLI |
|------|----------|-----|
| New mission file in `missions/` | `create_mission` | `esp32wb mission new "<name>" -d "<desc>" --target T --board B` |
| Check a mission has all sections | `validate_mission` | `esp32wb mission validate missions/<file>.md [--json]` |
| Draft a contract | `generate_contract` | `esp32wb contract "<summary>" --files "path\|ACTION\|desc" --risks "..." --tests "..." --rollback "..." [-o file]` |
| Pin / peripheral risk facts | `pin_audit`, `chip_info` | `esp32wb pins .`, `esp32wb chip <target>` |
| Whole-project baseline | `project_doctor` | `esp32wb doctor .` |

Templates if the tools are unavailable:
[mission template](assets/mission_template.md) and
[contract template](assets/implementation_contract.md).

## Steps

### 1. Open or create the mission

- If `missions/` has an active mission for this work, read it and resume from
  **Current Status** / **Next Step**.
- Otherwise create one (tool above). Fill **Board / Target** from
  `CONFIG_IDF_TARGET`, not from assumption. If the target is unknown, ask.

### 2. Gather requirements

- Requested change and expected behaviour after it.
- Target chip, board, ESP-IDF version.
- Hardware involved: which pins, buses, peripherals. Run `esp32wb pins .` and
  `esp32wb chip <target>` before proposing any GPIO; never quote pin limits from memory.
- For peripheral, ADC, timer, Wi-Fi or BLE work, search the docs first
  (`search_esp_docs` / `esp32wb docs search "<query>" --target T`).

### 3. Write the contract

Size the contract to the change:

| Change size | Contract |
|-------------|----------|
| Tiny (typo, log text, constant tweak, one file, no API/pin/timing impact) | **Brief**: summary, files, one-line risk, test, rollback |
| Normal | Full template |
| Touches boot, partitions, OTA, flash encryption, eFuses, strapping pins | Full template + explicit Boot risk = HIGH + hardware test |

Every contract contains:

1. **Change summary** — one paragraph.
2. **Non-goals** — what is explicitly out of scope.
3. **Affected files** — every file created/modified/deleted (`path|ACTION|description`).
4. **APIs touched** — public functions, Kconfig options, NVS keys, protocols.
5. **Risk assessment** (LOW/MED/HIGH with reasons):
   - Concurrency — shared state, ISRs, task priorities, core affinity.
   - Peripheral — pin conflicts, strapping/flash/PSRAM/USB pins **for this chip**, bus contention.
   - Memory — heap, stack (bytes), DMA-capable / internal RAM, PSRAM.
   - Boot — init order, strapping pins, partition table, bootloader config.
   - Backward compatibility — API, wire protocol, NVS schema, OTA from older versions.
6. **Test plan** — new/updated tests, no-hardware vs hardware tests, evidence required.
   For anything non-trivial hand off to the `esp32-test-plan` skill.
7. **Rollback** — how to revert, and what persistent state (NVS, filesystem, OTA
   slot, eFuses — the last are irreversible) must be considered.
8. **Acceptance criteria** — measurable.

### 4. Get acceptance

Present the contract and **stop**. Do not edit firmware until the user accepts it.
Record acceptance (date, who) in the mission **History** table.

### 5. During implementation

- Stay inside the contracted file list. If scope grows, update the contract and
  get re-acceptance before continuing.
- Update the mission **Current Status** and **Next Step** as you go.

### 6. After implementation

- Record test commands and results (pass/fail with output) in the mission.
- List what was **not** tested and why.
- Run `esp32wb mission validate` on the mission file.
- Prepare the PR with the `esp32-pr-prepare` skill, linking mission + contract.

## Rules

1. Never skip the contract; a brief one is fine for tiny changes.
2. Never edit code before acceptance.
3. Never invent hardware constraints — cite `esp32wb chip` output or Espressif docs.
4. Changes that burn eFuses or enable secure boot / flash encryption require an
   explicit, separate confirmation from the user: they are irreversible.
