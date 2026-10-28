---
name: esp32-repo-scout
description: Scans an ESP-IDF or PlatformIO firmware repository and produces a scout report of target chip, components, dependencies, pin usage, FreeRTOS tasks and risks before any change is planned. Use when the user says "scout the repo", "explain this firmware project", "what does this ESP32 project do", "map the components", "onboard me", "where is app_main", or before writing a mission or implementation contract on an unfamiliar ESP32, ESP32-S3, ESP32-C3, ESP32-C6 or other Espressif codebase.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Repo Scout

Build an accurate map of an ESP32 firmware project **before** proposing changes.
Read-only: this skill never edits files.

## Tools

Deterministic tools beat guessing. If the `esp32-workbench` MCP server is connected,
call its tools; otherwise run the `esp32wb` CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Analysis commands accept `--json`.

| Need | MCP tool | CLI |
|------|----------|-----|
| All static checks at once | `project_doctor` | `esp32wb doctor . [--target T]` |
| GPIO usage for the detected chip | `pin_audit` | `esp32wb pins .` |
| Verified pin facts for the chip | `chip_info` | `esp32wb chip <target>` |
| Task stack sizes | `stack_check` | `esp32wb stacks .` |

`esp32wb doctor .` is the fastest first step: it detects the target from
`CONFIG_IDF_TARGET` and runs pins, stacks, sdkconfig and partition checks together.

## Steps

### 1. Identify build system and target

- `CMakeLists.txt` with `include($ENV{IDF_PATH}/tools/cmake/project.cmake)` → ESP-IDF CMake project.
- `platformio.ini` → PlatformIO (check `framework = espidf` vs `arduino`, and `board =`).
- `Makefile` with `PROJECT_NAME` + `component.mk` → legacy GNU Make ESP-IDF (pre-v4); flag as a risk.
- Target chip: `grep CONFIG_IDF_TARGET sdkconfig sdkconfig.defaults*`. If absent, the
  default is classic `esp32` — say so explicitly instead of assuming the board.
- ESP-IDF version: `idf.py --version`, `dependencies.idf` in `main/idf_component.yml`,
  CI files, or `.devcontainer`. Record it; APIs differ a lot between v4.x and v5.x.

### 2. Map components

```bash
find . -name CMakeLists.txt -not -path "./build/*" -not -path "./managed_components/*"
grep -rn "idf_component_register" --include=CMakeLists.txt . | grep -v build/
```

For each component record: name, public headers (`include/`), `REQUIRES` /
`PRIV_REQUIRES`, and whether it has a `test/` folder or host tests.

### 3. Hardware configuration

- `sdkconfig.defaults*` (per-target variants like `sdkconfig.defaults.esp32s3`).
- Pin definitions: `GPIO_NUM_`, `#define .*_(PIN|GPIO|SDA|SCL|MOSI|MISO|CLK|CS)`, Kconfig `CONFIG_*_GPIO`.
- Peripherals: grep for `i2c_`, `spi_`, `uart_`, `ledc_`, `adc_`, `rmt_`, `i2s_`, `esp_wifi_`, `nimble`/`esp_bt`.
- Board: look for a `pin_map.h`, README, or schematic reference. Do not infer the board
  from pin numbers alone.
- Run `esp32wb pins .` and `esp32wb chip <target>` rather than recalling pin limits from memory.

### 4. Task architecture

```bash
grep -rnE "xTaskCreate(PinnedToCore|Static)?\(" --include=*.c --include=*.cpp .
```

Record task name, stack size (ESP-IDF stack depth is in **bytes**), priority, and core
affinity. `esp32wb stacks .` flags undersized stacks for you.

### 5. External dependencies

`idf_component.yml` / `dependencies.lock`, `managed_components/`, `.gitmodules`,
vendored `third_party/` or `lib/`, PlatformIO `lib_deps`.

### 6. Existing workflow state

- `missions/` — is there an active mission? Read it first and resume from its *Next Step*.
- `tests/`, `pytest.ini`, `test/` inside components, CI workflows.

## Output: Scout Report

Write to the active mission file if one exists, otherwise print:

```markdown
# Scout Report — <project>
- **Build system / ESP-IDF version**:
- **Target chip** (source: sdkconfig line or "not set → default esp32"):
- **Board** (source):
## Components
| Component | Purpose | Requires | Tests |
## Hardware
| GPIO | Function | Defined in |
(Include `esp32wb pins` findings verbatim.)
## Tasks
| Task | Stack (bytes) | Prio | Core |
## Dependencies
## Test coverage
## Risks / unknowns
```

## Rules

- State the evidence for every claim (file + line or tool output).
- Mark anything you could not confirm as **unknown** — never fill gaps with
  classic-ESP32 defaults.
- Next step after scouting is usually the `esp32-feature-contract` workflow
  (mission → contract → test plan).
