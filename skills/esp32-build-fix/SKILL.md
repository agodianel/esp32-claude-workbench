---
name: esp32-build-fix
description: Diagnoses and fixes ESP-IDF build, link, CMake/Kconfig, component-manager and flashing failures, using the build log to classify the error before changing anything. Use when "idf.py build fails", "undefined reference to", "No such file or directory" for a header, "Failed to resolve component", "CMake Error", "multiple definition of", "region `iram0_0_seg' overflowed", "app partition is too small", "implicit declaration of function", "esptool Failed to connect", "Wrong boot mode detected", "No serial data received", "Permission denied /dev/ttyUSB0", "A fatal error occurred", "set-target", or after an ESP-IDF version upgrade (v4 → v5) breaks the build.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Build & Flash Fix

Classify first, then make the smallest fix. Never "fix" a build by deleting
configuration the user cares about.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| Classify build / flash output | `build_log_triage` | `idf.py build 2>&1 \| tee build.log; esp32wb build-log build.log` (or `... \| esp32wb build-log -`) |
| App vs partition size | `partition_check` | `esp32wb partitions partitions.csv --sdkconfig sdkconfig --bin build/<app>.bin` |
| sdkconfig sanity | `sdkconfig_review` | `esp32wb sdkconfig sdkconfig` |
| Chip facts (toolchain, USB/strapping pins for download mode) | `chip_info` | `esp32wb chip <target>` |
| API changed / renamed? | `search_esp_docs` | `esp32wb docs search "<symbol or error>" --target T` |

## Steps

### 1. Capture the full first error

Rerun with output captured (`idf.py build 2>&1 | tee build.log`). The **first**
error matters; later ones are usually cascades. Run `esp32wb build-log build.log`.

### 2. Establish context

- `idf.py --version` and the version the project expects (README, CI,
  `idf_component.yml` `dependencies.idf`).
- Target: `CONFIG_IDF_TARGET` in `sdkconfig` vs the board you have. A stale
  `sdkconfig` from another target causes many confusing errors.
- Was it building before? What changed (IDF upgrade, new component, new target,
  `managed_components` update)?

### 3. Classify and fix

| Category | Signature | Usual fix |
|----------|-----------|-----------|
| Missing header | `fatal error: xyz.h: No such file or directory` | Add the owning component to `REQUIRES`/`PRIV_REQUIRES`; in recent IDF v5.x releases drivers are split into `esp_driver_*` components (e.g. `esp_driver_gpio`, `esp_driver_i2c`) — `driver` still works as an umbrella |
| Linker | `undefined reference to` | Source not in `SRCS`, missing `REQUIRES`, C/C++ name mangling (`extern "C"`), Kconfig option disabling the feature |
| Duplicate symbol | `multiple definition of` | Function defined in a header without `static inline`, or same file compiled twice |
| Kconfig | `'CONFIG_X' undeclared` | Option renamed/removed in this IDF version or not enabled; check `sdkconfig` and IDF migration guide |
| Component manager | `Failed to resolve component`, version solver errors | Fix `idf_component.yml` constraints; `idf.py update-dependencies` |
| API change after upgrade | `implicit declaration`, `unknown type name` | Follow the ESP-IDF migration guide (ADC → `adc_oneshot`, I2C → `i2c_master` driver, legacy timer/RMT/I2S drivers replaced) |
| Memory overflow | `region 'iram0_0_seg' overflowed`, `dram0_0_seg` overflow | Move code out of IRAM, reduce static buffers, enable PSRAM for large buffers |
| App too big | `app partition is too small` / partition check fails | Larger app partition or custom `partitions.csv`, `-Os` (`CONFIG_COMPILER_OPTIMIZATION_SIZE`), drop unused features |
| Target mismatch | static assertions, missing SoC symbols | `idf.py set-target <target>` (this regenerates `sdkconfig`) |
| Toolchain / env | `xtensa-...-elf-gcc: not found`, `riscv32-esp-elf-gcc: not found`, Python env errors | Re-run `install.sh <target>` / `export.sh` for the IDF version in use |

Details and examples: [build failure playbook](references/build_failure.md).

### 4. Flash and monitor failures

Read [flash failure playbook](references/flash_failures.md) for esptool errors
(`Failed to connect`, `Wrong boot mode detected`, `No serial data received`,
permission denied, `invalid header` boot loops after flashing). Download-mode entry
and the USB port differ per chip — check `esp32wb chip <target>` for the boot strapping
pins and USB-Serial-JTAG pins.

### 5. Verify

- Clean build of the affected configuration (`idf.py fullclean && idf.py build` only
  after the targeted fix, to prove it is not a stale-cache artifact).
- No new warnings.
- If flashing was the issue: flash + boot log showing `app_main` reached.
- Record commands and output as test evidence in the mission.

## Safety

- `idf.py set-target` and deleting `sdkconfig` **discard local menuconfig changes**.
  Back up `sdkconfig` first and make sure `sdkconfig.defaults` captures what matters.
- `idf.py erase-flash` wipes NVS (Wi-Fi credentials, calibration, bonds). Ask first.
- Never burn eFuses or enable secure boot/flash encryption to "fix" a flash error.
- A code change made to fix the build still needs a (brief) implementation contract.
