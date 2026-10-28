---
name: esp32-pin-audit
description: Audits GPIO assignments in ESP32-family firmware against the verified pin facts of the actual target chip and board, flagging flash/PSRAM pins, strapping pins, input-only pins, USB-Serial-JTAG and UART0 console pins, JTAG pins, ADC2/Wi-Fi conflicts, double assignments and pin-map drift. Use when the user asks "which pins can I use", "is GPIO 12 safe", "check my pinout", "pin conflict", "strapping pin", "board won't boot after wiring", "can't flash after connecting a sensor", "ADC2 with Wi-Fi", "input-only pin", "/pin-audit", or before adding any GPIO, I2C, SPI, UART, ADC or LEDC code on ESP32, ESP32-S2, ESP32-S3, ESP32-C2, ESP32-C3, ESP32-C5, ESP32-C6, ESP32-H2 or ESP32-P4.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Pin Audit

Pin rules are **chip-specific**. GPIO 6–11 are flash pins on classic ESP32 but
ordinary GPIOs on ESP32-S3; GPIO 34–39 are input-only on classic ESP32 but
bidirectional on ESP32-S3. Never apply one chip's table to another.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| Audit all GPIO usage (target auto-detected from `CONFIG_IDF_TARGET`) | `pin_audit` | `esp32wb pins <path> [--target T]` |
| Verified pin facts + Espressif source links | `chip_info` | `esp32wb chip <target>` (no arg lists targets) |
| Anything the chip table does not cover | `search_esp_docs` | `esp32wb docs search "<query>" --target T` |

Targets: `esp32 esp32s2 esp32s3 esp32c2 esp32c3 esp32c5 esp32c6 esp32h2 esp32p4`.

## Steps

### 1. Establish target, module and board

- Target from `CONFIG_IDF_TARGET` in `sdkconfig` / `sdkconfig.defaults`. If none is
  set, ESP-IDF defaults to `esp32`; state that assumption in the report.
- Module/variant matters: flash and PSRAM type (quad vs octal, in-package vs
  external) change which pins are reserved. Check `CONFIG_SPIRAM*` and
  `CONFIG_ESPTOOLPY_OCT_FLASH` / octal PSRAM settings.
- Board: if a profile exists (see References below), use it. Otherwise ask for the
  schematic. On-board peripherals (LEDs, buttons, displays,
  power-management chips) occupy pins even if the code never mentions them.

### 2. Run the audit tool

`esp32wb pins . ` — it collects `GPIO_NUM_*`, `#define *_PIN/_GPIO`, Kconfig GPIO
options and driver config structs, then checks them against the chip table.
Include its findings verbatim; add your own analysis on top.

If the tool is unavailable, collect manually:

```bash
grep -rnE "GPIO_NUM_[0-9]+|#define\s+\w*(GPIO|PIN|SDA|SCL|MOSI|MISO|SCLK|CLK|CS|TX|RX)\w*\s+[0-9]+" \
  --include=*.c --include=*.h --include=*.cpp .
grep -rn "CONFIG_.*\(GPIO\|PIN\)" sdkconfig sdkconfig.defaults* 2>/dev/null
```

and get the chip facts from `esp32wb chip <target>` or the Espressif datasheet.

### 3. Check every pin against the chip's categories

For each GPIO in use, look up the target's facts and classify:

| Category | Severity | What to check |
|----------|----------|---------------|
| Not a GPIO on this chip | ERROR | Number missing from the chip's GPIO list (e.g. S3 has no GPIO22–25) |
| Flash SPI pins | ERROR | Never use. Set differs per chip |
| PSRAM pins | ERROR if PSRAM enabled | Includes octal-mode extras on S3 |
| Input-only | ERROR if used as output or with internal pulls | Classic ESP32: 34–39 (no internal pulls). S2: GPIO46. S3/C-series: none |
| Strapping | WARN | External circuits must not force the wrong level at reset; note the boot function |
| USB D-/D+ (USB-Serial-JTAG / OTG) | WARN | Reusing them disables USB console/JTAG/flashing over USB. Present on S2/S3/C3/C5/C6/H2/P4 |
| UART0 console TX/RX | WARN | Breaks logs/flashing over UART unless the console is moved |
| JTAG pins | WARN if hardware debugging is used | |
| ADC2 + Wi-Fi | see below | |

### 4. ADC2 and Wi-Fi (chip-dependent)

- **Classic ESP32**: ADC2 pins **cannot** be used while Wi-Fi is in use — ERROR.
  Move analog inputs to ADC1.
- **ESP32-S2 / ESP32-S3**: ADC2 is shared with Wi-Fi and the driver arbitrates;
  `adc_oneshot_read()` can **fail** (returns an error) while Wi-Fi holds ADC2 — WARN,
  require retry/error handling or move to ADC1.
- **ESP32-C3**: ADC2 (GPIO5) oneshot is no longer supported by current ESP-IDF due
  to a hardware erratum — flag any ADC2 use regardless of Wi-Fi.
- **Other chips**: check `esp32wb chip <target>` (field "ADC2 + Wi-Fi" and notes) —
  several have no ADC2 at all.

### 5. Conflicts and electrical checks

- Same GPIO assigned to two functions (including board-fixed functions).
- Buses sharing pins unintentionally (I2C vs SPI vs UART).
- I2C lines: external pull-ups required; internal pull-ups are too weak for
  reliable I2C. Flag configs relying only on `*_pullup_en`.
- Floating inputs: buttons/sensors without external or internal pull.
- Voltage domain: some pins run at the flash/PSRAM voltage on 1.8 V parts
  (e.g. ESP32-S3 GPIO47/48 on R8V/R16V variants) — check `esp32wb chip`.

### 6. Documentation drift

Compare code against `pin_map.h`, README pin tables, board profile and schematic.
Every assignment must appear in one documented pin map (project rule).

## Report

```markdown
# Pin Audit — <project> (target: <chip>, board: <board>, module: <module>)
Evidence: `esp32wb pins` output + `esp32wb chip <target>` sources.

## Pin Allocation
| GPIO | Function | Defined in | Direction | Pull | Chip category |
## Errors (must fix)
## Warnings (strapping / USB / console / JTAG / ADC2)
## Conflicts
## Documentation drift
## Unknowns (facts not confirmed for this chip/board)
```

Use plain words (`ERROR`, `WARN`, `OK`) rather than emoji so the report renders in
every agent and terminal.

## References

- [Chip-by-chip hardware limitations](references/hardware_limitations.md)
- Board profiles ([index and how to add one](references/boards/README.md)):
  - [ESP32 DevKit V1 (classic ESP32, generic)](references/boards/esp32_devkit_v1.md)
  - [M5Stack M5StickC Plus2 (classic ESP32, ESP32-PICO-V3-02)](references/boards/m5stack_stickc_plus2.md)
  - [M5Stack StickS3 (ESP32-S3-PICO-1-N8R8)](references/boards/m5stack_sticks3.md)

## Rules

- Never hallucinate hardware constraints. If a fact is not in `esp32wb chip` output
  or a cited Espressif document, report it as **unknown**.
- Label any classic-ESP32 fact as "classic ESP32" when you mention it.
- Re-run the audit after every GPIO change and attach the output as test evidence.
