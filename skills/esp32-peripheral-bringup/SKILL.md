---
name: esp32-peripheral-bringup
description: Guides bring-up and debugging of ESP32 peripherals — I2C, SPI, UART, ADC and LEDC/PWM — using the ESP-IDF v5 driver APIs, chip-correct pin choices, and a step-by-step wiring → config → probe → speed-up procedure. Use when the user says "I2C sensor not responding", "i2c ESP_ERR_TIMEOUT", "no ACK", "I2C scan finds nothing", "SPI returns 0xFF", "SPI display blank", "UART garbage characters", "uart_read_bytes returns nothing", "ADC reads wrong value", "adc_oneshot", "analogRead with Wi-Fi", "PWM / LEDC not working", "set duty", "bring up a new sensor", or "connect a peripheral" on ESP32, ESP32-S3, ESP32-C3, ESP32-C6 or other Espressif chips.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Peripheral Bring-up

Bring a bus up in small, verifiable steps: **pins → electrical → minimal config →
probe → read known register → full speed → error handling**.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| Is this pin usable on this chip? | `chip_info`, `pin_audit` | `esp32wb chip <target>`, `esp32wb pins .` |
| Driver API / limits for the target (always before writing driver code) | `search_esp_docs` | `esp32wb docs search "i2c master probe" --target T` (index once: `esp32wb docs index`, needs `pip install 'esp32-workbench[docs]'`) |
| Summarise error output | `log_summary` | `esp32wb logs monitor.log` |

## Before writing driver code

1. Target chip + ESP-IDF version (API generation differs: v5.x has new I2C master,
   `esp_adc` oneshot, etc.).
2. Pin plan checked with `esp32wb pins .` — no flash/PSRAM/USB/console pins, strapping
   pins only with a reason, input-only pins only as inputs.
3. Peripheral instance count for the chip (number of I2C/SPI/UART ports, LEDC
   channels, ADC units differs per target) — check docs, do not assume classic ESP32.
4. Device datasheet: voltage levels (3.3 V I/O — never 5 V into a GPIO), address/mode,
   timing, power-up delay.
5. An implementation contract exists for the change.

## Per-peripheral quick guide

### I2C
- External pull-ups on SDA/SCL (typ. 2.2–4.7 kΩ to 3.3 V); internal pulls are too weak.
- Start at 100 kHz; use `i2c_master_probe()` (new driver) to scan.
- 7-bit addresses (datasheet `0xD0` 8-bit write address = `0x68`).
- Don't mix the legacy `driver/i2c.h` and new `driver/i2c_master.h` APIs in one build
  unless the IDF version explicitly allows it.
- Full procedure: [I2C bring-up](references/i2c_bringup.md).

### SPI
- Mode (CPOL/CPHA) from the device datasheet — the #1 failure cause.
- Start at 1 MHz; `.length` is in **bits**.
- IOMUX pins allow the highest clock; GPIO-matrix routing lowers the max — check the
  SPI master timing section for your chip.
- DMA buffers from `MALLOC_CAP_DMA` memory.
- Full procedure: [SPI bring-up](references/spi_bringup.md).

### UART
- Match baud, data bits, parity, stop bits, and flow control on both ends; cross TX→RX.
- Do not take UART0 console pins unless the console is moved.
- RX buffer passed to `uart_driver_install()` must exceed the hardware FIFO.
- Full procedure: [UART bring-up](references/uart_bringup.md).

### ADC
- Use `esp_adc` (`adc_oneshot_*` + `adc_cali_*`) on IDF v5; legacy `driver/adc.h` is deprecated.
- ADC2 + Wi-Fi is chip-specific (classic ESP32: unusable with Wi-Fi; S2/S3: arbitrated,
  reads can fail; C3: ADC2 oneshot unsupported) → prefer ADC1.
- Calibrate and report millivolts, not raw counts; attenuation sets the input range.
- Full procedure: [ADC bring-up](references/adc_bringup.md).

### LEDC (PWM)
- `LEDC_HIGH_SPEED_MODE` exists only on classic ESP32; other chips use `LEDC_LOW_SPEED_MODE`.
- Frequency × 2^resolution must be achievable from the timer clock source.
- Full procedure: [LEDC bring-up](references/ledc_bringup.md).

## Bring-up procedure (all buses)

1. **Power and wiring**: measure supply at the device, common ground, short wires.
2. **Minimal config** at the slowest sensible speed, explicit pins (never rely on defaults).
3. **Probe**: I2C probe / SPI read of a WHO_AM_I or ID register / UART loopback /
   ADC on a known voltage / LEDC on a scope or LED.
4. **Check every `esp_err_t`**; log with `esp_err_to_name()`.
5. **Scale up**: target speed, DMA, interrupts, multiple devices.
6. **Harden**: timeouts, retries, bus recovery, de-init path, mutex if the bus is shared
   between tasks.
7. **Evidence**: serial log of probe + readings, logic-analyzer capture if available.

## Report

```markdown
# Bring-up — <device> on <bus> (target: <chip>, IDF <ver>)
| Signal | GPIO | Chip category (esp32wb) | Notes |
## Configuration (speed, mode, address, pulls)
## Results (probe / ID register / readings, with log excerpts)
## Issues found and fixes
## Not verified (e.g. no scope available)
```

## Rules

- Never hallucinate peripheral limits; search the docs for the target first.
- Pin choices must pass `esp32wb pins` for the configured target.
- Any code change goes through the mission/contract workflow.
