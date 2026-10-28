# ESP32-Family Hardware Limitations (pin-level)

Use when selecting pins, starting a project, or diagnosing "cannot flash", "random
download mode", "ADC read fails with Wi-Fi", "pin won't drive output", "JTAG stopped
working".

> **Source of truth**: `esp32wb chip <target>` (MCP `chip_info`). It prints verified
> per-chip facts with links to the Espressif datasheet / ESP-IDF pages they came from.
> The tables below summarise those facts; if anything here disagrees with the tool or
> the current datasheet, the datasheet wins.

## Symptoms → likely pin cause

| Symptom | Check |
|---------|-------|
| Board enters download mode or won't boot with peripheral attached | Strapping pins held at the wrong level at reset |
| Flash / PSRAM errors, instant crash on boot | Code touches flash or PSRAM pins |
| Output pin never changes level | Input-only pin (classic ESP32 34–39, ESP32-S2 GPIO46) |
| USB serial/console disappears after init | USB D-/D+ pins reconfigured as GPIO |
| No boot log / can't flash over UART | UART0 console pins reused |
| ADC2 reads fail or return errors with Wi-Fi | ADC2/Wi-Fi interaction (chip-specific, below) |
| JTAG debugging unavailable | JTAG pins reused, or JTAG routed to USB-Serial-JTAG |

## Per-chip summary (from the workbench chip database)

| Target | Arch | Flash pins (never use) | Input-only | USB D-/D+ | UART0 TX/RX | Boot-mode strapping |
|--------|------|------------------------|------------|-----------|-------------|---------------------|
| esp32 (classic) | Xtensa | 6–11 | 34–39 | — | 1 / 3 | 0, 2 (plus 5, 12, 15 — see below) |
| esp32s2 | Xtensa | 27–32 (26 = PSRAM CS) | 46 | 19 / 20 (USB OTG) | 43 / 44 | 0, 46 (45 = VDD_SPI) |
| esp32s3 | Xtensa | 27–32 (26 = PSRAM CS; 33–37 with octal flash/PSRAM) | none | 19 / 20 | 43 / 44 | 0, 46 (45 = VDD_SPI, 3 = JTAG select) |
| esp32c2 | RISC-V | 11–17 | none | — | 20 / 19 | 8, 9 |
| esp32c3 | RISC-V | 12–17 | none | 18 / 19 | 21 / 20 | 2, 8, 9 |
| esp32c5 | RISC-V | 15–22 | none | 13 / 14 | 11 / 12 | 26, 27, 28 (+ others) |
| esp32c6 | RISC-V | 24–30 | none | 12 / 13 | 16 / 17 | 8, 9 (4, 5, 15 also strapping) |
| esp32h2 | RISC-V | 15–21 | none | 26 / 27 | 24 / 23 | 8, 9 (25 = JTAG select) |
| esp32p4 | RISC-V | none in the GPIO range (flash uses dedicated chip pins) | none | 24 / 25 | 37 / 38 | 35, 36 (34, 37, 38 also strapping) |

Run `esp32wb chip <target>` for full strapping semantics, JTAG pins, ADC channel
maps, PSRAM conditions and notes. The table omits detail on purpose.

## ADC2 and Wi-Fi

| Chip | Behaviour | Action |
|------|-----------|--------|
| Classic ESP32 | ESP-IDF: ADC2 pins cannot be used while Wi-Fi is used | Use ADC1 for any analog input in a Wi-Fi product |
| ESP32-S2, ESP32-S3 | ADC2 is also used by Wi-Fi; `adc_oneshot_read()` arbitrates and can return an error while Wi-Fi holds ADC2. ADC2 continuous (DMA) mode is not supported on S3 | Prefer ADC1; if ADC2 is unavoidable, handle the error and retry |
| Others | C2, C5, C6, H2 have no ADC2 in the chip database; C3 has one ADC2 channel (GPIO5) but current ESP-IDF no longer supports ADC2 oneshot on C3 (errata) — treat any ADC2 use on C3 as an error; P4 and H2 have no Wi-Fi. Confirm with `esp32wb chip <target>` | — |

## Classic ESP32 details (only for `esp32` target)

- **Input-only**: GPIO 34, 35, 36 (VP), 37, 38, 39 (VN) — no output, no internal
  pull-up/pull-down. Use external resistors for buttons.
- **Strapping**:
  - GPIO0 (default pull-up) and GPIO2 (default pull-down): GPIO0=0 with GPIO2=0 at
    reset enters download mode.
  - GPIO12 (MTDI): selects flash/PSRAM supply voltage at reset. Pulled high on a
    3.3 V-flash module → 1.8 V flash supply → boot failure. Can be overridden by eFuse.
  - GPIO15 (MTDO): ROM boot-log printing; GPIO5 + GPIO15: SDIO slave timing.
- **JTAG**: GPIO12 (MTDI), 13 (MTCK), 14 (MTMS), 15 (MTDO).
- **ADC1**: GPIO 32–39. **ADC2**: GPIO 0, 2, 4, 12–15, 25–27.
- **PSRAM modules (e.g. WROVER)**: GPIO16 (PSRAM CS) and normally GPIO17 (PSRAM clock) are used by PSRAM.

## Checklist

- [ ] Target chip and module variant confirmed (`CONFIG_IDF_TARGET`, flash/PSRAM type).
- [ ] No flash or PSRAM pins used.
- [ ] Outputs not on input-only pins (if the chip has any).
- [ ] Strapping pins either unused or verified safe at reset with the attached circuit.
- [ ] USB D-/D+ and UART0 console pins left alone unless the console is moved.
- [ ] Analog inputs on ADC1 when Wi-Fi is used (or ADC2 errors handled on S2/S3).
- [ ] JTAG pins free if hardware debugging is required.
