# M5Stack StickS3

- **Target**: `esp32s3` (Xtensa dual-core LX7, 240 MHz)
- **SoC**: ESP32-S3-PICO-1-N8R8 — 8 MB flash, 8 MB **octal** PSRAM
- **Not to be confused with**: M5StickC Plus2 (classic ESP32, see
  `m5stack_stickc_plus2.md`). Earlier versions of this file mixed the two boards
  (buttons on 37/39, I2C on 21/22, LCD on 13/15 are **Plus2** pins, not StickS3).

## On-board usage

| Function | GPIO | Notes |
|----------|------|-------|
| LCD ST7789P3 MOSI | 39 | 39–42 are the S3 pad-JTAG pins (JTAG defaults to USB-Serial-JTAG) |
| LCD SCK | 40 | |
| LCD RS / DC | 45 | Strapping pin (VDD_SPI voltage) |
| LCD CS | 41 | |
| LCD RST | 21 | |
| LCD backlight | 38 | |
| Internal I2C SDA | 47 | BMI270 IMU (0x68), M5PM1 PMIC (0x6E), ES8311 codec (0x18) |
| Internal I2C SCL | 48 | |
| ES8311 MCLK | 18 | |
| ES8311 DOUT | 14 | |
| ES8311 BCLK | 17 | |
| ES8311 LRCK | 15 | |
| ES8311 DIN | 16 | |
| KEY1 | 11 | Input |
| KEY2 | 12 | Input |
| IR TX | 46 | Strapping pin (boot mode / ROM print) |
| IR RX | 42 | Receive via RMT peripheral |
| Grove HY2.0-4P (PORT.CUSTOM) | 9 (yellow), 10 (white) | Black GND, red 5V |
| M5PM1 PYG1 ↔ ESP32-S3 | 13 | Per the docs' M5PM1 table; exact function — confirm in schematic |
| USB (Type-C) | 19 / 20 | S3 USB D-/D+ — do not reuse |

**Hat2-Bus 2.54-16P header** (pin: signal): 1 GND, 2 G5, 3 EXT_5V, 4 G4, 5 Boot,
6 G6, 7 G1, 8 G7, 9 G8, 10 G43, 11 BAT, 12 G44, 13 3V3_L2, 14 G2, 15 5V_IN, 16 G3.
G43/G44 are the S3 UART0 console pins; G3 is a (JTAG-select) strapping pin.

## Constraints

- **Octal PSRAM** (R8): GPIO33–37 are used by the PSRAM interface in addition to the
  flash pins 26–32 — never use them.
- Power management is the M5PM1 on the internal I2C bus (not AXP192/AXP2101).
  Initialise the I2C bus on 47/48 before relying on PMIC-controlled rails; the
  speaker amplifier and L3B rail are enabled through M5PM1 GPIOs (PYG3 / PYG2).
- No input-only pins on ESP32-S3; strapping pins on this board in use: 45 (LCD DC), 46 (IR TX).
- Console/flashing is normally over native USB-Serial-JTAG (GPIO19/20); the green LED
  flashing indicates download mode.

## Sources

- M5Stack docs, StickS3 product page: https://docs.m5stack.com/en/core/StickS3
  (SoC, LCD, I2C, M5PM1, audio, buttons, IR, PORT.CUSTOM and Hat2-Bus tables;
  accessed 2026-10-28).
- Schematic linked from that page: https://m5stack-doc.oss-cn-shenzhen.aliyuncs.com/1207/K150_Stick_S3_PRJ_V0.6_20251111_2025_11_17_16_10_24.pdf
- Chip facts: `esp32wb chip esp32s3` (octal PSRAM pins, strapping, USB, JTAG).
- UNVERIFIED: the exact ESP32-S3 GPIO behind each M5PM1 PYG line other than PYG1↔G13,
  and the GPIO of the internal "download mode" LED. Check the schematic.
