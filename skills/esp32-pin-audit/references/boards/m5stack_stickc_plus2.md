# M5Stack M5StickC Plus2

- **Target**: `esp32` (classic ESP32 — Xtensa dual-core)
- **SoC**: ESP32-PICO-V3-02, 8 MB flash, 2 MB quad PSRAM
- **USB-UART**: CH9102 (UART0 console on the classic ESP32 UART0 pins)
- **Not to be confused with**: M5StickC Plus (v1, ESP32-PICO-D4 + AXP192, different
  LCD/IR/LED pins) or the M5Stack StickS3 (ESP32-S3, see `m5stack_sticks3.md`).

## On-board usage

| Function | GPIO | Notes |
|----------|------|-------|
| Red LED **and** IR emitter | 19 | Shared pin — driving the LED also drives the IR LED |
| Button A | 37 | Classic ESP32 input-only pin |
| Button B | 39 | Classic ESP32 input-only pin |
| Button C (power / wake) | 35 | Classic ESP32 input-only pin |
| Power HOLD | 4 | Must be driven HIGH early after wake or the device powers off; LOW powers off |
| Passive buzzer | 2 | Strapping pin |
| Battery voltage (ADC) | 38 | ADC1 |
| LCD ST7789v2 MOSI | 15 | Strapping pin (MTDO) |
| LCD CLK | 13 | |
| LCD DC | 14 | |
| LCD RST | 12 | Strapping pin (MTDI, flash voltage select) |
| LCD CS | 5 | Strapping pin |
| LCD backlight | 27 | |
| Microphone SPM1423 CLK | 0 | Strapping pin |
| Microphone SPM1423 DATA | 34 | Input-only |
| Internal I2C SDA | 21 | MPU6886 IMU, BM8563 RTC |
| Internal I2C SCL | 22 | |
| Grove HY2.0-4P (PORT.CUSTOM) | 32 (yellow), 33 (white) | Black GND, red 5V |
| Other exposed pins (per spec table) | 0, 25/26, 36, 32, 33 | GPIO0 is shared with the mic clock |

## Constraints

- There is **no AXP192** on Plus2 (removed in the 2023.12 revision); power is held by
  GPIO4. Firmware must set HOLD=1 after boot.
- Many strapping pins (0, 2, 5, 12, 15) are wired to on-board peripherals by design;
  do not add external circuits that pull them at reset.
- With USB connected, a long press on Button C only sleeps the device instead of
  powering it off.
- Wi-Fi products must not read analog inputs on ADC2 pins (classic ESP32 rule);
  the battery ADC (GPIO38) is ADC1.

## Sources

- M5Stack docs, M5StickC Plus2 product page: https://docs.m5stack.com/en/core/M5StickC%20PLUS2
  (pin tables, revision history and Plus vs Plus2 comparison; accessed 2026-10-28).
- Schematic linked from that page: https://m5stack-doc.oss-cn-shenzhen.aliyuncs.com/512/Sch_M5StickC_Plus2_v0.5.pdf
- Chip facts: `esp32wb chip esp32`.
