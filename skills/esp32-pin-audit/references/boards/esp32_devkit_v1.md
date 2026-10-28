# ESP32 DevKit V1 (generic WROOM-32 board)

- **Target**: `esp32` (classic ESP32, Xtensa dual-core LX6)
- **Module**: ESP32-WROOM-32 / -32D / -32E (no PSRAM on WROOM; WROVER variants use GPIO16/17 for PSRAM)
- **Status**: generic profile. "DevKit V1" is a community name for many clone boards
  (30- and 38-pin). LED pin, USB-UART chip and exposed pins vary — confirm against
  your board's silkscreen/schematic.

## Typical on-board usage

| Function | GPIO | Notes |
|----------|------|-------|
| Built-in LED | 2 (active HIGH) on most clones | GPIO2 is a strapping pin — see below |
| BOOT button | 0 (active LOW, board pull-up) | Strapping pin; usable as input after boot |
| EN button | CHIP_PU (not a GPIO) | Reset |
| UART0 console (USB-UART bridge) | TX 1, RX 3 | Used for flashing and logs |

## Conventional (not mandatory) bus pins

ESP-IDF requires you to configure pins explicitly; these are just common choices.

| Bus | Pins | Notes |
|-----|------|-------|
| I2C | SDA 21, SCL 22 | Arduino-core default; any output-capable GPIO works via the GPIO matrix |
| SPI3 (VSPI) IOMUX pins | MOSI 23, MISO 19, SCLK 18, CS 5 | GPIO5 is a strapping pin (SDIO timing) |
| SPI2 (HSPI) IOMUX pins | MOSI 13, MISO 12, SCLK 14, CS 15 | GPIO12 and GPIO15 are strapping pins; 12–15 are JTAG |

## Classic ESP32 limits that apply (from `esp32wb chip esp32`)

- GPIO 6–11: internal flash — never use (often exposed on 38-pin boards).
- GPIO 34, 35, 36 (VP), 39 (VN): input-only, no internal pulls (37/38 are not exposed on WROOM).
- Strapping: GPIO0, GPIO2 (boot mode), GPIO12 (flash voltage — never pull high on a
  3.3 V-flash module), GPIO15 (boot log), GPIO5 (SDIO timing).
- ADC2 (GPIO 0, 2, 4, 12–15, 25–27) cannot be used while Wi-Fi is in use.

## Sources

- Chip facts: `esp32wb chip esp32` (cites the ESP32 datasheet and ESP-IDF GPIO/ADC pages).
- Board-level assignments: common clone schematics; **verify on your board**.
