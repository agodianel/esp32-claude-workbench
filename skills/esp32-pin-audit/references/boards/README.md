# Board Profiles

A board profile records which GPIOs a specific development board already uses
(LEDs, buttons, displays, sensors, power management, USB) so an audit can flag
conflicts that the chip table alone cannot see.

| Profile | Board | Chip (`CONFIG_IDF_TARGET`) | Status |
|---------|-------|----------------------------|--------|
| [esp32_devkit_v1.md](esp32_devkit_v1.md) | Generic "ESP32 DevKit V1" (WROOM-32 clones) | `esp32` (classic) | Generic — clones vary, check your board |
| [m5stack_stickc_plus2.md](m5stack_stickc_plus2.md) | M5Stack M5StickC Plus2 | `esp32` (classic, ESP32-PICO-V3-02) | Verified against M5Stack docs page |
| [m5stack_sticks3.md](m5stack_sticks3.md) | M5Stack StickS3 | `esp32s3` (ESP32-S3-PICO-1-N8R8) | Verified against M5Stack docs page |

## How to use

1. Identify the exact board and revision (silkscreen, product page, schematic).
2. Use the matching profile together with `esp32wb chip <target>` and
   `esp32wb pins .` — the profile adds board-level usage, the chip tool adds
   silicon-level limits.
3. If no profile matches, ask for the schematic. Do not reuse a profile for a
   "similar looking" board: the M5StickC Plus2 (classic ESP32) and StickS3 (ESP32-S3)
   share a form factor but have completely different pinouts.

## Adding a board

Create `<vendor>_<board>.md` with:

1. Board name, revision, module/SoC, `CONFIG_IDF_TARGET`, flash/PSRAM size and type.
2. Every on-board function → GPIO (LEDs with active level, buttons with active level
   and pull, displays, sensors, buses with addresses, power-hold pins, USB).
3. Exposed headers / ports and their GPIOs.
4. Board-specific constraints (power-management init, shared pins).
5. **Sources**: link the official product page and schematic with access date.
   Mark anything not confirmed from a primary source as `UNVERIFIED`.
