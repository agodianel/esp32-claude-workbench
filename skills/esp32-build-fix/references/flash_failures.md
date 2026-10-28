# Playbook: Flash & Monitor Failures

## Trigger

`idf.py flash` / `esptool` fails, or the device does not boot after flashing.

> Classify the output first: `idf.py -p PORT flash 2>&1 | tee flash.log` then
> `esp32wb build-log flash.log` (MCP `build_log_triage`).
> Get the chip's boot strapping pins and USB D-/D+ pins from `esp32wb chip <target>` —
> they differ per chip.

## Symptoms → causes

| Output | Likely cause | Fix |
|--------|-------------|-----|
| `Failed to connect to ESP32...: No serial data received` | Chip not in download mode; wrong port; charge-only USB cable; USB-UART driver missing | Hold BOOT while pressing/releasing RESET (EN), then flash; try another cable/port; install the CP210x / CH340 / CH9102 / FTDI driver |
| `Wrong boot mode detected (0x..)! The chip needs to be in download mode.` | Auto-reset circuit not pulling the boot strapping pin low | Manual download mode (BOOT + RESET); check external circuitry on the boot strapping pins |
| `Permission denied: '/dev/ttyUSB0'` (Linux) | User not in the serial group | Add user to `dialout` (or `uucp` on Arch), re-login |
| `Could not open port` / port busy | Another monitor/IDE holds the port | Close other serial monitors |
| `Invalid head of packet` / `Packet content transfer stopped` / timeouts mid-flash | Noisy USB, baud too high, weak power | Lower baud (`idf.py -b 115200 flash`), shorter cable, powered hub |
| `This chip is ESP32-XX not ESP32-YY. Wrong --chip argument?` | Built for a different target | `idf.py set-target <actual chip>` and rebuild |
| `A fatal error occurred: ... flash size` / boot shows wrong size | `CONFIG_ESPTOOLPY_FLASHSIZE` doesn't match the module | Set correct flash size; re-check partition table fits |
| Port disappears during/after flash (native USB) | Firmware reconfigured USB D-/D+ pins, crashed early, or entered deep sleep | Enter download mode manually; avoid reusing USB pins; keep console on USB-Serial-JTAG while debugging |

## Native USB vs USB-UART bridge

- **USB-UART bridge boards** (CP210x/CH340/CH9102/FTDI): console on UART0; auto-reset via
  DTR/RTS. Port names like `/dev/ttyUSB*`, `/dev/cu.usbserial-*`, `COMx`.
- **Native USB-Serial-JTAG** (S3, C3, C5, C6, H2, P4): port names like `/dev/ttyACM*`,
  `/dev/cu.usbmodem*`. The port re-enumerates on reset; if firmware crashes before USB
  is up or deep-sleeps, use manual download mode.
- **ESP32-S2** has USB OTG but no USB-Serial-JTAG; flashing over native USB uses
  the ROM's USB download mode or the TinyUSB CDC console.
- Console routing is set by `CONFIG_ESP_CONSOLE_*` — if you see no logs after a
  successful flash, check which console is configured.

## Boot problems right after flashing

| Boot log | Cause |
|----------|-------|
| `invalid header: 0xffffffff` repeating | Nothing valid at the boot offset: wrong offsets, partial flash, or flash erased |
| `flash read err` / checksum errors | Wrong flash mode/frequency for the module (`CONFIG_ESPTOOLPY_FLASHMODE`, `FLASHFREQ`), bad module |
| Boots into download mode every time (`waiting for download`) | Boot strapping pin held at the download level by external circuitry |
| Chip-ID mismatch error from bootloader | Image built for another target |
| Brownout right after boot | Power supply / cable cannot handle radio start-up current |

## Safe commands

```bash
idf.py -p PORT flash monitor          # normal path
idf.py -p PORT -b 115200 flash        # slower, more robust
esptool.py --port PORT chip_id        # connectivity test (esptool v5 also accepts `esptool chip-id`)
esptool.py --port PORT flash_id       # flash size/manufacturer
```

**Ask before running** `idf.py erase-flash` (wipes NVS, Wi-Fi credentials, BLE bonds,
calibration) and never run `espefuse.py burn_*` as a troubleshooting step — eFuse
writes are permanent.

## Resolution Checklist

- [ ] Correct target and port.
- [ ] Download mode entered (auto or manual).
- [ ] Cable/driver/permissions verified.
- [ ] Flash completes; boot log reaches `app_main`.
- [ ] Root cause recorded in the mission (e.g. strapping circuit, cable, config).
