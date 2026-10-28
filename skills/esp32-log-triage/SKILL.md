---
name: esp32-log-triage
description: Parses and classifies ESP32 serial monitor output — reset reasons, boot loops, panics, watchdog messages, brownouts, heap failures, Wi-Fi/BLE/MQTT errors and peripheral timeouts — and produces a triage report with a root-cause hypothesis and next steps. Use when the user pastes "idf.py monitor" output or a serial log, or says "boot loop", "keeps rebooting", "rst:0x", "Brownout detector was triggered", "Task watchdog got triggered", "Wi-Fi keeps disconnecting", "WIFI_EVENT_STA_DISCONNECTED", "BLE won't connect", "MQTT disconnects", "/triage-log", or "what does this log mean" for any ESP32-family chip.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Log Triage

Turn a raw serial log into a classified, evidence-backed diagnosis.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| Summarise a log (resets, errors, tags, patterns) | `log_summary` | `esp32wb logs monitor.log` or `... \| esp32wb logs -` |
| Decode a panic found in the log | `crash_decode` | `esp32wb crash panic.txt --elf build/<app>.elf [--target T]` |
| Chip facts (arch, toolchain, pins) | `chip_info` | `esp32wb chip <target>` |
| Driver / error-code meaning | `search_esp_docs` | `esp32wb docs search "<message>" --target T` |

Run `esp32wb logs` first and build on its output rather than re-reading the log by eye.

## Steps

### 1. Establish context

Target chip (`CONFIG_IDF_TARGET`), board, ESP-IDF version, what the device was doing,
and whether the log starts at power-on. Ask for the full log from reset if it is
truncated — the ROM boot lines carry the reset reason.

### 2. Read the log format correctly

ESP-IDF log lines look like:

```
E (12345) wifi_mgr: connect failed: ESP_ERR_TIMEOUT
^  ^       ^         ^
|  |       tag       message
|  ms since boot (or wall time if CONFIG_LOG_TIMESTAMP_SOURCE_SYSTEM)
level: E error, W warning, I info, D debug, V verbose
```

Lines without this prefix come from the ROM bootloader, `printf`, or the panic handler.

### 3. Reset reason

- The ROM prints `rst:0x.. (NAME)` on every boot. **Names and codes differ per
  chip** (classic ESP32 uses names like `POWERON_RESET`, `SW_CPU_RESET`,
  `TG0WDT_SYS_RESET`, `RTCWDT_BROWN_OUT_RESET`; newer chips use different names).
  Do not map codes from memory; use `esp32wb logs` or the target's TRM.
- The portable source is the app: log `esp_reset_reason()` at boot. Values:
  `ESP_RST_POWERON`, `ESP_RST_SW`, `ESP_RST_PANIC`, `ESP_RST_INT_WDT`,
  `ESP_RST_TASK_WDT`, `ESP_RST_WDT`, `ESP_RST_BROWNOUT`, `ESP_RST_DEEPSLEEP`, etc.
- After a panic the ROM usually reports a *software* CPU reset, not a watchdog code —
  look for the `Guru Meditation` text above it.

### 4. Extract critical events

| Pattern | Meaning |
|---------|---------|
| `Guru Meditation Error: Core N panic'ed (...)` | CPU exception / panic → hand to `esp32-crash-review` |
| `abort() was called at PC ...` | Assert / `ESP_ERROR_CHECK` failure — read the line above it |
| `ESP_ERROR_CHECK failed: esp_err_t 0x... (ESP_ERR_...)` | Fatal init error; file/line printed |
| `Stack canary watchpoint triggered (task)` / `Stack protection fault` / `***ERROR*** A stack overflow in task` | Stack overflow |
| `Task watchdog got triggered` | TWDT: a task starved IDLE. By default this only **prints** (resets only with `CONFIG_ESP_TASK_WDT_PANIC`) |
| `Interrupt wdt timeout on CPU0/1` | IWDT panic: ISR too long or interrupts disabled too long |
| `Brownout detector was triggered` | Supply dip (USB cable/port, regulator, Wi-Fi TX peaks) |
| `CORRUPT HEAP` / `heap_caps_alloc ... failed` / `ESP_ERR_NO_MEM` | Heap corruption or exhaustion |
| `invalid header: 0x...` repeating | Bad flash content/offset or wrong flash mode — see build/flash troubleshooting |
| `wifi:` state lines, `WIFI_EVENT_STA_DISCONNECTED` + reason | See [Wi-Fi playbook](references/wifi_debug.md) |
| `NimBLE` / `BT_*` errors | See [BLE playbook](references/ble_debug.md) |
| `i2c`, `spi_master`, `uart` errors / `ESP_ERR_TIMEOUT` | Peripheral bus problem (wiring, pull-ups, pins, mode) |

### 5. Classify

| Category | Indicators |
|----------|-----------|
| Boot loop | Repeating ROM banner; never reaches `app_main` / main task logs |
| Crash | Guru Meditation, abort, backtrace or stack dump |
| Watchdog | TWDT/IWDT messages or `ESP_RST_*WDT` |
| Memory | Allocation failures, falling free heap, heap corruption |
| Power | Brownout, resets correlated with radio TX or motor/LED load |
| Network | Disconnect reasons, DHCP/DNS/TLS failures, MQTT reconnect storms |
| Peripheral | Bus timeouts/NACKs, wrong data |

### 6. Backtraces

- Xtensa chips (ESP32, S2, S3) print `Backtrace: 0xPC:0xSP ...`.
- RISC-V chips (C2, C3, C5, C6, H2, P4) usually print a register dump with
  `MEPC`/`MCAUSE`/`MTVAL` and a stack dump; `idf.py monitor` decodes it.
- Decode with `esp32wb crash` + the matching ELF, or `idf.py monitor`. If you use
  addr2line directly, get the toolchain prefix from `esp32wb chip <target>`
  (e.g. `xtensa-esp32s3-elf-addr2line`, `riscv32-esp-elf-addr2line`); never assume
  `xtensa-esp32-elf`.

## Report

```markdown
# Log Triage — <device> (target: <chip>, IDF <ver>)
- **Issue type**: Boot loop | Crash | Watchdog | Memory | Power | Network | Peripheral
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Reset reason**: <ROM line> / <esp_reset_reason if logged>
## Key events (chronological, with timestamps)
## Error patterns (pattern — count — interpretation)
## Root-cause hypothesis (and confidence)
## Next steps (diagnostic config, instrumentation, fix candidates)
## Raw evidence (minimal excerpts)
```

## Tips

- Brownouts are usually power, not code: test with a better cable/supply before debugging firmware.
- A TWDT message without a reset is still a bug — the system was starved.
- A fault address of `0x00000000` or a small offset is almost always a NULL (struct) pointer.
- Repeated MQTT disconnects: compare keepalive with network latency and check TLS heap.
