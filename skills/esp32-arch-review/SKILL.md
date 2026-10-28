---
name: esp32-arch-review
description: Reviews ESP32 firmware architecture for FreeRTOS safety, memory and stack sizing, ISR rules, error handling, peripheral and pin safety, networking, storage, logging and build configuration, and returns a PASS / PASS WITH NOTES / FAIL report. Use when the user asks to "review the architecture", "code review this firmware", "is this RTOS-safe", "check for race conditions", "audit task stacks", "memory leak", "heap fragmentation", "stack overflow risk", "/arch-review", or before merging a significant ESP-IDF change or inheriting an existing ESP32 project.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Architecture Review

Structured, evidence-based review of embedded-specific correctness. Read-only unless
the user asks for fixes (fixes then go through an implementation contract).

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| All static checks (pins, stacks, sdkconfig, partitions) | `project_doctor` | `esp32wb doctor . [--target T]` |
| Undersized task stacks | `stack_check` | `esp32wb stacks .` |
| GPIO conflicts for the real chip | `pin_audit` | `esp32wb pins .` |
| Risky / debug sdkconfig options | `sdkconfig_review` | `esp32wb sdkconfig sdkconfig` |
| Partition table sanity | `partition_check` | `esp32wb partitions partitions.csv --sdkconfig sdkconfig` |
| API / peripheral behaviour questions | `search_esp_docs` | `esp32wb docs search "<query>" --target T` |

Start with `esp32wb doctor .` and paste its summary into the report; then do the
manual checklist below for what static tools cannot see.

## Review checklist

Mark each item `OK`, `ISSUE`, or `N/A`, with file:line evidence for every `ISSUE`.

### 1. FreeRTOS safety
- [ ] Every task has an explicit stack size. In ESP-IDF the stack argument is in
      **bytes** (not words). Guide: ≥2048 simple, ≥4096 with Wi-Fi/BLE/printf-heavy
      code, ≥8192 for TLS — verify with high-water marks, not guesses.
- [ ] No blocking calls in ISRs; only `*FromISR` APIs; `portYIELD_FROM_ISR` used.
- [ ] ISRs that run while flash cache is disabled are `IRAM_ATTR` and touch only DRAM data.
- [ ] Shared state protected by mutexes (priority inheritance), not by disabling interrupts.
- [ ] Spinlock critical sections (`portENTER_CRITICAL`) are very short (no logging, no allocation).
- [ ] No busy loops without blocking; `taskYIELD()` not `vTaskDelay(0)`.
- [ ] Core affinity deliberate and valid for the target's core count (dual-core:
      classic ESP32, S3, P4; S2 and the C/H series are single-core) — pin only to
      cores that exist, or use `tskNO_AFFINITY`.
- [ ] Bounded timeouts on queues/semaphores where a hang would be a bug.
- [ ] Task watchdog: long operations feed it or are restructured.

### 2. Memory
- [ ] No large stack buffers (> 512 bytes) — heap or static instead.
- [ ] Every allocation checked for NULL; freed on all error paths.
- [ ] DMA buffers from `heap_caps_malloc(..., MALLOC_CAP_DMA)` (PSRAM is generally
      not DMA-capable for all peripherals — check the peripheral docs for the target).
- [ ] No allocation in ISRs.
- [ ] Free heap and minimum free heap logged/monitored; fragmentation considered for
      long-running devices.
- See the [memory review playbook](references/memory_review.md) for diagnostics.

### 3. Error handling
- [ ] Every `esp_err_t` checked; `ESP_ERROR_CHECK()` only for fatal init.
- [ ] Runtime errors logged with `esp_err_to_name()` and handled (retry, degrade, reset state).
- [ ] Network operations retry with exponential backoff.
- [ ] Resources released on failure paths.

### 4. Peripherals and pins
- [ ] `esp32wb pins .` has no errors for the configured target.
- [ ] Pin map documented (`pin_map.h` or header comment) and matches code.
- [ ] I2C/SPI bus recovery and timeouts exist.
- [ ] ADC use respects the chip's ADC2/Wi-Fi behaviour (`esp32wb chip <target>`).
- [ ] Drivers use the API generation matching the ESP-IDF version (e.g. new I2C
      master driver vs legacy `driver/i2c.h` — do not mix both in one bus).

### 5. Networking
- [ ] Wi-Fi/IP event handlers registered before `esp_wifi_start()`.
- [ ] Reconnect with backoff; disconnect reason logged.
- [ ] TLS server verification enabled (no skip-verify in production).
- [ ] Socket/HTTP timeouts set; DNS failures handled.
- [ ] Wi-Fi + BLE coexistence enabled when both radios are used.

### 6. Persistent storage
- [ ] NVS namespaces per component; `ESP_ERR_NVS_NO_FREE_PAGES` /
      `ESP_ERR_NVS_NEW_VERSION_FOUND` handled at init.
- [ ] First-boot defaults when keys are missing.
- [ ] Partition table matches flash size and OTA scheme (`esp32wb partitions`).
- [ ] Filesystem (SPIFFS/LittleFS/FAT) mount errors handled.

### 7. Logging
- [ ] Correct levels; verbose output at `D`/`V`.
- [ ] No secrets in logs.
- [ ] Tags match component names; boot sequence traceable.

### 8. Build configuration
- [ ] `CONFIG_IDF_TARGET` matches the hardware.
- [ ] `sdkconfig.defaults` captures every non-default setting (not only a committed `sdkconfig`).
- [ ] Component `REQUIRES`/`PRIV_REQUIRES` complete and minimal.
- [ ] `esp32wb sdkconfig` shows no debug-only options in release configs.

## Report

```markdown
# Architecture Review — <project> (target: <chip>, ESP-IDF <ver>)
## Verdict: PASS | PASS WITH NOTES | FAIL
## Tool output
(esp32wb doctor summary)
## Critical issues (must fix before merge)
- [file:line] issue — why it matters — suggested fix
## Warnings
## Recommendations
## Checklist results
## Not reviewed / unknowns
```

FAIL if any critical issue exists (ISR misuse, unchecked allocation in a hot path,
flash/PSRAM pin use, missing TLS verification, stack overflow evidence).

## Rules

- Evidence for every finding; no generic advice without a location.
- Do not state chip limits from memory — cite `esp32wb chip` or Espressif docs.
- Recommendations that change code require an implementation contract first.
