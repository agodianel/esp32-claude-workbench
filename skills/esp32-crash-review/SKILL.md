---
name: esp32-crash-review
description: Decodes and analyses ESP32-family panics — Guru Meditation errors, register dumps, backtraces, aborts, stack overflows and watchdog timeouts on both Xtensa (ESP32, S2, S3) and RISC-V (C2, C3, C5, C6, H2, P4) chips — to find the root cause and recommend a fix plus diagnostic config. Use when the user pastes a crash or says "Guru Meditation", "LoadProhibited", "StoreProhibited", "InstrFetchProhibited", "IllegalInstruction", "Load access fault", "Store access fault", "MEPC", "MTVAL", "EXCVADDR", "Backtrace:", "abort() was called", "Stack canary watchpoint triggered", "Interrupt wdt timeout", "Task watchdog got triggered", "crash loop", "/review-crash", or "decode this backtrace".
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Crash Review

Crash analysis is architecture-specific. Identify the chip first.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Need | MCP tool | CLI |
|------|----------|-----|
| Decode panic (Xtensa + RISC-V; runs addr2line if the toolchain is on PATH) | `crash_decode` | `esp32wb crash panic.txt --elf build/<app>.elf [--target T]` (`-` reads stdin) |
| Arch + toolchain prefix for the chip | `chip_info` | `esp32wb chip <target>` |
| Surrounding log context | `log_summary` | `esp32wb logs monitor.log` |
| Stack sizes after a stack overflow | `stack_check` | `esp32wb stacks .` |

The ELF must be the **exact build** that produced the crash; a rebuilt ELF gives
wrong lines. If `--elf` is missing, ask for it or say decoding is unavailable.

## Steps

### 1. Identify target and architecture

From `CONFIG_IDF_TARGET`, the boot banner, or the dump itself:

| Dump contains | Architecture | Chips |
|---------------|--------------|-------|
| `PC`, `PS`, `A0..A15`, `EXCCAUSE`, `EXCVADDR`, `Backtrace: 0x..:0x..` | Xtensa | ESP32, ESP32-S2, ESP32-S3 |
| `MEPC`, `RA`, `SP`, `MCAUSE`, `MTVAL`, `MSTATUS`, stack memory dump | RISC-V | ESP32-C2, C3, C5, C6, H2, P4 |

Toolchains: Xtensa uses a per-chip prefix (`xtensa-esp32-elf-`,
`xtensa-esp32s2-elf-`, `xtensa-esp32s3-elf-`); RISC-V chips use `riscv32-esp-elf-`.
Confirm with `esp32wb chip <target>`.

### 2. Exception type

| Xtensa name | RISC-V equivalent | Typical cause |
|-------------|-------------------|---------------|
| `LoadProhibited` | Load access fault | Read through NULL / dangling / uninitialised pointer |
| `StoreProhibited` | Store access fault | Write through bad pointer, write to flash/ROM |
| `InstrFetchProhibited` | Instruction access fault | Bad function pointer, corrupted return address |
| `IllegalInstruction` | Illegal instruction | Corrupted stack/code, bad function pointer, `abort` paths |
| `LoadStoreAlignment` | Load/Store address misaligned | Misaligned access via cast or packed struct |
| `IntegerDivideByZero` | (no trap — RISC-V returns a defined result) | Unchecked divisor |
| `Unhandled debug exception` + "Stack canary watchpoint triggered" | `Breakpoint` (end-of-stack watchpoint) or `Stack protection fault` (HW stack guard) | Stack overflow |
| `Interrupt wdt timeout on CPU0/1` | same | ISR too long / interrupts disabled too long |
| `Cache error` / cache-disabled access | `Cache error` | Code/data in flash accessed while cache disabled (ISR not in IRAM) |

The fault address is `EXCVADDR` on Xtensa and `MTVAL` on RISC-V; the faulting
instruction is `PC` (Xtensa) / `MEPC` (RISC-V).

### 3. Decode the call chain

1. `esp32wb crash <file> --elf build/<app>.elf` (preferred), or
2. `idf.py monitor` with the matching ELF (decodes automatically; on RISC-V it
   unwinds the stack dump), or
3. Manual: `<prefix>addr2line -pfiaC -e build/<app>.elf <PC> <addr...>`.

RISC-V panics normally lack a `Backtrace:` line unless the firmware was built with
`CONFIG_ESP_SYSTEM_USE_EH_FRAME` or `CONFIG_ESP_SYSTEM_USE_FRAME_POINTER`; without the
monitor or the ELF you may only get the faulting function from `MEPC` and `RA`.

### 4. Interpret

- Fault address `0x00000000` or a small offset → NULL pointer / NULL struct member
  (offset ≈ member offset).
- Fault address that is garbage (not in any RAM/flash region of the chip) →
  uninitialised or corrupted pointer. Memory maps differ per chip; check the TRM
  rather than reusing classic-ESP32 ranges.
- Crash inside a FreeRTOS/driver call from an ISR → non-ISR-safe API in ISR.
- Different crash sites every time → memory corruption (overflow, use-after-free).
- `abort()` → read the preceding line (assert text, `ESP_ERROR_CHECK` error name and
  file:line).

Detailed patterns: [Guru Meditation playbook](references/guru_meditation.md),
[watchdog playbook](references/watchdog_reset.md).

### 5. Diagnostic configuration (debug builds only)

```
CONFIG_HEAP_POISONING_COMPREHENSIVE=y
CONFIG_HEAP_TASK_TRACKING=y
CONFIG_FREERTOS_WATCHPOINT_END_OF_STACK=y
CONFIG_FREERTOS_CHECK_STACKOVERFLOW_CANARY=y
CONFIG_COMPILER_STACK_CHECK_MODE_STRONG=y   # option name varies by IDF version
CONFIG_ESP_SYSTEM_PANIC_PRINT_HALT=y        # keep the dump on screen instead of rebooting
CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH=y       # capture field crashes; read with idf.py coredump-info
```

Verify each option exists for your ESP-IDF version in menuconfig before adding it,
and remove debug options before release (`esp32wb sdkconfig` flags them).

### 6. Report

```markdown
# Crash Analysis — <project> (target: <chip>, arch: Xtensa|RISC-V, IDF <ver>)
## Exception
- Type / core / task:
- PC|MEPC → function (file:line):
- Fault address (EXCVADDR|MTVAL):
## Call chain
| # | Address | Function | File:Line |
## Root cause (confidence: high/medium/low, with evidence)
## Classification: NULL pointer | stack overflow | ISR violation | memory corruption | watchdog | cache/IRAM | other
## Recommended fix (goes through an implementation contract)
## Diagnostic config to confirm
## Prevention
```

## Rules

- Never decode with the wrong toolchain or a different build's ELF.
- If you cannot decode, say so; do not guess function names from addresses.
- Fixes are code changes: create/update the mission and contract before editing.
