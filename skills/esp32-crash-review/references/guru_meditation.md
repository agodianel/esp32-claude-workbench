# Playbook: Guru Meditation Error Review

## Trigger

An ESP32-family chip outputs a "Guru Meditation Error" in the serial log and reboots.

> The register dump differs by architecture. **Xtensa** (ESP32, S2, S3): `PC`, `PS`,
> `A0..A15`, `EXCCAUSE`, `EXCVADDR`, `Backtrace:` line. **RISC-V** (C2, C3, C5, C6, H2,
> P4): `MEPC`, `RA`, `SP`, `MCAUSE`, `MTVAL`, plus a stack dump that `idf.py monitor`
> unwinds. Fastest path: `esp32wb crash <file> --elf build/<app>.elf` (MCP `crash_decode`).

## Symptoms

- Serial output shows `Guru Meditation Error: Core X panic'ed`.
- Device reboot loop.
- Backtrace in serial output.
- Register dump visible.

## Triage Steps

### Step 1: Capture the Full Panic Output

Ensure you have the complete output, from the `Guru Meditation` line to the reboot.

Xtensa example (classic ESP32 addresses):
```
Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled.
Core  0 register dump:
PC      : 0x400d1234  PS      : 0x00060030  A0      : 0x800d5678  A1      : 0x3ffb9a10
...
EXCVADDR: 0x00000000

Backtrace: 0x400d1234:0x3ffb9a10 0x400d5678:0x3ffb9a30 ...
```

RISC-V example (ESP32-C3 style):
```
Guru Meditation Error: Core  0 panic'ed (Load access fault). Exception was unhandled.
Core  0 register dump:
MEPC    : 0x420048b4  RA      : 0x420048b4  SP      : 0x3fc8f2f0  GP      : 0x3fc8a600
...
MCAUSE  : 0x00000005  MTVAL   : 0x00000000
...
Stack memory:
3fc8f2f0: 0x00000000 ...
```

### Step 2: Identify the Exception Type

| Xtensa | RISC-V | Meaning | Common Cause |
|--------|--------|---------|-------------|
| `LoadProhibited` | Load access fault | Invalid memory read | NULL pointer dereference, use-after-free |
| `StoreProhibited` | Store access fault | Invalid memory write | Writing through NULL, writing to flash/ROM |
| `InstrFetchProhibited` | Instruction access fault | Invalid code execution | Corrupted function pointer, smashed return address |
| `IllegalInstruction` | Illegal instruction | Invalid opcode | Stack corruption, buffer overflow, bad function pointer |
| `IntegerDivideByZero` | — (RISC-V does not trap on divide by zero) | Division by zero | Unchecked divisor |
| `LoadStoreAlignment` | Load/Store address misaligned | Unaligned access | Casting to wrong pointer type, packed structs |
| `LoadStoreError` | — | Wrong access width to IRAM/ROM | 8/16-bit access to word-only memory |

### Step 3: Check the fault address (`EXCVADDR` on Xtensa, `MTVAL` on RISC-V)

- `0x00000000` → NULL pointer dereference (most common).
- Small value (< `0x100`) → Struct member access on NULL pointer (offset is the member offset).
- Value outside every valid memory region of the chip → uninitialised or corrupted pointer.
- Address inside a valid RAM/flash region but wrong for the access → stack corruption,
  use-after-free, or bad function pointer.

Memory regions differ per chip (e.g. classic ESP32 DRAM is around `0x3FFxxxxx` and
flash-mapped code around `0x400Dxxxx`; S3 and RISC-V chips use other ranges). Look up
the target's TRM "System and Memory" map instead of reusing classic-ESP32 ranges.

### Step 4: Decode the Backtrace

```bash
# Method 0: workbench decoder (Xtensa + RISC-V, picks the toolchain for the target)
esp32wb crash panic.txt --elf build/project.elf --target esp32s3

# Method 1: addr2line — prefix depends on the chip (see `esp32wb chip <target>`):
#   classic ESP32: xtensa-esp32-elf-   S2: xtensa-esp32s2-elf-   S3: xtensa-esp32s3-elf-
#   C2/C3/C5/C6/H2/P4: riscv32-esp-elf-
xtensa-esp32s3-elf-addr2line -pfiaC -e build/project.elf 0x42001234 0x42005678

# Method 2: idf.py monitor (auto-decodes)
idf.py monitor

# Method 3: ESP-IDF monitor tool
python -m esp_idf_monitor --port /dev/ttyUSB0 build/project.elf
```

The decoded backtrace shows which function called which, leading to the crash site.

### Step 5: Analyze the Call Chain

Starting from the top of the backtrace (crash site):
1. What function crashed?
2. What was it trying to access?
3. Who called it? With what arguments?
4. Is there a pattern (e.g., always crashes after Wi-Fi disconnect)?

### Step 6: Check for Known Patterns

**Pattern: Crash after `free()` or `delete`** → Use-after-free. Enable heap poisoning:
```
CONFIG_HEAP_POISONING_COMPREHENSIVE=y
```

**Pattern: Crash during ISR** → ISR calling non-ISR-safe function. Check for `xSemaphoreTake`, `vTaskDelay`, `printf` in ISR context.

**Pattern: Random crashes, different locations** → Memory corruption. Enable:
```
CONFIG_HEAP_POISONING_COMPREHENSIVE=y
CONFIG_HEAP_TASK_TRACKING=y
CONFIG_COMPILER_STACK_CHECK=y
```

**Pattern: Crash on deep function call** → Stack overflow. Enable:
```
CONFIG_FREERTOS_WATCHPOINT_END_OF_STACK=y
```
Overflow signatures: `Stack canary watchpoint triggered (<task>)` (Xtensa debug
exception), `Breakpoint` or `Stack protection fault` (RISC-V), or
`***ERROR*** A stack overflow in task <name>` (FreeRTOS check).

**Pattern: `Cache error` / cache disabled access inside an ISR** → ISR or data it
touches lives in flash while flash cache is disabled (e.g. during flash writes). Mark
the ISR `IRAM_ATTR` and keep its data in DRAM.

## Resolution Checklist

- [ ] Full panic output captured.
- [ ] Exception type identified.
- [ ] Backtrace decoded to source lines.
- [ ] Root cause determined.
- [ ] Fix applied and tested.
- [ ] Diagnostic config disabled before release.

## Prevention

- Always check pointers before dereference.
- Size task stacks generously.
- Use `*FromISR` variants in interrupt handlers.
- Enable stack canaries in debug builds.
- Run with heap poisoning during development.
