"""Decode ESP32 panic output (Guru Meditation, abort, assert, stack overflow).

Works for both Xtensa (ESP32, S2, S3) and RISC-V (C2, C3, C5, C6, H2, P4) panic
formats. When the target's ``addr2line`` is on PATH and an ELF is supplied, code
addresses are resolved to ``function at file:line``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from esp32_workbench.chips import list_targets, load_chip

GURU_PATTERN = re.compile(r"Guru Meditation Error:\s*Core\s+(\d+)\s+panic'ed\s+\(([^)]+)\)")
REGISTER_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,8})\s*:\s*(0x[0-9a-fA-F]{8})")
BACKTRACE_PATTERN = re.compile(r"Backtrace:\s*(.+)")
ADDRESS_PAIR_PATTERN = re.compile(r"(0x[0-9a-fA-F]{8}):0x[0-9a-fA-F]{8}")
ABORT_PATTERN = re.compile(r"abort\(\) was called at PC (0x[0-9a-fA-F]+) on core (\d+)")
ASSERT_PATTERN = re.compile(r"assert failed:\s*(.+)")
STACK_OVERFLOW_PATTERN = re.compile(
    r"A stack overflow in task (\S+) has been detected|Stack canary watchpoint triggered \((\S+)\)"
)
HEAP_PATTERN = re.compile(r"CORRUPT HEAP:.*|heap_caps_free.*corrupt", re.IGNORECASE)
INT_WDT_PATTERN = re.compile(r"Interrupt wdt timeout on CPU(\d)")
CACHE_PATTERN = re.compile(r"Cache disabled but cached memory region accessed|Cache error")
ESP_ERROR_CHECK_PATTERN = re.compile(
    r"ESP_ERROR_CHECK failed: esp_err_t (0x[0-9a-fA-F]+) \((\w+)\)"
)

# Standard RISC-V mcause exception codes
RISCV_MCAUSE: dict[int, str] = {
    0: "Instruction address misaligned",
    1: "Instruction access fault",
    2: "Illegal instruction",
    3: "Breakpoint",
    4: "Load address misaligned",
    5: "Load access fault",
    6: "Store address misaligned",
    7: "Store access fault",
    8: "Environment call from U-mode",
    11: "Environment call from M-mode",
}

NULL_THRESHOLD = 0x100

HINTS: dict[str, str] = {
    "null_pointer": "Faulting address is near 0x0: a NULL (or NULL + small offset) pointer was "
    "dereferenced. Check handles/pointers returned by init functions before use.",
    "stack_overflow": "A task overflowed its stack. Increase the stack size in xTaskCreate*, move "
    "large local buffers to the heap, and check recursion. Use uxTaskGetStackHighWaterMark().",
    "heap_corruption": "Heap metadata was overwritten: buffer overflow, use-after-free or double "
    "free. Enable CONFIG_HEAP_POISONING_COMPREHENSIVE and CONFIG_HEAP_TASK_TRACKING to localise it.",
    "illegal_instruction": "CPU executed an invalid instruction: corrupted function pointer, stack "
    "corruption, or calling code from flash while the cache is disabled.",
    "cache_error": "Flash/PSRAM was accessed while the cache was disabled (e.g. during a flash "
    "write). ISRs and callbacks running then must be IRAM_ATTR and only touch DRAM data.",
    "interrupt_watchdog": "An ISR or critical section ran too long, or interrupts were disabled "
    "too long. Keep ISRs short and defer work to a task.",
    "abort": "abort() was called — usually a failed ESP_ERROR_CHECK, assert, or a FreeRTOS API "
    "misuse. Decode the abort PC and the backtrace to find the caller.",
    "assert": "An assert() failed. The message names the file, line and condition.",
    "esp_error_check": "ESP_ERROR_CHECK aborted on a non-ESP_OK return. Handle that error "
    "gracefully at runtime instead of aborting.",
    "unaligned": "Unaligned memory access — often a packed struct or a cast of a byte buffer to a "
    "wider type.",
    "divide_by_zero": "Integer division by zero.",
    "access_fault": "Invalid memory access at a non-NULL address: dangling pointer, out-of-bounds "
    "index, or a freed object.",
}


@dataclass
class Frame:
    """A code address from the panic, optionally decoded to a source location."""

    address: str
    source: str = "registers"  # registers | backtrace | abort
    decoded: str | None = None


@dataclass
class CrashReport:
    """Structured result of decoding a panic dump."""

    kind: str | None = None  # guru_meditation | abort | assert | stack_overflow | ...
    exception: str | None = None
    core: int | None = None
    arch: str | None = None
    fault_address: str | None = None
    registers: dict[str, str] = field(default_factory=dict)
    frames: list[Frame] = field(default_factory=list)
    backtrace_corrupted: bool = False
    task: str | None = None
    details: str | None = None
    classification: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    addr2line: str | None = None


def _classify(report: CrashReport) -> None:
    labels: list[str] = []
    exc = (report.exception or "").lower()

    if report.fault_address is not None and int(report.fault_address, 16) < NULL_THRESHOLD:
        if any(k in exc for k in ("load", "store", "access")):
            labels.append("null_pointer")
    if report.kind in ("stack_overflow",):
        labels.append("stack_overflow")
    if report.kind == "heap_corruption":
        labels.append("heap_corruption")
    if report.kind == "interrupt_watchdog":
        labels.append("interrupt_watchdog")
    if report.kind == "cache_error" or "cache" in exc:
        labels.append("cache_error")
    if "illegal" in exc:
        labels.append("illegal_instruction")
    if "unaligned" in exc or "misaligned" in exc:
        labels.append("unaligned")
    if "divide" in exc:
        labels.append("divide_by_zero")
    if report.kind in ("abort", "assert", "esp_error_check"):
        labels.append(report.kind)
    if not labels and any(k in exc for k in ("prohibited", "access fault")):
        labels.append("access_fault")

    report.classification = labels
    report.hints = [HINTS[label] for label in labels]


def parse_panic(text: str) -> CrashReport:
    """Parse raw serial output containing an ESP32 panic.

    Args:
        text: Serial log text (may contain surrounding log lines).

    Returns:
        CrashReport. ``kind`` is None when no panic signature was found.
    """
    report = CrashReport()

    if match := GURU_PATTERN.search(text):
        report.kind = "guru_meditation"
        report.core = int(match.group(1))
        report.exception = match.group(2)

    if match := STACK_OVERFLOW_PATTERN.search(text):
        report.kind = "stack_overflow"
        report.task = match.group(1) or match.group(2)
    elif match := ESP_ERROR_CHECK_PATTERN.search(text):
        report.kind = "esp_error_check"
        report.details = f"{match.group(2)} ({match.group(1)})"
    elif match := ASSERT_PATTERN.search(text):
        report.kind = "assert"
        report.details = match.group(1).strip()
    elif HEAP_PATTERN.search(text):
        report.kind = "heap_corruption"
    elif match := INT_WDT_PATTERN.search(text):
        report.kind = "interrupt_watchdog"
        report.core = int(match.group(1))
    elif CACHE_PATTERN.search(text):
        report.kind = "cache_error"

    if match := ABORT_PATTERN.search(text):
        report.kind = report.kind if report.kind not in (None, "guru_meditation") else "abort"
        report.core = int(match.group(2))
        report.frames.append(Frame(address=match.group(1).lower(), source="abort"))

    for name, value in REGISTER_PATTERN.findall(text):
        report.registers.setdefault(name, value.lower())

    regs = report.registers
    if "MEPC" in regs or "MCAUSE" in regs:
        report.arch = "riscv"
        if "MTVAL" in regs:
            report.fault_address = regs["MTVAL"]
        if "MCAUSE" in regs and not report.exception:
            cause = int(regs["MCAUSE"], 16) & 0x7FFFFFFF
            report.exception = RISCV_MCAUSE.get(cause, f"mcause {cause}")
        for reg in ("MEPC", "RA"):
            if reg in regs:
                report.frames.append(Frame(address=regs[reg]))
    elif "PC" in regs or "EXCVADDR" in regs:
        report.arch = "xtensa"
        if "EXCVADDR" in regs:
            report.fault_address = regs["EXCVADDR"]
        if "PC" in regs:
            report.frames.append(Frame(address=regs["PC"]))

    if match := BACKTRACE_PATTERN.search(text):
        line = match.group(1)
        report.backtrace_corrupted = "CORRUPTED" in line
        for pc in ADDRESS_PAIR_PATTERN.findall(line):
            report.frames.append(Frame(address=pc.lower(), source="backtrace"))
        if report.arch is None and report.frames:
            report.arch = "xtensa"

    if report.kind is None and report.frames:
        report.kind = "guru_meditation"

    _classify(report)
    return report


def find_addr2line(target: str | None, arch: str | None) -> str | None:
    """Locate an addr2line binary for the target, trying unified toolchain names too."""
    candidates: list[str] = []
    if target:
        candidates.append(load_chip(target).toolchain_prefix)
    if arch == "riscv":
        candidates.append("riscv32-esp-elf")
    elif arch == "xtensa":
        candidates.extend(["xtensa-esp-elf", "xtensa-esp32-elf", "xtensa-esp32s3-elf"])
    for prefix in dict.fromkeys(candidates):
        path = shutil.which(f"{prefix}-addr2line")
        if path:
            return path
    return None


def decode_frames(report: CrashReport, elf: Path, target: str | None = None) -> bool:
    """Resolve frame addresses with addr2line. Returns True if decoding ran."""
    tool = find_addr2line(target, report.arch)
    if tool is None or not report.frames or not elf.is_file():
        return False

    addresses = [f.address for f in report.frames]
    try:
        result = subprocess.run(
            [tool, "-pfiaC", "-e", str(elf), *addresses],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    # -a prints "0xADDR: func at file:line"; inlined frames add "(inlined by)" lines.
    decoded: dict[str, list[str]] = {}
    current: str | None = None
    for line in result.stdout.splitlines():
        if match := re.match(r"(0x[0-9a-fA-F]+):\s*(.*)", line):
            current = f"0x{int(match.group(1), 16):08x}"
            decoded[current] = [match.group(2).strip()]
        elif current and line.strip():
            decoded[current].append(line.strip())
    for frame in report.frames:
        if frame.address in decoded:
            frame.decoded = " | ".join(decoded[frame.address])
    report.addr2line = tool
    return True


def format_report(report: CrashReport) -> str:
    """Format a crash report as markdown."""
    if report.kind is None:
        return "# Crash Analysis\n\nNo panic signature found in the input.\n"

    lines = ["# Crash Analysis", "", "## Exception"]
    lines.append(f"- **Kind**: {report.kind}")
    if report.exception:
        lines.append(f"- **Exception**: {report.exception}")
    if report.core is not None:
        lines.append(f"- **Core**: {report.core}")
    if report.arch:
        lines.append(f"- **Architecture**: {report.arch}")
    if report.fault_address:
        lines.append(f"- **Fault address**: {report.fault_address}")
    if report.task:
        lines.append(f"- **Task**: {report.task}")
    if report.details:
        lines.append(f"- **Details**: {report.details}")
    lines.append("")

    if report.frames:
        lines.append("## Frames")
        lines.append("")
        lines.append("| # | Address | Source | Decoded |")
        lines.append("|---|---------|--------|---------|")
        for i, frame in enumerate(report.frames):
            lines.append(f"| {i} | `{frame.address}` | {frame.source} | {frame.decoded or '—'} |")
        if report.backtrace_corrupted:
            lines.append("")
            lines.append("⚠️ Backtrace is marked CORRUPTED — the stack itself was likely damaged.")
        if report.addr2line is None:
            lines.append("")
            lines.append(
                "Addresses not decoded. Pass `--elf build/<project>.elf` with the ESP-IDF "
                "toolchain on PATH (run `. $IDF_PATH/export.sh`)."
            )
        lines.append("")

    if report.classification:
        lines.append("## Likely Cause")
        for label, hint in zip(report.classification, report.hints):
            lines.append(f"- **{label}**: {hint}")
        lines.append("")

    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for ``esp32wb crash``."""
    parser.add_argument("path", help="File with the panic output, or - for stdin")
    parser.add_argument("--elf", type=Path, help="Application ELF for addr2line decoding")
    parser.add_argument("--target", choices=list_targets(), help="ESP-IDF target")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def run(args: argparse.Namespace) -> int:
    """Decode a panic for parsed CLI arguments and return the exit code."""
    if args.path == "-":
        text = sys.stdin.read()
    else:
        source = Path(args.path)
        if not source.is_file():
            print(f"ERROR: File not found: {source}")
            return 1
        text = source.read_text(encoding="utf-8", errors="replace")

    report = parse_panic(text)
    if args.elf:
        decode_frames(report, args.elf, args.target)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(format_report(report))
    return 0
