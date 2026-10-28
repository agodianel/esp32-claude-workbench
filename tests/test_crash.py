"""Tests for the panic decoder (Xtensa + RISC-V)."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from esp32_workbench.crash import decode_frames, find_addr2line, format_report, parse_panic

XTENSA_NULL = """\
I (312) app: starting
Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled.

Core  0 register dump:
PC      : 0x400d1a2b  PS      : 0x00060630  A0      : 0x800d1c3d  A1      : 0x3ffb5f60
A2      : 0x00000000  A3      : 0x3ffb5f8c  A4      : 0x00000001  A5      : 0x3ffb5f70
EXCVADDR: 0x00000004  LBEG    : 0x4000c2e0  LEND    : 0x4000c2f6  LCOUNT  : 0xffffffff

Backtrace: 0x400d1a28:0x3ffb5f60 0x400d1c3a:0x3ffb5f90 0x400d5b2e:0x3ffb5fb0
"""

RISCV_NULL = """\
Guru Meditation Error: Core  0 panic'ed (Load access fault). Exception was unhandled.

Core  0 register dump:
MEPC    : 0x4200a1b2  RA      : 0x4200a1a0  SP      : 0x3fc8f0e0  GP      : 0x3fc8b400
MSTATUS : 0x00001881  MTVEC   : 0x40380001  MCAUSE  : 0x00000005  MTVAL   : 0x00000000
MHARTID : 0x00000000
"""


def test_xtensa_null_pointer():
    r = parse_panic(XTENSA_NULL)
    assert (r.kind, r.exception, r.core, r.arch) == (
        "guru_meditation",
        "LoadProhibited",
        0,
        "xtensa",
    )
    assert r.fault_address == "0x00000004"
    assert [f.address for f in r.frames] == ["0x400d1a2b", "0x400d1a28", "0x400d1c3a", "0x400d5b2e"]
    assert "null_pointer" in r.classification


def test_riscv_null_pointer():
    r = parse_panic(RISCV_NULL)
    assert r.arch == "riscv"
    assert r.fault_address == "0x00000000"
    assert [f.address for f in r.frames] == ["0x4200a1b2", "0x4200a1a0"]
    assert "null_pointer" in r.classification


def test_riscv_mcause_used_when_no_guru_line():
    r = parse_panic("MEPC    : 0x42001000  MCAUSE  : 0x00000002  MTVAL   : 0x00000000\n")
    assert r.exception == "Illegal instruction"
    assert "illegal_instruction" in r.classification


def test_stack_overflow():
    r = parse_panic(
        "***ERROR*** A stack overflow in task sensor_task has been detected.\n\n"
        "Backtrace: 0x40081a2a:0x3ffb8e20 0x40086d55:0x3ffb8e40 |<-CORRUPTED\n"
    )
    assert (r.kind, r.task) == ("stack_overflow", "sensor_task")
    assert r.backtrace_corrupted
    assert "stack_overflow" in r.classification


def test_abort_and_esp_error_check():
    r = parse_panic(
        "ESP_ERROR_CHECK failed: esp_err_t 0x103 (ESP_ERR_INVALID_STATE) at 0x400d2f18\n"
        "abort() was called at PC 0x400d2f1b on core 1\n"
    )
    assert r.kind == "esp_error_check"
    assert r.details == "ESP_ERR_INVALID_STATE (0x103)"
    assert r.core == 1
    assert r.frames[0].source == "abort"


def test_assert():
    r = parse_panic("assert failed: xQueueGenericSend queue.c:832 (pxQueue)\n")
    assert r.kind == "assert"
    assert "queue.c:832" in r.details


def test_interrupt_watchdog():
    r = parse_panic("Guru Meditation Error: Core  1 panic'ed (Interrupt wdt timeout on CPU1).\n")
    assert r.kind == "interrupt_watchdog"
    assert "interrupt_watchdog" in r.classification


def test_no_panic():
    r = parse_panic("I (10) boot: all good\n")
    assert r.kind is None
    assert "No panic signature" in format_report(r)


def test_report_mentions_missing_elf():
    assert "--elf" in format_report(parse_panic(XTENSA_NULL))


def test_find_addr2line_prefers_target_prefix():
    with patch(
        "shutil.which", side_effect=lambda n: f"/x/{n}" if n.startswith("riscv32") else None
    ):
        assert find_addr2line("esp32c3", "riscv") == "/x/riscv32-esp-elf-addr2line"
    with patch(
        "shutil.which", side_effect=lambda n: f"/x/{n}" if n == "xtensa-esp-elf-addr2line" else None
    ):
        assert find_addr2line(None, "xtensa") == "/x/xtensa-esp-elf-addr2line"


def test_decode_frames(tmp_path: Path):
    elf = tmp_path / "app.elf"
    elf.write_bytes(b"\x7fELF")
    output = (
        "0x400d1a2b: app_main at /p/main/main.c:42\n"
        "0x400d1a28: read_sensor at /p/main/sensor.c:17\n"
        " (inlined by) sensor_task at /p/main/sensor.c:30\n"
    )
    report = parse_panic(XTENSA_NULL)
    fake = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
    with (
        patch("esp32_workbench.crash.find_addr2line", return_value="/x/addr2line"),
        patch("subprocess.run", return_value=fake) as run,
    ):
        assert decode_frames(report, elf, "esp32")
    assert run.call_args.args[0][:4] == ["/x/addr2line", "-pfiaC", "-e", str(elf)]
    assert report.frames[0].decoded == "app_main at /p/main/main.c:42"
    assert "inlined by" in report.frames[1].decoded
    assert "app_main at" in format_report(report)


def test_decode_frames_without_toolchain(tmp_path: Path):
    with patch("esp32_workbench.crash.find_addr2line", return_value=None):
        assert not decode_frames(parse_panic(XTENSA_NULL), tmp_path / "app.elf")
