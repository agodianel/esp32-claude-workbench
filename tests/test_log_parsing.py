"""Tests for ESP32 serial log parsing and summarization."""

import textwrap

from esp32_workbench.logs import format_summary, parse_log_line, summarize_logs


class TestParseLogLine:
    """Test individual log line parsing."""

    def test_parse_error_line(self):
        line = "E (12345) wifi: Connection failed"
        entry = parse_log_line(line)
        assert entry is not None
        assert entry.level == "E"
        assert entry.tag == "wifi"
        assert entry.message == "Connection failed"

    def test_parse_info_line(self):
        line = "I (500) main: Application started"
        entry = parse_log_line(line)
        assert entry is not None
        assert entry.level == "I"
        assert entry.tag == "main"
        assert entry.message == "Application started"

    def test_parse_warning_line(self):
        line = "W (1000) nvs: NVS partition truncated"
        entry = parse_log_line(line)
        assert entry is not None
        assert entry.level == "W"
        assert entry.tag == "nvs"

    def test_parse_debug_line(self):
        line = "D (200) heap_init: At 0x3FFAE6E0 len 00001920"
        entry = parse_log_line(line)
        assert entry is not None
        assert entry.level == "D"

    def test_parse_verbose_line(self):
        line = "V (50) system: Detailed trace info"
        entry = parse_log_line(line)
        assert entry is not None
        assert entry.level == "V"

    def test_unparseable_line_returns_none(self):
        line = "This is not a log line"
        entry = parse_log_line(line)
        assert entry is None

    def test_empty_line_returns_none(self):
        entry = parse_log_line("")
        assert entry is None


class TestSummarizeLogs:
    """Test log summarization."""

    def test_count_by_level(self):
        log = textwrap.dedent("""\
            I (100) main: Starting
            I (200) main: Running
            W (300) wifi: Weak signal
            E (400) mqtt: Connection lost
            E (500) mqtt: Retry failed
        """)
        summary = summarize_logs(log)
        assert summary.info_count == 2
        assert summary.warning_count == 1
        assert summary.error_count == 2

    def test_detect_reset_reason(self):
        log = "rst:0x1 (POWERON_RESET),boot:0x13"
        summary = summarize_logs(log)
        assert summary.reset_reason is not None
        assert "POWERON" in summary.reset_reason

    def test_detect_watchdog_reset(self):
        log = "rst:0x7 (TG0WDT_SYS_RESET),boot:0x13"
        summary = summarize_logs(log)
        assert summary.reset_reason is not None
        assert "watchdog" in summary.reset_reason.lower()

    def test_detect_guru_meditation(self):
        log = "Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled."
        summary = summarize_logs(log)
        assert summary.crash_detected is True
        assert summary.crash_type == "LoadProhibited"

    def test_detect_boot_success(self):
        log = textwrap.dedent("""\
            I (100) cpu_start: Starting scheduler
            I (200) main: Application started successfully
        """)
        summary = summarize_logs(log)
        assert summary.boot_successful is True

    def test_no_crash_by_default(self):
        log = "I (100) main: Normal operation"
        summary = summarize_logs(log)
        assert summary.crash_detected is False
        assert summary.crash_type is None

    def test_tag_counting(self):
        log = textwrap.dedent("""\
            I (100) wifi: Connecting
            I (200) wifi: Connected
            I (300) mqtt: Connecting
        """)
        summary = summarize_logs(log)
        assert summary.tags["wifi"] == 2
        assert summary.tags["mqtt"] == 1

    def test_empty_log(self):
        summary = summarize_logs("")
        assert summary.total_lines == 0
        assert summary.error_count == 0


class TestFormatSummary:
    """Test summary formatting."""

    def test_format_produces_markdown(self):
        log = textwrap.dedent("""\
            I (100) main: Starting
            E (200) wifi: Failed to connect
        """)
        summary = summarize_logs(log)
        report = format_summary(summary)
        assert "# Log Summary Report" in report
        assert "## Overview" in report
        assert "Errors" in report

    def test_format_includes_crash_info(self):
        log = "Guru Meditation Error: Core  0 panic'ed (StoreProhibited). Exception was unhandled."
        summary = summarize_logs(log)
        report = format_summary(summary)
        assert "Crash Detected" in report
        assert "StoreProhibited" in report

    def test_format_includes_errors(self):
        log = "E (100) driver: Init failed"
        summary = summarize_logs(log)
        report = format_summary(summary)
        assert "Init failed" in report
