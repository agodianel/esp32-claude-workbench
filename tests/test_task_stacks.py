"""Tests for the FreeRTOS task stack checker tool."""

import textwrap
from pathlib import Path

from esp32_workbench.stacks import (
    TaskInfo,
    analyze_tasks,
    scan_file,
    try_parse_int,
)


def _write_temp_c(tmp_path: Path, filename: str, content: str) -> Path:
    filepath = tmp_path / filename
    filepath.write_text(textwrap.dedent(content), encoding="utf-8")
    return filepath


class TestTryParseInt:
    """Test stack size expression parsing."""

    def test_simple_integer(self):
        assert try_parse_int("4096") == 4096

    def test_multiplication(self):
        assert try_parse_int("4 * 1024") == 4096

    def test_macro_returns_none(self):
        assert try_parse_int("configMINIMAL_STACK_SIZE") is None

    def test_complex_expr_returns_none(self):
        assert try_parse_int("STACK_BASE + 2048") is None


class TestScanFile:
    """Test task creation detection."""

    def test_detect_xTaskCreate(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.c",
            """\
            void my_task(void *arg) {}
            void init() {
                xTaskCreate(my_task, "sensor_task", 4096, NULL, 5, NULL);
            }
        """,
        )
        tasks = scan_file(f)
        assert len(tasks) == 1
        assert tasks[0].name == "sensor_task"
        assert tasks[0].stack_size == 4096
        assert tasks[0].function == "my_task"
        assert tasks[0].core is None

    def test_detect_xTaskCreatePinnedToCore(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.c",
            """\
            void wifi_task(void *arg) {}
            void init() {
                xTaskCreatePinnedToCore(wifi_task, "wifi_loop", 8192, NULL, 10, NULL, 0);
            }
        """,
        )
        tasks = scan_file(f)
        assert len(tasks) == 1
        assert tasks[0].name == "wifi_loop"
        assert tasks[0].stack_size == 8192
        assert tasks[0].core == "0"

    def test_detect_multiple_tasks(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.c",
            """\
            void task_a(void *arg) {}
            void task_b(void *arg) {}
            void init() {
                xTaskCreate(task_a, "alpha", 2048, NULL, 3, NULL);
                xTaskCreate(task_b, "beta", 4096, NULL, 5, NULL);
            }
        """,
        )
        tasks = scan_file(f)
        assert len(tasks) == 2
        names = {t.name for t in tasks}
        assert "alpha" in names
        assert "beta" in names

    def test_no_tasks_in_empty_file(self, tmp_path):
        f = _write_temp_c(tmp_path, "test.c", "// empty\n")
        tasks = scan_file(f)
        assert len(tasks) == 0


class TestAnalyzeTasks:
    """Test stack size analysis."""

    def test_tiny_stack_error(self):
        tasks = [
            TaskInfo(
                name="tiny",
                stack_size=512,
                stack_expr="512",
                priority="5",
                core=None,
                function="tiny_fn",
                file="test.c",
                line=1,
            )
        ]
        issues = analyze_tasks(tasks)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) >= 1
        assert any("below absolute minimum" in i.message for i in errors)

    def test_wifi_task_small_stack_warning(self):
        tasks = [
            TaskInfo(
                name="wifi_handler",
                stack_size=2048,
                stack_expr="2048",
                priority="5",
                core=None,
                function="wifi_handler_fn",
                file="test.c",
                line=1,
            )
        ]
        issues = analyze_tasks(tasks)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert any("WIFI" in i.message for i in warnings)

    def test_https_task_small_stack_warning(self):
        tasks = [
            TaskInfo(
                name="https_client",
                stack_size=4096,
                stack_expr="4096",
                priority="5",
                core=None,
                function="https_task",
                file="test.c",
                line=1,
            )
        ]
        issues = analyze_tasks(tasks)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert any("HTTP" in i.message for i in warnings)

    def test_adequate_stack_no_issues(self):
        tasks = [
            TaskInfo(
                name="sensor_read",
                stack_size=4096,
                stack_expr="4096",
                priority="5",
                core=None,
                function="read_sensor",
                file="test.c",
                line=1,
            )
        ]
        issues = analyze_tasks(tasks)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 0

    def test_macro_stack_info(self):
        tasks = [
            TaskInfo(
                name="dynamic",
                stack_size=None,
                stack_expr="MY_STACK_SIZE",
                priority="5",
                core=None,
                function="dynamic_fn",
                file="test.c",
                line=1,
            )
        ]
        issues = analyze_tasks(tasks)
        infos = [i for i in issues if i.severity == "INFO"]
        assert any("macro" in i.message.lower() for i in infos)
