"""Tests for ESP-IDF build/flash log triage."""

import pytest

from esp32_workbench.buildlog import format_report, triage


@pytest.mark.parametrize(
    ("line", "category", "fix_fragment"),
    [
        (
            "main.c:3:10: fatal error: esp_wifi.h: No such file or directory",
            "missing_header",
            "`esp_wifi`",
        ),
        (
            "main.c:(.text+0x1a): undefined reference to `sensor_init'",
            "undefined_reference",
            "sensor_init",
        ),
        ("region `iram0_0_seg' overflowed by 1312 bytes", "iram_overflow", "1312"),
        ("region `dram0_0_seg' overflowed by 88 bytes", "dram_overflow", "88"),
        (
            "Error: app partition is too small for binary app.bin size 0x110000",
            "app_partition_too_small",
            "partition",
        ),
        (
            "A fatal error occurred: This chip is ESP32-S3, not ESP32. Wrong --chip argument?",
            "target_mismatch",
            "set-target",
        ),
        ("Failed to resolve component 'esp_driver_i2s'.", "component_not_found", "esp_driver_i2s"),
        (
            "main.c:9:5: error: implicit declaration of function 'adc1_get_raw'",
            "implicit_declaration",
            "adc1_get_raw",
        ),
        (
            "main.c:12:9: error: 'CONFIG_BLINK_GPIO' undeclared (first use in this function)",
            "undeclared_config",
            "menuconfig",
        ),
        (
            "main.c:5:1: error: unknown type name 'i2c_master_bus_handle_t'",
            "unknown_type",
            "i2c_master_bus_handle_t",
        ),
        (
            "main.c:7:3: error: unused variable 'x' [-Werror=unused-variable]",
            "werror",
            "unused-variable",
        ),
        (
            "W (291) ADC: legacy adc driver is deprecated, please migrate",
            "legacy_driver",
            "migration guide",
        ),
        (
            "A fatal error occurred: Failed to connect to ESP32: No serial data received.",
            "no_serial_data",
            "BOOT",
        ),
        (
            "could not open port '/dev/ttyUSB0': [Errno 13] Permission denied: '/dev/ttyUSB0'",
            "serial_permission",
            "dialout",
        ),
    ],
)
def test_rules(line, category, fix_fragment):
    findings = triage(line)
    assert findings and findings[0].category == category
    assert fix_fragment in findings[0].fix


def test_dedup_and_order():
    log = "\n".join(
        [
            "W (1) ADC: legacy adc driver is deprecated",
            "fatal error: nvs_flash.h: No such file or directory",
            "fatal error: nvs_flash.h: No such file or directory",
        ]
    )
    findings = triage(log)
    assert [f.category for f in findings] == ["missing_header", "legacy_driver"]


def test_unknown_header_has_generic_fix():
    findings = triage("fatal error: my_lib.h: No such file or directory")
    assert "REQUIRES" in findings[0].fix and "provided by" not in findings[0].fix


def test_report_when_nothing_found():
    assert "No known failure pattern" in format_report(triage("Project build complete."))
