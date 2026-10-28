"""Tests for the GPIO pin scanner tool."""

import textwrap
from pathlib import Path

from esp32_workbench.pins import (
    PinAssignment,
    analyze_pins,
    scan_file,
)

# Write test C files to a temp location
ROOT = Path(__file__).parent.parent


def _write_temp_c(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper to write a temp C file."""
    filepath = tmp_path / filename
    filepath.write_text(textwrap.dedent(content), encoding="utf-8")
    return filepath


class TestScanFile:
    """Test GPIO pin detection from source files."""

    def test_detect_gpio_num_defines(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.h",
            """\
            #define LED_PIN GPIO_NUM_2
            #define BTN_PIN GPIO_NUM_0
        """,
        )
        pins = scan_file(f)
        gpios = {p.gpio for p in pins}
        assert 2 in gpios
        assert 0 in gpios

    def test_detect_named_pin_defines(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.h",
            """\
            #define I2C_SDA_PIN 21
            #define I2C_SCL_PIN 22
        """,
        )
        pins = scan_file(f)
        gpios = {p.gpio for p in pins}
        assert 21 in gpios
        assert 22 in gpios

    def test_detect_i2c_config(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.c",
            """\
            i2c_config_t conf = {
                .sda_io_num = 21,
                .scl_io_num = 22,
            };
        """,
        )
        pins = scan_file(f)
        gpios = {p.gpio for p in pins}
        assert 21 in gpios
        assert 22 in gpios

    def test_detect_spi_config(self, tmp_path):
        f = _write_temp_c(
            tmp_path,
            "test.c",
            """\
            spi_bus_config_t bus = {
                .mosi_io_num = 23,
                .miso_io_num = 19,
                .sclk_io_num = 18,
            };
        """,
        )
        pins = scan_file(f)
        gpios = {p.gpio for p in pins}
        assert 23 in gpios
        assert 19 in gpios
        assert 18 in gpios

    def test_no_pins_in_empty_file(self, tmp_path):
        f = _write_temp_c(tmp_path, "empty.c", "// no pins here\n")
        pins = scan_file(f)
        assert len(pins) == 0


class TestAnalyzePins:
    """Test pin conflict and reserved pin detection."""

    def test_forbidden_pin_detected(self):
        assignments = [PinAssignment(gpio=6, purpose="MY_CS", file="test.h", line=1, raw="")]
        issues = analyze_pins(assignments)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) >= 1
        assert any("internal SPI flash" in i.message for i in errors)

    def test_strapping_pin_warning(self):
        assignments = [PinAssignment(gpio=0, purpose="MY_BTN", file="test.h", line=1, raw="")]
        issues = analyze_pins(assignments)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert any("strapping" in i.message.lower() for i in warnings)

    def test_input_only_as_output_error(self):
        assignments = [PinAssignment(gpio=34, purpose="LED_PIN", file="test.h", line=1, raw="")]
        issues = analyze_pins(assignments)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("input-only" in i.message for i in errors)

    def test_double_assignment_detected(self):
        assignments = [
            PinAssignment(gpio=5, purpose="SPI_CS", file="spi.h", line=1, raw=""),
            PinAssignment(gpio=5, purpose="LED_PIN", file="led.h", line=2, raw=""),
        ]
        issues = analyze_pins(assignments)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("multiple functions" in i.message for i in errors)

    def test_adc2_with_wifi_error(self):
        assignments = [PinAssignment(gpio=25, purpose="adc_channel", file="test.c", line=1, raw="")]
        issues = analyze_pins(assignments, uses_wifi=True)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("ADC2" in i.message for i in errors)

    def test_adc2_without_wifi_ok(self):
        assignments = [PinAssignment(gpio=25, purpose="adc_channel", file="test.c", line=1, raw="")]
        issues = analyze_pins(assignments, uses_wifi=False)
        adc_errors = [i for i in issues if "ADC2" in i.message]
        assert len(adc_errors) == 0

    def test_valid_pins_no_errors(self):
        assignments = [
            PinAssignment(gpio=16, purpose="UART_TX", file="test.h", line=1, raw=""),
            PinAssignment(gpio=17, purpose="UART_RX", file="test.h", line=2, raw=""),
        ]
        issues = analyze_pins(assignments)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 0
