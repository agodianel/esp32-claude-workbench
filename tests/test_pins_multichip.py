"""Chip-aware pin audit tests (v0.2 behaviour)."""

import textwrap
from pathlib import Path

from esp32_workbench.chips import load_chip
from esp32_workbench.pins import PinAssignment, analyze_pins, scan, scan_file, strip_comments


def a(gpio: int, purpose: str) -> PinAssignment:
    return PinAssignment(gpio=gpio, purpose=purpose, file="main.c", line=1, raw="")


def messages(issues, severity):
    return [i.message for i in issues if i.severity == severity]


def test_s3_gpio38_output_is_fine():
    """GPIO 38 is input-only on classic ESP32 but a normal pin on ESP32-S3."""
    assert not messages(analyze_pins([a(38, "LED_PIN")], chip=load_chip("esp32s3")), "ERROR")
    assert messages(analyze_pins([a(38, "LED_PIN")], chip=load_chip("esp32")), "ERROR")


def test_s3_flash_pin_is_error():
    errors = messages(analyze_pins([a(27, "CS_PIN")], chip=load_chip("esp32s3")), "ERROR")
    assert any("SPI flash" in m for m in errors)


def test_s3_usb_pin_warning():
    warnings = messages(analyze_pins([a(19, "LED_PIN")], chip=load_chip("esp32s3")), "WARNING")
    assert any("USB" in m for m in warnings)


def test_c3_strapping_and_conditional_pin():
    chip = load_chip("esp32c3")
    assert any(
        "strapping" in m for m in messages(analyze_pins([a(9, "BTN_PIN")], chip=chip), "WARNING")
    )
    warnings = messages(analyze_pins([a(11, "LED_PIN")], chip=chip), "WARNING")
    assert any("VDD_SPI" in m for m in warnings)


def test_nonexistent_gpio():
    errors = messages(analyze_pins([a(25, "LED_PIN")], chip=load_chip("esp32c3")), "ERROR")
    assert any("does not exist" in m for m in errors)


def test_adc2_wifi_forbidden_vs_arbitrated():
    classic = analyze_pins([a(25, "SENSOR_ADC_PIN")], uses_wifi=True, chip=load_chip("esp32"))
    assert any("ADC2" in m for m in messages(classic, "ERROR"))
    s3 = analyze_pins([a(11, "SENSOR_ADC_PIN")], uses_wifi=True, chip=load_chip("esp32s3"))
    assert not any("ADC2" in m for m in messages(s3, "ERROR"))
    assert any("ADC2" in m for m in messages(s3, "WARNING"))


def test_comments_are_ignored(tmp_path: Path):
    f = tmp_path / "main.c"
    f.write_text("// #define LED_PIN GPIO_NUM_6\n/* .sda_io_num = 7, */\n#define BTN_PIN 4\n")
    pins = scan_file(f)
    assert [p.gpio for p in pins] == [4]
    assert pins[0].line == 3


def test_strip_comments_keeps_line_numbers():
    src = "a /* x\ny */ b\n// c\nd"
    assert strip_comments(src).count("\n") == src.count("\n")


def test_adc2_channel_macro_maps_to_gpio(tmp_path: Path):
    f = tmp_path / "adc.c"
    f.write_text("adc2_get_raw(ADC2_CHANNEL_8, ADC_WIDTH_BIT_12, &raw);\n")
    pins = scan_file(f, load_chip("esp32"))
    assert [(p.gpio, p.purpose) for p in pins] == [(25, "adc2_channel_8")]


def test_scan_autodetects_target(tmp_path: Path):
    (tmp_path / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n')
    main = tmp_path / "main"
    main.mkdir()
    (main / "main.c").write_text(
        textwrap.dedent("""\
        #define LED_PIN 38
        #define FLASH_CS_PIN 27
    """)
    )
    result = scan(main)
    assert result["target"] == "esp32s3"
    assert result["errors"] == 1
    assert result["issues"][0]["gpio"] == 27


def test_scan_single_file_and_skip_build(tmp_path: Path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "gen.c").write_text("#define LED_PIN 6\n")
    src = tmp_path / "main.c"
    src.write_text("#define LED_PIN 4\n")
    assert scan(tmp_path, "esp32")["errors"] == 0
    assert scan(src, "esp32")["assignments"][0]["gpio"] == 4
