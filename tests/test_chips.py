"""Tests for the chip database: schema consistency and spot-checked facts."""

from pathlib import Path

import pytest

from esp32_workbench.chips import detect_target, list_targets, load_chip, resolve_chip

ALL_TARGETS = list_targets()


def test_all_expected_targets_present():
    expected = {"esp32", "esp32s2", "esp32s3", "esp32c2", "esp32c3", "esp32c5", "esp32c6"}
    expected |= {"esp32h2", "esp32p4"}
    assert expected <= set(ALL_TARGETS)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_every_referenced_pin_exists(target):
    chip = load_chip(target)
    referenced = (
        chip.input_only
        | chip.flash_spi
        | chip.psram_pins
        | chip.usb_jtag
        | chip.jtag
        | set(chip.strapping)
        | set(chip.adc1)
        | set(chip.adc2)
        | set(chip.conditional)
    )
    assert referenced <= chip.gpios, referenced - chip.gpios


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_every_chip_cites_espressif_sources(target):
    chip = load_chip(target)
    assert chip.sources
    assert all("espressif.com" in url for url in chip.sources)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_toolchain_matches_architecture(target):
    chip = load_chip(target)
    if chip.arch == "riscv":
        assert chip.toolchain_prefix == "riscv32-esp-elf"
    else:
        assert chip.toolchain_prefix.startswith("xtensa-")


@pytest.mark.parametrize(
    ("target", "flash", "strapping", "uart0"),
    [
        ("esp32", {6, 7, 8, 9, 10, 11}, {0, 2, 5, 12, 15}, (1, 3)),
        ("esp32s3", {27, 28, 29, 30, 31, 32}, {0, 3, 45, 46}, (43, 44)),
        ("esp32c3", {12, 13, 14, 15, 16, 17}, {2, 8, 9}, (21, 20)),
        ("esp32c6", {24, 25, 26, 27, 28, 29, 30}, {4, 5, 8, 9, 15}, (16, 17)),
    ],
)
def test_known_facts(target, flash, strapping, uart0):
    chip = load_chip(target)
    assert chip.flash_spi == flash
    assert set(chip.strapping) == strapping
    assert (chip.uart0_tx, chip.uart0_rx) == uart0


def test_input_only_only_on_classic_and_s2():
    assert load_chip("esp32").input_only == {34, 35, 36, 37, 38, 39}
    assert load_chip("esp32s3").input_only == frozenset()
    assert load_chip("esp32c3").input_only == frozenset()


def test_adc2_wifi_policy():
    assert load_chip("esp32").adc2_wifi_policy == "forbidden"
    assert load_chip("esp32s3").adc2_wifi_policy == "arbitrated"
    assert load_chip("esp32c6").adc2_wifi_policy is None  # no ADC2
    assert load_chip("esp32p4").adc2_wifi_policy is None  # no Wi-Fi


def test_target_name_normalisation():
    assert load_chip("ESP32-S3").target == "esp32s3"


def test_unknown_target():
    with pytest.raises(ValueError, match="Unknown target"):
        load_chip("esp8266")


def test_detect_target_from_sdkconfig(tmp_path: Path):
    (tmp_path / "sdkconfig.defaults").write_text('CONFIG_IDF_TARGET="esp32c6"\n')
    (tmp_path / "main").mkdir()
    assert detect_target(tmp_path / "main") == "esp32c6"


def test_sdkconfig_wins_over_defaults(tmp_path: Path):
    (tmp_path / "sdkconfig.defaults").write_text('CONFIG_IDF_TARGET="esp32"\n')
    (tmp_path / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n')
    assert detect_target(tmp_path) == "esp32s3"


def test_resolve_chip_defaults_to_esp32(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert resolve_chip(None, tmp_path).target == "esp32"
