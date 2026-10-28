"""Tests for the partition table checker."""

from pathlib import Path

import pytest

from esp32_workbench.partitions import check_file, check_partitions, parse_csv, parse_size

OTA_TABLE = """\
# Name,   Type, SubType, Offset,  Size, Flags
nvs,      data, nvs,     0x9000,  0x6000,
otadata,  data, ota,     0xf000,  0x2000,
phy_init, data, phy,     0x11000, 0x1000,
ota_0,    app,  ota_0,   0x20000, 1M,
ota_1,    app,  ota_1,   ,        1M,
"""

FACTORY_TABLE = """\
nvs,      data, nvs,     ,        0x6000,
phy_init, data, phy,     ,        0x1000,
factory,  app,  factory, ,        1M,
"""


def severities(issues, level):
    return [i.message for i in issues if i.severity == level]


@pytest.mark.parametrize(
    ("text", "value"), [("0x1000", 4096), ("4K", 4096), ("1M", 1 << 20), ("100", 100)]
)
def test_parse_size(text, value):
    assert parse_size(text) == value


def test_blank_offsets_follow_idf_alignment():
    parts = {p.name: p for p in parse_csv(FACTORY_TABLE)}
    assert parts["nvs"].offset == 0x9000
    assert parts["phy_init"].offset == 0xF000
    assert parts["factory"].offset == 0x10000  # apps are 64 KB aligned


def test_valid_ota_table_has_no_errors():
    parts = parse_csv(OTA_TABLE)
    assert parts[-1].offset == 0x120000
    assert not severities(check_partitions(parts, flash_size=4 << 20), "ERROR")


def test_exceeds_flash():
    errors = severities(check_partitions(parse_csv(OTA_TABLE), flash_size=2 << 20), "ERROR")
    assert any("beyond the 2 MB flash" in m for m in errors)


def test_misaligned_app_and_overlap():
    parts = parse_csv("nvs, data, nvs, 0x9000, 0x6000,\nfactory, app, factory, 0xC000, 1M,\n")
    errors = severities(check_partitions(parts), "ERROR")
    assert any("not 64 KB" in m for m in errors)
    assert any("Overlaps" in m for m in errors)


def test_ota_without_otadata_and_single_slot():
    parts = parse_csv("nvs, data, nvs, , 0x6000,\nota_0, app, ota_0, , 1M,\n")
    issues = check_partitions(parts)
    assert any("otadata" in m for m in severities(issues, "ERROR"))
    assert any("Only one OTA slot" in m for m in severities(issues, "WARNING"))


def test_missing_nvs_small_nvs_and_long_name():
    issues = check_partitions(parse_csv("a_very_long_partition_name, app, factory, , 1M,\n"))
    assert any("No NVS" in m for m in severities(issues, "WARNING"))
    assert any("longer than 15" in m for m in severities(issues, "ERROR"))
    small = check_partitions(parse_csv("nvs, data, nvs, , 0x2000,\nfactory, app, factory, , 1M,\n"))
    assert any("smaller than the recommended" in m for m in severities(small, "WARNING"))


def test_coredump_required_when_enabled():
    errors = severities(check_partitions(parse_csv(FACTORY_TABLE), coredump_enabled=True), "ERROR")
    assert any("coredump" in m for m in errors)


def test_app_binary_fit():
    parts = parse_csv(FACTORY_TABLE)
    assert severities(check_partitions(parts, app_size=(1 << 20) + 1), "ERROR")
    assert any(
        "95%" in m
        for m in severities(check_partitions(parts, app_size=int(0.95 * (1 << 20))), "WARNING")
    )


def test_check_file_reads_sdkconfig(tmp_path: Path):
    csv = tmp_path / "partitions.csv"
    csv.write_text(OTA_TABLE)
    cfg = tmp_path / "sdkconfig"
    cfg.write_text('CONFIG_ESPTOOLPY_FLASHSIZE="2MB"\nCONFIG_PARTITION_TABLE_OFFSET=0x8000\n')
    result = check_file(csv, cfg)
    assert result["errors"] >= 1
    assert "Partition Table Report" in result["report"]


def test_invalid_size():
    with pytest.raises(ValueError):
        parse_csv("nvs, data, nvs, , lots,\n")
