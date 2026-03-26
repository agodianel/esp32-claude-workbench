"""Tests for CLI main entry points to ensure formatting and CLI logic works."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.analyze_sdkconfig import main as analyze_sdkconfig_main
from tools.check_task_stacks import main as check_task_stacks_main
from tools.scan_pins import main as scan_pins_main


def test_analyze_sdkconfig_main(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test analyze-sdkconfig CLI logic."""
    # Create mock sdkconfig
    sdkconfig_file = tmp_path / "sdkconfig"
    sdkconfig_file.write_text('CONFIG_IDF_TARGET="esp32"\nCONFIG_ESP_TASK_WDT_TIMEOUT_S=1\n')

    # Mock sys.argv
    with patch.object(sys, "argv", ["analyze-sdkconfig", str(sdkconfig_file)]):
        # We expect it to exit with 1 because there are warnings (timeout=1 will trigger a warning)
        # Actually our script returns 1 if there are ERRORs. Let's make an error:
        sdkconfig_file.write_text("CONFIG_ESP_TLS_SKIP_SERVER_CERT_VERIFY=y\n")
        with pytest.raises(SystemExit) as exc:
            analyze_sdkconfig_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Security" in out


def test_analyze_sdkconfig_main_no_args(capsys: pytest.CaptureFixture):
    """Test analyze-sdkconfig CLI without args."""
    with patch.object(sys, "argv", ["analyze-sdkconfig"]):
        with pytest.raises(SystemExit) as exc:
            analyze_sdkconfig_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Usage:" in out


def test_check_task_stacks_main(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test check-stacks CLI logic."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    source_file = main_dir / "test.c"
    source_file.write_text('void init() { xTaskCreate(mytask, "wifi", 512, NULL, 5, NULL); }')

    with patch.object(sys, "argv", ["check-stacks", str(main_dir)]):
        with pytest.raises(SystemExit) as exc:
            check_task_stacks_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Task Stack Analysis Report" in out
    assert "below absolute minimum" in out


def test_check_task_stacks_main_no_args(capsys: pytest.CaptureFixture):
    """Test check-stacks CLI without args."""
    with patch.object(sys, "argv", ["check-stacks"]):
        with pytest.raises(SystemExit) as exc:
            check_task_stacks_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Usage:" in out


def test_scan_pins_main(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test scan-pins CLI logic."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    source_file = main_dir / "test.c"
    source_file.write_text("#define BAD_PIN GPIO_NUM_6\n")  # 6 is internal flash (ERROR)

    with patch.object(sys, "argv", ["scan-pins", str(main_dir)]):
        with pytest.raises(SystemExit) as exc:
            scan_pins_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Pin Scan Report" in out
    assert "internal SPI flash" in out


def test_scan_pins_main_no_args(capsys: pytest.CaptureFixture):
    """Test scan-pins CLI without args."""
    with patch.object(sys, "argv", ["scan-pins"]):
        with pytest.raises(SystemExit) as exc:
            scan_pins_main()
        assert exc.value.code == 1

    out, _ = capsys.readouterr()
    assert "Usage:" in out
