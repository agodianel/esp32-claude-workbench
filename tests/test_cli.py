"""End-to-end tests for the esp32wb command line."""

import json
from pathlib import Path

import pytest

from esp32_workbench.cli import main


def run(capsys, *argv) -> tuple[int, str]:
    with pytest.raises(SystemExit) as exc:
        main(list(argv))
    return exc.value.code, capsys.readouterr().out


def make_project(root: Path) -> Path:
    (root / "main").mkdir(parents=True)
    (root / "sdkconfig.defaults").write_text('CONFIG_IDF_TARGET="esp32s3"\n')
    (root / "main" / "main.c").write_text(
        '#define LED_PIN 38\nvoid app_main(void) { xTaskCreate(t, "wifi", 8192, NULL, 5, NULL); }\n'
    )
    (root / "partitions.csv").write_text(
        "nvs, data, nvs, , 0x6000,\nfactory, app, factory, , 1M,\n"
    )
    return root


def test_version(capsys):
    code, out = run(capsys, "--version")
    assert code == 0 and "esp32wb" in out


def test_chip_list_and_detail(capsys):
    code, out = run(capsys, "chip")
    assert code == 0 and "esp32c6" in out
    code, out = run(capsys, "chip", "esp32s3", "--json")
    assert json.loads(out)["target"] == "esp32s3"


def test_doctor_clean_project(capsys, tmp_path: Path):
    code, out = run(capsys, "doctor", str(make_project(tmp_path)))
    assert code == 0
    assert "ESP32-S3" in out and "Partition Table Report" in out


def test_doctor_json_reports_errors(capsys, tmp_path: Path):
    project = make_project(tmp_path)
    (project / "main" / "bad.c").write_text("#define CS_PIN 27\n")
    code, out = run(capsys, "doctor", str(project), "--json")
    assert code == 1 and json.loads(out)["checks"]["pins"]["errors"] == 1


def test_pins_json(capsys, tmp_path: Path):
    code, out = run(capsys, "pins", str(make_project(tmp_path)), "--json")
    data = json.loads(out)
    assert code == 0 and data["target"] == "esp32s3" and "report" not in data


def test_mission_new_then_validate(capsys, tmp_path: Path):
    code, out = run(
        capsys, "mission", "new", "OTA rollback", "-d", "Add rollback", "--root", str(tmp_path)
    )
    assert code == 0
    path = next((tmp_path / "missions").glob("*-ota-rollback.md"))
    code, out = run(capsys, "mission", "validate", str(path), "--json")
    assert code == 0 and json.loads(out)[str(path)]["missing_required"] == []
    code, _ = run(capsys, "mission", "new", "OTA rollback", "--root", str(tmp_path))
    assert code == 1  # refuses to overwrite


@pytest.mark.parametrize(
    ("command", "content"),
    [
        ("crash", "Guru Meditation Error: Core  0 panic'ed (StoreProhibited)."),
        ("build-log", "fatal error: esp_wifi.h: No such file or directory"),
        ("logs", "E (100) wifi: connect failed"),
    ],
)
def test_text_commands(capsys, tmp_path: Path, command, content):
    f = tmp_path / "in.txt"
    f.write_text(content + "\n")
    code, out = run(capsys, command, str(f), "--json")
    assert code == 0 and json.loads(out)


def test_partitions_command(capsys, tmp_path: Path):
    csv = make_project(tmp_path) / "partitions.csv"
    code, out = run(capsys, "partitions", str(csv), "--flash-size", "4M")
    assert code == 0 and "factory" in out


def test_init_writes_rules_and_integrations(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("esp32_workbench.installer.default_server_command", lambda: ["esp32wb-mcp"])
    code, out = run(capsys, "init", str(tmp_path), "--agents", "claude", "codex")
    assert code == 0
    assert (tmp_path / "AGENTS.md").is_file() and (tmp_path / "CLAUDE.md").is_file()
    assert (tmp_path / "missions").is_dir()
    assert (tmp_path / ".claude/skills").is_dir() and (tmp_path / ".agents/skills").is_dir()
    assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["esp32-workbench"]
    assert "esp32-workbench" in (tmp_path / ".codex/config.toml").read_text()
    code, out = run(capsys, "init", str(tmp_path))
    assert "skip" in out  # keeps existing AGENTS.md without --force


def test_mcp_config_print(capsys):
    code, out = run(capsys, "mcp-config", "gemini", "--command", "esp32wb-mcp")
    assert json.loads(out) == {
        "mcpServers": {"esp32-workbench": {"command": "esp32wb-mcp", "args": []}}
    }


def test_missing_file_errors(capsys, tmp_path: Path):
    code, out = run(capsys, "sdkconfig", str(tmp_path / "nope"))
    assert code == 1 and "not found" in out
