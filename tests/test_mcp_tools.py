import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

mcp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp"))


def load_mcp_tool(name):
    file_path = os.path.join(mcp_path, "tools", f"{name}.py")
    module_name = f"mcp_local_tools_{name}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mission_generator = load_mcp_tool("mission_generator")
pin_audit = load_mcp_tool("pin_audit")
sdkconfig_check = load_mcp_tool("sdkconfig_check")
search_docs = load_mcp_tool("search_docs")

generate_mission = mission_generator.generate_mission
run_pin_audit = pin_audit.run_pin_audit
run_sdkconfig_check = sdkconfig_check.run_sdkconfig_check
_load_resources = search_docs._load_resources
search_esp_docs = search_docs.search_esp_docs


def test_run_pin_audit(tmp_path):
    c_file = tmp_path / "main.c"
    c_file.write_text(
        """
    #define LED_PIN GPIO_NUM_34
    gpio_set_direction(GPIO_NUM_34, GPIO_MODE_OUTPUT);
    gpio_set_direction(GPIO_NUM_0, GPIO_MODE_INPUT);
    #define SENSOR ADC1_CHANNEL_5 // Safe
    esp_wifi_init(&cfg);
    gpio_set_direction(GPIO_NUM_12, GPIO_MODE_INPUT);
    #define BAD_PIN GPIO_NUM_6
    """,
        encoding="utf-8",
    )
    result = run_pin_audit(str(c_file))
    assert "[CRITICAL]" in result


def test_run_pin_audit_safe(tmp_path):
    c_file = tmp_path / "main.c"
    c_file.write_text(
        """
    gpio_set_direction(GPIO_NUM_21, GPIO_MODE_OUTPUT);
    """,
        encoding="utf-8",
    )
    result = run_pin_audit(str(c_file))
    assert "[INFO]" in result


def test_run_pin_audit_file_not_found():
    result = run_pin_audit("nonexistent_file.c")
    assert "Error: File" in result


def test_sdkconfig_check_all_branches(tmp_path):
    sdk_file_1 = tmp_path / "sdk1"
    sdk_file_1.write_text(
        """
CONFIG_ESP_SYSTEM_PANIC_PRINT_REBOOT=y
CONFIG_BOOTLOADER_LOG_LEVEL=3
CONFIG_ESP_TASK_WDT_EN=y
CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH=y
CONFIG_MBEDTLS_CERTIFICATE_BUNDLE_DEFAULT_FULL=y
CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM="16"
CONFIG_FREERTOS_HZ="1000"
# CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG is not set
    """,
        encoding="utf-8",
    )
    res1 = run_sdkconfig_check(str(sdk_file_1))
    assert "[PASS] System Panic Mode is Print+Reboot." in res1

    sdk_file_2 = tmp_path / "sdk2"
    sdk_file_2.write_text(
        """
CONFIG_ESP_SYSTEM_PANIC_SILENT_REBOOT=y
CONFIG_BOOTLOADER_LOG_LEVEL=5
CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM="4"
CONFIG_FREERTOS_HZ="100"
CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG=y
    """,
        encoding="utf-8",
    )
    res2 = run_sdkconfig_check(str(sdk_file_2))
    assert "[FAIL] System Panic Mode is SILENT" in res2

    sdk_file_3 = tmp_path / "sdk3"
    sdk_file_3.write_text(
        """
CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM="bad"
    """,
        encoding="utf-8",
    )
    res3 = run_sdkconfig_check(str(sdk_file_3))
    assert "Panic mode not explicitly set" in res3


def test_sdkconfig_check_file_not_found():
    result = run_sdkconfig_check("nonexistent")
    assert "Error: sdkconfig file" in result


def test_mission_generator(tmp_path):
    with patch("os.makedirs"):
        with patch("builtins.open", new_callable=MagicMock):
            result = generate_mission("Wi-Fi Reconnect", "ESP32-S3", "Add robust reconnection.")
            assert "Mission successfully created at" in result


@patch("mcp_local_tools_search_docs._load_resources")
def test_search_docs_not_indexed(mock_load):
    mock_load.return_value = False
    result = search_esp_docs("ADC2")
    assert "Error: ESP32 documentation index not found" in result


@patch("mcp_local_tools_search_docs._load_resources")
@patch("mcp_local_tools_search_docs.encoder")
@patch("mcp_local_tools_search_docs.faiss_index")
@patch(
    "mcp_local_tools_search_docs.meta_data",
    new=[{"source_name": "Test TRM", "page": 5, "url": "http", "tags": [], "text": "ADC2 config"}],
)
def test_search_docs_indexed(mock_faiss, mock_enc, mock_load):
    mock_load.return_value = True
    mock_faiss.search.return_value = ([[0.1]], [[0]])
    mock_enc.encode.return_value = [[0.5, 0.5]]

    result = search_esp_docs("ADC2")
    assert "=== ESP32 Docs Search" in result


@patch("os.path.exists")
def test_load_resources_missing(mock_exists):
    mock_exists.return_value = False
    assert not _load_resources()


@patch("os.path.exists")
@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_load_resources_mocked_success(mock_open, mock_pickle, mock_exists):
    mock_faiss = MagicMock()
    sys.modules["faiss"] = mock_faiss
    mock_st = MagicMock()
    sys.modules["sentence_transformers"] = mock_st

    mock_exists.return_value = True

    search_docs.faiss_index = None
    search_docs.meta_data = None
    search_docs.encoder = None

    res = search_docs._load_resources()
    assert res
    assert search_docs.faiss_index is not None
