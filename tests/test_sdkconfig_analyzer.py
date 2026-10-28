"""Tests for the sdkconfig analyzer tool."""

from esp32_workbench.sdkconfig import analyze_config, parse_sdkconfig


class TestParseSDKConfig:
    """Test sdkconfig parsing."""

    def test_parse_key_value(self):
        content = 'CONFIG_IDF_TARGET="esp32"\nCONFIG_FREERTOS_HZ=1000'
        config = parse_sdkconfig(content)
        assert config["CONFIG_IDF_TARGET"] == '"esp32"'
        assert config["CONFIG_FREERTOS_HZ"] == "1000"

    def test_parse_not_set(self):
        content = "# CONFIG_BT_ENABLED is not set"
        config = parse_sdkconfig(content)
        assert config["CONFIG_BT_ENABLED"] == "__NOT_SET__"

    def test_ignore_comments(self):
        content = "# This is a comment\nCONFIG_LOG_DEFAULT_LEVEL=3"
        config = parse_sdkconfig(content)
        assert "CONFIG_LOG_DEFAULT_LEVEL" in config
        assert len([k for k in config if not k.startswith("CONFIG_")]) == 0

    def test_empty_content(self):
        config = parse_sdkconfig("")
        assert len(config) == 0


class TestAnalyzeConfig:
    """Test sdkconfig analysis rules."""

    def test_detect_tls_skip_verify(self):
        config = {"CONFIG_ESP_TLS_SKIP_SERVER_CERT_VERIFY": "y"}
        issues = analyze_config(config)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("TLS" in i.message for i in errors)

    def test_detect_low_tick_rate(self):
        config = {"CONFIG_FREERTOS_HZ": "50"}
        issues = analyze_config(config)
        assert any("tick rate" in i.message.lower() for i in issues)

    def test_detect_small_main_stack(self):
        config = {"CONFIG_ESP_MAIN_TASK_STACK_SIZE": "2048"}
        issues = analyze_config(config)
        assert any("stack" in i.message.lower() for i in issues)

    def test_detect_low_cpu_freq(self):
        config = {"CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ": "80"}
        issues = analyze_config(config)
        assert any("80 MHz" in i.message for i in issues)

    def test_detect_debug_log_level(self):
        config = {"CONFIG_LOG_DEFAULT_LEVEL_DEBUG": "y"}
        issues = analyze_config(config)
        assert any("DEBUG" in i.message for i in issues)

    def test_detect_verbose_log_level(self):
        config = {"CONFIG_LOG_DEFAULT_LEVEL_VERBOSE": "y"}
        issues = analyze_config(config)
        assert any("VERBOSE" in i.message for i in issues)

    def test_detect_wifi_bt_coexistence_missing(self):
        config = {
            "CONFIG_BT_ENABLED": "y",
            "CONFIG_ESP_WIFI_ENABLED": "y",
        }
        issues = analyze_config(config)
        assert any("coexistence" in i.message.lower() for i in issues)

    def test_wifi_bt_coex_ok_when_enabled(self):
        config = {
            "CONFIG_BT_ENABLED": "y",
            "CONFIG_ESP_WIFI_ENABLED": "y",
            "CONFIG_SW_COEXIST_ENABLE": "y",
        }
        issues = analyze_config(config)
        coex_issues = [i for i in issues if "coexistence" in i.message.lower()]
        assert len(coex_issues) == 0

    def test_detect_heap_poisoning_in_prod(self):
        config = {"CONFIG_HEAP_POISONING_COMPREHENSIVE": "y"}
        issues = analyze_config(config)
        assert any("heap poisoning" in i.message.lower() for i in issues)

    def test_clean_config_minimal_issues(self):
        config = {
            "CONFIG_IDF_TARGET": '"esp32"',
            "CONFIG_FREERTOS_HZ": "1000",
            "CONFIG_ESP_MAIN_TASK_STACK_SIZE": "8192",
            "CONFIG_ESP_TASK_WDT_EN": "y",
            "CONFIG_ESP_TASK_WDT_TIMEOUT_S": "10",
            "CONFIG_ESPTOOLPY_FLASHSIZE": '"4MB"',
            "CONFIG_LOG_DEFAULT_LEVEL_INFO": "y",
            "CONFIG_MBEDTLS_CERTIFICATE_BUNDLE": "y",
        }
        issues = analyze_config(config)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 0

    def test_detect_weak_wifi_password(self):
        config = {"CONFIG_ESP_WIFI_PASSWORD": '"password"'}
        issues = analyze_config(config)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("password" in i.message.lower() for i in errors)
