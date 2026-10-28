"""Tests for implementation contract generation."""

from esp32_workbench.contract import generate_contract


class TestGenerateContract:
    """Test contract generation with various inputs."""

    def test_basic_contract_generation(self):
        contract = generate_contract(summary="Add Wi-Fi reconnection handler.")
        assert "# Implementation Contract" in contract
        assert "Add Wi-Fi reconnection handler." in contract
        assert "## Change Summary" in contract
        assert "## Non-Goals" in contract
        assert "## Affected Files" in contract
        assert "## Risk Assessment" in contract
        assert "## Test Plan" in contract
        assert "## Rollback Strategy" in contract
        assert "## Acceptance Criteria" in contract
        assert "## Approval" in contract

    def test_contract_with_non_goals(self):
        contract = generate_contract(
            summary="Test change.",
            non_goals=["No OTA support", "No BLE integration"],
        )
        assert "No OTA support" in contract
        assert "No BLE integration" in contract

    def test_contract_with_files(self):
        contract = generate_contract(
            summary="Test change.",
            files=["main/wifi.c|MODIFY|Update reconnect logic"],
        )
        assert "main/wifi.c" in contract
        assert "MODIFY" in contract
        assert "Update reconnect logic" in contract

    def test_contract_with_risks(self):
        contract = generate_contract(
            summary="Test change.",
            risks=["Race condition on shared flag"],
        )
        assert "Race condition on shared flag" in contract

    def test_contract_with_tests(self):
        contract = generate_contract(
            summary="Test change.",
            tests=["Verify backoff calculation", "Test retry counter reset"],
        )
        assert "Verify backoff calculation" in contract
        assert "Test retry counter reset" in contract

    def test_contract_with_custom_rollback(self):
        contract = generate_contract(
            summary="Test change.",
            rollback="Revert commit abc123 and erase NVS partition.",
        )
        assert "Revert commit abc123 and erase NVS partition." in contract

    def test_contract_has_date(self):
        contract = generate_contract(summary="Test change.")
        assert "**Date**:" in contract

    def test_contract_default_rollback(self):
        contract = generate_contract(summary="Test change.")
        assert "Revert the commit." in contract

    def test_contract_with_all_parameters(self):
        contract = generate_contract(
            summary="Full featured change.",
            non_goals=["Not changing UART"],
            files=["main/app.c|MODIFY|Add init call"],
            risks=["Stack overflow if buffer too large"],
            tests=["Unit test for buffer allocation"],
            rollback="Revert and flash previous firmware.",
        )
        assert "Full featured change." in contract
        assert "Not changing UART" in contract
        assert "main/app.c" in contract
        assert "Stack overflow" in contract
        assert "Unit test for buffer" in contract
        assert "flash previous firmware" in contract

    def test_empty_lists_produce_placeholders(self):
        contract = generate_contract(summary="Test change.")
        assert "[To be defined]" in contract or "[To be assessed]" in contract
