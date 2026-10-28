"""Tests for the zero-dependency MCP stdio server."""

import asyncio
import io
import json
import sys
from pathlib import Path

import pytest

from esp32_workbench import mcp_server
from esp32_workbench.mcp_server import TOOLS, handle, serve


def call(name, **arguments):
    response = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return response["result"]


def test_initialize_negotiates_version():
    r = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert r["result"]["protocolVersion"] == "2025-06-18"
    assert r["result"]["serverInfo"]["name"] == "esp32-workbench"
    r = handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        }
    )
    assert r["result"]["protocolVersion"] == mcp_server.SUPPORTED_PROTOCOLS[0]


def test_notifications_get_no_reply():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_schemas():
    tools = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"pin_audit", "chip_info", "crash_decode", "project_doctor", "search_esp_docs"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def test_unknown_method_and_tool():
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601
    r = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope"}})
    assert r["error"]["code"] == -32602


def test_tool_errors_are_reported_as_results(tmp_path: Path):
    result = call("sdkconfig_review", path=str(tmp_path / "missing"))
    assert result["isError"] is True


def test_pin_audit_and_chip_info(tmp_path: Path):
    src = tmp_path / "main.c"
    src.write_text(
        "#define LED_PIN GPIO_NUM_34\ngpio_set_direction(GPIO_NUM_34, GPIO_MODE_OUTPUT);\n"
    )
    text = call("pin_audit", path=str(src), target="esp32")["content"][0]["text"]
    assert "input-only" in text
    assert "ESP32-C6" in call("chip_info", target="esp32c6")["content"][0]["text"]


def test_crash_and_build_tools():
    text = call("crash_decode", text="assert failed: foo main.c:1 (x)")["content"][0]["text"]
    assert "assert" in text
    text = call("build_log_triage", text="fatal error: nvs_flash.h: No such file or directory")
    assert "nvs_flash" in text["content"][0]["text"]


def test_mission_round_trip(tmp_path: Path):
    text = call(
        "create_mission",
        feature_name="Wi-Fi reconnect",
        description="Retry",
        project_root=str(tmp_path),
    )
    path = next((tmp_path / "missions").glob("*.md"))
    assert str(path) in text["content"][0]["text"]
    report = json.loads(call("validate_mission", path=str(path))["content"][0]["text"])
    assert report["missing_required"] == []


def test_search_docs_without_index(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ESP32WB_CACHE", str(tmp_path))
    assert "docs index" in call("search_esp_docs", query="ADC2")["content"][0]["text"]


def test_serve_loop_handles_bad_json():
    stdin = io.StringIO('not json\n\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n')
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_every_tool_has_description():
    assert all(len(t.description) > 30 for t in TOOLS)


def test_official_mcp_client_interop():
    """Drive the server with the official MCP Python SDK client over real stdio."""
    pytest.importorskip("mcp.client.stdio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    def field(obj, snake, camel):
        """SDK releases differ between camelCase and snake_case attribute names."""
        return getattr(obj, snake) if hasattr(obj, snake) else getattr(obj, camel)

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "esp32_workbench.mcp_server"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert field(init, "server_info", "serverInfo").name == "esp32-workbench"
                tools = await session.list_tools()
                assert len(tools.tools) == len(TOOLS)
                result = await session.call_tool("chip_info", {"target": "esp32s3"})
                assert not field(result, "is_error", "isError")
                assert "ESP32-S3" in result.content[0].text

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))
