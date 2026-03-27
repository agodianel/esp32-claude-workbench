from mcp.server.fastmcp import FastMCP
from tools.mission_generator import generate_mission
from tools.pin_audit import run_pin_audit
from tools.sdkconfig_check import run_sdkconfig_check
from tools.search_docs import search_esp_docs as _search_esp_docs

mcp = FastMCP("esp32-workbench")


@mcp.tool()
def pin_audit(file_path: str) -> str:
    """Scan a C/H file for ESP32 GPIO conflicts, input-only misuse, and bootstrap pin risks."""
    return run_pin_audit(file_path)


@mcp.tool()
def sdkconfig_review(sdkconfig_path: str) -> str:
    """Check sdkconfig for production readiness. Run this before any release."""
    return run_sdkconfig_check(sdkconfig_path)


@mcp.tool()
def create_mission(feature_name: str, board: str, description: str) -> str:
    """Create a structured mission markdown file for a new firmware task."""
    return generate_mission(feature_name, board, description)


@mcp.tool()
def search_esp_docs(query: str, top_k: int = 5) -> str:
    """Search ESP32 TRMs and datasheets semantically. Use before writing peripheral code."""
    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 5

    return _search_esp_docs(query, top_k)


if __name__ == "__main__":
    mcp.run()
