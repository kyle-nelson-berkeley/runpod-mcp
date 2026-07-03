"""Unit 5: server.py — exactly the 14 planned tools, importable without a key."""
import asyncio
import sys

from tests.conftest import MCP_ROOT

sys.path.insert(0, str(MCP_ROOT))

EXPECTED_TOOLS = {
    "gpu_availability", "ensure_pod", "pod_status", "run_pod_setup",
    "exec_on_pod", "run_job", "launch_training", "job_status", "sync_logs",
    "apply_bluerov_patches", "axis_sanity_sweep", "stop_pod",
    "terminate_pod", "spend_report",
}


def test_server_registers_exactly_the_14_tools():
    import server                      # import must NOT touch the Keychain
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
    assert len(EXPECTED_TOOLS) == 14


def test_every_tool_has_a_docstring_description():
    import server
    tools = asyncio.run(server.mcp.list_tools())
    for t in tools:
        assert t.description and len(t.description) > 20, t.name
