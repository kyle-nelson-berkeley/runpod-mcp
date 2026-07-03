"""Live ($0, read-only) verification — needs the real Keychain key + network.

Deselected by default; run explicitly with:
    RUNPOD_MCP_LIVE=1 runpod-mcp/.venv/bin/python -m pytest runpod-mcp/tests/test_live.py -q

NOTHING here mutates the account: GETs + unauthenticated GraphQL + a stdio
handshake against the locally-launched server. The unit-7 done-condition is
that these tests were SELECTED and passed (a skip is not verification).
"""
import asyncio
import subprocess

import pytest

from tests.conftest import MCP_ROOT

pytestmark = pytest.mark.live


def test_keychain_key_present_and_shaped():
    from runpod_mcp import config
    key = config.fetch_api_key(run=subprocess.run)
    assert key.startswith("rpa_") and len(key) > 20


def test_rest_api_read_only_smoke():
    from runpod_mcp import api, config
    client = api.RunPodClient(config.fetch_api_key())
    pods = client.list_pods()
    assert isinstance(pods, list)
    for pod in pods:                       # shape assumptions ensure_pod relies on
        assert {"id", "name", "desiredStatus"} <= set(pod)
    vols = client.list_network_volumes()
    assert isinstance(vols, list)
    billing = client.billing_pods()
    assert billing is None or isinstance(billing, list)
    # volume-storage billing endpoint existed on 2026-07-03 (spend_report
    # coverage label depends on this returning list OR None, never raising)
    vol_billing = client.billing_network_volumes()
    assert vol_billing is None or isinstance(vol_billing, list)


def test_graphql_gpu_types_unauthenticated():
    from runpod_mcp import api
    types = api.gpu_types()
    assert types and types[0]["id"] == "NVIDIA GeForce RTX 4090"
    lp = types[0]["lowestPrice"]
    assert "stockStatus" in lp and "uninterruptablePrice" in lp


def test_stdio_handshake_via_run_sh():
    """End-to-end transport check: run.sh venv bootstrap -> server startup
    (real Keychain fetch) -> MCP initialize -> tools/list == the 14 tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tests.test_server import EXPECTED_TOOLS

    async def handshake():
        params = StdioServerParameters(
            command="bash", args=[str(MCP_ROOT / "run.sh")])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                info = await session.initialize()
                tools = await session.list_tools()
                return info.serverInfo.name, {t.name for t in tools.tools}

    name, names = asyncio.run(asyncio.wait_for(handshake(), timeout=60))
    assert name == "runpod"
    assert names == EXPECTED_TOOLS
