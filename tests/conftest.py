"""Pytest configuration for the runpod-mcp suite.

Makes the runpod_mcp package importable and exposes repo-root paths so tests
can cross-check the code against the source-of-truth files (RUNBOOK.md,
config/bluerov2_heavy.yaml, patches/APPLY.md).
"""
import os
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]   # runpod-mcp/
REPO_ROOT = MCP_ROOT.parent                      # learning-to-swim-replication/

sys.path.insert(0, str(MCP_ROOT))


def merged_cfg(vehicle: str = "hippocampus") -> dict:
    """The flat per-vehicle config a Runtime expects when a test injects cfg.

    tools.Runtime(cfg=...) takes a MERGED view (config.merged_vehicle_cfg),
    not the raw pod_defaults.yaml — raw has no top-level pod_name. Fixture
    builders across the suite go through this helper. Default is hippocampus:
    the pre-existing lts-replication pod, so migrated fixtures keep their
    original values.
    """
    from runpod_mcp import config as _config          # import-time-safe (no key)
    return _config.merged_vehicle_cfg(_config.load_defaults(), vehicle)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: needs the real Keychain key / network; deselected by default"
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless explicitly requested with -m live or RUNPOD_MCP_LIVE=1."""
    if config.getoption("-m") == "live" or os.environ.get("RUNPOD_MCP_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test (run with -m live or RUNPOD_MCP_LIVE=1)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def mcp_root():
    return MCP_ROOT
