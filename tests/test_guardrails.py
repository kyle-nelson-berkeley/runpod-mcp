"""Unit 2: guardrails.py — the cost/safety rules are code, not prose."""
import pytest

from runpod_mcp import guardrails as g


POD = {"id": "p1", "name": "lts-replication"}
OTHER = {"id": "px", "name": "someone-elses-pod"}


# ------------------------------------------------------------- one-pod max

def test_only_pod_ok_with_empty_account():
    g.assert_only_pod([], "lts-replication")


def test_only_pod_ok_with_our_pod():
    g.assert_only_pod([POD], "lts-replication")


def test_only_pod_refuses_foreign_pod():
    with pytest.raises(g.GuardrailError, match="someone-elses-pod"):
        g.assert_only_pod([POD, OTHER], "lts-replication")


# --------------------------------------------------------- payload enforcer

def good_payload():
    return {
        "name": "lts-replication",
        "cloudType": "SECURE",
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuCount": 1,
        "interruptible": False,
        "networkVolumeId": "vol1",
        "volumeMountPath": "/workspace",
    }


def test_enforce_accepts_good_payload():
    out = g.enforce_pod_payload(good_payload())
    assert out["interruptible"] is False


def test_enforce_rejects_wrong_gpu():
    p = good_payload()
    p["gpuTypeIds"] = ["NVIDIA A100 80GB PCIe"]
    with pytest.raises(g.GuardrailError, match="4090"):
        g.enforce_pod_payload(p)


def test_enforce_rejects_multi_gpu():
    p = good_payload()
    p["gpuCount"] = 2
    with pytest.raises(g.GuardrailError):
        g.enforce_pod_payload(p)


def test_enforce_forces_interruptible_false():
    p = good_payload()
    p["interruptible"] = True          # someone tries spot to save pennies
    out = g.enforce_pod_payload(p)
    assert out["interruptible"] is False


def test_enforce_requires_network_volume():
    p = good_payload()
    p.pop("networkVolumeId")
    with pytest.raises(g.GuardrailError, match="network volume"):
        g.enforce_pod_payload(p)
    p["networkVolumeId"] = ""
    with pytest.raises(g.GuardrailError, match="network volume"):
        g.enforce_pod_payload(p)


def test_enforce_rejects_community_cloud():
    p = good_payload()
    p["cloudType"] = "COMMUNITY"
    with pytest.raises(g.GuardrailError, match="SECURE"):
        g.enforce_pod_payload(p)


# --------------------------------------------------------- terminate gate

def test_terminate_requires_verbatim_confirm():
    g.assert_terminate_confirm("terminate lts-replication", "lts-replication")
    for bad in ["", "terminate", "yes", "terminate lts_replication",
                "Terminate lts-replication", " terminate lts-replication"]:
        with pytest.raises(g.GuardrailError, match="terminate lts-replication"):
            g.assert_terminate_confirm(bad, "lts-replication")


# ------------------------------------------------------- concurrency guard

def test_concurrency_guard_refuses_live_jobs():
    with pytest.raises(g.GuardrailError, match="20260703_setup"):
        g.assert_no_live_jobs(["20260703_setup"], force=False)


def test_concurrency_guard_force_overrides():
    g.assert_no_live_jobs(["20260703_setup"], force=True)


def test_concurrency_guard_ok_when_idle():
    g.assert_no_live_jobs([], force=False)


# ------------------------------------------------------------ exec ceiling

def test_exec_timeout_ceiling():
    assert g.clamp_exec_timeout(30, 600) == 30
    assert g.clamp_exec_timeout(9999, 600) == 600
    assert g.clamp_exec_timeout(0, 600) == 1
