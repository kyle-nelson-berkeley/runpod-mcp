"""Unit: per-vehicle scoping — two pods+volumes (hippocampus, bluerov2).

The FIRST test in this file is the GOLDEN backward-compatibility contract:
`runtime("hippocampus")` must resolve to EXACTLY the flat config the server
had before the vehicles map existed (pod `lts-replication`, volume
`lts-replication`, `logs/pod`, `~/.ssh/known_hosts_runpod`, EU-RO-1 first).
That pod exists and holds every campaign's data — if this test ever goes red,
roll back rather than "fix" it.

Nothing here touches the real Keychain: Runtime's api/ssh clients are lazy,
and every test that would need one injects a fake.
"""
import pytest

from runpod_mcp import config, guardrails, tools


@pytest.fixture(autouse=True)
def _clear_registry():
    """Per-vehicle Runtime registry must not leak between tests (order-independence)."""
    tools._runtimes.clear()
    yield
    tools._runtimes.clear()


# ============================================================ (a) GOLDEN

def test_golden_hippocampus_is_byte_for_byte_the_old_flat_config():
    cfg = tools.runtime("hippocampus").cfg
    assert cfg["pod_name"] == "lts-replication"
    assert cfg["network_volume_name"] == "lts-replication"
    assert cfg["local_log_dir"] == "logs/pod"
    assert cfg["known_hosts_file"] == "~/.ssh/known_hosts_runpod"
    assert cfg["datacenter_preference"][0] == "EU-RO-1"


def test_golden_hippocampus_is_the_default_vehicle():
    """Bare runtime() — grandfathered chain scripts — still means lts-replication."""
    assert tools.runtime().cfg["pod_name"] == "lts-replication"
    assert tools.runtime() is tools.runtime("hippocampus")


def test_golden_hippocampus_keeps_every_shared_top_level_key():
    cfg = tools.runtime("hippocampus").cfg
    assert cfg["cloud_type"] == "SECURE"
    assert cfg["gpu_type_ids"] == ["NVIDIA GeForce RTX 4090"]
    assert cfg["gpu_count"] == 1
    assert cfg["network_volume_gb"] == 60
    assert cfg["volume_mount_path"] == "/workspace"
    assert cfg["ssh_identity"] == "~/.ssh/id_ed25519"
    assert cfg["idle_minutes"] == 60
    assert cfg["timeouts"]["exec_max_sec"] == 600
    assert cfg["min_driver_version"] == "535.129.03"


# ================================================ (b) per-vehicle resolution

def test_bluerov2_resolves_to_its_own_pod_volume_logs_and_known_hosts():
    cfg = tools.runtime("bluerov2").cfg
    assert cfg["pod_name"] == "lts-replication-bluerov2"
    assert cfg["network_volume_name"] == "bluerov2-lts"
    assert cfg["local_log_dir"] == "logs/pod/bluerov2"
    assert cfg["known_hosts_file"] == "~/.ssh/known_hosts_runpod_bluerov2"


def test_known_hosts_files_differ_per_vehicle():
    """ensure_pod truncates known_hosts on EVERY transition-to-running — a
    shared file would let one vehicle's bring-up wipe the other's live keys."""
    assert (tools.runtime("hippocampus").cfg["known_hosts_file"]
            != tools.runtime("bluerov2").cfg["known_hosts_file"])


def test_both_vehicles_share_the_same_datacenter_preference_list():
    assert (tools.runtime("bluerov2").cfg["datacenter_preference"]
            == tools.runtime("hippocampus").cfg["datacenter_preference"])


def test_unknown_vehicle_is_an_actionable_tool_error():
    with pytest.raises(tools.ToolError) as exc:
        tools.runtime("seaglider")
    msg = str(exc.value)
    assert "seaglider" in msg
    assert "hippocampus" in msg and "bluerov2" in msg


# ================================================= (c) assert_only_pod union

HIPPO_POD = {"id": "p1", "name": "lts-replication"}
BLUE_POD = {"id": "p2", "name": "lts-replication-bluerov2"}
FOREIGN = {"id": "px", "name": "someone-elses-pod"}


def _declared():
    return config.declared_pod_names(config.load_defaults())


def test_both_declared_pods_may_coexist():
    guardrails.assert_only_pod([HIPPO_POD, BLUE_POD], _declared())


def test_unknown_pod_is_refused_and_names_itself_and_the_allowed_union():
    with pytest.raises(guardrails.GuardrailError) as exc:
        guardrails.assert_only_pod([HIPPO_POD, FOREIGN], _declared())
    msg = str(exc.value)
    assert "someone-elses-pod" in msg
    assert "lts-replication" in msg and "lts-replication-bluerov2" in msg


def test_first_bring_up_tolerates_the_other_vehicles_running_pod():
    """THE first-bring-up trap: ensure_pod('bluerov2') runs while the
    hippocampus pod is RUNNING and the bluerov2 pod does not exist yet."""
    rt = tools.runtime("bluerov2")
    guardrails.assert_only_pod([dict(HIPPO_POD, desiredStatus="RUNNING")],
                               rt.allowed_pod_names)


def test_ensure_pod_allowed_names_is_the_union_not_the_single_vehicle():
    for vehicle in ("hippocampus", "bluerov2"):
        assert tools.runtime(vehicle).allowed_pod_names == {
            "lts-replication", "lts-replication-bluerov2"}


# ============================================================ (d) registry

def test_runtime_is_cached_per_vehicle():
    assert tools.runtime("hippocampus") is tools.runtime("hippocampus")
    assert tools.runtime("bluerov2") is tools.runtime("bluerov2")


def test_runtimes_are_distinct_objects_per_vehicle():
    assert tools.runtime("hippocampus") is not tools.runtime("bluerov2")


def test_conn_caches_are_independent():
    hippo, blue = tools.runtime("hippocampus"), tools.runtime("bluerov2")
    hippo.conn_cache.put("1.2.3.4", 22)
    assert hippo.conn_cache.get() == ("1.2.3.4", 22)
    assert blue.conn_cache.get() is None


def test_runtime_records_its_vehicle():
    assert tools.runtime("hippocampus").vehicle == "hippocampus"
    assert tools.runtime("bluerov2").vehicle == "bluerov2"


def test_injected_cfg_without_pod_name_fails_actionably_not_with_keyerror():
    with pytest.raises(tools.ToolError) as exc:
        tools.Runtime(cfg={"vehicles": {"hippocampus": {}}})
    msg = str(exc.value)
    assert "merged" in msg.lower()
    assert "merged_vehicle_cfg" in msg


def test_injected_merged_cfg_defaults_allowed_names_to_its_own_pod():
    rt = tools.Runtime(cfg=config.merged_vehicle_cfg(config.load_defaults(),
                                                     "hippocampus"))
    assert rt.allowed_pod_names == {"lts-replication"}


# ================================================ (e) training-vehicle axis

def test_training_vehicle_to_pod_map_is_explicit():
    assert tools.TRAINING_VEHICLE_TO_POD == {"curee": "hippocampus",
                                             "bluerov2": "bluerov2"}


# ============================================ (f) explicit-vehicle stop/terminate

def _no_runtime(monkeypatch):
    def boom(vehicle="hippocampus"):
        pytest.fail("the vehicle=None refusal must happen BEFORE any Runtime "
                    "is built")
    import server
    monkeypatch.setattr(server.tools, "runtime", boom)
    return server


def test_server_stop_pod_refuses_implicit_vehicle(monkeypatch):
    server = _no_runtime(monkeypatch)
    with pytest.raises(tools.ToolError) as exc:
        server.stop_pod()
    msg = str(exc.value)
    assert "hippocampus" in msg and "bluerov2" in msg
    assert "lts-replication" in msg


def test_server_terminate_pod_refuses_implicit_vehicle(monkeypatch):
    server = _no_runtime(monkeypatch)
    with pytest.raises(tools.ToolError) as exc:
        server.terminate_pod(confirm="terminate lts-replication")
    msg = str(exc.value)
    assert "hippocampus" in msg and "bluerov2" in msg
    assert "lts-replication" in msg


def test_server_stop_pod_routes_an_explicit_vehicle(monkeypatch):
    import server
    seen = {}
    monkeypatch.setattr(server.tools, "runtime", lambda v: seen.setdefault("v", v))
    monkeypatch.setattr(server.tools, "stop_pod",
                        lambda rt, force=False: {"status": "stopped", "rt": rt})
    out = server.stop_pod(vehicle="bluerov2")
    assert seen["v"] == "bluerov2"
    assert out["status"] == "stopped"


def test_tools_stop_and_terminate_signatures_are_unchanged(monkeypatch):
    """The vehicle check lives in the WRAPPER — supervise/deadman keep calling
    tools.stop_pod(rt, force) / tools.terminate_pod(rt, confirm)."""
    import inspect
    assert list(inspect.signature(tools.stop_pod).parameters) == ["rt", "force"]
    assert list(inspect.signature(tools.terminate_pod).parameters) == ["rt",
                                                                      "confirm"]


# ------------------------------------------------- server wrapper routing

def _route_spy(monkeypatch):
    import server
    seen = []
    monkeypatch.setattr(server.tools, "runtime",
                        lambda vehicle="hippocampus": seen.append(vehicle) or vehicle)
    return server, seen


def test_server_wrappers_default_to_hippocampus(monkeypatch):
    server, seen = _route_spy(monkeypatch)
    for name, call in [
        ("gpu_availability", lambda: server.gpu_availability()),
        ("pod_status", lambda: server.pod_status()),
        ("job_status", lambda: server.job_status("j1")),
        ("sync_logs", lambda: server.sync_logs()),
        ("spend_report", lambda: server.spend_report()),
    ]:
        monkeypatch.setattr(server.tools, name, lambda *a, **k: {})
        call()
    assert seen == ["hippocampus"] * 5


def test_server_wrappers_route_an_explicit_vehicle(monkeypatch):
    server, seen = _route_spy(monkeypatch)
    monkeypatch.setattr(server.tools, "pod_status", lambda rt: {"rt": rt})
    monkeypatch.setattr(server.tools, "sync_logs",
                        lambda rt, subdir="x": {"rt": rt})
    server.pod_status(vehicle="bluerov2")
    server.sync_logs(vehicle="bluerov2")
    assert seen == ["bluerov2", "bluerov2"]


def test_launch_training_derives_the_pod_from_the_training_vehicle(monkeypatch):
    server, seen = _route_spy(monkeypatch)
    monkeypatch.setattr(server.tools, "launch_training",
                        lambda rt, vehicle, dr_level, **kw: {"pod_rt": rt})
    server.launch_training("curee", "DR_2", dry_run=True)
    server.launch_training("bluerov2", "DR_2", dry_run=True)
    assert seen == ["hippocampus", "bluerov2"]


def test_launch_training_unknown_vehicle_falls_through_to_the_tool_error(monkeypatch):
    """An unknown training vehicle must not blow up in the ROUTER — it has to
    reach launch_training's own actionable 'use curee|bluerov2' error."""
    server, seen = _route_spy(monkeypatch)
    with pytest.raises(tools.ToolError, match="curee"):
        server.launch_training("seaglider", "DR_2", dry_run=True)
    assert seen == ["hippocampus"]


def test_bluerov2_only_tools_are_hardwired(monkeypatch):
    server, seen = _route_spy(monkeypatch)
    monkeypatch.setattr(server.tools, "apply_bluerov_patches",
                        lambda rt, dry_run=False, force=False: {})
    monkeypatch.setattr(server.tools, "axis_sanity_sweep",
                        lambda rt, auto_stop=False: {})
    server.apply_bluerov_patches(dry_run=True)
    server.axis_sanity_sweep()
    assert seen == ["bluerov2", "bluerov2"]


def test_pod_touching_wrappers_expose_a_vehicle_param():
    import inspect
    import server
    for name in ("gpu_availability", "ensure_pod", "pod_status",
                 "run_pod_setup", "exec_on_pod", "run_job", "job_status",
                 "sync_logs"):
        params = inspect.signature(getattr(server, name)).parameters
        assert params["vehicle"].default == "hippocampus", name
    for name in ("stop_pod", "terminate_pod"):
        assert inspect.signature(getattr(server, name)).parameters[
            "vehicle"].default is None, name
    # account-wide / hardwired tools deliberately expose no vehicle knob
    for name in ("spend_report", "apply_bluerov_patches", "axis_sanity_sweep"):
        assert "vehicle" not in inspect.signature(getattr(server, name)).parameters


def test_spend_report_is_labelled_account_wide(monkeypatch):
    server, _ = _route_spy(monkeypatch)
    monkeypatch.setattr(server.tools, "spend_report",
                        lambda rt: {"scope": "account-wide (all vehicles)"})
    assert server.spend_report()["scope"] == "account-wide (all vehicles)"


# ========================================== (g) no-GPU recovery names the pod

def test_no_gpu_recovery_names_the_resolved_pod_for_bluerov2():
    import json
    recipe = json.dumps(tools._no_gpu_recovery(tools.runtime("bluerov2").cfg))
    assert "terminate lts-replication-bluerov2" in recipe
    assert "terminate lts-replication'" not in recipe     # not the OTHER pod


def test_no_gpu_recovery_still_names_lts_replication_for_hippocampus():
    import json
    recipe = json.dumps(tools._no_gpu_recovery(tools.runtime("hippocampus").cfg))
    assert "terminate lts-replication" in recipe
    assert "lts-replication-bluerov2" not in recipe


# ======================================= (h) volumes stay by-NAME, never by id

def test_no_vehicle_declares_a_volume_id():
    """Volume resolution is by NAME with auto-create (ensure_pod creates the
    volume if absent) — a pinned id would break first bring-up."""
    raw = config.load_defaults()
    for vehicle, sub in raw["vehicles"].items():
        assert "network_volume_name" in sub, vehicle
        assert not [k for k in sub if k.endswith("_id") or k.endswith("Id")], vehicle
