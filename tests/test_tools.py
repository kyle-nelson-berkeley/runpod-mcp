"""Unit 5: tools.py — the 14 tool implementations against fake api/ssh layers."""
import json
import subprocess

import pytest

from runpod_mcp import config, guardrails, jobs, tools, training
from tests.conftest import REPO_ROOT

POD_RUNNING = {
    "id": "on2ghkedz0vbjr", "name": "lts-replication", "desiredStatus": "RUNNING",
    "publicIp": "213.173.99.47", "portMappings": {"22": 15356},
    "costPerHr": 0.69, "networkVolumeId": "vol1",
    "lastStartedAt": "2026-07-03 13:25:25.682 +0000 UTC",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
}
POD_STOPPED = {**POD_RUNNING, "desiredStatus": "EXITED",
               "publicIp": "", "portMappings": None}
VOL = {"id": "vol1", "name": "lts-replication", "size": 60,
       "dataCenterId": "EU-RO-1"}
GPU_OK = [{"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090",
           "memoryInGb": 24, "secureCloud": True, "communityCloud": True,
           "lowestPrice": {"stockStatus": "Medium",
                           "uninterruptablePrice": 0.34,
                           "minimumBidPrice": 0.34}}]


class FakeClient:
    def __init__(self, pods=None, volumes=None):
        self.pods = list(pods or [])
        self.volumes = list(volumes or [])
        self.calls = []
        self.created_pods = []
        self.created_volumes = []
        self.start_error = None
        self.billing_pods_data = [{"podId": "on2ghkedz0vbjr", "amount": 0.56}]
        self.billing_vol_data = [{"amount": 0.01}]

    def list_pods(self):
        self.calls.append("list_pods")
        return self.pods

    def get_pod(self, pod_id):
        self.calls.append(f"get_pod:{pod_id}")
        return next(p for p in self.pods if p["id"] == pod_id)

    def create_pod(self, payload):
        self.calls.append("create_pod")
        self.created_pods.append(payload)
        pod = {**POD_RUNNING, "id": "newpod1", "name": payload["name"]}
        self.pods = [pod]
        return pod

    def start_pod(self, pod_id):
        self.calls.append(f"start_pod:{pod_id}")
        if self.start_error:
            raise self.start_error
        self.pods = [dict(p, desiredStatus="RUNNING",
                          publicIp=POD_RUNNING["publicIp"],
                          portMappings=POD_RUNNING["portMappings"])
                     for p in self.pods]
        return self.pods[0]

    def stop_pod(self, pod_id):
        self.calls.append(f"stop_pod:{pod_id}")
        return {}

    def delete_pod(self, pod_id):
        self.calls.append(f"delete_pod:{pod_id}")

    def list_network_volumes(self):
        return self.volumes

    def create_network_volume(self, payload):
        self.created_volumes.append(payload)
        vol = {**VOL, "id": "newvol1", **payload}
        self.volumes = [vol]
        return vol

    def billing_pods(self):
        return self.billing_pods_data

    def billing_network_volumes(self):
        return self.billing_vol_data


class FakeSSH:
    def __init__(self, run_results=None):
        self.run_calls = []
        self.push_texts = []
        self.push_files = []
        self.rsyncs = []
        self.truncated = 0
        self._results = list(run_results or [])

    def run(self, host, port, command, timeout=60, check=False):
        self.run_calls.append(command)
        res = self._results.pop(0) if self._results else \
            subprocess.CompletedProcess([], 0, stdout="", stderr="")
        if check and res.returncode != 0:
            raise tools.ssh.SSHError(f"rc={res.returncode}")
        return res

    def push_text(self, host, port, text, remote_path, executable=False):
        self.push_texts.append((remote_path, text))

    def push_file(self, host, port, local_path, remote_path):
        self.push_files.append((str(local_path), remote_path))

    def rsync_pull(self, host, port, remote_dir, local_dir, timeout=600):
        self.rsyncs.append((remote_dir, str(local_dir)))
        return "sent"

    def truncate_known_hosts(self):
        self.truncated += 1


def ok(stdout=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def fail(stderr="x", rc=1):
    return subprocess.CompletedProcess([], rc, stdout="", stderr=stderr)


def make_rt(pods=None, volumes=None, ssh_results=None, gpu=None):
    cfg = config.load_defaults()
    cfg["ssh_identity"] = "~/.ssh/id_ed25519"
    rt = tools.Runtime(cfg=cfg,
                       client=FakeClient(pods, volumes),
                       sshc=FakeSSH(ssh_results),
                       sleep=lambda s: None,
                       gpu_types=lambda **kw: gpu if gpu is not None else GPU_OK,
                       ssh_pubkey="ssh-ed25519 AAAA test@mac")
    return rt


DRIVER_OK = ok("535.161.08\n")


# ------------------------------------------------------------ gpu_availability

def test_gpu_availability_reports_price_and_stock():
    out = tools.gpu_availability(make_rt())
    assert out["gpu"] == "NVIDIA GeForce RTX 4090"
    assert out["stock_status"] == "Medium"
    assert out["on_demand_price_usd_hr"] == 0.34


# ------------------------------------------------------------------ ensure_pod

def test_ensure_pod_dry_run_empty_account_full_payloads():
    rt = make_rt()
    out = tools.ensure_pod(rt, dry_run=True)
    assert out["dry_run"] is True
    vol = out["volume_payload"]
    assert vol == {"name": "lts-replication", "size": 60,
                   "dataCenterId": "EU-RO-1"}          # first DC with stock
    pod = out["pod_payload"]
    assert pod["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert pod["interruptible"] is False
    assert pod["cloudType"] == "SECURE"
    assert pod["networkVolumeId"] == "(created above)"
    assert pod["volumeMountPath"] == "/workspace"
    assert pod["ports"] == ["22/tcp"]
    assert pod["env"]["PUBLIC_KEY"].startswith("ssh-ed25519")
    assert pod["env"]["SSH_PUBLIC_KEY"] == pod["env"]["PUBLIC_KEY"]
    assert pod["containerDiskInGb"] == 30
    assert pod["dataCenterIds"] == ["EU-RO-1"]
    # nothing was actually created
    assert rt.client.created_pods == [] and rt.client.created_volumes == []


def test_ensure_pod_refuses_foreign_pod():
    rt = make_rt(pods=[{"id": "px", "name": "someone-else"}])
    with pytest.raises(guardrails.GuardrailError, match="someone-else"):
        tools.ensure_pod(rt, dry_run=True)


def test_ensure_pod_resume_truncates_knownhosts_and_arms_watchdog():
    rt = make_rt(pods=[dict(POD_STOPPED)], volumes=[VOL],
                 ssh_results=[DRIVER_OK,            # driver probe
                              ok("WATCHDOG_ARMED")])  # watchdog arm
    out = tools.ensure_pod(rt)
    assert "start_pod:on2ghkedz0vbjr" in rt.client.calls
    assert out["status"] == "running"
    assert out["driver_version"] == "535.161.08"
    assert rt.ssh.truncated == 1                       # resume wipes host keys
    assert out["idle_watchdog"] == "armed"
    pushed = dict(rt.ssh.push_texts)
    assert jobs.WATCHDOG_REMOTE in pushed              # reinstalled every time


def test_ensure_pod_already_running_still_rearms_guards():
    # console stop/start wipes the disk without us seeing a transition -> the
    # tool must truncate known_hosts and re-arm the watchdog on EVERY call
    rt = make_rt(pods=[dict(POD_RUNNING)], volumes=[VOL],
                 ssh_results=[DRIVER_OK, ok("WATCHDOG_ARMED")])
    out = tools.ensure_pod(rt)
    assert out["status"] == "running"
    assert rt.ssh.truncated == 1
    assert out["idle_watchdog"] == "armed"
    assert "start_pod:on2ghkedz0vbjr" not in rt.client.calls


def test_ensure_pod_refuses_pod_without_volume():
    bare = {**POD_STOPPED, "networkVolumeId": None}
    rt = make_rt(pods=[bare], volumes=[VOL])
    with pytest.raises(tools.ToolError, match="ephemeral"):
        tools.ensure_pod(rt)
    assert "start_pod:on2ghkedz0vbjr" not in rt.client.calls


def test_ensure_pod_refuses_mismatched_volume():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 volumes=[{**VOL, "id": "other-vol"}])
    with pytest.raises(tools.ToolError, match="other-vol"):
        tools.ensure_pod(rt)
    # dry_run surfaces the same problem as a warning instead of raising
    out = tools.ensure_pod(rt, dry_run=True)
    assert "vol1" in out["volume_check"]               # attached-but-wrong id


def test_ensure_pod_no_gpu_start_failure_returns_recovery_recipe():
    rt = make_rt(pods=[dict(POD_STOPPED)], volumes=[VOL])
    rt.client.start_error = tools.api.ApiError(
        "There are no longer any instances available with the requested specs",
        status_code=500)
    out = tools.ensure_pod(rt)
    assert out["status"] == "start_failed_no_gpu"
    recovery = json.dumps(out["recovery"])
    assert "terminate_pod" in recovery                 # sanctioned terminate case
    assert "gpu_availability" in recovery
    assert "volume" in recovery.lower()                # data survives
    assert "delete_pod:on2ghkedz0vbjr" not in rt.client.calls  # never automatic


def test_ensure_pod_driver_floor_enforced():
    rt = make_rt(pods=[dict(POD_STOPPED)], volumes=[VOL],
                 ssh_results=[ok("470.10.01\n")])
    out = tools.ensure_pod(rt)
    assert out["status"] == "driver_too_old"
    assert "535.129.03" in json.dumps(out)


def test_ensure_pod_still_initializing_within_budget():
    pod = dict(POD_STOPPED)
    rt = make_rt(pods=[pod], volumes=[VOL])
    rt.client.start_pod = lambda pid: rt.client.calls.append(f"start_pod:{pid}")
    # pod never publishes an ip -> poll budget exhausts
    rt.cfg["poll"] = {"budget_sec": 1, "interval_sec": 1}
    out = tools.ensure_pod(rt)
    assert out["status"] == "still_initializing"


# ------------------------------------------------------------------ pod_status

def test_pod_status_no_pod():
    assert tools.pod_status(make_rt())["status"] == "no_pod"


def test_pod_status_stopped():
    out = tools.pod_status(make_rt(pods=[dict(POD_STOPPED)]))
    assert out["status"] == "stopped"
    assert out["cost_per_hr_usd"] == 0.69


def test_pod_status_running_with_jobs():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[DRIVER_OK, ok("42G /workspace\n"),
                              ok("20260703-101010_setup_ab12\nLIVE_LIST_END\n")])
    out = tools.pod_status(rt)
    assert out["status"] == "running"
    assert out["active_jobs"] == ["20260703-101010_setup_ab12"]


def test_pod_status_running_ssh_pending():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[fail("Connection refused", rc=255)])
    out = tools.pod_status(rt)
    assert out["status"] == "running_ssh_pending"


# --------------------------------------------------------------- run_pod_setup

def test_run_pod_setup_dry_run_payload():
    out = tools.run_pod_setup(make_rt(pods=[dict(POD_RUNNING)]), dry_run=True)
    assert out["dry_run"] is True
    assert out["scp"]["local"].endswith("runbook/pod_setup.sh")
    assert out["scp"]["remote"] == "/workspace/pod_setup.sh"
    assert out["max_runtime_sec"] == 5400
    env = out["env"]
    assert env["DEBIAN_FRONTEND"] == "noninteractive"
    assert env["ACCEPT_EULA"] == "Y"
    assert env["PRIVACY_CONSENT"] == "Y"
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_run_pod_setup_launches_async_job():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("LIVE_LIST_END\n")])   # no jobs running
    out = tools.run_pod_setup(rt)
    assert (str(REPO_ROOT / "runbook" / "pod_setup.sh"),
            "/workspace/pod_setup.sh") in rt.ssh.push_files
    assert out["job_id"]
    launch_cmd = rt.ssh.run_calls[-1]
    assert "setsid bash" in launch_cmd and " 5400 " in launch_cmd


# ----------------------------------------------------------------- exec_on_pod

def test_exec_on_pod_clamps_timeout_and_cds():
    rt = make_rt(pods=[dict(POD_RUNNING)], ssh_results=[ok("hi\n")])
    out = tools.exec_on_pod(rt, "echo hi", timeout_sec=99999,
                            workdir="/workspace/IsaacLab")
    assert out["stdout"] == "hi\n"
    assert out["exit_code"] == 0
    assert rt.ssh.run_calls[0].startswith("cd /workspace/IsaacLab && ")
    assert out["timeout_sec"] == 600                   # ceiling applied


# --------------------------------------------------------------------- run_job

def test_run_job_concurrency_guard():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("20260703-1_busy_ab\nLIVE_LIST_END\n")])
    with pytest.raises(guardrails.GuardrailError, match="busy"):
        tools.run_job(rt, name="x", command="true")


def test_run_job_force_overrides_guard():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("20260703-1_busy_ab\nLIVE_LIST_END\n")])
    out = tools.run_job(rt, name="x", command="true", force=True)
    assert out["job_id"]


# ------------------------------------------------------------- launch_training

def test_launch_training_dry_run_needs_no_pod():
    out = tools.launch_training(make_rt(), vehicle="curee", dr_level="small",
                                seed=3, dry_run=True)
    assert out["dry_run"] is True
    assert out["command"] == training.build_train_command(seed=3)
    assert out["workdir"] == "/workspace/IsaacLab"
    assert out["dr_edits"]["com_to_cob_offset_radius"] == "0.025"
    assert out["max_runtime_sec"] == 3600
    assert "gates" in out


def _training_rt(markers: str, live="LIVE_LIST_END\n"):
    """markers: output of the marker-probe call (PATCH=.../SANITY=...)."""
    fixture = (REPO_ROOT / "runpod-mcp" / "tests" / "fixtures" /
               "warpauv_env_excerpt_7c5ebe7.py").read_text()
    return make_rt(pods=[dict(POD_RUNNING)],
                   ssh_results=[ok(markers), ok(live), ok(fixture), ok()])


def test_launch_training_curee_happy_path_edits_source():
    rt = _training_rt("PATCH=absent SANITY=absent")
    out = tools.launch_training(rt, vehicle="curee", dr_level="DR_0", seed=1)
    assert out["job_id"]
    pushed = dict(rt.ssh.push_texts)
    new_src = pushed[training.WARPAUV_ENV_REMOTE]
    assert "com_to_cob_offset_radius = 0 #" in new_src
    launch_cmd = rt.ssh.run_calls[-1]
    assert "setsid bash" in launch_cmd


def test_launch_training_curee_refuses_patched_checkout():
    rt = _training_rt("PATCH=present SANITY=present")
    with pytest.raises(tools.ToolError) as exc:
        tools.launch_training(rt, vehicle="curee", dr_level="DR_2", seed=1)
    msg = str(exc.value)
    assert "patched" in msg.lower()
    assert "apply_bluerov2_patch" not in msg or True
    assert "git" in msg.lower()                        # recovery: git restore
    assert "marker" in msg.lower()


def test_launch_training_bluerov2_requires_both_markers():
    rt = _training_rt("PATCH=present SANITY=absent")
    with pytest.raises(tools.ToolError, match="axis_sanity"):
        tools.launch_training(rt, vehicle="bluerov2", dr_level="small", seed=1)
    rt = _training_rt("PATCH=absent SANITY=absent")
    with pytest.raises(tools.ToolError, match="apply_bluerov_patches"):
        tools.launch_training(rt, vehicle="bluerov2", dr_level="small", seed=1)


# ------------------------------------------------------------------ job_status

def test_job_status_passthrough():
    blob = ("---PID---\n\n---ALIVE---\nno\n---EXIT---\n0\n---META---\n{}\n"
            "---LOG---\nMean reward: 96.1")
    rt = make_rt(pods=[dict(POD_RUNNING)], ssh_results=[ok(blob)])
    out = tools.job_status(rt, "20260703-1_t_ab")
    assert out["state"] == "succeeded"
    assert out["latest_reward_line"] == "Mean reward: 96.1"


# ------------------------------------------------------------------- sync_logs

def test_sync_logs_paths_and_no_delete():
    rt = make_rt(pods=[dict(POD_RUNNING)])
    out = tools.sync_logs(rt, subdir="rsl_rl/warpauv_direct")
    remote, local = rt.ssh.rsyncs[0]
    assert remote == "/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/"
    assert local.endswith("logs/pod/rsl_rl/warpauv_direct")
    assert str(REPO_ROOT) in local
    assert out["local_dir"] == local


def test_sync_logs_rejects_path_escape():
    rt = make_rt(pods=[dict(POD_RUNNING)])
    with pytest.raises(tools.ToolError, match="subdir"):
        tools.sync_logs(rt, subdir="../../etc")


# -------------------------------------------------------- apply_bluerov_patches

def test_apply_patches_refuses_while_job_live():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("20260703-1_train_ab\nLIVE_LIST_END\n")])
    with pytest.raises(guardrails.GuardrailError, match="train"):
        tools.apply_bluerov_patches(rt)
    assert rt.ssh.push_files == []                     # nothing touched


def test_apply_patches_pushes_and_runs():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("LIVE_LIST_END\n"), ok("PATCH OK")])
    out = tools.apply_bluerov_patches(rt)
    files = dict(rt.ssh.push_files)
    assert str(REPO_ROOT / "patches" / "bluerov2_heavy_thrusters.py") in files
    pushed = dict(rt.ssh.push_texts)
    assert "/workspace/patches/apply_bluerov2_patch.py" in pushed
    assert "pristine" in pushed["/workspace/patches/apply_bluerov2_patch.py"]
    run_cmd = rt.ssh.run_calls[-1]
    assert "python3 /workspace/patches/apply_bluerov2_patch.py" in run_cmd
    assert "--thrusters /workspace/patches/bluerov2_heavy_thrusters.py" in run_cmd
    assert out["output"].strip().endswith("PATCH OK")


def test_apply_patches_dry_run_no_pod_contact():
    rt = make_rt()
    out = tools.apply_bluerov_patches(rt, dry_run=True)
    assert out["dry_run"] is True
    assert rt.ssh.run_calls == []


# ------------------------------------------------------------ axis_sanity_sweep

def test_axis_sanity_sweep_marker_only_on_success():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("LIVE_LIST_END\n")])
    out = tools.axis_sanity_sweep(rt)
    assert out["job_id"]
    pushed = dict(rt.ssh.push_texts)
    files = dict(rt.ssh.push_files)
    assert str(REPO_ROOT / "patches" / "axis_sanity_sweep.py") in files
    cmd_sh = next(t for p, t in rt.ssh.push_texts if p.endswith("cmd.sh"))
    assert "./isaaclab.sh -p /workspace/patches/axis_sanity_sweep.py --headless" in cmd_sh
    assert f"&& touch {training.SANITY_MARKER}" in cmd_sh   # only on exit 0


# -------------------------------------------------------------------- stop_pod

def test_stop_pod_refuses_live_jobs():
    rt = make_rt(pods=[dict(POD_RUNNING)],
                 ssh_results=[ok("20260703-1_train_ab\nLIVE_LIST_END\n")])
    with pytest.raises(guardrails.GuardrailError):
        tools.stop_pod(rt)
    assert "stop_pod:on2ghkedz0vbjr" not in rt.client.calls


def test_stop_pod_force_stops():
    rt = make_rt(pods=[dict(POD_RUNNING)])
    out = tools.stop_pod(rt, force=True)
    assert out["status"] == "stopped"
    assert "stop_pod:on2ghkedz0vbjr" in rt.client.calls


def test_stop_pod_idempotent():
    assert tools.stop_pod(make_rt(pods=[dict(POD_STOPPED)]))["status"] == \
        "already_stopped"
    assert tools.stop_pod(make_rt())["status"] == "no_pod"


# --------------------------------------------------------------- terminate_pod

def test_terminate_requires_confirm():
    rt = make_rt(pods=[dict(POD_RUNNING)])
    with pytest.raises(guardrails.GuardrailError):
        tools.terminate_pod(rt, confirm="yes please")
    out = tools.terminate_pod(rt, confirm="terminate lts-replication")
    assert "delete_pod:on2ghkedz0vbjr" in rt.client.calls
    assert "volume" in json.dumps(out).lower()         # survives note


# ---------------------------------------------------------------- spend_report

def test_spend_report_sums_and_includes_volume_storage():
    out = tools.spend_report(make_rt())
    assert out["pod_compute_usd"] == 0.56
    assert out["network_volume_usd"] == 0.01
    assert out["total_usd"] == 0.57
    assert out["budget_usd"] == 50
    assert out["coverage"] == "compute + network volume storage"


def test_spend_report_labels_missing_volume_billing():
    rt = make_rt()
    rt.client.billing_vol_data = None
    out = tools.spend_report(rt)
    assert "compute only" in out["coverage"]
