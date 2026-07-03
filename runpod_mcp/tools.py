"""Tool implementations for the runpod MCP server.

server.py registers thin FastMCP wrappers around these; every function takes
an explicit Runtime so the whole surface is testable with fake api/ssh layers.

Design (consensus plan):
  - STATELESS: "the pod" = whatever GET /pods returns matching the configured
    name — the console and the MCP always agree. Only local state: a 60s-TTL
    (host, port) connection cache.
  - Guardrails protect money, not the pod filesystem.
  - dry_run paths return the exact would-be payloads without mutating.
"""
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

from . import api, guardrails, jobs, ssh, training
from .config import (REPO_ROOT, ConfigError, fetch_api_key, load_defaults,
                     read_ssh_public_key, scrub)

# Env for pod_setup.sh's FIRST-EVER detached run (only ever pasted
# interactively before): apt must never prompt, IsaacSim must not stop on
# EULA/telemetry consent (both env-var conventions set — harmless if unused).
SETUP_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "ACCEPT_EULA": "Y",
    "PRIVACY_CONSENT": "Y",
    "OMNI_KIT_ACCEPT_EULA": "YES",
}

POD_SETUP_LOCAL = REPO_ROOT / "runbook" / "pod_setup.sh"
POD_SETUP_REMOTE = "/workspace/pod_setup.sh"
SWEEP_LOCAL = REPO_ROOT / "patches" / "axis_sanity_sweep.py"
SWEEP_REMOTE = "/workspace/patches/axis_sanity_sweep.py"
THRUSTERS_LOCAL = REPO_ROOT / "patches" / "bluerov2_heavy_thrusters.py"
THRUSTERS_REMOTE = "/workspace/patches/bluerov2_heavy_thrusters.py"
PATCH_SCRIPT_LOCAL = Path(__file__).resolve().parent / "remote" / "apply_bluerov2_patch.py"
PATCH_SCRIPT_REMOTE = "/workspace/patches/apply_bluerov2_patch.py"


class ToolError(RuntimeError):
    """Refused/failed tool call; message is scrubbed and actionable."""


class Runtime:
    """Lazy holder for config + API/SSH clients. Tests inject fakes."""

    def __init__(self, cfg=None, client=None, sshc=None, sleep=time.sleep,
                 gpu_types=api.gpu_types, ssh_pubkey=None):
        self.cfg = cfg or load_defaults()
        self._client = client
        self._ssh = sshc
        self.sleep = sleep
        self.gpu_types = gpu_types
        self._ssh_pubkey = ssh_pubkey
        self.conn_cache = ssh.ConnCache()

    @property
    def client(self) -> api.RunPodClient:
        if self._client is None:
            self._client = api.RunPodClient(fetch_api_key())
        return self._client

    @property
    def ssh(self) -> ssh.SSHClient:
        if self._ssh is None:
            self._ssh = ssh.SSHClient(self.cfg["ssh_identity"],
                                      self.cfg["known_hosts_file"])
        return self._ssh

    @property
    def ssh_pubkey(self) -> str:
        if self._ssh_pubkey is None:
            self._ssh_pubkey = read_ssh_public_key(self.cfg)
        return self._ssh_pubkey


_runtime: Runtime | None = None


def runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


# ------------------------------------------------------------------- helpers

def _find_pod(rt: Runtime, pods=None) -> dict | None:
    pods = rt.client.list_pods() if pods is None else pods
    return next((p for p in pods if p.get("name") == rt.cfg["pod_name"]), None)


def _conn_info(rt: Runtime, pod: dict | None = None) -> tuple[str, int]:
    cached = rt.conn_cache.get()
    if cached:
        return cached
    pod = pod or _find_pod(rt)
    if pod is None:
        raise ToolError(f"no pod named '{rt.cfg['pod_name']}' exists — "
                        "run ensure_pod first")
    host = pod.get("publicIp")
    mapping = pod.get("portMappings") or {}
    port = mapping.get("22")
    if pod.get("desiredStatus") != "RUNNING" or not host or not port:
        raise ToolError(f"pod is not SSH-ready (status={pod.get('desiredStatus')}, "
                        f"publicIp={host!r}) — run ensure_pod and wait for "
                        "'running'")
    rt.conn_cache.put(host, int(port))
    return host, int(port)


def _driver_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def _ceiling(rt: Runtime, key: str, override: int | None = None) -> int:
    return int(override) if override else int(rt.cfg["timeouts"][key])


def _launch_guarded(rt: Runtime, *, name: str, command: str, workdir: str,
                    max_runtime_sec: int, auto_stop: bool, force: bool,
                    env: dict | None = None) -> dict:
    host, port = _conn_info(rt)
    pod = _find_pod(rt)
    live = jobs.list_live(rt.ssh, host, port)
    guardrails.assert_no_live_jobs(live, force=force)   # one job at a time
    job_id = jobs.launch(rt.ssh, host, port, name=name, command=command,
                         workdir=workdir, pod_id=pod["id"],
                         max_runtime_sec=max_runtime_sec,
                         auto_stop=auto_stop, env=env)
    return {"job_id": job_id, "max_runtime_sec": max_runtime_sec,
            "auto_stop": auto_stop,
            "hint": f"poll job_status('{job_id}') — do NOT block on it"}


# ============================================================ read-only tools

def gpu_availability(rt: Runtime, data_center_id: str | None = None) -> dict:
    types = rt.gpu_types(data_center_id=data_center_id)
    if not types:
        raise ToolError("GraphQL returned no RTX 4090 entry")
    g = types[0]
    lp = g.get("lowestPrice") or {}
    return {
        "gpu": g["id"],
        "memory_gb": g.get("memoryInGb"),
        "data_center": data_center_id or "(any)",
        "stock_status": lp.get("stockStatus"),
        "on_demand_price_usd_hr": lp.get("uninterruptablePrice"),
        "spot_price_usd_hr_UNUSED": lp.get("minimumBidPrice"),
        "secure_cloud": g.get("secureCloud"),
    }


def pod_status(rt: Runtime) -> dict:
    pod = _find_pod(rt)
    if pod is None:
        return {"status": "no_pod",
                "hint": "ensure_pod() creates volume + pod (Day 1)"}
    out = {"pod_id": pod["id"], "name": pod["name"],
           "cost_per_hr_usd": pod.get("costPerHr"),
           "image": pod.get("imageName"),
           "last_started_at": pod.get("lastStartedAt")}
    if pod.get("desiredStatus") != "RUNNING":
        out["status"] = "stopped"
        out["hint"] = "stopped pods bill only volume storage; ensure_pod resumes"
        return out
    try:
        host, port = _conn_info(rt, pod)
        driver = rt.ssh.run(host, port,
                            "nvidia-smi --query-gpu=driver_version "
                            "--format=csv,noheader", check=True).stdout.strip()
        disk = rt.ssh.run(host, port,
                          "df -h /workspace | tail -1").stdout.strip()
        out.update(status="running", ssh=f"root@{host}:{port}",
                   driver_version=driver, workspace_disk=disk)
        try:
            out["active_jobs"] = jobs.list_live(rt.ssh, host, port)
        except jobs.JobError as exc:
            out["active_jobs_error"] = scrub(exc)
    except (ToolError, ssh.SSHError, jobs.JobError) as exc:
        out["status"] = "running_ssh_pending"
        out["detail"] = scrub(exc)
    return out


def job_status(rt: Runtime, job_id: str, tail_lines: int = 40) -> dict:
    host, port = _conn_info(rt)
    return jobs.status(rt.ssh, host, port, job_id, tail_lines=tail_lines)


def spend_report(rt: Runtime) -> dict:
    def total(items):
        return round(sum(float(i.get("amount", 0)) for i in items), 4)

    pods_billing = rt.client.billing_pods() or []
    vol_billing = rt.client.billing_network_volumes()
    pod_usd = total(pods_billing)
    out = {"pod_compute_usd": pod_usd, "budget_usd": rt.cfg["budget_usd"]}
    if vol_billing is None:
        out["coverage"] = ("compute only, excludes ~$4.20/mo volume storage "
                           "(billing/networkvolumes endpoint absent)")
        out["total_usd"] = pod_usd
    else:
        vol_usd = total(vol_billing)
        out["network_volume_usd"] = vol_usd
        out["total_usd"] = round(pod_usd + vol_usd, 4)
        out["coverage"] = "compute + network volume storage"
    out["budget_remaining_usd"] = round(out["budget_usd"] - out["total_usd"], 4)
    return out


# ============================================================ lifecycle tools

def _pick_datacenter(rt: Runtime) -> str:
    """First preferred DC with any 4090 stock (live-verified filter)."""
    checked = {}
    for dc in rt.cfg["datacenter_preference"]:
        try:
            types = rt.gpu_types(data_center_id=dc)
            status = (types[0].get("lowestPrice") or {}).get("stockStatus") \
                if types else None
        except api.ApiError as exc:
            status = f"query failed: {scrub(exc)}"
        checked[dc] = status
        if status in ("High", "Medium", "Low"):
            return dc
    raise ToolError(
        f"no preferred datacenter shows 4090 stock right now: {checked}. "
        "Check gpu_availability() later, or extend datacenter_preference in "
        "runpod-mcp/pod_defaults.yaml.")


def _build_pod_payload(rt: Runtime, volume_id: str, data_center: str) -> dict:
    cfg = rt.cfg
    payload = {
        "name": cfg["pod_name"],
        "imageName": cfg["image_name"],
        "cloudType": cfg["cloud_type"],
        "gpuTypeIds": list(cfg["gpu_type_ids"]),
        "gpuCount": cfg["gpu_count"],
        "interruptible": False,
        "containerDiskInGb": cfg["container_disk_gb"],
        "networkVolumeId": volume_id,
        "volumeMountPath": cfg["volume_mount_path"],
        "ports": list(cfg["ports"]),
        "supportPublicIp": cfg["support_public_ip"],
        "allowedCudaVersions": list(cfg["allowed_cuda_versions"]),
        "dataCenterIds": [data_center],
        # PUBLIC_KEY is what runpod/pytorch images actually honor (live-
        # verified 2026-07-03); SSH_PUBLIC_KEY kept as belt-and-braces.
        "env": {"PUBLIC_KEY": rt.ssh_pubkey, "SSH_PUBLIC_KEY": rt.ssh_pubkey},
    }
    return guardrails.enforce_pod_payload(payload)


NO_GPU_RECOVERY = {
    "why": ("stopped pods do NOT reserve their GPU — this host has no free "
            "4090 right now; the network volume is untouched (data survives "
            "pod termination)"),
    "recipe": [
        "1. check stock: gpu_availability(data_center_id='<volume DC>')",
        "2. if stock exists: terminate_pod(confirm='terminate lts-replication')"
        "   [KYLE-APPROVAL-ONLY per CLAUDE.md] — the volume survives",
        "3. ensure_pod() — recreates the pod in the SAME datacenter, attached"
        "   to the same volume, zero data loss",
        "4. if the DC is dry: wait and retry, or (worst case) create a second"
        "   volume in another DC (pod_defaults datacenter_preference)",
    ],
    "note": "terminate stays confirm-gated — never automatic",
}


def ensure_pod(rt: Runtime, dry_run: bool = False) -> dict:
    cfg = rt.cfg
    pods = rt.client.list_pods()
    guardrails.assert_only_pod(pods, cfg["pod_name"])   # one-pod max
    pod = _find_pod(rt, pods)

    # ---- network volume first (survives termination; pins the DC) ----------
    vols = rt.client.list_network_volumes()
    vol = next((v for v in vols if v.get("name") == cfg["network_volume_name"]),
               None)
    volume_payload = None
    if vol is None:
        dc = _pick_datacenter(rt)
        volume_payload = {"name": cfg["network_volume_name"],
                          "size": cfg["network_volume_gb"],
                          "dataCenterId": dc}
    else:
        dc = vol["dataCenterId"]

    # An EXISTING pod must actually be attached to OUR volume — a pod created
    # outside this server without it would write setup/training results to
    # ephemeral container storage while we report a volume-backed workflow.
    volume_mismatch = None
    if pod is not None:
        if not pod.get("networkVolumeId"):
            volume_mismatch = (f"pod {pod['id']} has NO network volume attached "
                               "— its /workspace is ephemeral (lost on stop)")
        elif vol is not None and pod["networkVolumeId"] != vol["id"]:
            volume_mismatch = (f"pod {pod['id']} is attached to volume "
                               f"{pod['networkVolumeId']}, not "
                               f"'{cfg['network_volume_name']}' ({vol['id']})")
        elif vol is None:
            volume_mismatch = (f"pod {pod['id']} is attached to volume "
                               f"{pod['networkVolumeId']}, but no volume named "
                               f"'{cfg['network_volume_name']}' exists — "
                               "creating a second volume would split the data")

    if dry_run:
        pod_payload = _build_pod_payload(
            rt, vol["id"] if vol else "(created above)", dc)
        return {
            "dry_run": True,
            "volume_payload": volume_payload or
                {"exists": vol["id"], "dataCenterId": dc},
            "pod_payload": pod_payload if pod is None else
                {"exists": pod["id"], "desiredStatus": pod.get("desiredStatus")},
            "volume_check": volume_mismatch or "ok",
            "actions": ([f"POST /networkvolumes ({dc})"] if volume_payload else [])
                + (["POST /pods"] if pod is None else
                   [] if pod.get("desiredStatus") == "RUNNING" else
                   [f"POST /pods/{pod['id']}/start"]),
        }

    if volume_mismatch:
        raise ToolError(
            f"{volume_mismatch}. Refusing to reuse this pod. Fix in the "
            "console (or align network_volume_name in pod_defaults.yaml), "
            "or terminate_pod + ensure_pod to recreate it correctly.")

    if vol is None:
        vol = rt.client.create_network_volume(volume_payload)

    # ---- create / resume ----------------------------------------------------
    if pod is None:
        pod = rt.client.create_pod(_build_pod_payload(rt, vol["id"], dc))
    elif pod.get("desiredStatus") != "RUNNING":
        try:
            rt.client.start_pod(pod["id"])
        except api.ApiError as exc:
            if api.looks_like_no_gpu_error(str(exc)):
                return {"status": "start_failed_no_gpu", "pod_id": pod["id"],
                        "error": scrub(exc), "recovery": NO_GPU_RECOVERY}
            raise

    # ALWAYS treat this as a possible transition-to-running: the pod may have
    # been stopped/started from the console since we last looked (container
    # disk wiped -> new host keys, watchdog gone). Truncating + re-arming on
    # every call is idempotent and closes that hole.
    rt.ssh.truncate_known_hosts()
    rt.conn_cache.clear()

    # ---- poll for SSH readiness + driver floor ------------------------------
    poll = cfg["poll"]
    deadline = time.monotonic() + poll["budget_sec"]
    host = port = None
    driver = None
    while time.monotonic() < deadline:
        fresh = rt.client.get_pod(pod["id"])
        host = fresh.get("publicIp")
        port = (fresh.get("portMappings") or {}).get("22")
        if host and port:
            try:
                driver = rt.ssh.run(host, int(port),
                                    "nvidia-smi --query-gpu=driver_version "
                                    "--format=csv,noheader",
                                    check=True).stdout.strip()
                break
            except ssh.SSHError:
                pass                                    # sshd not up yet
        rt.sleep(poll["interval_sec"])
    if not driver:
        return {"status": "still_initializing", "pod_id": pod["id"],
                "hint": f"publicIp/SSH not ready within {poll['budget_sec']}s "
                        "— call ensure_pod again in a few minutes"}

    if _driver_tuple(driver) < _driver_tuple(cfg["min_driver_version"]):
        return {"status": "driver_too_old", "pod_id": pod["id"],
                "driver_version": driver,
                "required": cfg["min_driver_version"],
                "hint": "IsaacSim 4.5.0 floor (pod_setup.sh will refuse too) "
                        "— terminate and recreate to land on another host"}

    rt.conn_cache.put(host, int(port))
    out = {"status": "running", "pod_id": pod["id"], "ssh": f"root@{host}:{port}",
           "driver_version": driver, "datacenter": dc,
           "cost_per_hr_usd": pod.get("costPerHr")}

    # ---- idle watchdog: (re)armed on every successful ensure_pod ------------
    try:
        jobs.install_watchdog(rt.ssh, host, int(port), pod["id"],
                              cfg["idle_minutes"])
        out["idle_watchdog"] = "armed"
    except (jobs.JobError, ssh.SSHError) as exc:
        out["idle_watchdog"] = "FAILED"                 # loud, never silent
        out["idle_watchdog_warning"] = (
            f"{scrub(exc)} — the pod will NOT self-stop when idle; "
            "stop_pod() yourself when done")
    return out


def stop_pod(rt: Runtime, force: bool = False) -> dict:
    pod = _find_pod(rt)
    if pod is None:
        return {"status": "no_pod"}
    if pod.get("desiredStatus") != "RUNNING":
        return {"status": "already_stopped", "pod_id": pod["id"]}
    if not force:
        try:
            host, port = _conn_info(rt, pod)
            live = jobs.list_live(rt.ssh, host, port)
        except (ToolError, ssh.SSHError, jobs.JobError) as exc:
            raise ToolError(
                f"could not verify no jobs are running ({scrub(exc)}) — "
                "pass force=True to stop anyway (a live training run would "
                "be killed)") from None
        guardrails.assert_no_live_jobs(live, force=False)
    rt.client.stop_pod(pod["id"])
    rt.conn_cache.clear()
    return {"status": "stopped", "pod_id": pod["id"],
            "note": "GPU billing stopped; volume storage continues; the GPU "
                    "is NOT reserved for restart (see ensure_pod recovery)"}


def terminate_pod(rt: Runtime, confirm: str) -> dict:
    guardrails.assert_terminate_confirm(confirm, rt.cfg["pod_name"])
    pod = _find_pod(rt)
    if pod is None:
        return {"status": "no_pod", "note": "nothing to terminate"}
    rt.client.delete_pod(pod["id"])
    rt.conn_cache.clear()
    return {"status": "terminated", "pod_id": pod["id"],
            "note": "pod deleted; the NETWORK VOLUME SURVIVES — ensure_pod() "
                    "recreates the pod attached to it"}


# ============================================================== job tools

def run_pod_setup(rt: Runtime, auto_stop: bool = False,
                  dry_run: bool = False) -> dict:
    ceiling = _ceiling(rt, "setup_sec")
    command = f"bash {POD_SETUP_REMOTE}"
    if dry_run:
        return {"dry_run": True,
                "scp": {"local": str(POD_SETUP_LOCAL),
                        "remote": POD_SETUP_REMOTE},
                "cmd_script": jobs.build_cmd_script(command, "/workspace",
                                                    SETUP_ENV),
                "env": SETUP_ENV, "max_runtime_sec": ceiling,
                "auto_stop": auto_stop,
                "note": "first run 30-60 min; re-run all-skips in ~1-2 min "
                        "(Day-1 persistence drill)"}
    host, port = _conn_info(rt)
    rt.ssh.push_file(host, port, POD_SETUP_LOCAL, POD_SETUP_REMOTE)
    return _launch_guarded(rt, name="pod-setup", command=command,
                           workdir="/workspace", max_runtime_sec=ceiling,
                           auto_stop=auto_stop, force=False, env=SETUP_ENV)


def exec_on_pod(rt: Runtime, command: str, timeout_sec: int = 120,
                workdir: str | None = None) -> dict:
    timeout = guardrails.clamp_exec_timeout(
        timeout_sec, rt.cfg["timeouts"]["exec_max_sec"])
    host, port = _conn_info(rt)
    full = f"cd {shlex.quote(workdir)} && {command}" if workdir else command
    proc = rt.ssh.run(host, port, full, timeout=timeout)
    return {"exit_code": proc.returncode,
            "stdout": scrub(proc.stdout)[-8000:],
            "stderr": scrub(proc.stderr)[-4000:],
            "timeout_sec": timeout}


def run_job(rt: Runtime, name: str, command: str, workdir: str = "/workspace",
            auto_stop: bool = False, force: bool = False,
            dry_run: bool = False, max_runtime_sec: int | None = None) -> dict:
    ceiling = _ceiling(rt, "job_sec", max_runtime_sec)
    if dry_run:
        return {"dry_run": True,
                "cmd_script": jobs.build_cmd_script(command, workdir),
                "max_runtime_sec": ceiling, "auto_stop": auto_stop}
    return _launch_guarded(rt, name=name, command=command, workdir=workdir,
                           max_runtime_sec=ceiling, auto_stop=auto_stop,
                           force=force)


# ========================================================== training tools

def _probe_markers(rt: Runtime, host: str, port: int) -> dict:
    out = rt.ssh.run(
        host, port,
        f'p=absent; s=absent; '
        f'[ -f {training.PATCH_MARKER} ] && p=present; '
        f'[ -f {training.SANITY_MARKER} ] && s=present; '
        f'echo "PATCH=$p SANITY=$s"', check=True).stdout
    m = re.search(r"PATCH=(\w+) SANITY=(\w+)", out)
    if not m:
        raise ToolError(f"marker probe failed: {scrub(out)[:200]}")
    return {"patch": m.group(1) == "present", "sanity": m.group(2) == "present"}


def _check_vehicle_gates(vehicle: str, markers: dict) -> None:
    """Symmetric gates — both directions protect against silently poisoned runs."""
    if vehicle == "bluerov2":
        if not markers["patch"]:
            raise ToolError(
                "bluerov2 training refused: markers/bluerov2_patch_applied is "
                "missing — run apply_bluerov_patches() first (RUNBOOK Days 4-5)")
        if not markers["sanity"]:
            raise ToolError(
                "bluerov2 training refused: markers/axis_sanity_PASS is missing "
                "— run axis_sanity_sweep() and get exit 0 BEFORE any training "
                "(a wrong thruster sign would waste every subsequent run)")
    elif markers["patch"]:
        raise ToolError(
            "curee training refused: the checkout is PATCHED for BlueROV2 "
            "(markers/bluerov2_patch_applied exists) — CUREE on an 8-thruster "
            "checkout silently poisons results. Recovery: exec_on_pod("
            "'cd /workspace/isaac-auv-env && rm -f bluerov2_heavy_thrusters.py "
            "&& git checkout -- .'), then delete BOTH marker files under "
            "/workspace/markers/ and re-run.")


def launch_training(rt: Runtime, vehicle: str, dr_level: str, seed: int = 1,
                    auto_stop: bool = False, extra_args: str = "",
                    force: bool = False, dry_run: bool = False) -> dict:
    if vehicle not in training.VEHICLES:
        raise ToolError(f"unknown vehicle {vehicle!r} — use curee|bluerov2")
    level = training.canonical_level(dr_level)
    command = training.build_train_command(seed=seed, extra_args=extra_args)
    values = training.DR_TABLES[vehicle][level]
    ceiling = _ceiling(rt, "training_sec")

    if dry_run:
        return {"dry_run": True, "vehicle": vehicle, "dr_level": level,
                "seed": seed, "command": command,
                "workdir": training.ISAACLAB_DIR,
                "dr_edits": dict(values), "max_runtime_sec": ceiling,
                "auto_stop": auto_stop,
                "gates": ("bluerov2 needs patch+sanity markers; curee needs "
                          "the patch marker ABSENT (symmetric gate)")}

    host, port = _conn_info(rt)
    _check_vehicle_gates(vehicle, _probe_markers(rt, host, port))
    live = jobs.list_live(rt.ssh, host, port)
    guardrails.assert_no_live_jobs(live, force=force)

    # content-anchored DR edit: cat -> rewrite locally -> push back
    src = rt.ssh.run(host, port, f"cat {training.WARPAUV_ENV_REMOTE}",
                     check=True).stdout
    new_src = training.apply_dr_to_source(src, vehicle, level)
    rt.ssh.push_text(host, port, new_src, training.WARPAUV_ENV_REMOTE)

    pod = _find_pod(rt)
    job_id = jobs.launch(rt.ssh, host, port,
                         name=f"train-{vehicle}-{level}-s{seed}",
                         command=command, workdir=training.ISAACLAB_DIR,
                         pod_id=pod["id"], max_runtime_sec=ceiling,
                         auto_stop=auto_stop)
    return {"job_id": job_id, "vehicle": vehicle, "dr_level": level,
            "seed": seed, "dr_edits": dict(values), "auto_stop": auto_stop,
            "hint": f"poll job_status('{job_id}'); expect ~400 iters, "
                    "mean reward ~95-100 (RUNBOOK)"}


def apply_bluerov_patches(rt: Runtime, dry_run: bool = False,
                          force: bool = False) -> dict:
    remote_cmd = (f"python3 {PATCH_SCRIPT_REMOTE} "
                  f"--repo /workspace/isaac-auv-env "
                  f"--thrusters {THRUSTERS_REMOTE} "
                  f"--markers {training.MARKERS_DIR}")
    if dry_run:
        return {"dry_run": True,
                "pushes": [str(THRUSTERS_LOCAL), str(PATCH_SCRIPT_LOCAL)],
                "remote_command": remote_cmd,
                "note": "the script itself restores pristine 7c5ebe7, then "
                        "applies APPLY.md §§1-5 content-anchored (re-verifies "
                        "every anchor before writing)"}
    host, port = _conn_info(rt)
    # rewriting the shared checkout under a live setup/training job would
    # poison its results — same one-job guard as every mutating job tool
    guardrails.assert_no_live_jobs(jobs.list_live(rt.ssh, host, port),
                                   force=force)
    rt.ssh.push_text(host, port, PATCH_SCRIPT_LOCAL.read_text(),
                     PATCH_SCRIPT_REMOTE)
    rt.ssh.push_file(host, port, THRUSTERS_LOCAL, THRUSTERS_REMOTE)
    proc = rt.ssh.run(host, port, remote_cmd, timeout=120)
    if proc.returncode != 0:
        raise ToolError(f"apply_bluerov2_patch.py failed (rc={proc.returncode}):\n"
                        f"{scrub(proc.stdout)[-2000:]}\n{scrub(proc.stderr)[-2000:]}")
    return {"status": "patched", "output": scrub(proc.stdout)[-2000:],
            "next": "axis_sanity_sweep() is MANDATORY before bluerov2 training"}


def axis_sanity_sweep(rt: Runtime, auto_stop: bool = False) -> dict:
    host, port = _conn_info(rt)
    rt.ssh.push_file(host, port, SWEEP_LOCAL, SWEEP_REMOTE)
    command = (f"./isaaclab.sh -p {SWEEP_REMOTE} --headless "
               f"&& mkdir -p {training.MARKERS_DIR} "
               f"&& touch {training.SANITY_MARKER}")   # marker ONLY on exit 0
    out = _launch_guarded(rt, name="axis-sanity-sweep", command=command,
                          workdir=training.ISAACLAB_DIR,
                          max_runtime_sec=_ceiling(rt, "sweep_sec"),
                          auto_stop=auto_stop, force=False)
    out["gate"] = ("exit 0 writes markers/axis_sanity_PASS — required by "
                   "launch_training('bluerov2', ...); any FLIP/WEAK = STOP")
    return out


# ============================================================== log retrieval

def sync_logs(rt: Runtime, subdir: str = "rsl_rl/warpauv_direct") -> dict:
    clean = subdir.strip("/")
    if ".." in clean.split("/") or clean.startswith("~") or not clean:
        raise ToolError(f"bad subdir {subdir!r} — must be a relative path "
                        "under /workspace/IsaacLab/logs/")
    host, port = _conn_info(rt)
    remote = f"/workspace/IsaacLab/logs/{clean}/"
    local = REPO_ROOT / rt.cfg["local_log_dir"] / clean
    output = rt.ssh.rsync_pull(host, port, remote, local)
    return {"remote": remote, "local_dir": str(local),
            "rsync_output": scrub(output)[-1000:],
            "note": "never deletes on either side; logs/pod/ is gitignored"}
