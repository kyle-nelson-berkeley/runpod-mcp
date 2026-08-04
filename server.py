"""runpod MCP server — drives the Learning-to-Swim runbook on RunPod.

Thin layer only: FastMCP registrations around runpod_mcp.tools.* (all logic
and all tests live there). Launched by run.sh via .mcp.json at the repo root.

Two vehicles, two pods: `vehicle` selects the pod/volume a call acts on —
'hippocampus' (the lts-replication pod, the DEFAULT) or 'bluerov2'
(lts-replication-bluerov2). The two destructive tools, stop_pod and
terminate_pod, take NO default: in a two-pod world "which pod" must never be
implicit.

Startup fails fast (before serving) if the API key is missing from the macOS
Keychain — the key is fetched into memory and never written anywhere.
"""
import sys

from mcp.server.fastmcp import FastMCP

from runpod_mcp import tools
from runpod_mcp.config import ConfigError, fetch_api_key, scrub

mcp = FastMCP("runpod")


VEHICLE_HELP = (
    "pass vehicle='hippocampus' (the lts-replication pod) or "
    "vehicle='bluerov2' (lts-replication-bluerov2) — in a two-pod world "
    "'which pod' must never be implicit.")


def _require_vehicle(tool_name: str, vehicle: str | None) -> None:
    """Destructive tools refuse an implicit pod. Checked here, in the wrapper,
    so tools.stop_pod/terminate_pod keep their vehicle-bound-Runtime signature
    for supervise/deadman."""
    if vehicle is None:
        raise tools.ToolError(f"{tool_name} requires an explicit vehicle: "
                              f"{VEHICLE_HELP}")


@mcp.tool()
def gpu_availability(data_center_id: str | None = None,
                     vehicle: str = "hippocampus") -> dict:
    """RTX 4090 price + stock via RunPod's public GraphQL (read-only, $0).
    Optionally filter to one datacenter (e.g. 'EU-RO-1', where the network
    volume lives). vehicle: hippocampus|bluerov2 (which pod's config to use)."""
    return tools.gpu_availability(tools.runtime(vehicle), data_center_id)


@mcp.tool()
def ensure_pod(dry_run: bool = False, vehicle: str = "hippocampus") -> dict:
    """Ensure this vehicle's pod is RUNNING and SSH-ready: network volume
    first, then resume-or-create (4090/SECURE/non-interruptible enforced in
    code). vehicle: hippocampus (pod lts-replication) | bluerov2 (pod
    lts-replication-bluerov2) — each has its own volume, known_hosts file and
    log dir. Refuses if any UNDECLARED pod exists (the other vehicle's pod is
    fine). Polls ~5 min, verifies the driver floor, truncates known_hosts and
    re-arms the idle watchdog on every transition-to-running. dry_run returns
    the exact would-be payloads."""
    return tools.ensure_pod(tools.runtime(vehicle), dry_run=dry_run)


@mcp.tool()
def pod_status(vehicle: str = "hippocampus") -> dict:
    """Pod state (no_pod / stopped / running / running_ssh_pending) + driver,
    disk, active jobs, $/hr for one vehicle's pod. Read-only."""
    return tools.pod_status(tools.runtime(vehicle))


@mcp.tool()
def run_pod_setup(auto_stop: bool = False, dry_run: bool = False,
                  vehicle: str = "hippocampus") -> dict:
    """Push runbook/pod_setup.sh verbatim and run it as a DETACHED job
    (non-interactive env: DEBIAN_FRONTEND, IsaacSim EULA consent). First run
    30-60 min (5400s ceiling); re-run all-skips ~1-2 min (Day-1 drill).
    Poll with job_status."""
    return tools.run_pod_setup(tools.runtime(vehicle), auto_stop=auto_stop,
                               dry_run=dry_run)


@mcp.tool()
def exec_on_pod(command: str, timeout_sec: int = 120,
                workdir: str | None = None,
                vehicle: str = "hippocampus") -> dict:
    """Bounded synchronous SSH escape hatch (ceiling 600s — use run_job for
    anything longer). Unfiltered by design: guardrails protect money, not the
    pod filesystem."""
    return tools.exec_on_pod(tools.runtime(vehicle), command,
                             timeout_sec=timeout_sec, workdir=workdir)


@mcp.tool()
def run_job(name: str, command: str, workdir: str = "/workspace",
            auto_stop: bool = False, force: bool = False,
            dry_run: bool = False, max_runtime_sec: int | None = None,
            vehicle: str = "hippocampus") -> dict:
    """Generic detached job (eval scripts, pretrained cross-check, plots).
    Refuses while another job is live on that pod unless force=True. auto_stop
    stops the pod when the job ends (probed at launch — fails loudly if it
    can't work)."""
    return tools.run_job(tools.runtime(vehicle), name, command, workdir=workdir,
                         auto_stop=auto_stop, force=force, dry_run=dry_run,
                         max_runtime_sec=max_runtime_sec)


@mcp.tool()
def launch_training(vehicle: str, dr_level: str, seed: int = 1,
                    auto_stop: bool = False, extra_args: str = "",
                    force: bool = False, dry_run: bool = False) -> dict:
    """One RUNBOOK training run (vehicle: curee|bluerov2; dr_level: DR_0/1/2
    or none/small/large). The vehicle also picks the pod: curee trains on the
    hippocampus pod (lts-replication), bluerov2 on lts-replication-bluerov2.
    Rewrites the two DR cfg lines content-anchored (aborts on unrecognized
    current values), then launches the verbatim RUNBOOK train command. Gates:
    bluerov2 needs patch+sanity markers; curee refuses on a patched checkout.
    Poll with job_status."""
    rt = tools.runtime(tools.TRAINING_VEHICLE_TO_POD.get(vehicle, "hippocampus"))
    return tools.launch_training(rt, vehicle, dr_level,
                                 seed=seed, auto_stop=auto_stop,
                                 extra_args=extra_args, force=force,
                                 dry_run=dry_run)


@mcp.tool()
def job_status(job_id: str, tail_lines: int = 40,
               vehicle: str = "hippocampus") -> dict:
    """State of a detached job (running/succeeded/failed/orphaned/not_found)
    + log tail + latest 'Mean reward' line. exit_code 124 = wall-clock
    ceiling hit."""
    return tools.job_status(tools.runtime(vehicle), job_id,
                            tail_lines=tail_lines)


@mcp.tool()
def sync_logs(subdir: str = "rsl_rl/warpauv_direct",
              vehicle: str = "hippocampus") -> dict:
    """rsync -az --partial pod:/workspace/IsaacLab/logs/<subdir>/ down to that
    vehicle's local_log_dir/<subdir> (hippocampus -> logs/pod, bluerov2 ->
    logs/pod/bluerov2; both gitignored). Never deletes on either side."""
    return tools.sync_logs(tools.runtime(vehicle), subdir=subdir)


@mcp.tool()
def apply_bluerov_patches(dry_run: bool = False, force: bool = False) -> dict:
    """Automate patches/APPLY.md §§1-5 on the BLUEROV2 pod (bluerov2 by
    nature — no vehicle param): restore pristine 7c5ebe7, install the
    8-thruster drop-in, apply the content-anchored edits, write
    markers/bluerov2_patch_applied. Idempotent (pristine-then-apply). Refuses
    while a job is live unless force=True (the checkout is shared)."""
    return tools.apply_bluerov_patches(tools.runtime("bluerov2"),
                                       dry_run=dry_run, force=force)


@mcp.tool()
def axis_sanity_sweep(auto_stop: bool = False) -> dict:
    """Run patches/axis_sanity_sweep.py headless as a detached job on the
    BLUEROV2 pod (bluerov2 by nature — no vehicle param); writes
    markers/axis_sanity_PASS ONLY on exit 0. MANDATORY after patching and
    BEFORE any bluerov2 training (RUNBOOK Days 4-5)."""
    return tools.axis_sanity_sweep(tools.runtime("bluerov2"),
                                   auto_stop=auto_stop)


@mcp.tool()
def stop_pod(vehicle: str | None = None, force: bool = False) -> dict:
    """Default teardown — stop the pod whenever not training. vehicle is
    REQUIRED (no default): 'hippocampus' (lts-replication) or 'bluerov2'
    (lts-replication-bluerov2). Refuses if a job is live unless force=True.
    Idempotent. Stopped pods bill only volume storage."""
    _require_vehicle("stop_pod", vehicle)
    return tools.stop_pod(tools.runtime(vehicle), force=force)


@mcp.tool()
def terminate_pod(confirm: str, vehicle: str | None = None) -> dict:
    """DELETE one vehicle's pod (its network volume survives). vehicle is
    REQUIRED (no default): 'hippocampus' or 'bluerov2'. Requires the verbatim
    confirm string 'terminate <that vehicle's pod_name>' — i.e.
    'terminate lts-replication' for hippocampus, or
    'terminate lts-replication-bluerov2' for bluerov2. Human-approval-only per
    the root CLAUDE.md."""
    _require_vehicle("terminate_pod", vehicle)
    return tools.terminate_pod(tools.runtime(vehicle), confirm=confirm)


@mcp.tool()
def spend_report() -> dict:
    """Billing so far (pod compute + network volume storage) vs budget_usd.
    ACCOUNT-WIDE — RunPod billing has no per-pod filter, so this covers BOTH
    vehicles (no vehicle param). Informational only — the hard stop is
    RunPod's prepaid credit balance."""
    return tools.spend_report(tools.runtime())


def main() -> None:
    try:
        fetch_api_key()          # fail fast, before serving any tool call
    except ConfigError as exc:
        print(scrub(exc), file=sys.stderr)
        sys.exit(1)
    mcp.run()                    # stdio transport


if __name__ == "__main__":
    main()
