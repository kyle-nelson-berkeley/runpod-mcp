"""runpod MCP server — drives the Learning-to-Swim runbook on RunPod.

Thin layer only: FastMCP registrations around runpod_mcp.tools.* (all logic
and all tests live there). Launched by run.sh via .mcp.json at the repo root.

Startup fails fast (before serving) if the API key is missing from the macOS
Keychain — the key is fetched into memory and never written anywhere.
"""
import sys

from mcp.server.fastmcp import FastMCP

from runpod_mcp import tools
from runpod_mcp.config import ConfigError, fetch_api_key, scrub

mcp = FastMCP("runpod")


@mcp.tool()
def gpu_availability(data_center_id: str | None = None) -> dict:
    """RTX 4090 price + stock via RunPod's public GraphQL (read-only, $0).
    Optionally filter to one datacenter (e.g. 'EU-RO-1', where the network
    volume lives)."""
    return tools.gpu_availability(tools.runtime(), data_center_id)


@mcp.tool()
def ensure_pod(dry_run: bool = False) -> dict:
    """Ensure the lts-replication pod is RUNNING and SSH-ready: network volume
    first, then resume-or-create (4090/SECURE/non-interruptible enforced in
    code). Refuses if any other pod exists. Polls ~5 min, verifies the driver
    floor, truncates known_hosts and re-arms the idle watchdog on every
    transition-to-running. dry_run returns the exact would-be payloads."""
    return tools.ensure_pod(tools.runtime(), dry_run=dry_run)


@mcp.tool()
def pod_status() -> dict:
    """Pod state (no_pod / stopped / running / running_ssh_pending) + driver,
    disk, active jobs, $/hr. Read-only."""
    return tools.pod_status(tools.runtime())


@mcp.tool()
def run_pod_setup(auto_stop: bool = False, dry_run: bool = False) -> dict:
    """Push runbook/pod_setup.sh verbatim and run it as a DETACHED job
    (non-interactive env: DEBIAN_FRONTEND, IsaacSim EULA consent). First run
    30-60 min (5400s ceiling); re-run all-skips ~1-2 min (Day-1 drill).
    Poll with job_status."""
    return tools.run_pod_setup(tools.runtime(), auto_stop=auto_stop,
                               dry_run=dry_run)


@mcp.tool()
def exec_on_pod(command: str, timeout_sec: int = 120,
                workdir: str | None = None) -> dict:
    """Bounded synchronous SSH escape hatch (ceiling 600s — use run_job for
    anything longer). Unfiltered by design: guardrails protect money, not the
    pod filesystem."""
    return tools.exec_on_pod(tools.runtime(), command,
                             timeout_sec=timeout_sec, workdir=workdir)


@mcp.tool()
def run_job(name: str, command: str, workdir: str = "/workspace",
            auto_stop: bool = False, force: bool = False,
            dry_run: bool = False, max_runtime_sec: int | None = None) -> dict:
    """Generic detached job (eval scripts, pretrained cross-check, plots).
    Refuses while another job is live unless force=True. auto_stop stops the
    pod when the job ends (probed at launch — fails loudly if it can't work)."""
    return tools.run_job(tools.runtime(), name, command, workdir=workdir,
                         auto_stop=auto_stop, force=force, dry_run=dry_run,
                         max_runtime_sec=max_runtime_sec)


@mcp.tool()
def launch_training(vehicle: str, dr_level: str, seed: int = 1,
                    auto_stop: bool = False, extra_args: str = "",
                    force: bool = False, dry_run: bool = False) -> dict:
    """One RUNBOOK training run (vehicle: curee|bluerov2; dr_level: DR_0/1/2
    or none/small/large). Rewrites the two DR cfg lines content-anchored
    (aborts on unrecognized current values), then launches the verbatim
    RUNBOOK train command. Gates: bluerov2 needs patch+sanity markers; curee
    refuses on a patched checkout. Poll with job_status."""
    return tools.launch_training(tools.runtime(), vehicle, dr_level,
                                 seed=seed, auto_stop=auto_stop,
                                 extra_args=extra_args, force=force,
                                 dry_run=dry_run)


@mcp.tool()
def job_status(job_id: str, tail_lines: int = 40) -> dict:
    """State of a detached job (running/succeeded/failed/orphaned/not_found)
    + log tail + latest 'Mean reward' line. exit_code 124 = wall-clock
    ceiling hit."""
    return tools.job_status(tools.runtime(), job_id, tail_lines=tail_lines)


@mcp.tool()
def sync_logs(subdir: str = "rsl_rl/warpauv_direct") -> dict:
    """rsync -az --partial pod:/workspace/IsaacLab/logs/<subdir>/ down to
    logs/pod/<subdir> (gitignored). Never deletes on either side."""
    return tools.sync_logs(tools.runtime(), subdir=subdir)


@mcp.tool()
def apply_bluerov_patches(dry_run: bool = False, force: bool = False) -> dict:
    """Automate patches/APPLY.md §§1-5 on the pod: restore pristine 7c5ebe7,
    install the 8-thruster drop-in, apply the content-anchored edits, write
    markers/bluerov2_patch_applied. Idempotent (pristine-then-apply). Refuses
    while a job is live unless force=True (the checkout is shared)."""
    return tools.apply_bluerov_patches(tools.runtime(), dry_run=dry_run,
                                       force=force)


@mcp.tool()
def axis_sanity_sweep(auto_stop: bool = False) -> dict:
    """Run patches/axis_sanity_sweep.py headless as a detached job; writes
    markers/axis_sanity_PASS ONLY on exit 0. MANDATORY after patching and
    BEFORE any bluerov2 training (RUNBOOK Days 4-5)."""
    return tools.axis_sanity_sweep(tools.runtime(), auto_stop=auto_stop)


@mcp.tool()
def stop_pod(force: bool = False) -> dict:
    """Default teardown — stop the pod whenever not training. Refuses if a
    job is live unless force=True. Idempotent. Stopped pods bill only volume
    storage."""
    return tools.stop_pod(tools.runtime(), force=force)


@mcp.tool()
def terminate_pod(confirm: str) -> dict:
    """DELETE the pod (network volume survives). Requires the verbatim
    confirm string 'terminate lts-replication'. Human-approval-only per the
    root CLAUDE.md."""
    return tools.terminate_pod(tools.runtime(), confirm=confirm)


@mcp.tool()
def spend_report() -> dict:
    """Billing so far (pod compute + network volume storage) vs budget_usd.
    Informational only — the hard stop is RunPod's account spend limit."""
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
