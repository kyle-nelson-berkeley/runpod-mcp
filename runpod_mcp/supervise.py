"""supervise — one supervised RunPod job, from launch to a stopped pod.

Collapses the manual `launch -> poll job_status N times -> sync_logs ->
spend_report -> stop_pod` dance into a single call the agent fires once (as a
Claude Code `run_in_background` Bash task, wrapped in `caffeinate -i` via
`supervise.sh`) and is notified about on completion. See
`~/.claude/plans/runpod-supervise-job.md` (rev 2) for the full design
rationale; this module implements §4 (flow), §6 (CLI), §7 (safety
invariants).

This is a Mac-side background CLI, NOT an MCP tool — the stdio server must
never block for the 10-20 minutes a training run takes (runpod-mcp/CLAUDE.md:
"never add a blocking tool"). It reuses `runpod_mcp.tools.*` directly so it
inherits every guardrail (one-job guard, vehicle gates, driver floor) with
zero logic duplication.

TEST SEAM (read this before writing a test): every call into the tool surface
goes through the MODULE ATTRIBUTE — `tools.pod_status(rt)`,
`tools.launch_training(...)`, `tools.run_job(...)`, `tools.job_status(...)`,
`tools.sync_logs(...)`, `tools.spend_report(...)`, `tools.stop_pod(...)`,
`tools._conn_info(rt)` — via `from . import tools` + `tools.X(...)`, never
`from .tools import X`. Tests exercise this by monkeypatching attributes on
the `runpod_mcp.tools` module itself (e.g.
`monkeypatch.setattr(supervise.tools, "pod_status", fake)`), so a real
`Runtime`/API/SSH stack is never required for `supervise(rt, spec, ...)`
tests — only `rt.cfg` and `rt.ssh.rsync_pull` (the one call that bypasses
`tools.*`, per the plan: the job-dir pull is unconditional) are touched on
`rt` directly.

`supervise()` never talks to the real Keychain/network; only `main()` calls
`tools.runtime()` to build the real Runtime.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import guardrails, jobs, ssh, tools, training
from .config import REPO_ROOT, scrub

# The definitive "this call refused/failed cleanly, don't crash" set — five
# sibling RuntimeError subclasses, none subclassing another. Deliberately NOT
# bare RuntimeError/Exception: a genuine bug (KeyError, AttributeError, ...)
# must traceback loudly, never be silently masked into a fail-safe refusal.
REFUSE_ERRORS = (tools.ToolError, training.TrainingError, guardrails.GuardrailError,
                 jobs.JobError, ssh.SSHError)

TERMINAL_STATES = frozenset({"succeeded", "failed", "orphaned", "not_found"})


def _default_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobSpec:
    """mode ∈ {"training", "job"}. training uses vehicle/dr/seed/extra_args;
    job uses name/command/workdir/max_runtime_sec. The rest are common."""
    mode: str
    # --training
    vehicle: str | None = None
    dr: str | None = None
    seed: int = 1
    extra_args: str = ""
    # --job-name
    name: str | None = None
    command: str | None = None
    workdir: str = "/workspace"
    max_runtime_sec: int | None = None
    # common
    interval: int = 45
    max_wait: int | None = None
    backstop: int = 300
    no_stop: bool = False
    sync_subdir: str = "rsl_rl/warpauv_direct"
    summary_path: str | None = None


# ------------------------------------------------------------- launch helpers

def _dry_run(rt, spec: JobSpec) -> dict:
    if spec.mode == "training":
        return tools.launch_training(rt, vehicle=spec.vehicle, dr_level=spec.dr,
                                     seed=spec.seed, extra_args=spec.extra_args,
                                     dry_run=True)
    return tools.run_job(rt, name=spec.name, command=spec.command,
                         workdir=spec.workdir, max_runtime_sec=spec.max_runtime_sec,
                         dry_run=True)


def _launch(rt, spec: JobSpec) -> dict:
    # auto_stop is ALWAYS False here — supervise owns the stop (it must sync
    # first); auto_stop=True would stop the pod before sync_logs could run.
    if spec.mode == "training":
        return tools.launch_training(rt, vehicle=spec.vehicle, dr_level=spec.dr,
                                     seed=spec.seed, extra_args=spec.extra_args,
                                     auto_stop=False)
    return tools.run_job(rt, name=spec.name, command=spec.command,
                         workdir=spec.workdir, max_runtime_sec=spec.max_runtime_sec,
                         auto_stop=False)


# ---------------------------------------------------------------------- exit

def _determine_exit_code(*, final_state, force, errors, stop_escalated,
                         pod_may_still_be_running) -> int:
    if force or stop_escalated or pod_may_still_be_running or errors:
        return 1
    if final_state == "succeeded":
        return 0
    return 1   # failed / orphaned / not_found / unknown


def _default_summary_path(rt, job_id: str) -> Path:
    return REPO_ROOT / rt.cfg["local_log_dir"] / f"supervise-{job_id}.json"


# ------------------------------------------------------------------ refusal

def _refusal(spec: JobSpec, started_at: str, log, max_wait_sec, message: str) -> dict:
    """Gate/dry-run/launch refusal: no job_id, no poll loop, no stop. Exit 2.
    Refusal JSON always goes to stdout; the DEFAULT summary file is skipped
    (there's no job_id to name it after) but an EXPLICIT --summary-path is
    still honored."""
    log(f"[supervise] REFUSED: {message}")
    summary = {
        "job_id": None,
        "mode": spec.mode,
        "spec": asdict(spec),
        "max_wait_sec": max_wait_sec,
        "refused": True,
        "reason": message,
        "state": None,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "errors": [],
        "process_exit_code": 2,
    }
    if spec.summary_path:
        try:
            path = Path(spec.summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2))
            summary["summary_path"] = str(path)
        except Exception as exc:   # noqa: BLE001 — must never crash on the way out
            summary["errors"].append({"step": "summary_write", "error": scrub(str(exc))})
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------- capture + stop

def _capture_and_stop(rt, spec: JobSpec, job_id: str, last_known: dict | None,
                      force: bool, max_wait_sec: int, started_at: str, log) -> dict:
    """Best-effort, strictly ordered: job-dir pull -> analysis sync ->
    spend_report -> stop decision. Every step is wrapped in a BROAD
    `except Exception` — a capture failure must NEVER skip the stop."""
    errors: list[dict] = []
    pulled: dict = {}

    # (a) unconditional job-dir pull — captures stdout/exit even if
    # --sync-subdir is wrong/absent (closes the 2026-07-06 empty-output.csv
    # compute-and-discard failure mode).
    try:
        host, port = tools._conn_info(rt)
        remote_dir = f"/workspace/jobs/{job_id}/"
        local_dir = REPO_ROOT / rt.cfg["local_log_dir"] / "jobs" / job_id
        rsync_output = rt.ssh.rsync_pull(host, port, remote_dir, local_dir)
        pulled["job_dir"] = {"remote": remote_dir, "local": str(local_dir),
                             "rsync_output": scrub(rsync_output)[-1000:]}
    except Exception as exc:   # noqa: BLE001
        errors.append({"step": "job_dir_pull", "error": scrub(str(exc))})

    # (b) analysis sync — the curated subdir, unless explicitly skipped.
    if spec.sync_subdir == "none":
        pulled["analysis"] = {"skipped": True, "reason": "--sync-subdir none"}
    else:
        try:
            pulled["analysis"] = tools.sync_logs(rt, spec.sync_subdir)
        except Exception as exc:   # noqa: BLE001
            errors.append({"step": "sync_logs", "error": scrub(str(exc))})

    # (c) spend snapshot — informational, possibly lagging RunPod billing.
    spend = None
    try:
        spend = tools.spend_report(rt)
    except Exception as exc:   # noqa: BLE001
        errors.append({"step": "spend_report", "error": scrub(str(exc))})

    # (d) stop decision — exactly one of: skip (--no-stop), force (anomaly),
    # normal (with a single escalation-to-force retry on failure).
    stop_result = None
    stop_escalated = False
    pod_may_still_be_running = False
    if spec.no_stop:
        stop_result = {"skipped": True, "reason": "--no-stop"}
    elif force:
        try:
            stop_result = tools.stop_pod(rt, force=True)
        except Exception as exc:   # noqa: BLE001
            errors.append({"step": "stop_pod_force", "error": scrub(str(exc))})
            pod_may_still_be_running = True
    else:
        try:
            stop_result = tools.stop_pod(rt)
        except Exception as exc:   # noqa: BLE001
            errors.append({"step": "stop_pod", "error": scrub(str(exc))})
            stop_escalated = True
            try:
                stop_result = tools.stop_pod(rt, force=True)
            except Exception as exc2:   # noqa: BLE001
                errors.append({"step": "stop_pod_force_escalation",
                              "error": scrub(str(exc2))})
                pod_may_still_be_running = True

    final_state = last_known.get("state") if last_known else None
    exit_code = last_known.get("exit_code") if last_known else None
    latest_reward_line = last_known.get("latest_reward_line") if last_known else None

    process_exit_code = _determine_exit_code(
        final_state=final_state, force=force, errors=errors,
        stop_escalated=stop_escalated,
        pod_may_still_be_running=pod_may_still_be_running)

    summary = {
        "job_id": job_id,
        "mode": spec.mode,
        "spec": asdict(spec),
        "max_wait_sec": max_wait_sec,
        "state": final_state,
        "exit_code": exit_code,
        "force_stopped": force,
        "latest_reward_line": latest_reward_line,
        "pulled": pulled,
        "spend": spend,
        "spend_note": ("possibly-lagging snapshot — RunPod billing often "
                       "trails the just-finished job") if spend is not None else None,
        "stop": stop_result,
        "stop_escalated": stop_escalated,
        "pod_may_still_be_running": pod_may_still_be_running,
        "errors": errors,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "process_exit_code": process_exit_code,
    }

    summary_path = (Path(spec.summary_path) if spec.summary_path
                    else _default_summary_path(rt, job_id))
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
    except Exception as exc:   # noqa: BLE001 — a write failure must not crash us
        errors.append({"step": "summary_write", "error": scrub(str(exc))})
        summary["errors"] = errors
        if summary["process_exit_code"] == 0:
            summary["process_exit_code"] = 1

    print(json.dumps(summary, indent=2))
    log(f"[supervise] job_id={job_id} done state={final_state} "
       f"force_stopped={force} exit={summary['process_exit_code']}")
    return summary


# --------------------------------------------------------------------- core

def supervise(rt, spec: JobSpec, *, sleep=time.sleep, now=time.monotonic,
             log=_default_log) -> dict:
    started_at = _iso_now()

    # 1. GATE — pod must be RUNNING (this includes rejecting
    # "running_ssh_pending"). supervise never creates infrastructure.
    status = tools.pod_status(rt)
    if status.get("status") != "running":
        return _refusal(spec, started_at, log, None,
                        f"pod is not running (status={status.get('status')!r}) "
                        "— run ensure_pod first, then retry supervise")

    # 2. CAP DERIVATION — ALWAYS dry-run first (cheap $0 vehicle/dr
    # validation too), even when --max-wait was given explicitly.
    try:
        dry = _dry_run(rt, spec)
    except REFUSE_ERRORS as exc:
        return _refusal(spec, started_at, log, None,
                        f"launch dry-run refused: {scrub(str(exc))}")

    max_wait_sec = (spec.max_wait if spec.max_wait is not None
                    else int(dry["max_runtime_sec"]) + int(spec.backstop))

    # 3. LAUNCH — auto_stop=False always. A refusal here does NOT stop the
    # pod (the agent fixes-and-retries; a stop would force a ~5-min
    # ensure_pod re-create) and runs no poll loop.
    try:
        launched = _launch(rt, spec)
    except REFUSE_ERRORS as exc:
        return _refusal(spec, started_at, log, max_wait_sec,
                        f"launch refused: {scrub(str(exc))}")

    job_id = launched["job_id"]
    log(f"[supervise] launched job_id={job_id} mode={spec.mode} "
       f"max_wait_sec={max_wait_sec}")

    # 4. POLL LOOP — exactly two exits: terminal state, or deadline while
    # still non-terminal (ANOMALY -> force stop). A job_status call raising
    # any REFUSE_ERRORS is logged and treated as "state unknown, keep
    # polling" — the finite deadline still bounds the loop even if every
    # remaining poll errors.
    deadline = now() + max_wait_sec
    last_known: dict | None = None
    force = False
    while True:
        try:
            polled = tools.job_status(rt, job_id)
        except REFUSE_ERRORS as exc:
            log(f"[supervise] job_id={job_id} poll error (state unknown, "
               f"keep polling): {scrub(str(exc))}")
        else:
            last_known = polled
            state = polled.get("state")
            reward = polled.get("latest_reward_line")
            line = f"[supervise] job_id={job_id} state={state}"
            if reward:
                line += f" | {reward}"
            log(line)
            if state in TERMINAL_STATES:
                break
        # Bound the sleep by the remaining time to the deadline: a raw
        # sleep(spec.interval) would overshoot when interval > remaining,
        # billing the pod up to a full interval past --max-wait before the
        # next deadline check. This keeps --max-wait a HARD finite cap (one
        # final poll may land right at the deadline — fine).
        remaining = deadline - now()
        if remaining <= 0:
            force = True
            break
        sleep(min(spec.interval, remaining))

    # 5/6. CAPTURE THEN STOP, then durable summary.
    return _capture_and_stop(rt, spec, job_id, last_known, force, max_wait_sec,
                             started_at, log)


# --------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="supervise",
        description="Launch one job (training or generic), poll it to "
                    "completion, capture logs, and stop the pod — a single "
                    "supervised run fired once as a background task.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--training", metavar="VEHICLE",
                      help="e.g. curee, bluerov2 — validated by the launch "
                           "dry-run, never re-validated here")
    mode.add_argument("--job-name", metavar="NAME")

    p.add_argument("--dr", help="DR level (--training mode; required)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--extra-args", default="")

    p.add_argument("--command", help="--job-name mode; required")
    p.add_argument("--workdir", default="/workspace")
    p.add_argument("--max-runtime-sec", type=int, default=None)

    p.add_argument("--interval", type=int, default=45)
    p.add_argument("--max-wait", type=int, default=None)
    p.add_argument("--backstop", type=int, default=300)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--sync-subdir", default=None,
                   help="default rsl_rl/warpauv_direct in --training mode; "
                        "REQUIRED in --job-name mode ('none' skips the "
                        "analysis sync — the job-dir pull always happens)")
    p.add_argument("--summary-path", default=None)
    return p


def _spec_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> JobSpec:
    # Numeric hygiene (applies to both modes): a 0/negative --interval would
    # busy-spin the poll loop (and real time.sleep raises on a negative); a
    # non-positive --max-wait would make the finite cap non-positive; a
    # negative --backstop would push the derived cap below the job's own
    # timeout. Reject at argparse (exit 2) before any Runtime is built.
    if args.interval <= 0:
        parser.error("--interval must be a positive integer (seconds)")
    if args.max_wait is not None and args.max_wait <= 0:
        parser.error("--max-wait must be a positive integer (seconds)")
    if args.backstop < 0:
        parser.error("--backstop must be a non-negative integer (seconds)")

    if args.training:
        if not args.dr:
            parser.error("--training requires --dr")
        sync_subdir = (args.sync_subdir if args.sync_subdir is not None
                       else "rsl_rl/warpauv_direct")
        return JobSpec(mode="training", vehicle=args.training, dr=args.dr,
                       seed=args.seed, extra_args=args.extra_args,
                       interval=args.interval, max_wait=args.max_wait,
                       backstop=args.backstop, no_stop=args.no_stop,
                       sync_subdir=sync_subdir, summary_path=args.summary_path)

    if not args.command:
        parser.error("--job-name requires --command")
    if not args.sync_subdir:
        parser.error("--job-name requires --sync-subdir "
                     "(pass 'none' to skip the analysis sync)")
    return JobSpec(mode="job", name=args.job_name, command=args.command,
                   workdir=args.workdir, max_runtime_sec=args.max_runtime_sec,
                   interval=args.interval, max_wait=args.max_wait,
                   backstop=args.backstop, no_stop=args.no_stop,
                   sync_subdir=args.sync_subdir, summary_path=args.summary_path)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    spec = _spec_from_args(args, parser)
    summary = supervise(tools.runtime(), spec)
    return int(summary["process_exit_code"])


if __name__ == "__main__":
    sys.exit(main())
