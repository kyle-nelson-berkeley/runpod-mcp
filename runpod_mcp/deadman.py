"""deadman — an independent Mac-side fuse that stops the RunPod pod after a
fixed window, even if the process supervising a training run dies.

CONTEXT: a prior run (Experiment F) lost ~$4.5 when the process running
supervise.sh died and the pod idled ~7.2h before anyone noticed. The pod-side
idle watchdog (armed by ensure_pod) is the primary backstop, but it depends on
SSH/job-status reachability from the pod's own perspective; this deadman is a
SEPARATE, independent Mac-side fuse with no dependency on any supervising
process staying alive — arm it once at launch, and it fires on its own clock,
sleeping ~N hours then stopping the pod (with retries), unless cancelled.

USAGE:
  ./deadman.sh arm --hours 3.0 &     # background it (re-arm before it fires
                                      # to extend: `cancel` then a fresh `arm`)
  ./deadman.sh status                 # armed / LOST / last outcome — no network
  ./deadman.sh cancel                 # disarm cleanly before a normal stop_pod
  # Re-arm = cancel, then a FRESH `arm` in the background. There is no
  # compound "rearm" subcommand on purpose — the two-step keeps the state
  # machine trivial and the new fuse window explicit.

This is a Mac-side background CLI, NOT an MCP tool — invoke it directly,
never through .mcp.json. The stdio server's 14-tool surface is untouched;
server.py is not part of this deliverable.

TEST SEAM (identical to supervise.py — read this before writing a test):
every call into the tool surface goes through the MODULE ATTRIBUTE —
`tools.stop_pod(rt)` via `from . import tools`, never `from .tools import
stop_pod`; likewise `config.fetch_api_key()` / `config.scrub(...)` /
`config.REPO_ROOT` via `from . import config`, never `from .config import
X`. Tests exercise this by monkeypatching module attributes (e.g.
`monkeypatch.setattr(deadman.tools, "stop_pod", fake)`,
`monkeypatch.setattr(deadman.config, "fetch_api_key", fake)`), so a real
Runtime/API/SSH stack — or the real macOS Keychain — is never required for
any test in this module.

LAZY RUNTIME: `arm()` / `run_armed()` never build a Runtime themselves — they
take `rt_factory`, a zero-arg callable, and invoke it only once firing
actually begins (hours after arm time). Tests inject a fake factory
(`lambda: object()` is enough when `tools.stop_pod` is faked too); only
`main()` wires the real one (`tools.runtime`).

**NOT COVERED BY THESE UNIT TESTS: the LIVE stop path.** Every test in
`tests/test_deadman.py` mocks `tools.stop_pod` / `config.fetch_api_key` —
none of them exercise a real RunPod API call, real SSH, or the real
Keychain. The claim "`./deadman.sh arm --hours 3.0 &` really stops a real pod
N hours later" is verified only by hand, in the launch session — see the
runbook handoff for that check.
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, tools

# Statuses tools.stop_pod() can return without raising — all three mean the
# pod is confirmed not billing GPU time anymore (see tools.py::stop_pod).
_SUCCESS_STATUSES = frozenset({"stopped", "already_stopped", "no_pod"})

# How often the sleep-before-fire loop wakes to check for a cancel request.
# Bounded so a real SIGTERM is noticed promptly without busy-spinning; in
# FakeClock-driven tests this only affects iteration COUNT, never wall time.
_CANCEL_POLL_SEC = 30.0


class ArmRefused(RuntimeError):
    """arm() refused before the fuse was armed (exit 2) — a live deadman
    already holds the pid file, a pid-file create raced (TOCTOU), or the
    Keychain key probe failed. No pod interaction happened."""


def _default_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArmSpec:
    hours: float = 3.0
    retries: int = 5
    spacing_sec: int = 300
    pid_file: Path | None = None
    summary_path: Path | None = None
    probe_key: bool = True


# --------------------------------------------------------------- pid liveness

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just owned by someone else — treat as alive
    except OSError:
        return False
    return True


def _pid_cmdline(pid: int, run=subprocess.run) -> str:
    """macOS idiom: `ps -p <pid> -o command=`. Empty string if ps fails or
    the pid is gone (never raises — a probe, not an assertion)."""
    try:
        proc = run(["ps", "-p", str(pid), "-o", "command="],
                  capture_output=True, text=True, timeout=5)
    except Exception:   # noqa: BLE001 — a probe must never crash the caller
        return ""
    return (proc.stdout or "").strip()


def _is_deadman_process(pid: int | None) -> bool:
    """Alive AND its cmdline names this module — the PID-reuse-hazard guard.
    A dead pid, or a live pid some unrelated process now owns, is NOT a
    deadman — never signal it, never let it block a re-arm."""
    if pid is None:
        return False
    return _pid_alive(pid) and "runpod_mcp.deadman" in _pid_cmdline(pid)


def _read_pid_file(pid_file: Path) -> dict:
    try:
        return json.loads(pid_file.read_text())
    except Exception:   # noqa: BLE001 — a corrupt pid file is stale, not fatal
        return {}


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------------------- default paths

def _default_pid_file() -> Path:
    return config.REPO_ROOT / "logs" / "pod" / "deadman.pid"


def _iso_to_stamp(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y%m%d-%H%M%S")


def _default_summary_path(armed_at_iso: str) -> Path:
    return config.REPO_ROOT / "logs" / "pod" / f"deadman-{_iso_to_stamp(armed_at_iso)}.json"


def _default_summary_glob() -> str:
    return str(config.REPO_ROOT / "logs" / "pod" / "deadman-*.json")


def _add_hours_iso(iso: str, hours: float) -> str:
    return (datetime.fromisoformat(iso) + timedelta(hours=hours)).isoformat()


# ------------------------------------------------------------------- summary

def _base_summary(spec: ArmSpec, armed_at_iso: str, fire_at_iso: str, pid: int) -> dict:
    return {
        "armed_at": armed_at_iso,
        "fire_at": fire_at_iso,
        "hours": spec.hours,
        "retries": spec.retries,
        "spacing_sec": spec.spacing_sec,
        "pid": pid,
    }


def _write_summary(summary_path: Path, summary: dict, log) -> None:
    """Durable recovery contract. Wrapped so a write failure NEVER masks the
    real stop outcome already recorded in `summary` (mirrors supervise's
    summary_write handling) — on failure we log loudly (including the
    outcome, so it's at least visible in the session's stderr) and return;
    the caller's return value / exit code are unaffected."""
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
    except Exception as exc:   # noqa: BLE001 — must never crash on the way out
        log(f"[deadman] WARNING: failed to write summary to {summary_path}: "
           f"{config.scrub(str(exc))} — outcome was {summary.get('outcome')!r}, "
           f"stop_status={summary.get('stop_status')!r} (NOT lost, just not "
           "persisted to this path)")


# ------------------------------------------------------------- pid-file prep

def _reap_stale_pid_file(pid_file: Path, log) -> None:
    """Raises ArmRefused if the pid file names a currently-live deadman;
    otherwise removes it (loudly logged — a silent reap would hide a bug)."""
    if not pid_file.exists():
        return
    data = _read_pid_file(pid_file)
    pid = data.get("pid")
    if _is_deadman_process(pid):
        raise ArmRefused(
            f"a deadman process (pid={pid}) already holds {pid_file} — "
            "cancel it first (`deadman.sh cancel`), then re-arm")
    log(f"[deadman] reaping stale pid file {pid_file} (pid={pid!r} is not a "
       "live deadman process)")
    _remove(pid_file)


def _create_pid_file(pid_file: Path, pid: int, armed_at_iso: str, fire_at_iso: str) -> None:
    """O_EXCL create — the double-arm TOCTOU guard: if another arm() raced us
    between the reap check and here, this raises and we refuse cleanly."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(pid_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ArmRefused(
            f"pid file {pid_file} appeared concurrently (double-arm race) — "
            "refusing; check for another deadman and retry") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": pid, "armed_at": armed_at_iso, "fire_at": fire_at_iso}, fh)


# --------------------------------------------------------------------- core

def run_armed(spec: ArmSpec, *, pid_file: Path, summary_path: Path,
              armed_at_iso: str, fire_at_iso: str, fire_at_mono: float, pid: int,
              rt_factory, sleep=time.sleep, now=time.monotonic, iso_now=_iso_now,
              log=_default_log, cancel_requested=lambda: False,
              on_fire_start=lambda: None) -> dict:
    """Sleep-then-fire core, called by `arm()` after the pid file exists and
    the key probe (if any) has passed. Two exits: CANCEL (sleep phase,
    `cancel_requested()` observed True) or FIRE (deadline reached) -> up to
    `spec.retries` stop cycles. `on_fire_start()` is invoked exactly once, the
    instant firing begins and BEFORE the first stop attempt — the caller uses
    it to flip real SIGTERM to SIG_IGN so a late cancel can never overwrite a
    fired outcome (cancel_requested() is never consulted again after this
    point, by construction)."""
    while True:
        if cancel_requested():
            _remove(pid_file)
            summary = _base_summary(spec, armed_at_iso, fire_at_iso, pid)
            summary.update(outcome="cancelled", stop_status=None, attempts=[],
                          ended_at=iso_now(), pod_may_still_be_running=False)
            _write_summary(summary_path, summary, log)
            summary["exit_code"] = 0
            log(f"[deadman] cancelled during sleep at {summary['ended_at']}")
            return summary
        remaining = fire_at_mono - now()
        if remaining <= 0:
            break
        sleep(min(_CANCEL_POLL_SEC, remaining))

    # FIRE begins. From here, a cancel_requested() flip (e.g. a racing
    # SIGTERM) is never re-checked — the stop sequence owns the summary.
    on_fire_start()
    log(f"[deadman] fuse fired at {iso_now()} — starting stop sequence "
       f"(retries={spec.retries}, spacing_sec={spec.spacing_sec})")

    attempts: list[dict] = []
    outcome = "stop_failed"
    stop_status = None
    for cycle in range(1, spec.retries + 1):
        attempt = {"cycle": cycle, "at": iso_now()}
        try:
            rt = rt_factory()
        except Exception as exc:   # noqa: BLE001 — a failed attempt, not a crash
            attempt["error"] = f"runtime construction failed: {config.scrub(str(exc))}"
            attempts.append(attempt)
            log(f"[deadman] cycle {cycle}/{spec.retries}: {attempt['error']}")
            if cycle < spec.retries:
                sleep(spec.spacing_sec)
            continue

        result = None
        try:
            result = tools.stop_pod(rt)
            attempt["graceful"] = result
        except Exception as exc:   # noqa: BLE001 — escalate within the same cycle
            attempt["graceful_error"] = config.scrub(str(exc))
            log(f"[deadman] cycle {cycle}/{spec.retries}: graceful stop_pod "
               f"failed ({attempt['graceful_error']}) — escalating to force")
            try:
                result = tools.stop_pod(rt, force=True)
                attempt["forced"] = result
            except Exception as exc2:   # noqa: BLE001
                attempt["forced_error"] = config.scrub(str(exc2))
                result = None

        attempts.append(attempt)
        status_val = result.get("status") if isinstance(result, dict) else None
        if status_val in _SUCCESS_STATUSES:
            outcome, stop_status = "stopped", status_val
            log(f"[deadman] cycle {cycle}/{spec.retries}: stop succeeded "
               f"(status={status_val})")
            break
        log(f"[deadman] cycle {cycle}/{spec.retries}: stop attempt failed "
           f"(result={result!r})")
        if cycle < spec.retries:
            sleep(spec.spacing_sec)

    summary = _base_summary(spec, armed_at_iso, fire_at_iso, pid)
    summary.update(outcome=outcome, stop_status=stop_status, attempts=attempts,
                   ended_at=iso_now(), pod_may_still_be_running=(outcome != "stopped"))
    _remove(pid_file)
    _write_summary(summary_path, summary, log)
    summary["exit_code"] = 0 if outcome == "stopped" else 1
    return summary


def arm(spec: ArmSpec, *, rt_factory, sleep=time.sleep, now=time.monotonic,
       iso_now=_iso_now, log=_default_log, cancel_requested=lambda: False,
       on_fire_start=lambda: None) -> dict:
    """Full arm lifecycle: pid-file discipline -> (optional) Keychain key
    probe -> `run_armed()`. Raises ArmRefused (exit 2) for anything that goes
    wrong BEFORE the fuse is actually armed; nothing here talks to
    `tools.stop_pod` — that only ever happens inside `run_armed`'s fire
    sequence, hours later."""
    pid_file = spec.pid_file or _default_pid_file()
    armed_at_iso = iso_now()
    armed_at_mono = now()
    fire_at_mono = armed_at_mono + spec.hours * 3600.0
    fire_at_iso = _add_hours_iso(armed_at_iso, spec.hours)
    summary_path = spec.summary_path or _default_summary_path(armed_at_iso)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_pid_file(pid_file, log)

    pid = os.getpid()
    _create_pid_file(pid_file, pid, armed_at_iso, fire_at_iso)

    if spec.probe_key:
        try:
            config.fetch_api_key()   # fetch-and-DISCARD — never stored, never logged
        except Exception as exc:   # noqa: BLE001 — any probe failure refuses arm
            _remove(pid_file)
            raise ArmRefused(
                f"Keychain key probe failed: {config.scrub(str(exc))} — "
                "refusing to arm (fix the Keychain entry, or pass "
                "--no-probe-key for offline testing)") from None

    log(f"[deadman] armed pid={pid} armed_at={armed_at_iso} fire_at={fire_at_iso} "
       f"(hours={spec.hours}, retries={spec.retries}, spacing_sec={spec.spacing_sec})")

    return run_armed(spec, pid_file=pid_file, summary_path=summary_path,
                     armed_at_iso=armed_at_iso, fire_at_iso=fire_at_iso,
                     fire_at_mono=fire_at_mono, pid=pid, rt_factory=rt_factory,
                     sleep=sleep, now=now, iso_now=iso_now, log=log,
                     cancel_requested=cancel_requested, on_fire_start=on_fire_start)


# ------------------------------------------------------------------- cancel

def cancel(*, pid_file: Path | None = None, log=_default_log) -> dict:
    """SIGTERM the pid-file process, but ONLY after verifying it's actually a
    live deadman (PID-reuse hazard) — never signal an unverified pid. No pid
    file, or a dead/mismatched one, is an idempotent no-op success (and the
    stale file is reaped)."""
    pid_file = pid_file or _default_pid_file()
    if not pid_file.exists():
        return {"outcome": "not_armed", "message": "no pid file — nothing to cancel",
                "exit_code": 0}
    data = _read_pid_file(pid_file)
    pid = data.get("pid")
    if not _is_deadman_process(pid):
        log(f"[deadman] cancel: pid file {pid_file} (pid={pid!r}) is not a "
           "live deadman — reaping, nothing to signal")
        _remove(pid_file)
        return {"outcome": "not_armed",
                "message": "stale pid file reaped; nothing to cancel", "exit_code": 0}
    os.kill(pid, signal.SIGTERM)
    log(f"[deadman] cancel: SIGTERM sent to pid={pid}")
    return {"outcome": "cancel_requested", "pid": pid, "exit_code": 0}


# ------------------------------------------------------------------- status

def _find_latest_summary(glob_pattern: str) -> dict | None:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        return None
    try:
        return json.loads(Path(paths[-1]).read_text())
    except Exception:   # noqa: BLE001 — a corrupt summary is "no summary"
        return None


def _remaining_minutes(fire_at_iso: str | None, now_iso: str) -> float | None:
    if not fire_at_iso:
        return None
    try:
        fire_dt = datetime.fromisoformat(fire_at_iso)
        now_dt = datetime.fromisoformat(now_iso)
    except Exception:   # noqa: BLE001
        return None
    return round((fire_dt - now_dt).total_seconds() / 60.0, 1)


def status(*, pid_file: Path | None = None, summary_glob: str | None = None,
          now=_iso_now, log=_default_log) -> dict:
    """NO network, ever. armed (pid file names a verified-live process) |
    LOST (pid file exists but the process is dead/mismatched — the pod may
    still be running, exit 1 so scripts can alert) | last summary's outcome |
    not_armed (nothing on disk). A LOST fuse must never report as armed —
    that false comfort is exactly the failure mode this tool exists to kill."""
    pid_file = pid_file or _default_pid_file()
    if pid_file.exists():
        data = _read_pid_file(pid_file)
        pid = data.get("pid")
        fire_at = data.get("fire_at")
        if _is_deadman_process(pid):
            return {"state": "armed", "pid": pid, "fire_at": fire_at,
                    "remaining_minutes": _remaining_minutes(fire_at, now()),
                    "exit_code": 0}
        log(f"[deadman] status: pid file {pid_file} (pid={pid!r}) is LOST — "
           "process is dead or mismatched")
        return {"state": "LOST", "pid": pid, "fire_at": fire_at,
                "message": "deadman process died; the pod may still be running; "
                           "check/stop it manually", "exit_code": 1}

    summary = _find_latest_summary(summary_glob or _default_summary_glob())
    if summary is None:
        return {"state": "not_armed", "message": "no pid file, no prior summary",
                "exit_code": 0}
    return {"state": summary.get("outcome", "unknown"), "summary": summary,
            "exit_code": 0}


# --------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deadman",
        description="Independent Mac-side fuse: arm it once, it sleeps ~N "
                    "hours then stops the pod (with retries) unless "
                    "cancelled — a money-safety backstop that does not "
                    "depend on any supervising process staying alive.")
    sub = p.add_subparsers(dest="subcommand", required=True)

    arm_p = sub.add_parser("arm", help="arm the fuse (background it)")
    arm_p.add_argument("--hours", type=float, default=3.0)
    arm_p.add_argument("--retries", type=int, default=5)
    arm_p.add_argument("--spacing-sec", type=int, default=300)
    arm_p.add_argument("--pid-file", default=None)
    arm_p.add_argument("--summary-path", default=None)
    arm_p.add_argument("--no-probe-key", action="store_true",
                      help="skip the arm-time Keychain fetch-and-discard "
                           "probe (for tests/offline use)")

    cancel_p = sub.add_parser("cancel", help="disarm a running fuse")
    cancel_p.add_argument("--pid-file", default=None)

    status_p = sub.add_parser("status", help="report armed/LOST/last outcome — no network")
    status_p.add_argument("--pid-file", default=None)
    status_p.add_argument("--summary-glob", default=None)

    return p


def _validate_arm_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.hours <= 0:
        parser.error("--hours must be > 0")
    if args.retries < 1:
        parser.error("--retries must be >= 1")
    if args.spacing_sec < 0:
        parser.error("--spacing-sec must be >= 0 (seconds)")


def _path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _main_arm(spec: ArmSpec) -> int:
    """Wires the REAL SIGTERM handler (active only during the sleep phase)
    and the REAL runtime factory, then delegates to `arm()`. This is the
    only place in the module that touches `signal.signal` or `tools.runtime`."""
    cancel_flag = {"requested": False}

    def _on_sigterm(signum, frame):
        cancel_flag["requested"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    def _disarm_signal() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        summary = arm(spec, rt_factory=tools.runtime, sleep=time.sleep,
                     now=time.monotonic, iso_now=_iso_now, log=_default_log,
                     cancel_requested=lambda: cancel_flag["requested"],
                     on_fire_start=_disarm_signal)
    except ArmRefused as exc:
        result = {"outcome": "refused", "reason": str(exc), "exit_code": 2}
        print(json.dumps(result, indent=2))
        return 2

    print(json.dumps(summary, indent=2))
    return int(summary["exit_code"])


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "arm":
        _validate_arm_args(args, parser)
        spec = ArmSpec(hours=args.hours, retries=args.retries,
                       spacing_sec=args.spacing_sec,
                       pid_file=_path_or_none(args.pid_file),
                       summary_path=_path_or_none(args.summary_path),
                       probe_key=not args.no_probe_key)
        return _main_arm(spec)

    if args.subcommand == "cancel":
        result = cancel(pid_file=_path_or_none(args.pid_file))
        print(json.dumps(result, indent=2))
        return int(result["exit_code"])

    if args.subcommand == "status":
        result = status(pid_file=_path_or_none(args.pid_file),
                        summary_glob=args.summary_glob)
        print(json.dumps(result, indent=2))
        return int(result["exit_code"])

    parser.error(f"unknown subcommand {args.subcommand!r}")   # pragma: no cover
    return 2   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
