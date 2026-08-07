"""deadman — an independent Mac-side fuse that stops the RunPod pod after a
fixed window, even if the process supervising a training run dies.

CONTEXT: a prior run (campaign 007 · 5 s horizon) lost ~$4.5 when the
process running supervise.sh died and the pod idled ~7.2h before anyone
noticed. The pod-side idle watchdog (armed by ensure_pod) is the primary
backstop, but it depends on
SSH/job-status reachability from the pod's own perspective; this deadman is a
SEPARATE, independent Mac-side fuse with no dependency on any supervising
process staying alive — arm it once at launch, and it fires on its own clock,
sleeping ~N hours then stopping the pod (with retries), unless cancelled.

USAGE:
  ./deadman.sh arm --vehicle hippocampus --hours 3.0 &
                                      # background it (re-arm before it fires
                                      # to extend: `cancel` then a fresh `arm`)
  ./deadman.sh status                 # ALL vehicles: armed / LOST / last
                                      # outcome — no network
  ./deadman.sh status --vehicle bluerov2      # just one (flat shape)
  ./deadman.sh cancel --vehicle hippocampus   # disarm before a normal stop_pod
  # Re-arm = cancel, then a FRESH `arm` in the background. There is no
  # compound "rearm" subcommand on purpose — the two-step keeps the state
  # machine trivial and the new fuse window explicit.

VEHICLES (two pods, two fuses). `--vehicle hippocampus|bluerov2` selects which
pod a fuse guards; hippocampus IS the pre-existing lts-replication pod.
`arm` and `cancel` REQUIRE it explicitly — no default — because a fuse aimed at
the wrong pod is silently wrong for HOURS (it reports healthy while the real
pod bills). `status` takes it OPTIONALLY: with no `--vehicle` it reports EVERY
declared vehicle and returns the WORST result, so a LOST bluerov2 fuse can
never hide behind a quiet hippocampus exit-0.

Artifacts are separated by DIRECTORY, never by filename: each vehicle's
`local_log_dir` (from pod_defaults.yaml) holds the same `deadman.pid` /
`deadman-<stamp>.json` names. hippocampus therefore resolves byte-identically
to the pre-refactor paths (`logs/pod/...`) and keeps seeing every summary
written before vehicles existed; bluerov2 nests at `logs/pod/bluerov2/`, and
since a glob `*` never crosses `/`, neither vehicle's glob sees the other's.

This is a Mac-side background CLI, NOT an MCP tool — invoke it directly,
never through .mcp.json. The stdio server's 14-tool surface is untouched;
server.py is not part of this deliverable.

TEST SEAM (identical to supervise.py — read this before writing a test):
every call into the tool surface goes through the MODULE ATTRIBUTE —
`tools.stop_pod(rt)` via `from . import tools`, never `from .tools import
stop_pod`; likewise `config.fetch_api_key()` / `config.scrub(...)` /
`config.REPO_ROOT` / `config.load_defaults()` / `config.merged_vehicle_cfg()`
via `from . import config`, never `from .config import X`. Tests exercise this
by monkeypatching module attributes (e.g.
`monkeypatch.setattr(deadman.tools, "stop_pod", fake)`,
`monkeypatch.setattr(deadman.config, "fetch_api_key", fake)`,
`monkeypatch.setattr(deadman.config, "load_defaults", lambda: fake_raw)`), so a
real Runtime/API/SSH stack — or the real macOS Keychain — is never required for
any test in this module. Per-vehicle path resolution is CONFIG-ONLY: a local
YAML read, no Keychain and no network, so `status`'s "NO network, ever"
contract is untouched.

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
    """`vehicle` picks WHICH POD this fuse guards (and therefore which
    artifact directory it uses). The dataclass default is hippocampus — the
    pre-existing lts-replication pod — so a spec built in code keeps the
    historical behavior; the CLI deliberately does NOT default it (see
    `_require_vehicle`)."""
    hours: float = 3.0
    retries: int = 5
    spacing_sec: int = 300
    pid_file: Path | None = None
    summary_path: Path | None = None
    probe_key: bool = True
    vehicle: str = "hippocampus"


# --------------------------------------------------------------- pid liveness

def _pid_alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except (TypeError, ValueError):
        return False   # non-int pid (corrupt pid file) — provably not alive
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
    """Defensive parse: malformed JSON, a non-dict payload, or a non-integer
    `pid` value must never crash arm/cancel/status — a corrupt pid file is
    STALE (pid normalized to None -> _is_deadman_process False -> arm/cancel
    reap it, status reports LOST), never fatal."""
    try:
        data = json.loads(pid_file.read_text())
    except Exception:   # noqa: BLE001 — a corrupt pid file is stale, not fatal
        return {}
    if not isinstance(data, dict):
        return {}
    try:
        data["pid"] = int(data.get("pid"))
    except (TypeError, ValueError):
        data["pid"] = None
    return data


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------- per-vehicle default paths
#
# THE VEHICLE IS THE DIRECTORY, uniformly — filenames never change. Resolution
# is CONFIG-ONLY (a local YAML read: no Keychain, no network), so `status` keeps
# its "NO network, ever" contract. Consequences worth stating out loud:
#   * hippocampus resolves to REPO_ROOT/logs/pod/... — byte-identical to the
#     pre-vehicles paths, so every summary written before this refactor stays
#     visible to hippocampus `status`;
#   * bluerov2 nests at REPO_ROOT/logs/pod/bluerov2/, and since a glob `*`
#     never crosses `/`, neither vehicle's summary glob can see the other's.

def _vehicle_cfg(vehicle: str) -> dict:
    """The merged (flat) per-vehicle config. Raises ConfigError for an unknown
    vehicle or an unreadable pod_defaults.yaml — callers decide what that
    means (arm refuses; status reports a clean error)."""
    return config.merged_vehicle_cfg(config.load_defaults(), vehicle)


def _vehicle_log_dir(vehicle: str) -> Path:
    return config.REPO_ROOT / _vehicle_cfg(vehicle)["local_log_dir"]


def _pod_name_for(vehicle: str) -> str | None:
    """Best-effort, for the RECORD only (never a gate): a config read failure
    here must not crash the fire path hours after arm already validated it."""
    try:
        return _vehicle_cfg(vehicle)["pod_name"]
    except Exception:   # noqa: BLE001 — a record field, not an assertion
        return None


def _default_pid_file(vehicle: str) -> Path:
    return _vehicle_log_dir(vehicle) / "deadman.pid"


def _iso_to_stamp(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y%m%d-%H%M%S")


def _default_summary_path(vehicle: str, armed_at_iso: str) -> Path:
    return _vehicle_log_dir(vehicle) / f"deadman-{_iso_to_stamp(armed_at_iso)}.json"


def _default_summary_glob(vehicle: str) -> str:
    return str(_vehicle_log_dir(vehicle) / "deadman-*.json")


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
        # WHICH pod this fuse was aimed at — a mis-targeted fuse is otherwise
        # indistinguishable from a healthy one until the bill arrives.
        "vehicle": spec.vehicle,
        "pod_name": _pod_name_for(spec.vehicle),
    }


def _write_summary(summary_path: Path, summary: dict, log) -> bool:
    """Durable recovery contract. Wrapped so a write failure NEVER masks the
    real stop outcome already recorded in `summary` (mirrors supervise's
    summary_write handling) — on failure we log loudly (including the
    outcome, so it's at least visible in the session's stderr) and return
    False; the caller's return value / exit code are unaffected. run_armed
    uses the return value for its fail-safe ordering: the pid file may be
    removed only after the summary is durably on disk."""
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
        return True
    except Exception as exc:   # noqa: BLE001 — must never crash on the way out
        log(f"[deadman] WARNING: failed to write summary to {summary_path}: "
           f"{config.scrub(str(exc))} — outcome was {summary.get('outcome')!r}, "
           f"stop_status={summary.get('stop_status')!r} (NOT lost, just not "
           "persisted to this path)")
        return False


def _write_refusal_summary(spec: ArmSpec, summary_path: Path, armed_at_iso: str,
                           fire_at_iso: str, pid: int, reason: str, iso_now,
                           log) -> None:
    """Durable record of a post-pid-file-creation refusal (e.g. the key-probe
    failure): without it, a later `status` would report plain not_armed and
    the refusal reason would exist only in the (possibly lost) stdout of the
    backgrounded arm. `reason` must arrive already scrubbed. Same
    never-crash-on-the-way-out discipline as every other summary write.

    Deliberately NOT called for pre-create refusals (another live deadman
    holds the pid file, or the O_EXCL TOCTOU race): there the OTHER fuse's
    pid file is the authoritative state, and a stamped refusal file written
    now would sort lexicographically AFTER that fuse's own summary (named by
    its earlier armed_at), masking its eventual stopped/stop_failed outcome
    from `status` — the exact false comfort this tool exists to kill."""
    summary = _base_summary(spec, armed_at_iso, fire_at_iso, pid)
    summary.update(outcome="refused", reason=reason, stop_status=None,
                   attempts=[], ended_at=iso_now(),
                   pod_may_still_be_running=False, exit_code=2)
    _write_summary(summary_path, summary, log)


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


def _create_pid_file(pid_file: Path, pid: int, armed_at_iso: str, fire_at_iso: str,
                     vehicle: str | None = None, pod_name: str | None = None) -> None:
    """O_EXCL create — the double-arm TOCTOU guard: if another arm() raced us
    between the reap check and here, this raises and we refuse cleanly. The
    payload records vehicle/pod_name so a live fuse's TARGET is recoverable
    from disk alone (the arming session's stdout may be long gone)."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(pid_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ArmRefused(
            f"pid file {pid_file} appeared concurrently (double-arm race) — "
            "refusing; check for another deadman and retry") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": pid, "armed_at": armed_at_iso, "fire_at": fire_at_iso,
                   "vehicle": vehicle, "pod_name": pod_name}, fh)


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
            summary = _base_summary(spec, armed_at_iso, fire_at_iso, pid)
            # exit_code is set BEFORE the write so the PERSISTED summary
            # carries it too (repo convention: supervise persists
            # process_exit_code in its summary file).
            summary.update(outcome="cancelled", stop_status=None, attempts=[],
                          ended_at=iso_now(), pod_may_still_be_running=False,
                          exit_code=0)
            # Fail-safe ordering: summary FIRST, pid file second. A crash or
            # write failure between the two leaves the pid file in place, so
            # a later `status` reads LOST (exit 1, check manually) — never
            # not_armed or an older success.
            if _write_summary(summary_path, summary, log):
                _remove(pid_file)
            else:
                log("[deadman] cancel summary write failed — leaving the pid "
                    "file in place so status degrades to LOST, not silence")
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
    # exit_code is set BEFORE the write so the PERSISTED summary carries it
    # too (repo convention: supervise persists process_exit_code).
    summary.update(outcome=outcome, stop_status=stop_status, attempts=attempts,
                   ended_at=iso_now(), pod_may_still_be_running=(outcome != "stopped"),
                   exit_code=0 if outcome == "stopped" else 1)
    # Fail-safe ordering (applies to BOTH stopped and stop_failed): summary
    # FIRST, pid file second. If the process dies or the write fails right
    # here, the pid file remains and a later `status` reads LOST (exit 1,
    # check manually) — the desired degraded state; remove-then-write would
    # instead report not_armed or an older success in the exact
    # pod-may-still-be-billing scenario.
    if _write_summary(summary_path, summary, log):
        _remove(pid_file)
    else:
        log("[deadman] fire summary write failed — leaving the pid file in "
            "place so a later status reads LOST (fail-safe), not not_armed")
    return summary


def arm(spec: ArmSpec, *, rt_factory, sleep=time.sleep, now=time.monotonic,
       iso_now=_iso_now, log=_default_log, cancel_requested=lambda: False,
       on_fire_start=lambda: None) -> dict:
    """Full arm lifecycle: vehicle resolution -> pid-file discipline ->
    (optional) Keychain key probe -> `run_armed()`. Raises ArmRefused (exit 2)
    for anything that goes wrong BEFORE the fuse is actually armed; nothing
    here talks to `tools.stop_pod` — that only ever happens inside
    `run_armed`'s fire sequence, hours later."""
    # Resolve the vehicle FIRST and fail fast: if we cannot say which pod this
    # fuse would stop, arming it is worse than not arming it — an unaimed fuse
    # reports "armed" for hours while nothing is actually guarded.
    try:
        cfg = _vehicle_cfg(spec.vehicle)
    except Exception as exc:   # noqa: BLE001 — refusal, not a traceback
        raise ArmRefused(
            f"could not resolve vehicle {spec.vehicle!r} from pod_defaults.yaml: "
            f"{config.scrub(str(exc))} — refusing to arm (a fuse that cannot "
            "name its pod cannot guard it)") from None
    pod_name = cfg["pod_name"]
    artifact_dir = config.REPO_ROOT / cfg["local_log_dir"]

    pid_file = spec.pid_file or (artifact_dir / "deadman.pid")
    armed_at_iso = iso_now()
    armed_at_mono = now()
    fire_at_mono = armed_at_mono + spec.hours * 3600.0
    fire_at_iso = _add_hours_iso(armed_at_iso, spec.hours)
    summary_path = spec.summary_path or (
        artifact_dir / f"deadman-{_iso_to_stamp(armed_at_iso)}.json")

    log(f"[deadman] arming vehicle={spec.vehicle} pod={pod_name} "
       f"artifacts={artifact_dir}")

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_pid_file(pid_file, log)

    pid = os.getpid()
    _create_pid_file(pid_file, pid, armed_at_iso, fire_at_iso,
                     vehicle=spec.vehicle, pod_name=pod_name)

    if spec.probe_key:
        try:
            config.fetch_api_key()   # fetch-and-DISCARD — never stored, never logged
        except Exception as exc:   # noqa: BLE001 — any probe failure refuses arm
            _remove(pid_file)
            reason = (f"Keychain key probe failed: {config.scrub(str(exc))} — "
                      "refusing to arm (fix the Keychain entry, or pass "
                      "--no-probe-key for offline testing)")
            # Durable refusal record — the backgrounded arm's stdout may be
            # lost, and status must be able to surface WHY the fuse never armed.
            _write_refusal_summary(spec, summary_path, armed_at_iso, fire_at_iso,
                                   pid, reason, iso_now, log)
            raise ArmRefused(reason) from None

    log(f"[deadman] armed pid={pid} armed_at={armed_at_iso} fire_at={fire_at_iso} "
       f"(hours={spec.hours}, retries={spec.retries}, spacing_sec={spec.spacing_sec})")

    return run_armed(spec, pid_file=pid_file, summary_path=summary_path,
                     armed_at_iso=armed_at_iso, fire_at_iso=fire_at_iso,
                     fire_at_mono=fire_at_mono, pid=pid, rt_factory=rt_factory,
                     sleep=sleep, now=now, iso_now=iso_now, log=log,
                     cancel_requested=cancel_requested, on_fire_start=on_fire_start)


# ------------------------------------------------------------------- cancel

def cancel(*, vehicle: str = "hippocampus", pid_file: Path | None = None,
           log=_default_log) -> dict:
    """SIGTERM the pid-file process, but ONLY after verifying it's actually a
    live deadman (PID-reuse hazard) — never signal an unverified pid. No pid
    file, or a dead/mismatched one, is an idempotent no-op success (and the
    stale file is reaped).

    `vehicle` selects WHICH fuse (i.e. which artifact directory). The CLI
    requires it explicitly; the kwarg default keeps the historical
    hippocampus behavior for in-code callers."""
    if pid_file is None:
        try:
            cfg = _vehicle_cfg(vehicle)
        except Exception as exc:   # noqa: BLE001 — clean error, not a traceback
            message = (f"could not resolve vehicle {vehicle!r} from "
                       f"pod_defaults.yaml: {config.scrub(str(exc))}")
            log(f"[deadman] cancel: {message}")
            return {"outcome": "config_error", "message": message, "exit_code": 2}
        pid_file = config.REPO_ROOT / cfg["local_log_dir"] / "deadman.pid"
        log(f"[deadman] cancel: vehicle={vehicle} pod={cfg['pod_name']} "
           f"artifacts={pid_file.parent}")
    else:
        log(f"[deadman] cancel: vehicle={vehicle} "
           f"pod={_pod_name_for(vehicle)} artifacts={pid_file.parent}")

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
    """Most recent = lexicographically LAST filename. Deterministic because
    default summary names embed a zero-padded UTC stamp
    (deadman-YYYYMMDD-HHMMSS.json), which sorts chronologically; explicit
    --summary-path files live outside the default glob and are the
    operator's own bookkeeping."""
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


# Worst-wins ordering for the all-vehicles roll-up. The PRIMARY key is always
# the exit code (1 = the pod may still be running); this only breaks ties
# WITHIN an exit code, so the reported `state` is the most informative one
# rather than an arbitrary dict-order pick.
_STATE_SEVERITY = {
    "not_armed": 0, "stopped": 1, "cancelled": 1, "refused": 2, "armed": 3,
    "stop_failed": 4, "LOST": 5, "config_error": 6,
}


def status(*, vehicle: str | None = None, pid_file: Path | None = None,
          summary_glob: str | None = None, now=_iso_now, log=_default_log) -> dict:
    """NO network, ever (per-vehicle path resolution is a local YAML read).

    THREE call shapes:
      * `vehicle=None` and no explicit paths -> report EVERY declared vehicle:
        `{"vehicles": {...}, "state": <worst>, "exit_code": <worst>}`. This is
        the default for a bare `deadman.sh status`, and it exists because a
        LOST bluerov2 fuse hiding behind a hippocampus `not_armed`/exit-0 is
        exactly the false comfort this tool exists to kill.
      * an explicit `vehicle` -> the flat single-fuse shape below.
      * an explicit `pid_file`/`summary_glob` -> also flat: the override PINS
        one artifact set, so aggregating would be meaningless.
    """
    if vehicle is None and pid_file is None and summary_glob is None:
        return _status_all_vehicles(now=now, log=log)
    return _status_one(vehicle or "hippocampus", pid_file=pid_file,
                       summary_glob=summary_glob, now=now, log=log)


def _status_all_vehicles(*, now, log) -> dict:
    try:
        vehicles = sorted(config.load_defaults().get("vehicles") or {})
    except Exception as exc:   # noqa: BLE001 — clean error, not a traceback
        return {"vehicles": {}, "state": "config_error",
                "message": f"could not read pod_defaults.yaml: "
                           f"{config.scrub(str(exc))}",
                "exit_code": 2}
    per = {v: _status_one(v, pid_file=None, summary_glob=None, now=now, log=log)
           for v in vehicles}
    if not per:
        return {"vehicles": {}, "state": "not_armed",
                "message": "pod_defaults.yaml declares no vehicles",
                "exit_code": 0}
    # max() keeps the FIRST maximal element and `per` is in sorted-vehicle
    # order, so the roll-up is deterministic.
    worst = max(per, key=lambda v: (per[v]["exit_code"],
                                    _STATE_SEVERITY.get(per[v]["state"], 0)))
    return {"vehicles": per, "state": per[worst]["state"],
            "exit_code": per[worst]["exit_code"]}


def _status_one(vehicle: str, *, pid_file: Path | None, summary_glob: str | None,
               now, log) -> dict:
    """armed (pid file names a verified-live process) |
    LOST (pid file exists but the process is dead/mismatched — the pod may
    still be running, exit 1 so scripts can alert) | last summary's outcome |
    not_armed (nothing on disk). A LOST fuse must never report as armed —
    that false comfort is exactly the failure mode this tool exists to kill.

    Exit-code contract: 0 = armed / stopped / cancelled / refused / not_armed
    (a refused arm never touched the pod, but its reason is surfaced);
    1 = LOST or stop_failed (both mean the pod may still be running —
    status-based monitoring must be able to alert on them); 2 = usage errors
    (argparse) and an unreadable vehicle config. A stop_failed summary silently
    exiting 0 would defeat the monitoring in the exact case that matters. A
    malformed pid file (bad JSON / non-int pid) reads as LOST — we cannot prove
    the fuse is alive."""
    try:
        if pid_file is None:
            pid_file = _default_pid_file(vehicle)
        if summary_glob is None:
            summary_glob = _default_summary_glob(vehicle)
    except Exception as exc:   # noqa: BLE001 — clean error, not a traceback
        message = (f"could not resolve vehicle {vehicle!r} from "
                   f"pod_defaults.yaml: {config.scrub(str(exc))}")
        log(f"[deadman] status: {message}")
        return {"state": "config_error", "vehicle": vehicle,
                "message": message, "exit_code": 2}

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

    summary = _find_latest_summary(summary_glob)
    if summary is None:
        return {"state": "not_armed", "message": "no pid file, no prior summary",
                "exit_code": 0}
    outcome = summary.get("outcome", "unknown")
    if outcome == "stop_failed":
        log("[deadman] status: last fuse EXHAUSTED its stop retries — the pod "
           "may still be running/billing; check/stop it manually")
        return {"state": "stop_failed", "summary": summary,
                "message": "last fuse exhausted its stop retries; the pod may "
                           "still be running — check/stop it manually",
                "exit_code": 1}
    if outcome == "refused":
        # A refused arm never touched the pod, so exit 0 (not a pod-may-be-
        # billing state) — but the reason must surface, not read as not_armed.
        return {"state": "refused", "reason": summary.get("reason"),
                "summary": summary, "exit_code": 0}
    return {"state": outcome, "summary": summary, "exit_code": 0}


# --------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deadman",
        description="Independent Mac-side fuse: arm it once, it sleeps ~N "
                    "hours then stops the pod (with retries) unless "
                    "cancelled — a money-safety backstop that does not "
                    "depend on any supervising process staying alive.")
    sub = p.add_subparsers(dest="subcommand", required=True)

    # NOTE the asymmetry: arm/cancel default to None (and are then REJECTED by
    # _require_vehicle) while status leaves None meaning "all vehicles".
    # Deliberate — see _require_vehicle and status().
    vehicle_kwargs = dict(choices=("hippocampus", "bluerov2"), default=None)

    arm_p = sub.add_parser("arm", help="arm the fuse (background it)")
    arm_p.add_argument("--vehicle", **vehicle_kwargs,
                      help="REQUIRED — which pod this fuse guards "
                           "(hippocampus == the pre-existing lts-replication pod)")
    arm_p.add_argument("--hours", type=float, default=3.0)
    arm_p.add_argument("--retries", type=int, default=5)
    arm_p.add_argument("--spacing-sec", type=int, default=300)
    arm_p.add_argument("--pid-file", default=None)
    arm_p.add_argument("--summary-path", default=None)
    arm_p.add_argument("--no-probe-key", action="store_true",
                      help="skip the arm-time Keychain fetch-and-discard "
                           "probe (for tests/offline use)")

    cancel_p = sub.add_parser("cancel", help="disarm a running fuse")
    cancel_p.add_argument("--vehicle", **vehicle_kwargs,
                         help="REQUIRED — which pod's fuse to disarm")
    cancel_p.add_argument("--pid-file", default=None)

    status_p = sub.add_parser("status", help="report armed/LOST/last outcome — no network")
    status_p.add_argument("--vehicle", **vehicle_kwargs,
                         help="optional — OMIT to report EVERY declared "
                              "vehicle (worst result wins the exit code)")
    status_p.add_argument("--pid-file", default=None)
    status_p.add_argument("--summary-glob", default=None)

    return p


def _require_vehicle(args: argparse.Namespace, parser: argparse.ArgumentParser,
                     subcommand: str) -> None:
    """arm/cancel take NO default vehicle. A fuse pointed at the wrong pod is
    silently wrong for HOURS — it reports healthy while the real pod bills — so
    the target is always stated out loud, never inherited from a default."""
    if not args.vehicle:
        parser.error(
            f"{subcommand} requires an explicit --vehicle "
            "{hippocampus,bluerov2} — there is deliberately no default, "
            "because a fuse armed against the wrong pod looks healthy for "
            "hours while the real pod keeps billing. 'hippocampus' is the "
            "pre-existing lts-replication pod; 'bluerov2' is the "
            "lts-replication-bluerov2 pod.")


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
    and the REAL, VEHICLE-SCOPED runtime factory, then delegates to `arm()`.
    This is the only place in the module that touches `signal.signal` or
    `tools.runtime`. The factory stays LAZY — `tools.runtime(...)` is invoked
    only when the fuse actually fires, hours later."""
    cancel_flag = {"requested": False}

    def _on_sigterm(signum, frame):
        cancel_flag["requested"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    def _disarm_signal() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        summary = arm(spec, rt_factory=lambda: tools.runtime(spec.vehicle),
                     sleep=time.sleep,
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
        _require_vehicle(args, parser, "arm")
        _validate_arm_args(args, parser)
        spec = ArmSpec(hours=args.hours, retries=args.retries,
                       spacing_sec=args.spacing_sec,
                       pid_file=_path_or_none(args.pid_file),
                       summary_path=_path_or_none(args.summary_path),
                       probe_key=not args.no_probe_key,
                       vehicle=args.vehicle)
        return _main_arm(spec)

    if args.subcommand == "cancel":
        _require_vehicle(args, parser, "cancel")
        result = cancel(vehicle=args.vehicle,
                        pid_file=_path_or_none(args.pid_file))
        print(json.dumps(result, indent=2))
        return int(result["exit_code"])

    if args.subcommand == "status":
        result = status(vehicle=args.vehicle,
                        pid_file=_path_or_none(args.pid_file),
                        summary_glob=args.summary_glob)
        print(json.dumps(result, indent=2))
        return int(result["exit_code"])

    parser.error(f"unknown subcommand {args.subcommand!r}")   # pragma: no cover
    return 2   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
