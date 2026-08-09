"""Async-job convention on the pod (training runs 10-20 min, setup 30-60 min —
tools must never block on them).

One SSH call launches `setsid bash job_wrapper.sh <job_dir> <pod_id>
<max_runtime_sec> <auto_stop>`; all state lives on the network volume under
/workspace/jobs/<job_id>/{cmd.sh,pid,out.log,exit_code,meta.json} and thus
survives MCP restarts, Mac sleep, and pod stop.
"""
import json
import re
import secrets
import shlex
from datetime import datetime
from pathlib import Path

from .config import scrub

JOBS_ROOT = "/workspace/jobs"
WRAPPER_REMOTE = f"{JOBS_ROOT}/job_wrapper.sh"
WATCHDOG_REMOTE = "/workspace/idle_watchdog.sh"
KEEPALIVE = "/workspace/.keepalive"

REMOTE_DIR = Path(__file__).resolve().parent / "remote"

_REWARD_RE = re.compile(r"mean reward", re.IGNORECASE)


class JobError(RuntimeError):
    """Job launch/inspection failure (probe failed, pod unreachable, ...)."""


def new_job_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    # random suffix: same-second relaunches must never share a job dir
    return f"{stamp}_{slug}_{secrets.token_hex(2)}"


def build_cmd_script(command: str, workdir: str, env: dict | None = None) -> str:
    lines = ["#!/usr/bin/env bash", "set -uo pipefail"]
    for k, v in (env or {}).items():
        lines.append(f"export {k}={shlex.quote(str(v))}")
    lines.append(f"cd {shlex.quote(workdir)}")
    lines.append(command)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------- runpodctl probe (3-way)
# Both probe sites used to chain `command -v runpodctl && runpodctl get pod`,
# so binary-absent and auth-refused collapsed onto ONE exit code — nobody could
# tell which half failed on 2026-08-07. The split below NAMES the cause and, in
# the same pass, fixes the likeliest one: a gate must run in the environment it
# certifies, so the retry sources /etc/rp_environment exactly like the two
# consumers do (job_wrapper.sh:23-24, idle_watchdog.sh:32). ssh.run() opens a
# non-interactive, non-login shell, where that file is never read.
#
# HARD RULE — names, paths and booleans ONLY, never the file's CONTENTS. The
# retry SOURCES /etc/rp_environment; nothing ever prints it. A `cat` would put
# an API key into MCP output, and config.scrub() runs on error paths only.
RP_ENV = "/etc/rp_environment"
SENTINEL_NO_BIN = "NO_RUNPODCTL"
SENTINEL_AUTH_BARE = "NO_RUNPODCTL_AUTH_BARE"
SENTINEL_AUTH_SOURCED = "NO_RUNPODCTL_AUTH_SOURCED"
RC_NO_RUNPODCTL = 90      # binary genuinely absent (new; callers read rc != 0)
RC_NO_AUTH = 91           # terminal auth failure (unchanged)
# stdout is the ONLY place the sentinels and the PATH / ls -l captures exist,
# so the raised errors ship a slice big enough to actually contain them (the
# captures are multi-line; the old 100-char cap truncated inside the first one)
_OUT_CAP = 600

# candidate (d), the PATH suspect: a variable name/value that is not a secret,
# plus filesystem metadata for the two image-shipped install paths
_PROBE_DIAG = ('echo "PATH=$PATH"; '
               'ls -l /usr/local/bin/runpodctl /usr/bin/runpodctl 2>&1; ')

_PROBE_FIXES = {
    SENTINEL_NO_BIN: (
        "the runpodctl BINARY is unreachable even AFTER sourcing "
        f"{RP_ENV} (compare the two PATH= / ls -l captures in the probe "
        "output) — no pod-side stop path can exist until the binary is back"),
    SENTINEL_AUTH_BARE: (
        "runpodctl was refused in the BARE unsourced shell but accepted after "
        f"sourcing {RP_ENV} [H2] — the credential exists and only the "
        "non-interactive shell lacked it; that is what this probe now fixes"),
    SENTINEL_AUTH_SOURCED: (
        "runpodctl was refused BOTH bare AND after sourcing "
        f"{RP_ENV} [H1] — there is no usable credential anywhere on the pod, "
        "so sourcing cannot fix this one"),
}


def _runpodctl_probe_command(pod_id: str, *, fail_extra: str = "") -> str:
    """Three-way diagnostic probe, as a command PREFIX: it exits on failure and
    falls THROUGH on success, so each caller appends its own continuation.

    The shape is BARE-DIAGNOSTIC then SOURCED-VERDICT:

      1. The bare pair (`command -v`, then `get pod`) decides NOTHING. It exists
         only to answer the H1-vs-H2 question — a bare refusal followed by a
         sourced success is what confirms H2 — and to capture the BARE PATH,
         which is the §2.1d evidence when the binary is off the minimal PATH.
      2. `/etc/rp_environment` is then sourced UNCONDITIONALLY, whatever the
         bare result (plan §2.3-B: "a gate must run in the environment it
         certifies"). Both real consumers — job_wrapper.sh:23-24 and
         idle_watchdog.sh:32 — always source it before touching runpodctl, so a
         bare pass proves nothing about the stop path: a file that overrides
         PATH or the credential would leave the probe reporting PROBE_OK while
         every watchdog tick fails. That is the armed-but-dud $16/day leak this
         gate exists to prevent.
      3. The sourced pair is AUTHORITATIVE and carries the verdict sentinels:
         still no binary => NO_RUNPODCTL/90, still refused => AUTH_SOURCED/91.
         Falling through means the stop path works in the environment it will
         actually run in.

    `fail_extra` is echoed on every failure exit (the watchdog keeps emitting
    its legacy WATCHDOG_PROBE_FAILED sentinel alongside the finer one)."""
    qid = shlex.quote(pod_id)
    have = "command -v runpodctl >/dev/null 2>&1"
    get = f"runpodctl get pod {qid} >/dev/null 2>&1"
    return (
        # 1 · bare shell — diagnostics only, no verdict. The elif matters: a
        # missing binary must not masquerade as an auth refusal.
        f"if ! {have}; then {_PROBE_DIAG}"
        f"elif ! {get}; then echo {SENTINEL_AUTH_BARE}; {_PROBE_DIAG}fi; "
        # 2 · the environment the consumers actually run in
        f"[ -f {RP_ENV} ] && . {RP_ENV}; "
        # 3 · sourced shell — the authoritative checks
        f"if ! {have}; then echo {SENTINEL_NO_BIN}; {fail_extra}{_PROBE_DIAG}"
        f"exit {RC_NO_RUNPODCTL}; fi; "
        f"if ! {get}; then echo {SENTINEL_AUTH_SOURCED}; {fail_extra}"
        f"exit {RC_NO_AUTH}; fi; "
    )


def _probe_diagnosis(stdout: str) -> str:
    """Name WHICH half failed so the caller's warning names WHICH fix applies.

    Read off whole lines (the sentinels are echoed alone) and prefer the
    TERMINAL sentinel: an H1 failure emits AUTH_BARE first, then AUTH_SOURCED.
    Returned separately from the raw output slice because that slice is
    truncated and the diagnostic PATH/ls noise would otherwise bury it."""
    lines = {ln.strip() for ln in stdout.splitlines()}
    for sentinel in (SENTINEL_AUTH_SOURCED, SENTINEL_AUTH_BARE,
                     SENTINEL_NO_BIN):
        if sentinel in lines:
            return f"cause={sentinel} — {_PROBE_FIXES[sentinel]}"
    return ("cause=UNKNOWN — the runpodctl probe half reported no failure; "
            "the cause is elsewhere (see out=/err= below)")


def _probe_auto_stop(ssh, host: str, port: int, pod_id: str) -> None:
    """auto_stop must fail LOUDLY at launch if the stop path can't work —
    an armed-but-dud auto_stop is a silent $16/day leak."""
    proc = ssh.run(host, port,
                   _runpodctl_probe_command(pod_id) + "echo PROBE_OK",
                   timeout=60)
    if proc.returncode != 0 or "PROBE_OK" not in proc.stdout:
        raise JobError(
            "auto_stop requested but the runpodctl self-stop probe FAILED on "
            f"the pod (rc={proc.returncode}, {_probe_diagnosis(proc.stdout)}, "
            f"out={scrub(proc.stdout).strip()[:_OUT_CAP]}, "
            f"stderr={scrub(proc.stderr).strip()[:200]}). "
            "Refusing to launch with a dud auto_stop — fix runpodctl on the pod "
            "or launch with auto_stop=false and stop_pod() yourself.")


def launch(ssh, host: str, port: int, *, name: str, command: str, workdir: str,
           pod_id: str, max_runtime_sec: int, auto_stop: bool,
           env: dict | None = None) -> str:
    """Push cmd.sh + wrapper + meta, then detach the wrapper. Returns job_id."""
    if auto_stop:
        _probe_auto_stop(ssh, host, port, pod_id)

    job_id = new_job_id(name)
    job_dir = f"{JOBS_ROOT}/{job_id}"

    ssh.push_text(host, port, (REMOTE_DIR / "job_wrapper.sh").read_text(),
                  WRAPPER_REMOTE, executable=True)
    ssh.push_text(host, port, build_cmd_script(command, workdir, env),
                  f"{job_dir}/cmd.sh", executable=True)
    meta = {"name": name, "command": command, "workdir": workdir,
            "pod_id": pod_id, "max_runtime_sec": max_runtime_sec,
            "auto_stop": auto_stop, "started_at": datetime.now().isoformat()}
    ssh.push_text(host, port, json.dumps(meta, indent=1), f"{job_dir}/meta.json")

    qdir, qpod = shlex.quote(job_dir), shlex.quote(pod_id)
    qrun, qstop = shlex.quote(str(int(max_runtime_sec))), shlex.quote("1" if auto_stop else "0")
    ssh.run(host, port,
            f"touch {KEEPALIVE} && "
            f"setsid bash {WRAPPER_REMOTE} {qdir} {qpod} {qrun} {qstop} "
            f"</dev/null >/dev/null 2>&1 & echo LAUNCHED",
            timeout=60, check=True)
    return job_id


def list_live(ssh, host: str, port: int) -> list[str]:
    """Job ids with a live wrapper pid — the concurrency-guard input.

    Fails LOUDLY on SSH/remote failure: a transient failure must never read
    as "no jobs running" (that would bypass the one-job guard)."""
    proc = ssh.run(host, port,
                   f'for d in {JOBS_ROOT}/*/; do '
                   f'[ -f "$d/pid" ] || continue; '
                   f'kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null '
                   f'&& basename "$d"; done; echo LIVE_LIST_END',
                   timeout=60)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines or lines[-1] != "LIVE_LIST_END":
        raise JobError(f"could not inspect live jobs (rc={proc.returncode}, "
                       f"stderr={scrub(proc.stderr).strip()[:200]}) — refusing "
                       "to assume the pod is idle")
    return lines[:-1]


_STATUS_TEMPLATE = (
    'cd {job_dir} 2>/dev/null || {{ echo NO_SUCH_JOB; exit 0; }}; '
    'echo "---PID---"; cat pid 2>/dev/null; '
    'echo "---ALIVE---"; if [ -f pid ] && kill -0 "$(cat pid)" 2>/dev/null; '
    'then echo yes; else echo no; fi; '
    'echo "---EXIT---"; cat exit_code 2>/dev/null; '
    'echo "---META---"; cat meta.json 2>/dev/null; '
    'echo "---LOG---"; tail -n {tail_lines} out.log 2>/dev/null'
)


def status(ssh, host: str, port: int, job_id: str, tail_lines: int = 40) -> dict:
    job_dir = shlex.quote(f"{JOBS_ROOT}/{job_id}")
    proc = ssh.run(host, port,
                   _STATUS_TEMPLATE.format(job_dir=job_dir,
                                           tail_lines=int(tail_lines)),
                   timeout=60)
    out = proc.stdout
    if "NO_SUCH_JOB" in out.split("---PID---")[0]:
        return {"job_id": job_id, "state": "not_found"}
    # an unreachable pod must surface as an ERROR, not as "orphaned"
    if proc.returncode != 0 or "---PID---" not in out:
        raise JobError(f"could not inspect job {job_id} (rc={proc.returncode}, "
                       f"stderr={scrub(proc.stderr).strip()[:200]})")

    # Split ONLY on the exact section markers — log tails routinely contain
    # dashed separators of their own and must never be truncated by them.
    markers = ["PID", "ALIVE", "EXIT", "META", "LOG"]

    def section(marker: str) -> str:
        start = out.find(f"---{marker}---")
        if start < 0:
            return ""
        start += len(f"---{marker}---")
        later = [out.find(f"---{m}---", start) for m in markers]
        ends = [i for i in later if i >= 0]
        return out[start:min(ends)] .strip("\n") if ends else out[start:].strip("\n")

    alive = section("ALIVE").strip() == "yes"
    exit_raw = section("EXIT").strip()
    log_tail = section("LOG")
    try:
        meta = json.loads(section("META") or "{}")
    except json.JSONDecodeError:
        meta = {}

    result: dict = {"job_id": job_id, "meta": meta, "log_tail": log_tail}
    if alive:
        result["state"] = "running"
    elif exit_raw != "":
        code = int(exit_raw)
        result["exit_code"] = code
        result["state"] = "succeeded" if code == 0 else "failed"
        if code == 124:
            result["note"] = ("hit the wall-clock ceiling "
                              "(timeout -> exit 124); auto-stop still ran if armed")
    else:
        result["state"] = "orphaned"   # pid file present/dead, no exit_code

    reward_lines = [l.strip() for l in log_tail.splitlines() if _REWARD_RE.search(l)]
    if reward_lines:
        result["latest_reward_line"] = reward_lines[-1]
    return result


def watchdog_install_command(pod_id: str, idle_minutes: int) -> str:
    """Remote command that (re)arms the idle watchdog. Probes runpodctl AND
    the script's presence LOUDLY — a watchdog that fails silently every 5 min
    is worse than none."""
    qid = shlex.quote(pod_id)
    qmin = shlex.quote(str(int(idle_minutes)))
    return (
        # same three-way split as _probe_auto_stop, and it keeps emitting the
        # legacy WATCHDOG_PROBE_FAILED on every failure exit
        _runpodctl_probe_command(pod_id,
                                 fail_extra="echo WATCHDOG_PROBE_FAILED; ")
        + f"[ -f {WATCHDOG_REMOTE} ] || {{ echo WATCHDOG_MISSING; exit 92; }}; "
        f"[ -f /workspace/.idle_watchdog.pid ] && "
        f"kill \"$(cat /workspace/.idle_watchdog.pid)\" 2>/dev/null; "
        f"touch {KEEPALIVE} && "
        f"setsid bash {WATCHDOG_REMOTE} {qid} {qmin} "
        f">> /workspace/.idle_watchdog.log 2>&1 </dev/null & "
        f"echo $! > /workspace/.idle_watchdog.pid && echo WATCHDOG_ARMED"
    )


def install_watchdog(ssh, host: str, port: int, pod_id: str,
                     idle_minutes: int) -> None:
    """Push idle_watchdog.sh (the container disk wipes on stop — reinstall on
    EVERY transition-to-running) and (re)arm it. Fails loudly: the caller
    surfaces this as an ensure_pod warning, never swallows it."""
    ssh.push_text(host, port, (REMOTE_DIR / "idle_watchdog.sh").read_text(),
                  WATCHDOG_REMOTE, executable=True)
    proc = ssh.run(host, port, watchdog_install_command(pod_id, idle_minutes),
                   timeout=60)
    if proc.returncode != 0 or "WATCHDOG_ARMED" not in proc.stdout:
        raise JobError(
            f"idle watchdog install FAILED (rc={proc.returncode}, "
            f"{_probe_diagnosis(proc.stdout)}, "
            f"out={scrub(proc.stdout).strip()[:_OUT_CAP]}, "
            f"err={scrub(proc.stderr).strip()[:200]}) — the pod will NOT "
            "self-stop when idle; stop it manually when done")
