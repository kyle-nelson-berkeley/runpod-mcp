"""watch — an observation-loop for one already-launched RunPod job: discover
it, tail its out.log incrementally, parse rsl_rl training metrics, apply
advisory plateau/stall/failure heuristics, and report status until the job
reaches a terminal state.

This is a Mac-side background CLI, NOT an MCP tool — invoked directly (via
watch.sh), never through .mcp.json (the stdio server's tool surface stays 14
tools on purpose; see runpod-mcp/CLAUDE.md §D — "never add a blocking tool").
It runs ALONGSIDE `supervise` (as a second `run_in_background` Bash task) —
supervise owns launch/poll/capture/stop; watch only OBSERVES. It never stops,
terminates, or otherwise mutates the pod or the job: every remote command it
issues is read-only (tail/cat/ls).

AUTHORITATIVE VS ADVISORY (read this before trusting watch's output): the
`supervise` summary JSON (`supervise-<job_id>.json`) is the AUTHORITATIVE
terminal record of a job — it is written by the process that actually
launched/polled/stopped the job. This module's own status file
(`watch-<job_id>.json`) and its exit code are ADVISORY ONLY. In particular the
plateau heuristic (PlateauDetector) triggers no automated action of any kind
— it is a diagnostic hint for a human/agent reviewing the run, nothing more.

HEURISTIC DEFAULTS ARE UNVERIFIED IN PRODUCTION (GPU envelope SPENT): the
--plateau-window / --plateau-min-delta / --stall-sec defaults below were
derived from ONE real converging rsl_rl training log (fixture-tested — see
tests/test_watch.py and tests/fixtures/watch/) plus synthetic edge-case
fixtures. They were never validated against a live pod because the GPU
budget for this build was already spent. Treat them as a reasonable starting
point, not a calibrated threshold — re-tune after watching a few real runs.

TEST SEAM (read this before writing a test): every call into the tool
surface goes through the MODULE ATTRIBUTE — `tools.pod_status(rt)`,
`tools._conn_info(rt)` — via `from . import jobs, tools` + `tools.X(...)`,
never `from .tools import X`. Tests monkeypatch attributes on the
`runpod_mcp.tools` module itself. The tail/exit-code/artifact-listing reads
are the one call that bypasses `tools.*` (mirroring supervise's
`rt.ssh.rsync_pull`) — they go straight through `rt.ssh.run`, exercised in
tests via a scripted fake SSH client.

`watch()` never talks to the real Keychain/network; only `main()` calls
`tools.runtime()` to build the real Runtime.

EXIT CODES (the CLI's exit IS the page — it runs as one run_in_background
Bash task, same convention as supervise.sh):
    0   job completed OK — either (a) exit_code artifact == 0 and the job is
        gone from active jobs, or (b) the pod itself was AUTHORITATIVELY
        observed stopped/no_pod with no failure evidence cached (supervise's
        45s poll often stops the pod between the watcher's 60s polls before
        the exit_code artifact was ever read — only supervise/deadman stop
        pods, so an observed stop means the run is over and was handled by
        its owner; the status JSON carries a pod_stopped_note deferring to
        supervise's summary as the authoritative outcome record). Degraded
        signals (raised poll, running_ssh_pending, active_jobs_error) are
        UNREACHABLE, not stopped — they never take this branch and stay on
        the retry->stall path.
    2   argparse usage error
    3   plateau DETECTED mid-run — the watcher exits IMMEDIATELY at
        detection time, while the job is still running, so the page arrives
        while a human can still decide whether to intervene (that is the
        observation loop's entire marginal value). The watcher takes no
        action on the job (advisory only; it never touches stop paths) —
        the job keeps running and supervise remains authoritative.
        Precedence: failure evidence (-> 4) or already-terminal evidence
        (-> 0/4) visible in the SAME poll round beats a first-time plateau
        fire; a job that reaches terminal state having never plateaued
        exits 0/4 as usual. The condition is evaluated on the CURRENT
        trailing --plateau-window as of the newest parsed iteration — a
        HISTORICAL flat stretch the reward has since recovered from (e.g.
        seen while catching up on a backlog after a late attach) never
        pages; flatness must hold at the newest point.
    4   failure detected — either a traceback/CUDA-OOM pattern seen in the
        tailed log, or the pod-side exit_code artifact was nonzero
    5   stall / lost contact — no new iteration within --stall-sec AND, after
        exhausting the ordered exit-decision-tree checks (read exit_code
        artifact, check active-jobs disappearance), no terminal evidence was
        found. Reserved for genuinely-unreachable-with-no-terminal-evidence;
        "gone from active jobs + exit_code present" is NEVER reported as 5,
        always resolved to its own code (0 or 4).
"""
import argparse
import json
import re
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import jobs, ssh, tools
from .config import REPO_ROOT, scrub

REFUSE_ERRORS = (tools.ToolError, jobs.JobError, ssh.SSHError)

DEFAULT_INTERVAL = 60
DEFAULT_STARTUP_GRACE = 180
# Heuristic (unverified in production, GPU envelope SPENT): iteration cadence
# varies wildly (the real fixture ran ~1.4s/iter cold, ~0.02s/iter warm), and
# training legitimately pauses for checkpoint/eval — 10 minutes of silence is
# a conservative floor before treating it as a real stall.
DEFAULT_STALL_SEC = 600
# Heuristic (unverified in production, GPU envelope SPENT): fixture-derived —
# see tests/test_watch.py::TestPlateauDetector. window=50/min_delta=3.0 never
# fires on the real 200-iteration converging fixture (min trailing-50 spread
# observed there is ~3.98) but reliably fires on a synthetic flat tail.
DEFAULT_PLATEAU_WINDOW = 50
DEFAULT_PLATEAU_MIN_DELTA = 3.0
# Discovery poll cadence during --startup-grace; deliberately short and not a
# CLI flag — the grace window bounds total wait, not per-poll cost.
_DISCOVERY_POLL_SEC = 5


def _default_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class WatchError(RuntimeError):
    """Job discovery failed — no single unambiguous job found (see
    _discover_job_id). Mapped to exit 5 (lost contact) by main()."""


# --------------------------------------------------------------- ANSI/CR strip

# Covers both bracketed CSI sequences (\x1b[1m, \x1b[0m, \x1b[3g) and bare
# ESC-letter cursor-control codes (\x1bH — cursor home), plus bare CR (\r,
# used for same-line progress overwrites in the startup noise).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[A-Za-z]|\r")


def _strip_control(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --------------------------------------------------------------------- parser

@dataclass(frozen=True)
class MetricPoint:
    iteration: int
    total_iterations: int
    mean_reward: float
    value_loss: float
    surrogate_loss: float
    entropy_loss: float


_ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
_VALUE_LOSS_RE = re.compile(r"^Mean value_function loss:\s*([-+0-9.eE]+)")
_SURROGATE_RE = re.compile(r"^Mean surrogate loss:\s*([-+0-9.eE]+)")
_ENTROPY_RE = re.compile(r"^Mean entropy loss:\s*([-+0-9.eE]+)")
_REWARD_LINE_RE = re.compile(r"^Mean reward:\s*([-+0-9.eE]+)")


class MetricParser:
    """Incremental, stateful parser: feed() successive text chunks (which may
    split ANY line, including mid-block or mid-number) and get back the
    MetricPoints newly completed since the last feed() call.

    Design: line-buffered. An unterminated trailing partial line is held in
    `self._buf` and prefixed onto the next feed()'s text — this is what makes
    a block split mid-line across two chunks parse identically to the whole
    file (see tests/test_watch.py::test_split_block_across_two_chunks...).
    ANSI/CR control codes are stripped per-line before matching (extends the
    idea behind jobs._REWARD_RE — jobs.status only surfaces the latest raw
    reward line — to the full four-metric block + iteration index, parsed
    incrementally rather than from a single tail snapshot).
    """

    def __init__(self):
        self._buf = ""
        self._cur: dict | None = None   # {"iteration", "total_iterations", ...}

    def feed(self, chunk: str) -> list[MetricPoint]:
        text = self._buf + chunk
        lines = text.split("\n")
        self._buf = lines.pop()   # last element has no trailing \n yet — hold it
        points: list[MetricPoint] = []
        for raw_line in lines:
            pt = self._feed_line(_strip_control(raw_line).strip())
            if pt is not None:
                points.append(pt)
        return points

    def _feed_line(self, line: str) -> MetricPoint | None:
        m = _ITER_RE.search(line)
        if m:
            # A fresh header always starts a new block, discarding whatever
            # (necessarily incomplete) accumulator preceded it.
            self._cur = {"iteration": int(m.group(1)),
                        "total_iterations": int(m.group(2))}
            return None
        if self._cur is None:
            return None   # not inside a block yet (startup noise, etc.)
        for key, rx in (("value_loss", _VALUE_LOSS_RE),
                        ("surrogate_loss", _SURROGATE_RE),
                        ("entropy_loss", _ENTROPY_RE)):
            m = rx.match(line)
            if m:
                self._cur[key] = float(m.group(1))
                return None
        m = _REWARD_LINE_RE.match(line)
        if m:
            self._cur["mean_reward"] = float(m.group(1))
        if {"value_loss", "surrogate_loss", "entropy_loss", "mean_reward"} <= self._cur.keys():
            pt = MetricPoint(iteration=self._cur["iteration"],
                             total_iterations=self._cur["total_iterations"],
                             mean_reward=self._cur["mean_reward"],
                             value_loss=self._cur["value_loss"],
                             surrogate_loss=self._cur["surrogate_loss"],
                             entropy_loss=self._cur["entropy_loss"])
            self._cur = None   # block emitted — a stray duplicate metric line
                               # before the next header now lands harmlessly
            return pt
        return None


# --------------------------------------------------------------------- plateau

class PlateauDetector:
    """Advisory-only heuristic over the mean-reward series: fires once the
    trailing `window_iters`-wide span of reward has a max-min spread below
    `min_delta`. See module docstring — HEURISTIC, unverified in production
    (GPU envelope SPENT). Triggers no automated action on the job; watch.py
    surfaces it by exiting 3 IMMEDIATELY at detection time (the page), while
    the job keeps running untouched."""

    def __init__(self, window_iters: int = DEFAULT_PLATEAU_WINDOW,
                min_delta: float = DEFAULT_PLATEAU_MIN_DELTA):
        self.window_iters = window_iters
        self.min_delta = min_delta
        self._history: list[tuple[int, float]] = []

    def update(self, point: MetricPoint) -> bool:
        self._history.append((point.iteration, point.mean_reward))
        cutoff = point.iteration - self.window_iters
        self._history = [(i, r) for i, r in self._history if i >= cutoff]
        if len(self._history) < 2:
            return False
        span = self._history[-1][0] - self._history[0][0]
        if span < self.window_iters:
            return False
        rewards = [r for _, r in self._history]
        return (max(rewards) - min(rewards)) < self.min_delta


# ------------------------------------------------------------ failure detect

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
_CUDA_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
                          re.IGNORECASE)


def detect_failure(text: str) -> str | None:
    """Best-effort text-pattern failure detection over a tailed chunk.
    Returns a short human label, or None. This is independent of (and a
    faster proactive signal than) the pod-side exit_code artifact check in
    the exit decision tree — see module docstring, exit code 4."""
    if _CUDA_OOM_RE.search(text):
        return "cuda_oom"
    if _TRACEBACK_RE.search(text):
        return "traceback"
    return None


# ----------------------------------------------------------------------- spec

@dataclass
class WatchSpec:
    job_id: str | None = None
    interval: int = DEFAULT_INTERVAL
    startup_grace: int = DEFAULT_STARTUP_GRACE
    stall_sec: int = DEFAULT_STALL_SEC
    plateau_window: int = DEFAULT_PLATEAU_WINDOW
    plateau_min_delta: float = DEFAULT_PLATEAU_MIN_DELTA
    status_path: str | None = None


# ------------------------------------------------------------- job discovery

def _discover_job_id(rt, spec: WatchSpec, *, sleep=time.sleep, now=time.monotonic,
                     log=_default_log) -> str:
    """Poll the active-jobs seam (tools.pod_status(rt)["active_jobs"]) for
    THE single active job. --job-id (spec.job_id) overrides this entirely —
    no pod_status call at all in that case (attended use).

    Chosen seam rationale: tools.pod_status never raises for a list_live
    failure — it surfaces an "active_jobs_error" key instead — and a
    pre-SSH failure yields status="running_ssh_pending" rather than raising.
    BINDING invariant: during --startup-grace, ANY degraded signal (a raised
    exception, active_jobs_error present, or status != "running") means
    retry-and-keep-waiting, NEVER instant lost-contact — this is the same
    "the job detached before the reply arrived" benign-race pattern
    supervise.py's launch-adoption path documents. More than one active job
    is a DIFFERENT case — a real one-job-guard violation, not a transient
    glitch — and raises immediately rather than waiting out the grace."""
    if spec.job_id:
        return spec.job_id

    deadline = now() + spec.startup_grace
    while True:
        try:
            status = tools.pod_status(rt)
        except REFUSE_ERRORS as exc:
            log(f"[watch] discovery poll error (retrying): {scrub(str(exc))}")
            status = None

        if status is not None:
            if status.get("status") == "running" and status.get("active_jobs_error") is None:
                active = status.get("active_jobs")
                if active:
                    if len(active) > 1:
                        raise WatchError(
                            f"discovery found multiple active jobs {active!r} — "
                            "ambiguous; pass --job-id explicitly")
                    return active[0]
                # active == [] or None (key absent but not an error): job
                # hasn't started yet — armed-before-launch, keep waiting.
            else:
                log(f"[watch] discovery: degraded pod status "
                   f"({status.get('status')!r}, "
                   f"active_jobs_error={status.get('active_jobs_error')!r}) — retrying")

        remaining = deadline - now()
        if remaining <= 0:
            raise WatchError(
                f"no single active job found within --startup-grace "
                f"({spec.startup_grace}s) — pass --job-id explicitly")
        sleep(min(_DISCOVERY_POLL_SEC, remaining))


# --------------------------------------------------------------- remote reads

def _tail(rt, job_dir: str, offset: int, *, timeout: int = 30) -> tuple[str, int]:
    """Incremental read-only tail of <job_dir>/out.log starting at byte
    `offset` (0-indexed bytes already consumed). Returns (new_text,
    new_offset). Never raises on a transient SSH failure — returns ("",
    offset) unchanged so the caller's poll loop just tries again next
    interval; that failure is itself informative (often the happy-path pod
    stopping right after job end — see module docstring). NOTE: the
    tools._conn_info lookup itself lives INSIDE the try — it raises ToolError
    when the pod just stopped and the connection cache expired, which is
    exactly that same happy-path race, not a crash-worthy bug."""
    qpath = shlex.quote(f"{job_dir}/out.log")
    try:
        host, port = tools._conn_info(rt)
        proc = rt.ssh.run(host, port, f"tail -c +{offset + 1} {qpath} 2>/dev/null",
                          timeout=timeout)
    except (tools.ToolError, ssh.SSHError):
        return "", offset
    if proc.returncode != 0:
        return "", offset
    text = proc.stdout or ""
    # Byte-offset bookkeeping assumes the ssh transport round-trips text as
    # UTF-8 without loss — true for these logs (ANSI/CR control codes are all
    # single-byte ASCII); a lossy transcode would only ever cause a benign
    # re-read of a few trailing bytes, never data loss, since offset only
    # ever advances by what we actually decoded.
    return text, offset + len(text.encode("utf-8"))


def _read_exit_code(rt, job_dir: str, *, timeout: int = 30) -> int | None:
    """Best-effort read of the pod-side exit_code artifact (the same file
    jobs.status() reads) — step (a) of the ordered exit-decision-tree check.
    Returns None if absent/unreadable (job still running, or pod
    unreachable); never raises — including from the tools._conn_info lookup
    itself (pod-just-stopped + expired connection cache raises ToolError)."""
    qpath = shlex.quote(f"{job_dir}/exit_code")
    try:
        host, port = tools._conn_info(rt)
        proc = rt.ssh.run(host, port, f"cat {qpath} 2>/dev/null", timeout=timeout)
    except (tools.ToolError, ssh.SSHError):
        return None
    txt = (proc.stdout or "").strip()
    try:
        return int(txt)
    except ValueError:
        return None


def _snapshot_artifacts(rt, job_dir: str, *, timeout: int = 30) -> str:
    """Read-only directory LISTING (names/sizes/mtimes) of the job dir —
    listings only, never downloads. Best-effort: returns "" on failure —
    including a tools._conn_info ToolError (pod just stopped)."""
    qpath = shlex.quote(job_dir)
    try:
        host, port = tools._conn_info(rt)
        proc = rt.ssh.run(host, port, f"ls -la {qpath} 2>/dev/null", timeout=timeout)
    except (tools.ToolError, ssh.SSHError):
        return ""
    return proc.stdout or ""


def _poll_job_presence(rt, job_id: str) -> tuple[bool | None, bool]:
    """Step (b) of the ordered exit-decision-tree check. Returns
    (gone_from_active, pod_stopped):

    - gone_from_active: True/False when the active-jobs signal resolved this
      poll, None when it is degraded/unavailable (caller keeps its last known
      value rather than treating None as either answer).
    - pod_stopped: True ONLY on an AUTHORITATIVE pod_status of
      "stopped"/"no_pod" — the API answered and said the pod is not running.
      A raised poll, "running_ssh_pending", or an active_jobs_error is
      UNREACHABLE/degraded, NOT stopped, and must never set this flag (those
      stay on the retry->stall path). The distinction matters because only
      supervise/deadman ever stop pods: an authoritative stop observed by the
      watcher means the run is over and was already handled by its owner."""
    try:
        status = tools.pod_status(rt)
    except REFUSE_ERRORS:
        return None, False
    st = status.get("status")
    if st in ("stopped", "no_pod"):
        # pod itself is gone -> the job is certainly not active
        return True, True
    if st != "running" or status.get("active_jobs_error") is not None:
        return None, False
    active = status.get("active_jobs")
    if active is None:
        return None, False
    return job_id not in active, False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_status_path(rt, job_id: str) -> Path:
    return REPO_ROOT / rt.cfg["local_log_dir"] / f"watch-{job_id}.json"


def _trend_arrow(prev: float | None, curr: float) -> str:
    if prev is None:
        return "→"   # →
    if curr > prev:
        return "↑"   # ↑
    if curr < prev:
        return "↓"   # ↓
    return "→"


def _format_status_line(job_id: str, point: MetricPoint | None, trend: str,
                        plateau_fired: bool, stalled: bool) -> str:
    if point is None:
        core = "no metrics observed yet"
    else:
        pct = (100.0 * point.iteration / point.total_iterations
              if point.total_iterations else 0.0)
        core = (f"iter {point.iteration}/{point.total_iterations} "
                f"({pct:.1f}%) reward={point.mean_reward:.2f} {trend}")
    flags = []
    if plateau_fired:
        flags.append("PLATEAU(advisory)")
    if stalled:
        flags.append("STALLED")
    suffix = f" [{' '.join(flags)}]" if flags else ""
    return f"[watch] job_id={job_id} {core}{suffix}"


def _write_status_file(status_path: Path, status: dict) -> None:
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2))
    except Exception:   # noqa: BLE001 — a status-file write failure must
                        # never crash the watcher; it's advisory output only
        pass


def _build_status(*, job_id: str, spec: WatchSpec, point: MetricPoint | None,
                  trend: str, plateau_fired: bool, failure_detected: str | None,
                  exit_code_artifact: int | None, gone_from_active: bool | None,
                  stalled: bool, offset_bytes: int, artifacts: str,
                  started_at: str, process_exit_code: int | None) -> dict:
    return {
        "job_id": job_id,
        "spec": asdict(spec),
        "iteration": point.iteration if point else None,
        "total_iterations": point.total_iterations if point else None,
        "mean_reward": point.mean_reward if point else None,
        "trend": trend,
        "plateau_fired": plateau_fired,
        "failure_detected": failure_detected,
        "exit_code_artifact": exit_code_artifact,
        "gone_from_active": gone_from_active,
        "stalled": stalled,
        "offset_bytes": offset_bytes,
        "artifacts": artifacts,
        "started_at": started_at,
        "updated_at": _iso_now(),
        "process_exit_code": process_exit_code,
        "advisory_note": ("this file is ADVISORY ONLY — the supervise "
                          "summary JSON is the authoritative terminal "
                          "record; see watch.py module docstring"),
    }


# --------------------------------------------------------------------- core

def watch(rt, spec: WatchSpec, *, sleep=time.sleep, now=time.monotonic,
          log=_default_log) -> dict:
    started_at = _iso_now()

    try:
        job_id = _discover_job_id(rt, spec, sleep=sleep, now=now, log=log)
    except WatchError as exc:
        log(f"[watch] discovery failed: {scrub(str(exc))}")
        result = {
            "job_id": None, "spec": asdict(spec), "discovery_error": scrub(str(exc)),
            "process_exit_code": 5, "started_at": started_at, "ended_at": _iso_now(),
            "status_path": None,
        }
        if spec.status_path:
            _write_status_file(Path(spec.status_path), result)
            result["status_path"] = spec.status_path
        print(json.dumps(result, indent=2))
        return result

    job_dir = f"{jobs.JOBS_ROOT}/{job_id}"
    status_path = Path(spec.status_path) if spec.status_path else _default_status_path(rt, job_id)

    metric_parser = MetricParser()
    plateau = PlateauDetector(spec.plateau_window, spec.plateau_min_delta)

    offset = 0
    plateau_fired = False
    failure_detected: str | None = None
    last_point: MetricPoint | None = None
    trend = "→"
    last_new_iter_at = now()
    last_exit_code: int | None = None
    last_gone: bool | None = None

    log(f"[watch] observing job_id={job_id} interval={spec.interval}s "
       f"stall_sec={spec.stall_sec}s plateau_window={spec.plateau_window} "
       f"plateau_min_delta={spec.plateau_min_delta} (heuristic, unverified "
       "in production — GPU envelope SPENT)")

    while True:
        chunk, offset = _tail(rt, job_dir, offset)
        points = metric_parser.feed(chunk) if chunk else []
        for pt in points:
            trend = _trend_arrow(last_point.mean_reward if last_point else None,
                                 pt.mean_reward)
            last_point = pt
            last_new_iter_at = now()
            # Plateau reflects the CURRENT trailing window as of the NEWEST
            # point — deliberately NOT a per-point latch. A late attach's
            # first _tail reads the whole existing out.log as one backlog;
            # a historical flat stretch the reward has since recovered from
            # must not page (stale), while flatness that persists through
            # the newest point still fires exactly as before. With no new
            # points this round the last evaluation stands (still-flat data
            # stays flat until new evidence arrives).
            plateau_fired = plateau.update(pt)
        if chunk:
            fail = detect_failure(chunk)
            if fail:
                failure_detected = fail

        artifacts = _snapshot_artifacts(rt, job_dir)

        # Ordered exit-decision-tree check (a)/(b) — ATTEMPTED every
        # iteration (not just once stalled), so by the time `stalled` is
        # evaluated below, both have already been tried this same round.
        exit_code_now = _read_exit_code(rt, job_dir)
        if exit_code_now is not None:
            last_exit_code = exit_code_now
        gone_now, pod_stopped = _poll_job_presence(rt, job_id)
        if gone_now is not None:
            last_gone = gone_now

        stalled = (now() - last_new_iter_at) >= spec.stall_sec

        status = _build_status(job_id=job_id, spec=spec, point=last_point,
                               trend=trend, plateau_fired=plateau_fired,
                               failure_detected=failure_detected,
                               exit_code_artifact=last_exit_code,
                               gone_from_active=last_gone, stalled=stalled,
                               offset_bytes=offset, artifacts=artifacts,
                               started_at=started_at, process_exit_code=None)
        _write_status_file(status_path, status)
        print(_format_status_line(job_id, last_point, trend, plateau_fired, stalled),
             flush=True)

        # (1) terminal evidence — authoritative once both signals resolve;
        # "gone from active + exit_code present" is NEVER reported as 5.
        # It also beats a same-round plateau fire: the job is already over,
        # there is nothing left to intervene on.
        if last_exit_code is not None and last_gone:
            return _finalize(status, status_path,
                             0 if last_exit_code == 0 else 4)
        # (2) a definite-bad signal seen directly in the log — beats plateau
        # when both are visible in the same round.
        if failure_detected:
            return _finalize(status, status_path, 4)
        # (2b) pod AUTHORITATIVELY observed stopped/no_pod (never set by a
        # raised poll / running_ssh_pending / degraded signal — see
        # _poll_job_presence). Only supervise/deadman stop pods, so an
        # observed stop means the run is over and was already handled by its
        # owner. A nonzero cached exit_code was caught by (1) (stopped
        # implies gone) and log-failure evidence by (2), so reaching here
        # means NO failure evidence: resolve exit 0 promptly rather than
        # idling out --stall-sec into a false exit-5 page — the common case
        # is supervise's 45s poll stopping the pod between our 60s polls
        # before we ever cached the exit_code artifact.
        if pod_stopped:
            if last_exit_code is None:
                status["pod_stopped_note"] = (
                    "pod stopped by its supervisor before the watcher could "
                    "read exit_code — no failure evidence observed; "
                    "supervise's summary JSON is the authoritative outcome "
                    "record")
            return _finalize(status, status_path, 0)
        # (3) plateau is a WATCHER-terminal condition: the page (exit 3)
        # goes out at DETECTION time, while the job is still running, so a
        # human can decide whether to intervene. The watcher takes no action
        # on the job — it just exits (still advisory; never touches stop
        # paths). Supervise keeps running and remains authoritative.
        if plateau_fired:
            status["plateau_note"] = (
                "watcher exited at plateau DETECTION while the job was "
                "still running — no action was taken on the job; the "
                "supervise summary JSON remains the authoritative terminal "
                "record")
            return _finalize(status, status_path, 3)
        # (4) last resort — no terminal evidence found after (1)-(3) were
        # all attempted this round.
        if stalled:
            return _finalize(status, status_path, 5)

        sleep(spec.interval)


def _finalize(status: dict, status_path: Path, process_exit_code: int) -> dict:
    status = dict(status)
    status["process_exit_code"] = process_exit_code
    status["ended_at"] = _iso_now()
    _write_status_file(status_path, status)
    status["status_path"] = str(status_path)
    print(_format_status_line(status["job_id"],
                              None if status["iteration"] is None else
                              MetricPoint(status["iteration"], status["total_iterations"],
                                         status["mean_reward"], 0.0, 0.0, 0.0),
                              status["trend"], status["plateau_fired"], status["stalled"]))
    return status


# --------------------------------------------------------------------- CLI

_HEURISTIC_NOTE = ("heuristic, unverified in production (GPU envelope SPENT) "
                   "— see module docstring")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watch",
        description="Observe one already-launched RunPod job: discover it, "
                    "tail its out.log incrementally, parse rsl_rl training "
                    "metrics, and report status (advisory only — supervise's "
                    "own summary JSON remains the authoritative terminal "
                    f"record) until it reaches a terminal state. Plateau/"
                    f"stall defaults are {_HEURISTIC_NOTE}.")
    p.add_argument("--job-id", default=None,
                   help="skip discovery entirely and watch this job id "
                        "(attended use)")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                   help=f"seconds between tail polls (default {DEFAULT_INTERVAL})")
    p.add_argument("--startup-grace", type=int, default=DEFAULT_STARTUP_GRACE,
                   help="seconds to wait for exactly one active job to "
                        f"appear before giving up (default {DEFAULT_STARTUP_GRACE})")
    p.add_argument("--stall-sec", type=int, default=DEFAULT_STALL_SEC,
                   help="seconds with no new iteration before declaring a "
                        f"stall (exit 5) — {_HEURISTIC_NOTE} "
                        f"(default {DEFAULT_STALL_SEC})")
    p.add_argument("--plateau-window", type=int, default=DEFAULT_PLATEAU_WINDOW,
                   help="trailing-iteration window for the plateau heuristic "
                        f"— {_HEURISTIC_NOTE} (default {DEFAULT_PLATEAU_WINDOW})")
    p.add_argument("--plateau-min-delta", type=float, default=DEFAULT_PLATEAU_MIN_DELTA,
                   help="min reward spread over --plateau-window below which "
                        "the plateau heuristic fires (the watcher then exits "
                        "3 IMMEDIATELY, job still running, taking no action "
                        f"on it) — {_HEURISTIC_NOTE} "
                        f"(default {DEFAULT_PLATEAU_MIN_DELTA})")
    p.add_argument("--status-path", default=None,
                   help="default: REPO_ROOT/<local_log_dir>/watch-<job_id>.json")
    return p


def _spec_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> WatchSpec:
    if args.interval <= 0:
        parser.error("--interval must be a positive integer (seconds)")
    if args.startup_grace <= 0:
        parser.error("--startup-grace must be a positive integer (seconds)")
    if args.stall_sec <= 0:
        parser.error("--stall-sec must be a positive integer (seconds)")
    if args.plateau_window <= 0:
        parser.error("--plateau-window must be a positive integer (iterations)")
    if args.plateau_min_delta < 0:
        parser.error("--plateau-min-delta must be non-negative")

    return WatchSpec(job_id=args.job_id, interval=args.interval,
                     startup_grace=args.startup_grace, stall_sec=args.stall_sec,
                     plateau_window=args.plateau_window,
                     plateau_min_delta=args.plateau_min_delta,
                     status_path=args.status_path)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    spec = _spec_from_args(args, parser)
    result = watch(tools.runtime(), spec)
    return int(result["process_exit_code"])


if __name__ == "__main__":
    sys.exit(main())
