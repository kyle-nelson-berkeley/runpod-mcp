"""Unit: deadman.py — the independent Mac-side stop-pod fuse against fake
tools.*/config.* (never a real Runtime/API/SSH stack, never the real
Keychain). Mirrors test_supervise.py's FakeClock/NOLOG patterns.

deadman calls the injected tool surface by MODULE ATTRIBUTE (e.g.
`deadman.tools.stop_pod`, `deadman.config.fetch_api_key`), so every fake here
is wired via `monkeypatch.setattr(deadman.tools, "<name>", fake_fn)` /
`monkeypatch.setattr(deadman.config, "<name>", fake_fn)` — never by patching
the target module's globals directly.

NOT covered here (see the module docstring): the LIVE stop path — no test in
this file makes a real RunPod API call, opens a real SSH connection, or reads
the real macOS Keychain.
"""
import json
import subprocess
from pathlib import Path

import pytest

from runpod_mcp import config, guardrails, ssh, tools
from runpod_mcp import deadman

NOLOG = lambda *a, **kw: None  # noqa: E731 — silence progress lines in tests
ISO_T0 = "2026-07-15T00:00:00+00:00"


class FakeClock:
    """now()/sleep() pair — sleep ADVANCES now(), no real waiting ever."""

    def __init__(self, start: float = 0.0):
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _spec(tmp_path, **kw) -> deadman.ArmSpec:
    defaults = dict(hours=0.0, retries=1, spacing_sec=0,
                    pid_file=tmp_path / "deadman.pid",
                    summary_path=tmp_path / "summary.json", probe_key=False)
    defaults.update(kw)
    return deadman.ArmSpec(**defaults)


def _always_stopped(monkeypatch, status_val="stopped"):
    monkeypatch.setattr(deadman.tools, "stop_pod",
                        lambda rt, force=False: {"status": status_val})


# ================================================================ 1. timing

def test_fuse_fires_only_after_window_elapses(monkeypatch, tmp_path):
    """Money-safety invariant: stop_pod must not be reachable until the
    accumulated sleep time has reached the deadline."""
    clock = FakeClock(start=0.0)
    calls = []

    def fake_stop_pod(rt, force=False):
        calls.append(clock.now())
        # by the time stop_pod is ever called, the clock must have advanced
        # to (at least) the fire deadline — proves no early fire.
        assert clock.now() >= 3600.0
        return {"status": "stopped"}

    monkeypatch.setattr(deadman.tools, "stop_pod", fake_stop_pod)

    spec = _spec(tmp_path, hours=1.0)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
                          now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG)

    assert summary["outcome"] == "stopped"
    assert len(calls) == 1
    assert clock.now() >= 3600.0


def test_poll_sleep_chunks_bounded_by_remaining_time(monkeypatch, tmp_path):
    """The cancel-poll loop must bound each sleep by the remaining time to
    the deadline (never overshoot) — mirrors supervise's equivalent
    money-safety regression guard."""
    clock = FakeClock(start=0.0)
    _always_stopped(monkeypatch)

    spec = _spec(tmp_path, hours=1.0)   # 3600s, chunk=30s -> exactly 120 sleeps
    deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
               now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG)

    assert all(s <= 30.0 for s in clock.sleeps)
    assert clock.now() == pytest.approx(3600.0)


# =============================================================== 2. cancel

def test_cancel_during_sleep_writes_cancelled_summary_and_never_calls_stop(
        monkeypatch, tmp_path):
    def _must_not_stop(*a, **kw):
        pytest.fail("tools.stop_pod must NEVER be called on a cancel-before-fire path")
    monkeypatch.setattr(deadman.tools, "stop_pod", _must_not_stop)

    clock = FakeClock(start=0.0)
    checks = {"n": 0}

    def cancel_requested():
        checks["n"] += 1
        return checks["n"] > 3   # cancel on the 4th sleep-loop check

    spec = _spec(tmp_path, hours=3.0)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
                          now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG,
                          cancel_requested=cancel_requested)

    assert summary["outcome"] == "cancelled"
    assert summary["stop_status"] is None
    assert summary["attempts"] == []
    assert summary["exit_code"] == 0
    assert not spec.pid_file.exists()
    on_disk = json.loads(spec.summary_path.read_text())
    assert on_disk["outcome"] == "cancelled"
    assert on_disk["exit_code"] == 0   # persisted, not just in-memory
    # never reached the full 3h window — cancel fired early
    assert clock.now() < 3600 * 3


def test_cancel_after_fire_start_is_ignored_outcome_stays_fired(monkeypatch, tmp_path):
    """Signal discipline: once firing begins, a (racing) cancel must NEVER
    overwrite the fired outcome. Simulated by flipping the cancel flag inside
    on_fire_start itself — the strongest possible race a real SIGTERM could
    produce — and asserting the outcome is still 'stopped'."""
    calls = []
    monkeypatch.setattr(deadman.tools, "stop_pod",
                        lambda rt, force=False: (calls.append(1), {"status": "stopped"})[1])

    clock = FakeClock(start=0.0)
    cancel_flag = {"requested": False}

    def on_fire_start():
        cancel_flag["requested"] = True   # a SIGTERM landing right as we fire

    spec = _spec(tmp_path, hours=1.0)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
                          now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG,
                          cancel_requested=lambda: cancel_flag["requested"],
                          on_fire_start=on_fire_start)

    assert summary["outcome"] == "stopped"
    assert calls == [1]


# ============================================================ 3. escalation

def test_graceful_failure_escalates_to_force_same_cycle(monkeypatch, tmp_path):
    calls = []

    def stop_pod(rt, force=False):
        calls.append(force)
        if not force:
            raise tools.ToolError("could not verify no jobs are running")
        return {"status": "stopped"}
    monkeypatch.setattr(deadman.tools, "stop_pod", stop_pod)

    spec = _spec(tmp_path, retries=3, spacing_sec=100)
    clock = FakeClock()
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
                          now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG)

    assert calls == [False, True]   # ONE cycle: graceful then forced, no retry needed
    assert summary["outcome"] == "stopped"
    assert summary["stop_status"] == "stopped"
    assert len(summary["attempts"]) == 1
    assert "graceful_error" in summary["attempts"][0]
    assert summary["attempts"][0]["forced"]["status"] == "stopped"
    assert clock.sleeps == []   # succeeded on cycle 1 — no spacing sleep needed


@pytest.mark.parametrize("err", [
    tools.ToolError("could not verify no jobs are running"),
    guardrails.GuardrailError("Job(s) already running: busy."),
    ssh.SSHError("ssh timed out"),
])
def test_escalation_triggers_on_any_exception_type(monkeypatch, tmp_path, err):
    calls = []

    def stop_pod(rt, force=False):
        calls.append(force)
        if not force:
            raise err
        return {"status": "stopped"}
    monkeypatch.setattr(deadman.tools, "stop_pod", stop_pod)

    spec = _spec(tmp_path)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert calls == [False, True]
    assert summary["outcome"] == "stopped"


# ======================================================== 4. success statuses

@pytest.mark.parametrize("status_val", ["stopped", "already_stopped", "no_pod"])
def test_success_statuses_all_count_as_stopped(monkeypatch, tmp_path, status_val):
    _always_stopped(monkeypatch, status_val)
    spec = _spec(tmp_path)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert summary["outcome"] == "stopped"
    assert summary["stop_status"] == status_val
    assert summary["exit_code"] == 0
    assert summary["pod_may_still_be_running"] is False


# ============================================================ 5. retry cycles

def test_retry_cycles_respect_spacing_and_count(monkeypatch, tmp_path):
    forces = []

    def stop_pod(rt, force=False):
        forces.append(force)
        raise tools.ToolError("still busy")
    monkeypatch.setattr(deadman.tools, "stop_pod", stop_pod)

    clock = FakeClock()
    spec = _spec(tmp_path, retries=3, spacing_sec=120)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=clock.sleep,
                          now=clock.now, iso_now=lambda: ISO_T0, log=NOLOG)

    assert summary["outcome"] == "stop_failed"
    assert summary["stop_status"] is None
    assert summary["pod_may_still_be_running"] is True
    assert summary["exit_code"] == 1
    assert len(summary["attempts"]) == 3
    # graceful+forced attempted every cycle, 3 cycles
    assert forces == [False, True] * 3
    # spacing sleeps: exactly 2 (after cycle 1 and 2), NONE trailing cycle 3
    assert clock.sleeps == [120, 120]
    assert not spec.pid_file.exists()   # removed even on exhaustion
    on_disk = json.loads(spec.summary_path.read_text())
    assert on_disk["exit_code"] == 1   # persisted on the stop_failed path too


def test_runtime_construction_failure_counts_as_failed_attempt_not_a_crash(
        monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("keychain locked mid-fire")

    calls = {"stop_pod": 0}
    monkeypatch.setattr(deadman.tools, "stop_pod",
                        lambda rt, force=False: calls.__setitem__("stop_pod", calls["stop_pod"] + 1)
                        or {"status": "stopped"})

    spec = _spec(tmp_path, retries=2, spacing_sec=0)
    summary = deadman.arm(spec, rt_factory=boom, sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    assert summary["outcome"] == "stop_failed"
    assert calls["stop_pod"] == 0   # never reached — rt_factory always failed
    assert len(summary["attempts"]) == 2
    assert all("error" in a for a in summary["attempts"])


# =========================================== 6. pid-file lifecycle + TOCTOU

def test_arm_refuses_when_live_deadman_holds_pid_file(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 999, "armed_at": ISO_T0, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: pid == 999)

    spec = _spec(tmp_path, pid_file=pid_file)
    with pytest.raises(deadman.ArmRefused, match="already holds"):
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert pid_file.exists()   # untouched — still the live deadman's file
    assert json.loads(pid_file.read_text())["pid"] == 999


def test_arm_reaps_stale_pid_file_with_dead_process_loudly(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 999, "armed_at": ISO_T0, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: False)
    _always_stopped(monkeypatch)

    log_lines = []
    spec = _spec(tmp_path, pid_file=pid_file)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=log_lines.append)

    assert summary["outcome"] == "stopped"
    assert any("reaping stale pid file" in line for line in log_lines)


def test_pid_file_toctou_race_refuses(monkeypatch, tmp_path):
    """If os.open(O_EXCL) races (another arm() won concurrently), refuse
    cleanly rather than clobbering the winner's pid file."""
    pid_file = tmp_path / "deadman.pid"

    def fake_open(path, flags):
        raise FileExistsError()
    monkeypatch.setattr(deadman.os, "open", fake_open)

    spec = _spec(tmp_path, pid_file=pid_file)
    with pytest.raises(deadman.ArmRefused, match="concurrently"):
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert not pid_file.exists()


def test_is_deadman_process_requires_both_alive_and_cmdline_match(monkeypatch):
    monkeypatch.setattr(deadman, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(deadman, "_pid_cmdline", lambda pid: "python -m some.other.module")
    assert deadman._is_deadman_process(12345) is False   # PID reuse — cmdline mismatch

    monkeypatch.setattr(deadman, "_pid_cmdline",
                        lambda pid: "/venv/bin/python -m runpod_mcp.deadman arm --hours 3")
    assert deadman._is_deadman_process(12345) is True

    monkeypatch.setattr(deadman, "_pid_alive", lambda pid: False)
    assert deadman._is_deadman_process(12345) is False   # dead, regardless of cmdline

    assert deadman._is_deadman_process(None) is False


def test_pid_cmdline_uses_ps_p_o_command_idiom(monkeypatch):
    recorded = {}

    def fake_run(argv, **kw):
        recorded["argv"] = argv
        return subprocess.CompletedProcess(argv, 0,
                                           stdout="python -m runpod_mcp.deadman arm\n", stderr="")
    assert deadman._pid_cmdline(4242, run=fake_run) == "python -m runpod_mcp.deadman arm"
    assert recorded["argv"] == ["ps", "-p", "4242", "-o", "command="]


def test_pid_cmdline_never_raises_on_ps_failure(monkeypatch):
    def fake_run(argv, **kw):
        raise OSError("ps not found")
    assert deadman._pid_cmdline(4242, run=fake_run) == ""


# ------------------------------------------------ 6b. corrupt pid files

CORRUPT_PID_CONTENTS = [
    pytest.param(json.dumps({"pid": "not-a-number", "fire_at": ISO_T0}),
                 id="non-int-pid"),
    pytest.param("{{{ this is not json", id="malformed-json"),
]


def test_pid_alive_is_defensive_against_non_int_input():
    """os.kill(non-int, 0) raises TypeError/ValueError — _pid_alive must
    swallow those and answer False, never crash (a corrupt pid file must not
    brick arm/cancel/status until manually removed)."""
    assert deadman._pid_alive("not-a-number") is False
    assert deadman._pid_alive(None) is False


@pytest.mark.parametrize("content", CORRUPT_PID_CONTENTS)
def test_read_pid_file_normalizes_corrupt_pid_to_none(tmp_path, content):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(content)
    assert deadman._read_pid_file(pid_file).get("pid") is None


@pytest.mark.parametrize("content", CORRUPT_PID_CONTENTS)
def test_corrupt_pid_file_arm_reaps_loudly_and_proceeds(monkeypatch, tmp_path, content):
    """Uses the REAL _is_deadman_process/_pid_alive path — proves no
    TypeError escapes on a corrupt pid file; the file is stale, reaped, and
    the arm proceeds."""
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(content)
    _always_stopped(monkeypatch)

    log_lines = []
    spec = _spec(tmp_path, pid_file=pid_file)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=log_lines.append)

    assert summary["outcome"] == "stopped"
    assert any("reaping stale pid file" in line for line in log_lines)
    assert not pid_file.exists()   # consumed by the successful run


@pytest.mark.parametrize("content", CORRUPT_PID_CONTENTS)
def test_corrupt_pid_file_cancel_is_stale_noop_never_signals(monkeypatch, tmp_path, content):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(content)
    killed = []
    monkeypatch.setattr(deadman.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    result = deadman.cancel(pid_file=pid_file, log=NOLOG)

    assert killed == []
    assert not pid_file.exists()   # reaped
    assert result["outcome"] == "not_armed"
    assert result["exit_code"] == 0


@pytest.mark.parametrize("content", CORRUPT_PID_CONTENTS)
def test_corrupt_pid_file_status_reads_lost_exit_1(tmp_path, content):
    """We cannot prove the fuse is alive from a corrupt pid file, so status
    must alert (LOST, exit 1) — never crash, never report armed."""
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(content)

    result = deadman.status(pid_file=pid_file, log=NOLOG)

    assert result["state"] == "LOST"
    assert result["exit_code"] == 1
    assert "may still be running" in result["message"]


# ============================================================= 7. key probe

def test_key_probe_failure_refuses_arm_and_removes_pid_file(monkeypatch, tmp_path):
    def boom():
        raise config.ConfigError("Keychain lookup failed for rpa_SECRETVALUE123")
    monkeypatch.setattr(deadman.config, "fetch_api_key", boom)

    pid_file = tmp_path / "deadman.pid"
    log_lines = []
    spec = _spec(tmp_path, pid_file=pid_file, probe_key=True)
    with pytest.raises(deadman.ArmRefused) as exc_info:
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: ISO_T0, log=log_lines.append)

    assert not pid_file.exists()
    assert "rpa_SECRETVALUE123" not in str(exc_info.value)
    assert all("rpa_SECRETVALUE123" not in line for line in log_lines)


def test_key_probe_success_value_never_leaks_into_log_or_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(deadman.config, "fetch_api_key", lambda: "rpa_FAKEKEYVALUE456")
    _always_stopped(monkeypatch)

    log_lines = []
    spec = _spec(tmp_path, probe_key=True)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=log_lines.append)

    assert "rpa_FAKEKEYVALUE456" not in json.dumps(summary)
    assert all("rpa_FAKEKEYVALUE456" not in line for line in log_lines)
    assert "rpa_FAKEKEYVALUE456" not in spec.summary_path.read_text()


def test_no_probe_key_skips_the_keychain_entirely(monkeypatch, tmp_path):
    called = {"n": 0}

    def boom():
        called["n"] += 1
        raise config.ConfigError("must never be called")
    monkeypatch.setattr(deadman.config, "fetch_api_key", boom)
    _always_stopped(monkeypatch)

    spec = _spec(tmp_path, probe_key=False)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert called["n"] == 0
    assert summary["outcome"] == "stopped"


def test_key_probe_refusal_writes_durable_refused_summary_status_reports_it(
        monkeypatch, tmp_path):
    """The refusal must survive the (possibly lost) stdout of a backgrounded
    arm: a durable summary with outcome 'refused' is written, and a later
    status reports the reason with exit 0 (a refused arm never touched the
    pod) — not plain not_armed."""
    def boom():
        raise config.ConfigError("Keychain lookup failed (rc=44) for rpa_SECRET999")
    monkeypatch.setattr(deadman.config, "fetch_api_key", boom)

    pid_file = tmp_path / "deadman.pid"
    summary_path = tmp_path / "deadman-20260715-000000.json"
    spec = _spec(tmp_path, pid_file=pid_file, summary_path=summary_path,
                probe_key=True)
    with pytest.raises(deadman.ArmRefused):
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    assert not pid_file.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["outcome"] == "refused"
    assert on_disk["exit_code"] == 2
    assert "Keychain key probe failed" in on_disk["reason"]
    assert "rpa_SECRET999" not in summary_path.read_text()   # scrubbed

    result = deadman.status(pid_file=pid_file,
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "refused"
    assert result["exit_code"] == 0
    assert "Keychain key probe failed" in result["reason"]


def test_key_probe_refusal_writes_default_stamped_summary(monkeypatch, tmp_path):
    """Without an explicit --summary-path the refusal summary must land at
    the default-stamped deadman-<stamp>.json under REPO_ROOT/logs/pod — the
    path status's default glob actually searches."""
    monkeypatch.setattr(deadman.config, "REPO_ROOT", tmp_path)

    def boom():
        raise config.ConfigError("Keychain lookup failed")
    monkeypatch.setattr(deadman.config, "fetch_api_key", boom)

    spec = deadman.ArmSpec(hours=3.0, retries=5, spacing_sec=300, probe_key=True)
    with pytest.raises(deadman.ArmRefused):
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: "2026-07-15T12:00:00+00:00",
                   log=NOLOG)

    expected = tmp_path / "logs" / "pod" / "deadman-20260715-120000.json"
    assert expected.exists()
    assert json.loads(expected.read_text())["outcome"] == "refused"

    result = deadman.status(log=NOLOG)   # default pid file + default glob
    assert result["state"] == "refused"
    assert result["exit_code"] == 0


def test_refusal_summary_write_failure_never_masks_the_refusal(monkeypatch, tmp_path):
    """Same never-crash-on-the-way-out discipline as every other summary
    write: if the refusal summary can't be written, arm still refuses with
    ArmRefused (exit 2) — the write failure is logged, not raised."""
    def boom():
        raise config.ConfigError("Keychain lookup failed")
    monkeypatch.setattr(deadman.config, "fetch_api_key", boom)

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    spec = _spec(tmp_path, summary_path=blocker / "s.json", probe_key=True)

    log_lines = []
    with pytest.raises(deadman.ArmRefused, match="Keychain key probe failed"):
        deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                   now=lambda: 0.0, iso_now=lambda: ISO_T0, log=log_lines.append)
    assert any("failed to write summary" in line for line in log_lines)


# ============================================================= 8. summary

def test_summary_field_contract_present_in_memory_and_on_disk(monkeypatch, tmp_path):
    _always_stopped(monkeypatch)
    spec = _spec(tmp_path, retries=2, spacing_sec=7)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    required = ("armed_at", "fire_at", "hours", "retries", "spacing_sec",
               "outcome", "stop_status", "attempts", "ended_at", "pid",
               "exit_code")
    for key in required:
        assert key in summary, key
    on_disk = json.loads(spec.summary_path.read_text())
    for key in required:
        assert key in on_disk, key   # exit_code must be PERSISTED, not post-hoc
    assert on_disk["retries"] == 2
    assert on_disk["spacing_sec"] == 7
    assert on_disk["exit_code"] == 0   # stopped path


def test_summary_write_failure_never_masks_the_real_outcome(monkeypatch, tmp_path):
    _always_stopped(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad_summary_path = blocker / "summary.json"   # parent.mkdir will fail

    spec = _spec(tmp_path, summary_path=bad_summary_path)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    # the write failed silently-to-disk but the RETURNED outcome is still
    # the true one — a write failure must never flip stopped -> stop_failed.
    assert summary["outcome"] == "stopped"
    assert summary["exit_code"] == 0
    assert "summary_path" not in summary   # write never succeeded


def test_default_pid_and_summary_paths_are_repo_root_anchored(monkeypatch, tmp_path):
    monkeypatch.setattr(deadman.config, "REPO_ROOT", tmp_path)
    _always_stopped(monkeypatch)

    spec = deadman.ArmSpec(hours=0.0, retries=1, spacing_sec=0, probe_key=False)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: "2026-07-15T12:00:00+00:00",
                          log=NOLOG)

    assert summary["outcome"] == "stopped"
    expected_pid_file = tmp_path / "logs" / "pod" / "deadman.pid"
    assert not expected_pid_file.exists()   # removed on success
    expected_summary = tmp_path / "logs" / "pod" / "deadman-20260715-120000.json"
    assert expected_summary.exists()


# ============================== 8b. fail-safe ordering: summary before pid file

def _blocked_summary_path(tmp_path):
    """A summary path whose parent.mkdir will fail (parent is a file)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    return blocker / "summary.json"


@pytest.mark.parametrize("stop_behavior,expected_outcome,expected_exit", [
    ("succeed", "stopped", 0),
    ("fail", "stop_failed", 1),
])
def test_fire_summary_write_failure_leaves_pid_file_and_status_reads_lost(
        monkeypatch, tmp_path, stop_behavior, expected_outcome, expected_exit):
    """If the fire summary can't be written (stopped AND stop_failed paths),
    the pid file must be LEFT IN PLACE: once this process exits, a later
    `status` then reads LOST (exit 1, check manually) — the fail-safe state.
    Remove-then-write would instead report not_armed/an older success in the
    exact pod-may-still-be-billing scenario."""
    if stop_behavior == "succeed":
        _always_stopped(monkeypatch)
    else:
        def stop_pod(rt, force=False):
            raise tools.ToolError("API unreachable")
        monkeypatch.setattr(deadman.tools, "stop_pod", stop_pod)

    pid_file = tmp_path / "deadman.pid"
    log_lines = []
    spec = _spec(tmp_path, retries=1, pid_file=pid_file,
                summary_path=_blocked_summary_path(tmp_path))
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0,
                          log=log_lines.append)

    # the real outcome/exit code are unchanged by the write failure...
    assert summary["outcome"] == expected_outcome
    assert summary["exit_code"] == expected_exit
    # ...but the pid file survives as the LOST beacon
    assert pid_file.exists()
    assert any("leaving the pid file in place" in line for line in log_lines)

    # once the armed process is gone, status must read LOST, never not_armed
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: False)
    result = deadman.status(pid_file=pid_file,
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "LOST"
    assert result["exit_code"] == 1


def test_cancel_summary_write_failure_leaves_pid_file_and_status_reads_lost(
        monkeypatch, tmp_path):
    """Cancel-path equivalent: a cancel whose summary write fails must
    degrade to LOST, not to silence."""
    def _must_not_stop(*a, **kw):
        pytest.fail("stop_pod must not be called on the cancel path")
    monkeypatch.setattr(deadman.tools, "stop_pod", _must_not_stop)

    pid_file = tmp_path / "deadman.pid"
    log_lines = []
    spec = _spec(tmp_path, hours=3.0, pid_file=pid_file,
                summary_path=_blocked_summary_path(tmp_path))
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0,
                          log=log_lines.append, cancel_requested=lambda: True)

    assert summary["outcome"] == "cancelled"
    assert summary["exit_code"] == 0
    assert pid_file.exists()   # left as the LOST beacon
    assert any("leaving the pid file in place" in line for line in log_lines)

    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: False)
    result = deadman.status(pid_file=pid_file,
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "LOST"
    assert result["exit_code"] == 1


def test_fire_path_writes_summary_before_removing_pid_file(monkeypatch, tmp_path):
    """Ordering pin: on the fire path the durable summary hits disk BEFORE
    the pid file disappears, so a crash between the two degrades to LOST."""
    _always_stopped(monkeypatch)
    events = []
    real_write, real_remove = deadman._write_summary, deadman._remove

    def recording_write(path, summary, log):
        events.append(("write", path.name))
        return real_write(path, summary, log)

    def recording_remove(path):
        events.append(("remove", path.name))
        real_remove(path)

    monkeypatch.setattr(deadman, "_write_summary", recording_write)
    monkeypatch.setattr(deadman, "_remove", recording_remove)

    spec = _spec(tmp_path, summary_path=tmp_path / "deadman-20260715-000000.json")
    deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
               now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    assert events == [("write", "deadman-20260715-000000.json"),
                      ("remove", "deadman.pid")]


def test_cancel_path_writes_summary_before_removing_pid_file(monkeypatch, tmp_path):
    events = []
    real_write, real_remove = deadman._write_summary, deadman._remove

    def recording_write(path, summary, log):
        events.append(("write", path.name))
        return real_write(path, summary, log)

    def recording_remove(path):
        events.append(("remove", path.name))
        real_remove(path)

    monkeypatch.setattr(deadman, "_write_summary", recording_write)
    monkeypatch.setattr(deadman, "_remove", recording_remove)

    spec = _spec(tmp_path, hours=3.0,
                summary_path=tmp_path / "deadman-20260715-000000.json")
    deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
               now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG,
               cancel_requested=lambda: True)

    assert events == [("write", "deadman-20260715-000000.json"),
                      ("remove", "deadman.pid")]


# ============================================================== 9. cancel()

def test_cancel_no_pid_file_is_idempotent_noop(tmp_path):
    result = deadman.cancel(pid_file=tmp_path / "nope.pid", log=NOLOG)
    assert result == {"outcome": "not_armed",
                      "message": "no pid file — nothing to cancel", "exit_code": 0}


def test_cancel_sends_sigterm_only_to_a_verified_live_deadman(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 4242, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: pid == 4242)
    killed = []
    monkeypatch.setattr(deadman.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    result = deadman.cancel(pid_file=pid_file, log=NOLOG)

    assert killed == [(4242, deadman.signal.SIGTERM)]
    assert result["outcome"] == "cancel_requested"
    assert result["exit_code"] == 0


def test_cancel_never_signals_an_unverified_pid_pid_reuse_hazard(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 55, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: False)
    killed = []
    monkeypatch.setattr(deadman.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    result = deadman.cancel(pid_file=pid_file, log=NOLOG)

    assert killed == []
    assert not pid_file.exists()
    assert result["outcome"] == "not_armed"
    assert result["exit_code"] == 0


# ============================================================== 10. status()

def test_status_armed(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 111, "armed_at": "2026-07-15T00:00:00+00:00",
                                    "fire_at": "2026-07-15T03:00:00+00:00"}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: pid == 111)

    result = deadman.status(pid_file=pid_file, now=lambda: "2026-07-15T01:00:00+00:00",
                            log=NOLOG)

    assert result["state"] == "armed"
    assert result["pid"] == 111
    assert result["remaining_minutes"] == pytest.approx(120.0, abs=0.1)
    assert result["exit_code"] == 0


def test_status_lost_when_pid_file_process_is_dead(monkeypatch, tmp_path):
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 111, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: False)

    result = deadman.status(pid_file=pid_file, log=NOLOG)

    assert result["state"] == "LOST"
    assert result["exit_code"] == 1
    assert "may still be running" in result["message"]


def test_status_lost_never_masquerades_as_armed_when_cmdline_mismatches(monkeypatch, tmp_path):
    """PID reuse: a live pid whose cmdline is NOT ours must report LOST, not
    armed — the exact false-comfort failure mode this tool exists to kill."""
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 111, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(deadman, "_pid_cmdline", lambda pid: "/usr/bin/some-other-daemon")

    result = deadman.status(pid_file=pid_file, log=NOLOG)
    assert result["state"] == "LOST"
    assert result["exit_code"] == 1


def test_status_not_armed_no_pidfile_no_summary(tmp_path):
    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result == {"state": "not_armed", "message": "no pid file, no prior summary",
                      "exit_code": 0}


def test_status_reports_latest_summary_when_no_pidfile(tmp_path):
    (tmp_path / "deadman-20260714-000000.json").write_text(
        json.dumps({"outcome": "cancelled"}))
    (tmp_path / "deadman-20260715-000000.json").write_text(
        json.dumps({"outcome": "stopped", "ended_at": "..."}))

    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)

    assert result["state"] == "stopped"   # the later (lexicographically last) summary
    assert result["exit_code"] == 0


def test_status_after_stop_failed_summary_exits_1_with_warning(tmp_path):
    """The exhausted-fire path removes the pid file, so a later status lands
    on the last-summary branch — a stop_failed outcome there MUST exit 1 (the
    pod may still be running/billing); an exit-0 would silence status-based
    monitoring in the exact case it exists for."""
    (tmp_path / "deadman-20260715-000000.json").write_text(
        json.dumps({"outcome": "stop_failed", "pod_may_still_be_running": True}))

    log_lines = []
    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"),
                            log=log_lines.append)

    assert result["state"] == "stop_failed"
    assert result["exit_code"] == 1
    assert "may still be running" in result["message"]
    assert any("may still be running" in line for line in log_lines)


def test_status_after_stopped_summary_exits_0(tmp_path):
    (tmp_path / "deadman-20260715-000000.json").write_text(
        json.dumps({"outcome": "stopped", "stop_status": "stopped"}))

    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "stopped"
    assert result["exit_code"] == 0


def test_status_ordering_older_stop_failed_newer_stopped_exits_0(tmp_path):
    """Deterministic 'most recent' selection: default summary filenames embed
    a zero-padded UTC stamp, so lexicographic sort = chronological — a NEWER
    stopped supersedes an OLDER stop_failed."""
    (tmp_path / "deadman-20260714-090000.json").write_text(
        json.dumps({"outcome": "stop_failed"}))
    (tmp_path / "deadman-20260715-120000.json").write_text(
        json.dumps({"outcome": "stopped"}))

    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "stopped"
    assert result["exit_code"] == 0


def test_status_ordering_older_stopped_newer_stop_failed_exits_1(tmp_path):
    (tmp_path / "deadman-20260714-090000.json").write_text(
        json.dumps({"outcome": "stopped"}))
    (tmp_path / "deadman-20260715-120000.json").write_text(
        json.dumps({"outcome": "stop_failed"}))

    result = deadman.status(pid_file=tmp_path / "nope.pid",
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "stop_failed"
    assert result["exit_code"] == 1


def test_exhausted_fire_then_status_end_to_end_exits_1(monkeypatch, tmp_path):
    """The exact monitoring scenario from the review finding, end to end: a
    fuse exhausts its retries (pid file removed, stop_failed summary written)
    -> a later `status` over the same paths must exit 1, never 0."""
    def stop_pod(rt, force=False):
        raise tools.ToolError("API unreachable")
    monkeypatch.setattr(deadman.tools, "stop_pod", stop_pod)

    pid_file = tmp_path / "deadman.pid"
    summary_path = tmp_path / "deadman-20260715-000000.json"
    spec = _spec(tmp_path, retries=2, spacing_sec=0, pid_file=pid_file,
                summary_path=summary_path)
    fired = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                        now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)
    assert fired["outcome"] == "stop_failed"
    assert not pid_file.exists()

    result = deadman.status(pid_file=pid_file,
                            summary_glob=str(tmp_path / "deadman-*.json"), log=NOLOG)
    assert result["state"] == "stop_failed"
    assert result["exit_code"] == 1


# ========================================================= 11. CLI / argparse

@pytest.mark.parametrize("bad", ["0", "-1"])
def test_arm_non_positive_hours_is_argparse_error(bad):
    with pytest.raises(SystemExit) as exc_info:
        deadman.main(["arm", "--hours", bad, "--no-probe-key"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_arm_non_positive_retries_is_argparse_error(bad):
    with pytest.raises(SystemExit) as exc_info:
        deadman.main(["arm", "--retries", bad, "--no-probe-key"])
    assert exc_info.value.code == 2


def test_arm_negative_spacing_sec_is_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        deadman.main(["arm", "--spacing-sec", "-1", "--no-probe-key"])
    assert exc_info.value.code == 2


def test_missing_subcommand_is_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        deadman.main([])
    assert exc_info.value.code == 2


def test_main_status_wiring_no_pidfile(tmp_path, capsys):
    rc = deadman.main(["status", "--pid-file", str(tmp_path / "d.pid"),
                       "--summary-glob", str(tmp_path / "deadman-*.json")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "not_armed"


def test_main_cancel_wiring_no_pidfile(tmp_path, capsys):
    rc = deadman.main(["cancel", "--pid-file", str(tmp_path / "d.pid")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "not_armed"


def test_main_arm_refusal_exit_code_and_json(monkeypatch, tmp_path, capsys):
    """End-to-end through main() -> _main_arm(): a live pid file refuses with
    exit 2 and never touches tools.runtime (the real Runtime factory)."""
    pid_file = tmp_path / "deadman.pid"
    pid_file.write_text(json.dumps({"pid": 999, "armed_at": ISO_T0, "fire_at": ISO_T0}))
    monkeypatch.setattr(deadman, "_is_deadman_process", lambda pid: pid == 999)

    def _must_not_build_runtime():
        pytest.fail("main()'s refusal path must never touch tools.runtime")
    monkeypatch.setattr(deadman.tools, "runtime", _must_not_build_runtime)

    rc = deadman.main(["arm", "--hours", "3.0", "--no-probe-key",
                       "--pid-file", str(pid_file),
                       "--summary-path", str(tmp_path / "s.json")])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "refused"


# =============================================================== 12. deadman.sh

def test_deadman_sh_is_syntactically_valid():
    script = Path(__file__).resolve().parents[1] / "deadman.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
