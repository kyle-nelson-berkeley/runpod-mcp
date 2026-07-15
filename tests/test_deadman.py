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


# ============================================================= 8. summary

def test_summary_field_contract_present_in_memory_and_on_disk(monkeypatch, tmp_path):
    _always_stopped(monkeypatch)
    spec = _spec(tmp_path, retries=2, spacing_sec=7)
    summary = deadman.arm(spec, rt_factory=lambda: object(), sleep=lambda s: None,
                          now=lambda: 0.0, iso_now=lambda: ISO_T0, log=NOLOG)

    required = ("armed_at", "fire_at", "hours", "retries", "spacing_sec",
               "outcome", "stop_status", "attempts", "ended_at", "pid")
    for key in required:
        assert key in summary, key
    on_disk = json.loads(spec.summary_path.read_text())
    for key in required:
        assert key in on_disk, key
    assert on_disk["retries"] == 2
    assert on_disk["spacing_sec"] == 7


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
