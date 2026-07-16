"""Unit: supervise.py — the one-shot supervised-job CLI against fake tools.*.

supervise() calls the injected tools/training/guardrails/jobs/ssh surface by
MODULE ATTRIBUTE (e.g. `supervise.tools.pod_status`), so every fake here is
wired via `monkeypatch.setattr(supervise.tools, "<name>", fake_fn)` — never by
patching the target module's globals directly. `rt.ssh.rsync_pull` is the one
call that bypasses `tools.*` (per the plan, the job-dir pull is unconditional
and goes straight through the Runtime's ssh client), so it is exercised via a
purpose-built recording SSH fake that subclasses tests.test_tools.FakeSSH.
"""
import json

import pytest

from runpod_mcp import config, guardrails, jobs, ssh, tools, training
from runpod_mcp import supervise
from tests.test_tools import FakeSSH


# --------------------------------------------------------------------- clock

class FakeClock:
    """now()/sleep() pair — sleep ADVANCES now(), no real waiting ever."""

    def __init__(self, start: float = 1000.0):
        self.t = start
        self.sleeps = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


NOLOG = lambda *a, **kw: None  # noqa: E731 — silence progress lines in tests


# ----------------------------------------------------------------------- ssh

class RecordingSSH(FakeSSH):
    """Records rsync_pull into the SAME shared `calls` list the tools.* fakes
    use, so ordering assertions can see all four capture/stop steps in one
    place (the stock FakeSSH.rsyncs list is separate and insufficient here)."""

    def __init__(self, calls, run_results=None):
        super().__init__(run_results)
        self.calls = calls

    def rsync_pull(self, host, port, remote_dir, local_dir, timeout=600):
        self.calls.append(("rsync_pull", {"remote_dir": remote_dir,
                                          "local_dir": str(local_dir)}))
        self.rsyncs.append((remote_dir, str(local_dir)))
        return "sent"


class RaisingRsyncSSH(RecordingSSH):
    """Same recording behavior, but rsync_pull raises instead of succeeding —
    for the capture-breadth (non-ToolError-still-reaches-stop) test."""

    def __init__(self, calls, exc, run_results=None):
        super().__init__(calls, run_results)
        self._exc = exc

    def rsync_pull(self, host, port, remote_dir, local_dir, timeout=600):
        self.calls.append(("rsync_pull", {"remote_dir": remote_dir}))
        raise self._exc


def _make_rt(calls, ssh_results=None, local_log_dir=None, sshc=None):
    cfg = config.load_defaults()
    cfg["ssh_identity"] = "~/.ssh/id_ed25519"
    if local_log_dir is not None:
        cfg["local_log_dir"] = local_log_dir
    rt = tools.Runtime(cfg=cfg, client=None,
                       sshc=sshc if sshc is not None else RecordingSSH(calls, ssh_results),
                       sleep=lambda s: None, gpu_types=lambda **kw: [],
                       ssh_pubkey="ssh-ed25519 AAAA test@mac")
    return rt


# ----------------------------------------------------------- tools.* fakes

def make_launch_training(monkeypatch, *, dry_run_error=None, launch_error=None,
                         job_id="20260710-100000_train-curee-dr0-s1_ab12",
                         ceiling=3600):
    def fn(rt, vehicle, dr_level, seed=1, auto_stop=False, extra_args="",
          force=False, dry_run=False):
        if dry_run:
            if dry_run_error:
                raise dry_run_error
            return {"dry_run": True, "vehicle": vehicle, "dr_level": dr_level,
                    "max_runtime_sec": ceiling}
        if launch_error:
            raise launch_error
        assert auto_stop is False   # supervise must NEVER launch with auto_stop=True
        return {"job_id": job_id}
    monkeypatch.setattr(supervise.tools, "launch_training", fn)


def make_run_job(monkeypatch, *, dry_run_error=None, launch_error=None,
                 job_id="20260710-100000_eval_cd34", ceiling=3600):
    def fn(rt, name, command, workdir="/workspace", auto_stop=False,
          force=False, dry_run=False, max_runtime_sec=None):
        if dry_run:
            if dry_run_error:
                raise dry_run_error
            return {"dry_run": True, "max_runtime_sec": ceiling}
        if launch_error:
            raise launch_error
        assert auto_stop is False
        return {"job_id": job_id}
    monkeypatch.setattr(supervise.tools, "run_job", fn)


def make_job_status(monkeypatch, states):
    """states: list of dicts (job_status results) or Exception instances (to
    raise on that poll). Exhausted lists repeat their last element forever."""
    calls = {"i": 0}

    def fn(rt, job_id, tail_lines=40):
        i = min(calls["i"], len(states) - 1)
        calls["i"] += 1
        item = states[i]
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(supervise.tools, "job_status", fn)


def make_capture_fakes(monkeypatch, calls, *, sync_error=None, spend_error=None,
                       stop_error=None, force_stop_error=None):
    def sync_logs(rt, subdir):
        calls.append(("sync_logs", {"subdir": subdir}))
        if sync_error:
            raise sync_error
        return {"remote": f"/workspace/IsaacLab/logs/{subdir}/",
                "local_dir": f"logs/pod/{subdir}", "rsync_output": "sent"}

    def spend_report(rt):
        calls.append(("spend_report", {}))
        if spend_error:
            raise spend_error
        return {"total_usd": 1.23, "budget_usd": 50}

    def stop_pod(rt, force=False):
        calls.append(("stop_pod", {"force": force}))
        if force and force_stop_error:
            raise force_stop_error
        if not force and stop_error:
            raise stop_error
        return {"status": "stopped"}

    def conn_info(rt):
        return ("1.2.3.4", 2222)

    monkeypatch.setattr(supervise.tools, "sync_logs", sync_logs)
    monkeypatch.setattr(supervise.tools, "spend_report", spend_report)
    monkeypatch.setattr(supervise.tools, "stop_pod", stop_pod)
    monkeypatch.setattr(supervise.tools, "_conn_info", conn_info)


# --------------------------------------------------------------- pod-not-live

def _running(rt=None):
    return {"status": "running"}


@pytest.fixture(autouse=True)
def _guard_terminate_never_called(monkeypatch):
    """Every single test in this module implicitly asserts terminate_pod is
    never called by supervise — a spec-mandated invariant (money safety)."""
    def fn(rt, confirm):
        pytest.fail("tools.terminate_pod must NEVER be called by supervise")
    monkeypatch.setattr(supervise.tools, "terminate_pod", fn)
    yield


# ============================================================= 1. happy path

def test_full_run_success_exact_order_and_exit_zero(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch, job_id="20260710-100000_train-curee-dr0-s1_ab12")
    make_job_status(monkeypatch, [
        {"state": "running"},
        {"state": "running"},
        {"state": "succeeded", "exit_code": 0, "latest_reward_line": "Mean reward: 97.2"},
    ])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0", seed=1)
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    order = [name for name, _ in calls]
    assert order == ["rsync_pull", "sync_logs", "spend_report", "stop_pod"]
    assert calls[-1] == ("stop_pod", {"force": False})
    assert summary["process_exit_code"] == 0
    assert summary["state"] == "succeeded"
    assert summary["job_id"] == "20260710-100000_train-curee-dr0-s1_ab12"
    assert summary["latest_reward_line"] == "Mean reward: 97.2"

    summary_file = tmp_path / f"supervise-{summary['job_id']}.json"
    assert summary_file.exists()
    assert json.loads(summary_file.read_text())["process_exit_code"] == 0


# ============================================================ 2. job failed

def test_failed_job_still_captures_and_stops_nonzero_exit(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [
        {"state": "running"},
        {"state": "failed", "exit_code": 1},
    ])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["state"] == "failed"
    assert summary["exit_code"] == 1
    assert summary["process_exit_code"] != 0
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]
    sync_calls = [name for name, _ in calls if name == "sync_logs"]
    assert sync_calls == ["sync_logs"]


# ================================================== 3. max-wait exceeded

def test_max_wait_exceeded_while_running_force_stops(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "running"}])   # never terminates
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0",
                             max_wait=100, interval=45)
    clock = FakeClock(start=0.0)
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["force_stopped"] is True
    assert summary["max_wait_sec"] == 100
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": True}]
    assert summary["process_exit_code"] != 0
    assert clock.now() >= 100


def test_poll_sleep_never_overshoots_finite_cap(monkeypatch, tmp_path):
    """Regression guard (money-safety): when --interval is much larger than
    the remaining time to the deadline, the sleep MUST be bounded by that
    remaining time — otherwise the pod bills a whole extra interval past
    --max-wait before force-stopping."""
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "running"}])   # never terminates
    make_capture_fakes(monkeypatch, calls)

    # interval (100000) >> max_wait (100): the buggy full-interval sleep would
    # drive the clock to ~100000 before the next deadline check.
    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0",
                             max_wait=100, interval=100000)
    clock = FakeClock(start=0.0)
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["force_stopped"] is True
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": True}]
    # bounded sleep => clock lands right at the deadline (100), never near one
    # raw interval (100000).
    assert clock.now() >= 100
    assert clock.now() < 1000


# ===================================================== 4. pod not running

def test_pod_not_running_refuses_no_launch_attempted(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", lambda rt: {"status": "stopped"})

    def _must_not_launch(*a, **kw):
        raise AssertionError("launch_training must not be called when pod isn't running")
    monkeypatch.setattr(supervise.tools, "launch_training", _must_not_launch)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["process_exit_code"] == 2
    assert summary["refused"] is True
    assert summary["job_id"] is None
    assert calls == []


def test_pod_running_ssh_pending_also_refuses(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status",
                        lambda rt: {"status": "running_ssh_pending"})

    def _must_not_launch(*a, **kw):
        raise AssertionError("must not launch on running_ssh_pending")
    monkeypatch.setattr(supervise.tools, "launch_training", _must_not_launch)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
    assert summary["process_exit_code"] == 2
    assert summary["refused"] is True


# =========================================== 5. launch raises REFUSE_ERRORS

@pytest.mark.parametrize("err", [
    tools.ToolError("bluerov2 training refused: markers/axis_sanity_PASS is missing"),
    guardrails.GuardrailError("Job(s) already running: busy. One job at a time."),
])
def test_launch_refusal_no_poll_no_stop(monkeypatch, tmp_path, err):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch, launch_error=err)

    def _must_not_poll(*a, **kw):
        raise AssertionError("job_status must not be called — launch was refused")
    monkeypatch.setattr(supervise.tools, "job_status", _must_not_poll)
    make_capture_fakes(monkeypatch, calls)   # stop_pod etc should never fire either

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["process_exit_code"] == 2
    assert summary["refused"] is True
    assert calls == []
    assert str(err) in summary["reason"] or err.args[0] in summary["reason"]


# ============================ 5b. launch SSH-reply timeout -> adopt live job

def test_launch_ssh_timeout_adopts_running_job_and_stops(monkeypatch, tmp_path):
    """The documented-benign launch-SSH-reply timeout: the detach succeeded
    (job is RUNNING) but the reply was lost, so tools.launch_training raises
    ssh.SSHError. supervise must NOT refuse — it probes pod_status, finds the
    live job whose id is embedded in the timeout message, adopts it, and runs
    the full poll -> capture -> stop path."""
    JOB_ID = "20260713-142044_clean-rollout-dr0-model399_4362"
    timeout_exc = ssh.SSHError(
        "ssh to 1.2.3.4:2222 timed out after 60s running: touch "
        "/workspace/.keepalive && setsid bash /workspace/jobs/job_wrapper.sh "
        f"'/workspace/jobs/{JOB_ID}' 'pod5ln8' '1500' '0'")
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    # gate call: running pod, no live job yet; adoption-probe call (after the
    # failed-reply detach): our job is now live.
    seq = {"i": 0}

    def pod_status(rt):
        seq["i"] += 1
        return ({"status": "running"} if seq["i"] == 1
                else {"status": "running", "active_jobs": [JOB_ID]})
    monkeypatch.setattr(supervise.tools, "pod_status", pod_status)
    make_launch_training(monkeypatch, launch_error=timeout_exc)
    make_job_status(monkeypatch, [
        {"state": "succeeded", "exit_code": 0, "latest_reward_line": "Mean reward: 96.1"},
    ])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["job_id"] == JOB_ID
    assert summary["adopted_after_launch_timeout"] is True
    assert summary["state"] == "succeeded"
    assert summary["process_exit_code"] == 0
    # the capture/stop path ran against the ADOPTED id
    rsync_calls = [kw for name, kw in calls if name == "rsync_pull"]
    assert rsync_calls and rsync_calls[0]["remote_dir"] == f"/workspace/jobs/{JOB_ID}/"
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]
    # the durable recovery contract records the adoption
    on_disk = json.loads((tmp_path / f"supervise-{JOB_ID}.json").read_text())
    assert on_disk["adopted_after_launch_timeout"] is True


def test_launch_ssh_error_no_live_job_still_refuses(monkeypatch, tmp_path):
    """Safety case: an SSHError from an EARLIER launch call (nothing detached)
    leaves no live job. The adoption probe finds none -> supervise still
    refuses (exit 2, no poll/capture/stop) — the two-exits money-safety model
    is preserved; a genuine pre-detach failure is never falsely adopted."""
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status",
                        lambda rt: {"status": "running", "active_jobs": []})
    make_launch_training(monkeypatch, launch_error=ssh.SSHError(
        "ssh to 1.2.3.4:2222 timed out after 60s running: cat "
        "/workspace/isaac-auv-env/.../warpauv_env_cfg.py"))

    def _must_not_poll(*a, **kw):
        raise AssertionError("job_status must not be called — nothing was adopted")
    monkeypatch.setattr(supervise.tools, "job_status", _must_not_poll)
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["process_exit_code"] == 2
    assert summary["refused"] is True
    assert summary["job_id"] is None
    assert calls == []   # no poll, no capture, no stop


def test_launch_ssh_error_unparseable_id_adopts_sole_live_job(monkeypatch, tmp_path):
    """Fallback branch: the timeout message has no /workspace/jobs/<id> token,
    so the id can't be parsed — but the one-job guard means the single live job
    now must be the one we just launched, so supervise adopts it."""
    JOB_ID = "20260713-150000_eval_ab99"
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    seq = {"i": 0}

    def pod_status(rt):
        seq["i"] += 1
        return ({"status": "running"} if seq["i"] == 1
                else {"status": "running", "active_jobs": [JOB_ID]})
    monkeypatch.setattr(supervise.tools, "pod_status", pod_status)
    make_run_job(monkeypatch, job_id=JOB_ID, launch_error=ssh.SSHError(
        "ssh to 1.2.3.4:2222 timed out after 60s running: touch /workspace/.keepalive"))
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="job", name="eval", command="true", sync_subdir="none")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["job_id"] == JOB_ID
    assert summary["adopted_after_launch_timeout"] is True
    assert summary["process_exit_code"] == 0
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]


# ============================================== 6. capture/sync failure

def test_sync_logs_failure_still_stops_nonzero_exit(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls,
                      sync_error=tools.ToolError("bad subdir '../x'"))

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["process_exit_code"] != 0
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]
    assert any(e["step"] == "sync_logs" for e in summary["errors"])


# ==================================================== 7. --no-stop flag

def test_no_stop_flag_skips_stop_and_succeeded_job_is_exit_zero(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0", no_stop=True)
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert all(name != "stop_pod" for name, _ in calls)
    assert summary["stop"] == {"skipped": True, "reason": "--no-stop"}
    assert summary["process_exit_code"] == 0   # intentional skip = success


# ============================================== 8. generic job sync-subdir

def test_job_name_without_sync_subdir_is_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        supervise.main(["--job-name", "eval", "--command", "true"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_non_positive_interval_is_argparse_error(bad):
    """A 0/negative --interval would busy-spin (and real time.sleep raises on a
    negative) — reject it at argparse, exit 2."""
    with pytest.raises(SystemExit) as exc_info:
        supervise.main(["--training", "curee", "--dr", "DR_0", "--interval", bad])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("flag,bad", [("--max-wait", "0"), ("--max-wait", "-1"),
                                      ("--backstop", "-1")])
def test_non_positive_wait_and_negative_backstop_are_argparse_errors(flag, bad):
    with pytest.raises(SystemExit) as exc_info:
        supervise.main(["--training", "curee", "--dr", "DR_0", flag, bad])
    assert exc_info.value.code == 2


def test_job_name_sync_subdir_none_still_pulls_job_dir_skips_analysis(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_run_job(monkeypatch, job_id="20260710-100000_eval_ab12")
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="job", name="eval", command="true",
                             sync_subdir="none")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert any(name == "rsync_pull" for name, _ in calls)
    assert all(name != "sync_logs" for name, _ in calls)
    assert summary["pulled"]["analysis"] == {"skipped": True, "reason": "--sync-subdir none"}
    assert summary["process_exit_code"] == 0


# ===================================== 9. default summary path + max_wait

def test_default_summary_path_and_max_wait_derivation_job_mode(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_run_job(monkeypatch, job_id="20260710-120000_eval_cd34", ceiling=3600)
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="job", name="eval", command="true",
                             sync_subdir="none")   # no --max-runtime-sec given
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["max_wait_sec"] == rt.cfg["timeouts"]["job_sec"] + 300
    expected_path = tmp_path / f"supervise-{summary['job_id']}.json"
    assert expected_path.exists()
    on_disk = json.loads(expected_path.read_text())
    assert on_disk["job_id"] == summary["job_id"]
    assert on_disk["max_wait_sec"] == rt.cfg["timeouts"]["job_sec"] + 300


# ========================================== 10. dry-run derivation raises

@pytest.mark.parametrize("err", [
    tools.ToolError("unknown vehicle 'bogus' — use curee|bluerov2"),
    training.TrainingError("unknown dr_level 'bogus' — use DR_0..DR_4"),
])
def test_dry_run_refusal_no_poll_no_stop(monkeypatch, tmp_path, err):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch, dry_run_error=err)

    def _must_not_poll(*a, **kw):
        raise AssertionError("job_status must not be called — dry-run was refused")
    monkeypatch.setattr(supervise.tools, "job_status", _must_not_poll)
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["process_exit_code"] == 2
    assert summary["refused"] is True
    assert calls == []


# ================================================= 11. transient poll errors

def test_transient_poll_errors_then_deadline_still_forces_stop(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [
        jobs.JobError("could not inspect job (transient rc=255)"),
        jobs.JobError("could not inspect job (transient rc=255)"),
        jobs.JobError("could not inspect job (transient rc=255)"),
    ])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0",
                             max_wait=100, interval=45)
    clock = FakeClock(start=0.0)
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    assert summary["force_stopped"] is True
    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": True}]
    assert summary["process_exit_code"] != 0


# ============================== 12. capture-breadth: non-ToolError failures

def test_rsync_pull_ssh_error_still_reaches_stop(monkeypatch, tmp_path):
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path,
                 sshc=RaisingRsyncSSH(calls, ssh.SSHError("rsync rc=1: connection reset")))
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]
    assert summary["process_exit_code"] != 0
    assert any(e["step"] == "job_dir_pull" for e in summary["errors"])


def test_summary_write_oserror_still_stops_and_reports_nonzero(monkeypatch, tmp_path):
    calls = []
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")               # parent.mkdir will fail
    bad_summary_path = blocker / "summary.json"

    rt = _make_rt(calls, local_log_dir=tmp_path)
    monkeypatch.setattr(supervise.tools, "pod_status", _running)
    make_launch_training(monkeypatch)
    make_job_status(monkeypatch, [{"state": "succeeded", "exit_code": 0}])
    make_capture_fakes(monkeypatch, calls)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0",
                             summary_path=str(bad_summary_path))
    clock = FakeClock()
    summary = supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    stop_calls = [kw for name, kw in calls if name == "stop_pod"]
    assert stop_calls == [{"force": False}]   # stop already happened before the write attempt
    assert summary["process_exit_code"] != 0
    assert any(e["step"] == "summary_write" for e in summary["errors"])


# ================================================= misc: no bare RuntimeError

def test_genuine_bug_is_not_masked_as_a_refusal(monkeypatch, tmp_path):
    """A KeyError/AttributeError bug in the gate step must traceback loudly,
    not be silently swallowed into a 'refused' summary — REFUSE_ERRORS is an
    explicit tuple, never bare RuntimeError/Exception, in the gate/derive/
    launch phases."""
    calls = []
    rt = _make_rt(calls, local_log_dir=tmp_path)

    def _boom(rt):
        raise KeyError("not a REFUSE_ERRORS member")
    monkeypatch.setattr(supervise.tools, "pod_status", _boom)

    spec = supervise.JobSpec(mode="training", vehicle="curee", dr="DR_0")
    clock = FakeClock()
    with pytest.raises(KeyError):
        supervise.supervise(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
