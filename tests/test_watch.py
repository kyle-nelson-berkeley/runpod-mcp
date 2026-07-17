"""Unit: watch.py — the RunPod observation-loop watcher CLI, against fake
tools.*/jobs.*/ssh.* seams. Mirrors tests/test_supervise.py idioms: watch()
calls the injected tools/jobs surface by MODULE ATTRIBUTE
(`watch.tools.pod_status`, `watch.tools._conn_info`, ...), so every fake here
is wired via `monkeypatch.setattr(watch.tools, "<name>", fake_fn)`. The one
call that bypasses tools.* is rt.ssh.run — the tail/exit_code/ls read-only
commands go straight through the Runtime's ssh client, exercised via a
scripted FakeSSH (imported from tests.test_tools).

ZERO network, zero real sleeps — a FakeClock's sleep() advances now()
directly, exactly like test_supervise.py's.
"""
import json
import time
from pathlib import Path

import pytest

from runpod_mcp import config, jobs, ssh, tools
from runpod_mcp import watch
from tests.test_tools import FakeSSH, ok

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "watch"


def _read_fixture(name: str) -> str:
    """newline='' disables Python's universal-newline translation — the raw
    fixtures on disk deliberately keep bare \\r bytes (real cursor-control
    noise from the source log); reading with the default text mode would
    silently rewrite them to \\n before the parser/tests ever saw them."""
    return (FIXTURES / name).read_text(encoding="utf-8", newline="")


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

JOB_ID = "20260716-144932_train-curee-dr-2-s1_ba84"


@pytest.fixture(autouse=True)
def _guard_no_stop_or_terminate(monkeypatch):
    """watch.py must NEVER stop or terminate a pod — spec-mandated invariant.
    Every test in this module implicitly asserts this."""
    def _boom_stop(rt, force=False):
        pytest.fail("tools.stop_pod must NEVER be called by watch")

    def _boom_terminate(rt, confirm):
        pytest.fail("tools.terminate_pod must NEVER be called by watch")
    monkeypatch.setattr(watch.tools, "stop_pod", _boom_stop, raising=False)
    monkeypatch.setattr(watch.tools, "terminate_pod", _boom_terminate, raising=False)
    yield


def _make_rt(ssh_results=None, local_log_dir=None, sshc=None):
    cfg = config.load_defaults()
    cfg["ssh_identity"] = "~/.ssh/id_ed25519"
    if local_log_dir is not None:
        cfg["local_log_dir"] = local_log_dir
    rt = tools.Runtime(cfg=cfg, client=None,
                       sshc=sshc if sshc is not None else FakeSSH(ssh_results),
                       sleep=lambda s: None, gpu_types=lambda **kw: [],
                       ssh_pubkey="ssh-ed25519 AAAA test@mac")
    return rt


class ScriptedSSH:
    """Routes rt.ssh.run() calls to per-command result queues, keyed on which
    of watch's three read-only commands (tail/cat exit_code/ls) is being
    issued — order-independent (unlike test_supervise.py's FakeSSH positional
    queue), since watch's per-iteration command ORDER is an implementation
    detail this suite deliberately does not pin down. Each queue repeats its
    last element forever once exhausted, mirroring make_job_status's idiom."""

    def __init__(self, *, tail=None, exit_code=None, ls=None):
        self.run_calls = []
        self._tail = list(tail or [])
        self._exit_code = list(exit_code or [])
        self._ls = list(ls or [])

    def _pick(self, command):
        if command.startswith("tail "):
            return self._tail
        if command.startswith("cat "):
            return self._exit_code
        if command.startswith("ls "):
            return self._ls
        raise AssertionError(f"unexpected ssh command from watch: {command!r}")

    def run(self, host, port, command, timeout=60, check=False):
        self.run_calls.append(command)
        q = self._pick(command)
        if not q:
            return ok("")
        return q.pop(0) if len(q) > 1 else q[0]


def _conn_info_fake(rt):
    return ("1.2.3.4", 2222)


# =============================================================== 1. parser

class TestMetricParser:
    def test_parses_single_block(self):
        text = _read_fixture("two_iter_block.log")
        p = watch.MetricParser()
        points = p.feed(text)
        pt = points[0]
        assert pt.iteration == 0
        assert pt.total_iterations == 100
        assert pt.mean_reward == pytest.approx(3.17)
        assert pt.value_loss == pytest.approx(2.4427)
        assert pt.surrogate_loss == pytest.approx(-0.0029)
        assert pt.entropy_loss == pytest.approx(8.4917)

    def test_parses_two_iter_block_fixture_whole(self):
        text = _read_fixture("two_iter_block.log")
        p = watch.MetricParser()
        points = p.feed(text)
        assert [pt.iteration for pt in points] == [0, 1]
        assert points[1].mean_reward == pytest.approx(5.94)
        assert points[1].total_iterations == 100

    def test_parses_real_200_iteration_fixture(self):
        text = _read_fixture("converging_200iters.log")
        p = watch.MetricParser()
        points = p.feed(text)
        assert len(points) == 200
        assert [pt.iteration for pt in points] == list(range(200))
        assert all(pt.total_iterations == 2500 for pt in points)
        assert points[0].mean_reward == pytest.approx(3.17)
        assert points[-1].mean_reward == pytest.approx(71.59)

    def test_startup_noise_head_yields_no_points_and_does_not_crash(self):
        text = _read_fixture("converging_200iters.log")
        head = text.split("################", 1)[0]   # ANSI/CR garbage + INFO lines
        assert "\x1b[3g" in head and "\r" in head   # sanity: fixture really has it
        p = watch.MetricParser()
        points = p.feed(head)
        assert points == []

    def test_split_block_across_two_chunks_matches_whole_file_parse(self):
        text = _read_fixture("two_iter_block.log")
        whole = watch.MetricParser().feed(text)

        # Cut mid-line, inside the "Mean value_function loss:" line of block 1.
        cut_marker = "Mean value_function loss: 1.6946"
        idx = text.index(cut_marker)
        split_at = idx + len("Mean value_function loss: 1.6")   # mid-number
        chunk1, chunk2 = text[:split_at], text[split_at:]
        assert chunk1 and chunk2   # sanity: both halves non-empty

        p = watch.MetricParser()
        pts1 = p.feed(chunk1)
        pts2 = p.feed(chunk2)
        incremental = pts1 + pts2

        assert len(whole) == len(incremental) == 2
        for a, b in zip(whole, incremental):
            assert a == b

    def test_feed_called_multiple_times_accumulates_state(self):
        """Simulates real incremental polling: many small feed() calls, none
        aligned to line boundaries, must still recover every block."""
        text = _read_fixture("converging_200iters.log")
        p = watch.MetricParser()
        points = []
        chunk_size = 137   # deliberately not aligned to any line length
        for i in range(0, len(text), chunk_size):
            points.extend(p.feed(text[i:i + chunk_size]))
        assert len(points) == 200
        assert [pt.iteration for pt in points] == list(range(200))


# ============================================================ 2. plateau

class TestPlateauDetector:
    def _points(self, fixture_name):
        text = _read_fixture(fixture_name)
        return watch.MetricParser().feed(text)

    def test_does_not_fire_on_real_converging_fixture(self):
        points = self._points("converging_200iters.log")
        det = watch.PlateauDetector(watch.DEFAULT_PLATEAU_WINDOW,
                                    watch.DEFAULT_PLATEAU_MIN_DELTA)
        fired = [det.update(pt) for pt in points]
        assert not any(fired), (
            "plateau heuristic false-fired on the real converging-reward "
            "fixture — defaults are too tight")

    def test_fires_on_synthetic_flattened_tail_fixture(self):
        points = self._points("flattened_tail.log")
        det = watch.PlateauDetector(watch.DEFAULT_PLATEAU_WINDOW,
                                    watch.DEFAULT_PLATEAU_MIN_DELTA)
        fired = [det.update(pt) for pt in points]
        assert any(fired), "plateau heuristic never fired on the flattened tail fixture"

    def test_insufficient_history_never_fires(self):
        det = watch.PlateauDetector(window_iters=50, min_delta=3.0)
        pt = watch.MetricPoint(iteration=0, total_iterations=100, mean_reward=1.0,
                               value_loss=0.1, surrogate_loss=0.1, entropy_loss=0.1)
        assert det.update(pt) is False


# ======================================================= 3. failure detection

class TestFailureDetection:
    def test_detects_traceback(self):
        text = _read_fixture("traceback.log")
        assert watch.detect_failure(text) is not None

    def test_detects_cuda_oom(self):
        text = _read_fixture("cuda_oom.log")
        result = watch.detect_failure(text)
        assert result is not None
        assert "oom" in result.lower() or "memory" in result.lower()

    def test_normal_training_text_is_not_a_failure(self):
        text = _read_fixture("converging_200iters.log")
        assert watch.detect_failure(text) is None

    def test_empty_text_is_not_a_failure(self):
        assert watch.detect_failure("") is None


# ========================================================= 4. job discovery

def make_pod_status_sequence(monkeypatch, results):
    """results: list of dicts (pod_status() results) or Exception instances
    (raised on that poll). Exhausted lists repeat the last element forever —
    mirrors test_supervise.py's make_job_status idiom."""
    calls = {"i": 0}

    def fn(rt):
        i = min(calls["i"], len(results) - 1)
        calls["i"] += 1
        item = results[i]
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(watch.tools, "pod_status", fn)
    return calls


class TestJobDiscovery:
    def test_job_id_override_skips_discovery_entirely(self, monkeypatch):
        def _must_not_be_called(rt):
            raise AssertionError("pod_status must not be called — --job-id was given")
        monkeypatch.setattr(watch.tools, "pod_status", _must_not_be_called)

        rt = _make_rt()
        spec = watch.WatchSpec(job_id=JOB_ID)
        clock = FakeClock()
        found = watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert found == JOB_ID

    def test_immediate_single_active_job_is_found(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec()
        clock = FakeClock()
        found = watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert found == JOB_ID

    def test_armed_before_launch_then_degraded_poll_then_found_is_a_retry_not_exit5(self, monkeypatch):
        """The BINDING race case: (1) armed before the job launches (no active
        jobs yet), (2) a DEGRADED poll matching the chosen seam's signal
        (active_jobs_error present), (3) the job finally shows up. None of
        this may raise — it must retry through to a clean find."""
        calls = make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": []},              # armed before launch
            {"status": "running", "active_jobs_error": "could not inspect (rc=255)"},  # degraded
            {"status": "running", "active_jobs": [JOB_ID]},        # found
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=100)
        clock = FakeClock(start=0.0)
        found = watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert found == JOB_ID
        assert calls["i"] == 3
        assert clock.now() < 100   # never hit the grace deadline

    def test_running_ssh_pending_poll_is_also_a_retry_not_exit5(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            {"status": "running_ssh_pending"},
            {"status": "running", "active_jobs": [JOB_ID]},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=100)
        clock = FakeClock(start=0.0)
        found = watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert found == JOB_ID

    def test_raising_poll_is_also_a_retry_not_exit5(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            tools.ToolError("pod is not SSH-ready"),
            {"status": "running", "active_jobs": [JOB_ID]},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=100)
        clock = FakeClock(start=0.0)
        found = watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert found == JOB_ID

    def test_grace_expires_with_no_job_ever_seen_raises_watch_error(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": []},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=30)
        clock = FakeClock(start=0.0)
        with pytest.raises(watch.WatchError):
            watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert clock.now() >= 30

    def test_grace_expires_all_degraded_raises_watch_error(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs_error": "could not inspect (rc=255)"},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=30)
        clock = FakeClock(start=0.0)
        with pytest.raises(watch.WatchError):
            watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

    def test_multiple_active_jobs_raises_immediately_not_after_grace(self, monkeypatch):
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID, "other-job"]},
        ])
        rt = _make_rt()
        spec = watch.WatchSpec(startup_grace=1000)
        clock = FakeClock(start=0.0)
        with pytest.raises(watch.WatchError):
            watch._discover_job_id(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert clock.now() < 1000   # raised immediately, did not wait out the grace


# ========================================================== 5. tail/offset

class TestTailAndOffset:
    def test_tail_uses_byte_offset_and_advances(self, monkeypatch):
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        job_dir = f"{jobs.JOBS_ROOT}/{JOB_ID}"
        sshc = FakeSSH([ok("first chunk\n"), ok("second chunk\n")])
        rt = _make_rt()
        rt._ssh = sshc

        text1, offset1 = watch._tail(rt, job_dir, 0)
        assert text1 == "first chunk\n"
        assert offset1 == len("first chunk\n")
        assert f"tail -c +1 " in sshc.run_calls[0]

        text2, offset2 = watch._tail(rt, job_dir, offset1)
        assert text2 == "second chunk\n"
        assert offset2 == offset1 + len("second chunk\n")
        assert f"tail -c +{offset1 + 1} " in sshc.run_calls[1]

    def test_tail_never_issues_a_stop_or_kill_like_command(self, monkeypatch):
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        job_dir = f"{jobs.JOBS_ROOT}/{JOB_ID}"
        sshc = FakeSSH([ok("x\n")])
        rt = _make_rt()
        rt._ssh = sshc
        watch._tail(rt, job_dir, 0)
        for cmd in sshc.run_calls:
            assert "kill" not in cmd and "stop" not in cmd and "runpodctl" not in cmd


# ================================================== 6. exit decision tree

def _one_block(iteration, total, reward):
    return (
        "################################################################################\n"
        f"                     \x1b[1m Learning iteration {iteration}/{total} \x1b[0m\n"
        "\n"
        "                       Computation: 300000 steps/s\n"
        "             Mean action noise std: 0.11\n"
        "          Mean value_function loss: 0.5\n"
        "               Mean surrogate loss: 0.01\n"
        "                 Mean entropy loss: -4.0\n"
        f"                       Mean reward: {reward}\n"
        "               Mean episode length: 180.0\n"
        "--------------------------------------------------------------------------------\n"
        "                   Total timesteps: 1000\n"
        "                    Iteration time: 0.16s\n"
        "                      Time elapsed: 00:00:01\n"
        "                               ETA: 00:00:01\n"
        "\n"
    )


class TestExitDecisionTree:
    def test_happy_path_no_plateau_exit_zero(self, monkeypatch, tmp_path):
        chunk1 = _one_block(0, 100, 10.0) + _one_block(1, 100, 20.0)
        sshc = ScriptedSSH(
            tail=[ok(chunk1), ok("")],
            exit_code=[ok(""), ok("0\n")],
            ls=[ok("total 0\n-rw-r--r-- out.log\n")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
            {"status": "running", "active_jobs": []},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0
        assert result["job_id"] == JOB_ID
        assert result["plateau_fired"] is False

    def test_plateau_fires_mid_run_exits_three_immediately_job_still_active(
            self, monkeypatch, tmp_path):
        """The paging idiom (consensus plan): plateau is a WATCHER-terminal
        condition — the page (exit 3) must arrive at DETECTION time, while
        the job is still running, so a human can decide whether to
        intervene. NOT a completion-time recolor. The watcher takes no
        action on the job — it just exits."""
        flat_text = _read_fixture("flattened_tail.log")
        sshc = ScriptedSSH(
            tail=[ok(flat_text)],
            exit_code=[ok("")],                 # job has NOT finished
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # job STILL ACTIVE
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["plateau_fired"] is True
        assert result["process_exit_code"] == 3
        # exited in the FIRST poll round — never slept an interval, job never
        # observed terminal (still in active_jobs, no exit_code artifact).
        assert clock.sleeps == []
        assert result["gone_from_active"] is False
        assert result["exit_code_artifact"] is None
        # the final status JSON records that the job was still running and
        # points back at supervise as the authoritative record.
        on_disk = json.loads(Path(result["status_path"]).read_text())
        assert on_disk["plateau_fired"] is True
        assert on_disk["process_exit_code"] == 3
        assert "still running" in on_disk["plateau_note"]
        assert "supervise" in on_disk["plateau_note"]

    def test_historical_flat_window_with_recovered_trailing_window_does_not_page(
            self, monkeypatch, tmp_path):
        """[codex round-4 BLOCKING fix] Late attach: the first _tail reads the
        ENTIRE existing out.log as one backlog. Its MIDDLE window (iters
        40-99, constant reward) is flat — per-point latching would fire there
        during catch-up — but the TRAILING window has recovered (rising
        rewards through iter 119). Plateau must reflect the CURRENT trailing
        window as of the newest point: NO exit 3; the watcher keeps polling
        and resolves the job's real terminal state in round 2."""
        backlog = "".join(
            [_one_block(i, 2500, 2.0 * i) for i in range(40)]            # rising
            + [_one_block(i, 2500, 80.0) for i in range(40, 100)]        # flat (historical)
            + [_one_block(i, 2500, 80.0 + 2.0 * (i - 99)) for i in range(100, 120)]  # recovered
        )
        sshc = ScriptedSSH(
            tail=[ok(backlog), ok("")],
            exit_code=[ok(""), ok("0\n")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # round 1: still running
            {"status": "running", "active_jobs": []},         # round 2: job over
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0    # NOT 3 — the plateau is stale
        assert result["plateau_fired"] is False
        assert len(clock.sleeps) == 1              # survived round 1, exited round 2

    def test_backlog_whose_trailing_window_is_flat_pages_exit_three_first_round(
            self, monkeypatch, tmp_path):
        """Attach-late counterpart: rising history, but the backlog's TRAILING
        window IS flat as of the newest point — the plateau is CURRENT, so the
        first-round page (exit 3) is correct even though the flatness began in
        history."""
        backlog = "".join(
            [_one_block(i, 2500, 2.0 * i) for i in range(50)]            # rising
            + [_one_block(i, 2500, 100.0) for i in range(50, 110)]       # flat through newest
        )
        sshc = ScriptedSSH(
            tail=[ok(backlog)],
            exit_code=[ok("")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 3
        assert result["plateau_fired"] is True
        assert clock.sleeps == []                  # paged in the first round

    def test_plateau_and_failure_in_same_round_prefers_exit_four(self, monkeypatch, tmp_path):
        """Precedence: failure evidence beats plateau when both are visible
        in the same poll round."""
        flat_plus_traceback = (_read_fixture("flattened_tail.log")
                               + _read_fixture("traceback.log"))
        sshc = ScriptedSSH(
            tail=[ok(flat_plus_traceback)],
            exit_code=[ok("")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["plateau_fired"] is True   # it DID fire...
        assert result["process_exit_code"] == 4  # ...but failure wins the page

    def test_plateau_and_terminal_evidence_in_same_round_prefers_terminal_code(
            self, monkeypatch, tmp_path):
        """Precedence: if the same round that first fires the plateau also
        shows the job already over (gone + exit_code 0), prefer the terminal
        code — there is nothing left to intervene on."""
        flat_text = _read_fixture("flattened_tail.log")
        sshc = ScriptedSSH(
            tail=[ok(flat_text)],
            exit_code=[ok("0\n")],              # already finished OK
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": []},   # already gone
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["plateau_fired"] is True
        assert result["process_exit_code"] == 0   # terminal code, NOT 3

    def test_exit_code_nonzero_is_failure_exit_four(self, monkeypatch, tmp_path):
        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0)), ok("")],
            exit_code=[ok(""), ok("1\n")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
            {"status": "running", "active_jobs": []},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 4
        assert result["exit_code_artifact"] == 1

    def test_traceback_in_tail_is_failure_exit_four(self, monkeypatch, tmp_path):
        traceback_text = _read_fixture("traceback.log")
        sshc = ScriptedSSH(
            tail=[ok(traceback_text)],
            exit_code=[ok("")],   # never terminates via artifact
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # never gone
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 4
        assert result["failure_detected"] == "traceback"

    def test_stall_with_no_terminal_evidence_exit_five(self, monkeypatch, tmp_path):
        sshc = ScriptedSSH(
            tail=[ok("")],       # never any new iteration
            exit_code=[ok("")],  # never present
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # never gone
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=100, stall_sec=250)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 5
        assert clock.now() >= 250

    def test_pod_disappears_after_prior_successful_exit_code_read_still_resolves_exit_zero(
            self, monkeypatch, tmp_path):
        """The documented happy-path race: supervise's own poll loop stops
        the pod ~concurrently with watch's next tail. Round 1: watch reads
        exit_code=0 successfully but the gone-check hasn't caught up yet
        (job still listed active). Round 2: the pod is fully gone (ssh calls
        fail / pod_status reports status="stopped") — watch must use the
        CACHED exit_code from round 1 combined with the now-confirmed "gone"
        (via status="stopped") to resolve exit 0, never fall through to a
        stall."""
        def _tail_round2(host, port, command, timeout=60, check=False):
            raise ssh.SSHError("connection refused — pod is stopped")

        class FlakySSH(ScriptedSSH):
            """First 3 calls (round 1's tail/ls/cat) succeed normally; every
            call from the 4th onward (round 2+) fails, simulating the pod
            going away. NOTE: append exactly once per call — ScriptedSSH.run
            (via super()) already appends, so the failing branch must append
            itself instead of double-counting through the super() call."""

            def __init__(self):
                super().__init__(tail=[ok(_one_block(0, 100, 10.0))],
                                 exit_code=[ok("0\n")], ls=[ok("")])

            def run(self, host, port, command, timeout=60, check=False):
                if len(self.run_calls) >= 3:
                    self.run_calls.append(command)
                    raise ssh.SSHError("connection refused — pod is stopped")
                return super().run(host, port, command, timeout, check)

        sshc = FlakySSH()
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # round 1: not yet gone
            {"status": "stopped"},                            # round 2: pod fully gone
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0
        assert result["exit_code_artifact"] == 0

    def test_pod_stopped_before_exit_code_read_resolves_exit_zero_not_stall(
            self, monkeypatch, tmp_path):
        """[codex round-5 BLOCKING fix] Happy-path completion race: supervise
        (45s polls) observes completion and stops the pod BETWEEN watcher
        polls (60s), before the watcher ever cached an exit_code. From then
        on _read_exit_code can never succeed (pod stopped, SSH gone) while
        the pod itself is authoritatively observed stopped. This must
        resolve exit 0 PROMPTLY (with a status-JSON note deferring to
        supervise), never idle out --stall-sec into a false exit 5."""
        class DeadAfterRound1SSH(ScriptedSSH):
            def __init__(self):
                super().__init__(tail=[ok(_one_block(0, 100, 10.0))],
                                 exit_code=[ok("")],   # never readable
                                 ls=[ok("")])

            def run(self, host, port, command, timeout=60, check=False):
                if len(self.run_calls) >= 3:   # round 2+: pod is stopped
                    self.run_calls.append(command)
                    raise ssh.SSHError("connection refused — pod is stopped")
                return super().run(host, port, command, timeout, check)

        sshc = DeadAfterRound1SSH()
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # round 1
            {"status": "stopped"},                            # round 2: authoritative stop
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=60, stall_sec=600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0
        assert result["exit_code_artifact"] is None   # never got to read it
        # prompt: one interval slept, nowhere near --stall-sec
        assert len(clock.sleeps) == 1
        assert clock.now() < spec.stall_sec
        on_disk = json.loads(Path(result["status_path"]).read_text())
        assert "supervise" in on_disk["pod_stopped_note"]
        assert "exit_code" in on_disk["pod_stopped_note"]

    def test_pod_stopped_with_failure_evidence_seen_same_round_exits_four(
            self, monkeypatch, tmp_path):
        """Same stop race, but a traceback surfaces in the round-2 tail —
        failure evidence beats the benign pod-stopped resolution: exit 4."""
        traceback_text = _read_fixture("traceback.log")
        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0)), ok(traceback_text)],
            exit_code=[ok("")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},   # round 1
            {"status": "stopped"},                            # round 2
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=60, stall_sec=600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 4
        assert result["failure_detected"] == "traceback"

    def test_degraded_polls_do_not_take_the_stopped_branch_still_stall_exit_five(
            self, monkeypatch, tmp_path):
        """Guard: a raising pod_status / running_ssh_pending is UNREACHABLE
        (degraded), not an authoritative stop — it must stay on the
        retry->stall path and exit 5 when no terminal evidence ever appears,
        NOT be misread as 'pod stopped -> exit 0'."""
        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0)), ok("")],
            exit_code=[ok("")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},       # round 1
            tools.ToolError("pod is not SSH-ready"),              # degraded
            {"status": "running_ssh_pending"},                    # degraded (repeats)
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=100, stall_sec=250)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 5
        assert clock.now() >= spec.stall_sec

    def test_conn_info_toolerror_after_job_gone_resolves_via_cached_evidence_not_crash(
            self, monkeypatch, tmp_path):
        """[codex-review BLOCKING fix] tools._conn_info itself can raise
        ToolError (pod just stopped + connection cache expired) — exactly the
        happy-path pod-stop race. Round 1: everything works, exit_code=0 read
        and cached, but the job is still listed active. Round 2: _conn_info
        RAISES on every remote-read helper, and pod_status shows the pod
        stopped (gone). The watcher must NOT crash — the degraded reads are
        "signal unavailable this poll", and the decision tree resolves via
        the CACHED exit_code 0 + now-confirmed gone → exit 0."""
        rounds = {"n": 0}

        def conn_info(rt):
            if rounds["n"] >= 1:
                raise tools.ToolError("pod is not SSH-ready (status=EXITED)")
            return ("1.2.3.4", 2222)

        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0))],
            exit_code=[ok("0\n")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", conn_info)

        seq = [
            {"status": "running", "active_jobs": [JOB_ID]},   # round 1: not yet gone
            {"status": "stopped"},                            # round 2: pod stopped
        ]

        def pod_status(rt):
            i = min(rounds["n"], len(seq) - 1)
            rounds["n"] += 1        # pod_status is the LAST poll of a round —
                                    # advancing here flips _conn_info for round 2
            return seq[i]
        monkeypatch.setattr(watch.tools, "pod_status", pod_status)

        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0
        assert result["exit_code_artifact"] == 0

    def test_watch_writes_status_json_and_prints_a_status_line(self, monkeypatch, tmp_path, capsys):
        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0)), ok("")],
            exit_code=[ok(""), ok("0\n")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
            {"status": "running", "active_jobs": []},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)

        status_path = Path(result["status_path"])
        assert status_path.exists()
        assert status_path.name == f"watch-{JOB_ID}.json"
        out = capsys.readouterr().out
        assert JOB_ID[:8] in out or "iter" in out.lower()

    def test_watch_never_calls_stop_or_terminate_across_full_run(self, monkeypatch, tmp_path):
        """Redundant with the autouse guard fixture, but exercises it across
        a REAL multi-iteration watch() run end-to-end (not just a unit)."""
        sshc = ScriptedSSH(
            tail=[ok(_one_block(0, 100, 10.0)), ok(_one_block(1, 100, 12.0)), ok("")],
            exit_code=[ok(""), ok(""), ok("0\n")],
            ls=[ok("")],
        )
        rt = _make_rt(sshc=sshc, local_log_dir=tmp_path)
        monkeypatch.setattr(watch.tools, "_conn_info", _conn_info_fake)
        make_pod_status_sequence(monkeypatch, [
            {"status": "running", "active_jobs": [JOB_ID]},
            {"status": "running", "active_jobs": [JOB_ID]},
            {"status": "running", "active_jobs": []},
        ])
        spec = watch.WatchSpec(job_id=JOB_ID, interval=5, stall_sec=3600)
        clock = FakeClock(start=0.0)
        result = watch.watch(rt, spec, sleep=clock.sleep, now=clock.now, log=NOLOG)
        assert result["process_exit_code"] == 0
        for cmd in sshc.run_calls:
            assert "kill" not in cmd and "runpodctl" not in cmd


# ============================================================== 7. CLI

class TestCLI:
    def test_help_mentions_heuristic_unverified(self, capsys):
        parser = watch._build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "heuristic" in out.lower()
        assert "unverified" in out.lower()

    def test_prog_name_is_watch(self):
        assert watch._build_parser().prog == "watch"

    @pytest.mark.parametrize("flag,bad", [
        ("--interval", "0"), ("--interval", "-1"),
        ("--startup-grace", "0"), ("--startup-grace", "-1"),
        ("--stall-sec", "0"), ("--stall-sec", "-1"),
        ("--plateau-window", "0"), ("--plateau-window", "-1"),
    ])
    def test_non_positive_numeric_flags_are_argparse_errors(self, flag, bad):
        with pytest.raises(SystemExit) as exc_info:
            watch._spec_from_args(
                watch._build_parser().parse_args([flag, bad]),
                watch._build_parser())
        assert exc_info.value.code == 2

    def test_negative_plateau_min_delta_is_argparse_error(self):
        with pytest.raises(SystemExit) as exc_info:
            watch._spec_from_args(
                watch._build_parser().parse_args(["--plateau-min-delta", "-1"]),
                watch._build_parser())
        assert exc_info.value.code == 2

    def test_defaults_produce_a_valid_spec(self):
        parser = watch._build_parser()
        spec = watch._spec_from_args(parser.parse_args([]), parser)
        assert spec.job_id is None
        assert spec.interval == watch.DEFAULT_INTERVAL
        assert spec.startup_grace == watch.DEFAULT_STARTUP_GRACE
        assert spec.stall_sec == watch.DEFAULT_STALL_SEC
        assert spec.plateau_window == watch.DEFAULT_PLATEAU_WINDOW
        assert spec.plateau_min_delta == watch.DEFAULT_PLATEAU_MIN_DELTA

    def test_job_id_flag_is_captured(self):
        parser = watch._build_parser()
        spec = watch._spec_from_args(parser.parse_args(["--job-id", JOB_ID]), parser)
        assert spec.job_id == JOB_ID

    def test_default_status_path_uses_repo_root_and_local_log_dir(self, tmp_path):
        rt = _make_rt(local_log_dir=tmp_path)
        path = watch._default_status_path(rt, JOB_ID)
        assert path == tmp_path / f"watch-{JOB_ID}.json"
