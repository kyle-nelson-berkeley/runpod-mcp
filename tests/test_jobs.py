"""Unit 3: jobs.py — the async-job convention (launch, live-list, status)."""
import base64
import json
import subprocess

import pytest

from runpod_mcp import jobs


class FakeSSH:
    """Duck-typed SSHClient: records calls, returns scripted run() results."""

    def __init__(self, run_results=None):
        self.run_calls = []          # (host, port, command)
        self.push_texts = []         # (remote_path, text)
        self.push_files = []
        self._results = list(run_results or [])

    def run(self, host, port, command, timeout=60, check=False):
        self.run_calls.append(command)
        if self._results:
            res = self._results.pop(0)
            if check and res.returncode != 0:
                raise RuntimeError(f"rc={res.returncode}: {res.stderr}")
            return res
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def push_text(self, host, port, text, remote_path, executable=False):
        self.push_texts.append((remote_path, text))

    def push_file(self, host, port, local_path, remote_path):
        self.push_files.append((str(local_path), remote_path))


def ok(stdout=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def fail(stderr="nope", rc=1):
    return subprocess.CompletedProcess([], rc, stdout="", stderr=stderr)


# ------------------------------------------------------------------- job id

def test_new_job_id_is_sortable_slugged_and_collision_safe():
    jid = jobs.new_job_id("train CUREE DR_2 / seed 1!")
    stamp, slug, nonce = jid.split("_")
    assert len(stamp) == 15 and stamp[8] == "-"       # YYYYMMDD-HHMMSS
    assert slug == "train-curee-dr-2-seed-1"
    assert len(nonce) == 4                            # anti-collision suffix
    # same name, same second -> distinct job dirs
    assert jobs.new_job_id("x") != jobs.new_job_id("x")


# --------------------------------------------------------------- cmd script

def test_build_cmd_script_env_workdir_command():
    script = jobs.build_cmd_script("bash /workspace/pod_setup.sh",
                                   workdir="/workspace",
                                   env={"DEBIAN_FRONTEND": "noninteractive",
                                        "ACCEPT_EULA": "Y"})
    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "export DEBIAN_FRONTEND=noninteractive" in lines
    assert "export ACCEPT_EULA=Y" in lines
    assert "cd /workspace" in lines
    assert lines[-1] == "bash /workspace/pod_setup.sh"


# ------------------------------------------------------------------- launch

def test_launch_pushes_wrapper_cmd_meta_and_detaches():
    ssh = FakeSSH()
    jid = jobs.launch(ssh, "1.2.3.4", 15356,
                      name="setup", command="bash /workspace/pod_setup.sh",
                      workdir="/workspace", pod_id="on2ghkedz0vbjr",
                      max_runtime_sec=5400, auto_stop=False)
    job_dir = f"/workspace/jobs/{jid}"
    pushed = dict(ssh.push_texts)
    assert f"{job_dir}/cmd.sh" in pushed
    assert jobs.WRAPPER_REMOTE in pushed
    meta = json.loads(pushed[f"{job_dir}/meta.json"])
    assert meta["name"] == "setup"
    assert meta["max_runtime_sec"] == 5400
    assert meta["auto_stop"] is False
    # detached launch line: setsid + argv-injected pod id + full detach redirs
    launch_cmd = ssh.run_calls[-1]
    assert (f"setsid bash {jobs.WRAPPER_REMOTE} {job_dir} on2ghkedz0vbjr 5400 0"
            in launch_cmd)
    assert "</dev/null" in launch_cmd.replace("< /dev/null", "</dev/null")
    assert "&" in launch_cmd
    assert "touch /workspace/.keepalive" in launch_cmd


def test_launch_auto_stop_probes_runpodctl_first():
    ssh = FakeSSH(run_results=[ok("PROBE_OK"), ok(), ok()])
    jobs.launch(ssh, "h", 22, name="t", command="true", workdir="/workspace",
                pod_id="pid1", max_runtime_sec=60, auto_stop=True)
    probe = ssh.run_calls[0]
    assert "command -v runpodctl" in probe
    assert "runpodctl get pod pid1" in probe
    # wrapper armed with auto_stop=1
    assert " 1 </dev/null" in ssh.run_calls[-1]


def test_launch_auto_stop_probe_failure_is_loud():
    ssh = FakeSSH(run_results=[fail("runpodctl: not found", rc=127)])
    with pytest.raises(jobs.JobError, match="auto_stop"):
        jobs.launch(ssh, "h", 22, name="t", command="true",
                    workdir="/workspace", pod_id="pid1",
                    max_runtime_sec=60, auto_stop=True)
    # nothing was launched
    assert len(ssh.run_calls) == 1


# ---------------------------------------------------------------- live list

def test_list_live_parses_ids():
    ssh = FakeSSH(run_results=[
        ok("20260703-101010_setup\n20260703-111111_train\nLIVE_LIST_END\n")])
    assert jobs.list_live(ssh, "h", 22) == ["20260703-101010_setup",
                                            "20260703-111111_train"]


def test_list_live_empty():
    ssh = FakeSSH(run_results=[ok("LIVE_LIST_END\n")])
    assert jobs.list_live(ssh, "h", 22) == []


def test_list_live_ssh_failure_is_loud_never_empty():
    # a transient SSH failure must NOT read as "no jobs" (guard bypass)
    ssh = FakeSSH(run_results=[fail("connection reset rpa_LEAKME", rc=255)])
    with pytest.raises(jobs.JobError) as exc:
        jobs.list_live(ssh, "h", 22)
    assert "rpa_LEAKME" not in str(exc.value)          # scrubbed
    # truncated output (no sentinel) is also a failure
    ssh = FakeSSH(run_results=[ok("20260703-101010_setup\n")])
    with pytest.raises(jobs.JobError):
        jobs.list_live(ssh, "h", 22)


# ------------------------------------------------------------------- status

def _status_blob(pid="123", alive="no", exit_code="", meta=None, log=""):
    meta = json.dumps(meta or {"name": "t"})
    return (f"---PID---\n{pid}\n---ALIVE---\n{alive}\n---EXIT---\n{exit_code}\n"
            f"---META---\n{meta}\n---LOG---\n{log}")


def test_status_running():
    ssh = FakeSSH(run_results=[ok(_status_blob(alive="yes", log="iter 10"))])
    st = jobs.status(ssh, "h", 22, "20260703-101010_t", tail_lines=5)
    assert st["state"] == "running"
    assert "iter 10" in st["log_tail"]


def test_status_succeeded():
    ssh = FakeSSH(run_results=[ok(_status_blob(pid="", exit_code="0"))])
    assert jobs.status(ssh, "h", 22, "j")["state"] == "succeeded"


def test_status_failed_and_timeout_flagged():
    ssh = FakeSSH(run_results=[ok(_status_blob(pid="", exit_code="124"))])
    st = jobs.status(ssh, "h", 22, "j")
    assert st["state"] == "failed"
    assert st["exit_code"] == 124
    assert "wall-clock" in st["note"]                 # timeout ceiling hit


def test_status_orphaned():
    ssh = FakeSSH(run_results=[ok(_status_blob(pid="123", alive="no",
                                               exit_code=""))])
    assert jobs.status(ssh, "h", 22, "j")["state"] == "orphaned"


def test_status_not_found():
    ssh = FakeSSH(run_results=[ok("NO_SUCH_JOB\n")])
    assert jobs.status(ssh, "h", 22, "nope")["state"] == "not_found"


def test_status_extracts_latest_reward_line():
    log = "it 1\nMean reward: 12.3\nit 2\n  Mean reward:   95.7  \ndone"
    ssh = FakeSSH(run_results=[ok(_status_blob(pid="", exit_code="0", log=log))])
    st = jobs.status(ssh, "h", 22, "j")
    assert st["latest_reward_line"] == "Mean reward:   95.7"


def test_status_log_with_dashed_separators_survives_intact():
    # RSL-RL prints dashed rules; they must not truncate the parsed sections
    log = ("----------------------------------\n"
           "Learning iteration 399/400\n"
           "Mean reward: 96.1\n"
           "----------------------------------")
    ssh = FakeSSH(run_results=[ok(_status_blob(pid="", exit_code="0", log=log))])
    st = jobs.status(ssh, "h", 22, "j")
    assert st["state"] == "succeeded"
    assert "Learning iteration 399/400" in st["log_tail"]
    assert st["log_tail"].count("---") >= 2           # separators intact
    assert st["latest_reward_line"] == "Mean reward: 96.1"


def test_status_ssh_failure_raises_not_orphaned():
    # connection reset must be an ERROR, never misread as an orphaned job
    ssh = FakeSSH(run_results=[fail("reset by peer", rc=255)])
    with pytest.raises(jobs.JobError, match="could not inspect"):
        jobs.status(ssh, "h", 22, "j")


# ----------------------------------------------------------------- watchdog

def test_watchdog_install_command_probes_kills_and_relaunches():
    cmd = jobs.watchdog_install_command("podX", idle_minutes=60)
    assert "command -v runpodctl" in cmd
    assert "runpodctl get pod podX" in cmd            # loud probe at install
    assert "WATCHDOG_MISSING" in cmd                  # script presence guarded
    assert "kill" in cmd                              # old instance killed
    assert f"setsid bash {jobs.WATCHDOG_REMOTE} podX 60" in cmd
    assert ".idle_watchdog.pid" in cmd
    assert "touch /workspace/.keepalive" in cmd       # grace window on install


def test_install_watchdog_pushes_script_then_arms():
    ssh = FakeSSH(run_results=[ok("WATCHDOG_ARMED\n")])
    jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
    # the script content itself was pushed (survives container-disk wipe)
    paths = [p for p, _ in ssh.push_texts]
    assert jobs.WATCHDOG_REMOTE in paths
    text = dict(ssh.push_texts)[jobs.WATCHDOG_REMOTE]
    assert "idle_watchdog" in text and "runpodctl stop pod" in text


def test_install_watchdog_failure_is_loud():
    ssh = FakeSSH(run_results=[fail("WATCHDOG_PROBE_FAILED", rc=91)])
    with pytest.raises(jobs.JobError, match="NOT"):
        jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
