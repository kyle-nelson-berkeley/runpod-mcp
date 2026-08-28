"""Unit 3: jobs.py — the async-job convention (launch, live-list, status)."""
import base64
import json
import re
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
                      workdir="/workspace", pod_id="fakefakefake00",
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
    assert (f"setsid bash {jobs.WRAPPER_REMOTE} {job_dir} fakefakefake00 5400 0"
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


# ------------------------------------------- runpodctl probe: 3-way split
# OPS-DEFECTS-PLAN.md §2.3-A/B: the old probe chained
# `command -v runpodctl && runpodctl get pod <id>`, so binary-absent and
# auth-refused collapsed onto ONE rc and nobody could tell which half failed.
# The split names the cause AND runs the retry in the environment the gate
# certifies (`. /etc/rp_environment`, exactly like job_wrapper.sh:23-24 and
# idle_watchdog.sh:32).  HARD RULE: names/booleans only, never file CONTENTS.

def res(rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr=stderr)


def _probe_cmd(pod_id="pid1"):
    """The command _probe_auto_stop actually sends (captured via FakeSSH)."""
    ssh = FakeSSH(run_results=[res(0, "PROBE_OK"), ok(), ok()])
    jobs.launch(ssh, "h", 22, name="t", command="true", workdir="/workspace",
                pod_id=pod_id, max_runtime_sec=60, auto_stop=True)
    return ssh.run_calls[0]


def _both_commands():
    return {"probe": _probe_cmd("pid1"),
            "watchdog": jobs.watchdog_install_command("podX", idle_minutes=60)}


def test_both_probe_commands_carry_all_three_sentinels():
    for label, cmd in _both_commands().items():
        assert "NO_RUNPODCTL" in cmd, label            # binary absent
        assert "NO_RUNPODCTL_AUTH_BARE" in cmd, label  # refused unsourced
        assert "NO_RUNPODCTL_AUTH_SOURCED" in cmd, label   # refused sourced
        assert "exit 90" in cmd, label                 # binary-absent rc
        assert "exit 91" in cmd, label                 # terminal auth rc
        # candidate (d): PATH + binary metadata, names/paths only
        assert 'echo "PATH=$PATH"' in cmd, label
        assert "ls -l /usr/local/bin/runpodctl /usr/bin/runpodctl" in cmd, label


def test_legacy_probe_substrings_survive():
    cmds = _both_commands()
    assert "command -v runpodctl" in cmds["probe"]
    assert "runpodctl get pod pid1" in cmds["probe"]
    assert "PROBE_OK" in cmds["probe"]
    assert "command -v runpodctl" in cmds["watchdog"]
    assert "runpodctl get pod podX" in cmds["watchdog"]
    assert "WATCHDOG_PROBE_FAILED" in cmds["watchdog"]


def test_authoritative_checks_run_after_the_unconditional_source():
    # plan §2.3-B: "a gate must run in the environment it certifies".  The bare
    # pair is DIAGNOSTIC ONLY (it makes AUTH_BARE meaningful for H1 vs H2); the
    # source is UNCONDITIONAL, and the pair that decides the verdict runs after
    # it.  Both real consumers source before touching runpodctl, so a bare pass
    # proves nothing about the stop path.
    for label, cmd in _both_commands().items():
        haves = [m.start() for m in re.finditer(r"command -v runpodctl", cmd)]
        gets = [m.start() for m in re.finditer(r"runpodctl get pod", cmd)]
        srcs = [m.start() for m in re.finditer(r"\. /etc/rp_environment", cmd)]
        assert len(srcs) == 1, label            # sourced ONCE, unconditionally
        assert len(haves) == 2 and len(gets) == 2, label
        assert haves[0] < gets[0] < srcs[0] < haves[1] < gets[1], label
        # every VERDICT sentinel is emitted after the source; only the
        # diagnostic AUTH_BARE may precede it
        assert cmd.index("echo NO_RUNPODCTL_AUTH_BARE") < srcs[0], label
        assert srcs[0] < cmd.index("echo NO_RUNPODCTL;"), label
        assert srcs[0] < cmd.index("echo NO_RUNPODCTL_AUTH_SOURCED"), label
        assert haves[1] < cmd.index("exit 90") < gets[1] < cmd.index("exit 91")


def test_watchdog_command_starts_with_the_shared_probe_prefix():
    # one probe, two callers: the live-shell matrix below exercises the prefix
    # once and this pins that the watchdog ships the very same text
    prefix = jobs._runpodctl_probe_command(
        "podX", fail_extra="echo WATCHDOG_PROBE_FAILED; ")
    assert jobs.watchdog_install_command("podX", idle_minutes=60).startswith(
        prefix)


def test_probe_commands_never_print_rp_environment_contents():
    # a `cat` would put an API key into MCP output — config.scrub() runs on
    # error paths only, so the leak would already have escaped.
    for label, cmd in _both_commands().items():
        for verb in ("cat", "echo", "grep", "head", "tail", "sed", "awk",
                     "od", "xxd", "printf", "less", "more"):
            assert f"{verb} /etc/rp_environment" not in cmd, (label, verb)
        # every mention is either a presence test or a source, nothing else
        for line in cmd.split(";"):
            if "/etc/rp_environment" not in line:
                continue
            stripped = line.strip()
            assert (stripped.startswith("[ -f /etc/rp_environment ]")
                    or stripped.startswith(". /etc/rp_environment")), (label,
                                                                       stripped)


# --- the branch matrix, executed under a REAL bash ------------------------
# The sentinel/rc contract is a SHELL contract, so assert it against a shell.
# $0, Mac-side, no pod: a fake runpodctl + a fake rp_environment file.

def _run_probe_live(tmp_path, *, bare_path, env_adds_path=False,
                    bare_key=False, env_adds_key=False,
                    env_clears_key=False, env_hides_path=False):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)              # the leak test reuses tmp_path
    binary = bindir / "runpodctl"
    binary.write_text('#!/bin/sh\n[ -n "$RP_KEY" ] && exit 0\nexit 1\n')
    binary.chmod(0o755)

    env_file = tmp_path / "rp_environment"
    lines = []
    if env_adds_path:
        lines.append(f'PATH="$PATH:{bindir}"; export PATH')
    if env_hides_path:                       # the file OVERRIDES PATH
        lines.append('PATH="/usr/bin:/bin"; export PATH')
    if env_adds_key:
        lines.append("RP_KEY=NEVER_PRINT_ME; export RP_KEY")
    if env_clears_key:                       # the file OVERRIDES the credential
        lines.append("RP_KEY=; export RP_KEY")
    env_file.write_text("\n".join(lines) + "\n")

    cmd = (jobs._runpodctl_probe_command("pid1") + "echo PROBE_OK").replace(
        jobs.RP_ENV, str(env_file))
    env = {"PATH": (f"{bindir}:/usr/bin:/bin" if bare_path else "/usr/bin:/bin")}
    if bare_key:
        env["RP_KEY"] = "NEVER_PRINT_ME"
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          env=env)


def test_live_clean_success_emits_only_probe_ok(tmp_path):
    proc = _run_probe_live(tmp_path, bare_path=True, bare_key=True)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "PROBE_OK"          # no sentinel leak
    assert "NO_RUNPODCTL" not in proc.stdout


def test_live_binary_found_only_after_sourcing_still_passes(tmp_path):
    # THE REGRESSION: a bare-PATH miss must NOT exit 90 when sourcing
    # /etc/rp_environment puts runpodctl on PATH — both real consumers
    # (job_wrapper.sh, idle_watchdog.sh) source it before touching runpodctl.
    proc = _run_probe_live(tmp_path, bare_path=False, env_adds_path=True,
                           env_adds_key=True)
    assert proc.returncode == 0, proc.stdout
    assert "PROBE_OK" in proc.stdout
    assert "NO_RUNPODCTL" not in proc.stdout
    assert "PATH=" in proc.stdout                     # §2.1d: the bare miss


def test_live_binary_absent_even_after_sourcing_is_rc90(tmp_path):
    proc = _run_probe_live(tmp_path, bare_path=False, env_adds_path=False)
    assert proc.returncode == 90
    assert "NO_RUNPODCTL" in proc.stdout
    assert "PROBE_OK" not in proc.stdout


def test_live_binary_after_sourcing_but_no_credential_is_rc91(tmp_path):
    proc = _run_probe_live(tmp_path, bare_path=False, env_adds_path=True,
                           env_adds_key=False)
    assert proc.returncode == 91
    assert "NO_RUNPODCTL_AUTH_SOURCED" in proc.stdout
    assert "PROBE_OK" not in proc.stdout


def test_live_sourcing_rescues_the_credential_h2(tmp_path):
    proc = _run_probe_live(tmp_path, bare_path=True, env_adds_key=True)
    assert proc.returncode == 0, proc.stdout
    assert "NO_RUNPODCTL_AUTH_BARE" in proc.stdout     # H2 evidence
    assert "PROBE_OK" in proc.stdout


def test_live_no_credential_anywhere_is_rc91_h1(tmp_path):
    proc = _run_probe_live(tmp_path, bare_path=True)
    assert proc.returncode == 91
    assert "NO_RUNPODCTL_AUTH_BARE" in proc.stdout
    assert "NO_RUNPODCTL_AUTH_SOURCED" in proc.stdout


def test_live_bare_pass_then_sourced_credential_failure_is_rc91(tmp_path):
    # THE ARMED-BUT-DUD CASE: bare auth passes, but the file the real consumers
    # source overrides the credential — the probe must NOT report PROBE_OK.
    proc = _run_probe_live(tmp_path, bare_path=True, bare_key=True,
                           env_clears_key=True)
    assert proc.returncode == 91, proc.stdout
    assert "NO_RUNPODCTL_AUTH_SOURCED" in proc.stdout
    assert "PROBE_OK" not in proc.stdout
    assert "NO_RUNPODCTL_AUTH_BARE" not in proc.stdout   # bare really did pass


def test_live_bare_pass_then_sourced_path_override_is_rc90(tmp_path):
    # same shape, PATH flavour: the sourced env hides the binary the bare
    # shell could see, so the stop path would fail where the probe passed
    proc = _run_probe_live(tmp_path, bare_path=True, bare_key=True,
                           env_hides_path=True)
    assert proc.returncode == 90, proc.stdout
    assert "NO_RUNPODCTL" in proc.stdout
    assert "PROBE_OK" not in proc.stdout


def test_live_never_leaks_the_sourced_files_values(tmp_path):
    # the file's VALUES must never reach stdout/stderr on ANY branch
    for kwargs in ({"bare_path": True, "bare_key": True},
                   {"bare_path": True, "env_adds_key": True},
                   {"bare_path": True},
                   {"bare_path": False, "env_adds_path": True,
                    "env_adds_key": True},
                   {"bare_path": False}):
        proc = _run_probe_live(tmp_path, **kwargs)
        assert "NEVER_PRINT_ME" not in proc.stdout, kwargs
        assert "NEVER_PRINT_ME" not in proc.stderr, kwargs


def test_both_probe_commands_are_valid_shell():
    for label, cmd in _both_commands().items():
        proc = subprocess.run(["bash", "-n", "-c", cmd],
                              capture_output=True, text=True)
        assert proc.returncode == 0, (label, proc.stderr)


def test_auto_stop_probe_names_binary_absent_cause():
    ssh = FakeSSH(run_results=[res(90, "NO_RUNPODCTL\nPATH=/usr/bin\n")])
    with pytest.raises(jobs.JobError, match="auto_stop") as exc:
        jobs.launch(ssh, "h", 22, name="t", command="true",
                    workdir="/workspace", pod_id="pid1",
                    max_runtime_sec=60, auto_stop=True)
    assert "NO_RUNPODCTL" in str(exc.value)
    assert "AUTH" not in str(exc.value)                # not an auth failure
    assert len(ssh.run_calls) == 1                     # nothing launched


def test_auto_stop_probe_names_sourced_auth_cause_h1():
    # both attempts refused -> no credential ANYWHERE (H1); sourcing can't fix
    noise = "PATH=" + "/usr/local/bin:" * 20 + "\n-rwxr-xr-x runpodctl\n"
    ssh = FakeSSH(run_results=[res(91, "NO_RUNPODCTL_AUTH_BARE\n" + noise
                                   + "NO_RUNPODCTL_AUTH_SOURCED\n")])
    with pytest.raises(jobs.JobError) as exc:
        jobs.launch(ssh, "h", 22, name="t", command="true",
                    workdir="/workspace", pod_id="pid1",
                    max_runtime_sec=60, auto_stop=True)
    # the terminal sentinel wins over the BARE one, and survives truncation
    assert "NO_RUNPODCTL_AUTH_SOURCED" in str(exc.value)


def test_auto_stop_probe_passes_when_sourcing_rescues_it_h2():
    # AUTH_BARE then PROBE_OK == H2 confirmed: the fix works, launch proceeds
    ssh = FakeSSH(run_results=[res(0, "NO_RUNPODCTL_AUTH_BARE\nPROBE_OK\n"),
                               ok(), ok()])
    jobs.launch(ssh, "h", 22, name="t", command="true", workdir="/workspace",
                pod_id="pid1", max_runtime_sec=60, auto_stop=True)
    assert " 1 </dev/null" in ssh.run_calls[-1]        # armed with auto_stop=1


def test_install_watchdog_failure_names_the_distinguishing_sentinel():
    noise = "PATH=" + "/usr/local/bin:" * 20 + "\n-rwxr-xr-x runpodctl\n"
    ssh = FakeSSH(run_results=[res(91, "NO_RUNPODCTL_AUTH_BARE\n" + noise
                                   + "NO_RUNPODCTL_AUTH_SOURCED\n"
                                     "WATCHDOG_PROBE_FAILED\n")])
    with pytest.raises(jobs.JobError, match="NOT") as exc:
        jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
    assert "NO_RUNPODCTL_AUTH_SOURCED" in str(exc.value)


def test_install_watchdog_binary_absent_is_distinguishable():
    ssh = FakeSSH(run_results=[res(90, "NO_RUNPODCTL\nWATCHDOG_PROBE_FAILED\n")])
    with pytest.raises(jobs.JobError) as exc:
        jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
    assert "NO_RUNPODCTL" in str(exc.value)
    assert "AUTH" not in str(exc.value)


_CAPTURED = ("NO_RUNPODCTL_AUTH_BARE\n"
             "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
             "-rwxr-xr-x 1 root root 9000000 Aug  1 00:00 "
             "/usr/local/bin/runpodctl\n"
             "NO_RUNPODCTL_AUTH_SOURCED\n")


def test_auto_stop_probe_error_carries_the_captured_diagnostics():
    # stdout is the ONLY place the sentinels and the PATH / ls -l captures
    # exist; an error that tells the operator to compare captures must ship
    # them, and the cap must be big enough to actually contain them.
    ssh = FakeSSH(run_results=[res(91, _CAPTURED)])
    with pytest.raises(jobs.JobError) as exc:
        jobs.launch(ssh, "h", 22, name="t", command="true",
                    workdir="/workspace", pod_id="pid1",
                    max_runtime_sec=60, auto_stop=True)
    msg = str(exc.value)
    assert "NO_RUNPODCTL_AUTH_SOURCED" in msg
    assert "PATH=/usr/local/sbin" in msg
    assert "/usr/local/bin/runpodctl" in msg           # the ls -l capture


def test_install_watchdog_error_carries_the_captured_diagnostics():
    ssh = FakeSSH(run_results=[res(91, _CAPTURED + "WATCHDOG_PROBE_FAILED\n")])
    with pytest.raises(jobs.JobError) as exc:
        jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
    msg = str(exc.value)
    assert "NO_RUNPODCTL_AUTH_SOURCED" in msg
    assert "PATH=/usr/local/sbin" in msg
    assert "/usr/local/bin/runpodctl" in msg


def test_probe_errors_scrub_api_keys_out_of_the_captured_stdout():
    leaky = "PATH=/usr/bin\nrpa_LEAKMENOW\nNO_RUNPODCTL_AUTH_SOURCED\n"
    ssh = FakeSSH(run_results=[res(91, leaky)])
    with pytest.raises(jobs.JobError) as exc:
        jobs.launch(ssh, "h", 22, name="t", command="true",
                    workdir="/workspace", pod_id="pid1",
                    max_runtime_sec=60, auto_stop=True)
    assert "rpa_LEAKMENOW" not in str(exc.value)
    ssh = FakeSSH(run_results=[res(91, leaky)])
    with pytest.raises(jobs.JobError) as exc:
        jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)
    assert "rpa_LEAKMENOW" not in str(exc.value)


def test_install_watchdog_arms_after_the_sourced_retry():
    ssh = FakeSSH(run_results=[res(0, "NO_RUNPODCTL_AUTH_BARE\n"
                                      "PATH=/usr/bin\nWATCHDOG_ARMED\n")])
    jobs.install_watchdog(ssh, "h", 22, "podX", idle_minutes=60)   # no raise
