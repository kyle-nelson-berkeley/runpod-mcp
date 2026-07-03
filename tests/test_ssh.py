"""Unit 3: ssh.py — exact argv assertions on monkeypatched subprocess."""
import base64
import subprocess

import pytest

from runpod_mcp import ssh as sshmod


class Recorder:
    """Stands in for subprocess.run; records argv, returns scripted results."""

    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.results:
            return self.results.pop(0)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def make_client(tmp_path, runner):
    return sshmod.SSHClient(identity=tmp_path / "id_ed25519",
                            known_hosts=tmp_path / "known_hosts_runpod",
                            runner=runner)


def test_run_builds_exact_hardened_argv(tmp_path):
    rec = Recorder()
    c = make_client(tmp_path, rec)
    c.run("1.2.3.4", 15356, "nvidia-smi")
    argv, kwargs = rec.calls[0]
    assert argv == [
        "ssh",
        "-i", str(tmp_path / "id_ed25519"),
        "-p", "15356",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={tmp_path / 'known_hosts_runpod'}",
        "-o", "ConnectTimeout=15",
        "root@1.2.3.4",
        "nvidia-smi",
    ]
    assert kwargs["timeout"] == 60           # default
    assert kwargs["capture_output"] is True


def test_run_check_raises_scrubbed_ssherror(tmp_path):
    boom = subprocess.CompletedProcess([], 255, stdout="",
                                       stderr="auth failed for rpa_SECRET99")
    c = make_client(tmp_path, Recorder([boom]))
    with pytest.raises(sshmod.SSHError) as exc:
        c.run("1.2.3.4", 22, "true", check=True)
    assert "rpa_SECRET99" not in str(exc.value)
    assert "255" in str(exc.value)


def test_push_text_ships_base64_and_chmod(tmp_path):
    rec = Recorder()
    c = make_client(tmp_path, rec)
    content = "#!/bin/bash\necho hi 'quoted' $DOLLAR\n"
    c.push_text("h", 22, content, "/workspace/jobs/j1/cmd.sh", executable=True)
    argv, _ = rec.calls[0]
    remote_cmd = argv[-1]
    b64 = base64.b64encode(content.encode()).decode()
    assert b64 in remote_cmd                          # content survives quoting
    assert "base64 -d > /workspace/jobs/j1/cmd.sh" in remote_cmd
    assert "chmod +x /workspace/jobs/j1/cmd.sh" in remote_cmd
    assert "mkdir -p /workspace/jobs/j1" in remote_cmd


def test_push_file_uses_scp_with_same_hardening(tmp_path):
    rec = Recorder()
    c = make_client(tmp_path, rec)
    local = tmp_path / "pod_setup.sh"
    local.write_text("echo setup")
    c.push_file("5.6.7.8", 40022, local, "/workspace/pod_setup.sh")
    argv, _ = rec.calls[0]
    assert argv[0] == "scp"
    assert "-P" in argv and "40022" in argv           # scp uses -P, not -p
    assert "-o" in argv and "BatchMode=yes" in argv
    assert argv[-2] == str(local)
    assert argv[-1] == "root@5.6.7.8:/workspace/pod_setup.sh"


def test_rsync_pull_never_deletes(tmp_path):
    rec = Recorder()
    c = make_client(tmp_path, rec)
    c.rsync_pull("h", 22, "/workspace/IsaacLab/logs/rsl_rl/", tmp_path / "logs")
    argv, _ = rec.calls[0]
    assert argv[0] == "rsync"
    assert "-az" in argv and "--partial" in argv
    assert not any("--delete" in a for a in argv)
    assert argv[-2] == "root@h:/workspace/IsaacLab/logs/rsl_rl/"
    # the -e remote-shell string carries the full ssh hardening
    e_arg = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in e_arg and "-p 22" in e_arg


def test_truncate_known_hosts(tmp_path):
    c = make_client(tmp_path, Recorder())
    kh = tmp_path / "known_hosts_runpod"
    kh.write_text("stale-host-key\n")
    c.truncate_known_hosts()
    assert kh.read_text() == ""
    c.truncate_known_hosts()                          # idempotent, file empty
    assert kh.exists() and kh.read_text() == ""


def test_conn_cache_ttl():
    t = [1000.0]
    cache = sshmod.ConnCache(ttl_sec=60, clock=lambda: t[0])
    assert cache.get() is None
    cache.put("1.2.3.4", 15356)
    assert cache.get() == ("1.2.3.4", 15356)
    t[0] += 59
    assert cache.get() == ("1.2.3.4", 15356)
    t[0] += 2
    assert cache.get() is None                        # expired
    cache.put("h", 1)
    cache.clear()
    assert cache.get() is None
