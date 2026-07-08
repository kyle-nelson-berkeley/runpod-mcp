"""Unit 3: remote scripts — syntax lint + load-bearing content assertions.

The wrapper/watchdog carry the real cost hardening (timeout ceiling, exit_code
write, auto-stop ordering, argv-injected pod id). setsid doesn't exist on
macOS, so these are static checks: bash -n floor, shellcheck when available,
and explicit content asserts on every line the consensus plan depends on.
"""
import re
import shutil
import subprocess

import pytest

from tests.conftest import MCP_ROOT, REPO_ROOT

REMOTE = MCP_ROOT / "runpod_mcp" / "remote"
WRAPPER = REMOTE / "job_wrapper.sh"
WATCHDOG = REMOTE / "idle_watchdog.sh"
POD_SETUP = REPO_ROOT / "runbook" / "pod_setup.sh"


@pytest.mark.parametrize("script", [WRAPPER, WATCHDOG], ids=lambda p: p.name)
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", [WRAPPER, WATCHDOG], ids=lambda p: p.name)
def test_shellcheck_if_available(script):
    if not shutil.which("shellcheck"):
        pytest.skip("shellcheck not installed")
    proc = subprocess.run(["shellcheck", "-S", "warning", str(script)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout


# ------------------------------------------------------------- job_wrapper

def test_wrapper_takes_pod_id_via_argv_never_env():
    text = WRAPPER.read_text()
    assert 'POD_ID="$2"' in text
    assert "$RUNPOD_POD_ID" not in text       # env is unreliable when detached


def test_wrapper_sources_rp_environment():
    assert "/etc/rp_environment" in WRAPPER.read_text()


def test_wrapper_timeout_ceiling_and_exit_code():
    text = WRAPPER.read_text()
    assert re.search(
        r'timeout\s+--kill-after=\d+\s+"\$MAX_RUNTIME_SEC"\s+bash\s+cmd\.sh', text)
    assert re.search(r'echo\s+"\$rc"\s*>\s*exit_code', text)


def test_wrapper_auto_stop_after_exit_code_write():
    text = WRAPPER.read_text()
    exit_pos = text.index("> exit_code")
    stop_pos = text.index("runpodctl stop pod")
    assert stop_pos > exit_pos                # timeout can never defeat auto-stop
    assert re.search(r'runpodctl stop pod\s+"\$POD_ID"', text)


def test_wrapper_records_pid_and_keepalive():
    text = WRAPPER.read_text()
    assert re.search(r"echo\s+\$\$\s*>\s*pid", text)
    assert "touch /workspace/.keepalive" in text


def test_wrapper_exports_sane_term_before_running_cmd():
    # Detached BatchMode shells carry a bogus TERM ('ansi+tabs') that kills
    # isaaclab.sh with "unknown terminal type" (observed live: first
    # launch_training run, job 20260706-145815, 2026-07-06). Same fix class
    # as pod_setup.sh's nohup TERM pin.
    text = WRAPPER.read_text()
    term_pos = text.index("export TERM=xterm")
    cmd_pos = text.index("bash cmd.sh")
    assert term_pos < cmd_pos                 # TERM pinned before the payload runs


def test_wrapper_ensures_shader_cache_dirs_exist():
    # A missing target dir makes CUDA/GL silently DISABLE disk caching — the
    # wrapper must self-guarantee these exist before the vars point at them.
    text = WRAPPER.read_text()
    mkdir_match = re.search(
        r"mkdir\s+-p\s+[^\n]*", text)
    assert mkdir_match, "expected a mkdir -p line for the shader cache dirs"
    mkdir_line = mkdir_match.group(0)
    assert "/workspace/omniverse-cache/computecache" in mkdir_line
    assert "/workspace/omniverse-cache/glcache" in mkdir_line
    assert mkdir_match.start() < text.index("bash cmd.sh")


def test_wrapper_exports_shader_cache_vars():
    text = WRAPPER.read_text()
    cmd_pos = text.index("bash cmd.sh")
    for line in (
        "export CUDA_CACHE_PATH=/workspace/omniverse-cache/computecache",
        "export __GL_SHADER_DISK_CACHE=1",
        "export __GL_SHADER_DISK_CACHE_PATH=/workspace/omniverse-cache/glcache",
    ):
        assert line in text
        assert text.index(line) < cmd_pos


def test_wrapper_mkdir_before_shader_cache_exports():
    text = WRAPPER.read_text()
    mkdir_pos = text.index("mkdir -p /workspace/omniverse-cache/computecache")
    export_pos = text.index("export CUDA_CACHE_PATH=/workspace/omniverse-cache/computecache")
    assert mkdir_pos < export_pos


# ------------------------------------------------------------ idle_watchdog

def test_watchdog_pod_id_via_argv_never_env():
    text = WATCHDOG.read_text()
    assert 'POD_ID="$1"' in text
    assert "$RUNPOD_POD_ID" not in text


def test_watchdog_sources_rp_environment_inside_loop():
    text = WATCHDOG.read_text()
    loop_body = text.split("while true", 1)[1]
    assert "/etc/rp_environment" in loop_body  # same detached-env hole as wrapper


def test_watchdog_checks_all_three_idle_conditions():
    text = WATCHDOG.read_text()
    assert "/workspace/jobs/" in text and "kill -0" in text   # live job pids
    assert re.search(r"pgrep.*sshd", text)                    # ssh sessions
    assert "/workspace/.keepalive" in text                    # manual escape hatch
    assert "IDLE_MINUTES" in text


def test_watchdog_five_minute_cadence_and_stop():
    text = WATCHDOG.read_text()
    assert "sleep 300" in text
    assert re.search(r'runpodctl stop pod\s+"\$POD_ID"', text)


# ---------------------------------------------------------------- pod_setup

def test_pod_setup_creates_shader_cache_dirs():
    text = POD_SETUP.read_text()
    assert "$WORKSPACE/omniverse-cache/computecache" in text
    assert "$WORKSPACE/omniverse-cache/glcache" in text
