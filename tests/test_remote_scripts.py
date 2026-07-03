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

from tests.conftest import MCP_ROOT

REMOTE = MCP_ROOT / "runpod_mcp" / "remote"
WRAPPER = REMOTE / "job_wrapper.sh"
WATCHDOG = REMOTE / "idle_watchdog.sh"


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
