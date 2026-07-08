#!/usr/bin/env bash
# =============================================================================
# job_wrapper.sh <job_dir> <pod_id> <max_runtime_sec> <auto_stop 0|1>
#
# Detached (setsid) job runner for the runpod MCP async-job convention.
# State lives in <job_dir> on the network volume: pid, out.log, exit_code.
#
#  - `timeout` enforces the wall-clock ceiling: exit_code 124 on overrun.
#  - exit_code is written BEFORE the auto-stop suffix, and the suffix runs
#    regardless of how cmd.sh ended — a timeout can never defeat auto-stop.
#  - POD_ID arrives via ARGV: container env vars (RUNPOD_POD_ID) are NOT
#    reliable inside detached BatchMode SSH shells.
#  - /etc/rp_environment is sourced for runpodctl credentials (official
#    runpod/* images write API config there).
# =============================================================================
set -u

JOB_DIR="$1"
POD_ID="$2"
MAX_RUNTIME_SEC="$3"
AUTO_STOP="${4:-0}"

# shellcheck disable=SC1091
[ -f /etc/rp_environment ] && . /etc/rp_environment

cd "$JOB_DIR" || exit 97
echo $$ > pid
touch /workspace/.keepalive

# Detached BatchMode SSH shells carry a bogus TERM ('ansi+tabs') that kills
# isaaclab.sh with "unknown terminal type" — pin the same value as
# pod_setup.sh's nohup fix.
export TERM=xterm

# Redirect the CUDA-JIT + GL-shader DISK caches onto the volume so a completed
# compile survives a pod stop (/root is wiped on every stop). Defensive / one-
# time-ifying — NOT a confirmed fix for the boot-hang. mkdir first: a missing
# target dir makes CUDA/GL silently DISABLE caching.
mkdir -p /workspace/omniverse-cache/computecache /workspace/omniverse-cache/glcache
export CUDA_CACHE_PATH=/workspace/omniverse-cache/computecache
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_PATH=/workspace/omniverse-cache/glcache

# --kill-after: SIGKILL 60s after SIGTERM — a job that traps/ignores TERM
# must never outlive the ceiling (that would also defeat auto-stop)
timeout --kill-after=60 "$MAX_RUNTIME_SEC" bash cmd.sh >> out.log 2>&1
rc=$?
echo "$rc" > exit_code
rm -f pid
# job end = fresh idle window (idle_watchdog counts from .keepalive mtime)
touch /workspace/.keepalive

if [ "$AUTO_STOP" = "1" ]; then
  echo "job_wrapper: auto-stop (exit_code=$rc)" >> out.log
  runpodctl stop pod "$POD_ID" >> out.log 2>&1 || true
fi
