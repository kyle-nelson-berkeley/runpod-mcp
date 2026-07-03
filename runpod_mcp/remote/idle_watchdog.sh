#!/usr/bin/env bash
# =============================================================================
# idle_watchdog.sh <pod_id> <idle_minutes>
#
# Pod-side idle guard — the real cost leak is a RUNNING pod with no job and a
# dead Mac session ($16/day). Reinstalled by ensure_pod on EVERY transition-
# to-running (the container disk wipes on stop, taking this script with it).
#
# Every 5 min, stop the pod iff ALL THREE hold:
#   1. no live job pid under /workspace/jobs/*/pid
#   2. no interactive sshd session
#   3. /workspace/.keepalive older than <idle_minutes>
#      (`touch /workspace/.keepalive` = manual-session escape hatch)
#
# POD_ID via ARGV (detached shells can't trust RUNPOD_POD_ID); rp_environment
# is re-sourced each tick for runpodctl credentials — same detached-env hole
# as job_wrapper.sh.
# =============================================================================
set -u

POD_ID="$1"
IDLE_MINUTES="${2:-60}"
KEEPALIVE=/workspace/.keepalive

touch "$KEEPALIVE"
echo "idle_watchdog: armed for pod $POD_ID (idle_minutes=$IDLE_MINUTES)"

while true; do
  sleep 300

  # shellcheck disable=SC1091
  [ -f /etc/rp_environment ] && . /etc/rp_environment

  # 1. any live job?
  live=""
  for p in /workspace/jobs/*/pid; do
    [ -f "$p" ] || continue
    if kill -0 "$(cat "$p" 2>/dev/null)" 2>/dev/null; then live=1; break; fi
  done
  [ -n "$live" ] && continue

  # 2. any interactive ssh session?
  if pgrep -f 'sshd: .*@' >/dev/null 2>&1; then continue; fi

  # 3. keepalive fresh enough?
  if [ -f "$KEEPALIVE" ]; then
    now=$(date +%s)
    mtime=$(stat -c %Y "$KEEPALIVE" 2>/dev/null || echo 0)
    if [ $((now - mtime)) -lt $((IDLE_MINUTES * 60)) ]; then continue; fi
  fi

  echo "idle_watchdog: idle >= ${IDLE_MINUTES}m, stopping pod $POD_ID ($(date -u))"
  runpodctl stop pod "$POD_ID" && exit 0
  echo "idle_watchdog: stop attempt failed, will retry next tick"
done
