#!/usr/bin/env bash
# =============================================================================
# watch.sh — venv-bootstrapping launcher for the observation-loop watcher CLI
# (runpod_mcp.watch): discover the active job, tail its out.log incrementally,
# and report status until it reaches a terminal state. It runs ALONGSIDE
# `supervise` (as a second `run_in_background` Bash task) — supervise owns
# launch/poll/capture/stop; watch only OBSERVES and never mutates the pod.
#
# This is a Mac-side background CLI, NOT an MCP tool — invoke it directly,
# never through .mcp.json (the stdio server's tool surface stays 14 tools on
# purpose; see runpod-mcp/CLAUDE.md §D — "never add a blocking tool").
#
# HONESTY NOTE: live-pod behavior is UNVERIFIED — this CLI was built and
# tested entirely offline against fixture/mock data (see tests/test_watch.py,
# tests/fixtures/watch/); the GPU budget for this build was already spent
# before a real pod run could exercise it end-to-end.
#
# IDEMPOTENT venv bootstrap: mirrors supervise.sh/run.sh — creates
# runpod-mcp/.venv once; pip install is stamp-gated and re-runs only when
# requirements.txt changes.
#
# CAVEAT (documented, not solved): `caffeinate -i` holds off macOS *idle*
# sleep only for the lifetime of this process — it does NOT prevent
# clamshell/lid-close sleep on battery. If the Mac sleeps anyway, the
# pod-side idle watchdog + the job's own wall-clock `timeout --kill-after`
# are the money-safety backstop (see runpod-mcp/CLAUDE.md §D); only the
# completion-notification ergonomic degrades, not the stop guarantee. (watch
# itself never stops anything — it is read-only — so this caveat only
# affects how promptly its status page updates.)
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
STAMP="$VENV/.deps-stamp"

if [ ! -x "$VENV/bin/python" ]; then
  echo "runpod-mcp: creating venv..." >&2
  python3 -m venv "$VENV" >&2
fi

if [ ! -f "$STAMP" ] || [ "$DIR/requirements.txt" -nt "$STAMP" ]; then
  echo "runpod-mcp: installing deps..." >&2
  "$VENV/bin/pip" install -q -r "$DIR/requirements.txt" >&2
  touch "$STAMP"
fi

cd "$DIR" && exec caffeinate -i "$VENV/bin/python" -m runpod_mcp.watch "$@"
