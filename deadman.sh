#!/usr/bin/env bash
# =============================================================================
# deadman.sh — venv-bootstrapping launcher for the independent Mac-side
# stop-pod fuse (runpod_mcp.deadman): arm -> sleep ~N hours -> stop the pod
# with retries, unless cancelled. It exists ALONGSIDE supervise.sh /
# hand-driven sessions as a money-safety backstop: a prior run lost ~$4.5
# when the process supervising a training run died and the pod idled ~7.2h
# unnoticed — this fuse does not depend on that (or any) supervising process
# staying alive (see runpod-mcp/CLAUDE.md §D).
#
# This is a Mac-side background CLI, NOT an MCP tool — invoke it directly,
# never through .mcp.json (the stdio server's tool surface stays 14 tools on
# purpose; see runpod-mcp/CLAUDE.md §D — "never add a blocking tool").
#
# IDEMPOTENT venv bootstrap: mirrors supervise.sh/run.sh — creates
# runpod-mcp/.venv once; pip install is stamp-gated and re-runs only when
# requirements.txt changes.
#
# Usage:
#   ./deadman.sh arm --hours 3.0 &   # background it; re-arm before it fires
#                                     # to extend (cancel, then a fresh arm)
#   ./deadman.sh status               # armed / LOST / last outcome — no network
#   ./deadman.sh cancel                # disarm before a normal stop_pod
#
# CAVEAT (documented, not solved): `caffeinate -i` holds off macOS *idle*
# sleep only for the lifetime of this process — it does NOT prevent
# clamshell/lid-close sleep on battery. If the Mac sleeps anyway, the
# pod-side idle watchdog is the guaranteed money backstop either way; only
# this fuse's own guarantee degrades along with the sleeping Mac.
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

cd "$DIR" && exec caffeinate -i "$VENV/bin/python" -m runpod_mcp.deadman "$@"
