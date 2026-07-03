#!/usr/bin/env bash
# =============================================================================
# run.sh — venv-bootstrapping launcher for the runpod MCP server (stdio).
#
# Invoked by .mcp.json at the repo root:  bash runpod-mcp/run.sh
# IDEMPOTENT: creates runpod-mcp/.venv once; pip install is stamp-gated and
# re-runs only when requirements.txt changes. Python 3.14 verified compatible
# with mcp>=1.2 (pre-flight, 2026-07-03).
#
# NOTE: MCP stdio servers must not write to stdout outside the protocol —
# all bootstrap chatter goes to stderr.
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

exec "$VENV/bin/python" "$DIR/server.py"
