#!/usr/bin/env bash
# =============================================================================
# chain-011-CUREE_Adaptive-weights.sh — the ONE chain for campaign
# 011-CUREE_Adaptive-weights: does ADAPTING reward weights during training
# beat 010's fixed champion arm C (0.25/0.55/0.1)? Every link in this
# campaign trains at that SAME fixed pos/ang/act triple (unlike 010, there
# is no per-arm weight split) — DR_2, 2500 iters, 3.0 s episodes, checkpoint
# model_2499.pt:
#
#   PROBE     seed 1, 300 iters, cadence 25,  adapt ON  — quarantined, never
#             in the mapping; proves the patch + oracle wiring end to end
#             cheaply before any full-length training money.
#   CONTROL   seed 2, 2500 iters, adapt OFF   — the pre-registered "did the
#             patch change anything when the flag is off" determinism check
#             (must reproduce 010's arm-C seed-2 run: mean reward ~121.26).
#   ADAPTIVE  seeds 1/2/3, 2500 iters, cadence 100, adapt ON — the actual
#             experiment: 3 seeds of the adaptation rule running live.
#
# TWO PHASES, fired by the PARENT session (the one holding Kyle's direct
# spend-envelope approval), each as ONE `run_in_background` Bash task:
#
#   bash runpod-mcp/chains/chain-011-CUREE_Adaptive-weights.sh train
#       pre-flight (reconcile sync + mount-health gate + pod-side pins) ->
#       ADAPTIVE-PATCH LANDING (build the doubly-patched warpauv_env.py
#       Mac-side, push it pod-side, verify byte-for-byte) -> 300-iter probe
#       (quarantined, oracle-gated) -> CONTROL link (adapt OFF, determinism
#       record) -> 3 ADAPTIVE links (adapt ON, oracle-gated) -> emits
#       logs/pod/chain-011-rundirs.tsv (4 data rows + calibration comment
#       blocks) -> exits with the POD UP.
#
#   == STOP POINT == the parent session reviews chain-011-rundirs.tsv (4
#       rows, control's reward-proxy already enforced, calibration comments
#       for probe + every adaptive seed) and then fires:
#
#   bash runpod-mcp/chains/chain-011-CUREE_Adaptive-weights.sh rollouts [tsv]
#       read-only pod_status pre-check (stopped pod -> instructs ensure_pod,
#       NO auto bring-up: pod creation stays human-initiated, exit 3) ->
#       parse + validate the TSV BEFORE any spend (exactly 4 rows: the
#       CONTROL_2 / ADAPTIVE_1 / ADAPTIVE_2 / ADAPTIVE_3 combo set, char-exact
#       weights, model_2499.pt everywhere, local checkpoint file present —
#       any violation is a refusal: exit 3, no stop, no spend) -> 4 rollout
#       links in TSV order; links 1-3 --no-stop, link 4 OWNS the stop.
#
# THE ORACLE GATE (the money gate this whole campaign hinges on): every
# training/probe link that ran with adapt ON writes a JSONL event log
# pod-side (self.cfg.adapt_log_path); before that link is trusted for
# anything, its SYNCED LOCAL COPY is replayed through
# src/adaptive_analysis.py's `validate-jsonl` CLI (sequential replay against
# src/adaptive_weights.apply_rule — T1 is the oracle for what actually drove
# the GPU). A non-zero exit ABORTS the chain immediately: a corrupted or
# drifted adaptation rule must never be allowed to burn money on 3 more
# training links. WARN lines (non-fatal, e.g. an inert-trajectory heads-up)
# are logged but never abort. Each gate's SUMMARY_JSON line is appended to
# the TSV as a `# CALIBRATION(...)` comment for the offline scoring pass.
#
# THE ADAPTIVE-PATCH LANDING GATE (new vs. 010 — this campaign's whole
# premise depends on it): src/apply_adaptive_patch.py inserts the pod-side
# torch implementation of the SAME frozen rule as five content-anchored
# edits ON TOP OF the seed-respect patch. This is built FRESH Mac-side every
# chain run (never assumed already-landed), its sha256 is cross-checked
# against the module's own ADAPTIVE_PATCHED_SHA256 pin (catching a stale
# checkout of apply_adaptive_patch.py itself), pushed pod-side via
# `rt.ssh.push_text`, and — BEFORE any launch rewrites the DR lines — its
# on-pod sha256, ADAPTIVE-REW marker count, and surviving seed-respect
# invariant (exactly one `torch.manual_seed(0)`) are all re-verified. Every
# subsequent `launch_training` call's own content-anchored DR-rewrite
# (`training.apply_dr_to_source`) touches ONLY the two DR lines, so the
# adaptive patch survives every link untouched — verified in T2's own test
# suite, not merely assumed here.
#
# THE WEIGHT GUARD (retained from 010, unchanged mechanism): every link's
# guard reads the run's params/env.yaml BOTH pod-side (immediately post-link)
# AND the synced local copy, asserting char-exact fixed-string lines
# (count==1 each) for "rew_scale_pos: 0.25" / "rew_scale_ang: 0.55" /
# "rew_scale_actions: 0.1" — NEVER grep the SOURCE (warpauv_env.py always
# shows source defaults regardless of what a run actually trained with).
#
# THE FLAG GUARD (new vs. 010): a sibling single-line checker
# (`flag_line_check`, deliberately simpler than weight_lines_check — no
# fake-null branch, per the agreed "keep it simple" scope) asserts
# "adapt_rew_weights: true" (probe/adaptive) or "adapt_rew_weights: false"
# (control) is present char-exact, count==1, both pod-side and local — a
# Hydra override of adapt_rew_weights that silently fails to land would
# otherwise be invisible.
#
# THE JSONL FRESHNESS CONTRACT (existence == freshness): sync_logs/rsync is
# NON-DELETING, so a local adapt_011_*.jsonl left by a PREVIOUS chain
# attempt would otherwise survive untouched — if a rerun's log-path
# override then silently fails to land, the post-link sync would leave
# that stale file in place and the oracle would happily validate LAST
# attempt's events, wrongly passing the gate on a broken rule. Every
# oracle-gated link (probe + each adaptive seed) therefore clears BOTH
# sides (pod-side via `pod_rm_f`, local via plain `rm -f`) BEFORE
# launching, then requires the local file to EXIST AND BE NON-EMPTY
# afterward — since it was just deleted, existence can only mean THIS
# link wrote it. A `phase_train` pre-flight sweep additionally clears
# every adapt_011_*.jsonl on both sides once at the very start (belt and
# suspenders for a fresh chain run, on top of the per-link clears that
# also cover a mid-chain rerun).
#
# THE NO-JSONL GUARD (control link only): the control link passes NO
# adapt_* overrides at all (the patch's own class default is
# `adapt_rew_weights = False`), so it must write NOTHING to
# .../warpauv_direct/adapt_011_*.jsonl — the pod-side glob count must stay
# at exactly 1 (the probe's own file) after the control link completes.
#
# MEAN-REWARD DETERMINISM RECORD (control link only, train-phase PROXY --
# the hard determinism gate happens at offline scoring): the LAST "Mean
# reward" value in the job's out.log must sit within 1.0 of 010's committed
# arm-C seed-2 value (121.26), via a $ROOT_PY float compare.
#
# CHECKPOINT / DR / MEAN-REWARD-COUNT conventions (retained from 010): every
# 2500-iter link's final checkpoint is model_2499.pt (RSL-RL's
# max_iterations-1 convention); DR_2's two active cfg lines
# (com_to_cob_offset_radius = 0.05 / volume_range = [...]) are re-verified
# pod-side after every link (launch_training's own DR-rewrite recovers a
# checkout parked at a different level); reward-line count in out.log must
# equal max_iterations EXACTLY (equality, not >=, catches a dropped override).
#
# MOUNT-HEALTH GATE (new vs. 010 — Isaac boot-hang doctrine: it is the
# MooseFS mount, never the driver): before ANY spend beyond the already-
# running pod, one exec_on_pod times (a) a stat of /workspace/isaacsim and
# (b) an extension.toml find-storm under it, via `date +%s%N` deltas. PASS
# iff stat<2000ms, storm<10000ms, files>0 (history: healthy storm ~2-3s,
# degraded 17.6s, a wedged stat hangs >8s) — else abort and let the parent
# retry later rather than burn setup time on a sick mount.
#
# FAILURE DOCTRINE (verbatim 010): run_link retries ONCE after 60s, ONLY when
# the supervise summary shows NO job_id (a pre-launch failure); any other
# failure -> abort() -> stop_pod(force) x3 with 20s gaps -> exit 1. Never
# terminate_pod. Rollouts phase never brings the pod up itself. A non-fatal
# spend_report read failure is logged and the chain continues (supervise's
# own caps + the deadman fuse bound the burn either way).
#
# SPEND GUARDS: TRAIN_GUARD_USD=$2.75 + ROLLOUT_GUARD_USD=$1.25.
# check_spend runs after the probe, after each training link (control +
# each adaptive seed), and after rollout links 1-3 (010's placement — link 4
# owns the stop, so no post-check follows it).
#
# SECRETS: the RunPod API key stays in the macOS Keychain — tools.runtime()
# reads it internally; this script never sees, echoes, or writes it.
# =============================================================================
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../runpod-mcp/chains
MCPDIR="$(cd "$DIR/.." && pwd)"                           # .../runpod-mcp
REPO="$(cd "$MCPDIR/.." && pwd)"                          # repo root
VENV="$MCPDIR/.venv"
PY="$VENV/bin/python"
STAMPFILE="$VENV/.deps-stamp"
SUPERVISE="$MCPDIR/supervise.sh"
LOGDIR="$REPO/logs/pod"
RUNDIRS_LOCAL="$LOGDIR/rsl_rl/warpauv_direct"
MAPFILE_DEFAULT="$LOGDIR/chain-011-rundirs.tsv"
WARPAUV_ENV_REMOTE="/workspace/isaac-auv-env/warpauv_env.py"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

# ROOT repo venv (numpy/pytest/pyyaml, NO torch/mcp deps) — used for every
# oracle/builder shell-out and the mean-reward float compare. NEVER the MCP
# venv ($PY above), which carries fastmcp/httpx/paramiko instead.
ROOT_PY="$REPO/.venv/bin/python"
ORACLE="$REPO/src/adaptive_analysis.py"
BUILDER="$REPO/src/apply_adaptive_patch.py"

# Spend guards (deltas from each phase's own baseline).
TRAIN_GUARD_USD="2.75"
ROLLOUT_GUARD_USD="1.25"

# DR_2 is the only level this campaign ever touches (curee DR_TABLES["DR_2"],
# unit-tested char-exact against runpod_mcp/training.py).
DR2_RADIUS="0.05"
DR2_VRANGE="[0.019747843530591773, 0.02574784353059178]"

# Fixed reward-weight triple for EVERY link this campaign ever touches (char-
# exact guard values) — unlike 010, there is no per-arm split: probe,
# control, and all three adaptive seeds train at 010's champion arm C.
POS="0.25"
ANG="0.55"
ACT="0.1"
WEIGHTS="0.25/0.55/0.1"

REMOTE_JSONL_DIR="/workspace/IsaacLab/logs/rsl_rl/warpauv_direct"
PROBE_JSONL_REMOTE="$REMOTE_JSONL_DIR/adapt_011_probe.jsonl"
PROBE_JSONL_LOCAL="$RUNDIRS_LOCAL/adapt_011_probe.jsonl"

# Pinned externally (runbook/HANDOFF-experiment-G.md 1.7) — the play_video
# port-bug fix's committed sha256.
PLAY_ROLLOUT_EVAL_SHA="9caee847f11484c0b63bbd6d039efb3944d65b2e0f85845513d841bca338d486"

# 010's committed arm-C seed-2 champion mean reward (control determinism
# proxy target; the hard gate is adaptive_analysis.control_determinism_check
# at offline scoring).
CONTROL_TARGET_REWARD="121.26"
CONTROL_REWARD_BAND="1.0"

log() { printf '[chain-011 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

usage() {
  echo "usage: $0 train | rollouts [mapping.tsv]" >&2
  exit 2
}

# ------------------------------------------------------------ venv bootstrap
# Idempotent, mirrors supervise.sh/deadman.sh (the chain calls tools.* helpers
# before the first supervise.sh link, so it cannot rely on that bootstrap).
bootstrap_venv() {
  if [ ! -x "$PY" ]; then
    echo "runpod-mcp: creating venv..." >&2
    python3 -m venv "$VENV" >&2 || exit 1
  fi
  if [ ! -f "$STAMPFILE" ] || [ "$MCPDIR/requirements.txt" -nt "$STAMPFILE" ]; then
    echo "runpod-mcp: installing deps..." >&2
    "$VENV/bin/pip" install -q -r "$MCPDIR/requirements.txt" >&2 || exit 1
    touch "$STAMPFILE"
  fi
}

# ------------------------------------------------------- tools.* one-liners
# All run with cwd=$MCPDIR (same as supervise.sh/deadman.sh launchers).

spend_now() {
  (cd "$MCPDIR" && "$PY" -c \
    'from runpod_mcp import tools; print(tools.spend_report(tools.runtime())["total_usd"])' \
    2>/dev/null)
}

pod_status_now() {
  (cd "$MCPDIR" && "$PY" -c \
    'from runpod_mcp import tools; print(tools.pod_status(tools.runtime()).get("status"))' \
    2>/dev/null)
}

stop_pod_force() {
  (cd "$MCPDIR" && "$PY" - <<'PYEOF'
import sys, time
from runpod_mcp import tools
rt = tools.runtime()
for i in range(3):
    try:
        r = tools.stop_pod(rt, force=True)
        print(f"stop_pod attempt {i+1}: {r.get('status')}", flush=True)
        if r.get("status") in ("stopped", "already_stopped", "no_pod"):
            sys.exit(0)
    except Exception as exc:  # noqa: BLE001 — must keep retrying on the way out
        print(f"stop_pod attempt {i+1} FAILED: {exc}", flush=True)
    time.sleep(20)
sys.exit(1)
PYEOF
  )
}

sync_logs_now() {
  (cd "$MCPDIR" && "$PY" -c \
    'from runpod_mcp import tools; tools.sync_logs(tools.runtime()); print("SYNC_OK")')
}

# Pod-side DR guard: count the two ACTIVE cfg lines (fixed-string grep; the
# commented-out No-DR preset lines never collide with DR_2's values).
pod_dr_check() {  # $1 = radius string, $2 = volume_range string
  (cd "$MCPDIR" && "$PY" - "$1" "$2" "$WARPAUV_ENV_REMOTE" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools
rad, vr, path = sys.argv[1], sys.argv[2], sys.argv[3]
q1 = shlex.quote(f"com_to_cob_offset_radius = {rad} #")
q2 = shlex.quote(f"volume_range = {vr} #")
cmd = f"echo R=$(grep -Fc {q1} {path}) V=$(grep -Fc {q2} {path})"
out = tools.exec_on_pod(tools.runtime(), cmd, timeout_sec=60)
text = (out.get("stdout") or "").strip()
if out.get("exit_code") != 0:
    print(f"DR_CHECK_FAIL exec exit={out.get('exit_code')} out={text[:200]}")
    sys.exit(1)
if "R=1 V=1" in text:
    print(f"DR_OK radius={rad}")
    sys.exit(0)
print(f"DR_CHECK_FAIL want R=1 V=1, got '{text[:200]}' (radius={rad})")
sys.exit(1)
PYEOF
  )
}

pod_quarantine_probe() {  # $1 = probe run-dir basename
  (cd "$MCPDIR" && "$PY" - "$1" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools
d = shlex.quote(sys.argv[1])
cmd = ("cd /workspace/IsaacLab/logs/rsl_rl/warpauv_direct && "
       f"mkdir -p _probe_quarantine && mv {d} _probe_quarantine/ && echo QUARANTINED")
out = tools.exec_on_pod(tools.runtime(), cmd, timeout_sec=60)
text = (out.get("stdout") or "").strip()
print(text[:200])
sys.exit(0 if out.get("exit_code") == 0 and "QUARANTINED" in text else 1)
PYEOF
  )
}

# Best-effort pod-side `rm -f` of a stale JSONL BEFORE a link launches (the
# probe/adaptive links' oracle gate reads whatever is at this path post-sync,
# so a leftover file from an earlier aborted attempt must never be mistaken
# for this link's own log). $1 = remote path.
pod_rm_f() {
  (cd "$MCPDIR" && "$PY" - "$1" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools
path = shlex.quote(sys.argv[1])
out = tools.exec_on_pod(tools.runtime(), f"rm -f {path} && echo RM_OK", timeout_sec=60)
text = (out.get("stdout") or "").strip()
print(text[:200])
sys.exit(0 if out.get("exit_code") == 0 and "RM_OK" in text else 1)
PYEOF
  )
}

# NO-JSONL guard support: count adapt_011_*.jsonl files pod-side.
pod_jsonl_count() {
  (cd "$MCPDIR" && "$PY" - "$REMOTE_JSONL_DIR" <<'PYEOF'
import sys
from runpod_mcp import tools
d = sys.argv[1]
out = tools.exec_on_pod(tools.runtime(), f"ls {d}/adapt_011_*.jsonl 2>/dev/null | wc -l", timeout_sec=60)
text = (out.get("stdout") or "0").strip()
print(text or "0")
sys.exit(0 if out.get("exit_code") == 0 else 1)
PYEOF
  )
}

# PRE-FLIGHT JSONL SWEEP (codex round 2 finding, pod side): rm -f every
# adapt_011_*.jsonl pod-side ONCE at the very start of the train phase, so
# the campaign starts from ZERO on both sides and the control link's
# NO-JSONL guard (count==1, the probe's own) can never be poisoned by a
# leftover file from an earlier chain attempt. Per-link clearing
# (train_link_guarded's own pre-launch clear) remains in place too --
# belt and suspenders for a mid-chain rerun starting after this sweep.
pod_sweep_jsonl() {
  (cd "$MCPDIR" && "$PY" - "$REMOTE_JSONL_DIR" <<'PYEOF'
import sys
from runpod_mcp import tools
d = sys.argv[1]
out = tools.exec_on_pod(tools.runtime(), f"rm -f {d}/adapt_011_*.jsonl && echo SWEEP_OK", timeout_sec=60)
text = (out.get("stdout") or "").strip()
print(text[:200])
sys.exit(0 if out.get("exit_code") == 0 and "SWEEP_OK" in text else 1)
PYEOF
  )
}

# MOUNT-HEALTH GATE: one exec_on_pod, pod-side date +%s%N deltas around (a) a
# stat and (b) an extension.toml find-storm. Prints "STAT_MS=.. STORM_MS=..
# FILES=.."; exit 0 iff STAT_MS<2000, STORM_MS<10000, FILES>0.
mount_health_check() {
  (cd "$MCPDIR" && "$PY" - <<'PYEOF'
import re, sys
from runpod_mcp import tools
rt = tools.runtime()
cmd = (
    "t0=$(date +%s%N); timeout 10 stat /workspace/isaacsim >/dev/null 2>&1; "
    "t1=$(date +%s%N); STAT_MS=$(( (t1-t0)/1000000 )); "
    "t2=$(date +%s%N); "
    "FILES=$(timeout 60 find /workspace/isaacsim/exts -maxdepth 3 -name extension.toml 2>/dev/null | wc -l); "
    "t3=$(date +%s%N); STORM_MS=$(( (t3-t2)/1000000 )); "
    'echo "STAT_MS=$STAT_MS STORM_MS=$STORM_MS FILES=$FILES"'
)
out = tools.exec_on_pod(rt, cmd, timeout_sec=90)
text = (out.get("stdout") or "").strip()
if out.get("exit_code") != 0:
    print(f"MOUNT_HEALTH_FAIL exec exit={out.get('exit_code')} out={text[:200]}")
    sys.exit(1)
m = re.search(r"STAT_MS=(\d+)\s+STORM_MS=(\d+)\s+FILES=(\d+)", text)
if not m:
    print(f"MOUNT_HEALTH_FAIL unparseable output: {text[:200]}")
    sys.exit(1)
stat_ms, storm_ms, files = (int(x) for x in m.groups())
print(f"STAT_MS={stat_ms} STORM_MS={storm_ms} FILES={files}")
sys.exit(0 if (stat_ms < 2000 and storm_ms < 10000 and files > 0) else 1)
PYEOF
  )
}

# PIN: play_rollout_eval.py must match the committed post-fix sha256.
play_rollout_sha_check() {
  (cd "$MCPDIR" && "$PY" -c \
    'from runpod_mcp import tools; import sys
out = tools.exec_on_pod(tools.runtime(), "sha256sum /workspace/isaac-auv-env/custom_workflows/play_rollout_eval.py", timeout_sec=60)
text = (out.get("stdout") or "").strip()
sha = text.split()[0] if text.split() else ""
print(f"PLAY_ROLLOUT_EVAL_SHA={sha}")
sys.exit(0 if out.get("exit_code") == 0 and sha == sys.argv[1] else 1)' \
    "$PLAY_ROLLOUT_EVAL_SHA")
}

# PIN: the checkout must NOT be BlueROV2-patched (symmetric gate, see
# tools._check_vehicle_gates — this is an EXPLICIT early pre-flight of the
# same invariant launch_training checks internally later, so a bad checkout
# is caught before the mount-health/landing steps, not just at first launch).
markers_gate_check() {
  (cd "$MCPDIR" && "$PY" - <<'PYEOF'
import sys
from runpod_mcp import tools
out = tools.exec_on_pod(tools.runtime(), "ls /workspace/markers/ 2>/dev/null", timeout_sec=30)
text = (out.get("stdout") or "")
print(f"MARKERS_LS={text.strip()[:200]!r}")
sys.exit(1 if "bluerov2_patch_applied" in text else 0)
PYEOF
  )
}

# ADAPTIVE-PATCH LANDING: push the Mac-built doubly-patched source pod-side
# and re-verify sha256 + marker count + surviving seed-respect invariant --
# BEFORE any launch_training call ever rewrites the DR lines.
# $1 = local tmp file (the built patch bytes) $2 = pinned sha256
# $3 = pinned ADAPTIVE-REW marker count
pod_land_adaptive_patch() {
  (cd "$MCPDIR" && "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys
from runpod_mcp import tools

local_path, pin_sha, pin_markers = sys.argv[1], sys.argv[2], sys.argv[3]
rt = tools.runtime()
host, port = tools._conn_info(rt)
remote = "/workspace/isaac-auv-env/warpauv_env.py"

# Idempotent one-time backup — never overwrite an existing backup (the
# FIRST landing's pre-adaptive-patch bytes are what recovery needs).
backup_cmd = ("cd /workspace/isaac-auv-env && "
              "([ -f warpauv_env.py.pre-011 ] || cp warpauv_env.py warpauv_env.py.pre-011)")
out = tools.exec_on_pod(rt, backup_cmd, timeout_sec=60)
if out.get("exit_code") != 0:
    print(f"LANDING_FAIL backup: {(out.get('stdout') or '')[:200]} {(out.get('stderr') or '')[:200]}")
    sys.exit(1)

text = open(local_path).read()
rt.ssh.push_text(host, port, text, remote)

out = tools.exec_on_pod(rt, f"sha256sum {remote}", timeout_sec=60)
stdout = (out.get("stdout") or "").strip()
sha = stdout.split()[0] if stdout.split() else ""
if out.get("exit_code") != 0 or sha != pin_sha:
    print(f"LANDING_FAIL sha mismatch: got {sha!r} want {pin_sha!r}")
    sys.exit(1)

out = tools.exec_on_pod(rt, f"grep -c ADAPTIVE-REW {remote}", timeout_sec=60)
count_text = (out.get("stdout") or "").strip()
if out.get("exit_code") != 0 or count_text != pin_markers:
    print(f"LANDING_FAIL marker count: got {count_text!r} want {pin_markers!r}")
    sys.exit(1)

out = tools.exec_on_pod(rt, f"grep -c 'torch.manual_seed(0)' {remote}", timeout_sec=60)
seed_count_text = (out.get("stdout") or "").strip()
if out.get("exit_code") != 0 or seed_count_text != "1":
    print(f"LANDING_FAIL seed-respect count: got {seed_count_text!r} want '1'")
    sys.exit(1)

print(f"LANDING_OK sha={sha} markers={count_text} seed_count={seed_count_text}")
sys.exit(0)
PYEOF
  )
}

# Runtime weight guard, ONE side (local file or pod-side path via
# exec_on_pod). $1 = mode (local|remote) $2 = path $3=pos $4=ang $5=act.
# Exit 0 = char-exact match (count==1 each). Exit 2 = mismatch AND the
# fake-null signature (010's BASELINE triple 0.2/0.5/0.2, the source-file
# defaults) is present instead — the overrides were silently ignored.
# Exit 1 = mismatch, not fake-null.
weight_lines_check() {
  (cd "$MCPDIR" && "$PY" - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools

mode, path, pos, ang, act = sys.argv[1:6]
wanted = {
    "pos": f"rew_scale_pos: {pos}",
    "ang": f"rew_scale_ang: {ang}",
    "act": f"rew_scale_actions: {act}",
}
# The fake-null signature is warpauv_env.py's own SOURCE defaults --
# independent of what this campaign's triple is, a silently-ignored
# override always reproduces THIS, never some other wrong value.
fake_null = {
    "pos": "rew_scale_pos: 0.2",
    "ang": "rew_scale_ang: 0.5",
    "act": "rew_scale_actions: 0.2",
}

if mode == "local":
    try:
        content = open(path).read().splitlines()
    except OSError as exc:
        print(f"WEIGHT_CHECK_FAIL open error: {exc}")
        sys.exit(1)
    def get_count(want):
        return sum(1 for line in content if line == want)
elif mode == "remote":
    rt = tools.runtime()
    def get_count(want):
        q = shlex.quote(want)
        out = tools.exec_on_pod(rt, f"grep -Fxc {q} {shlex.quote(path)}", timeout_sec=60)
        try:
            return int((out.get("stdout") or "0").strip() or "0")
        except ValueError:
            return 0
else:
    print(f"WEIGHT_CHECK_FAIL bad mode {mode!r}")
    sys.exit(1)

counts = {k: get_count(v) for k, v in wanted.items()}
if all(v == 1 for v in counts.values()):
    print(f"WEIGHT_OK mode={mode} pos={pos} ang={ang} act={act}")
    sys.exit(0)

null_counts = {k: get_count(v) for k, v in fake_null.items()}
is_fake_null = all(v == 1 for v in null_counts.values())
print(f"WEIGHT_CHECK_FAIL mode={mode} counts={counts} "
      f"fake_null_counts={null_counts} fake_null={is_fake_null}")
sys.exit(2 if is_fake_null else 1)
PYEOF
  )
}

# Sibling to weight_lines_check, but for ONE arbitrary char-exact line (used
# for the adapt_rew_weights: true/false flag). Deliberately simpler than
# weight_lines_check -- no fake-null branch (a boolean flag has no separate
# "reproduces the source defaults" signature worth distinguishing from a
# plain mismatch; keeping this check to a single exit-code class is the
# agreed scope). $1=mode(local|remote) $2=path $3=wanted_line.
# Exit 0 = count==1. Exit 1 = anything else.
flag_line_check() {
  (cd "$MCPDIR" && "$PY" - "$1" "$2" "$3" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools

mode, path, wanted = sys.argv[1:4]

if mode == "local":
    try:
        content = open(path).read().splitlines()
    except OSError as exc:
        print(f"FLAG_CHECK_FAIL open error: {exc}")
        sys.exit(1)
    count = sum(1 for line in content if line == wanted)
elif mode == "remote":
    rt = tools.runtime()
    q = shlex.quote(wanted)
    out = tools.exec_on_pod(rt, f"grep -Fxc {q} {shlex.quote(path)}", timeout_sec=60)
    try:
        count = int((out.get("stdout") or "0").strip() or "0")
    except ValueError:
        count = 0
else:
    print(f"FLAG_CHECK_FAIL bad mode {mode!r}")
    sys.exit(1)

if count == 1:
    print(f"FLAG_OK mode={mode} line={wanted!r}")
    sys.exit(0)
print(f"FLAG_CHECK_FAIL mode={mode} line={wanted!r} count={count}")
sys.exit(1)
PYEOF
  )
}

# ------------------------------------------------------------ small helpers

summary_field() {  # $1 = summary json path, $2 = field
  "$PY" -c \
    'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
    "$1" "$2" 2>/dev/null
}

float_gt() {  # exit 0 iff $1 > $2
  "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$1" "$2"
}

# Mean-reward determinism band check: exit 0 iff |$1 - $2| <= $3. Uses
# ROOT_PY (not PY) -- this is a repo-level numeric compare, not an
# MCP-runtime call, and belongs with the oracle/builder shell-outs.
float_close() {
  "$ROOT_PY" -c \
    'import sys; a,b,tol=(float(x) for x in sys.argv[1:4]); sys.exit(0 if abs(a-b)<=tol else 1)' \
    "$1" "$2" "$3"
}

list_rundirs() {  # top-level local synced run dirs only (quarantine subdir excluded)
  [ -d "$RUNDIRS_LOCAL" ] || return 0
  (cd "$RUNDIRS_LOCAL" && find . -maxdepth 1 -type d -name '20*_*' | sed 's|^\./||' | sort)
}

reward_count() {  # $1 = job_id — prints a count (0 if the log is missing)
  local f="$LOGDIR/jobs/$1/out.log" c
  [ -f "$f" ] || { echo 0; return; }
  c=$(grep -c "Mean reward" "$f") || true   # grep -c prints 0 (exit 1) on no match
  echo "${c:-0}"
}

# Last "Mean reward: X" value in a job's out.log (empty string if none).
last_mean_reward() {  # $1 = job_id
  local f="$LOGDIR/jobs/$1/out.log"
  [ -f "$f" ] || { echo ""; return; }
  grep "Mean reward:" "$f" | tail -1 | sed -E 's/.*Mean reward: *([0-9.eE+-]+).*/\1/'
}

# One-line human summary of an oracle SUMMARY_JSON payload, for the TSV
# calibration comment block. $1 = raw JSON text (WITHOUT the "SUMMARY_JSON: "
# prefix).
oracle_summary_human_line() {
  "$ROOT_PY" -c \
    'import json, sys
d = json.loads(sys.argv[1])
hi = d.get("clamp_pin_fraction_hi")
warn = d.get("inert_warn")
n = d.get("n_samples")
print(f"clamp_pin_fraction_hi={hi} inert_warn={warn} n_samples={n}")' \
    "$1" 2>/dev/null
}

check_spend() {  # $1 = baseline, $2 = guard — exit 1 on breach
  local now delta
  now=$(spend_now)
  if [ -z "$now" ]; then
    log "spend guard: spend_report failed (non-fatal; supervise caps + deadman bound the burn)"
    return 0
  fi
  delta=$("$PY" -c 'import sys; print(f"{float(sys.argv[1])-float(sys.argv[2]):.3f}")' "$now" "$1")
  log "spend guard: spend_total=$now delta=$delta (guard=$2)"
  if float_gt "$delta" "$2"; then
    log "SPEND GUARD BREACH — aborting"
    return 1
  fi
  return 0
}

abort() {  # $* = reason; explicit stop, then non-zero exit
  log "ABORT: $*"
  log "ensuring pod stopped (force=True); chain rc=1"
  if stop_pod_force; then
    log "stop_pod done"
  else
    log "WARNING: stop_pod FAILED after 3 attempts — pod may still be running; check pod_status NOW"
  fi
  exit 1
}

run_link() {  # $1 = label, $2 = summary path, $3.. = supervise.sh args
  local label="$1" summary="$2" attempt rc jid
  shift 2
  for attempt in 1 2; do
    log "START $label (attempt $attempt)"
    "$SUPERVISE" "$@" --summary-path "$summary"
    rc=$?
    log "END $label rc=$rc (summary: $summary)"
    [ "$rc" -eq 0 ] && return 0
    jid=""
    [ -f "$summary" ] && jid=$(summary_field "$summary" job_id)
    if [ "$attempt" -eq 1 ] && [ -z "$jid" ]; then
      log "RETRY $label: failure was pre-launch (no job_id) — retrying once in 60s"
      sleep 60
      continue
    fi
    return "$rc"
  done
  return 1
}

# One training/probe link + every per-link guard. Sets NEW_DIR + JOB_ID.
# $1=label $2=seed $3=iters $4=max_wait $5=summary_path $6=extra_adapt_args
# ("" for control) $7=flag_wanted (true|false) $8=jsonl_remote_to_clear
# ("" to skip -- control only; the probe and every adaptive seed MUST pass
# their real remote path, no empty-arg special case).
train_link_guarded() {
  local label="$1" seed="$2" iters="$3" max_wait="$4" summary="$5"
  local extra_adapt="$6" flag_wanted="$7" jsonl_clear="$8"
  local before_f after_f new_dirs n_new rcount local_env remote_env
  local extra_args flag_line jsonl_local

  # JSONL FRESHNESS CONTRACT (codex round 2 finding): sync_logs/rsync is
  # non-deleting, so a local adapt_011_*.jsonl left by a PREVIOUS chain
  # attempt would otherwise survive untouched; if THIS link's log-path
  # override then silently fails to land, the post-link sync leaves that
  # stale file in place and the oracle would happily validate last
  # attempt's events instead of this run's -- a broken rule wrongly passing
  # the money gate. Fix: clear BOTH sides pre-launch (remote via
  # pod_rm_f, local via plain rm -f) so EXISTENCE after the link completes
  # can only mean THIS link wrote it -- checked below, post-launch.
  if [ -n "$jsonl_clear" ]; then
    jsonl_local="$RUNDIRS_LOCAL/$(basename "$jsonl_clear")"
    pod_rm_f "$jsonl_clear" || abort "$label: could not clear stale remote JSONL $jsonl_clear pre-launch"
    rm -f "$jsonl_local"
    log "$label: pre-launch JSONL clear ok (remote=$jsonl_clear local=$jsonl_local)"
  fi

  before_f=$(mktemp) || abort "$label: mktemp failed"
  after_f=$(mktemp) || abort "$label: mktemp failed"
  list_rundirs > "$before_f"

  extra_args="--max_iterations $iters env.episode_length_s=3.0 env.rew_scale_pos=$POS env.rew_scale_ang=$ANG env.rew_scale_actions=$ACT"
  if [ -n "$extra_adapt" ]; then
    extra_args="$extra_args $extra_adapt"
  fi

  run_link "$label" "$summary" \
    --training curee --dr 2 --seed "$seed" \
    --extra-args "$extra_args" \
    --max-wait "$max_wait" --no-stop \
    || abort "$label: supervise exited non-zero"

  JOB_ID=$(summary_field "$summary" job_id)
  [ -n "$JOB_ID" ] || abort "$label: no job_id in summary $summary"

  rcount=$(reward_count "$JOB_ID")
  if [ "$rcount" != "$iters" ]; then
    abort "$label: reward-line count $rcount != $iters (log: $LOGDIR/jobs/$JOB_ID/out.log) — override may not have landed"
  fi
  log "$label: reward-line count $rcount == $iters"

  # JSONL FRESHNESS CHECK: the pre-launch clear (above) deleted any stale
  # local file, so its EXISTENCE (and non-emptiness) now proves FRESHNESS
  # -- this link's own run produced it, never a leftover. A silently-missing
  # log-path override would otherwise only surface much later as a
  # confusing oracle failure (or, worse, an oracle PASS against stale data).
  if [ -n "$jsonl_clear" ]; then
    if [ ! -s "$jsonl_local" ]; then
      abort "$label: JSONL absent after pre-clear ($jsonl_local) — the log-path override likely did not land; data-capture violation"
    fi
    log "$label: JSONL freshness check ok ($jsonl_local exists and is non-empty)"
  fi

  list_rundirs > "$after_f"
  new_dirs=$(comm -13 "$before_f" "$after_f")
  rm -f "$before_f" "$after_f"
  n_new=$(printf '%s' "$new_dirs" | grep -c . || true)
  if [ "$n_new" != "1" ]; then
    abort "$label: expected exactly ONE new synced run dir, found $n_new [$new_dirs] — mapping integrity lost"
  fi
  NEW_DIR="$new_dirs"

  # RUNTIME WEIGHT GUARD — pod-side dump FIRST, then the synced local copy.
  # Never source-grep (see header): warpauv_env.py never shows runtime
  # overrides regardless of what actually trained.
  remote_env="/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$NEW_DIR/params/env.yaml"
  local_env="$RUNDIRS_LOCAL/$NEW_DIR/params/env.yaml"
  weight_lines_check remote "$remote_env" "$POS" "$ANG" "$ACT"
  case "$?" in
    0) log "$label: POD weight guard ok (pos=$POS ang=$ANG act=$ACT)" ;;
    2) abort "$label: WEIGHT GUARD FAKE-NULL (pod-side $remote_env) — the reward-weight overrides were silently ignored (fake null); run_dir=$NEW_DIR" ;;
    *) abort "$label: WEIGHT GUARD MISMATCH (pod-side $remote_env) — expected pos=$POS ang=$ANG act=$ACT, not found char-exact; run_dir=$NEW_DIR" ;;
  esac
  weight_lines_check local "$local_env" "$POS" "$ANG" "$ACT"
  case "$?" in
    0) log "$label: LOCAL weight guard ok (pos=$POS ang=$ANG act=$ACT)" ;;
    2) abort "$label: WEIGHT GUARD FAKE-NULL (local $local_env) — the reward-weight overrides were silently ignored (fake null); run_dir=$NEW_DIR" ;;
    *) abort "$label: WEIGHT GUARD MISMATCH (local $local_env) — expected pos=$POS ang=$ANG act=$ACT, not found char-exact; run_dir=$NEW_DIR" ;;
  esac

  # FLAG GUARD — same two-sided discipline, one char-exact line.
  flag_line="adapt_rew_weights: $flag_wanted"
  flag_line_check remote "$remote_env" "$flag_line" \
    || abort "$label: FLAG GUARD MISMATCH (pod-side $remote_env) — expected char-exact '$flag_line'; run_dir=$NEW_DIR"
  log "$label: POD flag guard ok ($flag_line)"
  flag_line_check local "$local_env" "$flag_line" \
    || abort "$label: FLAG GUARD MISMATCH (local $local_env) — expected char-exact '$flag_line'; run_dir=$NEW_DIR"
  log "$label: LOCAL flag guard ok ($flag_line)"

  if [ "$iters" = "2500" ] && [ ! -f "$RUNDIRS_LOCAL/$NEW_DIR/model_2499.pt" ]; then
    abort "$label: model_2499.pt missing in $NEW_DIR"   # (probe at 300 iters has no model_2499 -- skipped)
  fi
  grep -q '^episode_length_s: 3.0' "$local_env" \
    || abort "$label: episode_length_s: 3.0 not found in $NEW_DIR/params/env.yaml"

  if pod_dr_check "$DR2_RADIUS" "$DR2_VRANGE"; then
    log "$label: DR GUARD ok (radius=$DR2_RADIUS)"
  else
    abort "$label: pod-side DR guard failed for DR_2 (radius=$DR2_RADIUS)"
  fi
}

# Run the oracle gate against a synced local JSONL, appending its
# calibration to the TSV. Aborts on a non-zero oracle exit (hard money
# gate); WARN lines are logged, never fatal. $1=label $2=local_jsonl_path
# $3=cadence $4=expect_events $5=allow_inert(0|1) $6=calibration_tag (used in
# the "# CALIBRATION(<tag>): ..." comment).
oracle_gate() {
  local label="$1" jsonl_path="$2" cadence="$3" expect="$4" allow_inert="$5" tag="$6"
  local oracle_out oracle_rc summary_line summary_json human_line ln

  # Defense in depth: train_link_guarded's own freshness check (pre-clear +
  # post-launch existence/non-empty) already covers this for every
  # oracle-gated link; this is a redundant second check, not the primary
  # freshness guarantee.
  [ -s "$jsonl_path" ] || abort "$label: local JSONL missing or empty ($jsonl_path) — data-capture violation"

  if [ "$allow_inert" = "1" ]; then
    oracle_out=$("$ROOT_PY" "$ORACLE" validate-jsonl "$jsonl_path" --cadence "$cadence" --expect-events "$expect" --allow-inert)
  else
    oracle_out=$("$ROOT_PY" "$ORACLE" validate-jsonl "$jsonl_path" --cadence "$cadence" --expect-events "$expect")
  fi
  oracle_rc=$?

  while IFS= read -r ln; do
    case "$ln" in
      WARN:*) log "$label: ORACLE $ln" ;;
    esac
  done <<<"$oracle_out"

  if [ "$oracle_rc" -ne 0 ]; then
    log "$label: ORACLE GATE FAILED (exit=$oracle_rc):"
    printf '%s\n' "$oracle_out" | grep '^FAIL:' | while IFS= read -r ln; do log "$label: $ln"; done
    abort "$label: oracle validate-jsonl exited $oracle_rc — a corrupted/drifted adaptation rule must not spend further money"
  fi
  log "$label: ORACLE GATE PASS ($jsonl_path, cadence=$cadence, expect-events=$expect)"

  summary_line=$(printf '%s\n' "$oracle_out" | grep '^SUMMARY_JSON: ')
  summary_json="${summary_line#SUMMARY_JSON: }"
  human_line=$(oracle_summary_human_line "$summary_json")
  {
    echo "# CALIBRATION($tag): $summary_line"
    echo "# CALIBRATION($tag) human: $human_line"
  } >> "$MAPFILE_DEFAULT"
}

# ------------------------------------------------------------------- phases

phase_train() {
  log "chain start (train): campaign 011-CUREE_Adaptive-weights, stamp=$STAMP"
  mkdir -p "$LOGDIR"
  [ -x "$SUPERVISE" ] || abort "supervise.sh not found/executable at $SUPERVISE"

  # ---- pre-flight reconcile: bring the local mirror in sync with the pod so
  # per-link diffs can never pick up a stray pre-existing pod-side dir.
  sync_logs_now || abort "pre-flight sync_logs failed"
  log "pre-flight sync done"

  # ---- MOUNT-HEALTH GATE (before ANY spend beyond the already-running pod)
  local mh_out mh_rc
  mh_out=$(mount_health_check)
  mh_rc=$?
  log "mount-health gate: $mh_out"
  [ "$mh_rc" -eq 0 ] || abort "mount-health gate FAILED (degraded MooseFS backend — stop and retry later): $mh_out"

  # ---- PINS
  local sha_out sha_rc markers_out markers_rc
  sha_out=$(play_rollout_sha_check)
  sha_rc=$?
  log "play_rollout_eval pin: $sha_out"
  [ "$sha_rc" -eq 0 ] || abort "play_rollout_eval.py sha256 pin mismatch: $sha_out (want $PLAY_ROLLOUT_EVAL_SHA)"

  markers_out=$(markers_gate_check)
  markers_rc=$?
  log "markers gate: $markers_out"
  [ "$markers_rc" -eq 0 ] || abort "markers gate FAILED — checkout is BlueROV2-patched (bluerov2_patch_applied present); curee training refused"

  # ---- ADAPTIVE-PATCH LANDING (Mac-side build -> pod-side landing gate,
  # BEFORE any launch_training call ever rewrites the DR lines)
  local tmpf build_out build_rc pin_info pin_sha pin_markers landing_out landing_rc
  tmpf=$(mktemp) || abort "landing: mktemp failed"
  build_out=$("$ROOT_PY" "$BUILDER" --out "$tmpf" 2>&1)
  build_rc=$?
  [ "$build_rc" -eq 0 ] || abort "landing: apply_adaptive_patch.py build FAILED (rc=$build_rc): $build_out"

  pin_info=$("$ROOT_PY" - <<PYEOF
import sys
sys.path.insert(0, "$REPO/src")
import apply_adaptive_patch as m
print(m.ADAPTIVE_PATCHED_SHA256)
print(m.MARKER_COUNT)
PYEOF
)
  pin_sha=$(printf '%s\n' "$pin_info" | sed -n '1p')
  pin_markers=$(printf '%s\n' "$pin_info" | sed -n '2p')
  [ -n "$pin_sha" ] && [ -n "$pin_markers" ] || abort "landing: could not read module pins from apply_adaptive_patch.py"

  if [ "$build_out" != "$pin_sha" ]; then
    rm -f "$tmpf"
    abort "landing: built sha $build_out != module pin $pin_sha — apply_adaptive_patch.py drifted from its own pin"
  fi
  log "landing: Mac-side build OK, sha=$build_out matches module pin (markers pin=$pin_markers)"

  landing_out=$(pod_land_adaptive_patch "$tmpf" "$pin_sha" "$pin_markers")
  landing_rc=$?
  log "landing: $landing_out"
  rm -f "$tmpf"
  [ "$landing_rc" -eq 0 ] || abort "landing: pod-side landing gate FAILED"
  log "LANDING_OK"

  # ---- PRE-FLIGHT JSONL SWEEP (codex round 2 finding): clear every
  # adapt_011_*.jsonl on BOTH sides before anything trains, so the campaign
  # starts from zero and the control link's NO-JSONL guard can never be
  # poisoned by a stale file from an earlier chain attempt. Per-link
  # clearing (inside train_link_guarded) remains as belt-and-suspenders
  # for a mid-chain rerun that starts after this sweep already ran.
  local sweep_out sweep_rc
  sweep_out=$(pod_sweep_jsonl)
  sweep_rc=$?
  log "pre-flight JSONL sweep (pod-side): $sweep_out"
  [ "$sweep_rc" -eq 0 ] || abort "pre-flight JSONL sweep FAILED pod-side: $sweep_out"
  if [ -d "$RUNDIRS_LOCAL" ]; then
    rm -f "$RUNDIRS_LOCAL"/adapt_011_*.jsonl
  fi
  log "pre-flight JSONL sweep (local): $RUNDIRS_LOCAL/adapt_011_*.jsonl cleared"

  # ---- spend baseline
  local baseline
  baseline=$(spend_now)
  [ -n "$baseline" ] || abort "could not read spend baseline"
  log "spend baseline=\$$baseline guard=+\$$TRAIN_GUARD_USD (train phase)"

  # ---- mapping file (partial file stays valid row-by-row on a mid-chain abort)
  {
    echo "# chain-011-rundirs.tsv — arm/seed -> run-dir mapping for 011-CUREE_Adaptive-weights (stamp $STAMP)"
    echo "# INVARIANT: exactly ONE new run dir appeared per training link (guarded, weight-guard"
    echo "# + flag-guard char-exact against pos=$POS/ang=$ANG/act=$ACT and adapt_rew_weights"
    echo "# true/false); each fresh row was bound to its arm/seed AT LINK COMPLETION from a"
    echo "# per-link before/after diff of the local synced dir list. 4 DATA ROWS expected:"
    echo "# CONTROL seed 2, ADAPTIVE seeds 1-3. The PROBE run dir (300 iters, seed 1, cadence 25,"
    echo "# adapt ON) was identified by its own diff and EXCLUDED (quarantined pod-side in"
    echo "# _probe_quarantine/) — it never appears as a data row; its oracle CALIBRATION summary"
    echo "# is recorded below as comment lines instead, alongside every adaptive seed's."
    echo "# A PARTIAL file after a mid-chain abort remains VALID row-by-row for the rows it contains."
    echo "# columns: arm<TAB>seed<TAB>weights<TAB>run_dir<TAB>job_id<TAB>mean_reward_lines<TAB>checkpoint<TAB>jsonl"
  } > "$MAPFILE_DEFAULT"

  # ---- 300-iter probe at the champion triple, adapt ON, cadence 25
  NEW_DIR="" JOB_ID=""
  train_link_guarded "probe (300 iters, cadence 25)" 1 300 1500 \
    "$LOGDIR/supervise-011-probe-$STAMP.json" \
    "env.adapt_rew_weights=true env.adapt_cadence_iters=25 env.adapt_log_path=$PROBE_JSONL_REMOTE" \
    true "$PROBE_JSONL_REMOTE"
  log "probe run dir: $NEW_DIR"

  oracle_gate "probe" "$PROBE_JSONL_LOCAL" 25 12 0 "probe"

  log "probe run dir: $NEW_DIR — quarantining pod-side"
  if pod_quarantine_probe "$NEW_DIR"; then
    log "probe dir quarantined to _probe_quarantine/ (its local copy is EXCLUDED from the mapping by construction)"
  else
    abort "probe quarantine failed for $NEW_DIR"
  fi
  check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after probe"

  # ---- CONTROL link: seed 2, adapt OFF (NO adapt_* overrides at all)
  NEW_DIR="" JOB_ID=""
  train_link_guarded "CONTROL (seed 2, adapt OFF)" 2 2500 3600 \
    "$LOGDIR/supervise-011-train-control-s2-$STAMP.json" \
    "" false ""

  # NO-JSONL GUARD: control must write NOTHING — pod-side glob count stays
  # at exactly 1 (the probe's own file).
  local jc
  jc=$(pod_jsonl_count) || abort "CONTROL: could not count pod-side adapt_011_*.jsonl files"
  if [ "$jc" != "1" ]; then
    abort "CONTROL: expected exactly 1 adapt_011_*.jsonl pod-side (the probe's only), found $jc — control link wrote unexpected JSONL data"
  fi
  log "CONTROL: NO-JSONL guard ok (count=$jc, probe's only)"

  # MEAN-REWARD DETERMINISM RECORD (train-phase proxy; hard gate at scoring)
  local last_reward
  last_reward=$(last_mean_reward "$JOB_ID")
  [ -n "$last_reward" ] || abort "CONTROL: could not extract last Mean reward from $LOGDIR/jobs/$JOB_ID/out.log"
  if float_close "$last_reward" "$CONTROL_TARGET_REWARD" "$CONTROL_REWARD_BAND"; then
    log "CONTROL: mean-reward determinism record OK (last=$last_reward, target=$CONTROL_TARGET_REWARD, band=$CONTROL_REWARD_BAND)"
  else
    abort "CONTROL: mean-reward determinism OUT OF BAND (last=$last_reward, target=$CONTROL_TARGET_REWARD, band=$CONTROL_REWARD_BAND)"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "CONTROL" 2 "$WEIGHTS" "$NEW_DIR" "$JOB_ID" 2500 "model_2499.pt" "-" \
    >> "$MAPFILE_DEFAULT"
  log "CONTROL: mapped -> $NEW_DIR (row appended to $MAPFILE_DEFAULT)"
  check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after CONTROL"

  # ---- 3 ADAPTIVE links: seeds 1, 2, 3 — adapt ON, cadence 100
  local seed jsonl_remote jsonl_local label
  for seed in 1 2 3; do
    jsonl_remote="$REMOTE_JSONL_DIR/adapt_011_s${seed}.jsonl"
    jsonl_local="$RUNDIRS_LOCAL/adapt_011_s${seed}.jsonl"
    label="ADAPTIVE seed $seed (2500 iters, cadence 100)"
    NEW_DIR="" JOB_ID=""
    train_link_guarded "$label" "$seed" 2500 3600 \
      "$LOGDIR/supervise-011-train-adaptive-s$seed-$STAMP.json" \
      "env.adapt_rew_weights=true env.adapt_cadence_iters=100 env.adapt_log_path=$jsonl_remote" \
      true "$jsonl_remote"

    oracle_gate "$label" "$jsonl_local" 100 "23:25" 1 "s${seed}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "ADAPTIVE" "$seed" "$WEIGHTS" "$NEW_DIR" "$JOB_ID" 2500 "model_2499.pt" "adapt_011_s${seed}.jsonl" \
      >> "$MAPFILE_DEFAULT"
    log "$label: mapped -> $NEW_DIR (row appended to $MAPFILE_DEFAULT)"
    check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after $label"
  done

  log "train phase COMPLETE — probe + CONTROL + 3/3 ADAPTIVE links green; mapping: $MAPFILE_DEFAULT"
  log "POD LEFT UP by design. == STOP POINT == parent session: review the mapping"
  log "(4 data rows + calibration comment blocks, control reward proxy already enforced),"
  log "then fire promptly:"
  log "  bash $DIR/chain-011-CUREE_Adaptive-weights.sh rollouts"
  log "Idle burn while you review is bounded (~\$0.69/hr) and covered by deadman + watchdog."
  exit 0
}

phase_rollouts() {
  local mapfile="${1:-$MAPFILE_DEFAULT}"
  log "chain start (rollouts): campaign 011-CUREE_Adaptive-weights, stamp=$STAMP, mapping=$mapfile"

  # Read-only pre-check: NEVER brings the pod up itself (human-initiated only).
  local st
  st=$(pod_status_now)
  if [ "$st" != "running" ]; then
    log "pod is not running (status=$st) — run ensure_pod in the parent session, then re-fire:"
    log "  bash $DIR/chain-011-CUREE_Adaptive-weights.sh rollouts"
    exit 3
  fi

  [ -f "$mapfile" ] || { log "mapping file not found: $mapfile"; exit 3; }

  # Cheap re-verify of the play_rollout_eval pin before any spend.
  local sha_out sha_rc
  sha_out=$(play_rollout_sha_check)
  sha_rc=$?
  log "play_rollout_eval pin (rollouts pre-check): $sha_out"
  if [ "$sha_rc" -ne 0 ]; then
    log "play_rollout_eval.py sha256 pin mismatch: $sha_out (want $PLAY_ROLLOUT_EVAL_SHA) — refuse"
    exit 3
  fi

  # Parse + verify the mapping BEFORE any spend (bash-3.2-safe indexed arrays).
  local -a R_ARM=() R_SEED=() R_WEIGHTS=() R_DIR=() R_CKPT=()
  local arm seed weights_col dir jid rlines ckpt jsonl_col
  # jid/rlines/jsonl_col are read positionally (columns 5-6-8, informational
  # provenance only) and never referenced again — validation only needs
  # arm/seed/weights/dir/ckpt.
  # shellcheck disable=SC2034
  while IFS=$'\t' read -r arm seed weights_col dir jid rlines ckpt jsonl_col; do
    case "$arm" in \#*|"") continue ;; esac
    R_ARM+=("$arm"); R_SEED+=("$seed"); R_WEIGHTS+=("$weights_col")
    R_DIR+=("$dir"); R_CKPT+=("$ckpt")
  done < "$mapfile"

  local n=${#R_ARM[@]}
  if [ "$n" -ne 4 ]; then
    log "mapping has $n data rows, expected 4 — refuse (salvage/resume is a parent-session call)"
    exit 3
  fi

  local i combo combos_seen=""
  for i in 0 1 2 3; do
    combo="${R_ARM[$i]}_${R_SEED[$i]}"
    case " $combos_seen " in
      *" $combo "*) log "mapping row $((i + 1)): duplicate arm/seed combo $combo — refuse"; exit 3 ;;
    esac
    combos_seen="$combos_seen $combo"
  done
  local want
  for want in CONTROL_2 ADAPTIVE_1 ADAPTIVE_2 ADAPTIVE_3; do
    case " $combos_seen " in
      *" $want "*) : ;;
      *) log "mapping missing required row: $want — refuse"; exit 3 ;;
    esac
  done

  for i in 0 1 2 3; do
    arm="${R_ARM[$i]}"
    case "$arm" in
      CONTROL|ADAPTIVE) : ;;
      *) log "mapping row $((i + 1)): unknown arm '$arm' — refuse"; exit 3 ;;
    esac
    # Run-dir SHAPE guard (refuse-don't-sanitize): run_dir is later
    # interpolated into a pod-side shell command, so anything outside the
    # exact run-dir timestamp shape is refused OUTRIGHT pre-spend — never
    # quoted-through. A bash case glob is implicitly anchored to the whole
    # word (bash-3.2 safe; no =~).
    case "${R_DIR[$i]}" in
      20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]) : ;;
      *) log "mapping row $((i + 1)): run_dir '${R_DIR[$i]}' does not match the run-dir timestamp shape (YYYY-MM-DD_hh-mm-ss) — refuse"; exit 3 ;;
    esac
    if [ "${R_WEIGHTS[$i]}" != "$WEIGHTS" ]; then
      log "mapping row $((i + 1)) ($arm seed ${R_SEED[$i]}): weights '${R_WEIGHTS[$i]}' != expected '$WEIGHTS' — refuse"
      exit 3
    fi
    if [ "${R_CKPT[$i]}" != "model_2499.pt" ]; then
      log "mapping row $((i + 1)): arm $arm checkpoint must be model_2499.pt, got '${R_CKPT[$i]}' — refuse"
      exit 3
    fi
    if [ ! -f "$RUNDIRS_LOCAL/${R_DIR[$i]}/${R_CKPT[$i]}" ]; then
      log "mapping row $((i + 1)): local checkpoint missing: $RUNDIRS_LOCAL/${R_DIR[$i]}/${R_CKPT[$i]} — refuse"
      exit 3
    fi
  done
  log "mapping verified: 4 rows, CONTROL_2 x ADAPTIVE_1/2/3, weights + checkpoints all match"

  local baseline
  baseline=$(spend_now)
  [ -n "$baseline" ] || { log "could not read spend baseline — refusing pre-spend"; exit 3; }
  log "spend baseline=\$$baseline guard=+\$$ROLLOUT_GUARD_USD (rollouts phase)"

  local arm_lc csv label stop_flag summary cmd
  for i in 0 1 2 3; do
    arm="${R_ARM[$i]}"; seed="${R_SEED[$i]}"; dir="${R_DIR[$i]}"; ckpt="${R_CKPT[$i]}"
    case "$arm" in
      CONTROL) arm_lc="control" ;;
      ADAPTIVE) arm_lc="adaptive" ;;
      *) abort "rollout row $((i + 1)): unknown arm $arm" ;;
    esac
    csv="rollout_011_${arm}_seed${seed}_model_2499.csv"
    label="rollout $arm seed $seed ($dir @ $ckpt)"
    summary="$LOGDIR/supervise-011-rollout-$arm_lc-s$seed-$STAMP.json"
    cmd="cd /workspace/IsaacLab && test -f logs/rsl_rl/warpauv_direct/$dir/$ckpt && ./isaaclab.sh -p ../isaac-auv-env/custom_workflows/play_rollout_eval.py --task Isaac-WarpAUV-Direct-v1 --num_envs 1 --headless --load_run $dir --checkpoint $ckpt --out_csv /workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$csv"
    if [ "$i" -lt 3 ]; then
      stop_flag="--no-stop"
    else
      stop_flag=""   # link 4 OWNS the stop (supervise syncs first, then stops)
    fi
    # shellcheck disable=SC2086
    run_link "$label" "$summary" \
      --job-name "011-curee-${arm_lc}-rollout-s${seed}" \
      --sync-subdir rsl_rl/warpauv_direct --max-wait 2700 $stop_flag \
      --command "$cmd" \
      || abort "$label: supervise exited non-zero"
    if [ -s "$RUNDIRS_LOCAL/$csv" ]; then
      log "$label: CSV synced locally ($(wc -l < "$RUNDIRS_LOCAL/$csv" | tr -d ' ') lines)"
    else
      abort "$label: $csv missing/empty locally after sync"
    fi
    if [ "$i" -lt 3 ]; then
      check_spend "$baseline" "$ROLLOUT_GUARD_USD" || abort "spend guard breach after $label"
    fi
  done

  # Link 4's supervise owns the stop; verify it actually landed.
  st=$(pod_status_now)
  if [ "$st" = "running" ]; then
    log "WARNING: pod still running after final link — forcing stop now"
    if stop_pod_force; then log "stop_pod done"; else
      log "WARNING: stop_pod FAILED after 3 attempts — check pod_status NOW"
      exit 1
    fi
  else
    log "pod status after final link: $st (stop owned by link 4's supervise)"
  fi

  log "rollouts phase COMPLETE — 4/4 CSVs local; pod stopped. Offline scoring is next."
  exit 0
}

# ------------------------------------------------------------------ dispatch

[ $# -ge 1 ] || usage
bootstrap_venv
case "$1" in
  train)    phase_train ;;
  rollouts) phase_rollouts "${2:-}" ;;
  *)        usage ;;
esac
