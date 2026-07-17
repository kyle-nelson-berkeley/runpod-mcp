#!/usr/bin/env bash
# =============================================================================
# chain-010-CUREE_Weight-screen.sh — the ONE chain for campaign
# 010-CUREE_Weight-screen ("010 · weight screen"): a reward-weight STATIC
# screen. 4 arms (BASELINE + A/B/C), 3 trained (A/B/C), at the screening
# dose 1500 iters x 3.0 s x DR_2 x seeds {1,2,3}. pos+ang+act sums to 0.9 for
# every arm (Kyle-fixed constants):
#
#   BASELINE  0.2/0.5/0.2   NOT trained here — reuses 009's fresh-DR_2 dirs
#   A         0.3/0.4/0.2   trained x3 seeds
#   B         0.1/0.6/0.2   trained x3 seeds
#   C         0.25/0.55/0.1 trained x3 seeds (also the probe's triple)
#
# TWO PHASES, fired by the PARENT session (the one holding Kyle's direct
# spend-envelope approval), each as ONE `run_in_background` Bash task:
#
#   bash runpod-mcp/chains/chain-010-CUREE_Weight-screen.sh train
#       pre-flight (reconcile sync + verify the 3 BASELINE dirs exist, local
#       AND pod-side, with model_1500.pt) -> 5-iter probe at arm C's triple
#       (quarantined, never in the mapping) -> 9 training links, ascending
#       (A s1/2/3 -> B s1/2/3 -> C s1/2/3), each supervised (--no-stop) with
#       per-link guards -> emits logs/pod/chain-010-rundirs.tsv (3 BASELINE
#       rows at pre-flight + 9 fresh rows at link completion) -> exits with
#       the POD UP.
#
#   == STOP POINT == the parent session reviews chain-010-rundirs.tsv
#       (12 rows, invariants in the header) and then fires:
#
#   bash runpod-mcp/chains/chain-010-CUREE_Weight-screen.sh rollouts [tsv]
#       read-only pod_status pre-check (stopped pod -> instructs ensure_pod,
#       NO auto bring-up: pod creation stays human-initiated, exit 3) ->
#       parse + validate the TSV BEFORE any spend (exactly 12 rows, the full
#       BASELINE/A/B/C x seed 1-3 set, char-exact weights, right checkpoint
#       convention per row, local checkpoint file present — any violation is
#       a refusal: exit 3, no stop, no spend) -> 12 rollout links in TSV row
#       order (BASELINE s1-3 -> A -> B -> C); links 1-11 --no-stop, link 12
#       OWNS the stop.
#
# THE WEIGHT GUARD (NEW vs. 009 — the load-bearing correction): a Hydra
# override that silently fails to land (typo'd key, wrong dotpath, a stale
# checkout) reproduces the BASELINE run bit-for-bit and fabricates a clean
# "no effect" result. So every training/probe link's guard reads the run's
# params/env.yaml BOTH pod-side (immediately post-link, via exec_on_pod) AND
# the synced local copy, and asserts char-exact fixed-string lines (count==1
# each) for "rew_scale_pos: P" / "rew_scale_ang: A" / "rew_scale_actions: T"
# — note the trailing colon-space on "rew_scale_ang: " never collides with
# rew_scale_ang_vel (a different line entirely). NEVER grep the SOURCE for
# these — Hydra overrides only change the RUNTIME cfg dump; warpauv_env.py
# always shows the source defaults 0.2/0.5/0.2 regardless of what a run
# actually trained with, so a source-grep (009's DR-guard idiom) is the WRONG
# pattern here and would rubber-stamp every silent-ignore. On a mismatch the
# guard additionally tests the FAKE-NULL signature — all three of
# "rew_scale_pos: 0.2" + "rew_scale_ang: 0.5" + "rew_scale_actions: 0.2"
# present — and aborts with a DISTINCT message when it matches: the overrides
# were silently ignored (fake null). The probe fires FIRST, before any
# training-link money, using arm C's triple (0.25/0.55/0.1) precisely because
# ALL THREE keys are off-default there — one probe proves every key lands
# (a same-value key could pass by accident; C's triple cannot).
#
# CHECKPOINT OFFSET NOTE: trained links score model_1499.pt, not
# model_1500.pt — RSL-RL saves interval-50 multiples plus a FINAL checkpoint
# at max_iterations-1, so a 1500-iteration run's last checkpoint is numbered
# 1499. BASELINE rows (reused from 009, itself run at max_iterations=2500)
# score model_1500.pt, a mid-run interval checkpoint from that longer run —
# per-row in the TSV, NEVER hardcode one convention for both.
#
# DR GUARD (retained from 009 verbatim): every arm in this campaign sits at
# DR_2, so the pod-side fixed-string grep of the two ACTIVE lines of
# /workspace/isaac-auv-env/warpauv_env.py — `com_to_cob_offset_radius = 0.05 #`
# and `volume_range = [0.019747843530591773, 0.02574784353059178] #` — is
# always checked against the SAME two constants. launch_training rewrites
# these lines every launch (content-anchored), which also auto-recovers a
# checkout left parked at a different DR level by an earlier campaign; the
# probe proves this before any training-link money, same as 009.
#
# MEAN-REWARD GUARD (retained from 009): exactly one "Mean reward" line per
# iteration — equality (not >=) is load-bearing, since a >= check would wave
# through a dropped --max_iterations override (e.g. a stray 2500-line log on
# a 1500-iteration link) that an equality check catches immediately.
#
# FAILURE DOCTRINE (verbatim 009): run_link retries ONCE after 60s, ONLY when
# the supervise summary shows NO job_id (a pre-launch failure); any other
# failure -> abort() -> stop_pod(force) x3 with 20s gaps -> exit 1. Never
# terminate_pod. Rollouts phase never brings the pod up itself. A non-fatal
# spend_report read failure is logged and the chain continues (supervise's
# own caps + the deadman fuse bound the burn either way).
#
# SPEND GUARDS: TRAIN_GUARD_USD=$2.50 + ROLLOUT_GUARD_USD=$1.00 sum to the
# $3.50 slow-host ceiling of the Kyle-gated screen envelope (nominal
# predicted spend ~$2.0-2.5). check_spend runs after the probe, after each
# training link, and after rollout links 1-11 (009's placement — link 12 owns
# the stop, so no post-check follows it).
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
MAPFILE_DEFAULT="$LOGDIR/chain-010-rundirs.tsv"
WARPAUV_ENV_REMOTE="/workspace/isaac-auv-env/warpauv_env.py"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

# Spend guards (deltas from each phase's own baseline; they SUM to the
# Kyle-approved $3.50 envelope; nominal predicted total ~$2.0-2.5).
TRAIN_GUARD_USD="2.50"
ROLLOUT_GUARD_USD="1.00"

# DR_2 is the only level this campaign ever touches (curee DR_TABLES["DR_2"],
# unit-tested char-exact against runpod_mcp/training.py).
DR2_RADIUS="0.05"
DR2_VRANGE="[0.019747843530591773, 0.02574784353059178]"

# The three BASELINE dirs are 009's fresh-DR_2 seeds 1-3 (provenance:
# logs/pod/chain-009-rundirs.tsv) — never trained in this campaign.
BASELINE_DIR_1="2026-07-16_12-51-52"
BASELINE_DIR_2="2026-07-16_13-02-25"
BASELINE_DIR_3="2026-07-16_13-13-01"

# The 9 training links, ascending arm (bash-3.2-safe lock-step arrays).
ARM_NAMES=(A A A B B B C C C)
ARM_SEEDS=(1 2 3 1 2 3 1 2 3)

baseline_dir_for() {  # $1 = seed 1-3
  case "$1" in
    1) printf '%s' "$BASELINE_DIR_1" ;;
    2) printf '%s' "$BASELINE_DIR_2" ;;
    3) printf '%s' "$BASELINE_DIR_3" ;;
    *) return 1 ;;
  esac
}

# Per-arm pos/ang/act (Kyle-fixed; pos+ang+act = 0.9 for every arm).
pos_for() {
  case "$1" in
    BASELINE) printf '0.2' ;;
    A) printf '0.3' ;;
    B) printf '0.1' ;;
    C) printf '0.25' ;;
    *) return 1 ;;
  esac
}
ang_for() {
  case "$1" in
    BASELINE) printf '0.5' ;;
    A) printf '0.4' ;;
    B) printf '0.6' ;;
    C) printf '0.55' ;;
    *) return 1 ;;
  esac
}
act_for() {
  case "$1" in
    BASELINE) printf '0.2' ;;
    A) printf '0.2' ;;
    B) printf '0.2' ;;
    C) printf '0.1' ;;
    *) return 1 ;;
  esac
}
weights_for() {  # $1 = arm -> "pos/ang/act", char-exact TSV column
  local p a t
  p=$(pos_for "$1") || return 1
  a=$(ang_for "$1") || return 1
  t=$(act_for "$1") || return 1
  printf '%s/%s/%s' "$p" "$a" "$t"
}
slug_for() {  # $1 = arm -> rollout job-name slug
  case "$1" in
    BASELINE) printf 'baseline' ;;
    A) printf 'arm-a' ;;
    B) printf 'arm-b' ;;
    C) printf 'arm-c' ;;
    *) return 1 ;;
  esac
}

log() { printf '[chain-010 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

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

pod_path_exists() {  # $1 = remote path — exit 0 iff test -f succeeds pod-side
  (cd "$MCPDIR" && "$PY" - "$1" <<'PYEOF'
import shlex, sys
from runpod_mcp import tools
path = shlex.quote(sys.argv[1])
out = tools.exec_on_pod(tools.runtime(), f"test -f {path} && echo EXISTS", timeout_sec=60)
text = (out.get("stdout") or "").strip()
sys.exit(0 if out.get("exit_code") == 0 and "EXISTS" in text else 1)
PYEOF
  )
}

# Runtime weight guard, ONE side (local file or pod-side path via
# exec_on_pod). $1 = mode (local|remote) $2 = path $3=pos $4=ang $5=act.
# Exit 0 = char-exact match (count==1 each). Exit 2 = mismatch AND the
# fake-null signature (the BASELINE triple, verbatim) is present instead —
# the overrides were silently ignored. Exit 1 = mismatch, not fake-null.
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
# The fake-null signature is the literal BASELINE triple — independent of
# which arm we're checking (a silently-ignored override always reproduces
# THIS, never some other wrong value).
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

# ------------------------------------------------------------ small helpers

summary_field() {  # $1 = summary json path, $2 = field
  "$PY" -c \
    'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
    "$1" "$2" 2>/dev/null
}

float_gt() {  # exit 0 iff $1 > $2
  "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$1" "$2"
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
# $1=label $2=pos $3=ang $4=act $5=seed $6=iters $7=max_wait $8=summary_path
train_link_guarded() {
  local label="$1" pos="$2" ang="$3" act="$4" seed="$5" iters="$6" max_wait="$7" summary="$8"
  local before_f after_f new_dirs n_new rcount local_env remote_env

  before_f=$(mktemp) || abort "$label: mktemp failed"
  after_f=$(mktemp) || abort "$label: mktemp failed"
  list_rundirs > "$before_f"

  run_link "$label" "$summary" \
    --training curee --dr 2 --seed "$seed" \
    --extra-args "--max_iterations $iters env.episode_length_s=3.0 env.rew_scale_pos=$pos env.rew_scale_ang=$ang env.rew_scale_actions=$act" \
    --max-wait "$max_wait" --no-stop \
    || abort "$label: supervise exited non-zero"

  JOB_ID=$(summary_field "$summary" job_id)
  [ -n "$JOB_ID" ] || abort "$label: no job_id in summary $summary"

  rcount=$(reward_count "$JOB_ID")
  if [ "$rcount" != "$iters" ]; then
    abort "$label: reward-line count $rcount != $iters (log: $LOGDIR/jobs/$JOB_ID/out.log) — override may not have landed"
  fi
  log "$label: reward-line count $rcount == $iters"

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
  weight_lines_check remote "$remote_env" "$pos" "$ang" "$act"
  case "$?" in
    0) log "$label: POD weight guard ok (pos=$pos ang=$ang act=$act)" ;;
    2) abort "$label: WEIGHT GUARD FAKE-NULL (pod-side $remote_env) — the reward-weight overrides (pos=$pos ang=$ang act=$act) were silently ignored (fake null); run_dir=$NEW_DIR" ;;
    *) abort "$label: WEIGHT GUARD MISMATCH (pod-side $remote_env) — expected pos=$pos ang=$ang act=$act, not found char-exact; run_dir=$NEW_DIR" ;;
  esac
  weight_lines_check local "$local_env" "$pos" "$ang" "$act"
  case "$?" in
    0) log "$label: LOCAL weight guard ok (pos=$pos ang=$ang act=$act)" ;;
    2) abort "$label: WEIGHT GUARD FAKE-NULL (local $local_env) — the reward-weight overrides (pos=$pos ang=$ang act=$act) were silently ignored (fake null); run_dir=$NEW_DIR" ;;
    *) abort "$label: WEIGHT GUARD MISMATCH (local $local_env) — expected pos=$pos ang=$ang act=$act, not found char-exact; run_dir=$NEW_DIR" ;;
  esac

  if [ "$iters" = "1500" ] && [ ! -f "$RUNDIRS_LOCAL/$NEW_DIR/model_1499.pt" ]; then
    abort "$label: model_1499.pt missing in $NEW_DIR"   # (probe at 5 iters has no model_1499)
  fi
  grep -q '^episode_length_s: 3.0' "$local_env" \
    || abort "$label: episode_length_s: 3.0 not found in $NEW_DIR/params/env.yaml"

  if pod_dr_check "$DR2_RADIUS" "$DR2_VRANGE"; then
    log "$label: DR GUARD ok (radius=$DR2_RADIUS)"
  else
    abort "$label: pod-side DR guard failed for DR_2 (radius=$DR2_RADIUS)"
  fi
}

# Local + pod-side pre-flight for the 3 reused BASELINE dirs. ANY failure
# aborts BEFORE the probe / any training link (no supervise call yet).
baseline_preflight() {
  local seed dir local_ckpt local_env
  for seed in 1 2 3; do
    dir=$(baseline_dir_for "$seed") || abort "baseline pre-flight: unknown seed $seed"
    local_ckpt="$RUNDIRS_LOCAL/$dir/model_1500.pt"
    local_env="$RUNDIRS_LOCAL/$dir/params/env.yaml"
    [ -f "$local_ckpt" ] || abort "baseline pre-flight seed $seed: local checkpoint missing: $local_ckpt"
    [ -f "$local_env" ] || abort "baseline pre-flight seed $seed: local env.yaml missing: $local_env"
    if weight_lines_check local "$local_env" "$(pos_for BASELINE)" "$(ang_for BASELINE)" "$(act_for BASELINE)"; then
      log "baseline pre-flight seed $seed: local OK ($dir)"
    else
      abort "baseline pre-flight seed $seed: local env.yaml $local_env does not show the BASELINE triple $(weights_for BASELINE) char-exact"
    fi

    if pod_path_exists "/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$dir/model_1500.pt"; then
      log "baseline pre-flight seed $seed: pod-side checkpoint OK ($dir)"
    else
      abort "baseline pre-flight seed $seed: pod-side checkpoint missing for $dir"
    fi
  done
}

# ------------------------------------------------------------------- phases

phase_train() {
  log "chain start (train): campaign 010-CUREE_Weight-screen, stamp=$STAMP"
  mkdir -p "$LOGDIR"
  [ -x "$SUPERVISE" ] || abort "supervise.sh not found/executable at $SUPERVISE"

  # Pre-baseline reconcile: bring the local mirror in sync with the pod so
  # per-link diffs can never pick up a stray pre-existing pod-side dir.
  sync_logs_now || abort "pre-baseline sync_logs failed"
  log "pre-baseline sync done"

  local baseline
  baseline=$(spend_now)
  [ -n "$baseline" ] || abort "could not read spend baseline"
  log "spend baseline=\$$baseline guard=+\$$TRAIN_GUARD_USD (train phase)"

  # ---- mapping file (partial file stays valid row-by-row on a mid-chain abort)
  {
    echo "# chain-010-rundirs.tsv — arm -> run-dir mapping for 010-CUREE_Weight-screen (stamp $STAMP)"
    echo "# INVARIANT: exactly ONE new run dir appeared per training link (guarded, weight-guard"
    echo "# char-exact against the arm's pos/ang/act triple); each fresh row was bound to its arm"
    echo "# AT LINK COMPLETION from a per-link before/after diff of the local synced dir list."
    echo "# BASELINE rows are NOT trained here — they point at the three 009 fresh-DR_2 run dirs"
    echo "# (provenance: logs/pod/chain-009-rundirs.tsv, DR_2 seeds 1-3), reused as-is at"
    echo "# model_1500.pt. The probe run dir (arm C's triple, 5 iters) was identified by its own"
    echo "# diff and EXCLUDED (quarantined pod-side in _probe_quarantine/) — it never appears here."
    echo "# A PARTIAL file after a mid-chain abort remains VALID row-by-row for the rows it contains."
    echo "# columns: arm<TAB>seed<TAB>weights<TAB>run_dir<TAB>job_id<TAB>mean_reward_lines<TAB>checkpoint"
  } > "$MAPFILE_DEFAULT"

  # ---- BASELINE pre-flight (BEFORE the probe / any training money)
  baseline_preflight
  local seed dir
  for seed in 1 2 3; do
    dir=$(baseline_dir_for "$seed")
    printf 'BASELINE\t%s\t%s\t%s\t-\t-\tmodel_1500.pt\n' "$seed" "$(weights_for BASELINE)" "$dir" \
      >> "$MAPFILE_DEFAULT"
  done
  log "baseline pre-flight PASS — 3 BASELINE rows written to $MAPFILE_DEFAULT"

  # ---- 5-iter probe at arm C's triple (~\$0.10): ALL THREE keys off-default,
  # so this one probe proves every reward-weight key actually lands.
  NEW_DIR="" JOB_ID=""
  train_link_guarded "probe arm-C triple (5 iters)" \
    "$(pos_for C)" "$(ang_for C)" "$(act_for C)" 1 5 1500 \
    "$LOGDIR/supervise-010-probe-$STAMP.json"
  log "probe run dir: $NEW_DIR — quarantining pod-side"
  if pod_quarantine_probe "$NEW_DIR"; then
    log "probe dir quarantined to _probe_quarantine/ (its local copy is EXCLUDED from the mapping by construction)"
  else
    abort "probe quarantine failed for $NEW_DIR"
  fi
  check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after probe"

  # ---- 9 training links: A s1-3 -> B s1-3 -> C s1-3
  local i arm pos ang act label
  for i in 0 1 2 3 4 5 6 7 8; do
    arm=${ARM_NAMES[$i]}
    seed=${ARM_SEEDS[$i]}
    pos=$(pos_for "$arm") || abort "unknown arm $arm"
    ang=$(ang_for "$arm") || abort "unknown arm $arm"
    act=$(act_for "$arm") || abort "unknown arm $arm"
    label="train arm $arm seed $seed (1500 iters, ep3s)"
    NEW_DIR="" JOB_ID=""
    train_link_guarded "$label" "$pos" "$ang" "$act" "$seed" 1500 3600 \
      "$LOGDIR/supervise-010-train-arm$arm-s$seed-$STAMP.json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$arm" "$seed" "$(weights_for "$arm")" "$NEW_DIR" "$JOB_ID" 1500 "model_1499.pt" \
      >> "$MAPFILE_DEFAULT"
    log "$label: mapped -> $NEW_DIR (row appended to $MAPFILE_DEFAULT)"
    check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after $label"
  done

  log "train phase COMPLETE — 9/9 links green; mapping: $MAPFILE_DEFAULT"
  log "POD LEFT UP by design. == STOP POINT == parent session: review the mapping"
  log "(3 BASELINE + 9 fresh rows, invariants in header), then fire promptly:"
  log "  bash $DIR/chain-010-CUREE_Weight-screen.sh rollouts"
  log "Idle burn while you review is bounded (~\$0.69/hr) and covered by deadman + watchdog."
  exit 0
}

phase_rollouts() {
  local mapfile="${1:-$MAPFILE_DEFAULT}"
  log "chain start (rollouts): campaign 010-CUREE_Weight-screen, stamp=$STAMP, mapping=$mapfile"

  # Read-only pre-check: NEVER brings the pod up itself (human-initiated only).
  local st
  st=$(pod_status_now)
  if [ "$st" != "running" ]; then
    log "pod is not running (status=$st) — run ensure_pod in the parent session, then re-fire:"
    log "  bash $DIR/chain-010-CUREE_Weight-screen.sh rollouts"
    exit 3
  fi

  [ -f "$mapfile" ] || { log "mapping file not found: $mapfile"; exit 3; }

  # Parse + verify the mapping BEFORE any spend (bash-3.2-safe indexed arrays).
  local -a R_ARM=() R_SEED=() R_WEIGHTS=() R_DIR=() R_CKPT=()
  local arm seed weights_col dir jid rlines ckpt
  # jid/rlines are read positionally (columns 5-6, informational provenance
  # only) and never referenced again — validation only needs arm/seed/
  # weights/dir/ckpt.
  # shellcheck disable=SC2034
  while IFS=$'\t' read -r arm seed weights_col dir jid rlines ckpt; do
    case "$arm" in \#*|"") continue ;; esac
    R_ARM+=("$arm"); R_SEED+=("$seed"); R_WEIGHTS+=("$weights_col")
    R_DIR+=("$dir"); R_CKPT+=("$ckpt")
  done < "$mapfile"

  local n=${#R_ARM[@]}
  if [ "$n" -ne 12 ]; then
    log "mapping has $n data rows, expected 12 — refuse (salvage/resume is a parent-session call)"
    exit 3
  fi

  local i combo combos_seen=""
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    combo="${R_ARM[$i]}_${R_SEED[$i]}"
    case " $combos_seen " in
      *" $combo "*) log "mapping row $((i + 1)): duplicate arm/seed combo $combo — refuse"; exit 3 ;;
    esac
    combos_seen="$combos_seen $combo"
  done
  local want_arm want_seed
  for want_arm in BASELINE A B C; do
    for want_seed in 1 2 3; do
      case " $combos_seen " in
        *" ${want_arm}_${want_seed} "*) : ;;
        *) log "mapping missing required row: arm=$want_arm seed=$want_seed — refuse"; exit 3 ;;
      esac
    done
  done

  local want_weights
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    arm="${R_ARM[$i]}"
    want_weights=$(weights_for "$arm") || { log "mapping row $((i + 1)): unknown arm '$arm' — refuse"; exit 3; }
    if [ "${R_WEIGHTS[$i]}" != "$want_weights" ]; then
      log "mapping row $((i + 1)) ($arm seed ${R_SEED[$i]}): weights '${R_WEIGHTS[$i]}' != expected '$want_weights' — refuse"
      exit 3
    fi
    if [ "$arm" = "BASELINE" ]; then
      case " $BASELINE_DIR_1 $BASELINE_DIR_2 $BASELINE_DIR_3 " in
        *" ${R_DIR[$i]} "*) : ;;
        *) log "mapping row $((i + 1)): BASELINE run_dir '${R_DIR[$i]}' is not one of the known 009 dirs — refuse"; exit 3 ;;
      esac
      if [ "${R_CKPT[$i]}" != "model_1500.pt" ]; then
        log "mapping row $((i + 1)): BASELINE checkpoint must be model_1500.pt, got '${R_CKPT[$i]}' — refuse"
        exit 3
      fi
    else
      if [ "${R_CKPT[$i]}" != "model_1499.pt" ]; then
        log "mapping row $((i + 1)): arm $arm checkpoint must be model_1499.pt, got '${R_CKPT[$i]}' — refuse"
        exit 3
      fi
    fi
    if [ ! -f "$RUNDIRS_LOCAL/${R_DIR[$i]}/${R_CKPT[$i]}" ]; then
      log "mapping row $((i + 1)): local checkpoint missing: $RUNDIRS_LOCAL/${R_DIR[$i]}/${R_CKPT[$i]} — refuse"
      exit 3
    fi
  done
  log "mapping verified: 12 rows, BASELINE x A x B x C each x seeds 1-3, weights + checkpoints all match"

  local baseline
  baseline=$(spend_now)
  [ -n "$baseline" ] || { log "could not read spend baseline — refusing pre-spend"; exit 3; }
  log "spend baseline=\$$baseline guard=+\$$ROLLOUT_GUARD_USD (rollouts phase)"

  local slug ckpt_stem csv label stop_flag summary cmd
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    arm="${R_ARM[$i]}"; seed="${R_SEED[$i]}"; dir="${R_DIR[$i]}"; ckpt="${R_CKPT[$i]}"
    slug=$(slug_for "$arm") || abort "rollout row $((i + 1)): unknown arm $arm"
    ckpt_stem="${ckpt%.pt}"
    csv="rollout_010_${arm}_seed${seed}_${ckpt_stem}.csv"
    label="rollout $arm seed $seed ($dir @ $ckpt)"
    summary="$LOGDIR/supervise-010-rollout-$slug-s$seed-$STAMP.json"
    cmd="cd /workspace/IsaacLab && test -f logs/rsl_rl/warpauv_direct/$dir/$ckpt && ./isaaclab.sh -p ../isaac-auv-env/custom_workflows/play_rollout_eval.py --task Isaac-WarpAUV-Direct-v1 --num_envs 1 --headless --load_run $dir --checkpoint $ckpt --out_csv /workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$csv"
    if [ "$i" -lt 11 ]; then
      stop_flag="--no-stop"
    else
      stop_flag=""   # link 12 OWNS the stop (supervise syncs first, then stops)
    fi
    # shellcheck disable=SC2086
    run_link "$label" "$summary" \
      --job-name "010-curee-${slug}-rollout-s${seed}" \
      --sync-subdir rsl_rl/warpauv_direct --max-wait 2700 $stop_flag \
      --command "$cmd" \
      || abort "$label: supervise exited non-zero"
    if [ -s "$RUNDIRS_LOCAL/$csv" ]; then
      log "$label: CSV synced locally ($(wc -l < "$RUNDIRS_LOCAL/$csv" | tr -d ' ') lines)"
    else
      abort "$label: $csv missing/empty locally after sync"
    fi
    if [ "$i" -lt 11 ]; then
      check_spend "$baseline" "$ROLLOUT_GUARD_USD" || abort "spend guard breach after $label"
    fi
  done

  # Link 12's supervise owns the stop; verify it actually landed.
  st=$(pod_status_now)
  if [ "$st" = "running" ]; then
    log "WARNING: pod still running after final link — forcing stop now"
    if stop_pod_force; then log "stop_pod done"; else
      log "WARNING: stop_pod FAILED after 3 attempts — check pod_status NOW"
      exit 1
    fi
  else
    log "pod status after final link: $st (stop owned by link 12's supervise)"
  fi

  log "rollouts phase COMPLETE — 12/12 CSVs local; pod stopped. Offline scoring is next."
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
