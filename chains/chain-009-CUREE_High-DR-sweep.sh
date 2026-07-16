#!/usr/bin/env bash
# =============================================================================
# chain-009-CUREE_High-DR-sweep.sh — the ONE chain for campaign
# 009-CUREE_High-DR-sweep: 3-arm paired DR dose-response
# (DR_2 fresh control / DR_3 / DR_4 x seeds 1/2/3), all at the standing
# default recipe (2500 iters x 3.0 s), all scored at model_2499.pt.
#
# TWO PHASES, fired by the PARENT session (the one holding Kyle's direct
# $6.00-envelope approval), each as ONE `run_in_background` Bash task:
#
#   bash runpod-mcp/chains/chain-009-CUREE_High-DR-sweep.sh train
#       5-iter probe at DR_3 (verify + quarantine) -> 9 training links
#       ascending dose (DR_2 s1/2/3 -> DR_3 s1/2/3 -> DR_4 s1/2/3), each
#       supervised (--no-stop) with per-link guards -> emits the arm->run-dir
#       mapping logs/pod/chain-009-rundirs.tsv -> exits with the POD UP.
#
#   == STOP POINT == the parent session reviews chain-009-rundirs.tsv
#       (9 rows, one per arm, invariants in the header) and then fires:
#
#   bash runpod-mcp/chains/chain-009-CUREE_High-DR-sweep.sh rollouts [tsv]
#       read-only pod_status pre-check (stopped pod -> instructs ensure_pod,
#       NO auto bring-up: pod creation stays human-initiated) -> 9 rollout
#       links with EXPLICIT --load_run dirs from the tsv; links 1-8
#       --no-stop, link 9 owns stop_pod. Rollouts run the frozen eval
#       protocol (eval_mode, DR off, 3.0 s source default) BY DESIGN —
#       never "fix" it.
#
# FAILURE DOCTRINE (post-007 hardening): pre-launch-only link retry (one,
# after 60 s, only when the supervise summary shows NO job_id); ANY other
# failure -> explicit stop_pod(force) x3 -> abort non-zero. Never
# terminate_pod. The deadman fuse (armed by the parent per the handoff) is
# the independent money backstop.
#
# DR VERIFICATION NOTE (adaptation to reality, 2026-07-16): params/env.yaml
# does NOT record com_to_cob_offset_radius / volume_range (verified across
# all 21 synced run dirs — they are domain_randomization class attrs, not
# dumped cfg fields). The per-link DR guard therefore greps the two ACTIVE
# lines of /workspace/isaac-auv-env/warpauv_env.py pod-side right after each
# link: launch_training pushes the DR edit atomically with the launch and
# nothing else edits the file between a link's completion and its check.
# episode_length_s IS dumped, so that guard stays on the synced env.yaml.
#
# MEAN-REWARD GUARD: exactly one "Mean reward" line per iteration (equality
# verified on 7 real logs: 3x2500, 3x10000, 1x5). Equality is load-bearing:
# it is what catches a --max_iterations override that did not land (400- or
# 10000-line log); a >= check would wave that exact failure through.
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
MAPFILE_DEFAULT="$LOGDIR/chain-009-rundirs.tsv"
WARPAUV_ENV_REMOTE="/workspace/isaac-auv-env/warpauv_env.py"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

# Spend guards (deltas from each phase's own baseline; they SUM to the
# Kyle-approved $6.00 envelope; predicted total ~$1.7-3.0).
TRAIN_GUARD_USD="4.50"
ROLLOUT_GUARD_USD="1.50"

# The 9 arms, ascending dose. Kept in lock-step arrays (bash-3.2 safe).
ARM_LEVELS=(2 2 2 3 3 3 4 4 4)
ARM_SEEDS=(1 2 3 1 2 3 1 2 3)

# Expected ACTIVE-line value strings per level (verbatim from
# runpod_mcp/training.py DR_TABLES["curee"]; unit-tested char-exact).
radius_for() {
  case "$1" in
    2) printf '0.05' ;;
    3) printf '0.075' ;;
    4) printf '0.1' ;;
    *) return 1 ;;
  esac
}
vrange_for() {
  case "$1" in
    2) printf '[0.019747843530591773, 0.02574784353059178]' ;;
    3) printf '[0.018247843530591775, 0.027247843530591776]' ;;
    4) printf '[0.016747843530591777, 0.028747843530591774]' ;;
    *) return 1 ;;
  esac
}

log() { printf '[chain-009 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

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
# commented-out No-DR preset lines never collide with any 009 level's values).
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
# $1=label $2=level $3=seed $4=iters $5=max_wait $6=summary_path
train_link_guarded() {
  local label="$1" lvl="$2" seed="$3" iters="$4" max_wait="$5" summary="$6"
  local before_f after_f new_dirs n_new rcount rad vr
  rad=$(radius_for "$lvl") || abort "$label: unknown level $lvl"
  vr=$(vrange_for "$lvl") || abort "$label: unknown level $lvl"

  before_f=$(mktemp) && after_f=$(mktemp) || abort "$label: mktemp failed"
  list_rundirs > "$before_f"

  run_link "$label" "$summary" \
    --training curee --dr "$lvl" --seed "$seed" \
    --extra-args "--max_iterations $iters env.episode_length_s=3.0" \
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
    abort "$label: expected exactly ONE new synced run dir, found $n_new [$(echo $new_dirs)] — mapping integrity lost"
  fi
  NEW_DIR="$new_dirs"

  if [ "$iters" = "2500" ] && [ ! -f "$RUNDIRS_LOCAL/$NEW_DIR/model_2499.pt" ]; then
    abort "$label: model_2499.pt missing in $NEW_DIR"   # (probe at 5 iters has no model_2499)
  fi
  grep -q '^episode_length_s: 3.0' "$RUNDIRS_LOCAL/$NEW_DIR/params/env.yaml" \
    || abort "$label: episode_length_s: 3.0 not found in $NEW_DIR/params/env.yaml"

  if pod_dr_check "$rad" "$vr"; then
    log "$label: DR GUARD ok (radius=$rad landed in warpauv_env.py)"
  else
    abort "$label: pod-side DR guard failed for level DR_$lvl (radius=$rad)"
  fi
}

# ------------------------------------------------------------------- phases

phase_train() {
  log "chain start (train): campaign 009-CUREE_High-DR-sweep, stamp=$STAMP"
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

  # ---- 5-iter probe at DR_3 (~\$0.10): prove the NEW level lands end-to-end
  NEW_DIR="" JOB_ID=""
  train_link_guarded "probe DR_3 (5 iters)" 3 1 5 1500 \
    "$LOGDIR/supervise-009-probe-$STAMP.json"
  log "probe run dir: $NEW_DIR — quarantining pod-side"
  if pod_quarantine_probe "$NEW_DIR"; then
    log "probe dir quarantined to _probe_quarantine/ (its local copy is EXCLUDED from the mapping by construction)"
  else
    abort "probe quarantine failed for $NEW_DIR"
  fi
  check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after probe"

  # ---- mapping file (partial file stays valid row-by-row on a mid-chain abort)
  {
    echo "# chain-009-rundirs.tsv — arm -> run-dir mapping for 009-CUREE_High-DR-sweep (stamp $STAMP)"
    echo "# INVARIANT: exactly ONE new run dir appeared per training link (guarded); each row"
    echo "# was bound to its arm AT LINK COMPLETION from a per-link before/after diff of the"
    echo "# local synced dir list. The DR_3 probe dir was identified by its own diff and"
    echo "# EXCLUDED (quarantined pod-side in _probe_quarantine/). A PARTIAL file after a"
    echo "# mid-chain abort remains VALID row-by-row for the links it contains."
    echo "# columns: dr_level<TAB>seed<TAB>run_dir<TAB>job_id<TAB>mean_reward_lines"
  } > "$MAPFILE_DEFAULT"

  # ---- 9 training links, ascending dose
  local i lvl seed label
  for i in 0 1 2 3 4 5 6 7 8; do
    lvl=${ARM_LEVELS[$i]}
    seed=${ARM_SEEDS[$i]}
    label="train DR_${lvl} seed $seed (2500 iters, ep3s)"
    NEW_DIR="" JOB_ID=""
    train_link_guarded "$label" "$lvl" "$seed" 2500 3600 \
      "$LOGDIR/supervise-009-train-dr$lvl-s$seed-$STAMP.json"
    printf 'DR_%s\t%s\t%s\t%s\t%s\n' "$lvl" "$seed" "$NEW_DIR" "$JOB_ID" 2500 \
      >> "$MAPFILE_DEFAULT"
    log "$label: mapped -> $NEW_DIR (row appended to $MAPFILE_DEFAULT)"
    check_spend "$baseline" "$TRAIN_GUARD_USD" || abort "spend guard breach after $label"
  done

  log "train phase COMPLETE — 9/9 links green; mapping: $MAPFILE_DEFAULT"
  log "POD LEFT UP by design. == STOP POINT == parent session: review the mapping"
  log "(9 rows, invariants in header), then fire promptly:"
  log "  bash $DIR/chain-009-CUREE_High-DR-sweep.sh rollouts"
  log "Idle burn while you review is bounded (~\$0.69/hr) and covered by deadman + watchdog."
  exit 0
}

phase_rollouts() {
  local mapfile="${1:-$MAPFILE_DEFAULT}"
  log "chain start (rollouts): campaign 009-CUREE_High-DR-sweep, stamp=$STAMP, mapping=$mapfile"

  # Read-only pre-check: NEVER brings the pod up itself (human-initiated only).
  local st
  st=$(pod_status_now)
  if [ "$st" != "running" ]; then
    log "pod is not running (status=$st) — run ensure_pod in the parent session, then re-fire:"
    log "  bash $DIR/chain-009-CUREE_High-DR-sweep.sh rollouts"
    exit 3
  fi

  [ -f "$mapfile" ] || { log "mapping file not found: $mapfile"; exit 3; }

  # Parse + verify the mapping BEFORE any spend (bash-3.2-safe lock-step arrays).
  local lvls="" seeds="" dirs="" n=0 lvl seed dir jid rl
  while IFS=$'\t' read -r lvl seed dir jid rl; do
    case "$lvl" in \#*|"") continue ;; esac
    lvl="${lvl#DR_}"
    radius_for "$lvl" >/dev/null || { log "bad level '$lvl' in mapping"; exit 3; }
    [ -f "$RUNDIRS_LOCAL/$dir/model_2499.pt" ] \
      || { log "mapping row DR_$lvl s$seed: $dir has no local model_2499.pt — refuse"; exit 3; }
    lvls="$lvls $lvl"; seeds="$seeds $seed"; dirs="$dirs $dir"
    n=$((n + 1))
  done < "$mapfile"
  if [ "$n" -ne 9 ]; then
    log "mapping has $n data rows, expected 9 — refuse (salvage/resume is a parent-session call)"
    exit 3
  fi
  log "mapping verified: 9 rows, all with local model_2499.pt"

  local baseline
  baseline=$(spend_now)
  [ -n "$baseline" ] || { log "could not read spend baseline — refusing pre-spend"; exit 3; }
  log "spend baseline=\$$baseline guard=+\$$ROLLOUT_GUARD_USD (rollouts phase)"

  # Re-split the verified rows (sets positional params to the 9 triplets).
  set -- $lvls
  local L1=$1 L2=$2 L3=$3 L4=$4 L5=$5 L6=$6 L7=$7 L8=$8 L9=$9
  set -- $seeds
  local S1=$1 S2=$2 S3=$3 S4=$4 S5=$5 S6=$6 S7=$7 S8=$8 S9=$9
  set -- $dirs
  local D1=$1 D2=$2 D3=$3 D4=$4 D5=$5 D6=$6 D7=$7 D8=$8 D9=$9

  local i cmd csv label stop_flag summary
  for i in 1 2 3 4 5 6 7 8 9; do
    eval "lvl=\$L$i; seed=\$S$i; dir=\$D$i"
    csv="rollout_009_DR${lvl}_seed${seed}_model2499.csv"
    label="rollout DR_${lvl} seed $seed ($dir @ model_2499)"
    summary="$LOGDIR/supervise-009-rollout-dr$lvl-s$seed-$STAMP.json"
    cmd="cd /workspace/IsaacLab && test -f logs/rsl_rl/warpauv_direct/$dir/model_2499.pt && ./isaaclab.sh -p ../isaac-auv-env/custom_workflows/play_rollout_eval.py --task Isaac-WarpAUV-Direct-v1 --num_envs 1 --headless --load_run $dir --checkpoint model_2499.pt --out_csv /workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$csv"
    if [ "$i" -lt 9 ]; then
      stop_flag="--no-stop"
    else
      stop_flag=""   # link 9 OWNS the stop (supervise syncs first, then stops)
    fi
    # shellcheck disable=SC2086
    run_link "$label" "$summary" \
      --job-name "009-curee-dr$lvl-rollout-s$seed" \
      --sync-subdir rsl_rl/warpauv_direct --max-wait 2700 $stop_flag \
      --command "$cmd" \
      || abort "$label: supervise exited non-zero"
    if [ -s "$RUNDIRS_LOCAL/$csv" ]; then
      log "$label: CSV synced locally ($(wc -l < "$RUNDIRS_LOCAL/$csv" | tr -d ' ') lines)"
    else
      abort "$label: $csv missing/empty locally after sync"
    fi
    if [ "$i" -lt 9 ]; then
      check_spend "$baseline" "$ROLLOUT_GUARD_USD" || abort "spend guard breach after $label"
    fi
  done

  # Link 9's supervise owns the stop; verify it actually landed.
  st=$(pod_status_now)
  if [ "$st" = "running" ]; then
    log "WARNING: pod still running after final link — forcing stop now"
    if stop_pod_force; then log "stop_pod done"; else
      log "WARNING: stop_pod FAILED after 3 attempts — check pod_status NOW"
      exit 1
    fi
  else
    log "pod status after final link: $st (stop owned by link 9's supervise)"
  fi

  log "rollouts phase COMPLETE — 9/9 CSVs local; pod stopped. Phase D (offline scoring) is next."
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
