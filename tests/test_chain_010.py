"""Offline behavioral coverage for chain-010-CUREE_Weight-screen.sh.

The chain script (CUREE/chains/chain-010-CUREE_Weight-screen.sh) has NO
test hooks — it derives every path from BASH_SOURCE (DIR -> REPO -> MCPDIR),
exactly like chain-009. So this suite sandboxes it by COPYING the real script
into a scratch repo layout under pytest tmp_path, alongside:

  - a stub `supervise.sh` (bash) that fabricates realistic training/rollout
    side effects (new run dirs, env.yaml, job logs, CSVs, summary JSON) under
    env-var knobs, and logs its full argv for assertions;
  - a stub `.venv/bin/python` that `exec`s the REAL python3 with PYTHONPATH
    pointed at a FAKE `runpod_mcp` package materialized under tmp_path (never
    on this test suite's own sys.path — the real package must stay untouched
    when the full MCP suite runs);
  - a fake pod filesystem root (`<tmp>/podfs/workspace/...`) that
    `exec_on_pod` translates "/workspace" onto, so every pod-side grep/test-f/
    mkdir-quarantine the chain performs runs for real (locally) against
    fixture files;
  - a fake `sleep` on PATH so a mis-built stub that trips run_link's 60s
    pre-launch retry, or stop_pod_force's inter-attempt gap, costs a FAST
    test failure rather than a real-time hang. (stop_pod_force's own 20s
    gaps are `time.sleep()` in the fake tools.py, not the `sleep` binary —
    they're avoided structurally instead, by making the fake stop_pod always
    succeed on attempt 1, per the "not re-tested" retry-cadence note below.)

No test in this file ever imports runpod_mcp.tools directly, calls a runpod
MCP tool, or touches a real pod — every scenario runs the chain script as a
subprocess against the fixtures above.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]                    # runpod-mcp/
REPO_ROOT = MCP_ROOT.parent                                       # repo root
# The chains moved to CUREE/ on 2026-08-11 (problem-log C2): they are
# CUREE-owned campaign drivers, not MCP suite layout. The sandbox below must
# mirror that, or it would execute the script from the OLD layout and pass
# without ever exercising the relocation.
REAL_CHAIN = REPO_ROOT / "CUREE" / "chains" / "chain-010-CUREE_Weight-screen.sh"

BASELINE_DIRS = {
    1: "2026-07-16_12-51-52",
    2: "2026-07-16_13-02-25",
    3: "2026-07-16_13-13-01",
}
BASELINE_TRIPLE = ("0.2", "0.5", "0.2")
ARM_TRIPLES = {
    "A": ("0.3", "0.4", "0.2"),
    "B": ("0.1", "0.6", "0.2"),
    "C": ("0.25", "0.55", "0.1"),
}
ARM_SLUG = {"BASELINE": "baseline", "A": "arm-a", "B": "arm-b", "C": "arm-c"}

DR2_RADIUS = "0.05"
DR2_VRANGE = "[0.019747843530591773, 0.02574784353059178]"


def _weights_str(triple: tuple[str, str, str]) -> str:
    return "/".join(triple)


# ============================================================ fixture sources

FAKE_TOOLS_PY = '''
"""Fake runpod_mcp.tools — offline stand-in used ONLY by chain-010's test
harness. Every call is env-var driven so a single test process can script
many scenarios without touching a real pod or the Keychain."""
import os
import subprocess


class _RT:
    pass


def runtime():
    return _RT()


def _log(line):
    path = os.environ.get("STUB_CALLS_LOG")
    if not path:
        return
    with open(path, "a") as f:
        f.write(line + "\\n")


def spend_report(rt):
    path = os.environ["STUB_SPEND_STATE_FILE"]
    try:
        with open(path) as f:
            val = float((f.read() or "0").strip() or "0")
    except FileNotFoundError:
        val = 0.0
    return {"total_usd": val}


def pod_status(rt):
    path = os.environ["STUB_POD_STATE_FILE"]
    try:
        with open(path) as f:
            status = (f.read() or "running").strip() or "running"
    except FileNotFoundError:
        status = "running"
    return {"status": status}


def stop_pod(rt, force=False):
    path = os.environ["STUB_POD_STATE_FILE"]
    with open(path, "w") as f:
        f.write("stopped")
    _log(f"STOP_POD force={force}")
    return {"status": "stopped"}


def sync_logs(rt, subdir="rsl_rl/warpauv_direct"):
    _log(f"SYNC_LOGS subdir={subdir}")
    return {"status": "ok"}


def exec_on_pod(rt, cmd, timeout_sec=120, workdir=None):
    podfs = os.environ["FAKE_PODFS_ROOT"]
    full = f"cd {workdir} && {cmd}" if workdir else cmd
    translated = full.replace("/workspace", f"{podfs}/workspace")
    _log(f"EXEC_ON_POD {translated}")
    proc = subprocess.run(["bash", "-c", translated], capture_output=True,
                          text=True, timeout=timeout_sec)
    return {"exit_code": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr}
'''

STUB_SUPERVISE_SH = r'''#!/usr/bin/env bash
# Fake supervise.sh for chain-010 offline tests. NOT the real launcher — it
# never touches SSH/RunPod; it fabricates the same on-disk side effects a
# real supervised training/rollout job would leave, driven by STUB_* env
# knobs (documented in test_chain_010.py). Its own path derivation mirrors
# the real supervise.sh/chain scripts (BASH_SOURCE -> DIR -> REPO) so it
# writes to the SAME logs/pod tree the chain script reads back.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
LOGDIR="$REPO/logs/pod"
RUNDIRS_LOCAL="$LOGDIR/rsl_rl/warpauv_direct"
PYBIN="$DIR/.venv/bin/python"
mkdir -p "$RUNDIRS_LOCAL" "$LOGDIR/jobs"

next_counter() {  # $1 = counter file -> prints the NEW (incremented) value
  local f="$1" n
  [ -f "$f" ] || echo 0 > "$f"
  n=$(cat "$f")
  n=$((n + 1))
  echo "$n" > "$f"
  echo "$n"
}

fail_gate() {  # $1 = "1" if the knob was requested at all
  [ "$1" = "1" ] || return 1
  if [ -n "${STUB_FAIL_AT_CALL:-}" ] && [ "$CALLNUM" != "$STUB_FAIL_AT_CALL" ]; then
    return 1
  fi
  return 0
}

# ---- log the call verbatim (one arg per line, block-delimited) ------------
if [ -n "${STUB_CALLS_LOG:-}" ]; then
  {
    echo "=== SUPERVISE CALL START ==="
    for a in "$@"; do printf '%s\n' "$a"; done
    echo "=== SUPERVISE CALL END ==="
  } >> "$STUB_CALLS_LOG"
fi

MODE=""
EXTRA_ARGS=""
SUMMARY_PATH=""
NO_STOP="0"
COMMAND=""

while [ $# -gt 0 ]; do
  case "$1" in
    --training) MODE="training"; shift 2 ;;
    --dr) shift 2 ;;
    --seed) shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --job-name) MODE="job"; shift 2 ;;
    --command) COMMAND="$2"; shift 2 ;;
    --sync-subdir) shift 2 ;;
    --max-wait) shift 2 ;;
    --no-stop) NO_STOP="1"; shift ;;
    --summary-path) SUMMARY_PATH="$2"; shift 2 ;;
    *) shift ;;
  esac
done

CALLNUM=$(next_counter "${STUB_CALL_COUNTER_FILE:?}")
JOB_ID="stub-job-$(printf '%04d' "$CALLNUM")"

if [ "$MODE" = "training" ]; then
  MAXITERS=$(printf '%s' "$EXTRA_ARGS" | grep -oE -- '--max_iterations [0-9]+' | awk '{print $2}')
  [ -n "$MAXITERS" ] || MAXITERS=0
  POS=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'env\.rew_scale_pos=[0-9.]+' | cut -d= -f2)
  ANG=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'env\.rew_scale_ang=[0-9.]+' | cut -d= -f2)
  ACT=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'env\.rew_scale_actions=[0-9.]+' | cut -d= -f2)
  EPLEN=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'env\.episode_length_s=[0-9.]+' | cut -d= -f2 || true)
  [ -n "$EPLEN" ] || EPLEN="3.0"

  if fail_gate "${STUB_IGNORE_OVERRIDES:-0}"; then
    POS="0.2"; ANG="0.5"; ACT="0.2"
  elif fail_gate "${STUB_WRONG_WEIGHTS:-0}"; then
    POS="0.15"; ANG="0.35"; ACT="0.4"
  fi

  NEWDIRS=""
  if fail_gate "${STUB_ZERO_RUNDIRS:-0}"; then
    :   # deliberately create nothing
  elif fail_gate "${STUB_DOUBLE_RUNDIRS:-0}"; then
    d1=$(next_counter "$STUB_DIR_COUNTER_FILE"); n1=$(printf '2000-01-%02d_00-00-00' "$d1")
    d2=$(next_counter "$STUB_DIR_COUNTER_FILE"); n2=$(printf '2000-01-%02d_00-00-00' "$d2")
    NEWDIRS="$n1 $n2"
  else
    d1=$(next_counter "$STUB_DIR_COUNTER_FILE"); n1=$(printf '2000-01-%02d_00-00-00' "$d1")
    NEWDIRS="$n1"
  fi

  for nd in $NEWDIRS; do
    mkdir -p "$RUNDIRS_LOCAL/$nd/params"
    {
      echo "rew_scale_pos: $POS"
      echo "rew_scale_ang: $ANG"
      echo "rew_scale_actions: $ACT"
      echo "episode_length_s: $EPLEN"
    } > "$RUNDIRS_LOCAL/$nd/params/env.yaml"
    mkdir -p "$FAKE_PODFS_ROOT/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$nd/params"
    cp "$RUNDIRS_LOCAL/$nd/params/env.yaml" \
       "$FAKE_PODFS_ROOT/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$nd/params/env.yaml"
    if [ "${STUB_NO_CKPT:-0}" != "1" ]; then
      touch "$RUNDIRS_LOCAL/$nd/model_1499.pt"
      touch "$FAKE_PODFS_ROOT/workspace/IsaacLab/logs/rsl_rl/warpauv_direct/$nd/model_1499.pt"
    fi
  done

  RLINES="$MAXITERS"
  if [ -n "${STUB_REWARD_LINES:-}" ] && fail_gate 1; then
    RLINES="$STUB_REWARD_LINES"
  fi
  mkdir -p "$LOGDIR/jobs/$JOB_ID"
  if [ "$RLINES" -gt 0 ] 2>/dev/null; then
    yes "Mean reward: 0.0" | head -n "$RLINES" > "$LOGDIR/jobs/$JOB_ID/out.log"
  else
    : > "$LOGDIR/jobs/$JOB_ID/out.log"
  fi
fi

if [ "$MODE" = "job" ]; then
  CSVNAME=$(printf '%s' "$COMMAND" | grep -oE -- '--out_csv [^ ]+' | awk '{print $2}' | xargs -n1 basename)
  if [ -n "$CSVNAME" ]; then
    if [ "${STUB_EMPTY_CSV:-0}" = "1" ]; then
      : > "$RUNDIRS_LOCAL/$CSVNAME"
    else
      { echo "t,x,y,z"; echo "0,0,0,0"; } > "$RUNDIRS_LOCAL/$CSVNAME"
    fi
  fi
fi

if [ "${STUB_NO_JOBID:-0}" = "1" ]; then
  JOBFIELD="null"
else
  JOBFIELD="\"$JOB_ID\""
fi
if [ -n "$SUMMARY_PATH" ]; then
  mkdir -p "$(dirname "$SUMMARY_PATH")"
  printf '{"job_id": %s, "mode": "%s"}\n' "$JOBFIELD" "$MODE" > "$SUMMARY_PATH"
fi

if [ -n "${STUB_SPEND_STATE_FILE:-}" ]; then
  CUR=$(cat "$STUB_SPEND_STATE_FILE" 2>/dev/null || echo 0)
  BUMP="${STUB_SPEND_BUMP:-0.05}"
  NEW=$("$PYBIN" -c "print(float(\"$CUR\") + float(\"$BUMP\"))")
  echo "$NEW" > "$STUB_SPEND_STATE_FILE"
fi

if [ "$NO_STOP" != "1" ]; then
  "$PYBIN" -c 'from runpod_mcp import tools; tools.stop_pod(tools.runtime(), force=False)' >/dev/null
fi

exit "${STUB_SUPERVISE_RC:-0}"
'''

FAKE_SLEEP = '''#!/usr/bin/env bash
# Fast stand-in for /bin/sleep — any accidental retry/backoff sleep in the
# chain (or a mis-built stub) costs ~0s instead of a real 20-60s hang.
exit 0
'''


# ==================================================================== harness

class Harness:
    """One scratch repo (under a pytest tmp_path) wired for chain-010."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.repo = tmp_path / "repo"
        self.mcp = self.repo / "runpod-mcp"
        self.curee = self.repo / "CUREE"
        self.chains = self.curee / "chains"
        self.venv_bin = self.mcp / ".venv" / "bin"
        self.logdir = self.repo / "logs" / "pod"
        self.rundirs_local = self.logdir / "rsl_rl" / "warpauv_direct"
        self.podfs = tmp_path / "podfs"
        self.fakepkg = tmp_path / "fakepkg"
        self.bin = tmp_path / "bin"
        self.chain_script = self.chains / "chain-010-CUREE_Weight-screen.sh"
        self.calls_log = tmp_path / "calls.log"
        self.spend_state = tmp_path / "spend_state.txt"
        self.pod_state = tmp_path / "pod_state.txt"
        self.dir_counter = tmp_path / "dircounter.txt"
        self.call_counter = tmp_path / "callcounter.txt"
        self.default_tsv = self.logdir / "chain-010-rundirs.tsv"
        self._build()

    # ---------------------------------------------------------------- build

    def _build(self):
        self.chains.mkdir(parents=True)
        # Explicit: `chains` no longer lives under `mcp`, so creating it no
        # longer creates `mcp` as a side effect. The stub supervise.sh /
        # requirements.txt / .venv all land there.
        self.mcp.mkdir(parents=True, exist_ok=True)
        self.venv_bin.mkdir(parents=True)
        self.rundirs_local.mkdir(parents=True)
        (self.logdir / "jobs").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        self.fakepkg.mkdir(parents=True)
        (self.podfs / "workspace" / "isaac-auv-env").mkdir(parents=True)
        (self.podfs / "workspace" / "IsaacLab" / "logs" / "rsl_rl"
         / "warpauv_direct").mkdir(parents=True)

        # -- copy the REAL chain script under test (no test hooks inside it)
        shutil.copyfile(REAL_CHAIN, self.chain_script)
        self.chain_script.chmod(self.chain_script.stat().st_mode | stat.S_IEXEC)

        # -- stub supervise.sh
        supervise = self.mcp / "supervise.sh"
        supervise.write_text(STUB_SUPERVISE_SH)
        supervise.chmod(supervise.stat().st_mode | stat.S_IEXEC)

        # -- dummy requirements.txt + a NEWER .deps-stamp (bootstrap_venv no-op)
        req = self.mcp / "requirements.txt"
        req.write_text("# fake requirements, never installed\n")
        stamp = self.venv_bin.parent / ".deps-stamp"
        os.utime(req, (1000, 1000))
        stamp.write_text("stamp\n")
        os.utime(stamp, (2000, 2000))

        # -- fake runpod_mcp package (never on THIS test suite's sys.path)
        pkg = self.fakepkg / "runpod_mcp"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "tools.py").write_text(FAKE_TOOLS_PY)

        # -- stub .venv/bin/python: exec the REAL interpreter, fake PYTHONPATH
        py_stub = self.venv_bin / "python"
        py_stub.write_text(textwrap.dedent(f"""\
            #!/bin/bash
            export PYTHONPATH="{self.fakepkg}:${{PYTHONPATH:-}}"
            exec "{sys.executable}" "$@"
            """))
        py_stub.chmod(py_stub.stat().st_mode | stat.S_IEXEC)

        # -- fake sleep on PATH (defensive: fast-fail, never hang)
        sleep_stub = self.bin / "sleep"
        sleep_stub.write_text(FAKE_SLEEP)
        sleep_stub.chmod(sleep_stub.stat().st_mode | stat.S_IEXEC)

        # -- pod-side DR_2 fixture (chain-010's DR guard reads this)
        warpauv = self.podfs / "workspace" / "isaac-auv-env" / "warpauv_env.py"
        warpauv.write_text(
            "class DomainRandomization:\n"
            f"        com_to_cob_offset_radius = {DR2_RADIUS} # DR_2 (stub fixture)\n"
            f"        volume_range = {DR2_VRANGE} # DR_2 (stub fixture)\n"
        )

        # -- state files
        self.spend_state.write_text("10.00")
        self.pod_state.write_text("running")

    # ------------------------------------------------------------ fixtures

    def seed_baseline(self, missing_pod_ckpt_for: int | None = None,
                      missing_local_for: int | None = None):
        """Seed the 3 known 009 BASELINE dirs, locally + pod-side, complete
        by default. `missing_pod_ckpt_for`/`missing_local_for` (seed 1-3)
        deliberately omit ONE checkpoint to drive the pre-flight-failure
        test."""
        for seed, dirname in BASELINE_DIRS.items():
            if seed != missing_local_for:
                local_dir = self.rundirs_local / dirname / "params"
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "env.yaml").write_text(
                    f"rew_scale_pos: {BASELINE_TRIPLE[0]}\n"
                    f"rew_scale_ang: {BASELINE_TRIPLE[1]}\n"
                    f"rew_scale_actions: {BASELINE_TRIPLE[2]}\n"
                    "episode_length_s: 3.0\n")
                (self.rundirs_local / dirname / "model_1500.pt").write_text("x")
            if seed != missing_pod_ckpt_for:
                pod_dir = (self.podfs / "workspace" / "IsaacLab" / "logs"
                          / "rsl_rl" / "warpauv_direct" / dirname)
                pod_dir.mkdir(parents=True, exist_ok=True)
                (pod_dir / "model_1500.pt").write_text("x")

    def touch_checkpoint(self, dirname: str, ckpt: str):
        d = self.rundirs_local / dirname
        d.mkdir(parents=True, exist_ok=True)
        (d / ckpt).write_text("x")

    def write_tsv(self, rows: list[tuple], path: Path | None = None):
        path = path or self.default_tsv
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# chain-010-rundirs.tsv (test fixture)\n")
            f.write("# columns: arm\\tseed\\tweights\\trun_dir\\tjob_id\\t"
                    "mean_reward_lines\\tcheckpoint\n")
            for row in rows:
                f.write("\t".join(str(c) for c in row) + "\n")
        return path

    # ------------------------------------------------------------------ run

    def run(self, phase: str, *extra_args: str, env: dict | None = None,
            timeout: int = 30) -> subprocess.CompletedProcess:
        base_env = dict(os.environ)
        base_env["PATH"] = f"{self.bin}:{base_env.get('PATH', '')}"
        base_env.update({
            "FAKE_PODFS_ROOT": str(self.podfs),
            "STUB_CALLS_LOG": str(self.calls_log),
            "STUB_SPEND_STATE_FILE": str(self.spend_state),
            "STUB_SPEND_BUMP": "0.05",
            "STUB_POD_STATE_FILE": str(self.pod_state),
            "STUB_DIR_COUNTER_FILE": str(self.dir_counter),
            "STUB_CALL_COUNTER_FILE": str(self.call_counter),
        })
        if env:
            base_env.update(env)
        args = ["bash", str(self.chain_script), phase, *extra_args]
        return subprocess.run(args, capture_output=True, text=True,
                              cwd=str(self.repo), env=base_env, timeout=timeout)


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


# =============================================================== log parsing

def parse_calls(calls_log: Path) -> list[list[str]]:
    if not calls_log.exists():
        return []
    lines = calls_log.read_text().splitlines()
    calls: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if ln == "=== SUPERVISE CALL START ===":
            cur = []
        elif ln == "=== SUPERVISE CALL END ===":
            if cur is not None:
                calls.append(cur)
            cur = None
        elif cur is not None:
            cur.append(ln)
    return calls


def call_dict(argv: list[str]) -> dict:
    d = {"argv": argv, "no_stop": "--no-stop" in argv}
    i = 0
    named = {"--training", "--dr", "--seed", "--extra-args", "--job-name",
            "--command", "--sync-subdir", "--max-wait", "--summary-path"}
    while i < len(argv):
        a = argv[i]
        if a in named:
            key = a.lstrip("-").replace("-", "_")
            d[key] = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
        else:
            i += 1
    return d


def stop_count(calls_log: Path) -> int:
    if not calls_log.exists():
        return 0
    return calls_log.read_text().count("STOP_POD")


def read_tsv_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append(line.split("\t"))
    return rows


# ======================================================================= 0

def test_bash_syntax_clean():
    proc = subprocess.run(["bash", "-n", str(REAL_CHAIN)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_shellcheck_best_effort(tmp_path):
    """Best-effort means this test may SKIP but must never ERROR: an offline
    or slow-PyPI environment surfaces as TimeoutExpired/OSError, which must
    take the skip path. Only a real lint finding (rc != 0 from an installed
    shellcheck) is a failure."""
    sc_venv = tmp_path / "sc-venv"
    try:
        proc = subprocess.run([sys.executable, "-m", "venv", str(sc_venv)],
                              capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(f"could not create scratch venv for shellcheck: {exc}")
    if proc.returncode != 0:
        pytest.skip(f"could not create scratch venv for shellcheck: {proc.stderr}")
    pip = sc_venv / "bin" / "pip"
    try:
        install = subprocess.run([str(pip), "install", "-q", "shellcheck-py"],
                                 capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(f"shellcheck-py install timed out/errored (offline?): {exc}")
    if install.returncode != 0:
        pytest.skip(f"shellcheck-py install failed (offline?): {install.stderr[-500:]}")
    sc = sc_venv / "bin" / "shellcheck"
    try:
        result = subprocess.run([str(sc), str(REAL_CHAIN)],
                                capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(f"shellcheck invocation timed out/errored: {exc}")
    assert result.returncode == 0, result.stdout + result.stderr


# ================================================================ 1. happy path (train)

def test_train_happy_path(harness):
    harness.seed_baseline()
    proc = harness.run("train")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    rows = read_tsv_rows(harness.default_tsv)
    assert len(rows) == 12
    baseline_rows = [r for r in rows if r[0] == "BASELINE"]
    fresh_rows = [r for r in rows if r[0] != "BASELINE"]
    assert len(baseline_rows) == 3
    assert len(fresh_rows) == 9

    seen_baseline_seeds = set()
    for arm, seed, weights, dirname, job_id, rlines, ckpt in baseline_rows:
        assert weights == _weights_str(BASELINE_TRIPLE)
        assert dirname == BASELINE_DIRS[int(seed)]
        assert ckpt == "model_1500.pt"
        seen_baseline_seeds.add(seed)
    assert seen_baseline_seeds == {"1", "2", "3"}

    seen_fresh = set()
    for arm, seed, weights, dirname, job_id, rlines, ckpt in fresh_rows:
        assert arm in ARM_TRIPLES
        assert weights == _weights_str(ARM_TRIPLES[arm])
        assert rlines == "1500"
        assert ckpt == "model_1499.pt"
        assert job_id.startswith("stub-job-")
        assert (harness.rundirs_local / dirname / ckpt).exists()
        seen_fresh.add((arm, seed))
    assert seen_fresh == {(a, str(s)) for a in "ABC" for s in (1, 2, 3)}

    # probe dir: the one locally-synced run dir NOT referenced by any TSV row
    tsv_dirs = {r[3] for r in rows}
    all_local_dirs = {p.name for p in harness.rundirs_local.iterdir() if p.is_dir()}
    probe_candidates = all_local_dirs - tsv_dirs - set(BASELINE_DIRS.values())
    assert len(probe_candidates) == 1
    probe_dir = next(iter(probe_candidates))
    pod_run_root = (harness.podfs / "workspace" / "IsaacLab" / "logs" / "rsl_rl"
                    / "warpauv_direct")
    assert not (pod_run_root / probe_dir).exists()
    assert (pod_run_root / "_probe_quarantine" / probe_dir).exists()

    # exactly 10 training-mode supervise calls (1 probe + 9 links), all --no-stop
    calls = [call_dict(a) for a in parse_calls(harness.calls_log)]
    training_calls = [c for c in calls if c.get("training")]
    assert len(training_calls) == 10
    assert all(c["no_stop"] for c in training_calls)
    # pod left up by design: no STOP_POD anywhere in the train phase
    assert stop_count(harness.calls_log) == 0


# ============================================================ 2. reward-count guard

def test_train_reward_count_mismatch_aborts(harness):
    harness.seed_baseline()
    proc = harness.run("train", env={
        "STUB_FAIL_AT_CALL": "2",   # 1st trained link (arm A seed 1)
        "STUB_REWARD_LINES": "2500",
    })
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    assert "reward-line count" in (proc.stdout + proc.stderr)


# ========================================================= 3. new-dir-count guard

def test_train_zero_new_dirs_aborts(harness):
    harness.seed_baseline()
    proc = harness.run("train", env={
        "STUB_FAIL_AT_CALL": "2",
        "STUB_ZERO_RUNDIRS": "1",
    })
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    assert "exactly ONE new synced run dir" in (proc.stdout + proc.stderr)


def test_train_two_new_dirs_aborts(harness):
    harness.seed_baseline()
    proc = harness.run("train", env={
        "STUB_FAIL_AT_CALL": "2",
        "STUB_DOUBLE_RUNDIRS": "1",
    })
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    assert "exactly ONE new synced run dir" in (proc.stdout + proc.stderr)


# ============================================================ 4. weight guard miss

def test_train_weight_guard_mismatch_aborts(harness):
    harness.seed_baseline()
    proc = harness.run("train", env={
        "STUB_FAIL_AT_CALL": "2",
        "STUB_WRONG_WEIGHTS": "1",
    })
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    out = proc.stdout + proc.stderr
    assert "WEIGHT GUARD" in out
    assert "silently ignored (fake null)" not in out   # NOT the fake-null path


# ======================================================== 5. fake-null weight guard

def test_train_fake_null_weight_guard_aborts_with_distinct_message(harness):
    harness.seed_baseline()
    # Global (no STUB_FAIL_AT_CALL): the probe itself — arm C's triple, ALL
    # THREE keys off-default — is exactly the mechanism designed to catch a
    # silently-ignored override before any training link spends money.
    proc = harness.run("train", env={"STUB_IGNORE_OVERRIDES": "1"})
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    out = proc.stdout + proc.stderr
    assert "silently ignored (fake null)" in out
    # the probe was the only supervise call made before the abort
    calls = parse_calls(harness.calls_log)
    assert len(calls) == 1


# ==================================================== 6. baseline pre-flight failure

def test_train_baseline_preflight_failure_aborts_before_any_link(harness):
    harness.seed_baseline(missing_pod_ckpt_for=2)
    proc = harness.run("train")
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    assert parse_calls(harness.calls_log) == []   # no supervise call ever made
    assert "baseline pre-flight" in (proc.stdout + proc.stderr)


# ============================================================== 7. spend guard

def test_train_spend_guard_breach_aborts(harness):
    harness.seed_baseline()
    proc = harness.run("train", env={"STUB_SPEND_BUMP": "5.00"})
    assert proc.returncode != 0
    assert stop_count(harness.calls_log) >= 1
    assert "SPEND GUARD BREACH" in (proc.stdout + proc.stderr)


# ============================================== 7b. run_link retry doctrine

def test_train_prelaunch_failure_retries_once_then_aborts(harness):
    """run_link's ONE sanctioned retry: supervise fails WITHOUT a job_id in
    its summary (a pre-launch failure) -> exactly one retry (two supervise
    invocations for that link — the PATH-stubbed `sleep` makes the 60s wait
    instant); when the retry also fails, abort + stop. Knobs are global, so
    the probe (the first link) is where it trips — nothing runs after."""
    harness.seed_baseline()
    proc = harness.run("train", env={
        "STUB_SUPERVISE_RC": "1",
        "STUB_NO_JOBID": "1",
    })
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "RETRY" in out                       # the pre-launch retry fired
    calls = [call_dict(a) for a in parse_calls(harness.calls_log)]
    assert len(calls) == 2                      # probe: attempt 1 + retry, then abort
    assert all(c.get("training") for c in calls)
    assert "supervise exited non-zero" in out
    assert stop_count(harness.calls_log) >= 1


def test_train_failure_with_jobid_does_not_retry(harness):
    """The paired case (the double-launch/double-spend guard): supervise
    fails but its summary DOES carry a job_id -> NO retry (exactly one
    invocation), then abort + stop."""
    harness.seed_baseline()
    proc = harness.run("train", env={"STUB_SUPERVISE_RC": "1"})
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "RETRY" not in out
    calls = [call_dict(a) for a in parse_calls(harness.calls_log)]
    assert len(calls) == 1
    assert "supervise exited non-zero" in out
    assert stop_count(harness.calls_log) >= 1


# ================================================== 7c. checkpoint-missing guard

def test_train_missing_checkpoint_aborts(harness):
    """STUB_NO_CKPT (global): the 5-iter probe is exempt from the
    model_1499.pt check by design and passes all its guards, so the first
    1500-iter link (arm A seed 1) hits the checkpoint-missing abort + stop."""
    harness.seed_baseline()
    proc = harness.run("train", env={"STUB_NO_CKPT": "1"})
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "model_1499.pt missing" in out
    assert len(parse_calls(harness.calls_log)) == 2   # probe (green) + failing link 1
    assert stop_count(harness.calls_log) >= 1


# ======================================================= 8. rollouts refusal matrix

def _valid_12_rows(harness):
    rows = []
    for seed in (1, 2, 3):
        d = BASELINE_DIRS[seed]
        harness.touch_checkpoint(d, "model_1500.pt")
        rows.append(("BASELINE", seed, _weights_str(BASELINE_TRIPLE), d, "-", "-",
                    "model_1500.pt"))
    n = 0
    for arm in ("A", "B", "C"):
        for seed in (1, 2, 3):
            n += 1
            d = f"2001-01-{n:02d}_00-00-00"
            harness.touch_checkpoint(d, "model_1499.pt")
            rows.append((arm, seed, _weights_str(ARM_TRIPLES[arm]), d,
                        f"prior-job-{n}", 1500, "model_1499.pt"))
    return rows


def test_rollouts_refuses_wrong_row_count(harness):
    rows = _valid_12_rows(harness)[:-1]   # 11 rows
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0


def test_rollouts_refuses_bad_arm_seed_set(harness):
    rows = _valid_12_rows(harness)
    # duplicate BASELINE seed 1 in place of arm C seed 3 -> set is wrong
    rows[-1] = rows[0]
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0


@pytest.mark.parametrize("evil_suffix", [";touch {marker}", "$(touch {marker})"])
def test_rollouts_refuses_metachar_run_dir(harness, evil_suffix):
    """Codex finding 1: a TSV run_dir carrying shell metacharacters must be
    REFUSED at validation (rc 3, pre-spend, no stop) — never interpolated
    into the pod-side command. The local checkpoint fixture is created under
    the literal metachar dir name so the quoted `test -f` check alone would
    pass; only the anchored run-dir shape guard can refuse this row."""
    rows = _valid_12_rows(harness)
    marker = harness.tmp_path / "pwned"
    evil_dir = "2001-01-04_00-00-00" + evil_suffix.format(marker=marker)
    harness.touch_checkpoint(evil_dir, "model_1499.pt")
    row = list(rows[3])   # arm A seed 1
    row[3] = evil_dir
    rows[3] = tuple(row)
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0
    assert not marker.exists()   # the injected command demonstrably never ran


def test_rollouts_refuses_swapped_baseline_dirs(harness):
    """Codex finding 2: each BASELINE seed must bind to ITS exact 009 dir.
    A seed1<->seed2 dir swap passes a membership-only check (both dirs are
    known, both checkpoints exist) but silently mislabels the baseline
    rollout CSVs — it must be refused (rc 3, no supervise call, no stop)."""
    rows = _valid_12_rows(harness)
    r0, r1 = list(rows[0]), list(rows[1])   # BASELINE seed 1 / seed 2
    r0[3], r1[3] = r1[3], r0[3]
    rows[0], rows[1] = tuple(r0), tuple(r1)
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0


def test_rollouts_refuses_missing_local_checkpoint(harness):
    rows = _valid_12_rows(harness)
    # the last row's checkpoint file: remove it after _valid_12_rows touched it
    last = rows[-1]
    (harness.rundirs_local / last[3] / last[6]).unlink()
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0


# ============================================================ 9. rollouts happy path

def test_rollouts_happy_path(harness):
    rows = _valid_12_rows(harness)
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    calls = [call_dict(a) for a in parse_calls(harness.calls_log)]
    assert len(calls) == 12
    for i, (row, call) in enumerate(zip(rows, calls)):
        arm, seed, weights, dirname, job_id, rlines, ckpt = row
        expected_job_name = f"010-curee-{ARM_SLUG[arm]}-rollout-s{seed}"
        assert call.get("job_name") == expected_job_name, (i, call)
        cmd = call.get("command") or ""
        assert f"--load_run {dirname}" in cmd
        assert f"--checkpoint {ckpt}" in cmd
        ckpt_stem = ckpt[:-3] if ckpt.endswith(".pt") else ckpt
        assert f"rollout_010_{arm}_seed{seed}_{ckpt_stem}.csv" in cmd
        if i < 11:
            assert call["no_stop"] is True, (i, call)
        else:
            assert call["no_stop"] is False, (i, call)

    assert stop_count(harness.calls_log) == 1
    assert harness.pod_state.read_text().strip() == "stopped"


# ============================================== 9b. rollout empty-CSV abort

def test_rollouts_empty_csv_aborts(harness):
    """A rollout whose synced CSV lands empty is exactly the 07-06
    compute-and-discard failure mode: abort + stop, never continue to the
    next link."""
    rows = _valid_12_rows(harness)
    tsv = harness.write_tsv(rows)
    proc = harness.run("rollouts", str(tsv), env={"STUB_EMPTY_CSV": "1"})
    assert proc.returncode != 0
    assert "missing/empty locally after sync" in (proc.stdout + proc.stderr)
    assert len(parse_calls(harness.calls_log)) == 1   # aborted on link 1
    assert stop_count(harness.calls_log) >= 1


# =========================================================== 10. pod not running

def test_rollouts_pod_not_running_refuses(harness):
    rows = _valid_12_rows(harness)
    tsv = harness.write_tsv(rows)
    harness.pod_state.write_text("stopped")
    proc = harness.run("rollouts", str(tsv))
    assert proc.returncode == 3
    assert "ensure_pod" in (proc.stdout + proc.stderr)
    assert parse_calls(harness.calls_log) == []
    assert stop_count(harness.calls_log) == 0
