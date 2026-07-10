# runpod-mcp — CLAUDE.md

*Last updated: 2026-07-06 · Owner: Kyle*

Project context, operating rules, and the runbook-day → tool map live in the
**root [CLAUDE.md](../CLAUDE.md)** — read that first. This file covers only
this subfolder.

## A · What this folder is

A self-contained **FastMCP server (14 tools)** that lets agents drive the
entire Learning-to-Swim GPU runbook on RunPod: pod lifecycle
(`ensure_pod`/`stop_pod`/`terminate_pod`), setup (`run_pod_setup`), detached
training jobs (`launch_training`, `run_job`, `job_status`), BlueROV2
patching (`apply_bluerov_patches`, `axis_sanity_sweep`), plus
`sync_logs` / `spend_report` / `pod_status` / `gpu_availability` /
`exec_on_pod`. Cost guardrails are enforced **in code**, not by convention:
one pod max, RTX 4090 only, Secure Cloud, never interruptible, network
volume required, idle watchdog, per-job `auto_stop`.

## B · Why it exists

Manual REST+SSH pod driving (Day 1) was error-prone and burned money on
idle time; this server makes the runbook agent-drivable with the expensive
mistakes made impossible. Done = Days 2–7 run through these tools without
a human touching SSH.

## C · Stack & layout

- Python + FastMCP; RunPod REST v1 + GraphQL client; paramiko SSH
- **API key:** macOS Keychain (`runpod-api-key`) only — never echoed,
  written, or passed as an argument
- **Wired via** root `.mcp.json`; entry point `run.sh` → `server.py`
- `runpod_mcp/` — implementation; `pod_defaults.yaml` — pod spec +
  guardrail constants; `tests/` — the suite
- **Run tests:** `.venv/bin/python -m pytest tests -q` (from this folder;
  venv is gitignored — recreate with `python3 -m venv .venv &&
  .venv/bin/pip install -r requirements.txt`)
- Jobs are detached by design: pod-side nohup wrapper writes
  status/log files under `/workspace`; `job_status` polls them. Never
  add a blocking tool.

## D · The `supervise` CLI (a background command, NOT a 15th tool)

`python -m runpod_mcp.supervise` (wrapper: `supervise.sh`) chains the whole
run-a-job dance — verify-pod-running → derive a finite wall-clock cap →
`launch(auto_stop=false)` → poll `job_status` → **unconditionally pull the
job's own `/workspace/jobs/<job_id>/` dir** + `sync_logs` + `spend_report`
→ `stop_pod` → durable JSON summary — into **one command the agent fires
once (as a `run_in_background` Bash task) and is notified about on
completion**. The tool surface stays **14** on purpose: a poll-for-minutes
tool would block the stdio server ("never add a blocking tool" above), so
this lives Mac-side as a background CLI reusing `tools.*` — same guardrails,
zero logic duplication.

- **`auto_stop=false` always**: `auto_stop=true` stops the pod the instant
  the job ends, *before* `sync_logs` can run — so the supervisor owns the
  stop and sync-then-stops itself.
- **Money-safety = two exits, never a third**: normal completion →
  `stop_pod()`; or `--max-wait` elapsed while still `running` → ANOMALY →
  best-effort sync + `stop_pod(force=True)` + a `force_stopped` summary flag
  + non-zero exit. `--max-wait` is ALWAYS finite (dry-run derives
  `max_runtime_sec` from `pod_defaults.yaml` + a 300 s backstop) and the poll
  sleep is bounded by the remaining time, so the deadline is a hard cap on the
  *decision to stop*. A launch *refusal* (bad vehicle/dr, one-job guard,
  vehicle gate) → **no stop** (the agent fixes and retries; a stop would force
  a ~5-min `ensure_pod`), exit 2. **Never `terminate_pod`.**
  - *Residual (accepted, by design):* in the ANOMALY path the best-effort
    capture (rsync pull + `sync_logs`) runs *before* the force-stop — the
    DATA-CAPTURE directive wants a timed-out run's partial logs, and by the
    deadline the job is normally already past its own pod-side
    `timeout --kill-after` ceiling. If SSH is hung, that capture can delay the
    force-stop up to the rsync timeout; the stop still always follows (bounded,
    never unbounded) and the pod-side idle watchdog is the hard backstop.
- **The durable JSON summary is the recovery contract** (`logs/pod/
  supervise-<job_id>.json`): a later session reads it to confirm whether the
  pod was stopped, so *safety never depends on the completion notification
  landing*.
- **Liveness caveats (documented, not solved):** `supervise.sh` wraps in
  `caffeinate -i`, which holds off *idle* sleep only — it does **not** stop
  clamshell/lid-close sleep on battery. And whether a `run_in_background`
  Bash task survives WarmLifecycle idle-reaping of its parent session is
  **unverified**. Either way the *safety* outcome is held by the pod-side
  idle watchdog + the job's own `timeout --kill-after` ceiling; only the
  notification ergonomic degrades.
- `--sync-subdir` is REQUIRED in `--job-name` mode (pass `none` to skip the
  analysis sync — the job-dir pull still happens regardless); `--training`
  defaults it to `rsl_rl/warpauv_direct`.
