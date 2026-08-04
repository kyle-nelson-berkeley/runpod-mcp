# runpod-mcp — CLAUDE.md

*Last updated: 2026-08-04 · Owner: Kyle*

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
one pod PER VEHICLE (unknown pod names refused), RTX 4090 only, Secure
Cloud, never interruptible, network volume required, idle watchdog, per-job
`auto_stop`.

**Two-vehicle scoping (2026-08-04):** the server manages TWO pods+volumes —
`hippocampus` (== the original `lts-replication` pod + `lts-replication`
volume; every default resolves here, so bare calls behave exactly as before)
and `bluerov2` (`lts-replication-bluerov2` + volume `bluerov2-lts`,
auto-created by name on first bring-up). Every tool takes `vehicle=`
(default `hippocampus`); `stop_pod`/`terminate_pod` REQUIRE it explicitly;
`launch_training` derives it from its training vehicle
(`tools.TRAINING_VEHICLE_TO_POD`: curee→hippocampus, bluerov2→bluerov2 —
the training axis and the pod-routing axis are deliberately separate);
`apply_bluerov_patches`/`axis_sanity_sweep` are bluerov2-pod by nature.
Per-vehicle config lives under `vehicles:` in `pod_defaults.yaml` (pod/volume
names, DC preference, known_hosts file, `local_log_dir` — hippocampus keeps
`logs/pod`, bluerov2 gets `logs/pod/bluerov2`). `spend_report` is
account-wide (all vehicles) — never attribute its total to one arm.

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
- **The durable JSON summary is the recovery contract**
  (`supervise-<job_id>.json` in the vehicle's log dir — `logs/pod/` for
  hippocampus, `logs/pod/bluerov2/` for bluerov2): a later session reads it
  to confirm whether the pod was stopped, so *safety never depends on the
  completion notification landing*.
- **Pod routing:** `--training <vehicle>` derives the pod via
  `TRAINING_VEHICLE_TO_POD` (an explicit `--vehicle` that conflicts is
  refused at argparse); `--job-name` mode defaults to hippocampus — pass
  `--vehicle bluerov2` to supervise a bluerov2-pod job. The summary records
  the resolved `pod_vehicle`.
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

### The `watch` CLI — advisory observation loop (also NOT a 15th tool)

`python -m runpod_mcp.watch` (wrapper: `watch.sh`) is the mid-run companion to
`supervise`: armed as a **second** `run_in_background` Bash task alongside the
supervise task, it discovers the single active job (poll `pod_status`
`active_jobs` under a bounded `--startup-grace`, default 180 s; `--job-id`
skips discovery for attended use), tails the job's `out.log` incrementally by
byte offset (`tail -c +OFFSET`, `--interval` default 60 s), parses the rsl_rl
console blocks into a full metric series (iteration/total, mean reward,
value/surrogate/entropy losses — extending the `latest_reward_line` idea in
`jobs.status`), snapshots a LISTING of the job dir (names/sizes/mtimes — never
downloads weights), prints one compact status line per interval, and maintains
a JSON status file (`--status-path`, default `watch-<job_id>.json` in the
vehicle's log dir; `--vehicle` selects the pod, default hippocampus).

- **The exit IS the page** (same idiom as `supervise.sh`), with distinct
  codes: **0** job completed OK · **3** plateau detected (exits at DETECTION
  time, while the job still runs — the whole point is paging early enough to
  intervene) · **4** failure evidence (traceback / CUDA OOM patterns, or
  pod-side `exit_code` ≠ 0) · **5** stall / lost contact (only after
  `--stall-sec` with NO terminal evidence) · **2** usage errors.
- **Strictly read-only, strictly advisory.** The watcher only ever runs
  `tail`/`ls`/`cat`-class commands on the pod; it can NOT stop or terminate
  anything (stop stays owned by supervise / deadman / humans, and
  `terminate_pod` stays human-approval-only). Its pages — including plateau —
  trigger no automated action. **`supervise`'s durable summary JSON remains
  the AUTHORITATIVE terminal record**; the watcher's status file is advisory.
- **Ordered exit decision tree** (avoids false-paging normal completion:
  supervise stops the pod right after the job ends, so a 60 s tail often
  catches the pod already gone on the happy path): before concluding
  stall/lost-contact it (a) attempts to read the pod-side
  `/workspace/jobs/<job_id>/exit_code` artifact (the same file `jobs.status`
  reads) and (b) checks the job's disappearance from `active_jobs`. "Gone +
  exit_code present" exits **0 or 4 by that code, never 5**; exit 5 is
  reserved for genuinely-unreachable-with-no-terminal-evidence.
- **HONESTY NOTE (unverified live):** the watcher's live-pod behavior is
  **UNVERIFIED** — it was built and verified entirely offline against
  captured job logs and mocked SSH (the GPU envelope was SPENT; zero pod
  launches). The `--plateau-window` / `--plateau-min-delta` / `--stall-sec`
  defaults are **heuristics derived from (and tested against) the same
  captured fixtures** — a circularity we state rather than paper over.
  Treat the first live run as the real acceptance test.

### The `deadman` CLI — per-vehicle since 2026-08-04

`deadman.sh arm`/`cancel` now REQUIRE an explicit `--vehicle` (a fuse is a
fire-and-forget stop action — a wrong implicit vehicle would be silently
wrong for hours); `status` takes it optionally and with NO `--vehicle`
reports ALL declared vehicles with worst-exit-code-wins (a LOST or
stop_failed fuse on EITHER arm exits 1 — one arm's trouble can never hide
behind the other's clean report). Artifacts live in the vehicle's log dir
(filenames unchanged: `deadman.pid`, `deadman-<ts>.json` — hippocampus keeps
`logs/pod/`, so pre-refactor summaries stay visible); one armed fuse per
vehicle. Usage: `./deadman.sh arm --vehicle hippocampus --hours 3.0 &`.
