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
