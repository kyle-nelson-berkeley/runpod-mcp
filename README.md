# runpod-mcp — custom MCP server for the Learning-to-Swim replication

Task-shaped tools (14) mirroring [runbook/RUNBOOK.md](../runbook/RUNBOOK.md)
instead of ~50 generic API mirrors. Custom because **no RunPod API executes
commands on a pod** — the official MCP covers only the control plane; running
pod_setup.sh, the axis sanity sweep, and training needs SSH + rsync, encoded
here with cost guardrails in code.

## Architecture

```
.mcp.json → run.sh (venv bootstrap) → server.py (FastMCP, stdio; thin)
                                        └── runpod_mcp/
                                            config.py     Keychain key fetch + rpa_ scrubber
                                            api.py        REST v1 (pods/volumes/billing) + unauth GraphQL gpuTypes
                                            guardrails.py one-pod max · 4090-only · no spot · volume required · confirm gate
                                            ssh.py        hardened ssh/scp/rsync; known_hosts_runpod; 60s conn cache
                                            jobs.py       detached jobs: /workspace/jobs/<id>/{cmd.sh,pid,out.log,exit_code,meta.json}
                                            training.py   DR tables (RUNBOOK/yaml-cross-checked) + verbatim train cmd
                                            remote/       job_wrapper.sh · idle_watchdog.sh · apply_bluerov2_patch.py
```

- **Stateless**: "the pod" = whatever `GET /pods` returns named
  `lts-replication`; console and MCP always agree. Only local state: a
  60-second (host, port) cache.
- **Async jobs**: one SSH call runs `setsid bash job_wrapper.sh <dir> <pod_id>
  <ceiling> <auto_stop>`; state lives on the network volume, so it survives
  MCP restarts, Mac sleep, and pod stop. `timeout --kill-after` enforces
  wall-clock ceilings (exit 124); the auto-stop suffix runs AFTER exit_code
  is written, so a timeout can never defeat it. Pod id is argv-injected
  (container env vars are unreliable in detached BatchMode shells);
  `/etc/rp_environment` is sourced for runpodctl credentials; arming
  auto_stop probes runpodctl synchronously and fails loudly if it can't work.
- **Idle watchdog**: reinstalled on every transition-to-running (container
  disk wipes on stop). Every 5 min: no live job pid + no sshd session +
  `/workspace/.keepalive` older than 60 min → `runpodctl stop pod`.
  `touch /workspace/.keepalive` is the manual-session escape hatch.
- **Guardrails are code**: one pod max, RTX 4090 ×1, SECURE, interruptible
  forced false, network volume required, `terminate_pod` needs the verbatim
  string `terminate lts-replication`, one job at a time absent `force`.

## Setup

1. **API key** (never on disk/git/argv — Keychain only):

   ```
   security add-generic-password -a kyle -s runpod-api-key -w '<KEY>'
   ```

2. **SSH key**: `~/.ssh/id_ed25519(.pub)` must exist; the `.pub` is injected
   at pod-create via the `PUBLIC_KEY` env var (what `runpod/pytorch` images
   actually honor — live-verified; `SSH_PUBLIC_KEY` also set as
   belt-and-braces). Direct SSH to `root@publicIp:portMappings["22"]`;
   RunPod's proxy SSH is unused (no scp). Host keys land in a dedicated
   `~/.ssh/known_hosts_runpod`, truncated on every pod start (the container
   disk wipe regenerates host keys, stale entries only cause false MITM
   failures).

3. Nothing else — `run.sh` creates `.venv/` and installs
   [requirements.txt](requirements.txt) on first launch (stamp-gated).

## Testing

```
runpod-mcp/.venv/bin/python -m pytest runpod-mcp/tests -q          # offline (default)
RUNPOD_MCP_LIVE=1 runpod-mcp/.venv/bin/python -m pytest \
    runpod-mcp/tests/test_live.py -q                               # live $0 read-only
```

Offline tests use `httpx.MockTransport` + duck-typed fake SSH — no network,
no key. Live tests are read-only GETs + an MCP stdio handshake through
`run.sh` (asserts all 14 tools register). DR tables are cross-checked by
parsing [config/bluerov2_heavy.yaml](../config/bluerov2_heavy.yaml),
[RUNBOOK.md](../runbook/RUNBOOK.md) and [APPLY.md](../patches/APPLY.md);
the patch script is exercised against committed fixture excerpts of the
pinned `7c5ebe7` sources (plus a SHA-gated test against the real reference
clone when present — read-only, tmp copies).

Root-repo `pytest -q` ignores this folder (`conftest.py` `collect_ignore`) —
the lean root venv has no `mcp`/`httpx`.

## Dry runs

`ensure_pod`, `run_pod_setup`, `run_job`, `launch_training`,
`apply_bluerov_patches` all take `dry_run=true` and return the exact would-be
payloads/edits/commands without mutating anything ($0).

## NGC fallback image (manual swap — read first)

`nvcr.io/nvidia/isaac-sim:4.5.0` (RUNBOOK Day-1 fallback) has **no sshd** —
it breaks this server's entire SSH story. Switching requires a docker-start
command that installs/launches sshd (not a one-line change): flag to Kyle
before ever swapping `image_name` in [pod_defaults.yaml](pod_defaults.yaml).

## Known risks (accepted at plan time)

- The IsaacSim 4.5.0 download URL in pod_setup.sh may 404 — surfaces in
  `job_status` log tail; the fix is a runbook edit, not an MCP change.
- 4090 stock fluctuates per DC; the network volume pins one DC.
  `gpu_availability(data_center_id=...)` + ensure_pod's no-GPU recovery
  recipe cover it; worst case, create a second volume in another DC.
