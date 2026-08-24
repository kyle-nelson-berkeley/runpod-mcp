# runpod-mcp — custom MCP server for the Learning-to-Swim replication

> **Standalone note:** this server was extracted (full history) from the
> `learning-to-swim-replication` project. Relative links like
> `../runbook/RUNBOOK.md` refer to that parent project and only resolve when
> this repo sits inside it (or is symlinked there); the server itself runs
> standalone.

Task-shaped tools (14) mirroring the parent project's `runbook/RUNBOOK.md`
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
                                            guardrails.py one-pod-per-vehicle (unknown refused) · 4090-only · no spot · volume required · confirm gate
                                            ssh.py        hardened ssh/scp/rsync; known_hosts_runpod; 60s conn cache
                                            jobs.py       detached jobs: /workspace/jobs/<id>/{cmd.sh,pid,out.log,exit_code,meta.json}
                                            training.py   DR tables (RUNBOOK/yaml-cross-checked) + verbatim train cmd
                                            supervise.py  Mac-side background CLI: launch→poll→pull→sync→spend→stop (reuses tools.*)
                                            watch.py      Mac-side ADVISORY observation CLI: discover job→tail out.log→parse metrics→page on plateau/failure/stall (read-only; never stops pods)
                                            remote/       job_wrapper.sh · idle_watchdog.sh · apply_bluerov2_patch.py
                                            deadman.py    Mac-side stop-pod fuse: arm --vehicle → sleep → stop with retries (per-vehicle pid/summaries)
supervise.sh → caffeinate -i wrapper around  python -m runpod_mcp.supervise
watch.sh     → caffeinate -i wrapper around  python -m runpod_mcp.watch   (live-pod behavior UNVERIFIED — fixture/mock-verified only; see CLAUDE.md §D)
deadman.sh   → caffeinate -i wrapper around  python -m runpod_mcp.deadman (arm/cancel REQUIRE --vehicle; bare status reports all vehicles)
```

- **Stateless & per-vehicle**: "the pod" = whatever `GET /pods` returns
  matching the selected vehicle's configured name (`hippocampus` →
  `lts-replication`, `bluerov2` → `lts-replication-bluerov2`; every tool's
  `vehicle` param defaults to hippocampus, `stop_pod`/`terminate_pod` require
  it explicitly); console and MCP always agree. Only local state: a
  60-second (host, port) cache per vehicle Runtime.
- **Async jobs**: one SSH call runs `setsid bash job_wrapper.sh <dir> <pod_id>
  <ceiling> <auto_stop>`; state lives on the network volume, so it survives
  MCP restarts, Mac sleep, and pod stop. `timeout --kill-after` enforces
  wall-clock ceilings (exit 124); the auto-stop suffix runs AFTER exit_code
  is written, so a timeout can never defeat it. Pod id is argv-injected
  (container env vars are unreliable in detached BatchMode shells);
  `/etc/rp_environment` is sourced for runpodctl credentials; arming
  auto_stop probes runpodctl synchronously and fails loudly if it can't work.
  The probe (2026-08-09) is a three-way diagnostic: the bare-shell checks
  decide nothing (they answer H1-vs-H2 and capture the bare PATH),
  `/etc/rp_environment` is then sourced unconditionally, and the SOURCED
  pair carries the verdict — `NO_RUNPODCTL` (binary absent even after
  sourcing, exit 90), `NO_RUNPODCTL_AUTH_SOURCED` (still refused after
  sourcing, exit 91), `PROBE_OK` (sourced success only;
  `NO_RUNPODCTL_AUTH_BARE` is the mid-stream diagnostic that continues).
- **Idle watchdog**: reinstalled on every transition-to-running — the
  container-disk wipe removes runtime-installed material (`idle_watchdog.sh`
  itself, the apt X11/GL libs, `rsync`), which is why install-on-every-
  transition stays; `runpodctl` is IMAGE-SHIPPED and back on every boot (a
  wipe restores the disk from the image, it does not empty it — corrected
  2026-08-09). Every 5 min: no live job pid + no sshd session +
  `/workspace/.keepalive` older than 60 min → `runpodctl stop pod`.
  `touch /workspace/.keepalive` is the manual-session escape hatch.
  A successful install reports **`armed (stop path unverified)`** — the probe
  certifies READ (`get pod`), the watchdog needs WRITE (`stop pod`); the
  first real confirmation is a successful-stop entry in
  `/workspace/.idle_watchdog.log`. Status (2026-08-09): the install probe has
  failed on every recorded bring-up (opaque rc=91 pre-fix) — the watchdog has
  never yet armed; defect 2 ships DIAGNOSED, not CLOSED, and the next
  bring-up's sentinel settles it. `idle_watchdog: FAILED` ⇒ arm the Mac-side
  deadman before any job.
- **Guardrails are code**: one pod per declared vehicle (any other pod name
  on the account is refused), RTX 4090 ×1, SECURE, interruptible forced
  false, network volume required, `terminate_pod` needs an explicit
  `vehicle` plus the verbatim string `terminate <that vehicle's pod_name>`
  (e.g. `terminate lts-replication`), one job at a time per pod absent
  `force`.

## Install / registration

Register the server in a project's `.mcp.json` (Claude Code) with an
absolute path to `run.sh` — `run.sh` bootstraps its own `.venv` on first
launch:

```json
{
  "mcpServers": {
    "runpod": {
      "command": "bash",
      "args": ["/path/to/runpod-mcp/run.sh"]
    }
  }
}
```

## Setup

1. **API key** (never on disk/git/argv — macOS Keychain only; the server
   reads it via `security find-generic-password` and scrubs `rpa_` values
   from every error and log):

   ```
   security add-generic-password -a "$USER" -s runpod-api-key -w '<KEY>'
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
parsing [BLUEROV2/config/bluerov2_heavy.yaml](../BLUEROV2/config/bluerov2_heavy.yaml),
[RUNBOOK.md](../runbook/RUNBOOK.md) and [APPLY.md](../BLUEROV2/patches/APPLY.md);
the patch script is exercised against committed fixture excerpts of the
pinned `7c5ebe7` sources (plus a SHA-gated test against the real reference
clone when present — read-only, tmp copies).

`test_supervise.py` drives the `supervise` CLI's core with injected fakes +
a fake clock (no real waiting), covering every safety branch: normal
completion, job failure, max-wait force-stop, pod-not-running refusal, launch
refusal, transient poll errors, capture-failure-still-stops, `--no-stop`, and
`terminate_pod` is asserted never-called in every case.

Root-repo `pytest -q` ignores this folder (`conftest.py` `collect_ignore`) —
the lean root venv has no `mcp`/`httpx`.

## Supervised runs (`supervise.sh`)

One command that chains an entire run — verify-pod-running → dry-run-derive a
**finite** wall-clock cap → `launch(auto_stop=false)` → poll `job_status` →
unconditionally pull `/workspace/jobs/<job_id>/` + `sync_logs` +
`spend_report` → `stop_pod` → durable JSON summary — so the agent fires it
**once as a background task** and is notified on completion. It reuses
`runpod_mcp.tools.*` (no logic duplication, all guardrails inherited) and
never calls `terminate_pod`. This is a Mac-side CLI, **not** a 15th MCP tool:
a poll-for-minutes tool would block the stdio server.

```
# training run (background task)
supervise.sh --training curee --dr DR_0 --seed 1 \
    [--interval 45] [--max-wait N] [--backstop 300] [--no-stop] \
    [--sync-subdir rsl_rl/warpauv_direct] [--summary-path PATH]

# generic job — --sync-subdir REQUIRED (pass 'none' to skip the analysis sync;
# the job-dir pull always happens); --vehicle routes the pod (default
# hippocampus; --training mode derives it from the training vehicle instead)
supervise.sh --job-name eval --command "…" --workdir /workspace \
    --sync-subdir <dir|none> [--max-runtime-sec N] [--vehicle bluerov2]
```

Money-safety: the poll loop has exactly two exits — normal completion →
`stop_pod`; or `--max-wait` (always finite) elapsed while still `running` →
force-stop + non-zero exit + `force_stopped` summary flag. A launch *refusal*
→ no stop (fix and retry), exit 2. The `supervise-<job_id>.json` summary in
the vehicle's log dir (`logs/pod/` hippocampus, `logs/pod/bluerov2/`
bluerov2) is the recovery contract (a later session reconciles stop state
from it). Liveness caveats: `caffeinate -i` guards idle sleep but not lid-close;
`run_in_background` survival across WarmLifecycle reaping is unverified — the
job's `timeout` ceiling is the guaranteed backstop; the pod-side idle watchdog
would back it up but has never yet armed on a recorded bring-up (DIAGNOSED,
not CLOSED — see the Idle-watchdog bullet), so arm the Mac-side deadman when
`ensure_pod` reports `idle_watchdog: FAILED`.

## Campaign chains (`CUREE/chains/`)

One bash script per campaign (named by campaign ID, e.g.
`chain-011-CUREE_Adaptive-weights.sh`): the campaign's whole pod-side job
sequence — patches, gates, trainings, evals, syncs — as ordered, sha-pinned
links. Chains are launched through `supervise.sh` (which owns
capture-and-stop), never hand-driven; they are the durable record of exactly
what a campaign executed.

## Dry runs

`ensure_pod`, `run_pod_setup`, `run_job`, `launch_training`,
`apply_bluerov_patches` all take `dry_run=true` and return the exact would-be
payloads/edits/commands without mutating anything ($0). `supervise` uses this
dry-run path to derive its finite `--max-wait` before the real launch.

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

## License

MIT — see [LICENSE](LICENSE).
