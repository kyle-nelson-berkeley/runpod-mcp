"""Configuration + secrets for the runpod MCP server.

The RunPod API key lives ONLY in the macOS Keychain and is fetched here at
startup via `security`(1). It is held in memory, never written to disk, git,
argv of remote commands, or logs — and every error/log string that could
carry it is passed through scrub().
"""
import re
import subprocess
from pathlib import Path

import yaml

MCP_ROOT = Path(__file__).resolve().parents[1]         # runpod-mcp/
REPO_ROOT = MCP_ROOT.parent                            # repo root
DEFAULTS_PATH = MCP_ROOT / "pod_defaults.yaml"

KEYCHAIN_ARGV = ["security", "find-generic-password",
                 "-a", "kyle", "-s", "runpod-api-key", "-w"]

_KEY_RE = re.compile(r"rpa_[A-Za-z0-9]+")


class ConfigError(RuntimeError):
    """Fatal configuration problem (missing key, missing files)."""


def scrub(text) -> str:
    """Redact anything shaped like a RunPod API key from error/log text."""
    return _KEY_RE.sub("rpa_[REDACTED]", str(text))


def load_defaults(path: Path | None = None) -> dict:
    """Load the committed pod spec RAW (no secrets live there).

    Raw = shared top-level keys + the `vehicles:` map. Tools consume the
    per-vehicle FLAT view from merged_vehicle_cfg(), never this dict directly.
    """
    p = Path(path) if path else DEFAULTS_PATH
    with open(p) as fh:
        return yaml.safe_load(fh)


def merged_vehicle_cfg(raw: dict, vehicle: str) -> dict:
    """Flatten the raw spec for one vehicle: shared keys + that vehicle's overlay.

    The `vehicles` map itself is dropped — the result is exactly the flat shape
    the tools consumed before per-vehicle scoping existed.
    """
    vehicles = raw.get("vehicles") or {}
    if vehicle not in vehicles:
        raise ConfigError(
            f"unknown vehicle {vehicle!r} — pod_defaults.yaml declares: "
            f"{', '.join(sorted(vehicles))}")
    merged = {k: v for k, v in raw.items() if k != "vehicles"}
    merged.update(vehicles[vehicle])
    return merged


def declared_pod_names(raw: dict) -> set:
    """Every pod name this config declares — the union guardrails allow."""
    return {v["pod_name"] for v in (raw.get("vehicles") or {}).values()}


def fetch_api_key(run=subprocess.run) -> str:
    """Fetch the API key from the macOS Keychain. Fail fast with instructions."""
    add_help = ("API key not found in Keychain. Add it with:\n"
                "  security add-generic-password -a kyle -s runpod-api-key -w '<KEY>'\n"
                "(key never goes on disk or in git — Keychain only)")
    try:
        proc = run(KEYCHAIN_ARGV, capture_output=True, text=True, timeout=10)
    except Exception as exc:  # security(1) missing, timeout, ...
        raise ConfigError(f"Keychain lookup failed: {scrub(exc)}\n{add_help}") from None
    if proc.returncode != 0:
        raise ConfigError(f"Keychain lookup failed (rc={proc.returncode}): "
                          f"{scrub(proc.stderr).strip()}\n{add_help}")
    key = proc.stdout.strip()
    if not key:
        raise ConfigError(f"Keychain returned an empty key.\n{add_help}")
    return key


def read_ssh_public_key(cfg: dict) -> str:
    """Read the .pub side of the configured SSH identity (for SSH_PUBLIC_KEY env)."""
    pub = Path(cfg["ssh_identity"]).expanduser().with_name(
        Path(cfg["ssh_identity"]).name + ".pub")
    if not pub.exists():
        raise ConfigError(f"SSH public key not found: {pub} — "
                          "generate one with ssh-keygen -t ed25519")
    return pub.read_text().strip()
