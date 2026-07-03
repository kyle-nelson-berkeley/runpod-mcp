"""Unit 1: config.py — pod_defaults.yaml loading, Keychain fetch, key scrubbing."""
import subprocess

import pytest

from runpod_mcp import config


# ------------------------------------------------------------------ defaults

def test_load_defaults_reads_committed_yaml():
    cfg = config.load_defaults()
    assert cfg["pod_name"] == "lts-replication"
    assert cfg["gpu_type_ids"] == ["NVIDIA GeForce RTX 4090"]
    assert cfg["cloud_type"] == "SECURE"
    assert cfg["interruptible"] is False
    assert cfg["container_disk_gb"] == 30
    assert cfg["network_volume_gb"] == 60
    assert cfg["volume_mount_path"] == "/workspace"
    assert cfg["ports"] == ["22/tcp"]
    assert cfg["budget_usd"] == 50
    assert cfg["idle_minutes"] == 60


def test_defaults_have_runtime_ceilings():
    cfg = config.load_defaults()
    t = cfg["timeouts"]
    assert t["setup_sec"] == 5400          # first pod_setup run: 30-60 min
    assert t["training_sec"] == 3600       # 10-20 min training + margin
    assert t["sweep_sec"] == 3600
    assert t["job_sec"] == 3600
    assert t["exec_max_sec"] == 600        # exec_on_pod hard ceiling


def test_defaults_never_contain_secrets():
    text = (config.DEFAULTS_PATH).read_text()
    assert "rpa_" not in text


# ------------------------------------------------------------------ keychain

def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_fetch_api_key_uses_keychain_argv(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _completed(stdout="rpa_TESTKEY123\n")

    key = config.fetch_api_key(run=fake_run)
    assert key == "rpa_TESTKEY123"
    assert calls == [["security", "find-generic-password",
                      "-a", "kyle", "-s", "runpod-api-key", "-w"]]


def test_fetch_api_key_missing_fails_fast_with_instructions():
    def fake_run(argv, **kwargs):
        return _completed(returncode=44, stderr="could not be found")

    with pytest.raises(config.ConfigError) as exc:
        config.fetch_api_key(run=fake_run)
    msg = str(exc.value)
    assert "security add-generic-password" in msg  # tells Kyle how to add it
    assert "runpod-api-key" in msg


def test_fetch_api_key_empty_output_fails():
    def fake_run(argv, **kwargs):
        return _completed(stdout="\n")

    with pytest.raises(config.ConfigError):
        config.fetch_api_key(run=fake_run)


# ------------------------------------------------------------------ scrubber

def test_scrub_redacts_api_keys():
    dirty = "boom: Bearer rpa_ABCdef0123XYZ and rpa_Z9 too"
    clean = config.scrub(dirty)
    assert "rpa_ABCdef0123XYZ" not in clean
    assert "rpa_Z9" not in clean
    assert clean.count("rpa_[REDACTED]") == 2


def test_scrub_handles_non_string():
    assert "rpa_[REDACTED]" in config.scrub(ValueError("bad rpa_KEY1"))


def test_scrub_leaves_clean_text_alone():
    assert config.scrub("no secrets here") == "no secrets here"


# ------------------------------------------------------------- ssh identity

def test_ssh_public_key_read(monkeypatch, tmp_path):
    ident = tmp_path / "id_ed25519"
    ident.write_text("PRIVATE")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test@mac\n")
    cfg = config.load_defaults()
    cfg["ssh_identity"] = str(ident)
    assert config.read_ssh_public_key(cfg) == "ssh-ed25519 AAAA test@mac"


def test_ssh_public_key_missing_fails_with_path(tmp_path):
    cfg = config.load_defaults()
    cfg["ssh_identity"] = str(tmp_path / "nope")
    with pytest.raises(config.ConfigError) as exc:
        config.read_ssh_public_key(cfg)
    assert "nope.pub" in str(exc.value)
