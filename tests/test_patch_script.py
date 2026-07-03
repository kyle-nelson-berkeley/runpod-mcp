"""Unit 4: remote/apply_bluerov2_patch.py — exercised against a throwaway git
repo built from the committed fixture excerpts (canonical), plus a SHA-gated
verification against the real reference clone (READ-ONLY: tmp copy only)."""
import re
import shutil
import subprocess
import sys

import pytest

from tests.conftest import MCP_ROOT, REPO_ROOT

SCRIPT = MCP_ROOT / "runpod_mcp" / "remote" / "apply_bluerov2_patch.py"
FIXTURES = MCP_ROOT / "tests" / "fixtures"
THRUSTERS = REPO_ROOT / "patches" / "bluerov2_heavy_thrusters.py"
PIN = "7c5ebe7f7a08acd2570b5fba328e92b7f59f6794"
REFERENCE = REPO_ROOT / "reference" / "isaac-auv-env"


def make_repo(tmp_path, env_text=None, asset_text=None):
    """Throwaway git repo shaped like isaac-auv-env (fixture content)."""
    repo = tmp_path / "isaac-auv-env"
    (repo / "assets").mkdir(parents=True)
    (repo / "warpauv_env.py").write_text(
        env_text or (FIXTURES / "warpauv_env_excerpt_7c5ebe7.py").read_text())
    (repo / "assets" / "warpauv.py").write_text(
        asset_text or (FIXTURES / "assets_warpauv_7c5ebe7.py").read_text())
    def g(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True)
    g("init", "-q")
    g("config", "user.email", "t@t"); g("config", "user.name", "t")
    g("add", "-A"); g("commit", "-qm", "pristine")
    sha = g("rev-parse", "HEAD").stdout.strip()
    return repo, sha


def run_script(repo, sha, tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo),
         "--thrusters", str(THRUSTERS),
         "--markers", str(tmp_path / "markers"),
         "--expect-sha", sha, *extra],
        capture_output=True, text=True)


def test_patch_applies_all_sections(tmp_path):
    repo, sha = make_repo(tmp_path)
    proc = run_script(repo, sha, tmp_path)
    assert proc.returncode == 0, proc.stderr
    env = (repo / "warpauv_env.py").read_text()
    # §1 import swap
    assert "from .bluerov2_heavy_thrusters import get_thruster_com_and_orientations" in env
    assert "ConversionFunctionBasic, get_thruster_com_and_orientations" not in env
    # §2 all six dims flipped, none left behind
    assert "shape=(8,)" in env and "num_actions = 8" in env
    assert len(re.findall(r"torch\.zeros\(\(self\.num_envs, 8, 3\)", env)) == 2
    assert "DynamicsFirstOrder(self.num_envs, 8," in env
    assert "torch.zeros(self.num_envs, 8, device" in env
    # §3 physical params
    assert "volume = 0.0134" in env and "mass = 13.5" in env
    assert "inertia_tensors[:, 0] = 0.26" in env
    assert "inertia_tensors[:, 1] = 0.23" in env
    assert "inertia_tensors[:, 2] = 0.37" in env
    # §4 Small DR
    assert "com_to_cob_offset_radius = 0.0164285714" in env
    assert "volume_range = [0.0125164, 0.0142836]" in env
    # commented-out No-DR preset untouched
    assert "# com_to_cob_offset_radius = 0 #" in env
    # §5 mass_props inserted after rigid_props block
    asset = (repo / "assets" / "warpauv.py").read_text()
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=13.5)," in asset
    assert asset.index("mass_props") > asset.index("enable_gyroscopic_forces")
    # drop-in installed + marker written
    assert (repo / "bluerov2_heavy_thrusters.py").exists()
    marker = tmp_path / "markers" / "bluerov2_patch_applied"
    assert marker.exists() and sha in marker.read_text()


def test_patch_is_idempotent_via_pristine_restore(tmp_path):
    repo, sha = make_repo(tmp_path)
    assert run_script(repo, sha, tmp_path).returncode == 0
    env_first = (repo / "warpauv_env.py").read_text()
    proc = run_script(repo, sha, tmp_path)          # second run: restore+reapply
    assert proc.returncode == 0, proc.stderr
    assert (repo / "warpauv_env.py").read_text() == env_first


def test_patch_defuses_prior_curee_dr_edits(tmp_path):
    # a launch_training("curee", DR_0) edit must not break the patch
    from runpod_mcp import training
    pristine = (FIXTURES / "warpauv_env_excerpt_7c5ebe7.py").read_text()
    repo, sha = make_repo(tmp_path)
    (repo / "warpauv_env.py").write_text(
        training.apply_dr_to_source(pristine, "curee", "DR_0"))
    proc = run_script(repo, sha, tmp_path)
    assert proc.returncode == 0, proc.stderr        # pristine-restore defused it
    assert "com_to_cob_offset_radius = 0.0164285714" in \
        (repo / "warpauv_env.py").read_text()


def test_patch_refuses_wrong_sha(tmp_path):
    repo, _ = make_repo(tmp_path)
    proc = run_script(repo, PIN, tmp_path)          # tmp repo HEAD != real pin
    assert proc.returncode == 1
    assert "expected" in proc.stderr


def test_patch_aborts_on_anchor_mismatch(tmp_path):
    broken = (FIXTURES / "warpauv_env_excerpt_7c5ebe7.py").read_text().replace(
        "num_actions = 6", "num_actions = 12")      # unrecognized state
    repo, sha = make_repo(tmp_path, env_text=broken)
    proc = run_script(repo, sha, tmp_path)
    assert proc.returncode == 1
    assert "anchor mismatch" in proc.stderr
    # nothing half-applied: import swap must NOT have been committed to disk
    assert "bluerov2_heavy_thrusters import" not in (repo / "warpauv_env.py").read_text()


def test_dry_run_writes_nothing(tmp_path):
    repo, sha = make_repo(tmp_path)
    before = (repo / "warpauv_env.py").read_text()
    proc = run_script(repo, sha, tmp_path, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert (repo / "warpauv_env.py").read_text() == before
    assert not (tmp_path / "markers" / "bluerov2_patch_applied").exists()
    assert not (repo / "bluerov2_heavy_thrusters.py").exists()


# ------------------------------- SHA-gated verification vs the real checkout

def _reference_at_pin() -> bool:
    if not (REFERENCE / ".git").exists():
        return False
    proc = subprocess.run(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() == PIN


@pytest.mark.skipif(not _reference_at_pin(),
                    reason=f"reference/isaac-auv-env absent or not at {PIN[:12]}")
def test_patch_against_real_reference_copy(tmp_path):
    """READ-ONLY on the reference clone: full file copied to tmp, patched there."""
    repo, sha = make_repo(
        tmp_path,
        env_text=(REFERENCE / "warpauv_env.py").read_text(),
        asset_text=(REFERENCE / "assets" / "warpauv.py").read_text())
    proc = run_script(repo, sha, tmp_path)
    assert proc.returncode == 0, proc.stderr
    env = (repo / "warpauv_env.py").read_text()
    assert "num_actions = 8" in env
    assert "com_to_cob_offset_radius = 0.0164285714" in env
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=13.5)," in \
        (repo / "assets" / "warpauv.py").read_text()
