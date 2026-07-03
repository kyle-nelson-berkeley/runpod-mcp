"""Unit 4: training.py — DR tables cross-checked against the source-of-truth
files (config/bluerov2_heavy.yaml, RUNBOOK.md, APPLY.md), verbatim train
command, content-anchored DR edits with expected-current-value assertion."""
import re

import pytest
import yaml

from runpod_mcp import training
from tests.conftest import MCP_ROOT, REPO_ROOT

FIXTURE = (MCP_ROOT / "tests" / "fixtures" / "warpauv_env_excerpt_7c5ebe7.py").read_text()


# ----------------------------------------------- DR tables vs source of truth

def test_bluerov2_table_matches_config_yaml():
    cfg = yaml.safe_load((REPO_ROOT / "config" / "bluerov2_heavy.yaml").read_text())
    rec = cfg["domain_randomization"]["recommended"]
    t = training.DR_TABLES["bluerov2"]
    assert float(t["DR_1"]["com_to_cob_offset_radius"]) == \
        rec["small"]["com_to_cob_offset_radius"]
    assert [float(x) for x in t["DR_1"]["volume_range"].strip("[]").split(",")] == \
        rec["small"]["volume_range"]
    assert float(t["DR_2"]["com_to_cob_offset_radius"]) == \
        rec["large"]["com_to_cob_offset_radius"]
    assert [float(x) for x in t["DR_2"]["volume_range"].strip("[]").split(",")] == \
        rec["large"]["volume_range"]
    # DR_0: no offset noise, degenerate range at the BlueROV2 nominal volume
    assert float(t["DR_0"]["com_to_cob_offset_radius"]) == 0
    assert [float(x) for x in t["DR_0"]["volume_range"].strip("[]").split(",")] == \
        [cfg["vehicle"]["volume"]] * 2


def test_curee_table_matches_runbook_table():
    runbook = (REPO_ROOT / "runbook" / "RUNBOOK.md").read_text()
    rows = {}
    for level in ["DR_0", "DR_1", "DR_2"]:
        m = re.search(rf"\| {level} [^|]*\| `([^`]+)` \| `(\[[^`]+\])`", runbook)
        assert m, f"RUNBOOK.md CUREE DR table row for {level} not found"
        rows[level] = m.groups()
    t = training.DR_TABLES["curee"]
    for level, (radius, vrange) in rows.items():
        assert t[level]["com_to_cob_offset_radius"] == radius
        assert t[level]["volume_range"] == vrange


def test_bluerov2_table_matches_apply_md_section4():
    apply_md = (REPO_ROOT / "patches" / "APPLY.md").read_text()
    sec4 = apply_md.split("## 4.")[1].split("## 5.")[0]
    t = training.DR_TABLES["bluerov2"]
    for level, alias in [("DR_0", "No DR"), ("DR_1", "Small DR"), ("DR_2", "Large DR")]:
        m = re.search(rf"# {alias} \({level}\)\ncom_to_cob_offset_radius = (\S+)\n"
                      rf"volume_range = (\[[^\]]+\])", sec4)
        assert m, f"APPLY.md §4 block for {level} not found"
        assert t[level]["com_to_cob_offset_radius"] == m.group(1)
        assert t[level]["volume_range"] == m.group(2)


def test_level_aliases():
    for alias in ["DR_1", "dr_1", "small", "Small"]:
        assert training.canonical_level(alias) == "DR_1"
    assert training.canonical_level("none") == "DR_0"
    assert training.canonical_level("large") == "DR_2"
    with pytest.raises(training.TrainingError, match="dr_level"):
        training.canonical_level("huge")


# ------------------------------------------------------------- train command

def test_train_command_is_verbatim_runbook():
    runbook = (REPO_ROOT / "runbook" / "RUNBOOK.md").read_text()
    block = re.search(r"```bash\ncd /workspace/IsaacLab\n(\./isaaclab\.sh[^`]+?)```",
                      runbook, re.S)
    assert block, "RUNBOOK train command block not found"
    want = " ".join(block.group(1).replace("\\\n", " ").split())
    want = want.replace("<1|2|3>", "2")
    assert training.build_train_command(seed=2) == want
    assert training.ISAACLAB_DIR == "/workspace/IsaacLab"   # the cd line


def test_train_command_extra_args_appended():
    cmd = training.build_train_command(seed=1, extra_args="--video --video_length 200")
    assert cmd.endswith("--seed 1 --video --video_length 200")


# ------------------------------------------- content-anchored DR source edits

def test_apply_dr_pristine_curee_to_dr0():
    out = training.apply_dr_to_source(FIXTURE, "curee", "DR_0")
    assert "com_to_cob_offset_radius = 0 # uniform from sphere" in out
    assert "volume_range = [0.022747843530591776, 0.022747843530591776] # uniform" in out
    # the commented-out No-DR preset lines above stay untouched
    assert "# com_to_cob_offset_radius = 0 #" in out
    # mass_range untouched
    assert "mass_range = [2.2701e+01,2.2701e+01]" in out


def test_apply_dr_roundtrip_between_levels():
    dr1 = training.apply_dr_to_source(FIXTURE, "curee", "DR_1")
    assert "com_to_cob_offset_radius = 0.025 #" in dr1
    back = training.apply_dr_to_source(dr1, "curee", "DR_2")   # current=DR_1 is known
    assert "com_to_cob_offset_radius = 0.05 #" in back
    assert "volume_range = [0.019747843530591773, 0.02574784353059178] #" in back


def test_apply_dr_wrong_vehicle_aborts_with_diff():
    # pristine checkout carries CUREE values -> bluerov2 edit must refuse
    with pytest.raises(training.TrainingError) as exc:
        training.apply_dr_to_source(FIXTURE, "bluerov2", "DR_1")
    msg = str(exc.value)
    assert "0.05" in msg                 # what it found
    assert "expected one of" in msg      # what it wanted (abort-with-diff)


def test_apply_dr_unrecognized_current_value_aborts():
    mangled = FIXTURE.replace("com_to_cob_offset_radius = 0.05",
                              "com_to_cob_offset_radius = 0.123")
    with pytest.raises(training.TrainingError, match="expected one of"):
        training.apply_dr_to_source(mangled, "curee", "DR_2")


def test_apply_dr_missing_anchor_aborts():
    with pytest.raises(training.TrainingError, match="anchor"):
        training.apply_dr_to_source("nothing here", "curee", "DR_0")


# ----------------------------------------------------------------- markers

def test_marker_paths_are_on_network_volume():
    assert training.PATCH_MARKER.startswith("/workspace/markers/")
    assert training.SANITY_MARKER.startswith("/workspace/markers/")
    assert training.PATCH_MARKER.endswith("bluerov2_patch_applied")
    assert training.SANITY_MARKER.endswith("axis_sanity_PASS")
