"""DR-level tables + train-command builder for launch_training.

Sources of truth (unit tests cross-check all three, char-exact):
  - CUREE table:    runbook/RUNBOOK.md Days 2-3 (code is ground truth for the
                    0.05 m radius — the paper's Table II claims 10x more)
  - BlueROV2 table: config/bluerov2_heavy.yaml domain_randomization.recommended
                    == patches/APPLY.md §4
  - train command:  RUNBOOK.md verbatim ([README] + seed/headless flags)

DR edits are CONTENT-ANCHORED with an expected-current-value assertion —
never line numbers (APPLY.md's line numbers shift once the patch lands).
The value strings below are written into warpauv_env.py verbatim.

DR_3/DR_4 (CUREE only) are replication-extension levels added for campaign
009 — not in the paper. The ladder is linear: radius 0/0.025/0.05/0.075/0.1 m,
volume +/-0/1.5/3/4.5/6 L (+/-0/6.6/13.2/19.8/26.4% of displacement).
"""
import re

ISAACLAB_DIR = "/workspace/IsaacLab"
WARPAUV_ENV_REMOTE = "/workspace/isaac-auv-env/warpauv_env.py"
MARKERS_DIR = "/workspace/markers"
PATCH_MARKER = f"{MARKERS_DIR}/bluerov2_patch_applied"
SANITY_MARKER = f"{MARKERS_DIR}/axis_sanity_PASS"

VEHICLES = ("curee", "bluerov2")

DR_TABLES = {
    "curee": {
        "DR_0": {"com_to_cob_offset_radius": "0",
                 "volume_range": "[0.022747843530591776, 0.022747843530591776]"},
        "DR_1": {"com_to_cob_offset_radius": "0.025",
                 "volume_range": "[0.021247843530591776, 0.024247843530591776]"},
        "DR_2": {"com_to_cob_offset_radius": "0.05",
                 "volume_range": "[0.019747843530591773, 0.02574784353059178]"},
        # 009 replication-extension levels (CUREE-only, not in the paper) —
        # continue the linear ladder past DR_2.
        "DR_3": {"com_to_cob_offset_radius": "0.075",
                 "volume_range": "[0.018247843530591775, 0.027247843530591776]"},
        "DR_4": {"com_to_cob_offset_radius": "0.1",
                 "volume_range": "[0.016747843530591777, 0.028747843530591774]"},
    },
    "bluerov2": {
        "DR_0": {"com_to_cob_offset_radius": "0",
                 "volume_range": "[0.0134, 0.0134]"},
        "DR_1": {"com_to_cob_offset_radius": "0.0164285714",
                 "volume_range": "[0.0125164, 0.0142836]"},
        "DR_2": {"com_to_cob_offset_radius": "0.0328571429",
                 "volume_range": "[0.0116328, 0.0151672]"},
    },
}

_LEVEL_ALIASES = {"dr_0": "DR_0", "none": "DR_0", "no_dr": "DR_0", "0": "DR_0",
                  "dr_1": "DR_1", "small": "DR_1", "1": "DR_1",
                  "dr_2": "DR_2", "large": "DR_2", "shipped": "DR_2", "2": "DR_2",
                  "dr_3": "DR_3", "3": "DR_3",
                  "dr_4": "DR_4", "4": "DR_4"}


class TrainingError(RuntimeError):
    """Refused/failed training preparation (bad level, anchor mismatch, ...)."""


def canonical_level(level: str) -> str:
    key = str(level).strip().lower()
    if key not in _LEVEL_ALIASES:
        raise TrainingError(f"unknown dr_level {level!r} — use DR_0..DR_4 "
                            "(aliases: none/small/large, 0-4; DR_3/DR_4 are "
                            "CUREE-only extensions)")
    return _LEVEL_ALIASES[key]


def table_entry(vehicle: str, level: str) -> dict:
    """Look up a single DR-level entry for `vehicle`, refusing cleanly (never
    a raw KeyError) when the level is valid globally but not defined for this
    vehicle — e.g. DR_3/DR_4 are CUREE-only extensions."""
    if vehicle not in VEHICLES:
        raise TrainingError(f"unknown vehicle {vehicle!r} — use curee|bluerov2")
    lvl = canonical_level(level)
    table = DR_TABLES[vehicle]
    if lvl not in table:
        raise TrainingError(f"{lvl} not defined for vehicle {vehicle!r} "
                            f"(levels: {', '.join(sorted(table))})")
    return table[lvl]


def build_train_command(seed: int, extra_args: str = "") -> str:
    """The exact RUNBOOK.md Days 2-3 command (workdir must be ISAACLAB_DIR)."""
    cmd = ("./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py "
           f"--task Isaac-WarpAUV-Direct-v1 --num_envs 2048 --headless "
           f"--seed {int(seed)}")
    if extra_args:
        cmd += f" {extra_args.strip()}"
    return cmd


# Anchors: match the ACTIVE cfg lines only — the commented-out No-DR preset
# two lines above must never match (the '#' before the name breaks ^\s*NAME).
_RADIUS_RE = re.compile(
    r"^(?P<indent>[ \t]*)com_to_cob_offset_radius\s*=\s*(?P<value>[^#\n]+?)"
    r"(?P<comment>\s*#[^\n]*)?$", re.MULTILINE)
_VOLRANGE_RE = re.compile(
    r"^(?P<indent>[ \t]*)volume_range\s*=\s*(?P<value>\[[^\]\n]*\])"
    r"(?P<comment>\s*#[^\n]*)?$", re.MULTILINE)


def _swap(src: str, rx: re.Pattern, field: str, new_value: str,
          known_values: set[str], vehicle: str) -> str:
    m = rx.search(src)
    if not m:
        raise TrainingError(f"content anchor for '{field}' not found in "
                            "warpauv_env.py — file layout changed? Refusing "
                            "to guess (never line numbers).")
    current = m.group("value").strip()
    if current not in known_values:
        raise TrainingError(
            f"'{field}' is currently '{current}', expected one of "
            f"{sorted(known_values)} for vehicle '{vehicle}'.\n"
            f">>> {m.group(0).strip()}\n"
            "This usually means the checkout doesn't match the vehicle "
            "(CUREE values on a patched file, or vice versa) — fix the "
            "checkout state (apply_bluerov_patches / git restore), don't force.")
    replaced = (f"{m.group('indent')}{field} = {new_value}"
                f"{m.group('comment') or ''}")
    return src[:m.start()] + replaced + src[m.end():]


def apply_dr_to_source(src: str, vehicle: str, level: str) -> str:
    """Rewrite the two active DR cfg lines to `level`, asserting the current
    values belong to `vehicle` (abort-with-diff on any mismatch)."""
    entry = table_entry(vehicle, level)   # unknown vehicle/level -> clean TrainingError
    table = DR_TABLES[vehicle]
    src = _swap(src, _RADIUS_RE, "com_to_cob_offset_radius",
                entry["com_to_cob_offset_radius"],
                {t["com_to_cob_offset_radius"] for t in table.values()}, vehicle)
    src = _swap(src, _VOLRANGE_RE, "volume_range",
                entry["volume_range"],
                {t["volume_range"] for t in table.values()}, vehicle)
    return src
