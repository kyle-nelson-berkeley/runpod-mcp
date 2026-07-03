#!/usr/bin/env python3
"""apply_bluerov2_patch.py — automate patches/APPLY.md §§1-5 on the pod.

Converts the warplab isaac-auv-env checkout (pinned @ 7c5ebe7) from the
6-thruster CUREE to the 8-thruster BlueROV2 Heavy. Stdlib only — runs on the
pod's system python3 (scp'd there by the MCP's apply_bluerov_patches tool).

Strategy (consensus plan, "pristine-then-apply"):
  1. assert the checkout is the pinned SHA;
  2. delete the untracked drop-in (bluerov2_heavy_thrusters.py), then
     `git checkout -- .` and assert `git status --porcelain -uno` is EMPTY
     (-uno: the drop-in is untracked by design and must not fail the assert);
  3. re-copy the drop-in;
  4. apply every APPLY.md edit as a CONTENT-ANCHORED regex with an exact
     match-count assertion — never line numbers (they shift post-patch);
     abort with the surrounding context on any mismatch;
  5. py_compile both edited files;
  6. write the marker (markers/bluerov2_patch_applied).

Restoring pristine first makes idempotency deterministic and defuses
conflicts with earlier CUREE DR edits (launch_training rewrites DR lines).
Exit 0 on success, 1 on any failure.
"""
import argparse
import py_compile
import shutil
import subprocess
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

PIN_SHA = "7c5ebe7f7a08acd2570b5fba328e92b7f59f6794"
DROP_IN = "bluerov2_heavy_thrusters.py"

# (label, pattern, replacement, expected_match_count) — all against the
# PRISTINE 7c5ebe7 content, quoted from patches/APPLY.md.
ENV_EDITS = [
    ("§1 import swap (thruster geometry from the drop-in)",
     r"from \.thruster_dynamics import DynamicsFirstOrder, "
     r"ConversionFunctionBasic, get_thruster_com_and_orientations",
     "from .thruster_dynamics import DynamicsFirstOrder, "
     "ConversionFunctionBasic\n"
     "from .bluerov2_heavy_thrusters import get_thruster_com_and_orientations",
     1),
    ("§2 action_space shape 6->8",
     r"(action_space: gym\.spaces\.Space = gym\.spaces\.Box\("
     r"low=-1\.0, high=1\.0, shape=\()6(,\))",
     r"\g<1>8\g<2>", 1),
    ("§2 num_actions 6->8",
     r"(?m)^(\s*num_actions = )6$",
     r"\g<1>8", 1),
    ("§2 _actions buffer 6->8",
     r"(self\._actions = torch\.zeros\(self\.num_envs, )6(,)",
     r"\g<1>8\g<2>", 1),
    ("§2 DynamicsFirstOrder 6->8",
     r"(DynamicsFirstOrder\(self\.num_envs, )6(,)",
     r"\g<1>8\g<2>", 1),
    ("§2 thruster force/torque buffers 6->8 (x2)",
     r"(= torch\.zeros\(\(self\.num_envs, )6(, 3\))",
     r"\g<1>8\g<2>", 2),
    ("§3 volume -> 0.0134",
     r"(?m)^(\s*volume = )0\.022747843530591776( |$)",
     r"\g<1>0.0134\g<2>", 1),
    ("§3 mass -> 13.5",
     r"(?m)^(\s*mass = )2\.2701e\+01( |$)",
     r"\g<1>13.5\g<2>", 1),
    ("§3 hydro inertia Ix -> 0.26",
     r"(?m)^(\s*self\.inertia_tensors\[:, 0\] = )0\.37$",
     r"\g<1>0.26", 1),
    ("§3 hydro inertia Iy -> 0.23",
     r"(?m)^(\s*self\.inertia_tensors\[:, 1\] = )0\.97$",
     r"\g<1>0.23", 1),
    ("§3 hydro inertia Iz -> 0.37",
     r"(?m)^(\s*self\.inertia_tensors\[:, 2\] = )1\.19$",
     r"\g<1>0.37", 1),
    ("§4 DR radius -> Small (launch_training can re-level later)",
     r"(?m)^(\s*)com_to_cob_offset_radius = 0\.05( |$)",
     r"\g<1>com_to_cob_offset_radius = 0.0164285714\g<2>", 1),
    ("§4 DR volume_range -> Small",
     r"(?m)^(\s*)volume_range = "
     r"\[0\.019747843530591773, 0\.02574784353059178\]( |$)",
     r"\g<1>volume_range = [0.0125164, 0.0142836]\g<2>", 1),
]

ASSET_EDITS = [
    ("§5 PhysX mass_props (USD carries CUREE mass; inertia stays CUREE's — "
     "accepted, documented mismatch, see APPLY.md §5)",
     r"(?m)^(?P<indent>\s*)enable_gyroscopic_forces=True,\n"
     r"(?P<close>\s*\),)\n",
     "\\g<indent>enable_gyroscopic_forces=True,\n\\g<close>\n"
     "        mass_props=sim_utils.MassPropertiesCfg(mass=13.5),\n",
     1),
]


def fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def apply_edits(path: Path, edits: list, dry_run: bool) -> None:
    text = path.read_text()
    for label, pattern, repl, want in edits:
        found = len(re.findall(pattern, text))
        if found != want:
            context = "\n".join(
                line for line in text.splitlines()
                if re.search(pattern.split(r"\n")[0][:40] or "-", line)) or "(no similar lines)"
            fatal(f"anchor mismatch for [{label}] in {path.name}: expected "
                  f"{want} match(es), found {found}. Never guessing from line "
                  f"numbers. Nearby candidates:\n{context}")
        text = re.sub(pattern, repl, text)
        print(f"  ok: {label}")
    if not dry_run:
        path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="/workspace/isaac-auv-env",
                    help="isaac-auv-env checkout to patch")
    ap.add_argument("--thrusters", required=True,
                    help="path to patches/bluerov2_heavy_thrusters.py (drop-in)")
    ap.add_argument("--markers", default="/workspace/markers",
                    help="marker directory (network volume)")
    ap.add_argument("--expect-sha", default=PIN_SHA,
                    help="required HEAD SHA (test override only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify anchors + list edits, write nothing")
    args = ap.parse_args()

    repo = Path(args.repo)
    thrusters_src = Path(args.thrusters)
    env_py = repo / "warpauv_env.py"
    asset_py = repo / "assets" / "warpauv.py"

    for p in (repo / ".git", env_py, asset_py, thrusters_src):
        if not p.exists():
            fatal(f"missing: {p}")

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != args.expect_sha:
        fatal(f"checkout is at {head[:12]}, expected {args.expect_sha[:12]} — "
              "re-pin the repo (see runbook/pod_setup.sh step 6) before patching")

    # ---- pristine restore ---------------------------------------------------
    print("1/4 restoring pristine checkout")
    drop_in = repo / DROP_IN
    if not args.dry_run:
        if drop_in.exists():
            drop_in.unlink()                       # delete BEFORE the assert
        r = git(repo, "checkout", "--", ".")
        if r.returncode != 0:
            fatal(f"git checkout -- . failed: {r.stderr.strip()}")
    status = git(repo, "status", "--porcelain", "-uno").stdout.strip()
    if status and not args.dry_run:
        fatal(f"checkout not pristine after restore:\n{status}")

    # ---- drop-in ------------------------------------------------------------
    print("2/4 installing thruster drop-in")
    if not args.dry_run:
        shutil.copyfile(thrusters_src, drop_in)    # re-copy AFTER the assert

    # ---- content-anchored edits --------------------------------------------
    print(f"3/4 applying APPLY.md edits ({'DRY RUN' if args.dry_run else 'live'})")
    apply_edits(env_py, ENV_EDITS, args.dry_run)
    apply_edits(asset_py, ASSET_EDITS, args.dry_run)
    if not args.dry_run:
        for p in (env_py, asset_py, drop_in):
            py_compile.compile(str(p), doraise=True)

    # ---- marker -------------------------------------------------------------
    print("4/4 writing marker")
    if not args.dry_run:
        markers = Path(args.markers)
        markers.mkdir(parents=True, exist_ok=True)
        (markers / "bluerov2_patch_applied").write_text(
            f"applied {datetime.now(timezone.utc).isoformat()} on {head}\n"
            f"edits: {len(ENV_EDITS) + len(ASSET_EDITS)} (APPLY.md §§1-5)\n")
    print("PATCH OK — run the axis sanity sweep BEFORE any training "
          "(RUNBOOK Days 4-5 step 3).")


if __name__ == "__main__":
    main()
