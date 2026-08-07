"""Generator AND pin for the two SYNTHETIC watch fixtures in this directory.
Not a test file itself.

  python _gen.py           CHECK mode (the default): render both fixtures in
                           memory and byte-compare them against the committed
                           .log files. Exit 0 when every fixture matches; exit
                           1 with a per-file drift message when one does not.
  python _gen.py --write   Rewrite the .log files from the generators. Only
                           needed when a generator is deliberately changed.

Check mode is what makes this a pin rather than a one-off script: the same
byte-identity is asserted inline by tests/test_watch.py, so a hand-edited
fixture is caught by the suite, not just by whoever remembers to run this.
All paths resolve from THIS FILE's directory, so both modes behave
identically no matter what the current working directory is.

Only two of this directory's .log fixtures are generated here —
flattened_tail.log and two_iter_block.log. converging_200iters.log is REAL
captured pod output and the cuda_oom / traceback fixtures were written by
hand; none of those three has a generator, and none is touched by --write.

Provenance: format/spacing mirrors the REAL converging_200iters.log fixture
(source job 20260716-144932_train-curee-dr-2-s1_ba84, rsl_rl console output),
values here are synthetic.
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ESC = "\x1b"


def block(i, total, reward, *, vloss=0.5, sloss=0.01, eloss=-4.0, elen=180.0,
          steps=300000, ts=None):
    ts = ts if ts is not None else (i + 1) * 49152
    return (
        "################################################################################\n"
        f"                     {ESC}[1m Learning iteration {i}/{total} {ESC}[0m                      \n"
        "\n"
        f"                       Computation: {steps} steps/s (collection: 0.128s, learning 0.034s)\n"
        "             Mean action noise std: 0.11\n"
        f"          Mean value_function loss: {vloss:.4f}\n"
        f"               Mean surrogate loss: {sloss:.4f}\n"
        f"                 Mean entropy loss: {eloss:.4f}\n"
        f"                       Mean reward: {reward:.2f}\n"
        f"               Mean episode length: {elen:.2f}\n"
        "--------------------------------------------------------------------------------\n"
        f"                   Total timesteps: {ts}\n"
        "                    Iteration time: 0.16s\n"
        "                      Time elapsed: 00:07:00\n"
        "                               ETA: 00:01:00\n"
        "\n"
    )


def gen_flattened_tail():
    """60 iterations (2440-2499 of 2500) with mean reward flat within a ~1.2
    spread band — PlateauDetector(window=50, min_delta=3.0) MUST fire."""
    rng = random.Random(42)
    out = []
    for idx, i in enumerate(range(2440, 2500)):
        reward = 85.0 + rng.uniform(-0.6, 0.6)
        out.append(block(i, 2500, reward))
    return "".join(out)


def gen_two_iter_block():
    """Small 2-block fixture (iterations 0-1 of 100) used by the split-chunk
    incremental-parser test — whole-file parse must equal split-feed parse."""
    return block(0, 100, 3.17, vloss=2.4427, sloss=-0.0029, eloss=8.4917,
                elen=20.16, steps=34946, ts=49152) + \
           block(1, 100, 5.94, vloss=1.6946, sloss=-0.0020, eloss=8.4320,
                elen=43.61, steps=298077, ts=98304)


# Filename -> generator. Single source of truth for check mode, --write, and
# the byte-identity assertions in tests/test_watch.py.
GENERATED = {
    "flattened_tail.log": gen_flattened_tail,
    "two_iter_block.log": gen_two_iter_block,
}

# newline="" on BOTH read and write disables universal-newline translation,
# so what we compare (and what we write) is exactly what the generator
# produced — the same idiom tests/test_watch.py:_read_fixture uses.
_IO = {"encoding": "utf-8", "newline": ""}


def write():
    """Rewrite every generated fixture in place. Returns the paths written."""
    written = []
    for name, gen in GENERATED.items():
        path = HERE / name
        with open(path, "w", **_IO) as f:
            f.write(gen())
        written.append(path)
    return written


def check():
    """Byte-compare each committed fixture against its generator.

    Returns a list of human-readable drift messages — empty means clean.
    """
    drift = []
    for name, gen in GENERATED.items():
        path = HERE / name
        if not path.exists():
            drift.append(f"{name}: MISSING — run `python _gen.py --write`")
            continue
        with open(path, **_IO) as f:
            committed = f.read()
        if committed != gen():
            drift.append(
                f"{name}: DRIFT — the committed bytes no longer match "
                f"{gen.__name__}() in this file. Either the fixture was "
                "hand-edited (restore it with `python _gen.py --write`) or "
                "the generator changed on purpose (then --write and commit "
                "the regenerated fixture together)."
            )
    return drift


if __name__ == "__main__":
    _args = sys.argv[1:]
    if _args == ["--write"]:
        for _p in write():
            print(f"wrote {_p}")
    elif not _args:
        _drift = check()
        for _msg in _drift:
            print(f"_gen.py: {_msg}", file=sys.stderr)
        if _drift:
            sys.exit(1)
        print(f"_gen.py: OK — {len(GENERATED)} fixtures match their generators")
    else:
        print(f"usage: {Path(__file__).name} [--write]   "
              "(no argument = check mode)", file=sys.stderr)
        sys.exit(2)
