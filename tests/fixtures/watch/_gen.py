"""One-off generator for synthetic watch fixtures. Not a test file itself —
run manually to (re)produce the committed .log fixtures in this directory.
Provenance: format/spacing mirrors the REAL converging_200iters.log fixture
(source job 20260716-144932_train-curee-dr-2-s1_ba84, rsl_rl console output),
values here are synthetic.
"""
import random

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


if __name__ == "__main__":
    with open("flattened_tail.log", "w") as f:
        f.write(gen_flattened_tail())
    with open("two_iter_block.log", "w") as f:
        f.write(gen_two_iter_block())
