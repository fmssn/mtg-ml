"""Cost of feature set 6's simulated option previews per decision.

    python tools/bench_sim.py --engine native --games 20
    python tools/bench_sim.py --engine python --games 3

Plays random games (fixed seeds, every matchup) and times, at every
decision, `featurize_flat` in set 5 and in set 6 on the same state, and the
engine step. Set 6 copies the game, so the game takes a snapshot at every
step start from its first set-6 featurization on: the step time includes
that. Reports medians and means per decision, the share of decisions whose
options were skipped (more than `SIM_MAX_OPTIONS`), the options per
decision, and how many options run the second ("if the opponent passes")
simulation.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from mtg_ml.backend import game_class
from mtg_ml.encode import SIM_MAX_OPTIONS, option_previews
from mtg_ml.match import MATCHUPS, game_args
from mtg_ml.rl.features import featurize_flat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="native")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=30)
    a = ap.parse_args()
    t5, t6, ts, nopt = [], [], [], []
    skipped = second = 0
    for seed in range(a.games):
        m = sorted(MATCHUPS)[seed % len(MATCHUPS)]
        g = game_class(a.engine)(seed=seed, max_turns=a.max_turns, **game_args(1 + seed // 3 % 2, m))
        r = random.Random(seed)
        while not g.over:
            p = g.decision.player
            n = len(g.legal_options())
            t = time.perf_counter()
            featurize_flat(g, p, features=5)
            t5.append(time.perf_counter() - t)
            t = time.perf_counter()
            featurize_flat(g, p, features=6)
            t6.append(time.perf_counter() - t)
            nopt.append(n)
            second += sum("pv:sim:next:opponent:priority" in pv for pv in option_previews(g, p, 6))  # untimed
            skipped += n > SIM_MAX_OPTIONS
            t = time.perf_counter()
            g.step(r.randrange(n))
            ts.append(time.perf_counter() - t)
    us = 1e6

    def row(name, xs):
        return f"{name:<22} median {statistics.median(xs) * us:8.1f} us   mean {statistics.fmean(xs) * us:8.1f} us"

    print(f"engine {a.engine}, {a.games} games, {len(t5)} decisions, options/decision mean {statistics.fmean(nopt):.1f} max {max(nopt)}, skipped {skipped}, second (pv:simp:) simulation on {second / sum(nopt):.0%} of options")
    print(row("featurize set 5", t5))
    print(row("featurize set 6", t6))
    print(row("set 6 - set 5", [b - c for b, c in zip(t6, t5)]))
    print(row("step (snapshots on)", ts))
    print(f"set 6 / set 5 (means): {statistics.fmean(t6) / statistics.fmean(t5):.2f}x")


if __name__ == "__main__":
    main()
