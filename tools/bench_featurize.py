"""Microbenchmark and fingerprint of native `featurize_flat`.

    python tools/bench_featurize.py --games 30 --features 7 --reps 3

Plays random games (fixed seeds, every matchup), calls `featurize_flat` for the
deciding player at every decision, and reports us/call plus a sha256 over all
outputs (the fingerprint must not change under pure speed-ups).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import time

from mtg_ml.backend import game_class
from mtg_ml.match import EXPLICIT_ONLY, MATCHUPS, game_args


def states(games: int, max_turns: int = 30):
    pool = sorted(set(MATCHUPS) - EXPLICIT_ONLY)
    for seed in range(games):
        g = game_class("native")(seed=seed, max_turns=max_turns, **game_args(1 + seed // 3 % 2, pool[seed % len(pool)]))
        r = random.Random(seed)
        while not g.over:
            yield g
            g.step(r.randrange(len(g.legal_options())))


def fingerprint(games: int, features: int) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for g in states(games):
        for p in (g.decision.player, 1 - g.decision.player):
            s, ol, of = g.featurize_flat(p, 4096, 4096, features)
            h.update(repr((s, ol, of)).encode())
        n += 1
    return h.hexdigest(), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--features", type=int, default=7)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    best = []
    for _ in range(a.reps):
        ts = []
        for g in states(a.games):
            p = g.decision.player
            t = time.perf_counter()
            g.featurize_flat(p, 4096, 4096, a.features)
            ts.append(time.perf_counter() - t)
        best.append(ts)
    tot = [sum(t) for t in best]
    print(f"decisions {len(best[0])}  total {min(tot):.3f}s  mean {1e6 * min(tot) / len(best[0]):.1f} us/call  median {1e6 * statistics.median(best[0]):.1f} us")
    print("fingerprint", *fingerprint(a.games, a.features))


if __name__ == "__main__":
    main()
