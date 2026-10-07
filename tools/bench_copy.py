"""Game.copy() vs the replay fork (`fork(replay=True)`), and what snapshots
cost while stepping.

Scripted bots play each game up to decision 50 / 200 / 400 (games that end
earlier drop out of the later points); at each point both copies are timed
(median over games). `copy()` is timed after the game's first copy, which
replays once and turns snapshots on.

    python tools/bench_copy.py --engine native --games 40
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from mtg_ml.trace import Scenario, make_agents, new_game


def timed(f, reps: int) -> float:
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="python", choices=("python", "native"))
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--points", default="50,200,400")
    args = ap.parse_args()
    points = [int(x) for x in args.points.split(",")]
    reps = 50 if args.engine == "native" else 5
    replay: dict[int, list[float]] = {p: [] for p in points}
    copy: dict[int, list[float]] = {p: [] for p in points}
    since: dict[int, list[int]] = {p: [] for p in points}
    plain, snap = [], []
    for seed in range(args.games):
        sc = Scenario(seed=seed, agents=("bot", "bot"))
        agents = make_agents(sc)
        g = new_game(sc, args.engine, log=False)
        n = 0
        while not g.over and n < max(points):
            g.step(agents[g.decision.player].act(g))
            n += 1
            if n in points and not g.over:
                replay[n].append(timed(lambda: g.fork(replay=True), reps))
                g.copy()
                copy[n].append(timed(g.copy, reps))
                if args.engine == "python":
                    since[n].append(len(g.actions) - g._snap.n_actions)
        # per-decision stepping cost, random play, of a game never copied and of one copied (it snapshots every step)
        for on, out in ((False, plain), (True, snap)):
            g = new_game(sc, args.engine, log=False)
            if on:
                g.copy()  # the copied game takes snapshots from now on
            rng = random.Random(seed)
            k = 0
            t = time.perf_counter()
            while not g.over and k < 400:
                g.step(rng.randrange(len(g.legal_options())))
                k += 1
            out.append((time.perf_counter() - t) / max(k, 1))
    med = statistics.median
    print(f"engine={args.engine}, {args.games} bot games")
    print("| decision | games | replay fork | copy() | speed-up |")
    print("|---:|---:|---:|---:|---:|")
    for p in points:
        if replay[p]:
            r, c = med(replay[p]), med(copy[p])
            print(f"| {p} | {len(replay[p])} | {r * 1e3:.3f} ms | {c * 1e3:.3f} ms | {r / c:.0f}x |")
    if args.engine == "python":
        allsince = [x for p in points for x in since[p]]
        print(f"actions replayed by copy() (since the step began): median {med(allsince)}, max {max(allsince)}")
    print(f"step cost, random play: {med(plain) * 1e6:.1f} us without snapshots, {med(snap) * 1e6:.1f} us with")


if __name__ == "__main__":
    main()
