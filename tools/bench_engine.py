"""Engine throughput benchmark, single process or one pinned process per core.

Modes (decisions/s are counted the same way as `play bench`):
  engine    random choices by index only: the engine's own cost
  agents    RandomAgent through the public Game API (`play bench`)
  rollout   the CPU side of an RL rollout worker without the network:
            featurize() for the decider, hashed event tokens for both
            seats, random choice, step (rl/rollout.py minus torch)
  bots      scripted bot vs scripted bot

    python tools/bench_engine.py --engine native --mode rollout --seconds 20
    python tools/bench_engine.py --engine native --mode rollout --procs 64 --cpus 0-63

`--cpus` pins process i to the i-th listed CPU (Linux), so a shared box
stays usable: e.g. --procs 32 --cpus 32-63.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _cpus(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(engine: str, mode: str, seconds: float, seed0: int, cpu: int | None = None) -> tuple[int, int, float]:
    """(games, decisions, elapsed) after playing whole games for `seconds`."""
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    from mtg_ml.agents import RandomAgent
    from mtg_ml.backend import game_class
    from mtg_ml.bots import make_bot
    from mtg_ml.match import match_decks
    from mtg_ml.rl.features import encode_event_hashes, event_hashes, featurize

    Game = game_class(engine)
    decks = [match_decks(1), match_decks(2)]
    games = decisions = 0
    t0 = time.perf_counter()
    s = seed0
    while time.perf_counter() - t0 < seconds:
        g = Game(decks[s % 2], seed=s, match_game=1 + s % 2)
        r = random.Random(s)
        if mode == "engine":
            raw = getattr(g, "_g", None)
            if raw is not None:  # native: skip the Python wrapper entirely
                while not raw.over:
                    raw.step(0 if r.random() < 0.4 else r.randrange(raw.n_options()))
                    decisions += 1
            else:
                while not g.over:
                    n = len(g.decision.options)
                    g.step(0 if r.random() < 0.4 else r.randrange(n))
                    decisions += 1
        elif mode == "agents":
            agents = (RandomAgent(s * 2), RandomAgent(s * 2 + 1))
            while not g.over:
                g.step(agents[g.decision.player].act(g))
                decisions += 1
        elif mode == "bots":
            agents = (make_bot(0), make_bot(1))
            while not g.over:
                g.step(agents[g.decision.player].act(g))
                decisions += 1
        elif mode == "rollout":
            events = ([], [])
            while not g.over:
                d = g.decision
                p = d.player
                featurize(g, p)
                encode_event_hashes(events[p])
                events[p].clear()
                n = len(d.options)
                a = 0 if r.random() < 0.4 else r.randrange(n)
                mine, theirs = event_hashes(g, a)
                events[p].extend(mine)
                events[1 - p].extend(theirs)
                g.step(a)
                decisions += 1
        else:
            raise SystemExit(f"unknown mode {mode}")
        games += 1
        s += 1
    return games, decisions, time.perf_counter() - t0


def _worker(args):
    return run(*args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="python")
    ap.add_argument("--mode", default="engine", choices=("engine", "agents", "rollout", "bots"))
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cpus", default=None, help="CPU list to pin to, e.g. 0-31 or 0,2,4")
    ap.add_argument("--warmup", type=float, default=2, help="seconds of play before timing (JIT warm-up for PyPy)")
    args = ap.parse_args(argv)
    cpus = _cpus(args.cpus)
    if args.procs == 1:
        if args.warmup:
            run(args.engine, args.mode, args.warmup, 10**6, cpus[0] if cpus else None)
        g, d, dt = run(args.engine, args.mode, args.seconds, 0, cpus[0] if cpus else None)
        rate = d / dt
    else:
        work = [(args.engine, args.mode, args.seconds, i * 100_000, cpus[i % len(cpus)] if cpus else None) for i in range(args.procs)]
        with mp.get_context("spawn").Pool(args.procs) as pool:
            res = pool.map(_worker, work)
        g = sum(x[0] for x in res)
        d = sum(x[1] for x in res)
        rate = sum(x[1] / x[2] for x in res)
    impl = sys.implementation.name
    print(f"{args.engine:6s} {args.mode:7s} procs={args.procs:<3d} [{impl}] {g} games, {d} decisions: {rate:,.0f} decisions/s total, {rate / args.procs:,.0f} per process")


if __name__ == "__main__":
    main()
