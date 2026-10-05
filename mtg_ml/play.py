"""Command line: watch random games, play against the random agent, benchmark.

    python -m mtg_ml.play watch --seed 3          # random vs random, full log
    python -m mtg_ml.play human --seat 1          # you play Mono Blue Terror
    python -m mtg_ml.play bench --games 200       # throughput and results
"""

from __future__ import annotations

import argparse
import collections
import time

from .agents import HumanAgent, RandomAgent, play_game


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.play")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="random vs random with a full game log")
    w.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("human", help="play against the random agent")
    h.add_argument("--seat", type=int, default=0, help="0 = Jund Wildfire, 1 = Mono Blue Terror")
    h.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bench", help="random-play throughput")
    b.add_argument("--games", type=int, default=100)
    args = ap.parse_args(argv)

    if args.cmd == "watch":
        g = play_game([RandomAgent(args.seed), RandomAgent(args.seed + 1)], seed=args.seed, log=True)
        print("\n".join(g.log))
    elif args.cmd == "human":
        agents = [RandomAgent(args.seed), RandomAgent(args.seed + 1)]
        agents[args.seat] = HumanAgent()
        g = play_game(agents, seed=args.seed, log=True)
        print(f"Game over: winner {g.winner} ({g.end_reason})")
    elif args.cmd == "bench":
        t = time.perf_counter()
        decisions = 0
        results = collections.Counter()
        for s in range(args.games):
            g = play_game([RandomAgent(s), RandomAgent(s + 10_000)], seed=s)
            decisions += len(g.actions)
            results[("Jund" if g.winner == 0 else "Blue" if g.winner == 1 else "draw", g.end_reason)] += 1
        dt = time.perf_counter() - t
        print(f"{args.games} games, {decisions} decisions in {dt:.1f}s: {args.games / dt:.1f} games/s, {decisions / dt:.0f} decisions/s")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
