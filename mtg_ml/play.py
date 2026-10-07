"""Command line: watch random games, play against the random agent, benchmark.

    python -m mtg_ml.play watch --seed 3          # random vs random, full log
    python -m mtg_ml.play watch --agents bot,bot  # scripted bots
    python -m mtg_ml.play human --seat 1          # you play Mono Blue Terror
    python -m mtg_ml.play human --opponent bot    # ... against the scripted bot
    python -m mtg_ml.play bench --games 200       # throughput and results
    python -m mtg_ml.play match --agents bot,bot --matches 200   # best-of-three with sideboards
    python -m mtg_ml.play bench --engine native   # any command: --engine python|native (or $MTG_ENGINE)
    python -m mtg_ml.play watch --agents model:runs/x/final.pt,bot --features 2   # model agents in feature set 2
"""

from __future__ import annotations

import argparse
import collections
import time

from .agents import HumanAgent, RandomAgent, play_game
from .backend import engine_name
from .bots import make_bot
from .encode import FEATURE_VERSIONS
from .match import matchup_decks, play_match


def make_agent(kind: str, seat: int, seed: int, greedy: bool = False, deck: str | None = None, features: int = 0):
    """`deck`: the deck in `seat` (default: the seat's usual one), for the bot.
    `features`: for a model, the feature-set version to featurize in (0: its checkpoint's)."""
    if kind.startswith("model:"):  # model:<checkpoint path>
        from .rl.agent import ModelAgent

        return ModelAgent(kind[len("model:"):], seat, sample=not greedy, seed=seed, features=features)
    if kind == "bot":
        return make_bot(seat, deck)
    if kind.startswith("search"):  # search or search:<playouts>
        from .bots.search import SearchBot

        playouts = int(kind.split(":")[1]) if ":" in kind else 8
        return SearchBot(seat, playouts=playouts, seed=seed, deck=deck)
    if kind == "random":
        return RandomAgent(seed + seat)
    raise SystemExit(f"unknown agent {kind!r} (random, bot, search[:playouts], model:<checkpoint>)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.play")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="random vs random with a full game log")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--agents", default="random,random", help="seat0,seat1 from: random, bot, search[:playouts]")
    h = sub.add_parser("human", help="play against the random agent")
    h.add_argument("--seat", type=int, default=0, help="0 = Jund Wildfire, 1 = Mono Blue Terror")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--opponent", default="random", help="random or bot")
    b = sub.add_parser("bench", help="random-play throughput")
    b.add_argument("--games", type=int, default=100)
    b.add_argument("--agents", default="random,random", help="seat0,seat1 from: random, bot, search[:playouts]")
    m = sub.add_parser("match", help="best-of-three matches with sideboarding")
    m.add_argument("--matches", type=int, default=100)
    m.add_argument("--agents", default="bot,bot", help="seat0,seat1 from: random, bot, search[:playouts]")
    m.add_argument("--log", action="store_true", help="print the game logs of the first match")
    m.add_argument("--matchup", default="jund_blue", help="match.MATCHUPS: jund_blue or jund_madness")
    for p in (w, h, b, m):
        p.add_argument("--engine", default=None, help="python (reference) or native (Rust); default: $MTG_ENGINE, else python")
        p.add_argument("--features", type=int, default=0, choices=FEATURE_VERSIONS, help="featurize model:<checkpoint> agents in this feature-set version instead of the one their config records (docs/features.md)")
    args = ap.parse_args(argv)
    engine = args.engine
    feats = args.features

    if args.cmd == "watch":
        kinds = args.agents.split(",")
        g = play_game([make_agent(k, i, args.seed, features=feats) for i, k in enumerate(kinds)], seed=args.seed, log=True, engine=engine)
        print("\n".join(g.log))
    elif args.cmd == "human":
        agents = [make_agent(args.opponent, i, args.seed, features=feats) for i in (0, 1)]
        agents[args.seat] = HumanAgent()
        g = play_game(agents, seed=args.seed, log=True, engine=engine)
        print(f"Game over: winner {g.winner} ({g.end_reason})")
    elif args.cmd == "bench":
        t = time.perf_counter()
        decisions = 0
        results = collections.Counter()
        for s in range(args.games):
            kinds = args.agents.split(",")
            g = play_game([make_agent(k, i, s * 10_000, features=feats) for i, k in enumerate(kinds)], seed=s, engine=engine)
            decisions += len(g.actions)
            results[("Jund" if g.winner == 0 else "Blue" if g.winner == 1 else "draw", g.end_reason)] += 1
        dt = time.perf_counter() - t
        print(f"[{engine_name(engine)}] {args.games} games, {decisions} decisions in {dt:.1f}s: {args.games / dt:.1f} games/s, {decisions / dt:.0f} decisions/s")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")


    elif args.cmd == "match":
        kinds = args.agents.split(",")
        matches, games = collections.Counter(), collections.Counter()
        for s in range(args.matches):
            decks = matchup_decks(args.matchup)
            r = play_match([make_agent(k, i, s * 10_000, deck=decks[i], features=feats) for i, k in enumerate(kinds)], seed=s, log=args.log and s == 0, engine=engine, matchup=args.matchup)
            matches[r.winner] += 1
            for n, (_, w, _) in enumerate(r.games, 1):
                games[(n, w)] += 1
        name = {0: "seat 0", 1: "seat 1", None: "draw"}
        print(f"{args.matches} matches: " + ", ".join(f"{name[k]} {v}" for k, v in sorted(matches.items(), key=lambda kv: str(kv[0]))))
        for n in (1, 2, 3):
            tot = sum(v for (gn, _), v in games.items() if gn == n)
            if tot:
                print(f"  game {n}: seat 0 wins {games[(n, 0)]}/{tot} ({games[(n, 0)] / tot:.1%})")


if __name__ == "__main__":
    main()
