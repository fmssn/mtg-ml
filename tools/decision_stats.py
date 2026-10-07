"""Decisions per game and per turn, with and without the action-decomposition
options (`Game(auto_mana=..., auto_pass=...)`), over scripted bot-vs-bot games.

    python tools/decision_stats.py --games 200 --engine native

Every configuration plays the same seeds (game 1 decks, alternating starting
player). Reports decisions per game / per turn, the split by decision kind,
and the Jund (seat 0) win rate, so the options can be checked to shorten
trajectories without changing who wins.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from mtg_ml.backend import game_class
from mtg_ml.bots import make_bot
from mtg_ml.match import match_decks

CONFIGS = {
    "default": {},
    "auto_mana": {"auto_mana": True},
    "auto_pass": {"auto_pass": True},
    "both": {"auto_mana": True, "auto_pass": True},
}


def run(cfg: dict, games: int, engine: str | None, start: int = 0) -> dict:
    Game = game_class(engine)
    bots = [make_bot(0), make_bot(1)]
    kinds: Counter = Counter()
    decisions = turns = 0
    wins = [0, 0]
    draws = 0
    for seed in range(start, start + games):
        g = Game(match_decks(1), seed=seed, starting_player=seed % 2, **cfg)
        while not g.over:
            kinds[g.decision.kind] += 1
            g.step(bots[g.decision.player].act(g))
        decisions += len(g.actions)
        turns += g.turn
        if g.winner is None:
            draws += 1
        else:
            wins[g.winner] += 1
    return {"decisions": decisions, "turns": turns, "kinds": kinds, "wins": wins, "draws": draws, "games": games}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--engine", default=None, help="python | native (default: $MTG_ENGINE or python)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated subset of {list(CONFIGS)}")
    args = ap.parse_args(argv)
    names = args.configs.split(",")
    results = {}
    for name in names:
        t = time.perf_counter()
        results[name] = r = run(CONFIGS[name], args.games, args.engine, args.start)
        n = r["games"]
        print(
            f"{name:>10}: {r['decisions'] / n:6.1f} decisions/game  {r['decisions'] / r['turns']:5.2f} /turn  "
            f"{r['turns'] / n:5.1f} turns/game  Jund wins {r['wins'][0] / n:5.1%} (draws {r['draws']})  "
            f"[{time.perf_counter() - t:.0f}s]",
            flush=True,
        )
    all_kinds = sorted({k for r in results.values() for k in r["kinds"]}, key=lambda k: -results[names[0]]["kinds"].get(k, 0))
    print("\ndecisions per game by kind")
    print(f"{'kind':>16} " + " ".join(f"{n:>10}" for n in names))
    for k in all_kinds:
        print(f"{k:>16} " + " ".join(f"{results[n]['kinds'].get(k, 0) / results[n]['games']:10.2f}" for n in names))


if __name__ == "__main__":
    main()
