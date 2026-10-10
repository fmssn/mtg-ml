"""Deck-by-deck head to head of a candidate Jund model against a baseline, on paired seeds.

    python tools/eval_matrix.py --candidate pilot.pt --baseline lr075.pt --out results.json --games 800 --workers 8

For each opponent deck X (matchup `jund_<x>`; the Jund mirror for jund_wildfire) three cells are
played with the same seeds and deals, the Jund model always in canonical seat 0 (`evaluate.paired_specs`
with `balance_seats`: every seed once per starting player and once per physical seat, 4 games per seed):

  cand_vs_base  candidate as Jund vs baseline as X
  base_vs_base  baseline as Jund vs baseline as X (the reference)
  base_vs_cand  baseline as Jund vs candidate as X (did the candidate's play of X get better?)

The Jund delta per deck is cand_vs_base minus base_vs_base; the 4 games of a seed form one block, so
the interval is a paired one over seeds (normal approximation on the per-seed differences). A draw
counts as half a win. In the mirror the delta of the first cell is the head to head.
Sampled play by default, `--greedy` for argmax play.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mtg_ml.backend import ENV_VAR, engine_name  # noqa: E402
from mtg_ml.rl.evaluate import EVAL_SEED, paired_specs, wilson  # noqa: E402
from mtg_ml.rl.rollout import Job, create_pool, play  # noqa: E402

DECKS = {
    "mono_blue_terror": "jund_blue",
    "red_madness": "jund_madness",
    "grixis_affinity": "jund_affinity",
    "elves": "jund_elves",
    "tron": "jund_tron",
    "jund_wildfire": "jund_mirror",
}
CELLS = ("cand_vs_base", "base_vs_base", "base_vs_cand")


def per_seed(games: list) -> dict[int, float]:
    """Jund model's mean score per seed (canonical seat 0 won = 1, draw 0.5)."""
    blocks = defaultdict(list)
    for seats, winner, _reason, _turn, _actions, seed in games:
        blocks[seed].append(1.0 if winner == 0 else 0.5 if winner is None else 0.0)
    return {s: sum(v) / len(v) for s, v in blocks.items()}


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float]:
    """Mean of a-b over shared seeds and its 95% half width."""
    d = [a[s] - b[s] for s in sorted(a.keys() & b.keys())]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def run_cell(pool, workers: int, jund: str, other: str, matchup: str, games: int, greedy: bool) -> list:
    specs = paired_specs(other, games, matchup=matchup, seat=0, balance_seats=True)
    return play(pool, specs, Job([], jund, 0, record=False, greedy=greedy), workers).games


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--games", type=int, default=800, help="games per cell (4 per seed)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--decks", default=",".join(DECKS), help="opponent decks X (comma separated)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.environ[ENV_VAR] = engine_name(args.engine)
    out = {
        "candidate": args.candidate,
        "baseline": args.baseline,
        "games": args.games,
        "greedy": args.greedy,
        "engine": args.engine,
        "seed": EVAL_SEED,
        "decks": {},
    }
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
    with create_pool(args.workers) as pool:
        for deck in args.decks.split(","):
            if deck in out["decks"]:
                continue
            matchup, t0, rows = DECKS[deck], time.time(), {}
            for cell, (jund, other) in zip(
                CELLS,
                ((args.candidate, args.baseline), (args.baseline, args.baseline), (args.baseline, args.candidate)),
            ):
                rows[cell] = per_seed(run_cell(pool, args.workers, jund, other, matchup, args.games, args.greedy))
            res = {"matchup": matchup, "seeds": {c: {str(s): v for s, v in r.items()} for c, r in rows.items()}}
            for c, r in rows.items():
                n = len(r) * 4
                res[c] = {"score": sum(r.values()) / len(r), "ci": wilson(sum(r.values()) * 4, n), "games": n}
            res["delta_cand"] = paired(rows["cand_vs_base"], rows["base_vs_base"])
            res["delta_x"] = paired(
                rows["base_vs_cand"], rows["base_vs_base"]
            )  # negative: the candidate plays X better
            out["decks"][deck] = res
            print(
                f"{deck}: {res['cand_vs_base']['score']:.3f} vs {res['base_vs_base']['score']:.3f}  delta {res['delta_cand'][0]:+.3f} +-{res['delta_cand'][1]:.3f}  "
                f"X-delta {res['delta_x'][0]:+.3f} +-{res['delta_x'][1]:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
            with open(args.out + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()
