"""Head-to-head evaluation on paired seeds.

    python -m mtg_ml.rl.evaluate runs/a/latest.pt runs/b/latest.pt --games 600
    python -m mtg_ml.rl.evaluate runs/a/latest.pt random
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot      # scripted bots
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --bo3   # best-of-three matches

Every seed is played twice with the seats swapped, so both sides see the
same shuffles from both decks. Scores count a draw as half a win; the
interval is Wilson 95%. With --bo3 each seed is a best-of-three match
(`mtg_ml.match` rules: sideboarded games 2/3, loser starts the next game).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

from ..backend import ENV_VAR, engine_name
from ..match import MatchResult, game_seed
from .rollout import BOT, LEARNER, RANDOM, GameSpec, Job, run_job, split_games, worker_init

EVAL_SEED = 10_000_000


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def paired_specs(opponent: str, games: int, seed: int = EVAL_SEED) -> list[GameSpec]:
    specs = []
    for s in range(games // 2):
        specs.append(GameSpec(seed=seed + s, seats=(LEARNER, opponent)))
        specs.append(GameSpec(seed=seed + s, seats=(opponent, LEARNER)))
    return specs


def score(games: list) -> dict:
    """Learner score per deck and overall from `Result.games` rows."""
    out = {}
    rows = {"jund": [], "blue": []}
    for g in games:
        seat = 0 if g[0][0] == LEARNER else 1
        rows["jund" if seat == 0 else "blue"].append(1.0 if g[1] == seat else 0.5 if g[1] is None else 0.0)
    rows["all"] = rows["jund"] + rows["blue"]
    for k, xs in rows.items():
        lo, hi = wilson(sum(xs), len(xs))
        out[k] = (sum(xs) / max(len(xs), 1), (round(lo, 3), round(hi, 3)), len(xs))
    return out


def head_to_head(procs, learner_path: str, opponent: str, games: int, n_jobs: int, version: int = 0, max_turns: int = 100) -> dict:
    jobs = [Job(chunk, learner_path, version, record=False, max_turns=max_turns) for chunk in split_games(paired_specs(opponent, games), n_jobs)]
    return score([g for r in procs.map(run_job, jobs) for g in r.games])


def head_to_head_bo3(procs, learner_path: str, opponent: str, matches: int, n_jobs: int, version: int = 0, max_turns: int = 100) -> dict:
    """Best-of-three matches on paired seeds. Each round plays the next game
    of every unfinished match as one batch. Returns match scores like `score`."""
    live = [(EVAL_SEED + s, seats, MatchResult()) for s in range(matches // 2) for seats in ((LEARNER, opponent), (opponent, LEARNER))]
    while True:
        todo = {}
        for seed, seats, res in live:
            if not res.over:
                n, start = res.next_game(seed)
                todo[(game_seed(seed, n), seats)] = (res, GameSpec(seed=game_seed(seed, n), seats=seats, starting_player=start, match_game=n))
        if not todo:
            break
        jobs = [Job(chunk, learner_path, version, record=False, max_turns=max_turns) for chunk in split_games([sp for _, sp in todo.values()], n_jobs)]
        for r in procs.map(run_job, jobs):
            for seats, winner, reason, _, _, seed in r.games:
                res, spec = todo[(seed, seats)]
                res.games.append((spec.starting_player, winner, reason))
    return score([(seats, res.winner) for _, seats, res in live])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.rl.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("opponent", help=f"a checkpoint path, '{RANDOM}' or '{BOT}'")
    ap.add_argument("--games", type=int, default=400, help="games (or matches with --bo3)")
    ap.add_argument("--bo3", action="store_true", help="best-of-three matches with sideboarding")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default=None, help="python or native (default: $MTG_ENGINE, else python)")
    args = ap.parse_args(argv)
    if args.engine:
        os.environ[ENV_VAR] = engine_name(args.engine)
    with mp.get_context("spawn").Pool(args.workers, initializer=worker_init) as procs:
        fn = head_to_head_bo3 if args.bo3 else head_to_head
        res = fn(procs, args.checkpoint, args.opponent, args.games, args.workers)
    for k, (s, ci, n) in res.items():
        print(f"{k:5s} {s:.3f}  95% CI {ci}  ({n} games)")


if __name__ == "__main__":
    main()
