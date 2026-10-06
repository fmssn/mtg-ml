"""Head-to-head evaluation on paired seeds.

    python -m mtg_ml.rl.evaluate runs/a/latest.pt runs/b/latest.pt --games 600
    python -m mtg_ml.rl.evaluate runs/a/latest.pt random
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot      # scripted bots
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --bo3   # best-of-three matches
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --jund  # the benchmark: learner Jund vs the blue bot

Every seed is played twice with the seats swapped, so both sides see the
same shuffles from both decks. With --jund the learner always plays Jund
Wildfire (seat 0) and every seed is played once per starting player instead;
against `bot` that is the project benchmark (learner Jund vs the Mono Blue
Terror / Delver bot), see `benchmark`. Scores count a draw as half a win; the
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


def paired_specs(opponent: str, games: int, seed: int = EVAL_SEED, jund_only: bool = False) -> list[GameSpec]:
    specs = []
    for s in range(games // 2):
        if jund_only:  # learner keeps seat 0 (Jund); the pair swaps who starts
            specs.append(GameSpec(seed=seed + s, seats=(LEARNER, opponent), starting_player=0))
            specs.append(GameSpec(seed=seed + s, seats=(LEARNER, opponent), starting_player=1))
            continue
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


def head_to_head(procs, learner_path: str, opponent: str, games: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", jund_only: bool = False) -> dict:
    specs = paired_specs(opponent, games, jund_only=jund_only)
    jobs = [Job(chunk, learner_path, version, record=False, max_turns=max_turns, inference=inference) for chunk in split_games(specs, n_jobs)]
    return score([g for r in procs.map(run_job, jobs) for g in r.games])


def head_to_head_bo3(procs, learner_path: str, opponent: str, matches: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", jund_only: bool = False) -> dict:
    """Best-of-three matches on paired seeds (`jund_only`: one match per seed,
    learner on Jund). Each round plays the next game of every unfinished match
    as one batch. Returns match scores like `score`."""
    if jund_only:
        live = [(EVAL_SEED + s, (LEARNER, opponent), MatchResult()) for s in range(matches)]
    else:
        live = [(EVAL_SEED + s, seats, MatchResult()) for s in range(matches // 2) for seats in ((LEARNER, opponent), (opponent, LEARNER))]
    while True:
        todo = {}
        for seed, seats, res in live:
            if not res.over:
                n, start = res.next_game(seed)
                todo[(game_seed(seed, n), seats)] = (res, GameSpec(seed=game_seed(seed, n), seats=seats, starting_player=start, match_game=n))
        if not todo:
            break
        jobs = [Job(chunk, learner_path, version, record=False, max_turns=max_turns, inference=inference) for chunk in split_games([sp for _, sp in todo.values()], n_jobs)]
        for r in procs.map(run_job, jobs):
            for seats, winner, reason, _, _, seed in r.games:
                res, spec = todo[(seed, seats)]
                res.games.append((spec.starting_player, winner, reason))
    return score([(seats, res.winner) for _, seats, res in live])


def benchmark(procs, learner_path: str, games: int, bo3_matches: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local") -> dict:
    """The fixed benchmark: the learner plays Jund Wildfire against the scripted
    Mono Blue Terror (Delver) bot. Game-1 decks on paired seeds (each seed once
    per starting player), plus best-of-three matches. Same seeds every call."""
    out = {}
    if games:
        s, ci, n = head_to_head(procs, learner_path, BOT, games, n_jobs, version, max_turns, inference, jund_only=True)["jund"]
        out.update({"bench/jund_vs_bot": s, "bench/jund_vs_bot_ci": ci, "bench/jund_vs_bot_n": n})
    if bo3_matches:
        s, ci, n = head_to_head_bo3(procs, learner_path, BOT, bo3_matches, n_jobs, version, max_turns, inference, jund_only=True)["jund"]
        out.update({"bench/jund_vs_bot_bo3": s, "bench/jund_vs_bot_bo3_ci": ci, "bench/jund_vs_bot_bo3_n": n})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.rl.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("opponent", help=f"a checkpoint path, '{RANDOM}' or '{BOT}'")
    ap.add_argument("--games", type=int, default=400, help="games (or matches with --bo3)")
    ap.add_argument("--bo3", action="store_true", help="best-of-three matches with sideboarding")
    ap.add_argument("--jund", action="store_true", help="learner always plays Jund (seat 0); with 'bot' this is the benchmark")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default=None, help="python or native (default: $MTG_ENGINE, else python)")
    args = ap.parse_args(argv)
    if args.engine:
        os.environ[ENV_VAR] = engine_name(args.engine)
    with mp.get_context("spawn").Pool(args.workers, initializer=worker_init) as procs:
        fn = head_to_head_bo3 if args.bo3 else head_to_head
        res = fn(procs, args.checkpoint, args.opponent, args.games, args.workers, jund_only=args.jund)
    for k, (s, ci, n) in res.items():
        print(f"{k:5s} {s:.3f}  95% CI {ci}  ({n} games)")


if __name__ == "__main__":
    main()
