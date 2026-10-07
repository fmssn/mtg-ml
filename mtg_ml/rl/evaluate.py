"""Head-to-head evaluation on paired seeds.

    python -m mtg_ml.rl.evaluate runs/a/latest.pt runs/b/latest.pt --games 600
    python -m mtg_ml.rl.evaluate runs/a/latest.pt random
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot      # scripted bots
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --bo3   # best-of-three matches
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --jund  # the benchmark: learner Jund vs the blue bot
    python -m mtg_ml.rl.evaluate runs/a/latest.pt bot --jund --greedy   # ... with argmax play
    python -m mtg_ml.rl.evaluate ladder a.pt b.pt c.pt --games 200 --out ladder.json   # rate a reference ladder

Every seed is played twice with the seats swapped, so both sides see the
same shuffles from both decks. With --jund the learner always plays Jund
Wildfire (seat 0) and every seed is played once per starting player instead;
against `bot` that is the project benchmark (learner Jund vs the Mono Blue
Terror / Delver bot), see `benchmark`. Scores count a draw as half a win; the
interval is Wilson 95%. With --bo3 each seed is a best-of-three match
(`mtg_ml.match` rules: sideboarded games 2/3, loser starts the next game).
`evaluate_policy` is the trainer's periodic evaluation, run in a process of
its own (`rl/collect.py`, `rl/train.py`).

--greedy: every network (both sides when the opponent is a checkpoint)
takes its most likely option instead of sampling. The engine, the bots and
the random agent are seeded, so on paired seeds a greedy evaluation is
deterministic: the same checkpoint scores exactly the same every time, and
its spread across checkpoints is only the seeds' (the Wilson interval still
describes how well N seeds pin down the score). Sampled play stays the
default; the two measure different things (the policy as trained vs its mode).

Reference ladder: `ladder` plays a round robin of fixed checkpoints on paired
seeds and writes their Elo ratings (`fit_ratings`, the first checkpoint at
0) as JSON. The trainer's `--ladder` plays the learner against each rung and
fits the learner's Elo on that scale (`fit_elo`): unlike the score against
one opponent, it does not saturate as long as the ladder has a rung near
the learner.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from ..backend import ENV_VAR, engine_name
from ..match import MatchResult, game_seed
from .rollout import BOT, LEARNER, RANDOM, GameSpec, Job, create_pool, play

EVAL_SEED = 10_000_000
ELO = 400 / math.log(10)  # Elo points per natural-log unit of odds
EVAL_BLOCKS = ("random", "bot", "pool0")


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


def head_to_head(procs, learner_path: str, opponent: str, games: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", jund_only: bool = False, greedy: bool = False,
                 auto_mana: bool = False, auto_pass: bool = False) -> dict:
    specs = paired_specs(opponent, games, jund_only=jund_only)
    job = Job([], learner_path, version, record=False, max_turns=max_turns, inference=inference, greedy=greedy, auto_mana=auto_mana, auto_pass=auto_pass)
    return score(play(procs, specs, job, n_jobs).games)


def head_to_head_bo3(procs, learner_path: str, opponent: str, matches: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", jund_only: bool = False, greedy: bool = False,
                     auto_mana: bool = False, auto_pass: bool = False) -> dict:
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
        job = Job([], learner_path, version, record=False, max_turns=max_turns, inference=inference, greedy=greedy, auto_mana=auto_mana, auto_pass=auto_pass)
        for seats, winner, reason, _, _, seed in play(procs, [sp for _, sp in todo.values()], job, n_jobs).games:
            res, spec = todo[(seed, seats)]
            res.games.append((spec.starting_player, winner, reason))
    return score([(seats, res.winner) for _, seats, res in live])


def benchmark(procs, learner_path: str, games: int, bo3_matches: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", greedy_games: int = 0,
              auto_mana: bool = False, auto_pass: bool = False) -> dict:
    """The fixed benchmark: the learner plays Jund Wildfire against the scripted
    Mono Blue Terror (Delver) bot. Game-1 decks on paired seeds (each seed once
    per starting player), sampled (`games`) and greedy (`greedy_games`,
    keys `bench/jund_vs_bot_greedy*`), plus best-of-three matches. Same seeds
    every call."""
    out = {}
    for n_games, sfx, greedy in ((games, "", False), (greedy_games, "_greedy", True)):
        if n_games:
            s, ci, n = head_to_head(procs, learner_path, BOT, n_games, n_jobs, version, max_turns, inference, jund_only=True, greedy=greedy, auto_mana=auto_mana, auto_pass=auto_pass)["jund"]
            out.update({f"bench/jund_vs_bot{sfx}": s, f"bench/jund_vs_bot{sfx}_ci": ci, f"bench/jund_vs_bot{sfx}_n": n})
    if bo3_matches:
        s, ci, n = head_to_head_bo3(procs, learner_path, BOT, bo3_matches, n_jobs, version, max_turns, inference, jund_only=True, auto_mana=auto_mana, auto_pass=auto_pass)["jund"]
        out.update({"bench/jund_vs_bot_bo3": s, "bench/jund_vs_bot_bo3_ci": ci, "bench/jund_vs_bot_bo3_n": n})
    return out


def evaluate_policy(procs, policy: str, pool0: str, version: int, n_jobs: int, eval_games: int, eval_bo3_matches: int, bench_games: int, bench_bo3_matches: int,
                    max_turns: int = 100, inference: str = "local", blocks: tuple = EVAL_BLOCKS, bench_greedy_games: int = 0,
                    ladder: tuple = (), ladder_games: int = 200, ladder_ratings: str = "", ladder_greedy: bool = False,
                    auto_mana: bool = False, auto_pass: bool = False) -> dict:
    """The trainer's evaluation of a policy file: learner vs the opponents of
    `blocks` (the random agent, the scripted bots, the oldest pool snapshot
    `pool0`) on paired seeds (game 1 decks), best-of-three matches against
    the bots, the benchmark (sampled and greedy) and, with `ladder`
    (checkpoint paths) in place of `pool0`, the reference ladder
    (`ladder_eval`; rung ratings from `load_or_rate_ladder`). Keys
    `eval/<opponent>/<deck>` (+ `_ci`), `bench/*` and `ladder/*`. Runs on
    any pool (`rl.collect` calls it as `fn(pool, *args)`)."""
    out = {}
    opponents = {"random": RANDOM, "bot": BOT, "pool0": pool0}
    for name in blocks:
        if eval_games and not (name == "pool0" and ladder):
            res = head_to_head(procs, policy, opponents[name], eval_games, n_jobs, version, max_turns, inference, auto_mana=auto_mana, auto_pass=auto_pass)
            for deck in ("jund", "blue"):
                out[f"eval/{name}/{deck}"], out[f"eval/{name}/{deck}_ci"], _ = res[deck]
    if eval_bo3_matches:
        res = head_to_head_bo3(procs, policy, BOT, eval_bo3_matches, n_jobs, version, max_turns, inference, auto_mana=auto_mana, auto_pass=auto_pass)
        for deck in ("jund", "blue"):
            out[f"eval/bot_bo3/{deck}"], out[f"eval/bot_bo3/{deck}_ci"], _ = res[deck]
    out.update(benchmark(procs, policy, bench_games, bench_bo3_matches, n_jobs, version, max_turns, inference, bench_greedy_games, auto_mana, auto_pass))
    if ladder and ladder_games:
        ratings = load_or_rate_ladder(procs, list(ladder), ladder_ratings, ladder_games, n_jobs, max_turns, inference, ladder_greedy, auto_mana, auto_pass)
        out.update(ladder_eval(procs, policy, ratings, ladder_games, n_jobs, version, max_turns, inference, ladder_greedy, auto_mana, auto_pass))
    return out


# -- reference ladder and Elo ------------------------------------------------


def _expected(diff: float) -> float:
    """Expected score at an Elo difference `diff` (own minus opponent's)."""
    return 1 / (1 + 10 ** (-diff / 400))


def fit_elo(results: list[tuple[float, float, int]], prior_games: float = 1.0) -> tuple[float, float]:
    """Maximum-likelihood Elo of a player from (opponent rating, score in
    [0, 1], games) against fixed opponents, and its standard error. Draws
    count as half a win (score). Each opponent also adds `prior_games`
    virtual drawn games, so a clean sweep gives a finite rating. The score
    equation sum n (s - E) = 0 is monotone in the rating: bisection."""
    rows = [(r, (s * n + 0.5 * prior_games) / (n + prior_games), n + prior_games) for r, s, n in results if n + prior_games > 0]
    if not rows:
        return float("nan"), float("nan")
    grad = lambda x: sum(n * (s - _expected(x - r)) for r, s, n in rows)  # noqa: E731
    lo, hi = min(r for r, _, _ in rows) - 4000, max(r for r, _, _ in rows) + 4000
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if grad(mid) > 0 else (lo, mid)
    x = (lo + hi) / 2
    info = sum(n * _expected(x - r) * (1 - _expected(x - r)) for r, _, n in rows) / ELO**2
    return x, (1 / math.sqrt(info) if info > 0 else float("inf"))


def fit_ratings(n_players: int, results: list, prior_games: float = 1.0, iters: int = 5000) -> list[float]:
    """Elo ratings of `n_players` from pairwise results (i, j, score of i in
    [0, 1], games): maximum likelihood under the Elo (Bradley-Terry) model,
    with `prior_games` virtual drawn games per pair that played, by Hunter's
    MM iteration. Player 0 is rated 0."""
    wins = [0.0] * n_players
    games: dict[tuple[int, int], float] = {}
    for i, j, s, n in results:
        a, b = min(i, j), max(i, j)
        games[(a, b)] = games.get((a, b), 0.0) + n + prior_games
        wins[i] += s * n + 0.5 * prior_games
        wins[j] += (1 - s) * n + 0.5 * prior_games
    g = [1.0] * n_players
    for _ in range(iters):
        denom = [0.0] * n_players
        for (a, b), n in games.items():
            denom[a] += n / (g[a] + g[b])
            denom[b] += n / (g[a] + g[b])
        new = [wins[k] / denom[k] if denom[k] else g[k] for k in range(n_players)]
        new = [x / new[0] for x in new]
        done = max(abs(math.log(x / y)) for x, y in zip(new, g)) < 1e-12
        g = new
        if done:
            break
    return [ELO * math.log(x) for x in g]


def rung_names(paths: list[str]) -> list[str]:
    """Short names for ladder rungs: the file name without `.pt` (with its
    index when names repeat)."""
    stems = [os.path.basename(p).removesuffix(".pt") for p in paths]
    return [f"{k}_{s}" if stems.count(s) > 1 else s for k, s in enumerate(stems)]


def rate_ladder(procs, paths: list[str], games: int, n_jobs: int, max_turns: int = 100, inference: str = "local", greedy: bool = False,
                auto_mana: bool = False, auto_pass: bool = False) -> dict:
    """Round robin of the checkpoints `paths` on paired seeds, `games` per
    pair. Returns the ratings JSON: {"ratings": {path: Elo}, "games",
    "greedy", "auto_mana", "auto_pass", "results": [[i, j, score of i, games], ...]}."""
    results = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            s, _, n = head_to_head(procs, paths[i], paths[j], games, n_jobs, 0, max_turns, inference, greedy=greedy, auto_mana=auto_mana, auto_pass=auto_pass)["all"]
            results.append([i, j, s, n])
    elo = fit_ratings(len(paths), results)
    return {"ratings": {p: round(r, 1) for p, r in zip(paths, elo)}, "games": games, "greedy": greedy, "auto_mana": auto_mana, "auto_pass": auto_pass, "results": results}


def load_or_rate_ladder(procs, paths: list[str], path: str, games: int, n_jobs: int, max_turns: int = 100, inference: str = "local", greedy: bool = False,
                        auto_mana: bool = False, auto_pass: bool = False) -> dict:
    """{rung path: Elo} from the ratings JSON at `path` if it rates every
    rung; otherwise rate the ladder (`rate_ladder`) and write it there."""
    if path and os.path.exists(path):
        with open(path) as f:
            ratings = json.load(f)["ratings"]
        if all(p in ratings for p in paths):
            return {p: ratings[p] for p in paths}
    out = rate_ladder(procs, paths, games, n_jobs, max_turns, inference, greedy, auto_mana, auto_pass)
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, path)
    return out["ratings"]


def ladder_eval(procs, policy: str, ratings: dict, games: int, n_jobs: int, version: int = 0, max_turns: int = 100, inference: str = "local", greedy: bool = False,
                auto_mana: bool = False, auto_pass: bool = False) -> dict:
    """The learner against every rung of the ladder ({path: Elo}) on paired
    seeds (`games` each, both seats): keys `ladder/<rung>` (score) with
    `_ci`, and the learner's fitted Elo on the ladder's scale `ladder/elo`
    with its standard error `ladder/elo_se`."""
    out, results = {}, []
    paths = list(ratings)
    for name, p in zip(rung_names(paths), paths):
        s, ci, n = head_to_head(procs, policy, p, games, n_jobs, version, max_turns, inference, greedy=greedy, auto_mana=auto_mana, auto_pass=auto_pass)["all"]
        out[f"ladder/{name}"], out[f"ladder/{name}_ci"] = s, ci
        results.append((ratings[p], s, n))
    elo, se = fit_elo(results)
    out["ladder/elo"], out["ladder/elo_se"] = round(elo, 1), round(se, 1)
    return out


def ladder_main(argv) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.rl.evaluate ladder", description="Rate a reference ladder: round robin of fixed checkpoints, Elo ratings as JSON (the first at 0).")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--games", type=int, default=200, help="games per pair, split over both seats")
    ap.add_argument("--out", default="ladder.json")
    ap.add_argument("--greedy", action="store_true", help="argmax play instead of sampling")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default=None, help="python or native (default: $MTG_ENGINE, else python)")
    ap.add_argument("--auto-mana", action="store_true", help="games with Game(auto_mana=True), for policies trained with it")
    ap.add_argument("--auto-pass", action="store_true", help="games with Game(auto_pass=True), for policies trained with it")
    args = ap.parse_args(argv)
    if len(args.checkpoints) < 2:
        ap.error("a ladder needs at least two checkpoints")
    if args.engine:
        os.environ[ENV_VAR] = engine_name(args.engine)
    with create_pool(args.workers) as procs:
        res = rate_ladder(procs, args.checkpoints, args.games, args.workers, greedy=args.greedy, auto_mana=args.auto_mana, auto_pass=args.auto_pass)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, args.out)
    for name, (p, r) in zip(rung_names(list(res["ratings"])), res["ratings"].items()):
        print(f"{name:20s} {r:7.1f}  {p}")
    print(f"wrote {args.out}")


def main(argv=None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "ladder":
        return ladder_main(argv[1:])
    ap = argparse.ArgumentParser(prog="mtg_ml.rl.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("opponent", help=f"a checkpoint path, '{RANDOM}' or '{BOT}'")
    ap.add_argument("--games", type=int, default=400, help="games (or matches with --bo3)")
    ap.add_argument("--bo3", action="store_true", help="best-of-three matches with sideboarding")
    ap.add_argument("--jund", action="store_true", help="learner always plays Jund (seat 0); with 'bot' this is the benchmark")
    ap.add_argument("--greedy", action="store_true", help="networks take their most likely option instead of sampling (deterministic on paired seeds)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default=None, help="python or native (default: $MTG_ENGINE, else python)")
    ap.add_argument("--auto-mana", action="store_true", help="games with Game(auto_mana=True), for policies trained with it")
    ap.add_argument("--auto-pass", action="store_true", help="games with Game(auto_pass=True), for policies trained with it")
    args = ap.parse_args(argv)
    if args.engine:
        os.environ[ENV_VAR] = engine_name(args.engine)
    with create_pool(args.workers) as procs:
        fn = head_to_head_bo3 if args.bo3 else head_to_head
        res = fn(procs, args.checkpoint, args.opponent, args.games, args.workers, jund_only=args.jund, greedy=args.greedy, auto_mana=args.auto_mana, auto_pass=args.auto_pass)
    for k, (s, ci, n) in res.items():
        print(f"{k:5s} {s:.3f}  95% CI {ci}  ({n} games)")


if __name__ == "__main__":
    main()
