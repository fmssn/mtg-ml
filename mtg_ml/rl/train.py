"""Masked PPO self-play training loop.

    python -m mtg_ml.rl.train --run runs/ppo1 --iterations 200
    python -m mtg_ml.rl.train --run runs/ppo1 --iterations 400   # resumes

One deck-conditioned network plays both seats (the seat is a state feature).
It has a GRU memory over each player's decisions in a game, fed with the
player's own previous choice and the opponent's public actions.
Each iteration:
  1. collect `games_per_iter` games in parallel workers. A fraction
     `self_play_frac` are learner vs learner (both seats recorded), the rest
     are learner vs a frozen checkpoint from the opponent pool (learner seat
     random, only the learner recorded), and `bot_frac` vs the scripted bots;
  2. one PPO update on all recorded decisions;
  3. write the new weights to `policy/vNNNNN.pt` (weights only, what workers,
     the evaluation and the inference server load); every `checkpoint_every`
     iterations and at the end, the resume checkpoint `latest.pt` (weights,
     Adam state, counters, rng; ~3x the weights) in a background thread;
     every `snapshot_every` iterations freeze the weights into `pool/`;
     every `eval_every` iterations (and/or every `eval_every_games` training
     games) evaluate: against the opponents of `eval_blocks` (the random
     agent, the scripted bots, the oldest pool checkpoint; all three are
     saturated sanity checks, hence the small `eval_games`) on fixed paired
     seeds (both seats, both starting players), best-of-three vs the bots,
     the benchmark (`evaluate.benchmark`: learner as Jund Wildfire vs the
     blue Delver bot, `bench_games` sampled and `bench_greedy_games` greedy)
     and, with `--ladder a.pt,b.pt,...`, the reference ladder in place of the
     oldest pool checkpoint (`ladder_games` games against each rung and the
     learner's Elo on the ladder's scale; the rung ratings come from
     `--ladder-ratings`, else `<run>/ladder.json`, rated once by the first
     evaluation if missing); logged as `eval/*`, `bench/*` and `ladder/*`.

A new deck against a fixed opponent (`--matchup`, `--learner-seat`,
`--opponent`, `--init-from`): e.g. Red Madness (seat 1 of `jund_madness`)
against a frozen Jund checkpoint, warm-started from that checkpoint's
weights (the network's rules knowledge carries over; the `self:deck:`
feature tells it which deck it holds). The learner then only plays its
seat, every non-bot game is against `--opponent`, no pool snapshots are
taken, and the evaluation scores the learner's deck (`eval/opponent/<deck>`
in place of pool0, `bench/<deck>_vs_bot*`):

    python -m mtg_ml.rl.train --run runs/red --matchup jund_madness --learner-seat 1 \
        --opponent jund.pt --init-from jund.pt --self-play-frac 0

Processes (`rl/collect.py`). The trainer's main thread only updates and
publishes. The rollout workers belong to a collector process, which
streams the games (`rollout.play`), merges the results and hands the merged
batch over in one shared-memory block (`--collector thread` keeps the pool
and the merge in a thread of the trainer, the old layout). The evaluation
runs in a process of its own with its own pool (`--eval-workers`, local CPU
inference) on cores the run does not use, from the policy file of the
iteration it evaluates (pinned against `KEEP_POLICIES` pruning until it is
done); the loop never waits for it. While an evaluation runs, a newer due one
waits and replaces any older one still waiting, so a slow evaluation skips
iterations instead of piling up. `--eval-process 0` evaluates on the
training pool and blocks, as before.

`--pipeline 1` (the default): while the update of iteration k runs on the
GPU, the workers already play the games of iteration k+1 with the weights of
iteration k, so an iteration takes max(rollout, update) instead of their
sum. That is the usual one-step policy lag: the recorded log-probs are the
behaviour policy's, so the clipped ratio stays valid. A policy file is never
rewritten, so the update cannot change the weights under a rollout that is
still loading them. The trainer is then pinned to its cores (also with
`--trainer-cpus` given) and uses one torch thread per core. `--pipeline 0`
runs the steps one after the other.

CPU layout (`collect.cpu_layout`, printed at start): unless given with
`--trainer-cpus`, `--worker-cpus`, `--eval-cpus` (lists like `32-47,50`),
the trainer takes one core on the GPU's NUMA node (sysfs), the inference
server (`--inference server`) the next, every rollout worker one core from
the other end of the affinity mask, the evaluation up to 8 of the cores left
(none left: it shares the workers' cores, niced), the trainer the rest; the
collector moves between the workers' cores off the trainer's node. Workers
on the trainer's NUMA node slow the update (~12% with 15 of them on the box;
none: as fast as alone), so its node gets workers last.

`--total-games` stops after that many training games (counted across
resumes). Resuming from `latest.pt` at iteration k drops what the lost
iterations after k left behind: pool snapshots and `metrics.jsonl` rows of
later iterations (the rows are moved to `metrics-dropped.jsonl`).

Metrics: `<run>/metrics.jsonl`, one JSON row per iteration, in order.
Besides the PPO statistics (`ppo.ppo_update`: entropy and KL overall, of
the first and last epoch, over the non-trivial decisions and per decision
kind) a row has `rollout_stats`: win rates against the pool (bot games
excluded), per pool opponent and against the bots, decisions per game and
the share of `pay_mana` decisions.
Evaluation results are merged into the row of the iteration they evaluate
when they arrive, a few iterations later (the file is rewritten through a
temporary file and `os.replace`, so a reader sees the old or the new file),
together with `eval_s` (the evaluation's wall time) and `eval_lag`
(iterations trained meanwhile); the final evaluations are merged before
`train()` returns. Timing: `rollout_s` (submit to merged result) is for the
games the row trains on, which with `--pipeline 1` were played during the
previous update; `update_s` covers the update and the publish (and the
checkpoint snapshot when one is due); `wait_s` is the time the iteration
blocked on the next rollout after its update (including mapping its
result); `wall_s` is the whole iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields

import torch

from ..backend import ENV_VAR, engine_name
from ..match import matchup_decks
from .collect import PoolProcess, PoolThread, cpu_layout, release
from .evaluate import EVAL_BLOCKS, evaluate_policy
from .model import PolicyNet
from .ppo import PPOConfig, load_optimizer_state, make_optimizer, ppo_update
from .rollout import BOT, KIND_ID, LEARNER, GameSpec, Job, create_pool, play

TRAIN_SEED_BASE = 1 << 40  # training game seeds start here: disjoint from the fixed evaluation/benchmark seeds (EVAL_SEED + s, and 4x that for best-of-three games)
KEEP_POLICIES = 3  # the newest, the one a lagged rollout may still be loading, one spare (plus those pinned by evaluations)


REQUEST_INTS_PER_GAME = 1 << 15  # worst seen in the overnight run: ~17k int32 words per decision (state, events since the last one, options)


def request_ints_for(games_per_iter: int, workers: int, groups: int = Job.groups) -> int:
    """Request slot size for the inference server: one decision per live game of
    a worker's request group, at REQUEST_INTS_PER_GAME each, rounded up to a
    power of two and at least 1 << 18. With 13 workers and 2048 games per
    iteration a request carries ~79 decisions; 1 << 18 overflowed at 1.3M."""
    games = -(-games_per_iter // max(workers, 1))
    per_request = -(-games // max(groups, 1))
    return max(1 << 18, 1 << (per_request * REQUEST_INTS_PER_GAME - 1).bit_length())


@dataclass
class TrainConfig:
    run: str = "runs/ppo"
    iterations: int = 100
    total_games: int = 0  # stop once this many training games are played (0 = only --iterations)
    games_per_iter: int = 256  # >= ~32 per worker keeps batched inference cheap next to the engine
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    pipeline: int = 1  # 1: play iteration k+1 with the weights of k during the update of k (one-step policy lag); 0: one after the other
    collector: str = "process"  # "process": the worker pool and the result merge in a process of their own; "thread": in a thread of the trainer
    checkpoint_every: int = 10  # write latest.pt (weights + Adam state) every this many iterations, and at the end
    hidden: int = 512  # policy width (128 before 2026-10; 7x455 / 6x512 MLPs are typical for PPO card-game agents)
    trunk: str = "mlp"  # "mlp" or "transformer" (2 layers over the active state features)
    value_net: str = "separate"  # "separate": own embeddings, trunk and memory; "shared": linear head on the policy core
    value_hidden: int = 0  # width of a separate value net (0 = hidden)
    memory: str = "gru"  # "gru" (recurrent over the player's decisions) or "none"
    gamma: float = 0.995
    lam: float = 0.95
    shaping: float = 0.2  # life-difference potential shaping, linearly annealed to 0
    shaping_anneal_iters: int = 100
    self_play_frac: float = 0.5
    bot_frac: float = 0.0  # share of games against the scripted bots (taken out of the pool share)
    postboard_frac: float = 0.5  # share of training games played with sideboarded decks (match games 2/3)
    eval_bo3_matches: int = 100  # best-of-three matches vs the bots at each evaluation (0 = off)
    snapshot_every: int = 10
    pool_recent_frac: float = 0.5  # share of pool games against the newest snapshot
    eval_every: int = 10  # evaluate every this many iterations (0 = only by eval_every_games)
    eval_every_games: int = 0  # also evaluate whenever the training games cross a multiple of this (e.g. 250000)
    eval_process: int = 1  # 1: evaluate in a process of its own, never waiting for it; 0: on the training pool, blocking
    eval_workers: int = 0  # workers of the evaluation process (0: one per evaluation core, at most 8)
    eval_blocks: str = ",".join(EVAL_BLOCKS)  # opponents of the eval/<opponent>/<deck> games, among random, bot, pool0 ("" = none)
    eval_games: int = 40  # per eval block, split over both seats (they saturate: a sanity check)
    bench_games: int = 1000  # benchmark games at each evaluation: learner Jund vs blue bot, sampled (0 = off)
    bench_greedy_games: int = 1000  # the same with greedy play (0 = off)
    bench_bo3_matches: int = 200  # benchmark best-of-three matches at each evaluation (0 = off)
    ladder: str = ""  # reference ladder: comma-separated fixed checkpoints, replacing the pool0 block
    ladder_games: int = 200  # paired games against each rung
    ladder_ratings: str = ""  # JSON with the rungs' Elo (`evaluate ladder`); "" = <run>/ladder.json, rated by the first evaluation if missing
    ladder_greedy: int = 0  # 1: ladder games (and the rating round robin) with greedy play
    max_turns: int = 100
    matchup: str = "jund_blue"  # match.MATCHUPS: the deck in each seat
    learner_seat: int = -1  # -1: the learner plays both seats; 0 / 1: only that seat (its deck)
    opponent: str = ""  # a frozen checkpoint every non-bot game is played against (instead of the pool)
    init_from: str = ""  # start a new run from these weights (its network config wins over --hidden etc.)
    auto_mana: int = 0  # 1: pay non-strategic mana costs automatically (colour-preserving payer; docs/action-decomposition.md)
    auto_pass: int = 0  # 1: auto-pass priority when the only other options are side-effect-free sacrifice-for-mana abilities
    seed: int = 0
    device: str = "cpu"
    engine: str = "python"  # rules engine for rollouts: python (reference) or native (Rust, mtg_ml_native)
    inference: str = "local"  # policy inference: local (CPU torch in each worker) or server (one GPU process, rl/inference.py)
    server_device: str = ""  # device of the inference server ("" = --device, or cuda when --device is cpu and a GPU exists)
    server_max_rows: int = 16384  # largest batch the server builds from queued requests
    server_request_ints: int = 0  # int32 words per worker request slot (0 = auto: `request_ints_for`, from the games each request carries)
    server_policy_slots: int = 32  # policies the server's stack holds before it grows (pool snapshots + learner versions; each ~68 MB for h128 entity)
    trainer_cpus: str = ""  # CPUs of the trainer process, e.g. "32" or "32-35" ("": see the CPU layout)
    worker_cpus: str = ""  # CPUs of the rollout workers, one worker per CPU round robin
    eval_cpus: str = ""  # CPUs of the evaluation workers
    ppo: PPOConfig = field(default_factory=PPOConfig)


@dataclass
class _Rollout:
    """A training rollout in flight on the collector."""

    pending: object  # collect.Pending
    shaping: float
    lag: int  # updates its weights are behind the update that trains on it
    rng: tuple  # trainer rng state before its games were drawn: the checkpoint of its iteration stores it
    data: object = None  # the Result once waited for
    waited: float = 0.0  # seconds the trainer blocked on it

    def wait(self) -> _Rollout:
        if self.data is None:
            t = time.monotonic()
            self.data = self.pending.wait()
            self.waited = time.monotonic() - t
        return self

    @property
    def rollout_s(self) -> float:
        return self.pending.t_ready - self.pending.t_submit


@dataclass
class _Eval:
    iteration: int
    games_total: int
    policy: str
    version: int
    pending: object = None
    t_submit: float = 0.0


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.environ[ENV_VAR] = engine_name(cfg.engine)  # spawned processes inherit it
        for d in ("pool", "policy"):
            os.makedirs(os.path.join(cfg.run, d), exist_ok=True)
        self.latest = os.path.join(cfg.run, "latest.pt")
        self.metrics = os.path.join(cfg.run, "metrics.jsonl")
        self.blocks = tuple(b for b in cfg.eval_blocks.split(",") if b)
        if set(self.blocks) - set(EVAL_BLOCKS):
            raise ValueError(f"eval blocks must be among {EVAL_BLOCKS}, not {cfg.eval_blocks!r}")
        self.ladder = tuple(p for p in cfg.ladder.split(",") if p)
        for path in self.ladder + ((cfg.ladder_ratings,) if cfg.ladder_ratings else ()):
            if not os.path.exists(path):
                raise FileNotFoundError(f"ladder file {path} does not exist")
        self.ladder_ratings = cfg.ladder_ratings or os.path.join(cfg.run, "ladder.json")
        self.evaluating = bool(cfg.eval_every or cfg.eval_every_games)
        if cfg.learner_seat not in (-1, 0, 1):
            raise ValueError("--learner-seat must be -1, 0 or 1")
        if cfg.learner_seat >= 0 and cfg.self_play_frac > 0:
            raise ValueError("--learner-seat: the learner only plays one seat, so --self-play-frac must be 0")
        matchup_decks(cfg.matchup)  # validates the name
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        init = torch.load(cfg.init_from, map_location="cpu", weights_only=False) if cfg.init_from and not os.path.exists(self.latest) else None
        net_cfg = init["config"] if init else dict(hidden=cfg.hidden, memory=cfg.memory, trunk=cfg.trunk, value_net=cfg.value_net, value_hidden=cfg.value_hidden)
        self.net = PolicyNet(**net_cfg).to(cfg.device)
        if init:
            self.net.load_state_dict(init["model"])
            print(f"initialized from {cfg.init_from}: {net_cfg}", flush=True)
        self.opt = make_optimizer(self.net.parameters(), cfg.ppo.lr, cfg.device)
        self.iteration = 0
        self.games_total = 0
        self.checkpointed = -1  # iteration of the newest latest.pt
        self.saver = ThreadPoolExecutor(1, thread_name_prefix="checkpoint")
        self.saving = None  # the latest.pt write in flight
        self.pinned: dict[str, int] = {}  # policy files an evaluation still needs
        if os.path.exists(self.latest):
            ck = torch.load(self.latest, map_location=cfg.device, weights_only=False)
            self.net.load_state_dict(ck["model"])
            load_optimizer_state(self.opt, ck["optim"])
            self.iteration = self.checkpointed = ck["iteration"]
            self.games_total = ck.get("games_total", self.iteration * cfg.games_per_iter)
            self.rng.setstate(ck["rng"])
            if "torch_rng" in ck:  # older checkpoints lack it: the update's minibatch shuffles then restart from cfg.seed
                torch.set_rng_state(ck["torch_rng"])
                if ck.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ck["cuda_rng"])
            self._drop_lost_iterations()
            print(f"resumed {self.latest} at iteration {self.iteration}")
        self._remove_partial_writes()
        pool_dir = os.path.join(cfg.run, "pool")
        self.pool: list[str] = sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.startswith("iter_") and f.endswith(".pt"))
        weights = self._weights()
        self._publish(weights)
        if not self.pool:  # also with --opponent: the initial snapshot is the evaluation's pool0
            self._snapshot(weights)
            self._checkpoint(weights, self.rng.getstate())
        self.layout = cpu_layout(cfg.workers, cfg.device, cfg.inference == "server", bool(self.evaluating and cfg.eval_process),
                                 cfg.trainer_cpus, cfg.worker_cpus, cfg.eval_cpus, available_cpus())
        print(self.layout.describe(), flush=True)
        self.server = None
        self.collector = self._collector()
        self.evaluator = None
        if self.evaluating and cfg.eval_process:
            ev = self.layout.evaluator
            self.eval_workers = cfg.eval_workers or (4 if self.layout.shared_eval else max(1, min(8, len(ev))))
            self.evaluator = PoolProcess(self.eval_workers, "local", worker_cpus=ev or None, cpus=ev or None, nice=10, name="evaluator")
        self.evals: list[_Eval] = []  # running first, then at most one waiting

    def _collector(self):
        c, lay = self.cfg, self.layout
        server_cfg = None
        if c.inference == "server":
            from .inference import ServerConfig, default_device

            dev = c.server_device or (c.device if c.device != "cpu" else default_device())
            server_cfg = ServerConfig(device=dev, max_rows=c.server_max_rows, request_ints=c.server_request_ints or request_ints_for(c.games_per_iter, c.workers), policy_slots=c.server_policy_slots, cpus=lay.server or None)
        if c.collector == "process":  # merging as jobs end, off the trainer's NUMA node if it can
            return PoolProcess(c.workers, c.inference, server_cfg, lay.workers or None, lay.collector or None)
        if c.collector != "thread":
            raise ValueError(f"collector must be process or thread, not {c.collector!r}")
        if c.inference == "server":
            from .inference import InferenceServer

            self.server = InferenceServer(c.workers, server_cfg, lay.workers or None)
        return PoolThread(create_pool(c.workers, c.inference, self.server, worker_cpus=lay.workers or None))

    # -- checkpoints ---------------------------------------------------------

    def _weights(self) -> dict:
        """CPU copies of the weights: they are written while the next update changes the originals."""
        return _cpu(self.net.state_dict())

    def _publish(self, weights: dict) -> None:
        """Write the weights that rollouts and evaluations load from now on. A new
        file per version: a rollout still loading the previous one is unaffected."""
        d = os.path.join(self.cfg.run, "policy")
        self.policy = _save({"config": self.net.config, "model": weights}, os.path.join(d, f"v{self.iteration:05d}.pt"))
        self._prune_policies()

    def _prune_policies(self) -> None:
        d = os.path.join(self.cfg.run, "policy")
        old = sorted((f for f in os.listdir(d) if f.startswith("v") and f.endswith(".pt")), key=lambda f: int(f[1:-3]))
        for f in old[:-KEEP_POLICIES]:
            if os.path.join(d, f) not in self.pinned:
                os.remove(os.path.join(d, f))

    def _snapshot(self, weights: dict) -> None:
        """Freeze the weights into the opponent pool (only rollout workers load them)."""
        self.pool.append(_save({"config": self.net.config, "model": weights}, os.path.join(self.cfg.run, "pool", f"iter_{self.iteration:05d}.pt")))

    def _checkpoint(self, weights: dict, rng: tuple) -> None:
        """Write the resume checkpoint `latest.pt` in the background: with the
        Adam state it is ~3x the weights, and the next update need not wait."""
        self._join_checkpoint()
        ck = {
            "config": self.net.config,
            "model": weights,
            "optim": _cpu(self.opt.state_dict()),
            "iteration": self.iteration,
            "games_total": self.games_total,
            "rng": rng,
            "torch_rng": torch.get_rng_state(),  # the update's minibatch shuffles
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_config": asdict(self.cfg),
        }
        self.saving = self.saver.submit(_save, ck, self.latest)
        self.checkpointed = self.iteration

    def _join_checkpoint(self) -> None:
        if self.saving is not None:
            self.saving.result()  # re-raises a failed write
            self.saving = None

    def _remove_partial_writes(self) -> None:
        """Delete `*.tmp` files a killed run left behind mid-`_save`."""
        for d in (self.cfg.run, os.path.join(self.cfg.run, "pool"), os.path.join(self.cfg.run, "policy")):
            for f in os.listdir(d):
                if f.endswith(".tmp"):
                    os.remove(os.path.join(d, f))

    def _drop_lost_iterations(self) -> None:
        """After resuming at iteration k: policy files, pool snapshots and
        metrics rows of the iterations after k are from a past that did not
        happen (and newer policy files would make `_prune_policies` delete
        the one being published)."""
        k = self.iteration
        for sub, prefix in (("pool", "iter_"), ("policy", "v")):
            d = os.path.join(self.cfg.run, sub)
            for f in os.listdir(d):
                if f.startswith(prefix) and f.endswith(".pt") and int(f[len(prefix) : -3]) > k:
                    os.remove(os.path.join(d, f))
        if os.path.exists(self.metrics):
            with open(self.metrics) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            lost = [ln for ln in lines if json.loads(ln)["iteration"] > k]
            if lost:
                with open(os.path.join(self.cfg.run, "metrics-dropped.jsonl"), "a") as f:
                    f.writelines(ln + "\n" for ln in lost)
                _write_lines(self.metrics, [ln for ln in lines if json.loads(ln)["iteration"] <= k])

    # -- rollouts ------------------------------------------------------------

    def _submit(self, it: int) -> _Rollout:
        """Start the training rollout for the update of iteration `it` (0-based)
        with the newest policy file, without waiting for it. The games are
        drawn here, in the main thread (`self.rng`)."""
        c, rng = self.cfg, self.rng.getstate()
        shaping = c.shaping * max(0.0, 1 - it / max(c.shaping_anneal_iters, 1))
        job = Job([], self.policy, self.iteration + 1, record=True, gamma=c.gamma, lam=c.lam, shaping=shaping, max_turns=c.max_turns, inference=c.inference, auto_mana=bool(c.auto_mana), auto_pass=bool(c.auto_pass))
        specs = self._train_specs(it)
        return _Rollout(self.collector.call(play, specs, job, c.workers), shaping, it - self.iteration, rng)

    def _train_specs(self, it: int) -> list[GameSpec]:
        c, specs = self.cfg, []
        base = TRAIN_SEED_BASE + (it + 1) * 1_000_003 + c.seed * 7919
        for k in range(c.games_per_iter):
            r = self.rng.random()
            if r < c.self_play_frac:
                seats = (LEARNER, LEARNER)
            else:
                if r < c.self_play_frac + c.bot_frac:
                    opp = BOT
                elif c.opponent:
                    opp = c.opponent
                else:
                    opp = self.pool[-1] if self.rng.random() < c.pool_recent_frac else self.rng.choice(self.pool)
                seat = c.learner_seat if c.learner_seat >= 0 else (0 if self.rng.random() < 0.5 else 1)
                seats = (LEARNER, opp) if seat == 0 else (opp, LEARNER)
            game_no = 2 if self.rng.random() < c.postboard_frac else 1
            specs.append(GameSpec(seed=base + k, seats=seats, match_game=game_no, matchup=c.matchup))
        return specs

    # -- evaluation ----------------------------------------------------------

    def _eval_args(self, e: _Eval, n_jobs: int, inference: str) -> tuple:
        c = self.cfg
        return (e.policy, self.pool[0], e.version, n_jobs, c.eval_games, c.eval_bo3_matches, c.bench_games, c.bench_bo3_matches, c.max_turns, inference,
                self.blocks, c.bench_greedy_games, self.ladder, c.ladder_games, self.ladder_ratings, bool(c.ladder_greedy), bool(c.auto_mana), bool(c.auto_pass),
                c.matchup, None if c.learner_seat < 0 else c.learner_seat, c.opponent)

    def _request_eval(self) -> dict:
        """Evaluate the newest policy file. Inline (`--eval-process 0`): on the
        training pool after the rollout in flight, returning the results.
        Otherwise queued for the evaluation process; returns {}."""
        e = _Eval(self.iteration, self.games_total, self.policy, self.iteration + 1)
        if self.evaluator is None:
            t = time.monotonic()
            out = self.collector.call(evaluate_policy, *self._eval_args(e, self.cfg.workers, self.cfg.inference)).wait()
            return {**out, "eval_s": round(time.monotonic() - t, 2)}
        if len(self.evals) > 1:  # a newer request replaces the one still waiting
            self._unpin(self.evals.pop().policy)
        self.pinned[e.policy] = self.pinned.get(e.policy, 0) + 1
        self.evals.append(e)
        self._start_eval()
        return {}

    def _start_eval(self) -> None:
        if self.evals and self.evals[0].pending is None:
            e = self.evals[0]
            e.t_submit = time.monotonic()
            e.pending = self.evaluator.call(evaluate_policy, *self._eval_args(e, self.eval_workers, "local"))

    def _unpin(self, path: str) -> None:
        self.pinned[path] -= 1
        if not self.pinned[path]:
            del self.pinned[path]
        self._prune_policies()

    def _collect_evals(self, block: bool = False) -> None:
        """Merge finished evaluations into their rows (all of them if `block`)."""
        while self.evals and (block or self.evals[0].pending.done()):
            e = self.evals.pop(0)
            out = e.pending.wait()
            self._unpin(e.policy)
            out.update(eval_s=round(time.monotonic() - e.t_submit, 2), eval_lag=self.iteration - e.iteration)
            _amend(self.metrics, e.iteration, out)
            print(_fmt({"iteration": e.iteration, **out}), flush=True)
            self._start_eval()

    # -- main loop -----------------------------------------------------------

    def _eval_due(self, games: int) -> bool:
        """Whether the iteration just done (which played `games`) is evaluated."""
        c, g = self.cfg, self.cfg.eval_every_games
        return bool((c.eval_every and self.iteration % c.eval_every == 0) or (g and self.games_total // g > (self.games_total - games) // g))

    def _more(self, iteration: int, games_total: int) -> bool:
        """Whether an iteration starting at these counters runs."""
        c = self.cfg
        return iteration < c.iterations and not (c.total_games and games_total >= c.total_games)

    def train(self) -> None:
        c = self.cfg
        threads = torch.get_num_threads()
        affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        if c.pipeline or c.trainer_cpus:  # the update shares the CPUs with the workers: spinning OMP threads on busy cores slowed it by 60%
            torch.set_num_threads(max(1, min(threads, len(self.layout.trainer))))
            if affinity is not None and self.layout.trainer:
                os.sched_setaffinity(0, set(self.layout.trainer))
        roll = None  # the collected rollout for the coming update, if it was played during the last one
        ok = False
        try:
            while self._more(self.iteration, self.games_total):
                t0 = time.monotonic()
                if roll is None:
                    roll = self._submit(self.iteration).wait()
                data = roll.data
                nxt = None
                if c.pipeline and self._more(self.iteration + 1, self.games_total + len(data.games)):
                    nxt = self._submit(self.iteration + 1)  # plays with the current weights during the update
                t1 = time.monotonic()
                stats = ppo_update(self.net, self.opt, data, c.ppo, device=c.device)
                release(data)  # unmaps the shared block of the samples
                self.iteration += 1
                self.games_total += len(data.games)
                weights = self._weights()
                self._publish(weights)
                if self.iteration % c.snapshot_every == 0 and not c.opponent:
                    self._snapshot(weights)
                if c.checkpoint_every and self.iteration % c.checkpoint_every == 0:
                    self._checkpoint(weights, nxt.rng if nxt else self.rng.getstate())
                t2 = time.monotonic()

                row = {
                    "iteration": self.iteration,
                    "decisions": len(data.actions),
                    "games": len(data.games),
                    "games_total": self.games_total,
                    "game_turns": sum(g[3] for g in data.games) / max(len(data.games), 1),
                    "draws": sum(g[1] is None for g in data.games) / max(len(data.games), 1),
                    "jund_wins_selfplay": _rate([g for g in data.games if g[0] == (LEARNER, LEARNER)], 0),
                    **rollout_stats(data.games, data.kinds),
                    "shaping": roll.shaping,
                    "pool_size": len(self.pool),
                    "rollout_s": round(roll.rollout_s, 2),
                    "update_s": round(t2 - t1, 2),
                    "policy_lag": roll.lag,
                    "decisions_per_s": round(len(data.actions) / max(roll.rollout_s, 1e-9)),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                }
                if self._eval_due(len(data.games)):
                    row.update(self._request_eval())
                if nxt:
                    nxt.wait()
                row["wait_s"] = round(nxt.waited if nxt else 0.0, 2)
                row["wall_s"] = round(time.monotonic() - t0, 2)
                _append(self.metrics, row)
                print(_fmt(row), flush=True)
                if self.evaluator is not None:
                    self._collect_evals()
                roll = nxt
            if self.checkpointed != self.iteration:
                self._checkpoint(self._weights(), self.rng.getstate())
            if self.evaluator is not None:
                self._collect_evals(block=True)
            ok = True
        finally:
            for p in (self.collector, self.evaluator):
                if p is not None:
                    p.close() if ok else p.terminate()  # terminate: abandon a rollout or evaluation in flight
            if self.server is not None:
                self.server.close()
            torch.set_num_threads(threads)
            if affinity is not None:
                os.sched_setaffinity(0, affinity)
            self._join_checkpoint()
            self.saver.shutdown()


def _save(obj: dict, path: str) -> str:
    """torch.save through a temporary file: readers see the old file or the new one, never part of one."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def _cpu(x):
    """CPU copies of the tensors in a (nested) state dict (`.cpu()` returns CPU tensors themselves)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_cpu(v) for v in x]
    return x


def available_cpus() -> tuple[int, ...]:
    """CPUs this process may use (its affinity)."""
    return tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else tuple(range(os.cpu_count() or 1))


def _append(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _write_lines(path: str, lines: list[str]) -> None:
    """Replace a text file atomically: readers see the old file or the new one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(ln + "\n" for ln in lines)
    os.replace(tmp, path)


def _amend(path: str, iteration: int, values: dict) -> None:
    """Merge `values` into the newest row of `iteration` in a JSON-lines file
    (appended as a row of its own if there is none)."""
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        row = json.loads(lines[i])
        if row.get("iteration") == iteration:
            row.update(values)
            lines[i] = json.dumps(row)
            _write_lines(path, lines)
            return
    _append(path, {"iteration": iteration, **values})


def rollout_stats(games: list, kinds) -> dict:
    """Row statistics of a training rollout from its `Result.games` rows
    (seats, winner, end reason, turns, decisions, seed) and recorded decision
    kinds: the learner's win rate against pool checkpoints (`win_vs_pool`;
    self-play and bot games excluded), per pool opponent
    (`win_vs_pool_by_opp`: {snapshot name: [win rate, learner games]}),
    against the scripted bots (`win_vs_bot`), decisions per game (every
    seat's, forced moves included) and the share of the recorded learner
    decisions that are mana payments (`pay_mana_share`)."""
    by_opp: dict[str, list[int]] = {}
    pool_won = pool_n = bot_won = bot_n = 0
    for g in games:
        for s in (0, 1):
            opp = g[0][1 - s]
            if g[0][s] != LEARNER or opp == LEARNER:
                continue
            won = int(g[1] == s)
            if opp == BOT:
                bot_won, bot_n = bot_won + won, bot_n + 1
                continue
            pool_won, pool_n = pool_won + won, pool_n + 1
            w = by_opp.setdefault(os.path.basename(opp).removesuffix(".pt"), [0, 0])
            w[0] += won
            w[1] += 1
    pay = KIND_ID["pay_mana"]
    return {
        "win_vs_pool": pool_won / max(pool_n, 1),
        "win_vs_pool_by_opp": {k: [round(w / n, 4), n] for k, (w, n) in sorted(by_opp.items())},
        "win_vs_bot": bot_won / max(bot_n, 1),
        "decisions_per_game": sum(g[4] for g in games) / max(len(games), 1),
        "pay_mana_share": kinds.count(pay) / len(kinds) if len(kinds) else float("nan"),
    }


def _rate(games, seat: int) -> float:
    return sum(g[1] == seat for g in games) / len(games) if games else float("nan")


def _fmt(row: dict) -> str:
    keys = ["iteration", "decisions", "game_turns", "draws", "jund_wins_selfplay", "win_vs_pool", "entropy", "approx_kl", "explained_var", "decisions_per_s", "wall_s"]
    s = " ".join(f"{k}={row[k]:.3f}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in keys if k in row)
    ev = {k.split("/", 1)[1]: v for k, v in row.items() if k.startswith(("eval/", "bench/")) and not k.endswith(("_ci", "_n"))}
    return s + (" | eval " + " ".join(f"{k}={v:.2f}" for k, v in ev.items()) if ev else "")


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(prog="mtg_ml.rl.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg, ppo = TrainConfig(), PPOConfig()
    for obj, prefix in ((cfg, ""), (ppo, "ppo-")):
        for f in fields(obj):
            if f.name == "ppo":
                continue
            v = getattr(obj, f.name)
            typ = float if f.name == "target_kl" else type(v)
            ap.add_argument(f"--{prefix}{f.name.replace('_', '-')}", type=typ, default=v)
    a = vars(ap.parse_args(argv))
    for f in fields(ppo):
        setattr(ppo, f.name, a.pop("ppo_" + f.name))
    return TrainConfig(**a, ppo=ppo)


def main(argv=None) -> None:
    Trainer(parse_args(argv)).train()


if __name__ == "__main__":
    main()
