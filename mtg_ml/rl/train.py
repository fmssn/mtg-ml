"""Masked PPO self-play training loop.

    python -m mtg_ml.rl.train --run runs/ppo1 --iterations 200
    python -m mtg_ml.rl.train --run runs/ppo1 --iterations 400   # resumes
    python -m mtg_ml.rl.train --run runs/attn1 --trunk entity --entity-attn 1 --init runs/ppo1/latest.pt   # fine-tunes

One deck-conditioned network plays both seats (the seat is a state feature).
It has a GRU memory over each player's decisions in a game, fed with the
player's own previous choice and the opponent's public actions.
Each iteration:
  1. collect `games_per_iter` games in parallel workers. A fraction
     `self_play_frac` are learner vs learner (both seats recorded), the rest
     are learner vs a frozen checkpoint from the opponent pool (learner seat
     random, only the learner recorded), and `bot_frac` vs the scripted bots
     (`--pool-sampling pfsp`, `--bot-seat jund`, exploiter mode `--exploit`,
     lr anneal, per-turn discounting: docs/training-options.md);
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

A new deck against a fixed opponent: `--matchup` picks each seat's deck,
and exploiter mode puts the learner on one of them, e.g. Red Madness
(`--exploit-deck red`, seat 1 of `jund_madness`) against a frozen Jund
checkpoint (`--exploit`), starting from that checkpoint's weights (the
network's rules knowledge carries over; the `self:deck:` feature tells it
which deck it holds). The evaluation then scores the learner's deck, with
the frozen policy in place of pool0 (`eval/opponent/<deck>`,
`bench/<deck>_vs_bot*`):

    python -m mtg_ml.rl.train --run runs/red --matchup jund_madness --exploit jund.pt --exploit-deck red

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
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields

import torch

from ..backend import ENGINES, ENV_VAR, engine_name
from ..encode import BELIEF_FEATURES, FEATURES, FEATURE_VERSIONS, information_contract
from ..match import game_seed, matchup_decks, parse_matchups
from .collect import PoolProcess, PoolThread, cpu_layout, fmt_cpus, parse_cpus, release
from .evaluate import DECK_KEYS, EVAL_BLOCKS, evaluate_policy
from .belief import BeliefSpec
from .model import PolicyNet, load_partial
from .ppo import PPOConfig, load_optimizer_state, make_optimizer, ppo_update, set_lr
from .rollout import BOT, KIND_ID, LEARNER, GameSpec, Job, checkpoint_config, create_pool, play

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


GAMES_PER_MATCH = 2.5  # --match-rollouts: matches per iteration = games_per_iter / this; budgets count the games actually played


@dataclass
class TrainConfig:
    run: str = "runs/ppo"
    iterations: int = 100
    total_games: int = 0  # stop once this many training games are played (0 = only --iterations)
    duration_seconds: float = 0.0  # wall-clock budget for this invocation; finish the current update before stopping
    games_per_iter: int = 256  # >= ~32 per worker keeps batched inference cheap next to the engine
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    pipeline: int = 1  # 1: play iteration k+1 with the weights of k during the update of k (one-step policy lag); 0: one after the other
    collector: str = "process"  # "process": the worker pool and the result merge in a process of their own; "thread": in a thread of the trainer
    checkpoint_every: int = 10  # write latest.pt (weights + Adam state) every this many iterations, and at the end
    hidden: int = 512  # policy width (128 before 2026-10; 7x455 / 6x512 MLPs are typical for PPO card-game agents)
    trunk: str = "mlp"  # "mlp" or "transformer" (2 layers over the active state features)
    value_net: str = "separate"  # "separate": own embeddings, trunk and memory; "shared": linear head on the policy core
    value_hidden: int = 0  # width of a separate value net (0 = hidden)
    value_bound: str = "none"  # "tanh": squash the value head into (-1, 1) (new runs and --init only; a resumed run keeps its checkpoint's)
    value_clamp: float = 1.0  # GAE clamps unshaped values to +-this, then subtracts shaping * phi; 0 = raw values
    entity_attn: int = 0  # trunk "entity": self-attention layers over each decision's entities (4 heads, FFN 2x; 0 = none)
    features: int = 0  # feature-set version the policy reads (encode.FEATURE_VERSIONS; docs/features.md). 0: a new run takes the latest, --init / --exploit the source's, a resumed run its checkpoint's; a version overrides it (and is written into the config)
    memory: str = "gru"  # "gru" (recurrent over the player's decisions) or "none"
    gamma: float = 0.995
    lam: float = 0.95
    gamma_turn: float = 0.0  # > 0: discount per game turn (each player's turn counts) instead of per decision; replaces gamma
    lam_turn: float = 0.0  # > 0: GAE lambda per game turn; replaces lam
    shaping: float = 0.2  # life-difference potential shaping, linearly annealed to 0
    shaping_anneal_iters: int = 100
    shaping_offset_iters: int = 0  # continuation position in the parent's shaping schedule
    lr_anneal_games: int = 0  # > 0: anneal --ppo-lr to --ppo-lr-final over this many training games (from the run's start or the first resume with annealing on)
    lr_anneal_offset_games: int = 0  # continuation position with a fresh optimizer/game counter
    lr_schedule: str = "linear"  # "linear" or "cosine"
    self_play_frac: float = 0.5
    bot_frac: float = 0.0  # share of games against the scripted bots (taken out of the pool share)
    bot_seat: str = "both"  # bot games: "both" seats at random, or "jund": the learner always Jund (seat 0) vs the blue bot, the benchmark matchup
    postboard_frac: float = 0.5  # share of training games played with sideboarded decks (match games 2/3); isolated-game mode only
    match_rollouts: int = 0  # 1: training games are whole best-of-three matches (games 1 -> 2 -> 3, knowledge carried; about games_per_iter / GAMES_PER_MATCH matches per iteration); 0: isolated games (legacy)
    variants: str = ""  # registered 75s of training matches: "" = the stock lists, else an `engine.variants` split ("train") sampled per match and position
    belief: int = 0  # 1: feature set 8 with the belief head (docs/belief-head.md); needs --match-rollouts 1 and --variants
    eval_bo3_matches: int = 100  # best-of-three matches vs the bots at each evaluation (0 = off)
    snapshot_every: int = 10
    pool_recent_frac: float = 0.5  # share of pool games against the newest snapshot
    pool_sampling: str = "uniform"  # the other pool games: "uniform" over the pool, or "pfsp": weighted (1 - learner's win rate vs it) ** pfsp_power
    opponent_pool: str = ""  # immutable external *.pt opponents, separate from this run's snapshot namespace
    pfsp_power: float = 2.0
    pfsp_ema: float = 0.05  # per-game step of the running win rate vs each pool opponent (starts at 0.5)
    init: str = ""  # a new run starts from these weights (a policy file or checkpoint; fresh optimizer): its architecture, with --value-bound, --entity-attn and --features on top (new attention layers start as the identity)
    exploit: str = ""  # exploiter mode: every training game is the learner on --exploit-deck vs this frozen policy file
    exploit_deck: str = "jund"  # the learner's deck: "jund", "blue", "red", "affinity", "elves" or "tron", one of --matchup's (evaluate.DECK_KEYS)
    eval_every: int = 10  # evaluate every this many iterations (0 = only by eval_every_games)
    eval_final: int = 0  # evaluate the final policy even when the stop is between regular evaluation points
    eval_every_games: int = 0  # also evaluate whenever the training games cross a multiple of this (e.g. 250000)
    eval_extra_matchups: int = 1  # 0: only primary matchup in routine evaluation; matrix evaluation runs separately
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
    matchup: str = "jund_blue"  # match.MATCHUPS: the deck in each seat; a mix "a:w,b:w,..." draws each game's matchup by weight (default 1), the first is the primary (full evaluation; the others: benchmark of both seats)
    auto_mana: int = 0  # 1: pay non-strategic mana costs automatically (colour-preserving payer; docs/action-decomposition.md)
    auto_pass: int = 0  # 1: auto-pass priority when the only other options are side-effect-free sacrifice-for-mana abilities
    seed: int = 0
    device: str = "cpu"
    learner_devices: str = ""  # comma-separated devices; first matches --device, others are persistent replicas
    learner_cpus: str = ""  # semicolon-separated CPU lists, one per learner rank
    engine: str = ""  # rules engine for rollouts: python (reference) or native (Rust, mtg_ml_native); "" = $MTG_ENGINE, else python (the saved config records the resolved name)
    inference: str = "local"  # policy inference: local (CPU torch in each worker) or server (one GPU process, rl/inference.py)
    server_device: str = ""  # device of the inference server ("" = --device, or cuda when --device is cpu and a GPU exists)
    server_devices: str = ""  # comma-separated inference devices; learner uses --device
    server_cpus: str = ""  # semicolon-separated CPU lists, one per inference device
    server_resident_limit: int = 0  # bounded residency waves; start new large-model runs with 64
    server_stacked_attention: int = 1  # 0: eager per-policy attention baseline, 1: stacked graphs
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
    descriptor: dict | None = None  # exact immutable rollout inputs, for lag-2 resume

    def wait(self) -> _Rollout:
        if self.data is None:
            t = time.monotonic()
            self.data = self.pending.wait()
            self.waited = time.monotonic() - t
        return self

    @property
    def rollout_s(self) -> float:
        return self.pending.t_ready - (self.pending.t_start or self.pending.t_submit)


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
        cfg.engine = engine_name(cfg.engine)  # "" -> $MTG_ENGINE or python; the saved config records the concrete engine
        os.environ[ENV_VAR] = cfg.engine  # spawned processes inherit it
        for d in ("pool", "policy"):
            os.makedirs(os.path.join(cfg.run, d), exist_ok=True)
        self.latest = os.path.join(cfg.run, "latest.pt")
        self.metrics = os.path.join(cfg.run, "metrics.jsonl")
        _check(cfg)
        self.blocks = tuple(b for b in cfg.eval_blocks.split(",") if b)
        if set(self.blocks) - set(EVAL_BLOCKS):
            raise ValueError(f"eval blocks must be among {EVAL_BLOCKS}, not {cfg.eval_blocks!r}")
        self.ladder = tuple(p for p in cfg.ladder.split(",") if p)
        for path in self.ladder + ((cfg.ladder_ratings,) if cfg.ladder_ratings else ()):
            if not os.path.exists(path):
                raise FileNotFoundError(f"ladder file {path} does not exist")
        self.ladder_ratings = cfg.ladder_ratings or os.path.join(cfg.run, "ladder.json")
        self.evaluating = bool(cfg.eval_every or cfg.eval_every_games or cfg.eval_final)
        self.matchups = parse_matchups(cfg.matchup)
        self.exploit_seat = exploit_seat(cfg)  # also validates --matchup
        self.spec_matchup: dict[int, str] = {}  # seed -> matchup of the training games in flight (a mix only)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        ck = torch.load(self.latest, map_location=cfg.device, weights_only=False) if os.path.exists(self.latest) else None
        init = cfg.init or cfg.exploit  # an exploiter starts from the policy it exploits unless --init says otherwise
        if ck is not None:  # the architecture is the checkpoint's, whatever the flags say
            if cfg.entity_attn and cfg.entity_attn != ck["config"].get("entity_attn", 0):  # the Adam state would not fit
                raise ValueError(f"{self.latest} holds a {ck['config']} network, not entity_attn={cfg.entity_attn}: resume without --entity-attn, or start a new --run with --init {self.latest}")
            self.net = PolicyNet(**{**ck["config"], **({"features": cfg.features} if cfg.features else {})})
        elif init:  # the source's architecture, with --value-bound and --entity-attn on top
            config = {k: v for k, v in torch.load(init, map_location="cpu", weights_only=False)["config"].items() if k != "value_bound"}
            config["entity_attn"] = cfg.entity_attn or config.get("entity_attn", 0)
            if cfg.features:  # no weights depend on it: a fine-tune may move to a newer feature set
                config["features"] = cfg.features
            if cfg.belief and "belief" not in config:  # explicit upgrade: new belief head, zero projection (`_init_from`)
                config.update(features=BELIEF_FEATURES, belief=BeliefSpec.default().to_config())
            self.net = PolicyNet(**config, value_bound=cfg.value_bound)
        else:
            belief = BeliefSpec.default() if cfg.belief else None
            self.net = PolicyNet(hidden=cfg.hidden, memory=cfg.memory, trunk=cfg.trunk, value_net=cfg.value_net, value_hidden=cfg.value_hidden, value_bound=cfg.value_bound, entity_attn=cfg.entity_attn,
                                 features=BELIEF_FEATURES if cfg.belief else cfg.features or FEATURES, belief=belief)
        if self.net.belief is not None and not (cfg.match_rollouts and cfg.variants and not cfg.exploit):
            raise ValueError("training a belief head needs varied-list match rollouts: --match-rollouts 1 --variants train (and no --exploit)")
        if self.net.belief is not None and len([d for d in cfg.learner_devices.split(",") if d]) > 1:
            raise ValueError("the multi-device update (rl/distributed.py) does not carry belief targets yet: train a belief head on one learner device")
        if cfg.variants:
            from ..engine.variants import sampling_prior

            self.variant_prior = sampling_prior(cfg.variants)
            for m, _ in self.matchups:
                missing = [d for d in matchup_decks(m) if d not in self.variant_prior]
                if missing:
                    raise ValueError(f"--variants {cfg.variants}: no {missing} variants in that split")
        start_iteration = ck["iteration"] if ck is not None else 0
        active_shaping = cfg.shaping * max(0.0, 1 - start_iteration / max(cfg.shaping_anneal_iters, 1))
        if active_shaping and self.net.value_bound == "tanh":
            raise ValueError("active shaping requires an unbounded value head")
        self.net.to(cfg.device)
        # a tensor lr when annealing: the update's CUDA graphs read it, so a new value each iteration costs no recapture
        self.opt = make_optimizer(self.net.parameters(), cfg.ppo.lr, cfg.device, tensor_lr=cfg.lr_anneal_games > 0)
        self.iteration = 0
        self.games_total = 0
        self.lr_origin = None  # games_total where the lr anneal starts
        self.pfsp: dict[str, float] = {}  # pool file name -> running win rate of the learner against it
        self.checkpointed = -1  # iteration of the newest latest.pt
        self.saver = ThreadPoolExecutor(1, thread_name_prefix="checkpoint")
        self.saving = None  # the latest.pt write in flight
        self.pinned: dict[str, int] = {}  # policy files an evaluation still needs
        self.checkpoint_policies = []
        self.retiring_checkpoint_policies = []
        self.resume_rollouts = ck.get("pending_rollouts", []) if ck is not None else []
        self.elapsed_total = ck.get("elapsed_total", 0.0) if ck is not None else 0.0
        self.started_at = None
        if self.resume_rollouts and cfg.pipeline != 2:
            raise ValueError("checkpoint has queued lag-2 rollouts; resume with --pipeline 2")
        for desc in self.resume_rollouts:
            path = desc["job"].learner_path
            self._pin(path)
            self.checkpoint_policies.append(path)
        if ck is not None:
            self.net.load_state_dict(ck["model"])
            load_optimizer_state(self.opt, ck["optim"])
            self.iteration = self.checkpointed = ck["iteration"]
            self.games_total = ck.get("games_total", self.iteration * cfg.games_per_iter)
            self.rng.setstate(ck["rng"])
            restore_torch_rng(ck)
            self.lr_origin = ck.get("lr_anneal_origin")
            self.pfsp = dict(ck.get("pfsp") or {})
            self._drop_lost_iterations()
            print(f"resumed {self.latest} at iteration {self.iteration}")
        elif init:
            self._init_from(init)
        if cfg.lr_anneal_games and self.lr_origin is None:
            self.lr_origin = self.games_total - cfg.lr_anneal_offset_games
        set_lr(self.opt, self._lr())
        self._remove_partial_writes()
        pool_dir = os.path.join(cfg.run, "pool")
        saved_pool = ck.get("imported_pool", []) if ck is not None else []
        imported = sorted(os.path.abspath(os.path.join(cfg.opponent_pool, f)) for f in os.listdir(cfg.opponent_pool) if f.endswith(".pt")) if cfg.opponent_pool else saved_pool
        if saved_pool and imported != saved_pool:
            raise ValueError("resumed run must retain its original immutable opponent pool")
        if any(not os.path.isfile(p) for p in imported):
            raise FileNotFoundError("an imported opponent checkpoint is missing")
        self.imported_pool = imported
        self.pool: list[str] = imported + sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.startswith("iter_") and f.endswith(".pt"))
        # a resumed run's snapshots that record no version are the run's own from before --features stamped it (docs/features.md): its version
        self.pool_features = {p: self.net.features for p in self.pool if "features" not in checkpoint_config(p)} if ck is not None and self.net.features != 1 else {}
        weights = self._weights()
        self._publish(weights)
        if not self.pool:
            self._snapshot(weights)
        if ck is None:
            self._checkpoint(weights, self.rng.getstate())
        self.layout, self.learner_cpus, self.server_cpus = training_cpu_layout(cfg, self.evaluating)
        print(self.layout.describe(), flush=True)
        self.server = None
        self.collector = self._collector()
        self.evaluator = None
        if self.evaluating and cfg.eval_process:
            ev = self.layout.evaluator
            self.eval_workers = cfg.eval_workers or (4 if self.layout.shared_eval else max(1, min(8, len(ev))))
            self.evaluator = PoolProcess(self.eval_workers, "local", worker_cpus=ev or None, cpus=ev or None, nice=10, name="evaluator")
        self.evals: list[_Eval] = []  # running first, then at most one waiting
        self.learner = None
        if cfg.learner_devices:
            from .distributed import DistributedLearner

            devices = tuple(cfg.learner_devices.split(","))
            if len(devices) > 1:
                try:
                    self.learner = DistributedLearner(self.net, self.opt, cfg.ppo, devices, self.learner_cpus)
                except BaseException:
                    self.collector.terminate()
                    if self.evaluator is not None:
                        self.evaluator.terminate()
                    if self.server is not None:
                        self.server.close()
                    self._join_checkpoint()
                    self.saver.shutdown()
                    raise

    def _init_from(self, path: str) -> None:
        """Start from the weights of checkpoint `path` (a policy file or
        latest.pt; not its optimizer or counters). Its network must have this
        config except for layers that start as the identity (`entity_attn`),
        which keep their initialisation: the first rollouts play like the
        checkpoint."""
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        with torch.device("meta"):  # its config with today's defaults filled in
            config = PolicyNet(**ck["config"]).config
        skip = ("entity_attn", "value_bound", "features", "belief", "belief_coef")  # no weights, or new layers (belief head, zero projection) `load_partial` initialises
        own = {k: v for k, v in self.net.config.items() if k not in skip}
        theirs = {k: v for k, v in config.items() if k not in skip}
        if own != theirs or config.get("entity_attn", 0) > self.net.config.get("entity_attn", 0):
            raise ValueError(f"--init {path}: its network {ck['config']} does not fit {self.net.config}")
        new = load_partial(self.net, ck["model"])
        print(f"initialised from {path}" + (f"; new layers at their identity init: {', '.join(new)}" if new else ""), flush=True)

    def _collector(self):
        c, lay = self.cfg, self.layout
        server_cfg = None
        if c.inference == "server":
            from .inference import ServerConfig, default_device

            dev = c.server_device or (c.device if c.device != "cpu" else default_device())
            devices = tuple(c.server_devices.split(",")) if c.server_devices else (dev,)
            per_cpu = self.server_cpus
            server_cfg = ServerConfig(device=dev, devices=devices, server_cpus=per_cpu, resident_limit=c.server_resident_limit, stacked_attention=bool(c.server_stacked_attention),
                                      max_rows=c.server_max_rows, request_ints=c.server_request_ints or request_ints_for(c.games_per_iter, c.workers),
                                      policy_slots=c.server_policy_slots, cpus=lay.server or None)
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

    def _checkpoint(self, weights: dict, rng: tuple, pending_rollouts=()) -> None:
        """Write the resume checkpoint `latest.pt` in the background: with the
        Adam state it is ~3x the weights, and the next update need not wait."""
        self._join_checkpoint()
        new_policies = [desc["job"].learner_path for desc in pending_rollouts]
        for path in new_policies:
            self._pin(path)
        self.retiring_checkpoint_policies = self.checkpoint_policies
        self.checkpoint_policies = new_policies
        ck = {
            "config": self.net.config,
            "model": weights,
            "optim": _cpu(self.opt.state_dict()),
            "iteration": self.iteration,
            "games_total": self.games_total,
            "rng": rng,
            "torch_rng": torch.get_rng_state(),  # the update's minibatch shuffles
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "lr_anneal_origin": self.lr_origin,
            "pfsp": dict(self.pfsp),
            "train_config": asdict(self.cfg),
            "pending_rollouts": list(pending_rollouts),
            "elapsed_total": self.elapsed_total + (time.monotonic() - self.started_at if self.started_at is not None else 0.0),
            "imported_pool": self.imported_pool,
        }
        self.saving = self.saver.submit(_save, ck, self.latest)
        self.checkpointed = self.iteration

    def _join_checkpoint(self) -> None:
        if self.saving is not None:
            self.saving.result()  # re-raises a failed write
            self.saving = None
            for path in self.retiring_checkpoint_policies:
                self._unpin(path)
            self.retiring_checkpoint_policies = []

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
        shaping = c.shaping * max(0.0, 1 - (it + c.shaping_offset_iters) / max(c.shaping_anneal_iters, 1))
        job = Job([], self.policy, self.iteration + 1, record=True, gamma=c.gamma, lam=c.lam, shaping=shaping, max_turns=c.max_turns, inference=c.inference,
                  value_clamp=c.value_clamp, gamma_turn=c.gamma_turn, lam_turn=c.lam_turn, auto_mana=bool(c.auto_mana), auto_pass=bool(c.auto_pass),
                  features=self.pool_features)
        specs = self._train_specs(it)
        desc = {"it": it, "job": job, "specs": specs, "rng": rng}
        if c.pipeline == 2:
            self._pin(job.learner_path)
        return _Rollout(self.collector.call(play, specs, job, c.workers), shaping, it - self.iteration, rng, descriptor=desc)

    def _resubmit(self, desc):
        job, specs = desc["job"], desc["specs"]
        self._pin(job.learner_path)
        if len(self.matchups) > 1:
            self.spec_matchup.update({seed: s.matchup for s in specs for seed in ([game_seed(s.seed, n) for n in (1, 2, 3)] if s.bo3 else [s.seed])})
        pending = self.collector.call(play, specs, job, self.cfg.workers)
        return _Rollout(pending, job.shaping, desc["it"] - job.learner_version + 1, desc["rng"], descriptor=desc)

    def _train_specs(self, it: int) -> list[GameSpec]:
        """The games of a training rollout. Exploiter mode: the learner on
        --exploit-deck against the frozen --exploit policy, every game.
        Otherwise self-play, bot games (--bot-seat) and pool games: the
        newest snapshot with probability pool_recent_frac, else one drawn
        uniformly or by PFSP weight (`_pfsp_weights`)."""
        c, specs = self.cfg, []
        base = TRAIN_SEED_BASE + (it + 1) * 1_000_003 + c.seed * 7919
        weights = self._pfsp_weights() if c.pool_sampling == "pfsp" else None
        for k in range(math.ceil(c.games_per_iter / GAMES_PER_MATCH) if c.match_rollouts else c.games_per_iter):
            if c.exploit:
                main = os.path.abspath(c.exploit)
                seats = (LEARNER, main) if self.exploit_seat == 0 else (main, LEARNER)
            else:
                r = self.rng.random()
                if r < c.self_play_frac:
                    seats = (LEARNER, LEARNER)
                elif r < c.self_play_frac + c.bot_frac:
                    if c.bot_seat == "jund":
                        seats = (LEARNER, BOT)
                    else:
                        seats = (LEARNER, BOT) if self.rng.random() < 0.5 else (BOT, LEARNER)
                else:
                    if self.rng.random() < c.pool_recent_frac:
                        opp = self.pool[-1]
                    else:
                        opp = self.rng.choices(self.pool, weights)[0] if weights else self.rng.choice(self.pool)
                    seats = (LEARNER, opp) if self.rng.random() < 0.5 else (opp, LEARNER)
            game_no = 1 if c.match_rollouts else 2 if self.rng.random() < c.postboard_frac else 1
            matchup = self.matchups[0][0]
            if len(self.matchups) > 1:
                matchup = self.rng.choices([m for m, _ in self.matchups], [w for _, w in self.matchups])[0]
                for seed in ([game_seed(base + k, n) for n in (1, 2, 3)] if c.match_rollouts else [base + k]):
                    self.spec_matchup[seed] = matchup
            swap = self.rng.random() < 0.5 if self.net.features >= 7 else False
            variants = None
            if c.variants:
                from ..engine.variants import sample_variant

                variants = tuple(sample_variant(d, self.rng, c.variants).id for d in matchup_decks(matchup))
            specs.append(GameSpec(seed=base + k, seats=seats, match_game=game_no, matchup=matchup, swap_seats=swap,
                                  bo3=bool(c.match_rollouts), variants=variants))
        return specs

    def _matchup_stats(self, games: list) -> dict:
        """A mix: each matchup's share of the games and seat 0's self-play win rate on it."""
        by = {m: [] for m, _ in self.matchups}
        for g in games:
            by[self.spec_matchup.pop(g[-1])].append(g)
        if self.cfg.match_rollouts:  # games a match did not need (a 2-0)
            for seed in {(g[-1] - 1) // 4 for g in games}:
                for n in (1, 2, 3):
                    self.spec_matchup.pop(game_seed(seed, n), None)
        out = {}
        for m, gs in by.items():
            out[f"matchup_share/{m}"] = len(gs) / max(len(games), 1)
            out[f"seat0_wins_selfplay/{m}"] = _rate([g for g in gs if g[0] == (LEARNER, LEARNER)], 0)
        return out

    def _pfsp_weights(self) -> list[float] | None:
        """Prioritised fictitious self-play: pool snapshot i weighted (1 -
        p_i) ** pfsp_power, p_i the learner's running win rate against it
        (draws half; 0.5 until it has played it), so opponents that still beat
        the learner come up most. None (uniform) if every weight is 0."""
        w = [(1.0 - self.pfsp.get(self._pool_name(p), 0.5)) ** self.cfg.pfsp_power for p in self.pool]
        return w if sum(w) > 0 else None

    def _update_pfsp(self, games: list) -> None:
        """Move the running win rates toward the results of the learner's games
        against pool snapshots (step pfsp_ema per game)."""
        pool = {p: self._pool_name(p) for p in self.pool}
        for seats, winner, *_ in games:
            for s in (0, 1):
                if seats[s] == LEARNER and seats[1 - s] in pool:
                    name = pool[seats[1 - s]]
                    result = 0.5 if winner is None else float(winner == s)
                    p = self.pfsp.get(name, 0.5)
                    self.pfsp[name] = p + self.cfg.pfsp_ema * (result - p)

    def _pool_name(self, path):
        return ("imported/" if path in getattr(self, "imported_pool", ()) else "") + os.path.basename(path)

    def _lr(self) -> float:
        """The learning rate of the next update: --ppo-lr, or annealed to
        --ppo-lr-final over --lr-anneal-games training games counted from
        `lr_origin` (linear or cosine), then held."""
        c = self.cfg
        if not c.lr_anneal_games:
            return c.ppo.lr
        f = min(1.0, max(0.0, (self.games_total - self.lr_origin) / c.lr_anneal_games))
        if c.lr_schedule == "cosine":
            f = 0.5 * (1 - math.cos(math.pi * f))
        return c.ppo.lr + (c.ppo.lr_final - c.ppo.lr) * f

    # -- evaluation ----------------------------------------------------------

    def _eval_args(self, e: _Eval, n_jobs: int, inference: str) -> tuple:
        c = self.cfg
        return (e.policy, self.pool[0], e.version, n_jobs, c.eval_games, c.eval_bo3_matches, c.bench_games, c.bench_bo3_matches, c.max_turns, inference,
                self.blocks, c.bench_greedy_games, self.ladder, c.ladder_games, self.ladder_ratings, bool(c.ladder_greedy), bool(c.auto_mana), bool(c.auto_pass),
                self.matchups[0][0], self.exploit_seat, os.path.abspath(c.exploit) if c.exploit else "", self.pool_features, tuple(m for m, _ in self.matchups[1:]) if c.eval_extra_matchups else ())

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

    def _pin(self, path: str) -> None:
        self.pinned[path] = self.pinned.get(path, 0) + 1

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
        return (iteration < c.iterations and not (c.total_games and games_total >= c.total_games)
                and not (c.duration_seconds and self.started_at is not None and time.monotonic() - self.started_at >= c.duration_seconds))

    def train(self) -> None:
        c = self.cfg
        self.started_at = time.monotonic()
        threads = torch.get_num_threads()
        affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        if c.pipeline or c.trainer_cpus:  # the update shares the CPUs with the workers: spinning OMP threads on busy cores slowed it by 60%
            torch.set_num_threads(max(1, min(threads, len(self.layout.trainer))))
            if affinity is not None and self.layout.trainer:
                os.sched_setaffinity(0, set(self.layout.trainer))
        roll = None  # the collected rollout for the coming update, if it was played during the last one
        queue = [self._resubmit(desc) for desc in self.resume_rollouts]
        self.resume_rollouts = []
        ok = False
        try:
            while self._more(self.iteration, self.games_total):
                t0 = time.monotonic()
                if c.pipeline == 2 and queue:
                    roll = queue.pop(0).wait()
                elif roll is None:
                    roll = self._submit(self.iteration).wait()
                data = roll.data
                if c.pool_sampling == "pfsp":  # before the next games are drawn, so a resume draws the same ones
                    self._update_pfsp(data.games)
                nxt = None
                if c.pipeline == 2:
                    while len(queue) < 2 and self._more(self.iteration + 1 + len(queue), self.games_total + len(data.games) + len(queue) * c.games_per_iter):
                        queue.append(self._submit(self.iteration + 1 + len(queue)))
                    nxt = queue[0] if queue else None
                elif c.pipeline and self._more(self.iteration + 1, self.games_total + len(data.games)):
                    nxt = self._submit(self.iteration + 1)  # plays with the current weights during the update
                t1 = time.monotonic()
                lr = self._lr()
                set_lr(self.opt, lr)
                stats = self.learner.update(self.net, self.opt, data, c.ppo, lr) if self.learner else ppo_update(self.net, self.opt, data, c.ppo, device=c.device)
                t_learned = time.monotonic()
                release(data)  # unmaps the shared block of the samples
                self.iteration += 1
                self.games_total += len(data.games)
                weights = self._weights()
                self._publish(weights)
                if self.iteration % c.snapshot_every == 0:
                    self._snapshot(weights)
                if c.checkpoint_every and self.iteration % c.checkpoint_every == 0:
                    if c.pipeline == 2:
                        self._checkpoint(weights, self.rng.getstate(), [r.descriptor for r in queue])
                    else:
                        self._checkpoint(weights, nxt.rng if nxt else self.rng.getstate())
                t2 = time.monotonic()

                row = {
                    "features": self.net.features,
                    "information_contract": information_contract(self.net.features),
                    "iteration": self.iteration,
                    "decisions": len(data.actions),
                    "games": len(data.games),
                    "games_total": self.games_total,
                    "matches": len(getattr(data, "matches", [])),
                    "game_turns": sum(g[3] for g in data.games) / max(len(data.games), 1),
                    "draws": sum(g[1] is None for g in data.games) / max(len(data.games), 1),
                    "jund_wins_selfplay": _rate([g for g in data.games if g[0] == (LEARNER, LEARNER)], 0),
                    **rollout_stats(data.games, data.kinds),
                    "shaping": roll.shaping,
                    "lr": lr,
                    "pool_size": len(self.pool),
                    "rollout_s": round(roll.rollout_s, 2),
                    "update_s": round(t2 - t1, 2),
                    "ppo_s": round(t_learned - t1, 3),
                    "publish_s": round(t2 - t_learned, 3),
                    "learner_precision": c.ppo.precision,
                    "learner_peak_bytes": torch.cuda.max_memory_allocated(next(self.net.parameters()).device) if next(self.net.parameters()).is_cuda else 0,
                    "inference_stats": getattr(data, "inference_stats", {}),
                    "residency": getattr(data, "residency", {}),
                    "policy_lag": roll.lag,
                    "decisions_per_s": round(len(data.actions) / max(roll.rollout_s, 1e-9)),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                }
                if len(self.matchups) > 1:
                    row.update(self._matchup_stats(data.games))
                if c.exploit:  # every training game is against the frozen main policy
                    row["win_vs_main"] = row["win_vs_pool"]
                if self._eval_due(len(data.games)):
                    row.update(self._request_eval())
                if nxt and c.pipeline != 2:
                    nxt.wait()
                row["wait_s"] = round(roll.waited if c.pipeline == 2 else nxt.waited if nxt else 0.0, 2)
                row["queue_s"] = round(max(0, roll.pending.t_start - roll.pending.t_submit), 3)
                row["wall_s"] = round(time.monotonic() - t0, 2)
                row["elapsed_s"] = round(self.elapsed_total + time.monotonic() - self.started_at, 3)
                _append(self.metrics, row)
                print(_fmt(row), flush=True)
                if self.evaluator is not None:
                    self._collect_evals()
                if c.pipeline == 2:
                    self._unpin(roll.descriptor["job"].learner_path)
                roll = nxt
            if self.checkpointed != self.iteration or queue:
                rng = roll.rng if c.pipeline == 1 and roll is not None else self.rng.getstate()
                self._checkpoint(self._weights(), rng, [r.descriptor for r in queue])
            if c.eval_final and self.iteration and not self._eval_due(c.games_per_iter):
                final_eval = self._request_eval()
                if final_eval:
                    _amend(self.metrics, self.iteration, final_eval)
            if self.evaluator is not None:
                self._collect_evals(block=True)
            ok = True
        finally:
            if self.learner is not None:
                self.learner.close()
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


def restore_torch_rng(ck: dict) -> None:
    """Restore the torch (and CUDA) generator states saved in a checkpoint.
    `torch.load(map_location=cuda)` moves the saved ByteTensors to the GPU,
    which set_rng_state rejects, so they go back to the CPU first. Older
    checkpoints lack them: the update's minibatch shuffles then restart from
    cfg.seed."""
    if "torch_rng" not in ck:
        return
    torch.set_rng_state(ck["torch_rng"].cpu())
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])


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


def training_cpu_layout(cfg, evaluating):
    """Reserve every GPU driver process before assigning rollout workers."""
    avail = available_cpus()
    learners = tuple(cfg.learner_devices.split(",")) if cfg.learner_devices else (cfg.device,)
    servers = tuple(cfg.server_devices.split(",")) if cfg.server_devices else (cfg.server_device or cfg.device,)
    lc = tuple(parse_cpus(c) for c in cfg.learner_cpus.split(";")) if cfg.learner_cpus else ()
    sc = tuple(parse_cpus(c) for c in cfg.server_cpus.split(";")) if cfg.server_cpus else ()
    given = [parse_cpus(c) for c in (cfg.trainer_cpus, cfg.worker_cpus, cfg.eval_cpus) if c]
    if lc and cfg.trainer_cpus and lc[0] != parse_cpus(cfg.trainer_cpus):
        raise ValueError("trainer_cpus must match the first learner_cpus list")
    reserved = {c for group in (*lc[1:], *sc) for c in group}
    primary = lc[0] if lc else parse_cpus(cfg.trainer_cpus)
    if reserved & {c for group in (*given, primary) for c in group}:
        raise ValueError("GPU process CPU lists overlap the trainer, workers or evaluation")
    groups = (*lc, *sc)
    if groups and (any(not g for g in groups) or any(c not in avail for g in groups for c in g)):
        raise ValueError("GPU CPU lists must be nonempty and inside the process affinity")
    flat = [c for g in (*lc[1:], *sc) for c in g]
    if len(set(flat)) != len(flat):
        raise ValueError("GPU process CPU lists must not overlap")
    free = [c for c in avail if c not in reserved and not any(c in g for g in given) and c not in primary]
    if len(learners) > 1 and not lc:
        if len(free) < len(learners):
            raise ValueError("not enough CPUs for learner ranks; provide learner_cpus")
        primary = parse_cpus(cfg.trainer_cpus) or (free.pop(0),)
        lc = (primary, *((free.pop(0),) for _ in learners[1:]))
        reserved.update(c for g in lc[1:] for c in g)
    if cfg.inference == "server" and len(servers) > 1 and not sc:
        if len(free) < len(servers):
            raise ValueError("not enough CPUs for inference servers; provide server_cpus")
        sc = tuple((free.pop(0),) for _ in servers)
        reserved.update(c for g in sc for c in g)
    lay = cpu_layout(cfg.workers, cfg.device, cfg.inference == "server" and not sc, bool(evaluating and cfg.eval_process),
                     fmt_cpus(primary) if primary else cfg.trainer_cpus, cfg.worker_cpus, cfg.eval_cpus, tuple(c for c in avail if c not in reserved))
    if sc:
        lay.server = tuple(c for g in sc for c in g)
    return lay, lc, sc


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


CHOICES = {"bot_seat": ("both", "jund"), "pool_sampling": ("uniform", "pfsp"), "exploit_deck": tuple(DECK_KEYS.values()), "lr_schedule": ("linear", "cosine"), "value_bound": ("none", "tanh"), "features": (0, *FEATURE_VERSIONS), "engine": ("", *ENGINES)}


def exploit_seat(cfg: TrainConfig) -> int | None:
    """The learner's seat in exploiter mode (where --exploit-deck sits in --matchup), else None."""
    matchups = parse_matchups(cfg.matchup)
    if not cfg.exploit:
        return None
    if len(matchups) > 1:
        raise ValueError(f"exploiter mode plays one matchup, not the mix {cfg.matchup!r}")
    decks = [DECK_KEYS[d] for d in matchup_decks(matchups[0][0])]
    if cfg.exploit_deck not in decks:
        raise ValueError(f"--exploit-deck {cfg.exploit_deck} is not in --matchup {cfg.matchup} ({', '.join(decks)})")
    return decks.index(cfg.exploit_deck)


def _check(cfg: TrainConfig) -> None:
    for name, allowed in CHOICES.items():
        if getattr(cfg, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {getattr(cfg, name)!r}")
    if cfg.exploit and not os.path.exists(cfg.exploit):
        raise FileNotFoundError(f"--exploit {cfg.exploit}: no such policy file")
    if cfg.duration_seconds < 0:
        raise ValueError("duration_seconds must be nonnegative")
    if cfg.lr_anneal_offset_games < 0 or cfg.shaping_offset_iters < 0:
        raise ValueError("continuation schedule offsets must be nonnegative")
    if cfg.lr_anneal_games < 0 or cfg.gamma_turn < 0 or cfg.lam_turn < 0 or cfg.value_clamp < 0:
        raise ValueError("lr_anneal_games, gamma_turn, lam_turn and value_clamp must be >= 0")
    if cfg.server_resident_limit < 0:
        raise ValueError("server_resident_limit must be nonnegative")
    if cfg.pipeline not in (0, 1, 2):
        raise ValueError("pipeline must be 0, 1 or 2")
    if cfg.server_devices and cfg.inference != "server":
        raise ValueError("server_devices requires inference=server")
    if cfg.server_cpus:
        devices = cfg.server_devices.split(",") if cfg.server_devices else [cfg.server_device or cfg.device]
        if len(cfg.server_cpus.split(";")) != len(devices):
            raise ValueError("server_cpus needs one CPU list per inference device")
    if cfg.ppo.precision not in ("fp32", "bf16"):
        raise ValueError("ppo precision must be fp32 or bf16")
    if cfg.learner_devices:
        devices = cfg.learner_devices.split(",")
        if torch.device(devices[0]) != torch.device(cfg.device):
            raise ValueError("first learner device must match --device")
        if cfg.learner_cpus and len(cfg.learner_cpus.split(";")) != len(devices):
            raise ValueError("learner_cpus needs one CPU list per learner device")


def _fmt(row: dict) -> str:
    keys = ["iteration", "decisions", "game_turns", "draws", "jund_wins_selfplay", "win_vs_pool", "win_vs_main", "entropy", "approx_kl", "explained_var", "decisions_per_s", "wall_s"]
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
            typ = (lambda x: None if x.lower() == "none" else float(x)) if f.name == "target_kl" else (
                lambda x: None if x.lower() == "none" else tuple(float(y) for y in x.split(","))) if f.name == "belief_coef" else type(v)
            ap.add_argument(f"--{prefix}{f.name.replace('_', '-')}", type=typ, default=v, choices=CHOICES.get(f.name) if obj is cfg else None)
    a = vars(ap.parse_args(argv))
    for f in fields(ppo):
        setattr(ppo, f.name, a.pop("ppo_" + f.name))
    return TrainConfig(**a, ppo=ppo)


def main(argv=None) -> None:
    import signal

    def stop(signum, _frame):
        raise SystemExit(128 + signum)  # unwind Trainer.train's owned processes/checkpoint writer

    signal.signal(signal.SIGTERM, stop)
    Trainer(parse_args(argv)).train()


if __name__ == "__main__":
    main()
