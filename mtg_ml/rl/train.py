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
  3. save `latest.pt`; every `snapshot_every` iterations freeze a copy into
     `pool/`; every `eval_every` iterations play against the random agent,
     the scripted bots and the oldest pool checkpoint on fixed paired seeds (both seats, both
     starting players), and run the benchmark (`evaluate.benchmark`: learner
     as Jund Wildfire vs the blue Delver bot), logged as `bench/*`.
`--total-games` stops after that many training games (counted across resumes).
Metrics go to `<run>/metrics.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields

import torch

from ..backend import ENV_VAR, engine_name
from .evaluate import benchmark, head_to_head, head_to_head_bo3
from .model import PolicyNet
from .ppo import PPOConfig, ppo_update
from .rollout import BOT, LEARNER, RANDOM, GameSpec, Job, Result, run_job, split_games, worker_init


@dataclass
class TrainConfig:
    run: str = "runs/ppo"
    iterations: int = 100
    total_games: int = 0  # stop once this many training games are played (0 = only --iterations)
    games_per_iter: int = 256  # >= ~32 per worker keeps batched inference cheap next to the engine
    workers: int = max(1, (os.cpu_count() or 2) - 1)
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
    eval_every: int = 10
    eval_games: int = 100  # per opponent, split over both seats
    bench_games: int = 1000  # benchmark games at each evaluation: learner Jund vs blue bot (0 = off)
    bench_bo3_matches: int = 200  # benchmark best-of-three matches at each evaluation (0 = off)
    max_turns: int = 100
    seed: int = 0
    device: str = "cpu"
    engine: str = "python"  # rules engine for rollouts: python (reference) or native (Rust, mtg_ml_native)
    inference: str = "local"  # policy inference: local (CPU torch in each worker) or server (one GPU process, rl/inference.py)
    server_device: str = ""  # device of the inference server ("" = --device, or cuda when --device is cpu and a GPU exists)
    server_max_rows: int = 16384  # largest batch the server builds from queued requests
    ppo: PPOConfig = field(default_factory=PPOConfig)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.environ[ENV_VAR] = engine_name(cfg.engine)  # spawned rollout workers inherit it
        os.makedirs(os.path.join(cfg.run, "pool"), exist_ok=True)
        self.latest = os.path.join(cfg.run, "latest.pt")
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.net = PolicyNet(hidden=cfg.hidden, memory=cfg.memory, trunk=cfg.trunk, value_net=cfg.value_net, value_hidden=cfg.value_hidden).to(cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self.iteration = 0
        self.games_total = 0
        self.pool: list[str] = []
        if os.path.exists(self.latest):
            ck = torch.load(self.latest, map_location=cfg.device, weights_only=False)
            self.net.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["optim"])
            self.iteration = ck["iteration"]
            self.games_total = ck.get("games_total", self.iteration * cfg.games_per_iter)
            self.rng.setstate(ck["rng"])
            print(f"resumed {self.latest} at iteration {self.iteration}")
        self.pool = sorted(os.path.join(cfg.run, "pool", f) for f in os.listdir(os.path.join(cfg.run, "pool")))
        if not self.pool:
            self.pool.append(self._save(os.path.join(cfg.run, "pool", "iter_00000.pt")))
            self._save(self.latest)
        self.server = None
        if cfg.inference == "server":
            from .inference import InferenceServer, ServerConfig, default_device

            dev = cfg.server_device or (cfg.device if cfg.device != "cpu" else default_device())
            self.server = InferenceServer(cfg.workers, ServerConfig(device=dev, max_rows=cfg.server_max_rows))
            self.procs = self.server.pool()
        elif cfg.inference == "local":
            self.procs = mp.get_context("spawn").Pool(cfg.workers, initializer=worker_init)
        else:
            raise ValueError(f"inference must be local or server, not {cfg.inference!r}")

    # -- checkpoints ---------------------------------------------------------

    def _save(self, path: str) -> str:
        tmp = path + ".tmp"
        torch.save(
            {
                "config": self.net.config,
                "model": {k: v.cpu() for k, v in self.net.state_dict().items()},
                "optim": self.opt.state_dict(),
                "iteration": self.iteration,
                "games_total": self.games_total,
                "rng": self.rng.getstate(),
                "train_config": asdict(self.cfg),
            },
            tmp,
        )
        os.replace(tmp, path)
        return path

    # -- rollouts ------------------------------------------------------------

    def _run(self, specs: list[GameSpec], record: bool, shaping: float = 0.0) -> Result:
        c = self.cfg
        jobs = [
            Job(chunk, self.latest, self.iteration + 1, record=record, gamma=c.gamma, lam=c.lam, shaping=shaping, max_turns=c.max_turns, inference=c.inference)
            for chunk in split_games(specs, c.workers)
        ]
        merged = Result()
        for r in self.procs.map(run_job, jobs):
            for f in fields(Result):
                getattr(merged, f.name).extend(getattr(r, f.name))
        return merged

    def _train_specs(self) -> list[GameSpec]:
        c, specs = self.cfg, []
        base = (self.iteration + 1) * 1_000_003 + c.seed * 7919
        for k in range(c.games_per_iter):
            r = self.rng.random()
            if r < c.self_play_frac:
                seats = (LEARNER, LEARNER)
            elif r < c.self_play_frac + c.bot_frac:
                seats = (LEARNER, BOT) if self.rng.random() < 0.5 else (BOT, LEARNER)
            else:
                opp = self.pool[-1] if self.rng.random() < c.pool_recent_frac else self.rng.choice(self.pool)
                seats = (LEARNER, opp) if self.rng.random() < 0.5 else (opp, LEARNER)
            game_no = 2 if self.rng.random() < c.postboard_frac else 1
            specs.append(GameSpec(seed=base + k, seats=seats, match_game=game_no))
        return specs

    def evaluate(self) -> dict:
        """Learner vs random agent, scripted bots and the oldest snapshot on paired
        seeds (game 1 decks), plus best-of-three matches against the bots."""
        c, out = self.cfg, {}
        for name, opp in (("random", RANDOM), ("bot", BOT), ("pool0", self.pool[0])):
            res = head_to_head(self.procs, self.latest, opp, c.eval_games, c.workers, self.iteration + 1, c.max_turns, c.inference)
            for deck in ("jund", "blue"):
                out[f"eval/{name}/{deck}"], out[f"eval/{name}/{deck}_ci"], _ = res[deck]
        if c.eval_bo3_matches:
            res = head_to_head_bo3(self.procs, self.latest, BOT, c.eval_bo3_matches, c.workers, self.iteration + 1, c.max_turns, c.inference)
            for deck in ("jund", "blue"):
                out[f"eval/bot_bo3/{deck}"], out[f"eval/bot_bo3/{deck}_ci"], _ = res[deck]
        out.update(benchmark(self.procs, self.latest, c.bench_games, c.bench_bo3_matches, c.workers, self.iteration + 1, c.max_turns, c.inference))
        return out

    # -- main loop -----------------------------------------------------------

    def train(self) -> None:
        c = self.cfg
        log = open(os.path.join(c.run, "metrics.jsonl"), "a")
        try:
            while self.iteration < c.iterations and not (c.total_games and self.games_total >= c.total_games):
                t0 = time.perf_counter()
                shaping = c.shaping * max(0.0, 1 - self.iteration / max(c.shaping_anneal_iters, 1))
                data = self._run(self._train_specs(), record=True, shaping=shaping)
                t1 = time.perf_counter()
                stats = ppo_update(self.net, self.opt, data, c.ppo, device=c.device)
                self.iteration += 1
                self.games_total += len(data.games)
                self._save(self.latest)
                if self.iteration % c.snapshot_every == 0:
                    self.pool.append(self._save(os.path.join(c.run, "pool", f"iter_{self.iteration:05d}.pt")))
                t2 = time.perf_counter()

                learner_games = [(g, s) for g in data.games for s in (0, 1) if g[0][s] == LEARNER and g[0][1 - s] != LEARNER]
                row = {
                    "iteration": self.iteration,
                    "decisions": len(data.actions),
                    "games": len(data.games),
                    "games_total": self.games_total,
                    "game_turns": sum(g[3] for g in data.games) / max(len(data.games), 1),
                    "draws": sum(g[1] is None for g in data.games) / max(len(data.games), 1),
                    "jund_wins_selfplay": _rate([g for g in data.games if g[0] == (LEARNER, LEARNER)], 0),
                    "win_vs_pool": sum(g[1] == s for g, s in learner_games) / max(len(learner_games), 1),
                    "shaping": shaping,
                    "pool_size": len(self.pool),
                    "rollout_s": round(t1 - t0, 2),
                    "update_s": round(t2 - t1, 2),
                    "decisions_per_s": round(len(data.actions) / max(t1 - t0, 1e-9)),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                }
                if c.eval_every and self.iteration % c.eval_every == 0:
                    row.update(self.evaluate())
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(_fmt(row), flush=True)
        finally:
            log.close()
            self.procs.close()
            self.procs.join()
            if self.server is not None:
                self.server.close()


def _rate(games, seat: int) -> float:
    return sum(g[1] == seat for g in games) / len(games) if games else float("nan")


def _fmt(row: dict) -> str:
    keys = ["iteration", "decisions", "game_turns", "draws", "jund_wins_selfplay", "win_vs_pool", "entropy", "approx_kl", "explained_var", "decisions_per_s"]
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
