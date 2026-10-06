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
  3. write the new weights to `policy/vNNNNN.pt` (weights only, what workers
     and the inference server load; the newest 3 are kept) and the resume
     checkpoint `latest.pt` (weights, Adam state, counters, rng) in a
     background thread; every `snapshot_every` iterations freeze the weights
     into `pool/`; every `eval_every` iterations play against the random
     agent, the scripted bots and the oldest pool checkpoint on fixed paired
     seeds (both seats, both starting players), and run the benchmark
     (`evaluate.benchmark`: learner as Jund Wildfire vs the blue Delver bot),
     logged as `bench/*`.
With `--pipeline 1` (the default) steps 1 and 2 overlap: while the update of
iteration k runs on the GPU, the workers already play the games of
iteration k+1 with the weights of iteration k, so an iteration takes
max(rollout, update) instead of their sum. That is the usual one-step policy
lag: the recorded log-probs are the behaviour policy's, so the clipped ratio
stays valid. A policy file is never rewritten, so the update cannot change
the weights under a rollout that is still loading them, and evaluation waits
for the rollout in flight. The update then shares the CPUs with the workers,
so it uses at most as many torch threads as the workers leave cores free
(at least one). `--pipeline 0` runs the steps one after the other.
`--total-games` stops after that many training games (counted across resumes).
Metrics go to `<run>/metrics.jsonl`. Timing: `rollout_s` (submit to last job
done) is for the games the row trains on, which with `--pipeline 1` were
played during the previous update; `update_s` covers the update and the
checkpoints; `wait_s` is the time the iteration blocked on the next rollout
after its update; `wall_s` is the whole iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields

import torch

from ..backend import ENV_VAR, engine_name
from .evaluate import benchmark, head_to_head, head_to_head_bo3
from .model import PolicyNet
from .ppo import PPOConfig, ppo_update
from .rollout import BOT, LEARNER, RANDOM, GameSpec, Job, Result, create_pool, play

KEEP_POLICIES = 3  # the newest, the one a lagged rollout may still be loading, one spare


@dataclass
class TrainConfig:
    run: str = "runs/ppo"
    iterations: int = 100
    total_games: int = 0  # stop once this many training games are played (0 = only --iterations)
    games_per_iter: int = 256  # >= ~32 per worker keeps batched inference cheap next to the engine
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    pipeline: int = 1  # 1: play iteration k+1 with the weights of k during the update of k (one-step policy lag); 0: one after the other
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


@dataclass
class _Rollout:
    """A training rollout playing on the worker pool in a thread of its own
    (`rollout.play` streams games and merges results as jobs end); `wait`
    collects it. A daemon thread, so a rollout abandoned after an error
    (`Pool.terminate`) cannot block the exit."""

    shaping: float
    lag: int  # updates its weights are behind the update that trains on it
    rng: tuple  # trainer rng state before its games were drawn: the checkpoint of its iteration stores it
    t_submit: float
    thread: threading.Thread | None = None
    t_ready: float = 0.0  # when the last job was merged
    data: Result | None = None
    error: BaseException | None = None

    def start(self, fn, *args) -> _Rollout:
        def run():
            try:
                self.data = fn(*args)
            except BaseException as e:  # noqa: BLE001 - re-raised by wait()
                self.error = e
            self.t_ready = time.perf_counter()

        self.thread = threading.Thread(target=run, name="rollout", daemon=True)
        self.thread.start()
        return self

    def wait(self) -> _Rollout:
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.environ[ENV_VAR] = engine_name(cfg.engine)  # spawned rollout workers inherit it
        for d in ("pool", "policy"):
            os.makedirs(os.path.join(cfg.run, d), exist_ok=True)
        self.latest = os.path.join(cfg.run, "latest.pt")
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.net = PolicyNet(hidden=cfg.hidden, memory=cfg.memory, trunk=cfg.trunk, value_net=cfg.value_net, value_hidden=cfg.value_hidden).to(cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self.iteration = 0
        self.games_total = 0
        self.saver = ThreadPoolExecutor(1, thread_name_prefix="checkpoint")
        self.saving = None  # the latest.pt write in flight
        if os.path.exists(self.latest):
            ck = torch.load(self.latest, map_location=cfg.device, weights_only=False)
            self.net.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["optim"])
            self.iteration = ck["iteration"]
            self.games_total = ck.get("games_total", self.iteration * cfg.games_per_iter)
            self.rng.setstate(ck["rng"])
            print(f"resumed {self.latest} at iteration {self.iteration}")
        self.pool: list[str] = sorted(os.path.join(cfg.run, "pool", f) for f in os.listdir(os.path.join(cfg.run, "pool")))
        weights = self._weights()
        self._publish(weights)
        if not self.pool:
            self._snapshot(weights)
            self._checkpoint(weights, self.rng.getstate())
        self.server = None
        if cfg.inference == "server":
            from .inference import InferenceServer, ServerConfig, default_device

            dev = cfg.server_device or (cfg.device if cfg.device != "cpu" else default_device())
            self.server = InferenceServer(cfg.workers, ServerConfig(device=dev, max_rows=cfg.server_max_rows))
        self.procs = create_pool(cfg.workers, cfg.inference, self.server)

    # -- checkpoints ---------------------------------------------------------

    def _weights(self) -> dict:
        """CPU copies of the weights: they are written while the next update changes the originals."""
        return _cpu(self.net.state_dict())

    def _publish(self, weights: dict) -> None:
        """Write the weights that rollouts and evaluations load from now on. A new
        file per version: a rollout still loading the previous one is unaffected."""
        d = os.path.join(self.cfg.run, "policy")
        self.policy = _save({"config": self.net.config, "model": weights}, os.path.join(d, f"v{self.iteration:05d}.pt"))
        old = sorted((f for f in os.listdir(d) if f.startswith("v") and f.endswith(".pt")), key=lambda f: int(f[1:-3]))
        for f in old[:-KEEP_POLICIES]:
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
            "train_config": asdict(self.cfg),
        }
        self.saving = self.saver.submit(_save, ck, self.latest)

    def _join_checkpoint(self) -> None:
        if self.saving is not None:
            self.saving.result()  # re-raises a failed write
            self.saving = None

    # -- rollouts ------------------------------------------------------------

    def _submit(self, it: int) -> _Rollout:
        """Start the training rollout for the update of iteration `it` (0-based)
        with the newest policy file, without waiting for it. The games are
        drawn here, in the main thread (`self.rng`)."""
        c, t, rng = self.cfg, time.perf_counter(), self.rng.getstate()
        shaping = c.shaping * max(0.0, 1 - it / max(c.shaping_anneal_iters, 1))
        job = Job([], self.policy, self.iteration + 1, record=True, gamma=c.gamma, lam=c.lam, shaping=shaping, max_turns=c.max_turns, inference=c.inference)
        return _Rollout(shaping, it - self.iteration, rng, t).start(play, self.procs, self._train_specs(it), job, c.workers)

    def _train_specs(self, it: int) -> list[GameSpec]:
        c, specs = self.cfg, []
        base = (it + 1) * 1_000_003 + c.seed * 7919
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
        seeds (game 1 decks), plus best-of-three matches against the bots. Uses
        the newest policy file and the worker pool, so no rollout may be in flight."""
        c, out = self.cfg, {}
        for name, opp in (("random", RANDOM), ("bot", BOT), ("pool0", self.pool[0])):
            res = head_to_head(self.procs, self.policy, opp, c.eval_games, c.workers, self.iteration + 1, c.max_turns, c.inference)
            for deck in ("jund", "blue"):
                out[f"eval/{name}/{deck}"], out[f"eval/{name}/{deck}_ci"], _ = res[deck]
        if c.eval_bo3_matches:
            res = head_to_head_bo3(self.procs, self.policy, BOT, c.eval_bo3_matches, c.workers, self.iteration + 1, c.max_turns, c.inference)
            for deck in ("jund", "blue"):
                out[f"eval/bot_bo3/{deck}"], out[f"eval/bot_bo3/{deck}_ci"], _ = res[deck]
        out.update(benchmark(self.procs, self.policy, c.bench_games, c.bench_bo3_matches, c.workers, self.iteration + 1, c.max_turns, c.inference))
        return out

    # -- main loop -----------------------------------------------------------

    def _more(self, iteration: int, games_total: int) -> bool:
        """Whether an iteration starting at these counters runs."""
        c = self.cfg
        return iteration < c.iterations and not (c.total_games and games_total >= c.total_games)

    def train(self) -> None:
        c = self.cfg
        log = open(os.path.join(c.run, "metrics.jsonl"), "a")
        threads = torch.get_num_threads()
        if c.pipeline:  # the update shares the CPUs with the workers: spinning OMP threads on busy cores slowed it by 60%
            torch.set_num_threads(max(1, min(threads, _cpus() - c.workers)))
        roll = None  # the collected rollout for the coming update, if it was played during the last one
        try:
            while self._more(self.iteration, self.games_total):
                t0 = time.perf_counter()
                if roll is None:
                    roll = self._submit(self.iteration).wait()
                data = roll.data
                nxt = None
                if c.pipeline and self._more(self.iteration + 1, self.games_total + len(data.games)):
                    nxt = self._submit(self.iteration + 1)  # plays with the current weights during the update
                t1 = time.perf_counter()
                stats = ppo_update(self.net, self.opt, data, c.ppo, device=c.device)
                self.iteration += 1
                self.games_total += len(data.games)
                weights = self._weights()
                self._publish(weights)
                if self.iteration % c.snapshot_every == 0:
                    self._snapshot(weights)
                self._checkpoint(weights, nxt.rng if nxt else self.rng.getstate())
                t2 = time.perf_counter()
                if nxt:
                    nxt.wait()  # before evaluating, which needs the pool

                rollout_s = roll.t_ready - roll.t_submit
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
                    "shaping": roll.shaping,
                    "pool_size": len(self.pool),
                    "rollout_s": round(rollout_s, 2),
                    "update_s": round(t2 - t1, 2),
                    "wait_s": round(max(0.0, nxt.t_ready - t2) if nxt else 0.0, 2),
                    "policy_lag": roll.lag,
                    "decisions_per_s": round(len(data.actions) / max(rollout_s, 1e-9)),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                }
                if c.eval_every and self.iteration % c.eval_every == 0:
                    row.update(self.evaluate())
                row["wall_s"] = round(time.perf_counter() - t0, 2)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(_fmt(row), flush=True)
                roll = nxt
        except BaseException:
            self.procs.terminate()  # abandon a rollout or evaluation in flight
            raise
        finally:
            torch.set_num_threads(threads)
            log.close()
            self.procs.close()
            self.procs.join()
            if self.server is not None:
                self.server.close()
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


def _cpus() -> int:
    """CPUs this process may use (its affinity, which the workers inherit)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


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
