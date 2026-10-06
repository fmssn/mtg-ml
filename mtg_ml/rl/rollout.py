"""Self-play rollouts: many games in lockstep with batched policy inference.

A job is a list of games. Each game has a policy per seat:
    "learner"   the current weights; its decisions are recorded for PPO,
    "random"    `agents.RandomAgent` (evaluation baseline),
    "bot"       the scripted bot for that seat's deck (`mtg_ml.bots`),
    <path>      a frozen checkpoint from the opponent pool.
Each (game, seat) carries its own recurrent state and the events it saw
since its last decision. Every recorded player trajectory gets terminal reward +1 / -1 / 0 plus
optional potential-based life shaping, then GAE.

Policies are evaluated either in the worker (`inference="local"`: a CPU
copy of each network, one torch thread; checkpoints are loaded from disk and
cached per process, pool checkpoints never change, the learner is keyed by
its version), or by the central GPU server (`inference="server"`,
`rl/inference.py`), in which case the worker never imports torch and keeps
two groups of games in flight so its CPU work overlaps the server round trip.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..agents import RandomAgent
from ..bots import make_bot
from ..backend import game_class
from ..engine import Game
from ..match import match_decks
from .features import encode_event_hashes, event_hashes, featurize
from .samples import PackedSamples

LEARNER = "learner"
RANDOM = "random"
BOT = "bot"
SCRIPTED = (RANDOM, BOT)

_DECKS: dict[int, tuple] = {}
_MODELS: dict[tuple, object] = {}


def _decks(match_game: int = 1):
    """Maindecks for game 1, sideboarded decks for games 2 and 3."""
    key = min(match_game, 2)
    if key not in _DECKS:
        _DECKS[key] = match_decks(key)
    return _DECKS[key]


def load_policy(path: str, version: int = 0):
    import torch

    from .model import PolicyNet

    key = (path, version)
    if key not in _MODELS:
        if version:  # drop older learner versions
            for k in [k for k in _MODELS if k[0] == path]:
                del _MODELS[k]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyNet(**ckpt["config"])
        net.load_state_dict(ckpt["model"])
        net.eval()
        _MODELS[key] = net
    return _MODELS[key]


def worker_init() -> None:
    import torch

    torch.set_num_threads(1)


@dataclass
class GameSpec:
    seed: int
    seats: tuple[str, str]  # policy spec per seat
    starting_player: int | None = None
    match_game: int = 1  # 1 = maindecks, 2/3 = after sideboarding


@dataclass
class Job:
    games: list[GameSpec]
    learner_path: str
    learner_version: int
    record: bool = True
    gamma: float = 0.995
    lam: float = 0.95
    shaping: float = 0.0
    max_turns: int = 100
    engine: str | None = None  # None: $MTG_ENGINE, else python
    inference: str = "local"  # "local": CPU torch in the worker; "server": the central inference server
    groups: int = 2  # server mode: requests in flight per worker (its games are split into this many groups)


@dataclass
class Trajectory:
    samples: list = field(default_factory=list)  # (state, opts, events)
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    values: list = field(default_factory=list)
    potentials: list = field(default_factory=list)


@dataclass
class Result:
    """Recorded decisions, whole trajectories laid end to end (`lengths`)."""

    samples: PackedSamples = field(default_factory=PackedSamples)  # (state, opts, events), packed
    lengths: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    # per game: (seats, winner, end_reason, turns, decisions, seed)
    games: list = field(default_factory=list)
    # per job: (wall seconds, seconds waiting for inference, decisions)
    timing: list = field(default_factory=list)


def _potential(game: Game, p: int) -> float:
    if getattr(game, "NATIVE", False):
        return (game._g.life(p) - game._g.life(1 - p)) / 20.0
    return (game.players[p].life - game.players[1 - p].life) / 20.0


def _finish(traj: Trajectory, outcome: float, job: Job, out: Result) -> None:
    n = len(traj.actions)
    if n == 0:
        return
    rewards = [0.0] * n
    rewards[-1] = outcome
    if job.shaping:
        phis = traj.potentials + [0.0]  # terminal potential 0, so shaping telescopes to -phi(s0)
        for t in range(n):
            rewards[t] += job.shaping * (job.gamma * phis[t + 1] - phis[t])
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        next_v = traj.values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + job.gamma * next_v - traj.values[t]
        last = delta + job.gamma * job.lam * last
        adv[t] = last
    out.lengths.append(n)
    out.samples += traj.samples
    out.actions += traj.actions
    out.logps += traj.logps
    out.advantages += adv
    out.returns += [a + v for a, v in zip(adv, traj.values)]


class _Seat:
    """Per (game, seat) memory: recurrent state and events since the last decision."""

    __slots__ = ("hidden", "events")

    def __init__(self):
        self.hidden = None
        self.events: list[int] = []  # hashed event tokens


def _step(g: Game, seats: tuple[_Seat, _Seat], a: int) -> None:
    """Take option `a` and tell both players what they saw of it."""
    p = g.decision.player
    mine, theirs = event_hashes(g, a)
    seats[p].events += mine
    seats[1 - p].events += theirs
    g.step(a)


class _LocalEvaluator:
    """Runs the policies in this process (CPU torch), hidden states per (game, seat)."""

    def __init__(self, job: Job):
        import torch

        self.torch = torch
        self.learner = load_policy(job.learner_path, job.learner_version)
        self.hidden: dict = {}

    def submit(self, group: int, items: list):
        from .model import collate

        torch = self.torch
        acts_out, logp_out, val_out = [0] * len(items), [0.0] * len(items), [0.0] * len(items)
        by_pol: dict = {}
        for k, it in enumerate(items):
            by_pol.setdefault(it[0], []).append(k)
        with torch.no_grad():
            for pol, ks in by_pol.items():
                net = self.learner if pol == LEARNER else load_policy(pol)
                hidden = None
                if net.memory != "none":
                    hs = [self.hidden.get((items[k][1], items[k][2])) for k in ks]
                    hidden = torch.stack([net.initial_state(1)[0] if h is None else h for h in hs])
                logits, values, hn = net(collate([items[k][3] for k in ks]), hidden)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
                for r, k in enumerate(ks):
                    if hn is not None:
                        self.hidden[(items[k][1], items[k][2])] = hn[r]
                    acts_out[k], logp_out[k], val_out[k] = int(acts[r]), float(logps[r]), float(values[r])
        return acts_out, logp_out, val_out

    def collect(self, handle):
        return handle


class _ServerEvaluator:
    """Sends decisions to the central inference server (rl/inference.py)."""

    def __init__(self, job: Job):
        from .inference import client

        self.client = client()
        self.learner_key = (job.learner_path, job.learner_version)
        self.started: set = set()

    def submit(self, group: int, items: list):
        rows = []
        for pol, i, p, x in sorted(items, key=lambda it: it[0]):
            key = self.learner_key if pol == LEARNER else (pol, 0)
            fresh = (i, p) not in self.started
            self.started.add((i, p))
            rows.append((key, 2 * i + p, int(fresh), x[0], x[1], x[2]))
        order = sorted(range(len(items)), key=lambda k: items[k][0])
        return self.client.submit(group, rows), order

    def collect(self, handle):
        h, order = handle
        acts, logps, vals = self.client.collect(h)
        n = len(order)
        a, lp, v = [0] * n, [0.0] * n, [0.0] * n
        for r, k in enumerate(order):
            a[k], lp[k], v[k] = acts[r], logps[r], vals[r]
        return a, lp, v


def run_job(job: Job) -> Result:
    Game = game_class(job.engine)
    if 2 * len(job.games) > 8192:
        raise ValueError("a job holds at most 4096 games")
    ev = _ServerEvaluator(job) if job.inference == "server" else _LocalEvaluator(job)
    games, trajs, mem, scripted = [], [], [], {}
    for i, spec in enumerate(job.games):
        games.append(
            Game(_decks(spec.match_game), seed=spec.seed, starting_player=spec.starting_player, max_turns=job.max_turns, match_game=spec.match_game)
        )
        trajs.append((Trajectory(), Trajectory()))
        mem.append((_Seat(), _Seat()))
        for seat, pol in enumerate(spec.seats):
            if pol == RANDOM:
                scripted[(i, seat)] = RandomAgent(seed=spec.seed * 2 + seat)
            elif pol == BOT:
                scripted[(i, seat)] = make_bot(seat)
    t_start = time.perf_counter()
    out = Result()
    n_groups = max(1, min(job.groups, len(games))) if job.inference == "server" else 1
    groups = [[i for i in range(len(games)) if i % n_groups == k] for k in range(n_groups)]
    inflight: list = [None] * n_groups  # (handle, items) per group

    def finish(i: int) -> None:
        g, spec = games[i], job.games[i]
        out.games.append((spec.seats, g.winner, g.end_reason, g.turn, len(g.actions), spec.seed))
        if job.record:
            for p in (0, 1):
                if spec.seats[p] == LEARNER:
                    outcome = 0.0 if g.winner is None else (1.0 if g.winner == p else -1.0)
                    _finish(trajs[i][p], outcome, job, out)
        trajs[i] = mem[i] = None

    def prepare(k: int) -> list:
        """Play scripted moves until each live game of group k needs a policy; featurize those."""
        items, live = [], []
        for i in groups[k]:
            g = games[i]
            while not g.over:
                p = g.decision.player
                pol = job.games[i].seats[p]
                if pol not in SCRIPTED:
                    break
                _step(g, mem[i], scripted[(i, p)].act(g))
            if g.over:
                finish(i)
                continue
            live.append(i)
            seat = mem[i][p]
            state, opts = featurize(g, p)
            x = (state, opts, encode_event_hashes(seat.events))
            seat.events = []
            pot = _potential(g, p) if job.record and pol == LEARNER else 0.0
            items.append((pol, i, p, x, pot))
        groups[k] = live
        return items

    waited = [0.0]

    def apply(k: int) -> None:
        handle, items = inflight[k]
        tw = time.perf_counter()
        acts, logps, values = ev.collect(handle)
        waited[0] += time.perf_counter() - tw
        for (pol, i, p, x, pot), a, lp, v in zip(items, acts, logps, values):
            if job.record and pol == LEARNER:
                tr = trajs[i][p]
                tr.samples.append(x)
                tr.actions.append(a)
                tr.logps.append(lp)
                tr.values.append(v)
                tr.potentials.append(pot)
            _step(games[i], mem[i], a)
        inflight[k] = None

    while True:
        busy = False
        for k in range(n_groups):
            if inflight[k] is not None:
                apply(k)
            items = prepare(k)
            if items:
                inflight[k] = (ev.submit(k, [it[:4] for it in items]), items)
                busy = True
        if not busy and all(f is None for f in inflight):
            break
    out.timing.append((time.perf_counter() - t_start, waited[0], sum(g[4] for g in out.games)))
    return out


def split_games(specs: list[GameSpec], n_jobs: int) -> list[list[GameSpec]]:
    n_jobs = max(1, min(n_jobs, len(specs)))
    return [specs[k::n_jobs] for k in range(n_jobs)]

