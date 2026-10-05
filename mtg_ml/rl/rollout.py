"""Self-play rollouts: many games in lockstep with batched policy inference.

A job is a list of games. Each game has a policy per seat:
    "learner"   the current weights; its decisions are recorded for PPO,
    "random"    `agents.RandomAgent` (evaluation baseline),
    "bot"       the scripted bot for that seat's deck (`mtg_ml.bots`),
    <path>      a frozen checkpoint from the opponent pool.
Each (game, seat) carries its own recurrent state and the events it saw
since its last decision. Every recorded player trajectory gets terminal reward +1 / -1 / 0 plus
optional potential-based life shaping, then GAE.

Workers run on CPU with one torch thread each; checkpoints are loaded from
disk and cached per process (pool checkpoints never change, the learner is
keyed by its version).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..agents import RandomAgent
from ..bots import make_bot
from ..backend import game_class
from ..engine import Game
from ..match import match_decks
from .features import encode_event_hashes, event_hashes, featurize
from .model import PolicyNet, collate

LEARNER = "learner"
RANDOM = "random"
BOT = "bot"
SCRIPTED = (RANDOM, BOT)

_DECKS: dict[int, tuple] = {}
_MODELS: dict[tuple, PolicyNet] = {}


def _decks(match_game: int = 1):
    """Maindecks for game 1, sideboarded decks for games 2 and 3."""
    key = min(match_game, 2)
    if key not in _DECKS:
        _DECKS[key] = match_decks(key)
    return _DECKS[key]


def load_policy(path: str, version: int = 0) -> PolicyNet:
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

    samples: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    # per game: (seats, winner, end_reason, turns, decisions, seed)
    games: list = field(default_factory=list)


def _potential(game: Game, p: int) -> float:
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


@torch.no_grad()
def run_job(job: Job) -> Result:
    learner = load_policy(job.learner_path, job.learner_version)
    Game = game_class(job.engine)
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
    out = Result()
    live = [i for i, g in enumerate(games) if not g.over]
    while live:
        pending: dict[str, list[tuple[int, int, tuple]]] = {}
        for i in live:
            g = games[i]
            p = g.decision.player
            pol = job.games[i].seats[p]
            if pol in SCRIPTED:
                _step(g, mem[i], scripted[(i, p)].act(g))
                continue
            seat = mem[i][p]
            state, opts = featurize(g, p)
            x = (state, opts, encode_event_hashes(seat.events))
            seat.events = []
            pending.setdefault(pol, []).append((i, p, x))
        for pol, items in pending.items():
            net = learner if pol == LEARNER else load_policy(pol)
            hs = [mem[i][p].hidden for i, p, _ in items]
            hidden = None if net.memory == "none" else torch.stack([net.initial_state(1)[0] if h is None else h for h in hs])
            logits, values, hn = net(collate([x for _, _, x in items]), hidden)
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)
            for k, (i, p, x) in enumerate(items):
                g, a = games[i], int(acts[k])
                if hn is not None:
                    mem[i][p].hidden = hn[k]
                if job.record and pol == LEARNER:
                    tr = trajs[i][p]
                    tr.samples.append(x)
                    tr.actions.append(a)
                    tr.logps.append(float(logps[k]))
                    tr.values.append(float(values[k]))
                    tr.potentials.append(_potential(g, p))
                _step(g, mem[i], a)
        still = []
        for i in live:
            g = games[i]
            if not g.over:
                still.append(i)
                continue
            spec = job.games[i]
            out.games.append((spec.seats, g.winner, g.end_reason, g.turn, len(g.actions), spec.seed))
            if job.record:
                for p in (0, 1):
                    if spec.seats[p] == LEARNER:
                        outcome = 0.0 if g.winner is None else (1.0 if g.winner == p else -1.0)
                        _finish(trajs[i][p], outcome, job, out)
            trajs[i] = mem[i] = None
        live = still
    return out


def split_games(specs: list[GameSpec], n_jobs: int) -> list[list[GameSpec]]:
    n_jobs = max(1, min(n_jobs, len(specs)))
    return [specs[k::n_jobs] for k in range(n_jobs)]

