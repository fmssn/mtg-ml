"""Self-play rollouts: many games in lockstep with batched policy inference.

A job plays games (`GameSpec`). Each game has a policy per seat:
    "learner"   the current weights; its decisions are recorded for PPO,
    "random"    `agents.RandomAgent` (evaluation baseline),
    "bot"       the scripted bot for that seat's deck (`mtg_ml.bots`),
    <path>      a frozen checkpoint from the opponent pool.
Each (game, seat) carries its own recurrent state and the events it saw
since its last decision. Every recorded player trajectory gets terminal reward +1 / -1 / 0 plus
optional potential-based life shaping, then GAE (`_finish`): per decision
(gamma, lam), or per game turn (`Job.gamma_turn`, `Job.lam_turn`), with the
recorded values clamped to the reachable return range first.

A worker advances all its live games by one policy decision per step, with
one forward pass per policy present. On a Xeon core a pass of the h128
entity net costs ~0.5 ms plus 20-40 µs per decision, so what matters is
how many games share a step and how many policies they bring. Two ways to
spread games over a pool:
  * fixed split (`split_games`, `pool.map(run_job, jobs)`): each job plays
    its own games, all at once. Its batch shrinks as games end, the round
    waits for the slowest job, and with the opponent pool a job's games
    bring a dozen checkpoints, so a dozen passes per step.
  * shared game queue (`create_pool` + `run_specs`): every job gets all
    specs, keeps `Job.inflight` games live and claims the next spec from
    pool-wide counters whenever a game ends, so batches stay full and jobs
    end close together. Specs are grouped by matchup (the opponent
    network) and a worker sticks to one matchup at a time, so a step needs
    about two passes. Recorded samples come back through shared memory.

Policies are evaluated either in the worker (`inference="local"`: a CPU
copy of each network, one torch thread; checkpoints are memory-mapped and
cached per process, pool checkpoints never change, the learner is keyed by
its version), or by the central GPU server (`inference="server"`,
`rl/inference.py`), in which case the worker never imports torch and keeps
`Job.groups` groups of games in flight so its CPU work overlaps the server
round trip. Both keep recurrent state in a table indexed by slot (two per
live game, reused when a game ends; a seat's first decision is `fresh`).
"""

from __future__ import annotations

import itertools
import os
import time
from array import array
from dataclasses import dataclass, field, fields, replace
from operator import itemgetter

from ..agents import RandomAgent
from ..bots import make_bot
from ..backend import game_class
from ..engine import Game
from ..engine.objects import (
    ASSIGN_DAMAGE,
    CHOOSE_CARD,
    CHOOSE_MODE,
    CHOOSE_X,
    DECLARE_ATTACKER,
    DECLARE_BLOCKER,
    EXILE_FROM_GY,
    MULLIGAN,
    ORDER,
    ORDER_TRIGGERS,
    PAY_MANA,
    PRIORITY,
    SACRIFICE,
    TARGET,
    YES_NO,
)
from ..match import match_decks
from .features import encode_event_hashes, event_hashes, featurize_flat
from .samples import FIELDS, PackedSamples

LEARNER = "learner"
RANDOM = "random"
BOT = "bot"
SCRIPTED = (RANDOM, BOT)
MAX_LIVE = 4096  # games in play per job: the server keeps 8192 hidden-state slots per worker
MAX_MATCHUPS = 1024  # claim counters a pool shares (run_specs)
# Decision kinds as recorded per decision (`Result.kinds`, small ints); id 0
# is any kind not listed. The PPO statistics break entropy and KL down by them.
KINDS = ("other", PRIORITY, PAY_MANA, TARGET, DECLARE_ATTACKER, DECLARE_BLOCKER, YES_NO, CHOOSE_CARD, CHOOSE_X, SACRIFICE,
         EXILE_FROM_GY, ORDER, ORDER_TRIGGERS, ASSIGN_DAMAGE, CHOOSE_MODE, MULLIGAN)  # fmt: skip
KIND_ID = {k: i for i, k in enumerate(KINDS)}

_DECKS: dict[int, tuple] = {}
_MODELS: dict[tuple, object] = {}
_CLAIM = None  # the pool's shared next-spec index per matchup (multiprocessing Array), set by the pool initializer
_PROFILE = None


def _decks(match_game: int = 1):
    """Maindecks for game 1, sideboarded decks for games 2 and 3."""
    key = min(match_game, 2)
    if key not in _DECKS:
        _DECKS[key] = match_decks(key)
    return _DECKS[key]


def load_net(path: str):
    """A checkpoint's network on the CPU, in eval mode. The file is
    memory-mapped and only `model` is read: built on the meta device and
    given the mapped tensors (`assign`), so there is no random init, no copy,
    and the optimizer state that makes up 2/3 of a training checkpoint is
    never touched (~5 ms instead of ~220 ms for the h128 entity net). Workers
    share the pages through the page cache. Checkpoints must be replaced
    atomically (write + os.replace, as the trainer does), never rewritten in
    place while mapped."""
    import torch

    from .model import PolicyNet

    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    with torch.device("meta"):
        net = PolicyNet(**ckpt["config"])
    net.load_state_dict(ckpt["model"], assign=True)
    return net.eval()


def load_policy(path: str, version: int = 0):
    """`load_net`, cached per process. Frozen checkpoints (version 0) stay
    cached; a learner version replaces every other learner version, whatever
    its path (the trainer may write a new file per iteration)."""
    key = (path, version)
    if key not in _MODELS:
        if version:
            for k in [k for k in _MODELS if k[1]]:
                del _MODELS[k]
        _MODELS[key] = load_net(path)
    return _MODELS[key]


def worker_init(claim=None, ids=None, cpus=None) -> None:
    """Initializer of a local-inference pool: one torch thread. `create_pool`
    also hands over the pool's spec counters and, to pin worker i to
    cpus[i % len(cpus)] (Linux), a shared id counter."""
    import torch

    torch.set_num_threads(1)
    connect(claim)
    if ids is not None and cpus and hasattr(os, "sched_setaffinity"):
        with ids.get_lock():
            wid = ids.value
            ids.value += 1
        os.sched_setaffinity(0, {cpus[wid % len(cpus)]})


def connect(claim) -> None:
    """Use `claim` (a shared `Array("i", MAX_MATCHUPS)`) as this process's
    spec counters for streaming jobs. Shared arrays cannot be pickled into
    tasks, only inherited, so pool initializers call this."""
    global _CLAIM
    _CLAIM = claim


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
    # GAE bootstraps from values clamped to +-(value_clamp + |shaping|), the
    # range of the returns (0 = off). An unbounded value head drifts past +-1.
    value_clamp: float = 1.0
    # > 0: discount per game turn instead of per decision: between consecutive
    # decisions gamma_turn ** (turns elapsed) (lam_turn likewise), so a line's
    # cost no longer depends on how many clicks (passes, mana payments) it takes.
    gamma_turn: float = 0.0
    lam_turn: float = 0.0
    max_turns: int = 100
    engine: str | None = None  # None: $MTG_ENGINE, else python
    inference: str = "local"  # "local": CPU torch in the worker; "server": the central inference server
    groups: int = 2  # server mode: requests in flight per worker (its live games are split into this many groups)
    # 0: play all `games` at once. n > 0: keep n games live, claiming more of
    # `games` from the pool's shared counters as games end (`run_specs`).
    inflight: int = 0
    # every network seat (learner and frozen checkpoints) takes its most
    # likely option instead of sampling (ties: the first). Evaluation only:
    # recorded games must come from the sampling policy.
    greedy: bool = False


@dataclass
class Trajectory:
    samples: list = field(default_factory=list)  # (state, option lengths, option tokens, events), int32 arrays
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    values: list = field(default_factory=list)
    potentials: list = field(default_factory=list)
    turns: list = field(default_factory=list)  # the game's turn counter at each decision (each player's turn counts)
    kinds: list = field(default_factory=list)


@dataclass
class Result:
    """Recorded decisions, whole trajectories laid end to end (`lengths`)."""

    samples: PackedSamples = field(default_factory=PackedSamples)  # (state, opts, events), packed
    lengths: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    kinds: list = field(default_factory=list)  # decision kind per recorded decision (`KIND_ID`)
    # per game: (seats, winner, end_reason, turns, decisions, seed)
    games: list = field(default_factory=list)
    # per job: (wall seconds, seconds waiting for inference, decisions)
    timing: list = field(default_factory=list)


def _potential(game: Game, p: int) -> float:
    if getattr(game, "NATIVE", False):
        return (game._g.life(p) - game._g.life(1 - p)) / 20.0
    return (game.players[p].life - game.players[1 - p].life) / 20.0


def _discounts(traj: Trajectory, job: Job) -> tuple[list[float], list[float]]:
    """(gamma, lam) between decision t and t + 1, per t: job.gamma / job.lam
    per decision, or gamma_turn / lam_turn to the power of the game turns
    between the two decisions when those are set (decisions within one
    turn: 1). The last entry (to the terminal state) is 1 per turn: the
    terminal value and potential are 0, so only the bootstrapping matters."""
    n = len(traj.actions)
    turns = traj.turns
    if (job.gamma_turn > 0 or job.lam_turn > 0) and len(turns) != n:
        raise ValueError("per-turn discounting needs the turn of every recorded decision")
    gaps = [turns[t + 1] - turns[t] for t in range(n - 1)] + [0] if turns else None
    gam = [job.gamma_turn**d for d in gaps] if job.gamma_turn > 0 else [job.gamma] * n
    lam = [job.lam_turn**d for d in gaps] if job.lam_turn > 0 else [job.lam] * n
    return gam, lam


def _finish(traj: Trajectory, outcome: float, job: Job, out: Result) -> None:
    n = len(traj.actions)
    if n == 0:
        return
    gam, lam = _discounts(traj, job)
    rewards = [0.0] * n
    rewards[-1] = outcome
    if job.shaping:
        phis = traj.potentials + [0.0]  # terminal potential 0, so shaping telescopes to -phi(s0)
        for t in range(n):
            rewards[t] += job.shaping * (gam[t] * phis[t + 1] - phis[t])
    values = traj.values  # raw, as the network said them (traj.values stays untouched)
    if job.value_clamp > 0:
        hi = job.value_clamp + abs(job.shaping)
        values = [min(hi, max(-hi, v)) for v in values]
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gam[t] * next_v - values[t]
        last = delta + gam[t] * lam[t] * last
        adv[t] = last
    out.lengths.append(n)
    out.samples += traj.samples
    out.actions += traj.actions
    out.logps += traj.logps
    out.advantages += adv
    out.returns += [a + v for a, v in zip(adv, values)]
    out.kinds += traj.kinds


class _Seat:
    """Per (game, seat) memory: events since the last decision; `fresh`
    until the seat's first policy decision (its recurrent state starts at zero)."""

    __slots__ = ("events", "fresh")

    def __init__(self):
        self.events: list[int] = []  # hashed event tokens
        self.fresh = True


class _Live:
    """A game in play: its spec, engine, slot (hidden-state rows 2 * slot + seat),
    trajectories, seat memories and scripted agents."""

    __slots__ = ("spec", "game", "slot", "trajs", "seats", "agents")

    def __init__(self, spec: GameSpec, game: Game, slot: int):
        self.spec, self.game, self.slot = spec, game, slot
        self.trajs = (Trajectory(), Trajectory())
        self.seats = (_Seat(), _Seat())
        self.agents = tuple(RandomAgent(seed=spec.seed * 2 + s) if pol == RANDOM else make_bot(s) if pol == BOT else None for s, pol in enumerate(spec.seats))


def _step(g: Game, seats: tuple[_Seat, _Seat], a: int) -> None:
    """Take option `a` and tell both players what they saw of it."""
    p = g.decision.player
    mine, theirs = event_hashes(g, a)
    seats[p].events += mine
    seats[1 - p].events += theirs
    g.step(a)


def _batch(torch, xs: list):
    """Step-mode model input of decisions xs = (state, option lengths, option
    tokens, events) as int32 arrays: values and offsets go into one int32
    buffer with array copies (no per-token Python work), viewed as one tensor
    and split. Returns (Batch, most options of a decision)."""
    from .model import Batch

    s, e, o = array("i"), array("i"), array("i")
    s_off, e_off, o_off, o_row, o_pos, n_opts = (array("i") for _ in range(6))
    for r, (st, ol, of, ev) in enumerate(xs):
        s_off.append(len(s))
        s.extend(st)
        e_off.append(len(e))
        e.extend(ev)
        k = len(ol)
        n_opts.append(k)
        o_row.extend([r] * k)
        o_pos.extend(range(k))
        o_off.extend(itertools.accumulate(ol[:-1], initial=len(o)))
        o.extend(of)
    parts = (s, s_off, e, e_off, o, o_off, o_row, o_pos, n_opts)
    flat = array("i")
    for p in parts:
        flat.extend(p)
    return Batch(*torch.split(torch.frombuffer(flat, dtype=torch.int32), [len(p) for p in parts])), max(n_opts)


class _LocalEvaluator:
    """Runs the policies in this process (CPU torch). Items are (policy, slot,
    fresh, x), sorted by policy; `collect` runs one forward per policy (so
    the time a job waits for inference is the inference time) and keeps
    recurrent state in a (slots, state size) table per policy."""

    def __init__(self, job: Job, slots: int):
        import torch

        self.torch = torch
        self.learner = load_policy(job.learner_path, job.learner_version)
        self.slots = slots
        self.greedy = job.greedy
        self.hidden: dict = {}

    def submit(self, group: int, items: list):
        return items

    def collect(self, items: list):
        torch = self.torch
        acts, logps, vals = [], [], []
        with torch.inference_mode():
            for pol, run in itertools.groupby(items, key=itemgetter(0)):
                run = list(run)
                net = self.learner if pol == LEARNER else load_policy(pol)
                batch, width = _batch(torch, [it[3] for it in run])
                hidden = None
                if net.memory != "none":
                    table = self.hidden.get(pol)
                    if table is None:
                        table = self.hidden[pol] = torch.zeros(self.slots, net.state_size)
                    slots = torch.tensor([it[1] for it in run])
                    hidden = table[slots]
                    fresh = [k for k, it in enumerate(run) if it[2]]
                    if fresh:
                        hidden[fresh] = 0.0
                logits, values, hn = net(batch, hidden, max_options=width)
                if hn is not None:
                    table[slots] = hn
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                a = logits.argmax(-1) if self.greedy else dist.sample()  # padded options are -inf
                acts += a.tolist()
                logps += dist.log_prob(a).tolist()
                vals += values.tolist()
        return acts, logps, vals


class _ServerEvaluator:
    """Sends decisions to the central inference server (rl/inference.py)."""

    def __init__(self, job: Job, slots: int):
        from .inference import GREEDY_FLAG, client

        self.client = client()
        self.learner_key = (job.learner_path, job.learner_version)
        self.flag = GREEDY_FLAG if job.greedy else 0

    def submit(self, group: int, items: list):
        key, flag = self.learner_key, self.flag
        return self.client.submit(group, [(key if it[0] == LEARNER else (it[0], 0), it[1], it[2] | flag, *it[3]) for it in items])

    def collect(self, handle):
        return self.client.collect(handle)


def run_job(job: Job) -> Result:
    """Play a job. `MTG_WORKER_PROFILE=path` cProfiles every call in each
    worker process (cumulative per process, written to path.<pid>)."""
    path = os.environ.get("MTG_WORKER_PROFILE")
    if not path:
        return _play(job)
    global _PROFILE
    import cProfile

    _PROFILE = _PROFILE or cProfile.Profile()
    _PROFILE.enable()
    try:
        return _play(job)
    finally:
        _PROFILE.disable()
        _PROFILE.dump_stats(f"{path}.{os.getpid()}")


def _matchup(spec: GameSpec) -> tuple:
    """The networks a game needs besides the learner's (() for self-play and
    games against scripted seats)."""
    return tuple(sorted(set(spec.seats) - {LEARNER, *SCRIPTED}))


def _matchup_bounds(specs: list[GameSpec]) -> list[int]:
    """Starts of the runs of equal `_matchup` in `specs` (sorted by it in
    `run_specs`), then the end. Run 0 is always the learner-only one,
    possibly empty; runs past MAX_MATCHUPS are merged into the last."""
    keys = [_matchup(sp) for sp in specs]
    starts = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    if keys and keys[0]:
        starts.insert(0, 0)
    return starts[:MAX_MATCHUPS] + [len(specs)]


def _claimer(job: Job):
    """claim(n) -> indices of up to n more of `job.games`: all of them for a
    plain job. For a streaming job they come from the pool's shared array of
    next index per matchup, under its lock (a few µs per claim). A worker
    sticks to one opponent network: it claims from its current matchup,
    then learner-only games, then switches to the matchup with the most
    games left. So a step needs ~2 forward passes, not one per pool
    checkpoint its games happen to bring."""
    if not job.inflight:
        rest = iter(range(len(job.games)))
        return lambda n: list(itertools.islice(rest, n))
    if _CLAIM is None:
        raise RuntimeError("Job.inflight needs a pool from rollout.create_pool or InferenceServer.pool (run_specs)")
    ends = _matchup_bounds(job.games)[1:]
    pos = _CLAIM.get_obj()
    left = lambda m: ends[m] - pos[m]  # noqa: E731
    largest = lambda: max(range(1, len(ends)), key=left, default=0)  # noqa: E731
    cur, dry = None, False

    def take(n: int) -> list[int]:
        nonlocal cur, dry
        got: list[int] = []
        if n <= 0 or dry:
            return got
        with _CLAIM.get_lock():
            if cur is None:
                cur = largest()
            while len(got) < n:
                m = cur if left(cur) > 0 else 0 if left(0) > 0 else largest()
                k = min(n - len(got), left(m))
                if k <= 0:
                    dry = True
                    break
                cur = m or cur
                got.extend(range(pos[m], pos[m] + k))
                pos[m] += k
        return got

    return take


def _play(job: Job) -> Result:
    t_start = time.perf_counter()
    if job.greedy and job.record:
        raise ValueError("greedy games are for evaluation: PPO needs games of the sampling policy (record=False)")
    Game = game_class(job.engine)
    cap = job.inflight or len(job.games)
    if cap > MAX_LIVE:
        raise ValueError(f"a job holds at most {MAX_LIVE} live games")
    ev = _ServerEvaluator(job, 2 * cap) if job.inference == "server" else _LocalEvaluator(job, 2 * cap)
    claim = _claimer(job)
    free = list(range(cap - 1, -1, -1))  # game slots
    n_groups = max(1, min(job.groups, cap)) if job.inference == "server" else 1
    room = -(-cap // n_groups)  # live games per group
    groups: list[list[_Live]] = [[] for _ in range(n_groups)]
    inflight: list = [None] * n_groups  # (handle, items) per group
    out = Result()

    def start(i: int) -> _Live:
        spec = job.games[i]
        g = Game(_decks(spec.match_game), seed=spec.seed, starting_player=spec.starting_player, max_turns=job.max_turns, match_game=spec.match_game)
        return _Live(spec, g, free.pop())

    def finish(lv: _Live) -> None:
        g, spec = lv.game, lv.spec
        out.games.append((spec.seats, g.winner, g.end_reason, g.turn, len(g.actions), spec.seed))
        if job.record:
            for p in (0, 1):
                if spec.seats[p] == LEARNER:
                    outcome = 0.0 if g.winner is None else (1.0 if g.winner == p else -1.0)
                    _finish(lv.trajs[p], outcome, job, out)
        free.append(lv.slot)

    def prepare(k: int) -> list:
        """Play scripted moves until each game of group k needs a policy (top
        the group up with newly claimed games as others end); featurize
        those decisions. Items come back sorted by policy."""
        items, live, todo = [], [], groups[k]
        while True:
            for lv in todo:
                g, seats = lv.game, lv.spec.seats
                while not g.over:
                    p = g.decision.player
                    pol = seats[p]
                    if pol not in SCRIPTED:
                        break
                    _step(g, lv.seats, lv.agents[p].act(g))
                if g.over:
                    finish(lv)
                    continue
                live.append(lv)
                seat = lv.seats[p]
                state, o_len, o_flat = featurize_flat(g, p)
                # int32 arrays once: batching and recording then only copy memory
                x = (array("i", state), array("i", o_len), array("i", o_flat), array("i", encode_event_hashes(seat.events)))
                seat.events = []
                pot = _potential(g, p) if job.record and pol == LEARNER else 0.0
                items.append((pol, 2 * lv.slot + p, seat.fresh, x, lv, p, pot, g.turn))
                seat.fresh = False
            todo = [start(i) for i in claim(min(room - len(live), len(free)))]
            if not todo:
                break
        groups[k] = live
        items.sort(key=itemgetter(0))
        return items

    waited = 0.0

    def apply(k: int) -> None:
        nonlocal waited
        handle, items = inflight[k]
        tw = time.perf_counter()
        acts, logps, values = ev.collect(handle)
        waited += time.perf_counter() - tw
        for (pol, _, _, x, lv, p, pot, turn), a, lp, v in zip(items, acts, logps, values):
            if job.record and pol == LEARNER:
                tr = lv.trajs[p]
                tr.kinds.append(KIND_ID.get(lv.game.decision.kind, 0))
                tr.samples.append(x)
                tr.actions.append(a)
                tr.logps.append(lp)
                tr.values.append(v)
                tr.potentials.append(pot)
                tr.turns.append(turn)
            _step(lv.game, lv.seats, a)
        inflight[k] = None

    try:
        while True:
            busy = False
            for k in range(n_groups):
                if inflight[k] is not None:
                    apply(k)
                items = prepare(k)
                if items:
                    inflight[k] = (ev.submit(k, items), items)
                    busy = True
            if not busy and all(f is None for f in inflight):
                break
    except BaseException:
        # Collect every request still in flight before the error leaves the
        # job: the server may still be reading a request area, and the next
        # job in this worker writes its requests to the same areas.
        for f in inflight:
            if f is not None:
                try:
                    ev.collect(f[0])
                except Exception:  # noqa: BLE001, S110 - the server failed too; the original error matters
                    pass
        raise
    out.timing.append((time.perf_counter() - t_start, waited, sum(g[4] for g in out.games)))
    return out


def split_games(specs: list[GameSpec], n_jobs: int) -> list[list[GameSpec]]:
    n_jobs = max(1, min(n_jobs, len(specs)))
    return [specs[k::n_jobs] for k in range(n_jobs)]


def create_pool(workers: int, inference: str = "local", server=None, worker_cpus: tuple[int, ...] | None = None):
    """A worker pool for `run_specs` (`pool.map(run_job, jobs)` works too),
    carrying its shared spec counters as `pool.claim`. Local inference:
    spawned workers with one torch thread each, pinned one per CPU of
    `worker_cpus` (Linux). Server: `server.pool()` (the server pins its
    workers itself). `Trainer` switches with `self.procs =
    create_pool(cfg.workers, cfg.inference, self.server)`."""
    if inference == "server":
        if server is None:
            raise ValueError("inference='server' needs the InferenceServer")
        return server.pool()
    if inference != "local":
        raise ValueError(f"inference must be local or server, not {inference!r}")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    claim = ctx.Array("i", MAX_MATCHUPS)
    pool = ctx.Pool(workers, initializer=worker_init, initargs=(claim, ctx.Value("i", 0), worker_cpus))
    pool.claim = claim
    return pool


def run_specs(pool, specs: list[GameSpec], job: Job, workers: int, inflight: int = 64) -> Result:
    """Play `specs` on `pool` (from `create_pool`) through a shared game queue
    and merge the results; `job` gives everything but the games. Each of the
    `workers` jobs gets every spec (2048 pickle to ~60 KB) and keeps up to
    `inflight` games live, fewer when there are too few specs to go round
    (2048 games on 31 workers: 64 measured best; 32-48 lose more to smaller
    batches than they gain in balance). Specs are sorted by matchup (the networks they need besides the
    learner's) so each worker can stick to one opponent (`_claimer`).
    Results come in finishing order (trajectories are independent).
    `Trainer._run` and `evaluate.head_to_head(_bo3)` switch to it with
    `run_specs(procs, specs, Job([], ...), n_jobs)` (one merged Result
    instead of a list) once their pool comes from `create_pool`."""
    merged = Result()
    if not specs:
        return merged
    specs = sorted(specs, key=_matchup)
    starts = _matchup_bounds(specs)[:-1]
    with pool.claim.get_lock():
        pool.claim.get_obj()[: len(starts)] = starts
    n = max(1, min(inflight, -(-len(specs) // workers)))
    for r in pool.imap_unordered(_run_parked, [replace(job, games=specs, inflight=n) for _ in range(workers)]):
        r.samples.unpark(merged.samples)  # merged as jobs end, while others still play
        for f in fields(Result):
            if f.name != "samples":
                getattr(merged, f.name).extend(getattr(r, f.name))
    return merged


def play(pool, specs: list[GameSpec], job: Job, workers: int, inflight: int = 64) -> Result:
    """`run_specs` when `pool` carries the shared counters (`create_pool`,
    `InferenceServer.pool`); otherwise a fixed split over `pool.map` (a plain
    `Pool`, or an in-process stand-in with `map`). What the trainer and the
    evaluation call."""
    if getattr(pool, "claim", None) is not None:
        return run_specs(pool, specs, job, workers, inflight)
    merged = Result()
    for r in pool.map(run_job, [replace(job, games=chunk) for chunk in split_games(specs, workers)]):
        for f in fields(Result):
            getattr(merged, f.name).extend(getattr(r, f.name))
    return merged


class _Parked:
    """A job's samples parked in a shared-memory block for the trip back to
    `run_specs`. Through the result pipe they cost the parent four copies
    (read, unpickle, frombytes, merge; ~25 ms per job of 12 MB, in one
    thread, mostly after the last job ends); from the block, one."""

    def __init__(self, ps: PackedSamples):
        from multiprocessing import shared_memory

        parts = [memoryview(getattr(ps, f)).cast("B") for f in FIELDS]
        self.sizes = [len(b) for b in parts]
        shm = shared_memory.SharedMemory(create=True, size=max(sum(self.sizes), 1))
        pos = 0
        for b in parts:
            shm.buf[pos : pos + len(b)] = b
            pos += len(b)
        self.name = shm.name
        shm.close()

    def unpark(self, into: PackedSamples) -> None:
        """Append the samples to `into` (being built: its cached views are not
        reset) and free the block."""
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=self.name)
        pos = 0
        for f, n in zip(FIELDS, self.sizes):
            getattr(into, f).frombytes(shm.buf[pos : pos + n])
            pos += n
        shm.close()
        shm.unlink()


def _run_parked(job: Job) -> Result:
    out = run_job(job)
    out.samples = _Parked(out.samples)
    return out
