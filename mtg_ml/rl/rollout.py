"""Self-play rollouts: many games in lockstep with batched policy inference.

A job plays games (`GameSpec`). Each game has a policy per seat:
    "learner"   the current weights; its decisions are recorded for PPO,
    "random"    `agents.RandomAgent` (evaluation baseline),
    "bot"       the scripted bot for that seat's deck (`mtg_ml.bots`),
    <path>      a frozen checkpoint from the opponent pool.
Each (game, seat) carries its own recurrent state and the events it saw
since its last decision, and is featurized in the feature-set version of
its own policy (`policy_features`), so a pool snapshot or ladder rung
trained before a feature set existed never sees its features. Every recorded player trajectory gets terminal reward +1 / -1 / 0 plus
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

import io
import itertools
import os
import pickle
import time
import zipfile
from array import array
from dataclasses import dataclass, field, fields, replace
from operator import itemgetter

from ..agents import RandomAgent
from ..bots import make_bot
from ..backend import game_class
from ..encode import BELIEF_FEATURES, check_features
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
from ..knowledge import MatchKnowledge
from ..match import DEFAULT_MATCHUP, first_starting_player, game_args, game_seed, matchup_decks, next_starting_player
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
         EXILE_FROM_GY, ORDER, ORDER_TRIGGERS, ASSIGN_DAMAGE, CHOOSE_MODE, MULLIGAN, "assign_damage_amount")  # fmt: skip
KIND_ID = {k: i for i, k in enumerate(KINDS)}

_DECKS: dict[tuple, dict] = {}
_MODELS: dict[tuple, object] = {}
_CLAIM = None  # the pool's shared next-spec index per matchup (multiprocessing Array), set by the pool initializer
_PROFILE = None


def _game_args(match_game: int = 1, matchup: str = DEFAULT_MATCHUP, variants: tuple[str, str] | None = None) -> dict:
    """Decks (maindecks for game 1, sideboarded for games 2 and 3) and deck
    names, as Game arguments. `variants`: each canonical position's
    registered 75 (`engine.variants`) instead of the stock lists; games 2
    and 3 apply that variant's plan against the opponent's archetype, which
    changes the partition, never the 75."""
    key = (min(match_game, 2), matchup, variants)
    if key not in _DECKS:
        _DECKS[key] = game_args(key[0], matchup) if variants is None else _variant_args(key[0], matchup, variants)
    return {**_DECKS[key], "match_game": match_game}


def _variant_args(game_no: int, matchup: str, variants: tuple[str, str]) -> dict:
    from ..engine.decks import expand
    from ..engine.variants import VARIANTS, plan_for_variant
    from ..match import deck_names

    archetypes = matchup_decks(matchup)
    vs = tuple(VARIANTS[v] for v in variants)
    for s, v in enumerate(vs):
        if v.archetype != archetypes[s]:
            raise ValueError(f"variant {v.id!r} is {v.archetype}, but {matchup} seats {archetypes[s]} at position {s}")
    if game_no == 1:
        decks = tuple(expand(v.main) for v in vs)
    else:
        decks = tuple(expand(plan_for_variant(v, archetypes[1 - s]).apply()) for s, v in enumerate(vs))
    return {"decks": decks, "deck_names": deck_names(matchup),
            "registered_main": tuple(expand(v.main) for v in vs),
            "registered_sideboards": tuple(expand(v.sideboard) for v in vs)}


def registered_lists(matchup: str, variants: tuple[str, str] | None, position: int) -> tuple[str, dict[str, int]]:
    """(archetype, registered 75 as card counts) of a canonical position:
    the belief head's training target, never one of its inputs."""
    archetype = matchup_decks(matchup)[position]
    if variants is None:
        from ..engine.decks import DECKS, SIDEBOARDS

        main, side = DECKS[archetype], SIDEBOARDS[archetype]
    else:
        from ..engine.variants import VARIANTS

        v = VARIANTS[variants[position]]
        main, side = v.main, v.sideboard
    out = dict(main)
    for name, n in side.items():
        out[name] = out.get(name, 0) + n
    return archetype, out


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


class _Stub:
    """Whatever a checkpoint pickles besides plain containers (tensors,
    storages, numpy arrays): `checkpoint_config` only wants the config."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __setstate__(self, state):
        pass


class _ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(module, name)
        return _Stub

    def persistent_load(self, pid):
        return None


def checkpoint_config(path: str) -> dict:
    """A checkpoint's `config` without torch: rollout workers that use the
    inference server never import it. Reads the pickle of torch's zip format
    with every tensor stubbed out (no tensor data is read); falls back to
    `torch.load` for anything else."""
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
            return dict(_ConfigUnpickler(io.BytesIO(z.read(name))).load()["config"])
    except (zipfile.BadZipFile, StopIteration, KeyError, TypeError, pickle.UnpicklingError, AttributeError):
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)["config"]


_FEATURES: dict[tuple, int] = {}


def policy_features(path: str, version: int = 0) -> int:
    """Feature-set version of the policy in checkpoint `path` (its config's
    `features`; absent = 1), cached per process like `load_policy`: a
    learner version replaces the previous learner version."""
    key = (path, version)
    if key not in _FEATURES:
        if version:
            for k in [k for k in _FEATURES if k[1]]:
                del _FEATURES[k]
        _FEATURES[key] = check_features(checkpoint_config(path).get("features", 1))
    return _FEATURES[key]


_BELIEFS: dict[tuple, object] = {}


def policy_belief(path: str, version: int = 0):
    """The `BeliefSpec` of a feature-set-8 checkpoint (config `belief`), or
    None for a policy without a belief head; cached like `policy_features`."""
    key = (path, version)
    if key not in _BELIEFS:
        if version:
            for k in [k for k in _BELIEFS if k[1]]:
                del _BELIEFS[k]
        cfg = checkpoint_config(path).get("belief")
        if cfg:
            from .belief import BeliefSpec

            _BELIEFS[key] = BeliefSpec.from_config(cfg)
        else:
            _BELIEFS[key] = None
    return _BELIEFS[key]


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
    matchup: str = DEFAULT_MATCHUP  # canonical deck positions, independent of physical seats
    swap_seats: bool = False
    # True: the spec is a whole best-of-three match from game 1 (`seed` is the
    # match seed, game n plays with `match.game_seed(seed, n)`), with fixed
    # registrations, policies and seat mapping, each player's witnessed
    # opponent cards carried into the next game (`MatchKnowledge`).
    bo3: bool = False
    # each canonical position's registered 75 (`engine.variants` ids); None: the stock lists
    variants: tuple[str, str] | None = None
    # each canonical position's knowledge from earlier games of its match
    # (`MatchKnowledge.to_dict`), for a single game scheduled from outside;
    # None: a fresh match
    knowledge: tuple[dict | None, dict | None] = (None, None)

    @property
    def physical_seats(self) -> tuple[str, str]:
        return self.seats[::-1] if self.swap_seats else self.seats

    def canonical(self, physical: int) -> int:
        return 1 - physical if self.swap_seats else physical

    def game_args(self, match_game: int | None = None, starting_player: int | None = -1) -> dict:
        """Game keyword arguments of game `match_game` (default: the spec's)
        starting with canonical `starting_player` (default: the spec's)."""
        args = _game_args(self.match_game if match_game is None else match_game, self.matchup, self.variants)
        if self.swap_seats:
            for name in ("decks", "registered_main", "registered_sideboards"):
                args[name] = args[name][::-1]
            # Legacy policies still need labels relative to their physical seats.
            from ..match import DECK_NAMES
            decks = matchup_decks(self.matchup)[::-1]
            args["deck_names"] = tuple(d if d != DECK_NAMES[s] else None for s, d in enumerate(decks))
        start = self.starting_player if starting_player == -1 else starting_player
        args["starting_player"] = 1 - start if self.swap_seats and start is not None else start
        return args


@dataclass
class Job:
    games: list[GameSpec]
    learner_path: str
    learner_version: int
    record: bool = True
    gamma: float = 0.995
    lam: float = 0.95
    shaping: float = 0.0
    # GAE clamps unshaped value to +-value_clamp, then shifts by -shaping * phi.
    # 0 disables clamping. Raw predictions remain available for logging.
    value_clamp: float = 1.0
    # > 0: discount per game turn instead of per decision: between consecutive
    # decisions gamma_turn ** (turns elapsed) (lam_turn likewise), so a line's
    # cost no longer depends on how many clicks (passes, mana payments) it takes.
    gamma_turn: float = 0.0
    lam_turn: float = 0.0
    max_turns: int = 100
    auto_mana: bool = False  # Game(auto_mana=...): colour-preserving auto payment of non-strategic mana costs
    auto_pass: bool = False  # Game(auto_pass=...): collapse uneventful priority passes
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
    # {policy: feature-set version} for network seats (LEARNER or a checkpoint
    # path) whose checkpoint's own version is wrong: models trained on set 2
    # before checkpoints recorded it (docs/features.md). Others: `policy_features`.
    features: dict = field(default_factory=dict)


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
    trajectory_ids: list = field(default_factory=list)  # (game seed, seat), for stable residency-wave merging
    actions: list = field(default_factory=list)
    logps: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    kinds: list = field(default_factory=list)  # decision kind per recorded decision (`KIND_ID`)
    # per game: (seats, winner, end_reason, turns, decisions, seed); the
    # games of a best-of-three spec carry `match.game_seed(match seed, n)`
    games: list = field(default_factory=list)
    # per best-of-three spec: (seats, match seed, matchup, ((canonical
    # starting player, canonical winner, reason) per game), match winner)
    matches: list = field(default_factory=list)
    # per recorded trajectory (aligned with `lengths`) of a learner with a
    # belief head: (archetype index, per-card copies of the opponent's
    # registered 75), `BeliefSpec.target`. Training data only.
    belief_targets: list = field(default_factory=list)
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
        if len(traj.potentials) != n:
            raise ValueError("shaping needs the potential of every recorded decision")
        phis = traj.potentials + [0.0]  # terminal potential 0, so shaping telescopes to -phi(s0)
        for t in range(n):
            rewards[t] += job.shaping * (gam[t] * phis[t + 1] - phis[t])
    values = traj.values  # raw, as the network said them (traj.values stays untouched)
    if job.value_clamp > 0:
        shifts = [job.shaping * phi for phi in traj.potentials] if job.shaping else [0.0] * n
        values = [min(job.value_clamp - shift, max(-job.value_clamp - shift, v)) for v, shift in zip(values, shifts)]
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
    trajectories, seat memories and scripted agents. A best-of-three spec
    keeps its `_Live` (and slot) for the whole match: `next_game` starts the
    next game with fresh trajectories and seat memories (the recurrent state
    restarts each game), and `knowledge` (per canonical position) carries
    what each player witnessed of the other into it."""

    __slots__ = ("spec", "game", "slot", "trajs", "seats", "agents", "game_no", "start", "results", "knowledge")

    def __init__(self, spec: GameSpec, slot: int, Game, kw: dict):
        self.spec, self.slot = spec, slot
        self.knowledge = [MatchKnowledge.from_dict(k) for k in spec.knowledge]
        self.results: list = []  # (canonical starting player, canonical winner, reason) per finished game
        self.game_no = spec.match_game
        if spec.bo3:
            if spec.match_game != 1:
                raise ValueError("a best-of-three spec starts at game 1")
            self.start = first_starting_player(spec.seed) if spec.starting_player is None else spec.starting_player
        else:
            self.start = spec.starting_player
        self._new_game(Game, kw)

    def _new_game(self, Game, kw: dict) -> None:
        spec = self.spec
        seed = game_seed(spec.seed, self.game_no) if spec.bo3 else spec.seed
        self.game = Game(**spec.game_args(self.game_no, self.start), seed=seed, **kw)
        self.trajs = (Trajectory(), Trajectory())
        self.seats = (_Seat(), _Seat())
        decks = matchup_decks(spec.matchup)[::-1] if spec.swap_seats else matchup_decks(spec.matchup)
        self.agents = tuple(RandomAgent(seed=seed * 2 + s) if pol == RANDOM else make_bot(s, decks[s]) if pol == BOT else None for s, pol in enumerate(spec.physical_seats))

    @property
    def seed(self) -> int:
        return game_seed(self.spec.seed, self.game_no) if self.spec.bo3 else self.spec.seed

    def next_game(self, Game, kw: dict) -> bool:
        """After a finished game: fold each player's witnessed record into its
        knowledge and start the match's next game. False when the match is over
        (or the spec is a single game)."""
        g, spec = self.game, self.spec
        winner = None if g.winner is None else spec.canonical(g.winner)
        self.results.append((self.start, winner, g.end_reason))
        if not spec.bo3:
            return False
        wins = [sum(1 for _, w, _ in self.results if w == c) for c in (0, 1)]
        if max(wins) >= 2 or len(self.results) >= 3:
            return False
        for p in (0, 1):
            c = spec.canonical(p)
            self.knowledge[c] = self.knowledge[c].with_game(g.witnessed(p))
        self.game_no += 1
        self.start = next_starting_player(self.start, winner)
        self._new_game(Game, kw)
        return True

    @property
    def match_winner(self) -> int | None:
        wins = [sum(1 for _, w, _ in self.results if w == c) for c in (0, 1)]
        return None if wins[0] == wins[1] else int(wins[1] > wins[0])


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
                xs = [it[3] for it in run]
                if len(xs[0]) > 4:  # set 8: belief evidence
                    from .model import step_batch

                    batch, width = step_batch(xs)
                else:
                    batch, width = _batch(torch, xs)
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
    if job.record and job.shaping and checkpoint_config(job.learner_path).get("value_bound") == "tanh":
        raise ValueError("active shaping requires an unbounded value head")
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

    def features(pol: str) -> int:
        if pol in job.features:
            return check_features(job.features[pol])
        return policy_features(job.learner_path, job.learner_version) if pol == LEARNER else policy_features(pol)

    def belief(pol: str):
        if features(pol) < BELIEF_FEATURES:
            return None
        return policy_belief(job.learner_path, job.learner_version) if pol == LEARNER else policy_belief(pol)

    game_kw = {"max_turns": job.max_turns, "auto_mana": job.auto_mana, "auto_pass": job.auto_pass}

    def start(i: int) -> _Live:
        return _Live(job.games[i], free.pop(), Game, game_kw)

    def finish(lv: _Live) -> bool:
        """Record a finished game; True if its match goes on in `lv`."""
        g, spec = lv.game, lv.spec
        winner = 1 - g.winner if spec.swap_seats and g.winner is not None else g.winner
        out.games.append((spec.seats, winner, g.end_reason, g.turn, len(g.actions), lv.seed))
        if job.record:
            for p in (0, 1):
                if spec.physical_seats[p] == LEARNER:
                    outcome = 0.0 if g.winner is None else (1.0 if g.winner == p else -1.0)
                    _finish(lv.trajs[p], outcome, job, out)
                    if lv.trajs[p].actions:
                        out.trajectory_ids.append((lv.seed, p))
                        learner_belief = belief(LEARNER)
                        if learner_belief is not None:
                            out.belief_targets.append(learner_belief.target(*registered_lists(spec.matchup, spec.variants, 1 - spec.canonical(p))))
        if lv.next_game(Game, game_kw):
            return True
        if spec.bo3:
            out.matches.append((spec.seats, spec.seed, spec.matchup, tuple(lv.results), lv.match_winner))
        free.append(lv.slot)
        return False

    def prepare(k: int) -> list:
        """Play scripted moves until each game of group k needs a policy (top
        the group up with newly claimed games as others end); featurize
        those decisions. Items come back sorted by policy."""
        items, live, todo = [], [], groups[k]
        while True:
            for lv in todo:
                seats = lv.spec.physical_seats
                while True:
                    g = lv.game
                    while not g.over:
                        p = g.decision.player
                        pol = seats[p]
                        if pol not in SCRIPTED:
                            break
                        _step(g, lv.seats, lv.agents[p].act(g))
                    if not g.over or not finish(lv):
                        break
                if g.over:
                    continue
                live.append(lv)
                seat = lv.seats[p]
                state, o_len, o_flat = featurize_flat(g, p, features=features(pol))
                # int32 arrays once: batching and recording then only copy memory
                x = (array("i", state), array("i", o_len), array("i", o_flat), array("i", encode_event_hashes(seat.events)))
                spec_b = belief(pol)
                if spec_b is not None:  # set 8: the evidence the belief head reads, nothing else of the opponent
                    x += (spec_b.evidence(lv.knowledge[lv.spec.canonical(p)], g.witnessed(p)),)
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


def residency_waves(specs: list[GameSpec], capacity: int):
    """Schedule already sampled games; capacity includes one learner slot.

    Every game stays whole, and each wave finishes before slots are reused.
    Sorting groups minimizes reloads without trimming the opponent pool.
    """
    if capacity < 1:
        raise ValueError("resident capacity must be positive")
    wave, policies = [], set()
    for spec in sorted(specs, key=_matchup):
        need = set(_matchup(spec))
        if len(need) + 1 > capacity:
            raise ValueError("one game's policies exceed resident capacity (including learner)")
        if wave and len(policies | need) + 1 > capacity:
            yield wave
            wave, policies = [], set()
        policies |= need
        wave.append(spec)
    if wave:
        yield wave


def run_specs(pool, specs: list[GameSpec], job: Job, workers: int, inflight: int = 64) -> Result:
    server = getattr(pool, "server", None)
    if server is None or not server.cfg.resident_limit:
        return _run_specs(pool, specs, job, workers, inflight)
    merged = Result()
    selection_s, waves = 0.0, 0
    for wave in residency_waves(specs, server.cfg.resident_limit):
        keys = {(job.learner_path, job.learner_version)} | {(p, 0) for spec in wave for p in _matchup(spec)}
        t = time.monotonic()
        server.select_policies(sorted(keys))
        selection_s += time.monotonic() - t
        waves += 1
        res = _run_specs(pool, wave, job, workers, inflight)
        for f in fields(Result):
            getattr(merged, f.name).extend(getattr(res, f.name))
    t_merge = time.monotonic()
    if merged.trajectory_ids:
        order = sorted(range(len(merged.lengths)), key=lambda i: merged.trajectory_ids[i])
        starts = list(itertools.accumulate(merged.lengths, initial=0))
        ranges = [(starts[i], starts[i + 1]) for i in order]
        merged.samples = merged.samples.take_ranges(ranges)
        for name in ("actions", "logps", "advantages", "returns", "kinds"):
            src = getattr(merged, name)
            setattr(merged, name, [v for lo, hi in ranges for v in src[lo:hi]])
        merged.lengths = [merged.lengths[i] for i in order]
        if merged.belief_targets:
            merged.belief_targets = [merged.belief_targets[i] for i in order]
        merged.trajectory_ids = [merged.trajectory_ids[i] for i in order]
    merged.games.sort(key=lambda g: g[-1])
    merged.residency = {"waves": waves, "selection_s": selection_s, "merge_s": time.monotonic() - t_merge}
    return merged


def _run_specs(pool, specs: list[GameSpec], job: Job, workers: int, inflight: int = 64) -> Result:
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
