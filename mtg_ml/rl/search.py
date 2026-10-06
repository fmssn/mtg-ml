"""Own-turn search: a policy improvement operator for deep combos.

Why: PPO only improves actions it samples. A multi-step line whose last
step looks bad on its own (Krark-Clan Shaman + Toxin Analysis: give the
Shaman deathtouch, then sacrifice an artifact to kill every creature) has
its last action pushed to ~1e-7 before it was ever tried with the setup in
place, and no amount of further self-play recovers it (docs/search.md).
Search tries the line explicitly; the rollout records the search's policy
as a distillation target, so inference stays one forward pass.

What is searched: the searcher's own decisions on its own turn of the
branching kinds (`SearchConfig.kinds`: priority with an empty stack,
targets, sacrifices, X, modes, attackers). Everything else inside the tree
is played by the same network with argmax: the searcher's bookkeeping
(mana payment, holding priority over its own spell, ...) and all of the
opponent's decisions, on one determinization of the hidden cards
(`view.determinize`). A path ends when the turn passes to the opponent,
the game ends or `max_depth` branching decisions were made; the leaf value
is the value head from the searcher's seat (+-1 when the game is over).

Algorithm: Gumbel AlphaZero (Danihelka et al. 2022): Sequential Halving
over the Gumbel top-m root actions, and below the root the deterministic
"improved policy minus visit share" rule with completed Q values. Changes
for this tree: (1) a node's unexpanded children are expanded before the
selection rule applies, activated abilities first, so an action the policy
has learned to ignore is still looked at once its parent has a handful of
visits (otherwise the search inherits the blind spot it is meant to cure);
(2) the backup is a max over children (`backup="max"`): every node is the
searcher's own decision in a deterministic tree, so a node is worth its
best line, not the average over the lines tried (with "mean" the seed-6
combo is found but valued below the policy's line); (3) a leaf child is
never revisited, since its value is already exact; (4) the tree branches
only within the root's phase and not on mana abilities, and paths end at
the searcher's next decision after the turn, where the value head is in
distribution. Priors are mixed with `prior_floor` uniform mass.

Cost: every node and every auto-played decision is one forward pass of the
network, so a search of budget B costs about B * (1 + auto-played decisions
per node) passes plus as many engine steps; forks replay the action history
(~1 ms on the native engine at turn 13).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..engine import objects as O
from ..engine.view import determinize
from .features import encode_event_hashes, event_hashes, featurize_flat

# Sacrifice costs are left to the policy (it picks the Wellspring at 0.998): then an
# "activate, pay by sacrificing" line is one node whose child is the resolved board,
# instead of a sacrifice node the critic cannot judge plus one level more to resolve.
BRANCH_KINDS = frozenset({O.PRIORITY, O.TARGET, O.CHOOSE_X, O.CHOOSE_MODE, O.DECLARE_ATTACKER})
MAIN_PHASES = ("main1", "main2")


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 64  # node expansions per search (32 misses the Shaman sweep on the seed-6 position, docs/search.md)
    max_root: int = 8  # Gumbel top-m root actions considered
    max_depth: int = 8  # branching decisions along one path
    prior_floor: float = 0.02  # uniform mass mixed into the policy priors
    c_visit: float = 50.0  # Gumbel AlphaZero sigma(q) = (c_visit + max N) * c_scale * q, inside the tree
    c_scale: float = 1.0
    # The distillation target uses its own, softer scale. At 1.0 a 0.05 difference
    # in leaf value, the critic's noise level here, is 3+ logits: the targets were
    # one-hot on whichever leaf the critic overrated, and distilling them collapsed
    # the policy within three updates (win rate vs the pool 0.56 -> 0.31). At 0.25
    # that noise stays under a logit while the Shaman sweep's +0.5 is still 6+.
    # The tree keeps the full scale: with 0.25 the visits follow the prior and
    # never resolve the sweep (the seed-6 probe misses it at budget 64).
    target_scale: float = 0.25
    # The search intervenes (its action is played, its policy recorded as a target)
    # only when its choice differs from the policy's argmax and beats that action's
    # value by this margin. Across the seed-6 game the search disagreed with the
    # policy at half of the eligible decisions, mostly within +-0.1 (the critic's
    # noise plus the max backup's optimism), while the real finds stood out: the
    # Shaman lines at +0.17, +0.40 and +1.19. Distilling the noisy half dropped the
    # benchmark from 0.50 to 0.29 in five iterations.
    margin: float = 0.15
    backup: str = "max"  # "max" (deterministic own-turn tree) or "mean"
    kinds: frozenset = BRANCH_KINDS
    steps: tuple = MAIN_PHASES  # root decisions are searched only in these steps
    min_options: int = 2  # non-pass options a root decision needs
    determinize: bool = True
    max_evals: int = 0  # cap on network evaluations per search (0 = 16 * budget)
    # "turn": a path ends at the searcher's first decision after the turn passed
    # to the opponent; "opponent": at its first decision of its next turn
    horizon: str = "turn"


@dataclass
class SearchResult:
    action: int
    policy: list[float]  # improved policy over the root's legal options (the distillation target)
    value: float  # value of the chosen root action
    root_value: float  # value head at the root, for comparison
    reference: float = 0.0  # value of the policy's own argmax action (same horizon as `value`)
    improved: bool = False  # the search's action differs from the policy's argmax and beats it by `margin`
    q: list = field(default_factory=list)  # searched value per root action (None where never visited)
    line: list[str] = field(default_factory=list)  # principal variation (option labels)
    expansions: int = 0
    evals: int = 0


def eligible(game, player: int, cfg: SearchConfig) -> bool:
    """Whether the current decision is one the search handles at the root."""
    d = game.decision
    if d is None or d.player != player or game.active != player or game.over:
        return False
    if d.kind not in cfg.kinds or game.step_name not in cfg.steps:
        return False
    if d.kind == O.PRIORITY and game.stack:
        return False
    return sum(1 for o in d.options if o.key[0] != "pass") >= cfg.min_options


class NetEvaluator:
    """Evaluates one decision at a time with a PolicyNet on the CPU:
    (logits over the legal options, value, new recurrent state)."""

    def __init__(self, net):
        import torch

        self.torch = torch
        self.net = net

    def __call__(self, game, player: int, hidden, events: list[int]):
        from .model import collate
        from .samples import PackedSamples

        torch = self.torch
        state, o_len, o_flat = featurize_flat(game, player)
        ps = PackedSamples()
        ps.extend([(state, o_len, o_flat, encode_event_hashes(events))])
        h = None
        if self.net.memory != "none":
            h = self.net.initial_state(1) if hidden is None else hidden[None]
        with torch.inference_mode():
            logits, values, hn = self.net(collate(ps), h)
        return logits[0, : len(o_len)].tolist(), float(values[0]), None if hn is None else hn[0]


class _Seat:
    """A player's recurrent state and the events since their last evaluation, inside the tree."""

    __slots__ = ("hidden", "events")

    def __init__(self, hidden=None, events=()):
        self.hidden = hidden
        self.events = list(events)

    def copy(self) -> "_Seat":
        return _Seat(self.hidden, self.events)


class _Node:
    __slots__ = ("game", "seats", "depth", "logits", "value", "leaf", "n", "q", "children", "labels", "keys", "allowed")

    def __init__(self, game, seats, depth: int):
        self.game = game
        self.seats = seats
        self.depth = depth
        self.logits: list[float] = []
        self.value = 0.0
        self.leaf = True
        self.n: list[int] = []
        self.q: list[float] = []  # backed-up value per edge (max or mean over the subtree)
        self.children: dict[int, _Node] = {}
        self.labels: list[str] = []
        self.keys: list[tuple] = []
        self.allowed: list[int] = []  # actions the tree branches on (mana abilities are left to the policy)


class _Search:
    def __init__(self, player: int, evaluator, cfg: SearchConfig, rng: random.Random):
        self.player, self.ev, self.cfg, self.rng = player, evaluator, cfg, rng
        self.expansions = 0
        self.evals = 0
        self.max_evals = cfg.max_evals or 16 * cfg.budget
        self.turn = self.step = None  # the root's turn and step: the tree branches within this phase only

    # -- engine side ---------------------------------------------------------

    def _evaluate(self, game, player: int, seat: _Seat):
        self.evals += 1
        logits, value, hn = self.ev(game, player, seat.hidden, seat.events)
        seat.hidden, seat.events = hn, []
        return logits, value

    def _step(self, game, seats, a: int) -> None:
        p = game.decision.player
        mine, theirs = event_hashes(game, a)
        seats[p].events += mine
        seats[1 - p].events += theirs
        game.step(a)

    def _branching(self, game, depth: int) -> bool:
        """A searcher decision of a branching kind in the root's phase. Once
        the searcher moves on (passes to combat, to main 2, ...) the rest of
        the horizon is the policy's: searching main 2 from main 1 would
        duplicate the whole tree one phase later."""
        d = game.decision
        if d.player != self.player or game.turn != self.turn or game.step_name != self.step or d.kind not in self.cfg.kinds:
            return False
        if d.kind == O.PRIORITY and game.stack:
            return False  # holding priority over our own spell: let the policy decide
        return depth < self.cfg.max_depth

    def _horizon(self, game) -> bool:
        """Whether a path ends here: at the searcher's first own decision after
        the root turn ("turn", typically priority in the opponent's upkeep) or
        after the opponent's turn ("opponent"). Leaves are always the
        searcher's decisions: the value head is trained on nothing else."""
        if game.decision.player != self.player:
            return False
        if self.cfg.horizon == "opponent":
            return game.active == self.player and game.turn > self.turn
        return game.active != self.player

    def _open(self, node: _Node, game) -> None:
        """Make `node` a branching node at the current decision of `game`."""
        node.leaf = False
        node.logits, node.value = self._evaluate(game, self.player, node.seats[self.player])
        opts = game.decision.options
        node.labels = [o.label for o in opts]
        node.keys = [o.key for o in opts]
        node.allowed = [a for a, k in enumerate(node.keys) if k[0] != "mana"] or list(range(len(opts)))
        n = len(opts)
        node.n, node.q = [0] * n, [node.value] * n

    def _advance(self, node: _Node) -> None:
        """Auto-play `node.game` until a branching decision of the searcher,
        the end of the searcher's turn, the game's end or the eval cap; then
        evaluate the node (logits for a branching node, value for a leaf)."""
        g, seats, me = node.game, node.seats, self.player
        while not g.over and not self._horizon(g):
            if self._branching(g, node.depth):
                self._open(node, g)
                return
            if self.evals >= self.max_evals:
                break
            p = g.decision.player
            logits, _ = self._evaluate(g, p, seats[p])
            self._step(g, seats, max(range(len(logits)), key=logits.__getitem__))
        node.leaf = True
        if g.over:
            node.value = 0.0 if g.winner is None else (1.0 if g.winner == me else -1.0)
        else:
            _, node.value = self._evaluate(g, me, seats[me])

    def _expand(self, parent: _Node, a: int) -> _Node:
        self.expansions += 1
        g = parent.game.fork()
        seats = (parent.seats[0].copy(), parent.seats[1].copy())
        child = _Node(g, seats, parent.depth + 1)
        self._step(g, seats, a)
        self._advance(child)
        return child

    # -- Gumbel AlphaZero ----------------------------------------------------

    def _priors(self, logits: list[float]) -> list[float]:
        """Log of the policy mixed with `prior_floor` uniform mass."""
        m = max(logits)
        z = sum(math.exp(x - m) for x in logits)
        n, eps = len(logits), self.cfg.prior_floor
        return [math.log((1 - eps) * math.exp(x - m) / z + eps / n) for x in logits]

    def _sigma(self, node: _Node, q: float, scale: float | None = None) -> float:
        return (self.cfg.c_visit + max(node.n, default=0)) * (self.cfg.c_scale if scale is None else scale) * q

    def _completed_q(self, node: _Node) -> list[float]:
        """Q per edge: the backed-up value where visited, else the node's own
        value head estimate (what the policy would get without search)."""
        return [node.q[a] if node.n[a] else node.value for a in range(len(node.n))]

    def _improved(self, node: _Node, scale: float | None = None) -> list[float]:
        """pi' = softmax(prior + sigma(completed Q))."""
        pri = self._priors(node.logits)
        s = [p + self._sigma(node, q, scale) for p, q in zip(pri, self._completed_q(node))]
        m = max(s)
        e = [math.exp(x - m) for x in s]
        z = sum(e)
        return [x / z for x in e]

    def _select(self, node: _Node) -> int:
        """Below the root: unexpanded children first (activated abilities,
        whose worth depends on the setup the line just made, then by prior),
        then the action maximizing pi'(a) - N(a) / (1 + sum N)."""
        unexpanded = [a for a in node.allowed if a not in node.children]
        if unexpanded:
            return max(unexpanded, key=lambda a: (node.keys[a][0] == "activate", node.logits[a]))
        pi = self._improved(node)
        total = sum(node.n)
        # A leaf child's value is known exactly (the tree is deterministic), so
        # revisiting it learns nothing: descend into the lines still open.
        open_ = [a for a in node.allowed if not node.children[a].leaf] or node.allowed
        return max(open_, key=lambda a: pi[a] - node.n[a] / (1 + total))

    def _simulate(self, root: _Node, a: int) -> None:
        path = []
        node, action = root, a
        while True:
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                child = node.children[action] = self._expand(node, action)
                value = child.value
                break
            node = child
            if node.leaf:
                value = node.value
                break
            action = self._select(node)
        for node, action in reversed(path):
            node.n[action] += 1
            if self.cfg.backup == "max":
                node.q[action] = value if node.n[action] == 1 else max(node.q[action], value)
            else:
                node.q[action] += (value - node.q[action]) / node.n[action]
            # the value of taking `action` at `node` is the best (or mean) line below it
            value = node.q[action] if self.cfg.backup == "max" else value

    def run(self, root: _Node) -> SearchResult:
        cfg, n = self.cfg, len(root.logits)
        pri = self._priors(root.logits)
        gumbel = [-math.log(-math.log(self.rng.random() or 1e-12)) for _ in range(n)]
        considered = sorted(root.allowed, key=lambda a: gumbel[a] + pri[a], reverse=True)[: max(1, min(cfg.max_root, len(root.allowed)))]
        phases = max(1, math.ceil(math.log2(len(considered)))) if len(considered) > 1 else 1
        per_phase = max(1, cfg.budget // phases)
        while True:
            visits = max(1, per_phase // len(considered))
            for a in considered:
                for _ in range(visits):
                    if self.expansions >= cfg.budget and root.n[a]:
                        break
                    self._simulate(root, a)
            if len(considered) == 1:
                break
            score = lambda a: gumbel[a] + pri[a] + self._sigma(root, root.q[a])  # noqa: E731
            considered = sorted(considered, key=score, reverse=True)[: max(1, len(considered) // 2)]
        best = considered[0]
        pol = max(range(n), key=root.logits.__getitem__)
        reference = root.q[pol] if root.n[pol] else root.value
        improved = best != pol and root.q[best] - reference >= cfg.margin
        line = []
        node, a = root, best
        while True:
            line.append(node.labels[a])
            child = node.children.get(a)
            if child is None or child.leaf or not child.n or not sum(child.n):
                break
            node = child
            a = max(range(len(node.n)), key=lambda i: (node.n[i], node.q[i]))
        q = [root.q[a] if root.n[a] else None for a in range(n)]
        return SearchResult(best, self._improved(root, cfg.target_scale), root.q[best], root.value, reference, improved, q, line, self.expansions, self.evals)


def search(game, player: int, evaluator, cfg: SearchConfig, rng: random.Random, root_hidden=None, root_events=(), root_logits=None, root_value=None, root_hn=None, opp_hidden=None, opp_events=()) -> SearchResult:
    """Search the current decision of `game` for `player`. `root_hidden` and
    `root_events` are the player's recurrent state and the events since their
    last decision; if the root was already evaluated, pass `root_logits`,
    `root_value` and the new state `root_hn` instead. `opp_hidden` and
    `opp_events` are the same for the opponent, when known (self-play): the
    network plays the opponent inside the tree, and a GRU started from zero in
    the middle of a game is a different, worse player. The game is not
    modified (the tree works on forks, determinized unless disabled)."""
    s = _Search(player, evaluator, cfg, rng)
    s.turn, s.step = game.turn, game.step_name
    base = determinize(game, player, rng) if cfg.determinize else game.fork()
    seats = tuple(_Seat(root_hidden, root_events) if p == player else _Seat(opp_hidden, opp_events) for p in (0, 1))
    root = _Node(base, seats, 0)
    if root_logits is None:
        s._open(root, base)
    else:
        root.leaf = False
        opts = base.decision.options
        root.logits, root.value = list(root_logits), float(root_value)
        seats[player].hidden, seats[player].events = root_hn, []
        root.labels, root.keys = [o.label for o in opts], [o.key for o in opts]
        root.allowed = [a for a, k in enumerate(root.keys) if k[0] != "mana"] or list(range(len(opts)))
        root.n, root.q = [0] * len(opts), [root.value] * len(opts)
    return s.run(root)
