"""Search on top of a scripted bot: determinized flat Monte Carlo.

At a decision that matters, the bot keeps its base bot's top few options,
and for each one plays `playouts` games to the end from the current
position: hidden cards (opponent's hand, both libraries) are re-sampled
consistently with what this player knows (`view.determinize`), the option
is taken, and both sides continue with the scripted bots. The option with
the best average result wins (draw = half). All options share the same
re-sampled worlds (common random numbers), so the comparison is fair.

Slow (tens of milliseconds per playout), meant as a stronger fixed opponent
for evaluation and as a probe of how much better the scripted plans could
play, not for training-scale self-play.
"""

from __future__ import annotations

import random

from ..engine.game import Game
from ..engine.view import determinize
from . import make_bot
from .base import Bot

# Decisions worth searching; the rest (mana payment, trigger order...) use the base bot.
SEARCHED = {"priority", "target", "declare_attacker", "declare_blocker", "sacrifice", "choose_card", "choose_x", "mulligan"}


class SearchBot:
    def __init__(self, seat: int, playouts: int = 8, max_options: int = 4, seed: int = 0, max_decisions: int = 4000):
        self.p = seat
        self.base: Bot = make_bot(seat)
        self.playouts = playouts
        self.max_options = max_options
        self.rng = random.Random(seed)
        self.max_decisions = max_decisions
        self.name = f"search({self.base.name})"

    def act(self, g: Game) -> int:
        d = g.decision
        if len(d.options) == 1 or d.kind not in SEARCHED:
            return self.base.act(g)
        handler = getattr(self.base, f"score_{d.kind}")
        scores = [handler(g, d, o) for o in d.options]
        ranked = sorted(range(len(d.options)), key=lambda i: -scores[i])[: self.max_options]
        if len(ranked) == 1:
            return ranked[0]
        seeds = [self.rng.randrange(1 << 30) for _ in range(self.playouts)]
        best, best_v = ranked[0], -1.0
        for i in ranked:  # ties keep the base bot's preference (ranked order)
            v = sum(self._playout(g, i, s) for s in seeds) / len(seeds)
            if v > best_v:
                best, best_v = i, v
        return best

    def _playout(self, g: Game, option: int, seed: int) -> float:
        h = determinize(g, self.p, random.Random(seed))
        h.rng = random.Random(seed ^ 0x5EED)  # future shuffles must not mirror the real game's
        h.step(option)
        bots = (make_bot(0), make_bot(1))
        n = 0
        while not h.over and n < self.max_decisions:
            h.step(bots[h.decision.player].act(h))
            n += 1
        if h.winner is None:
            return 0.5
        return 1.0 if h.winner == self.p else 0.0
