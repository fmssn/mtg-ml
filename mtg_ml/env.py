"""Minimal two-player environment around `Game` (no gymnasium dependency).

Each `step` is one decision by `current_player`. Rewards are terminal only:
+1 / -1 from each player's perspective, 0 for a draw or the turn limit.
"""

from __future__ import annotations

from .encode import action_keys, encode_state
from .backend import game_class
from .engine import JUND_WILDFIRE, MONO_BLUE_TERROR, expand
from .engine.view import observe


class MTGEnv:
    def __init__(self, decks=None, max_turns: int | None = None, auto_single: bool = True, engine: str | None = None):
        self.decks = decks or (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))
        self.max_turns = max_turns
        self.auto_single = auto_single
        self.game_cls = game_class(engine)
        self.game = None

    def reset(self, seed: int = 0, starting_player: int | None = None) -> dict:
        self.game = self.game_cls(self.decks, seed=seed, starting_player=starting_player, max_turns=self.max_turns, auto_single=self.auto_single)
        return self._obs()

    @property
    def current_player(self) -> int | None:
        d = self.game.decision
        return None if d is None else d.player

    def legal_actions(self) -> list[int]:
        return list(range(len(self.game.legal_options())))

    def step(self, action: int):
        self.game.step(action)
        done = self.game.over
        rewards = (0.0, 0.0)
        if done and self.game.winner is not None:
            rewards = (1.0, -1.0) if self.game.winner == 0 else (-1.0, 1.0)
        return self._obs(), rewards, done, {"reason": self.game.end_reason}

    def _obs(self) -> dict:
        g = self.game
        p = self.current_player
        if p is None:
            return {"player": None, "done": True}
        return {
            "player": p,
            "view": observe(g, p),
            "features": encode_state(g, p),
            "action_keys": action_keys(g),
            "decision_kind": g.decision.kind,
        }
