"""A trained policy as a plain `act(game)` agent, for watching and recording games.

    from mtg_ml.rl.agent import ModelAgent
    agent = ModelAgent("runs/x/latest.pt", seat=0)
    agent = ModelAgent("runs/x/latest.pt", seat=0, search=SearchConfig(budget=64))  # own-turn search

Mirrors what rollouts do for the learner (`rollout._LocalEvaluator`): the
recurrent state persists across the agent's decisions, and the events it saw
since its previous decision (its own and the opponent's public actions) are
fed in with the next one. For that the game loop must call `observe(game, a)`
before every `game.step(a)`, whoever decides. After `act`, `last_info` holds
the policy (one probability per option) and the value estimate; with a
search, also `search`: its policy, the value of the chosen action and the
principal variation (`rl/search.py`).
"""

from __future__ import annotations

import random

from .features import encode_event_hashes, event_hashes, featurize_flat
from .rollout import load_policy
from .samples import PackedSamples
from .search import NetEvaluator, SearchConfig, eligible, search


class ModelAgent:
    def __init__(self, path: str, seat: int, sample: bool = True, seed: int = 0, search: SearchConfig | None = None):
        import torch

        self.torch = torch
        self.net = load_policy(path)
        self.seat = seat
        self.sample = sample
        self.gen = torch.Generator().manual_seed(seed * 2 + seat)
        self.hidden = None
        self.events: list[int] = []
        self.last_info: dict | None = None
        self.search = search
        self.rng = random.Random(seed * 2 + seat)
        self.evaluator = NetEvaluator(self.net) if search is not None else None

    def observe(self, game, index: int) -> None:
        mine, theirs = event_hashes(game, index)
        self.events += mine if game.decision.player == self.seat else theirs

    def act(self, game) -> int:
        from .model import collate

        torch = self.torch
        state, o_len, o_flat = featurize_flat(game, self.seat)
        ps = PackedSamples()
        ps.extend([(state, o_len, o_flat, encode_event_hashes(self.events))])
        self.events = []
        hidden = None
        if self.net.memory != "none":
            h = self.net.initial_state(1)[0] if self.hidden is None else self.hidden
            hidden = torch.stack([h])
        with torch.no_grad():
            logits, values, hn = self.net(collate(ps), hidden)
        if hn is not None:
            self.hidden = hn[0]
        logits = logits[0, : len(o_len)]
        probs = torch.softmax(logits, dim=-1)
        a = int(torch.multinomial(probs, 1, generator=self.gen)) if self.sample else int(probs.argmax())
        self.last_info = {"policy": [round(float(p), 4) for p in probs], "value": round(float(values[0]), 4)}
        if self.search is not None and eligible(game, self.seat, self.search):
            found = search(game, self.seat, self.evaluator, self.search, self.rng, root_logits=logits.tolist(), root_value=float(values[0]), root_hn=self.hidden)
            a = found.action
            self.last_info["search"] = {"policy": [round(p, 4) for p in found.policy], "value": round(found.value, 4), "line": found.line, "evals": found.evals}
        return a
