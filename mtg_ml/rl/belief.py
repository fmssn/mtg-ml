"""Inputs and targets of the belief head (feature set 8), without torch.

The belief head reads only witnessed opponent-card evidence: the current
game's `Game.witnessed(viewer)` and the earlier games of the match
(`knowledge.MatchKnowledge`). It never receives the opponent's registered
list, archetype, seed or sideboard choice; those are training targets only
and travel in `Result` beside the recorded samples, never into inference.

Cards are indexed exactly (no hashing) by a vocabulary saved in the
checkpoint (`BeliefSpec`), so an old checkpoint keeps its card order when
cards are added later: evidence for cards it does not know is dropped.

Evidence is a flat int32 array of (card, slot, copies) triples:
    slot 0   the current game,
    slot g   finished game g of this match (1 or 2),
so historical cards are never confused with cards in play now. Copies are
capped at `MAX_BASIC` (basic lands) or `MAX_NONBASIC`.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..engine.sideboard import BASIC_LANDS
from ..knowledge import MatchKnowledge

SCHEMA = 1
MAX_NONBASIC = 4
MAX_BASIC = 75
SLOTS = 3  # current game, finished game 1, finished game 2


def default_vocab() -> tuple[str, ...]:
    """Every registered non-token card, sorted."""
    from ..engine.cards import CARDS

    return tuple(sorted(CARDS))


def default_archetypes() -> tuple[str, ...]:
    from ..engine.decks import DECKS

    return tuple(DECKS)


@dataclass(frozen=True)
class BeliefSpec:
    """The belief head's vocabulary and archetype order (checkpoint config `belief`)."""

    vocab: tuple[str, ...]
    archetypes: tuple[str, ...]
    schema: int = SCHEMA

    @classmethod
    def default(cls) -> "BeliefSpec":
        return cls(default_vocab(), default_archetypes())

    @classmethod
    def from_config(cls, cfg: Mapping) -> "BeliefSpec":
        if cfg.get("schema") != SCHEMA:
            raise ValueError(f"belief schema {cfg.get('schema')!r}, expected {SCHEMA}")
        return cls(tuple(cfg["vocab"]), tuple(cfg["archetypes"]), SCHEMA)

    def to_config(self) -> dict:
        return {"schema": self.schema, "vocab": list(self.vocab), "archetypes": list(self.archetypes)}

    @property
    def index(self) -> dict[str, int]:
        idx = self.__dict__.get("_index")
        if idx is None:
            idx = {n: i for i, n in enumerate(self.vocab)}
            object.__setattr__(self, "_index", idx)
        return idx

    @property
    def basic(self) -> tuple[bool, ...]:
        """Per vocabulary card: a basic land (counts 0..MAX_BASIC) or not (0..MAX_NONBASIC)."""
        return tuple(n in BASIC_LANDS for n in self.vocab)

    def cap(self, name: str) -> int:
        return MAX_BASIC if name in BASIC_LANDS else MAX_NONBASIC

    def evidence(self, knowledge: MatchKnowledge, current: Mapping[str, int] | None) -> array:
        """The flat (card, slot, copies) triples of one decision."""
        out = array("i")
        idx = self.index
        parts = [(0, current or {})] + [(g + 1, dict(w)) for g, w in enumerate(knowledge.games)]
        for slot, seen in parts:
            for name in sorted(seen):
                i = idx.get(name)
                n = min(int(seen[name]), self.cap(name))
                if i is not None and n > 0:
                    out.extend((i, slot, n))
        return out

    def target(self, archetype: str, registered: Mapping[str, int]) -> tuple[int, array]:
        """(archetype index, per-vocabulary-card copies in the registered 75):
        what the belief head is trained to predict. Training data only."""
        counts = array("i", [0]) * len(self.vocab)
        idx = self.index
        for name, n in registered.items():
            if n > self.cap(name):
                raise ValueError(f"{name!r}: {n} copies in a registered 75 exceed the belief head's range")
            i = idx.get(name)
            if i is None:
                raise ValueError(f"{name!r} is not in the belief vocabulary")
            counts[i] += int(n)
        return self.archetypes.index(archetype), counts


def registered_75(main: Sequence[str], sideboard: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in list(main) + list(sideboard):
        out[name] = out.get(name, 0) + 1
    return out
