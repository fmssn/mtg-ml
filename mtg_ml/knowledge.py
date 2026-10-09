"""What one player has witnessed of the opponent's cards during a match.

A `MatchKnowledge` belongs to one player (seat-relative: "the opponent" is
always the other player) and one best-of-three match. It holds, per finished
game of the match, the opponent's cards that player witnessed and the
number of copies of each that the game legally established: the largest
number of distinct copies visible to that player at one moment
(`Game.witnessed(viewer)`), counting public zones, cards revealed to it
(including transient reveals later shuffled away) and known library
positions. Tokens are excluded and transformed cards are folded to their
registered (front) name.

Copy counts are lower bounds. Across games they combine by maximum, never by
addition: two games that each showed one Lightning Bolt establish one copy,
not two. Game 3 retains games 1 and 2; a new match starts from `EMPTY`.

The record is plain data (`to_dict` / `from_dict`, JSON-safe) so it can move
between rollout workers, inference slots and live-play sessions, and it
holds nothing the player did not see: no registered list, archetype label,
seed or sideboard choice of the opponent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SCHEMA = 1
MAX_GAMES = 3


def _canonical(witnessed: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    out = []
    for name, n in witnessed.items():
        n = int(n)
        if n < 0:
            raise ValueError(f"negative copy count for {name!r}: {n}")
        if n:
            out.append((str(name), n))
    return tuple(sorted(out))


@dataclass(frozen=True)
class MatchKnowledge:
    """games[i]: game i + 1's witnessed opponent cards, sorted (name, copies)."""

    games: tuple[tuple[tuple[str, int], ...], ...] = ()

    def __post_init__(self):
        if len(self.games) > MAX_GAMES:
            raise ValueError(f"a best-of-three has at most {MAX_GAMES} games")

    def with_game(self, witnessed: Mapping[str, int]) -> "MatchKnowledge":
        """This record plus one more finished game (`Game.witnessed(viewer)` at its end)."""
        return MatchKnowledge(self.games + (_canonical(witnessed),))

    def counts(self, current: Mapping[str, int] | None = None) -> dict[str, int]:
        """Established copies per card over the earlier games and, if given,
        the current game's witnessed record: the maximum, never the sum."""
        out: dict[str, int] = {}
        for game in self.games + ((_canonical(current),) if current else ()):
            for name, n in game:
                if n > out.get(name, 0):
                    out[name] = n
        return out

    @property
    def game_no(self) -> int:
        """The number of the game this knowledge is carried into."""
        return len(self.games) + 1

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "games": [dict(g) for g in self.games]}

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "MatchKnowledge":
        if not d:
            return EMPTY
        if d.get("schema") != SCHEMA:
            raise ValueError(f"match knowledge schema {d.get('schema')!r}, expected {SCHEMA}")
        return cls(tuple(_canonical(g) for g in d["games"]))


EMPTY = MatchKnowledge()
