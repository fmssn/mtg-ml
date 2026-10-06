"""Best-of-three matches with sideboarding.

Game 1 uses the maindecks and a random starting player. Before games 2 and
3 both decks apply their sideboard plan (`decks.SIDEBOARD_PLANS`), and the
loser of the previous game chooses to play first (always "play", the right
call in this tempo matchup); after a drawn game the previous starting
player starts again. A match ends when a player has two game wins or after
three games.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .backend import game_class
from .engine import DECKS, expand, postboard

DECK_NAMES = ("jund_wildfire", "mono_blue_terror")


def match_decks(game_no: int) -> tuple[list[str], list[str]]:
    if game_no == 1:
        return tuple(expand(DECKS[d]) for d in DECK_NAMES)
    return tuple(expand(postboard(d)) for d in DECK_NAMES)


def game_seed(match_seed: int, game_no: int) -> int:
    return match_seed * 4 + game_no


def first_starting_player(match_seed: int) -> int:
    return random.Random(match_seed).randrange(2)


def next_starting_player(prev_start: int, prev_winner: int | None) -> int:
    return prev_start if prev_winner is None else 1 - prev_winner


@dataclass
class MatchResult:
    games: list[tuple[int, int | None, str]] = field(default_factory=list)  # (starting player, winner, reason)

    @property
    def wins(self) -> list[int]:
        return [sum(1 for _, w, _ in self.games if w == p) for p in (0, 1)]

    @property
    def over(self) -> bool:
        return max(self.wins) >= 2 or len(self.games) >= 3

    @property
    def winner(self) -> int | None:
        a, b = self.wins
        return None if a == b else (0 if a > b else 1)

    def next_game(self, match_seed: int) -> tuple[int, int]:
        """(game number, starting player) of the next game."""
        if not self.games:
            return 1, first_starting_player(match_seed)
        start, winner, _ = self.games[-1]
        return len(self.games) + 1, next_starting_player(start, winner)


def play_match(agents, seed: int = 0, engine: str | None = None, **game_kw) -> MatchResult:
    """agents: [seat0, seat1] objects with act(game), reused across games."""
    from .agents import take  # agents imports the engine; keep this module light

    Game = game_class(engine)
    res = MatchResult()
    while not res.over:
        n, start = res.next_game(seed)
        g = Game(match_decks(n), seed=game_seed(seed, n), starting_player=start, match_game=n, **game_kw)
        while not g.over:
            take(g, agents, agents[g.decision.player].act(g))
        res.games.append((start, g.winner, g.end_reason))
    return res
