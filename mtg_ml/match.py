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

# The deck each seat plays unless a matchup says otherwise. Seat 0 is always
# Jund Wildfire; the network's `seat:` feature has always meant this pairing.
DECK_NAMES = ("jund_wildfire", "mono_blue_terror")

# Matchups by name: the deck in seat 0 and seat 1.
MATCHUPS = {
    "jund_blue": ("jund_wildfire", "mono_blue_terror"),
    "jund_madness": ("jund_wildfire", "red_madness"),
}
DEFAULT_MATCHUP = "jund_blue"


def matchup_decks(matchup: str = DEFAULT_MATCHUP) -> tuple[str, str]:
    if matchup not in MATCHUPS:
        raise ValueError(f"unknown matchup {matchup!r}: expected one of {sorted(MATCHUPS)}")
    return MATCHUPS[matchup]


def match_decks(game_no: int, matchup: str = DEFAULT_MATCHUP) -> tuple[list[str], list[str]]:
    a, b = matchup_decks(matchup)
    if game_no == 1:
        return expand(DECKS[a]), expand(DECKS[b])
    return expand(postboard(a, b)), expand(postboard(b, a))


def deck_names(matchup: str = DEFAULT_MATCHUP) -> tuple[str | None, str | None]:
    """`Game(deck_names=...)`: a deck's name where it is not its seat's
    usual deck (a `self:deck:` state feature), None where it is, so
    networks trained on the default matchup see exactly the inputs they
    were trained on."""
    return tuple(d if d != DECK_NAMES[s] else None for s, d in enumerate(matchup_decks(matchup)))


def game_args(game_no: int, matchup: str = DEFAULT_MATCHUP) -> dict:
    """Decks and deck names of game `game_no` of a match, as Game keyword arguments."""
    return {"decks": match_decks(game_no, matchup), "match_game": game_no, "deck_names": deck_names(matchup)}


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


def play_match(agents, seed: int = 0, engine: str | None = None, matchup: str = DEFAULT_MATCHUP, **game_kw) -> MatchResult:
    """agents: [seat0, seat1] objects with act(game), reused across games."""
    from .agents import take  # agents imports the engine; keep this module light

    Game = game_class(engine)
    res = MatchResult()
    while not res.over:
        n, start = res.next_game(seed)
        g = Game(**game_args(n, matchup), seed=game_seed(seed, n), starting_player=start, **game_kw)
        while not g.over:
            take(g, agents, agents[g.decision.player].act(g))
        res.games.append((start, g.winner, g.end_reason))
    return res
