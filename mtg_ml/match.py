"""Best-of-three matches with sideboarding.

Game 1 uses the maindecks and a random starting player. Before games 2 and
3 each seat's `SideboardPolicy` picks a sideboard plan, a decision step
between games that sees the earlier games of the match from that seat's
point of view (`SeatGame`: who started, who won, the opponent's cards it
saw). The default `PlanMatrixPolicy` plays the fixed plan table
(`engine/sideboard_plans.toml`); docs/sideboarding.md describes the
interface a learned policy implements. The loser of the previous game
chooses to play first (always "play", the right call in these tempo
matchups); after a drawn game the previous starting player starts again. A
match ends when a player has two game wins or after three games.

`game_args` / `match_decks` (training games, the benchmark's batched
best-of-three) use the plan table directly: they equal `play_match` with
the default policy, which does not depend on the match history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .backend import game_class
from .engine import DECKS, SIDEBOARDS, SideboardPlan, expand, plan_for, postboard
from .engine.sideboard import validate_plan

# The deck each seat plays unless a matchup says otherwise. Seat 0 is always
# Jund Wildfire; the network's `seat:` feature has always meant this pairing.
DECK_NAMES = ("jund_wildfire", "mono_blue_terror")

# Matchups by name: the deck in seat 0 and seat 1.
MATCHUPS = {
    "jund_blue": ("jund_wildfire", "mono_blue_terror"),
    "jund_madness": ("jund_wildfire", "red_madness"),
    "blue_madness": ("mono_blue_terror", "red_madness"),
    "jund_affinity": ("jund_wildfire", "grixis_affinity"),
    "blue_affinity": ("mono_blue_terror", "grixis_affinity"),
    "madness_affinity": ("red_madness", "grixis_affinity"),
    "jund_elves": ("jund_wildfire", "elves"),
    "blue_elves": ("mono_blue_terror", "elves"),
    "madness_elves": ("red_madness", "elves"),
    "affinity_elves": ("grixis_affinity", "elves"),
    "jund_tron": ("jund_wildfire", "tron"),
    "blue_tron": ("mono_blue_terror", "tron"),
    "madness_tron": ("red_madness", "tron"),
    "affinity_tron": ("grixis_affinity", "tron"),
    "elves_tron": ("elves", "tron"),
    "blue_mirror": ("mono_blue_terror", "mono_blue_terror"),  # the Delver mirror (play UI); see EXPLICIT_ONLY
    "jund_mirror": ("jund_wildfire", "jund_wildfire"),
    "madness_mirror": ("red_madness", "red_madness"),
    "affinity_mirror": ("grixis_affinity", "grixis_affinity"),
    "elves_mirror": ("elves", "elves"),
    "tron_mirror": ("tron", "tron"),
}
# Matchups used only where they are named (a --matchup list, the play
# config): code that enumerates MATCHUPS for training mixes, benchmarks or
# tools skips them, so earlier runs and numbers stay comparable.
EXPLICIT_ONLY = frozenset(name for name, (a, b) in MATCHUPS.items() if a == b)
DEFAULT_MATCHUP = "jund_blue"


def parse_matchups(spec: str) -> list[tuple[str, float]]:
    """A training mix: comma-separated matchups, each with an optional
    `:weight` (default 1), e.g. "jund_blue:2,jund_madness,blue_madness"."""
    out = []
    for part in spec.split(","):
        name, _, w = part.strip().partition(":")
        matchup_decks(name)
        weight = float(w) if w else 1.0
        if weight <= 0 or name in dict(out):
            raise ValueError(f"matchup mix {spec!r}: weights must be > 0 and each matchup listed once")
        out.append((name, weight))
    return out


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
    names = matchup_decks(matchup)
    return {"decks": match_decks(game_no, matchup), "match_game": game_no, "deck_names": deck_names(matchup),
            "registered_main": tuple(expand(DECKS[d]) for d in names),
            "registered_sideboards": tuple(expand(SIDEBOARDS[d]) for d in names)}


def game_seed(match_seed: int, game_no: int) -> int:
    return match_seed * 4 + game_no


def first_starting_player(match_seed: int) -> int:
    return random.Random(match_seed).randrange(2)


def next_starting_player(prev_start: int, prev_winner: int | None) -> int:
    return prev_start if prev_winner is None else 1 - prev_winner


@dataclass(frozen=True)
class SeatGame:
    """One finished game of a match, as one seat saw it: the input a
    sideboarding policy gets for each earlier game."""

    game_no: int
    on_the_play: bool
    result: str  # "win" | "loss" | "draw"
    reason: str
    plan: SideboardPlan  # the plan this seat played the game with
    opponent_seen: tuple[str, ...]  # opponent's cards this seat saw (public zones at the end, revealed hand cards), sorted


class SideboardPolicy(Protocol):
    def choose(self, deck: str, opponent: str, game_no: int, history: Sequence[SeatGame]) -> SideboardPlan:
        """The plan for game `game_no` (2 or 3) of `deck` against
        `opponent`, given this seat's view of the earlier games."""
        ...


class PlanMatrixPolicy:
    """The scripted default: the plan table's row for (deck, opponent),
    whatever happened in the match so far."""

    def choose(self, deck: str, opponent: str, game_no: int, history: Sequence[SeatGame]) -> SideboardPlan:
        return plan_for(deck, opponent)


class NoSideboardPolicy:
    """Every game with the maindeck (a baseline for measuring the plans)."""

    def choose(self, deck: str, opponent: str, game_no: int, history: Sequence[SeatGame]) -> SideboardPlan:
        return SideboardPlan.none(deck, opponent)


def _front_names() -> dict[str, str]:
    from .engine.cards import CARDS

    return {d.back.name: name for name, d in CARDS.items() if d.back is not None}


def opponent_seen(g, seat: int) -> tuple[str, ...]:
    """The opponent's cards `seat` can name at the end of game `g`: its
    graveyard, exile, nontoken permanents and hand cards revealed to `seat`
    (back faces folded to the card's name). Cards seen earlier that went back
    to a hidden zone are not included."""
    from .engine.view import PLOTTED, observe

    obs = observe(g, seat)
    opp = obs["opponent"]
    names = list(opp["graveyard"]) + [x.removesuffix(PLOTTED) for x in opp["exile"]] + list(opp["hand_known"])
    names += [c["name"] for c in obs["battlefield"] if c["controller"] == "opponent" and not c["token"]]
    front = _front_names()
    return tuple(sorted(front.get(n, n) for n in names))


@dataclass
class MatchResult:
    games: list[tuple[int, int | None, str]] = field(default_factory=list)  # (starting player, winner, reason)
    plans: list[tuple[SideboardPlan, SideboardPlan]] = field(default_factory=list)  # per game, each seat's plan (play_match only)
    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)  # per game, opponent_seen of each seat (play_match only)

    def history(self, seat: int) -> list[SeatGame]:
        """The finished games from `seat`'s point of view (needs `plans` and `seen`)."""
        out = []
        for i, (start, winner, reason) in enumerate(self.games):
            result = "draw" if winner is None else "win" if winner == seat else "loss"
            out.append(SeatGame(i + 1, start == seat, result, reason, self.plans[i][seat], self.seen[i][seat]))
        return out

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


def choose_plans(res: MatchResult, game_no: int, matchup: str, policies: Sequence[SideboardPolicy]) -> tuple[SideboardPlan, SideboardPlan]:
    """Each seat's plan for game `game_no`: maindecks in game 1, otherwise its
    policy's choice (validated) given its view of the match so far."""
    decks = matchup_decks(matchup)
    out = []
    for s in (0, 1):
        deck, opp = decks[s], decks[1 - s]
        if game_no == 1:
            out.append(SideboardPlan.none(deck, opp))
            continue
        plan = policies[s].choose(deck, opp, game_no, res.history(s))
        if (plan.deck, plan.opponent) != (deck, opp):
            raise ValueError(f"seat {s}: policy returned a plan for {plan.deck} vs {plan.opponent}, expected {deck} vs {opp}")
        validate_plan(plan)
        out.append(plan)
    return out[0], out[1]


def play_match(agents, seed: int = 0, engine: str | None = None, matchup: str = DEFAULT_MATCHUP, policies: Sequence[SideboardPolicy] | None = None, **game_kw) -> MatchResult:
    """agents: [seat0, seat1] objects with act(game), reused across games.
    policies: each seat's SideboardPolicy (default: PlanMatrixPolicy for both)."""
    from .agents import take  # agents imports the engine; keep this module light

    policies = policies or (PlanMatrixPolicy(), PlanMatrixPolicy())
    Game = game_class(engine)
    res = MatchResult()
    while not res.over:
        n, start = res.next_game(seed)
        plans = choose_plans(res, n, matchup, policies)
        decks = tuple(expand(p.apply()) for p in plans)
        args = game_args(n, matchup)
        args["decks"] = decks
        args.update(game_kw)
        g = Game(**args, seed=game_seed(seed, n), starting_player=start)
        while not g.over:
            take(g, agents, agents[g.decision.player].act(g))
        res.games.append((start, g.winner, g.end_reason))
        res.plans.append(plans)
        res.seen.append((opponent_seen(g, 0), opponent_seen(g, 1)))
    return res
