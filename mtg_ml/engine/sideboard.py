"""Sideboard plans: what a deck boards in and out for games 2 and 3.

The plans are data, `sideboard_plans.toml` next to this module, one
`[[plan]]` per ordered (deck, opponent) pair:

    [[plan]]
    deck = "jund_wildfire"
    opponent = "mono_blue_terror"
    why = "one line: what the swap is for"
    in = { "Duress" = 3, "Pyroblast" = 2 }
    out = { "Lembas" = 3, "Nyxborn Hydra" = 1, "Toxin Analysis" = 1 }

`load_plans` checks every plan against `decks.DECKS` / `decks.SIDEBOARDS`
(both decks exist, `out` only names maindeck cards and `in` only sideboard
cards, in the counts available, the same number in as out, and the result
is 60 cards with at most 4 copies of anything but basic lands). Plans are
applied Python-side: both engines receive the resulting decklists, so the
Rust port needs no copy of this data.

A pair without a plan plays its maindeck unchanged (`SideboardPlan.none`);
tests/test_sideboard.py requires a plan for every pair of decks, so a new
deck must bring one row per opponent (and each opponent one against it).
How games 2/3 choose a plan (the `SideboardPolicy` interface, and the
default `PlanMatrixPolicy` that reads this table): `mtg_ml/match.py` and
docs/sideboarding.md.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .decks import DECKS, SIDEBOARDS

PLANS_PATH = os.path.join(os.path.dirname(__file__), "sideboard_plans.toml")
BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"})
MAIN_SIZE, SIDE_SIZE, MAX_COPIES = 60, 15, 4


def _frozen(d: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(d))


@dataclass(frozen=True)
class SideboardPlan:
    """Cards swapped between maindeck and sideboard, relative to the
    maindeck as registered (`decks.DECKS[deck]`)."""

    deck: str
    opponent: str
    cards_in: Mapping[str, int] = field(default_factory=dict)
    cards_out: Mapping[str, int] = field(default_factory=dict)
    why: str = ""

    def __post_init__(self):
        object.__setattr__(self, "cards_in", _frozen(self.cards_in))
        object.__setattr__(self, "cards_out", _frozen(self.cards_out))

    @classmethod
    def none(cls, deck: str, opponent: str) -> "SideboardPlan":
        return cls(deck, opponent, {}, {}, "no changes")

    @property
    def swaps(self) -> int:
        return sum(self.cards_in.values())

    def apply(self, main: Mapping[str, int] | None = None) -> dict[str, int]:
        """The maindeck after the swap (insertion order: the maindeck's, then new cards)."""
        main = dict(DECKS[self.deck] if main is None else main)
        for name, n in self.cards_out.items():
            main[name] = main.get(name, 0) - n
            if main[name] <= 0:
                del main[name]
        for name, n in self.cards_in.items():
            main[name] = main.get(name, 0) + n
        return main

    def side_after(self, side: Mapping[str, int] | None = None) -> dict[str, int]:
        """The sideboard after the swap."""
        side = dict(SIDEBOARDS[self.deck] if side is None else side)
        for name, n in self.cards_in.items():
            side[name] = side.get(name, 0) - n
            if side[name] <= 0:
                del side[name]
        for name, n in self.cards_out.items():
            side[name] = side.get(name, 0) + n
        return side


def check_deck(name: str, main: Mapping[str, int], side: Mapping[str, int], side_size: int | None = SIDE_SIZE) -> None:
    """60-card maindeck, `side_size`-card sideboard (None: at most 15), at
    most 4 copies of any non-basic across both."""
    if sum(main.values()) != MAIN_SIZE:
        raise ValueError(f"{name}: maindeck has {sum(main.values())} cards, expected {MAIN_SIZE}")
    n_side = sum(side.values())
    if (side_size is not None and n_side != side_size) or n_side > SIDE_SIZE:
        raise ValueError(f"{name}: sideboard has {n_side} cards, expected {side_size if side_size is not None else f'<= {SIDE_SIZE}'}")
    for card in set(main) | set(side):
        n = main.get(card, 0) + side.get(card, 0)
        if card not in BASIC_LANDS and n > MAX_COPIES:
            raise ValueError(f"{name}: {n} copies of {card} across maindeck and sideboard")


def validate_plan(plan: SideboardPlan, decks: Mapping = DECKS, sideboards: Mapping = SIDEBOARDS) -> None:
    where = f"sideboard plan {plan.deck} vs {plan.opponent}"
    for d in (plan.deck, plan.opponent):
        if d not in decks:
            raise ValueError(f"{where}: unknown deck {d!r}")
    main, side = decks[plan.deck], sideboards.get(plan.deck, {})
    for name, n in plan.cards_out.items():
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"{where}: out {name!r} = {n!r}, expected a positive count")
        if main.get(name, 0) < n:
            raise ValueError(f"{where}: cannot board out {n} {name} (maindeck has {main.get(name, 0)})")
    for name, n in plan.cards_in.items():
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"{where}: in {name!r} = {n!r}, expected a positive count")
        if side.get(name, 0) < n:
            raise ValueError(f"{where}: cannot board in {n} {name} (sideboard has {side.get(name, 0)})")
    if sum(plan.cards_in.values()) != sum(plan.cards_out.values()):
        raise ValueError(f"{where}: {sum(plan.cards_in.values())} cards in but {sum(plan.cards_out.values())} out")
    if set(plan.cards_in) & set(plan.cards_out):
        raise ValueError(f"{where}: {sorted(set(plan.cards_in) & set(plan.cards_out))} both in and out")
    check_deck(where, plan.apply(main), plan.side_after(side), side_size=None)


def parse_plans(text: str, decks: Mapping = DECKS, sideboards: Mapping = SIDEBOARDS) -> dict[tuple[str, str], SideboardPlan]:
    spec = tomllib.loads(text)
    if set(spec) - {"plan"}:
        raise ValueError(f"sideboard plans: unknown top-level keys {sorted(set(spec) - {'plan'})}")
    plans: dict[tuple[str, str], SideboardPlan] = {}
    for row in spec.get("plan", ()):
        unknown = set(row) - {"deck", "opponent", "why", "in", "out"}
        missing = {"deck", "opponent", "why"} - set(row)
        if unknown or missing:
            raise ValueError(f"sideboard plan {row.get('deck')} vs {row.get('opponent')}: unknown fields {sorted(unknown)}, missing {sorted(missing)}")
        plan = SideboardPlan(row["deck"], row["opponent"], row.get("in", {}), row.get("out", {}), row["why"].strip())
        if (plan.deck, plan.opponent) in plans:
            raise ValueError(f"duplicate sideboard plan {plan.deck} vs {plan.opponent}")
        validate_plan(plan, decks, sideboards)
        plans[(plan.deck, plan.opponent)] = plan
    return plans


def load_plans(path: str = PLANS_PATH) -> dict[tuple[str, str], SideboardPlan]:
    with open(path, encoding="utf-8") as f:
        return parse_plans(f.read())


PLANS = load_plans()


def plan_for(deck: str, opponent: str) -> SideboardPlan:
    """The table's plan for `deck` against `opponent` (no changes if there is none)."""
    return PLANS.get((deck, opponent)) or SideboardPlan.none(deck, opponent)


def postboard(deck: str, opponent: str) -> dict[str, int]:
    """The maindeck of `deck` after applying its plan against `opponent`."""
    return plan_for(deck, opponent).apply()


# Back-compatible view of the table: {(deck, opponent): {"in": {...}, "out": {...}}}.
SIDEBOARD_PLANS = {k: {"in": dict(p.cards_in), "out": dict(p.cards_out)} for k, p in PLANS.items()}
