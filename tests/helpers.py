"""Scenario helpers: build a game in an arbitrary position and drive it by
option labels."""

from __future__ import annotations

from mtg_ml.backend import game_class
from mtg_ml.engine import Game
from mtg_ml.engine import objects as O


def new_game(*args, **kw) -> Game:
    """A game on the engine selected by $MTG_ENGINE (set per test by conftest.py)."""
    return game_class()(*args, **kw)


def scenario(p0: dict | None = None, p1: dict | None = None, active: int = 0, step: str = "main1", auto_single: bool = False, **game_kw) -> Game:
    """p0/p1: dict with optional keys hand, battlefield, graveyard, exile,
    library (top first), life, drawn. Battlefield entries are names or
    (name, kwargs-for-add_card). Libraries default to 20 basic lands.
    `game_kw` go to the Game constructor (e.g. auto_mana=True)."""

    def setup(g: Game) -> None:
        for idx, spec in ((0, p0 or {}), (1, p1 or {})):
            for zone in ("hand", "graveyard", "exile"):
                for name in spec.get(zone, []):
                    g.add_card(name, idx, zone)
            for entry in spec.get("battlefield", []):
                name, kw = (entry, {}) if isinstance(entry, str) else entry
                g.add_card(name, idx, "battlefield", **kw)
            for name in spec.get("library", ["Swamp" if idx == 0 else "Island"] * 20):
                g.add_card(name, idx, "library")
            g.players[idx].life = spec.get("life", 20)
            g.players[idx].cards_drawn_this_turn = spec.get("drawn", 0)

    g = new_game(([], []), seed=0, starting_player=active, setup=setup, start_step=step, auto_single=auto_single, log=True, **game_kw)
    settle(g)
    return g


def settle(g: Game) -> None:
    """Take every forced (single-option) decision except priority, so tests
    only have to drive real choices and priority passes."""
    while g.decision is not None and g.decision.kind != O.PRIORITY and len(g.legal_options()) == 1:
        g.step(0)


def labels(g: Game) -> list[str]:
    return [o.label for o in g.legal_options()]


def choose(g: Game, text: str) -> None:
    """Pick the option whose label equals `text`, else the unique one containing it."""
    opts = g.legal_options()
    exact = [i for i, o in enumerate(opts) if o.label == text]
    if exact:
        g.step(exact[0])
    else:
        hits = [i for i, o in enumerate(opts) if text in o.label]
        assert len(hits) == 1, f"{text!r} matches {len(hits)} of {[o.label for o in opts]} ({g.decision})"
        g.step(hits[0])
    settle(g)


def has(g: Game, text: str) -> bool:
    return any(text in o.label for o in g.legal_options())


def pass_priority(g: Game, times: int = 1) -> None:
    for _ in range(times):
        assert g.decision.kind == O.PRIORITY, g.decision
        choose(g, "Pass priority")


def pay(g: Game) -> None:
    """Pay any pending mana cost with the first offered option each time."""
    while g.decision is not None and g.decision.kind == O.PAY_MANA:
        g.step(0)
    settle(g)


def resolve_stack(g: Game) -> None:
    """Both players pass until the stack is empty (and no decision interrupts)."""
    while g.stack and g.decision is not None and g.decision.kind == O.PRIORITY:
        choose(g, "Pass priority")


def names(cards) -> list[str]:
    return [c.name for c in cards]


def bf(g: Game, player: int | None = None) -> list[str]:
    return [c.name for c in g.battlefield if player is None or c.controller == player]


def find(g: Game, name: str, player: int | None = None):
    hits = [c for c in g.battlefield if c.name == name and (player is None or c.controller == player)]
    assert hits, f"{name} not on battlefield: {bf(g)}"
    return hits[0]
