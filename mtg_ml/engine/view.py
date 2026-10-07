"""Player views: what one player may know, plus determinization for search.

`observe(game, viewer)` returns only information available to `viewer`:
public zones, their own hand, the opponent's hand size (plus cards revealed
to them), and library cards whose identity they learned (Brainstorm,
Ponder, Delver, Deem Inferior, scry...). `determinize(game, viewer, rng)`
returns a copy in which every card hidden from `viewer` is re-sampled
consistently with that view, the standard trick (Cowling et al. 2012) for
running perfect-information search such as MCTS under hidden information.
"""

from __future__ import annotations

import random

from .game import Game


def _perm_view(g: Game, c, viewer: int) -> dict:
    return {
        "oid": c.oid,
        "name": c.name,
        "controller": "self" if c.controller == viewer else "opponent",
        "token": c.is_token,
        "types": sorted(g.types(c)),
        "tapped": c.tapped,
        "sick": c.sick,
        "damage": c.damage,
        "counters": c.counters,
        "power": g.power(c) if g.is_creature(c) else None,
        "toughness": g.toughness(c) if g.is_creature(c) else None,
        "keywords": sorted(g.keywords(c)),
        "attached_to": c.attached_to,
        "attacking": c.oid in g.attackers,
        "blocking": g.blocks.get(c.oid),
        "skip_untap": c.skip_untap,
    }


def observe(g: Game, viewer: int) -> dict:
    if getattr(g, "NATIVE", False):
        return g.observe(viewer)
    opp = 1 - viewer
    me, them = g.players[viewer], g.players[opp]

    def known_library(p):
        return [(i, c.name) for i, c in enumerate(p.library) if viewer in c.known_to]

    d = g.decision
    obs = {
        "turn": g.turn,
        "match_game": g.match_game,
        "active": "self" if g.active == viewer else "opponent",
        "step": g.step_name,
        "over": g.over,
        "winner": None if g.winner is None else ("self" if g.winner == viewer else "opponent"),
        "self": {
            "life": me.life,
            "hand": [c.name for c in me.hand],
            "library_count": len(me.library),
            "library_known": known_library(me),
            "graveyard": [c.name for c in me.graveyard],
            "exile": [_exiled(c) for c in me.exile],
            "pool": dict(me.pool),
            "cards_drawn_this_turn": me.cards_drawn_this_turn,
            "mulligans": g.mulligans_taken[viewer],
        },
        "opponent": {
            "life": them.life,
            "hand_count": len(them.hand),
            "hand_known": [c.name for c in them.hand if viewer in c.known_to],
            "library_count": len(them.library),
            "library_known": known_library(them),
            "graveyard": [c.name for c in them.graveyard],
            "exile": [_exiled(c) for c in them.exile],
            "pool": dict(them.pool),
            "mulligans": g.mulligans_taken[opp],
        },
        "lands_played": g.lands_played if g.active == viewer else None,
        "battlefield": [_perm_view(g, c, viewer) for c in g.battlefield],
        "stack": [
            {
                "sid": it.sid,
                "name": it.name,
                "kind": it.kind,
                "controller": "self" if it.controller == viewer else "opponent",
                "targets": [g.describe_ref(t, viewer)[0] for t in it.targets if _ref_exists(g, t)],
                "x": it.x,
                "method": it.method,
            }
            for it in g.stack
        ],
        "decision": None,
    }
    if d is not None and d.player == viewer:
        obs["decision"] = {"kind": d.kind, "prompt": d.prompt, "options": [o.label for o in d.options], "keys": [o.key for o in d.options]}
    elif d is not None:
        obs["decision"] = {"kind": d.kind, "waiting_for": "opponent"}
    return obs


PLOTTED = " (plotted)"


def _exiled(c) -> str:
    """An exiled card's name; plotted cards (castable later) are marked."""
    return f"{c.name}{PLOTTED}" if c.plotted_turn else c.name


def _ref_exists(g: Game, ref: tuple) -> bool:
    if ref[0] == "player":
        return True
    if ref[0] == "stack":
        return g.stack_item(ref[1]) is not None
    return g.perm(ref[1]) is not None


def determinize(g: Game, viewer: int, rng: random.Random) -> Game:
    """Fork `g` and re-sample everything `viewer` cannot see.

    Hidden cards of the opponent (unknown hand cards and unknown library
    cards) are pooled and redistributed; the viewer's own unknown library
    cards are shuffled among their positions. Known cards stay put. Card
    objects keep their identity (oid/uid), only their definition changes, so
    engine bookkeeping stays valid.
    """
    if getattr(g, "NATIVE", False):
        return g.determinize(viewer, rng)
    f = g.fork()
    for p in f.players:
        hidden = [c for c in p.library if viewer not in c.known_to]
        if p.idx != viewer:
            hidden += [c for c in p.hand if viewer not in c.known_to]
        # Sort first, so the result depends only on what `viewer` knows (the
        # multiset of hidden cards), not on where they really are.
        defs = sorted((c.defn for c in hidden), key=lambda d: d.name)
        rng.shuffle(defs)
        for c, d in zip(hidden, defs):
            c.defn = d
            c.transformed = False
    f._state_edited()  # its copy snapshot predates the re-deal
    return f
