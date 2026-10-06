"""Baseline agents and a game runner."""

from __future__ import annotations

import random

from .backend import game_class
from .engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, expand
from .engine.view import observe


class RandomAgent:
    """Uniform over options, but passes priority with probability `pass_bias`
    when passing is allowed (pure uniform play rarely finishes a turn), and
    always keeps its opening hand."""

    def __init__(self, seed: int = 0, pass_bias: float = 0.4):
        self.rng = random.Random(seed)
        self.pass_bias = pass_bias

    def act(self, game: Game) -> int:
        opts = game.legal_options()
        if game.decision.kind == "mulligan":
            return 0  # keep
        if opts[0].key == ("pass",) and self.rng.random() < self.pass_bias:
            return 0
        return self.rng.randrange(len(opts))


class HumanAgent:
    """Terminal prompt. Shows only what the deciding player may see."""

    def act(self, game: Game) -> int:
        d = game.decision
        print(render(game, d.player))
        print(f"\n[{d.kind}] {d.prompt}")
        for i, o in enumerate(d.options):
            print(f"  {i:2d}: {o.label}")
        while True:
            raw = input("> ").strip()
            if raw.isdigit() and int(raw) < len(d.options):
                return int(raw)
            print("enter an option number")


def render(game: Game, viewer: int) -> str:
    o = observe(game, viewer)
    lines = [f"Turn {o['turn']} ({o['active']} active), step {o['step']}"]
    for side in ("opponent", "self"):
        s = o[side]
        hand = s["hand"] if side == "self" else f"{s['hand_count']} cards"
        lines.append(f"{side.upper()}: life {s['life']}, library {s['library_count']}, hand: {hand}")
        lines.append(f"   graveyard: {', '.join(s['graveyard']) or '-'}")
        perms = []
        for p in o["battlefield"]:
            if p["controller"] != side:
                continue
            tag = p["name"]
            if p["power"] is not None:
                tag += f" {p['power']}/{p['toughness']}"
            flags = [k for k in ("tapped", "sick", "attacking") if p[k]] + (["blocking"] if p["blocking"] else [])
            if p["counters"]:
                flags.append(f"+{p['counters']}")
            if p["damage"]:
                flags.append(f"{p['damage']} dmg")
            perms.append(tag + (f" ({', '.join(flags)})" if flags else ""))
        lines.append(f"   battlefield: {'; '.join(perms) or '-'}")
    if o["stack"]:
        lines.append("STACK (top first): " + " | ".join(f"{it['name']} -> {it['targets']}" for it in reversed(o["stack"])))
    return "\n".join(lines)


def take(g: Game, agents, index: int) -> None:
    """Take option `index`, first telling every agent with an `observe` hook
    (recurrent models track both players' actions) what is being done."""
    for a in agents:
        if hasattr(a, "observe"):
            a.observe(g, index)
    g.step(index)


def play_game(agents, seed: int = 0, decks=None, engine: str | None = None, **game_kw) -> Game:
    """engine: "python" / "native" (default: $MTG_ENGINE, else python)."""
    decks = decks or (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))
    g = game_class(engine)(decks, seed=seed, **game_kw)
    while not g.over:
        take(g, agents, agents[g.decision.player].act(g))
    return g
