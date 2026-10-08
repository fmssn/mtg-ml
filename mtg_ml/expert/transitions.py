"""Reconstruction-only transition facts, derived from legal engine execution.

These include private completion identities. They are never policy inputs or
source observations. Matching an engine fact does not approve an extraction.
"""

from __future__ import annotations

from collections import Counter
import re
from types import SimpleNamespace

from .scenarios import canonical


CHECKABLE = {
    "cast",
    "play_land",
    "activate",
    "cycle",
    "turn",
    "draw",
    "mill",
    "discard",
    "exile",
    "return_hand",
    "reveal",
    "shuffle",
    "token",
    "counter",
    "trigger",
    "attack",
    "block",
    "choice",
    "order_choice",
    "scry",
    "result",
    "sacrifice",
    "explore",
}


def frozen_option(option):
    """Materialize native decision proxies before advancing the engine."""
    return SimpleNamespace(key=canonical(option.key), label=option.label)


def capture(g):
    cards = {}
    for p, player in enumerate(g.players):
        for zone in ("hand", "library", "graveyard", "exile"):
            for c in getattr(player, zone):
                cards[c.uid] = (c.name, p, zone, c.oid, c.counters, c.is_token, tuple(sorted(c.known_to)))
    for c in g.battlefield:
        cards[c.uid] = (c.name, c.controller, "battlefield", c.oid, c.counters, c.is_token, tuple(sorted(c.known_to)))

    def item(it):
        return {
            "sid": it.sid,
            "kind": it.kind,
            "name": it.name,
            "player": it.controller,
            "method": it.method,
            "targets": canonical(it.targets),
        }

    return {
        "cards": cards,
        "stack": [item(it) for it in g.stack],
        "log_size": len(g.log),
        "turn": g.turn,
        "active": g.active,
        "libraries": [[c.uid for c in p.library] for p in g.players],
        "drawn": [p.cards_drawn_this_turn for p in g.players],
        "rng": g.rng.getstate(),
        "over": g.over,
        "winner": g.winner,
    }


def transition(g, before, decision, option):
    after = capture(g)
    facts = []
    key = canonical(option.key)
    p, kind = decision
    resolving = next(
        (it for it in reversed(before["stack"]) if "resolve " + it["name"] in g.log[before["log_size"] :]), None
    )
    cause = resolving or (after["stack"][-1] if after["stack"] else before["stack"][-1] if before["stack"] else None)
    cause_id = ("stack", cause["sid"]) if cause else ("turn", after["turn"])

    def add(t, player=None, **kw):
        facts.append({"type": t, "player": player, "cause": cause_id, **kw})

    if kind == "declare_attacker" and key[0] == "attack" and key[1] is None:
        add("attack", p, cards=[c.name for c in g.battlefield if c.oid in g.attackers])
    if kind == "declare_blocker" and key[0] == "block" and key[-1] is not None:
        add("block", p, card=key[1], targets=[key[2]])
    if kind == "yes_no":
        add("choice", p, cards=[it["name"] for it in before["stack"]], yes=key[-1] == "yes")
        if key[0] == "shuffle" and key[-1] == "yes":
            add("shuffle", p)
    if kind == "order":
        add("order_choice", p, cards=[it["name"] for it in before["stack"]], key=key)
    if key[0] == "scry":
        if len(key) == 2:
            top = int(key[1] == "top")
            bottom = 1 - top
        else:
            split = key.index("bottom")
            top = split - 2
            bottom = len(key) - split - 1
        add("scry", p, key=key, count=top + bottom, top=top, bottom=bottom)
    names = {c[3]: c[0] for c in before["cards"].values()} | {c[3]: c[0] for c in after["cards"].values()}
    stack_names = {it["sid"]: it["name"] for it in before["stack"] + after["stack"]}
    # The engine log attests completion of casting, after targets and costs.
    lines = list(g.log[before["log_size"] :])
    if kind != "priority" and g.decision is not None and g.decision.kind == "priority":
        # Casting logs before target/cost prompts. Emit the cast constraint
        # only after these finish; a resolving spell is not a new cast.
        for it in after["stack"]:
            if it["kind"] not in {"spell", "ability"}:
                continue
            pattern = (
                r"p[01] casts " + re.escape(it["name"]) + r" \("
                if it["kind"] == "spell"
                else r"p[01] activates " + re.escape(it["name"]) + "$"
            )
            cast_index = next(
                (i for i in range(len(g.log) - 1, -1, -1) if re.match(pattern, g.log[i])),
                None,
            )
            cast_line = g.log[cast_index] if cast_index is not None else None
            resolving = (
                any(line == "resolve " + it["name"] for line in g.log[cast_index + 1 :])
                if cast_index is not None
                else True
            )
            if cast_line and not resolving and cast_line not in lines:
                lines.append(cast_line)
    for line in lines:
        activation = re.match(r"p([01]) activates (.+): (.+)$", line)
        if activation and g.decision is not None and g.decision.kind == "priority":
            name = activation[2]
            it = next((it for it in after["stack"] if it["name"] == name + ": " + activation[3]), None)
            targets = []
            target_player = None
            for target_kind, oid in (it or {}).get("targets", []):
                if target_kind == "player":
                    target_player = oid
                elif oid in names:
                    targets.append(names[oid])
            add(
                "cycle" if "cycl" in activation[3].lower() else "activate",
                int(activation[1]),
                card=name,
                targets=targets,
                target_player=target_player,
            )
        m = re.match(r"p([01]) casts (.+) \(([^)]+)\) from (\w+)", line)
        if m and g.decision is not None and g.decision.kind == "priority":
            actor, name, method, zone = int(m[1]), m[2], m[3], m[4]
            it = next((it for it in after["stack"] if it["name"] == name and it["kind"] == "spell"), None)
            targets, target_player = [], None
            for target_kind, oid in (it or {}).get("targets", []):
                if target_kind == "player":
                    target_player = oid
                elif target_kind == "stack" and oid in stack_names:
                    targets.append(stack_names[oid])
                elif oid in names:
                    targets.append(names[oid])
            add("cast", actor, card=name, method=method, targets=targets, target_player=target_player, zone=zone)
        m = re.match(r"p([01]) plays (.+)", line)
        if m:
            add("play_land", int(m[1]), card=m[2])
        m = re.match(r"p([01]) sacrifices (.+)#\d+", line)
        if m:
            add("sacrifice", int(m[1]), card=m[2])
        m = re.match(r"trigger -> stack: (.+)", line)
        if m:
            it = next((it for it in after["stack"] if it["name"] == m[1]), None)
            from ..engine.cards import CARDS

            source = next((name for name in sorted(CARDS, key=len, reverse=True) if m[1].startswith(name)), m[1])
            add("trigger", (it or {}).get("player"), card=source)
        m = re.match(r"p([01]) reveals their hand for (.+): (.*)", line)
        if m:
            add("reveal", int(m[1]), cards=m[3].split(", ") if m[3] != "empty" else [])
        m = re.match(r"=== Turn (\d+): player ([01]) ===", line)
        if m:
            add("turn", int(m[2]), engine_turn=int(m[1]))
    moved = {
        (a, b, p): []
        for a, b in [("library", "hand"), ("library", "graveyard"), ("hand", "graveyard"), ("battlefield", "hand")]
        for p in (0, 1)
    }
    tokens = {0: [], 1: []}
    exiled = {0: [], 1: []}
    counters = []
    for uid, c in after["cards"].items():
        old = before["cards"].get(uid)
        name, actor, zone, oid, count, token, known = c
        if old:
            pair = (old[2], zone, actor)
            if old[2] != zone and pair in moved:
                moved[pair].append(name)
            if zone == "exile" and old[2] != zone:
                exiled[actor].append(name)
            if count > old[4]:
                counters.append((actor, name, count - old[4]))
            if set(known) == {0, 1} and set(old[6]) != {0, 1} and zone in {"hand", "library"}:
                add("reveal", actor, cards=[name])
        if zone == "battlefield" and token and (old is None or old[2] != zone):
            tokens[actor].append(name)
    for (a, b, actor), cards in moved.items():
        if not cards:
            continue
        t = {
            ("library", "hand"): "draw",
            ("library", "graveyard"): "mill",
            ("hand", "graveyard"): "discard",
            ("battlefield", "hand"): "return_hand",
        }[a, b]
        if t == "draw":
            count = after["drawn"][actor] - (before["drawn"][actor] if before["turn"] == after["turn"] else 0)
            if count <= 0:
                continue
            cards = cards[:count]
        # A sacrificed/cast instant leaves hand via stack, not a discard.
        if t == "discard" and any(f["type"] == "cast" and f["card"] in cards for f in facts):
            cards = [c for c in cards if not any(f["type"] == "cast" and f["card"] == c for f in facts)]
        if cards:
            add(t, actor, cards=cards, count=len(cards))
    for actor, cards in tokens.items():
        if cards:
            add("token", actor, cards=cards, count=len(cards))
    for actor, cards in exiled.items():
        if cards:
            add("exile", actor, cards=cards)
    for it in before["stack"]:
        if "Map" in it["name"] and "resolve " + it["name"] in g.log[before["log_size"] :]:
            target = next((names.get(oid) for kind, oid in it["targets"] if kind == "permanent"), None)
            if target:
                add("explore", it["player"], card=target, cards=[target])
    for actor, name, count in counters:
        add("counter", actor, card=name, count=count)
    if before["rng"] != after["rng"]:
        for actor in (0, 1):
            if (
                set(before["libraries"][actor]) == set(after["libraries"][actor])
                and before["libraries"][actor] != after["libraries"][actor]
            ):
                if not any(f["type"] == "shuffle" and f["player"] == actor for f in facts):
                    add("shuffle", actor)
    if after["over"] and not before["over"]:
        add("result", after["winner"])
    return facts


def compatible(event, fact):
    if event["type"] != fact["type"]:
        return False
    if event.get("player") is not None and event["player"] != fact.get("player"):
        return False
    for field in ("card", "engine_turn", "count", "yes", "method", "target_player", "top", "bottom"):
        if event.get(field) is not None and event[field] != fact.get(field):
            return False
    if event.get("targets") and Counter(event["targets"]) != Counter(fact.get("targets", [])):
        return False
    cards = event.get("cards", [])
    if event["type"] == "token":
        from ..engine.cards import TOKENS

        cards = [c for c in cards if c in TOKENS]
    if event["type"] in {"attack", "mill", "discard", "exile", "return_hand", "token", "reveal"} and cards:
        if event["type"] == "attack" and Counter(cards) != Counter(fact.get("cards", [])):
            return False
        if Counter(cards) - Counter(fact.get("cards", [])):
            return False
    return True


def advance(events, cursor, facts, pending=None):
    """Group split cost choices from one engine cause, never unrelated draws."""
    remaining = list(pending or []) + list(facts)
    matched = []
    while cursor < len(events):
        event = events[cursor]["event"]
        hit = next((i for i, f in enumerate(remaining) if compatible(event, f)), None)
        if hit is None:
            groups = {}
            if event["type"] in {"exile", "discard", "mill", "draw", "reveal"}:
                for i, f in enumerate(remaining):
                    if f["type"] == event["type"] and event.get("player") in (None, f.get("player")):
                        groups.setdefault((f.get("player"), f.get("cause")), []).append(i)
            group_hit = None
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                aggregate = {
                    **remaining[indices[0]],
                    "cards": [c for i in indices for c in remaining[i].get("cards", [])],
                    "count": sum(remaining[i].get("count", len(remaining[i].get("cards", []))) for i in indices),
                }
                if compatible(event, aggregate):
                    group_hit = indices
                    break
            if group_hit is None:
                break
            for i in reversed(group_hit):
                remaining.pop(i)
        else:
            remaining.pop(hit)
        matched.append(events[cursor]["id"])
        cursor += 1
    if cursor < len(events):
        wanted = events[cursor]["event"]
        remaining = [
            f for f in remaining if f["type"] == wanted["type"] and wanted.get("player") in (None, f.get("player"))
        ]
    else:
        remaining = []
    return cursor, matched, remaining
