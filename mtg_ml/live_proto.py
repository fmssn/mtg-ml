"""Machine-readable decisions and events for the play client (apps/play).

The engine describes each option with a label, an id-free key and an
engine-internal value. The play client needs more: which card in hand an
option casts, which permanent an ability belongs to, which creature a
blocker would block. `option_refs` derives that from `Option.key` and
`Option.value` without touching the engine, so both engines (whose values
carry live card objects) give the same answer.

Nothing here may leak hidden information. Every object reference is checked
against the snapshot the player is sent (`visible_ids`): a uid or oid the
player cannot see there (a card in a library, the opponent's hand) is
dropped, and only the label's public part (a name the label already shows)
stays. `parse_events` reads only log lines that `replay.visible_events`
already let through.
"""

from __future__ import annotations

import re

_OID = re.compile(r"#(\d+)")


def visible_ids(state: dict) -> dict[str, set]:
    """uids, oids and stack ids that appear in a (player-view) snapshot."""
    uids, oids = set(), set()
    for p in state["players"]:
        for zone in ("hand", "graveyard", "exile"):
            uids.update(c["uid"] for c in p[zone] if c["uid"] >= 0)
    for c in state["battlefield"]:
        uids.add(c["uid"])
        oids.add(c["oid"])
    return {"uid": uids, "oid": oids, "sid": {it["sid"] for it in state["stack"]}}


def _card_ref(card, vis: dict, out: dict) -> None:
    """Add `card`'s ids to `out` when the player can see them."""
    if card is None:
        return
    out["name"] = card.name
    zone = getattr(card, "zone", None)
    if zone == "battlefield" and card.oid in vis["oid"]:
        out["oid"] = card.oid
        out["zone"] = zone
    elif zone in ("hand", "graveyard", "exile") and card.uid in vis["uid"]:
        out["uid"] = card.uid
        out["zone"] = zone


def _target(ref, vis: dict) -> dict | None:
    if ref is None:
        return None
    kind, n = ref[0], ref[1]
    if kind == "player":
        return {"player": n}
    if kind == "perm" and n in vis["oid"]:
        return {"oid": n}
    if kind == "stack" and n in vis["sid"]:
        return {"sid": n}
    return {}


def _label_oids(label: str) -> list[int]:
    return [int(m) for m in _OID.findall(label)]


def option_ref(kind: str, label: str, key: tuple, value, vis: dict) -> dict:
    """One option as the client sees it: `type` plus whatever refs apply."""
    head = key[0] if key else None
    r: dict = {"type": head or kind}
    if kind == "mulligan":
        r["type"] = "mulligan" if value else "keep"
        return r
    if kind == "priority":
        if head == "pass":
            return r
        act = value[0] if isinstance(value, tuple) else None
        card = value[1] if act in ("land", "cast", "activate", "plot", "mana") else None
        _card_ref(card, vis, r)
        if head == "cast":
            r["from"], r["mode"] = key[2], key[3]
            if len(key) > 4:
                r["spell_mode"] = key[4]
        elif head in ("activate", "mana"):
            r["ability"] = key[2]
        return r
    if kind == "target":
        r["type"] = "target"
        t = _target(value, vis)
        if t is None:
            r["none"] = True
        else:
            r.update(t)
        return r
    if kind == "pay_mana":
        r["type"] = "pay"
        if value[0] == "pool":
            r["pool"] = value[1]
        else:
            r["via"] = value[0]
            _card_ref(value[1], vis, r)
        r["color"] = value[-1]
        return r
    if kind == "declare_attacker":
        r["type"] = "attack"
        oids = _label_oids(label)
        if value is None or not oids:
            r["done"] = True
        elif oids[0] in vis["oid"]:
            r["attacker"] = oids[0]
            r["name"] = key[1]
        return r
    if kind == "declare_blocker":
        r["type"] = "block"
        oids = [o for o in _label_oids(label) if o in vis["oid"]]
        if oids:
            r["blocker"] = oids[0]
        if value is None:
            r["none"] = True
        elif len(oids) > 1:
            r["attacker"] = oids[1]
            r["attacker_name"] = key[2]
        return r
    if kind == "assign_damage_amount":
        r.update(type="damage_amount", amount=value, remaining=key[5], assigned=list(key[6]), player_damage=key[7], player=key[3] == -1)
        oids = [o for o in _label_oids(label) if o in vis["oid"]]
        if oids:
            r["attacker"] = oids[-1]
            if key[3] >= 0:
                r["recipient"] = oids[0]
        return r
    if kind == "assign_damage":
        r["type"] = "damage"
        r["split"] = list(value)
        r["to"] = [o for o in _label_oids(label) if o in vis["oid"]]
        r["player"] = label.endswith("to player")
        return r
    if kind == "choose_x":
        r["type"] = "x"
        r["x"] = value
        return r
    # Library orderings: Ponder ("order", names...) and scry N ("scry",
    # "top", names..., "bottom", names...). The names are the decider's own
    # known cards; the label already lists them.
    if kind == "order" and key and key[0] == "order":
        r["type"] = "order"
        r["top"], r["bottom"] = list(key[1:]), []
        return r
    if kind == "order" and key and key[0] == "scry" and "bottom" in key:
        k = key.index("bottom")
        r["type"] = "order"
        r["top"], r["bottom"] = list(key[2:k]), list(key[k + 1 :])
        return r
    # Card choices (sacrifice, discard, search, exile from graveyard, ...):
    # the value is a card, a (verb, card) pair (Highway Robbery) or None;
    # ids only when the card is in plain view.
    r["type"] = kind
    if isinstance(value, tuple) and len(value) == 2 and hasattr(value[1], "uid"):
        r["verb"] = value[0]
        value = value[1]
    if hasattr(value, "uid"):
        _card_ref(value, vis, r)
        if r.get("zone") == "battlefield" and getattr(value, "tapped", False):
            r["tapped"] = True
    return r


def decision_cards(g, d, seat: int) -> list[str]:
    """Names of the decider's own known cards (top of library, hand) that a
    decision talks about (scry, surveil, explore, Delver, Ponder), longest
    first, so the client can show them. Only `seat`'s own knowledge."""
    p = g.players[seat]
    known = [c.name for c in list(p.library)[:5] if seat in c.known_to] + [c.name for c in p.hand]
    text = d.prompt + " | " + " | ".join(o.label for o in d.options)
    seen: list[str] = []
    for n in sorted(set(known), key=len, reverse=True):
        if n in text and not any(n in m for m in seen):
            seen.append(n)
    return seen


def option_refs(decision, state: dict) -> list[dict]:
    """Structured refs for every option of `decision`, filtered by what the
    player sees in `state` (their snapshot at this decision)."""
    vis = visible_ids(state)
    return [option_ref(decision.kind, o.label, o.key, o.value, vis) for o in decision.options]


# ---------------------------------------------------------------------------
# Events: the visible log lines, parsed for animation
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^=== Turn (?P<turn>\d+): player (?P<p>\d) ===$"), "turn"),
    (re.compile(r"^-- (?P<step>\w+)$"), "step"),
    (re.compile(r"^p(?P<p>\d) plays (?P<name>.+)$"), "play"),
    (re.compile(r"^p(?P<p>\d) casts (?P<name>.+) \((?P<mode>[^)]*)\) from (?P<zone>\w+)$"), "cast"),
    (re.compile(r"^p(?P<p>\d) activates (?P<name>.+)#(?P<oid>\d+) for (?P<color>\w)$"), "mana"),
    (re.compile(r"^p(?P<p>\d) activates (?P<name>.+)$"), "activate"),
    (re.compile(r"^p(?P<p>\d) plots (?P<name>.+)$"), "plot"),
    (re.compile(r"^trigger -> stack: (?P<name>.+)$"), "trigger"),
    (re.compile(r"^resolve (?P<name>.+)$"), "resolve"),
    (re.compile(r"^enters: (?P<name>.+)#(?P<oid>\d+) \(p(?P<p>\d)\)$"), "enter"),
    (re.compile(r"^leaves: (?P<name>.+)#(?P<oid>\d+) \(p(?P<p>\d)\) -> (?P<to>\w+)$"), "leave"),
    (re.compile(r"^SBA: (?P<name>.+) dies$"), "dies"),
    (re.compile(r"^p(?P<p>\d) attacks with (?P<list>.*)$"), "attack"),
    (re.compile(r"^p(?P<p>\d) blocks: (?P<map>.*)$"), "block"),
    (re.compile(r"^p(?P<p>\d) discards (?P<name>.+)$"), "discard"),
    (re.compile(r"^p(?P<p>\d) sacrifices (?P<name>.+)#(?P<oid>\d+)$"), "sacrifice"),
    (re.compile(r"^p(?P<p>\d) mulligans \((?P<n>\d+)\)$"), "mulligan"),
    (re.compile(r"^(?P<name>.+) is countered$"), "countered"),
    (re.compile(r"^(?P<name>.+) fizzles .*$"), "fizzle"),
    (re.compile(r"^GAME OVER: winner=(?P<winner>\w+) \((?P<reason>.*)\)$"), "game_over"),
    (re.compile(r"^p(?P<p>\d) concedes$"), "concede"),
    (re.compile(r"^combat: (?P<name>.+)#(?P<oid>\d+) deals (?P<n>\d+) damage to p(?P<p>\d)$"), "hit"),
    (re.compile(r"^combat: (?P<name>.+)#(?P<oid>\d+) deals (?P<n>\d+) damage to (?P<to_name>.+)#(?P<to_oid>\d+)$"), "hit"),
    (re.compile(r"^life: p(?P<p>\d) (?P<old>-?\d+) -> (?P<new>-?\d+)$"), "life"),
]
_INTS = {"p", "turn", "oid", "n", "to_oid", "old", "new"}


def parse_event(line: str) -> dict | None:
    """One log line as {"t": kind, ...}; decision echoes (indented) are None."""
    if line.startswith("  "):
        return None
    for pat, t in _PATTERNS:
        m = pat.match(line)
        if not m:
            continue
        ev = {"t": t}
        for k, v in m.groupdict().items():
            if k == "list":
                ev["oids"] = _label_oids(v)
            elif k == "map":
                ev["pairs"] = [[int(b), int(a)] for b, a in re.findall(r"(\d+): (\d+)", v)]
            else:
                ev[k] = int(v) if k in _INTS else v
        return ev
    return {"t": "note", "text": line}


def parse_events(lines: list[str]) -> list[dict]:
    """The visible log lines of a frame as structured events, in order."""
    return [ev for ev in map(parse_event, lines) if ev is not None]


# ---------------------------------------------------------------------------
# Combat damage and life changes: the engine does not log them, so the live
# layer adds lines (to the player's log and the saved replay) by comparing
# the game before and after each action. Life lines are exact; combat hits
# use the powers before damage (pump effects in the damage step and first
# strike are approximated).
# ---------------------------------------------------------------------------


def status(g) -> dict:
    return {
        "life": [p.life for p in g.players],
        "perms": {c.oid: (c.name, g.power(c) if g.is_creature(c) else 0, c.controller) for c in g.battlefield},
        # toughness left and trample, for splitting a trampler's damage
        "body": {c.oid: (g.toughness(c) - c.damage, g.has(c, "trample")) for c in g.battlefield if g.is_creature(c)},
        "attackers": list(g.attackers),
        "blocks": dict(g.blocks),
        "active": g.active,
    }


_BLOCKS = re.compile(r"^p\d blocks: \{(.*)\}$")


def combat_and_life_lines(before: dict, after: dict, new_lines: list[str]) -> list[str]:
    """Log lines for what happened between two statuses: combat hits (when
    the combat damage step began) and every life change."""
    out = []
    if "-- combat_damage" in new_lines and before["attackers"]:
        blocks = dict(before["blocks"])
        for line in new_lines:
            m = _BLOCKS.match(line)
            if m:
                blocks.update({int(b): int(a) for b, a in re.findall(r"(\d+): (\d+)", m.group(1))})
        perms, defender = before["perms"], 1 - before["active"]
        for a in before["attackers"]:
            if a not in perms:
                continue
            name, power, _ = perms[a]
            mine = [b for b, at in blocks.items() if at == a and b in perms]
            if not mine:
                if power > 0:
                    out.append(f"combat: {name}#{a} deals {power} damage to p{defender}")
                continue
            if len(mine) == 1 and power > 0:
                bname = perms[mine[0]][0]
                trample = before["body"].get(a, (0, False))[1]
                need = max(0, before["body"].get(mine[0], (power, False))[0])
                if trample and power > need:  # lethal to the blocker, the rest tramples over (the default split)
                    out.append(f"combat: {name}#{a} deals {need} damage to {bname}#{mine[0]}")
                    out.append(f"combat: {name}#{a} deals {power - need} damage to p{defender}")
                else:
                    out.append(f"combat: {name}#{a} deals {power} damage to {bname}#{mine[0]}")
            for b in mine:
                bname, bpower, _ = perms[b]
                if bpower > 0:
                    out.append(f"combat: {bname}#{b} deals {bpower} damage to {name}#{a}")
    for p, (old, new) in enumerate(zip(before["life"], after["life"])):
        if old != new:
            out.append(f"life: p{p} {old} -> {new}")
    return out


# ---------------------------------------------------------------------------
# Mana: the auto-pay choice, and which sources a cast would tap
# ---------------------------------------------------------------------------

BASICS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"})


def auto_pay_index(g, d) -> int:
    """The pay_mana option the play client takes when paying automatically:
    floating mana first; then the source that keeps the most options open
    (lands before artifacts before creatures, sources without another {T}
    ability (a Bridge before Twisted Landscape), single-colour sources,
    basics); sacrifices and filters last; ties in option order. The engine offers only
    options that keep the payment completable, so greedy is safe."""
    counts: dict[int, int] = {}
    for o in d.options:
        if o.value[0] != "pool":
            counts[o.value[1].oid] = counts.get(o.value[1].oid, 0) + 1
    best, best_key = 0, None
    for i, o in enumerate(d.options):
        v = o.value
        if v[0] == "pool":
            key = (0, 0, 0, 0, 0, 0, 0, i)
        else:
            card = v[1]
            types = g.types(card)
            kind = 2 if "Creature" in types else 0 if "Land" in types else 1
            other_tap = any(a.tap and a.mana is None and a.zone == "battlefield" for a in card.face.abilities)
            key = (1, int(o.label.startswith("Sacrifice")), int(v[0] == "filter"), kind, int(other_tap), counts[card.oid], int(card.name not in BASICS), i)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def tap_preview(g, index: int, seat: int, limit: int = 60) -> tuple[list[int], list[int]] | None:
    """(tapped, sacrificed): the oids of `seat`'s permanents that taking
    priority option `index` and paying with `auto_pay_index` would tap, and
    those it would sacrifice (a Spawn for mana, an additional cost; other
    choices on the way take their first option). Runs on a copy; None if it
    cannot be simulated."""
    try:
        before = {c.oid for c in g.battlefield if c.controller == seat and not c.tapped}
        mine = {c.oid for c in g.battlefield if c.controller == seat}
        c = g.copy()
        c.step(index)
        for _ in range(limit):
            d = c.decision
            if d is None or d.player != seat or d.kind == "priority":
                break
            c.step(auto_pay_index(c, d) if d.kind == "pay_mana" else 0)
        after = {x.oid: x for x in c.battlefield}
        return sorted(o for o in before if o in after and after[o].tapped), sorted(o for o in mine if o not in after)
    except Exception:  # a preview must never break the game
        return None


PREVIEWED = frozenset({"cast", "activate", "plot"})


def add_tap_previews(g, refs: list[dict], seat: int, vis: dict, max_options: int = 16) -> None:
    """Give each cast/activate/plot ref `taps`: the sources auto-pay would use."""
    import gc

    done = 0
    for i, r in enumerate(refs):
        if r["type"] not in PREVIEWED or done >= max_options:
            continue
        done += 1
        res = tap_preview(g, i, seat)
        if res is not None:
            r["taps"] = [o for o in res[0] if o in vis["oid"]]
            if res[1]:
                r["sacs"] = [o for o in res[1] if o in vis["oid"]]
    if done:
        gc.collect()  # the copies hold reference cycles; native objects must die on this (the game's) thread
