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
    # Card choices (sacrifice, discard, search, exile from graveyard, ...):
    # the value is a card or None; ids only when the card is in plain view.
    r["type"] = kind
    if hasattr(value, "uid"):
        _card_ref(value, vis, r)
    return r


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
]
_INTS = {"p", "turn", "oid", "n"}


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
# Mana: the auto-pay choice, and which sources a cast would tap
# ---------------------------------------------------------------------------

BASICS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"})


def auto_pay_index(g, d) -> int:
    """The pay_mana option the play client takes when paying automatically:
    floating mana first; then the source that keeps the most options open
    (lands before artifacts before creatures, single-colour sources, basics);
    sacrifices and filters last; ties in option order. The engine offers only
    options that keep the payment completable, so greedy is safe."""
    counts: dict[int, int] = {}
    for o in d.options:
        if o.value[0] != "pool":
            counts[o.value[1].oid] = counts.get(o.value[1].oid, 0) + 1
    best, best_key = 0, None
    for i, o in enumerate(d.options):
        v = o.value
        if v[0] == "pool":
            key = (0, 0, 0, 0, 0, 0, i)
        else:
            card = v[1]
            types = g.types(card)
            kind = 2 if "Creature" in types else 0 if "Land" in types else 1
            key = (1, int(o.label.startswith("Sacrifice")), int(v[0] == "filter"), kind, counts[card.oid], int(card.name not in BASICS), i)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def tap_preview(g, index: int, seat: int, limit: int = 60) -> list[int] | None:
    """The oids of `seat`'s permanents that taking priority option `index`
    and paying with `auto_pay_index` would tap (other choices on the way take
    their first option). Runs on a copy; None if it cannot be simulated."""
    try:
        before = {c.oid for c in g.battlefield if c.controller == seat and not c.tapped}
        c = g.copy()
        c.step(index)
        for _ in range(limit):
            d = c.decision
            if d is None or d.player != seat or d.kind == "priority":
                break
            c.step(auto_pay_index(c, d) if d.kind == "pay_mana" else 0)
        return sorted(x.oid for x in c.battlefield if x.oid in before and x.tapped)
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
        taps = tap_preview(g, i, seat)
        if taps is not None:
            r["taps"] = [o for o in taps if o in vis["oid"]]
    if done:
        gc.collect()  # the copies hold reference cycles; native objects must die on this (the game's) thread
