"""Sparse hashed state features and stable action keys.

Inspired by MageZero's StateEncoder: a state is a set of string features
(hierarchical: a permanent contributes its name, its name+state and coarse
type-level features), hashed into a fixed index space. Everything here works
from `observe()`, so encoders never see hidden information.

The hashed state is a set, so anything that matters as a number has to be
spelled out as features: counts of identical objects (`_counted`) and
thermometers for turn, life, hand and graveyard sizes and board power
(`_thermo`). Before 2026-10 duplicates collapsed, so the network could not
tell one Forest from three, and it saw neither the turn nor exact life.
"""

from __future__ import annotations

import re
import zlib

from .engine.view import observe

DEFAULT_DIM = 1 << 16
_ID = re.compile(r"#(\d+)")


def _bucket(n: int, edges=(0, 1, 2, 3, 5, 8, 13, 20)) -> int:
    b = 0
    for i, e in enumerate(edges):
        if n >= e:
            b = i
    return b


COUNT_CAP = 8  # copies of one object feature that stay distinguishable


# Thermometer steps: every value where single steps matter, coarser above.
LIFE_STEPS = (*range(1, 11), 12, 14, 16, 18, 20, 25)
TURN_STEPS = (*range(1, 13), 14, 16, 18, 20)
POWER_STEPS = (*range(1, 11), 12, 15, 20)
COUNT_STEPS = tuple(range(1, 11))
GY_STEPS = (*range(1, 11), 12, 15)


def _thermo(name: str, n: int, steps: tuple = COUNT_STEPS) -> list[str]:
    """`name>=k` for every step k <= n: a number as a set of features, so a
    summed embedding grows with it and nothing is lost to deduplication."""
    return [f"{name}>={k}" for k in steps if n >= k]


def _counted(raw: list[str]) -> list[str]:
    """Each distinct feature once (first-occurrence order), followed by
    `#2` .. `#n` for its extra copies (n capped at COUNT_CAP). Three
    untapped Forests stay three Forests after the hashed set is built."""
    n: dict[str, int] = {}
    for x in raw:
        n[x] = n.get(x, 0) + 1
    out = []
    for x, c in n.items():
        out.append(x)
        out.extend(f"{x}#{k}" for k in range(2, min(c, COUNT_CAP) + 1))
    return out


def state_features(game, viewer: int) -> list[str]:
    """Global features of what `viewer` can see. Numbers (turn, life, hand
    and graveyard sizes, board power) are thermometers; cards in hand,
    graveyard and exile and the per-type board counts keep their
    multiplicity through `_counted`. Permanents and stack items themselves
    are entities (`entity_features`)."""
    if getattr(game, "NATIVE", False):
        return game.state_features(viewer)
    o = observe(game, viewer)
    f = [f"step:{o['step']}", f"active:{o['active']}", f"postboard:{o['match_game'] > 1}"]
    f += _thermo("turn", o["turn"], TURN_STEPS)
    if o["lands_played"] is not None:
        f.append(f"land_played:{o['lands_played'] > 0}")
    raw = []
    for side in ("self", "opponent"):
        s = o[side]
        f += _thermo(f"{side}:life", s["life"], LIFE_STEPS)
        f.append(f"{side}:library:{_bucket(s['library_count'])}")
        f.append(f"{side}:mulligans:{s['mulligans']}")
        f += _thermo(f"{side}:gy_count", len(s["graveyard"]), GY_STEPS)
        f += _thermo(f"{side}:exile_count", len(s["exile"]))
        for name in s["graveyard"]:
            raw.append(f"{side}:gy:{name}")
        for name in s["exile"]:
            raw.append(f"{side}:exile:{name}")
        for c, n in s["pool"].items():
            f.append(f"{side}:pool:{c}:{n}")
        for _, name in s["library_known"][:3]:
            raw.append(f"{side}:known_library:{name}")
    f += _thermo("self:hand_count", len(o["self"]["hand"]))
    for name in o["self"]["hand"]:
        raw.append(f"self:hand:{name}")
    f += _thermo("opponent:hand_count", o["opponent"]["hand_count"])
    for name in o["opponent"]["hand_known"]:
        raw.append(f"opponent:hand_known:{name}")
    power = {"self": 0, "opponent": 0}
    for p in o["battlefield"]:
        side = p["controller"]
        for t in p["types"]:
            raw.append(f"{side}:bf_type:{t}")
            if not p["tapped"]:
                raw.append(f"{side}:bf_type:{t}:untapped")
        if p["power"] is not None:
            power[side] += max(p["power"], 0)
    for side in ("self", "opponent"):
        f += _thermo(f"{side}:power", power[side], POWER_STEPS)
    f += _thermo("stack_count", len(o["stack"]))
    return f + _counted(raw)


ENT_STEPS = (*range(1, 9), 10, 12, 15)
MAX_ENTITIES = 64  # permanents and stack items beyond this are left out


def entity_features(game, viewer: int) -> tuple[list[list[str]], dict[int, int]]:
    """One feature list per object the cards can point at: every permanent
    (battlefield order), then the stack from the top. Returns the lists and
    {oid or stack id: entity index}; the two id spaces are shared, so labels
    like `Tolarian Terror#12` resolve unambiguously. Everything about one
    object stays together, so two Tolarian Terrors, one tapped and damaged,
    are two different entities rather than a bag of shared name features."""
    if getattr(game, "NATIVE", False):
        return game.entity_features(viewer)
    o = observe(game, viewer)
    ents, index = [], {}
    for p in o["battlefield"]:
        e = [f"e:name:{p['name']}", f"e:ctrl:{p['controller']}"]
        e += [f"e:type:{t}" for t in p["types"]]
        e += [f"e:kw:{k}" for k in p["keywords"]]
        for flag in ("tapped", "sick", "attacking", "token"):
            if p[flag]:
                e.append(f"e:{flag}")
        if p["blocking"] is not None:
            e.append("e:blocking")
        if p["attached_to"] is not None:
            e.append("e:attached")
        if p["power"] is not None:
            e += _thermo("e:power", p["power"], ENT_STEPS)
            e += _thermo("e:toughness", p["toughness"], ENT_STEPS)
        e += _thermo("e:damage", p["damage"], ENT_STEPS)
        if p["counters"] > 0:
            e += _thermo("e:counters", p["counters"], ENT_STEPS)
        elif p["counters"] < 0:
            e.append(f"e:counters:{p['counters']}")
        index[p["oid"]] = len(ents)
        ents.append(e)
    for i, it in enumerate(reversed(o["stack"])):
        index[it["sid"]] = len(ents)
        ents.append(["e:stack", f"e:name:{it['name']}", f"e:ctrl:{it['controller']}", f"e:stack_pos:{min(i, 3)}", f"e:stack_kind:{it['kind']}"])
    if len(ents) > MAX_ENTITIES:
        ents = ents[:MAX_ENTITIES]
        index = {k: v for k, v in index.items() if v < MAX_ENTITIES}
    return ents, index


def option_object_ids(option) -> list[int]:
    """Ids of the objects an option is about: every `Name#id` in its label
    (attackers, blockers, mana sources, sacrifices, targets, damage
    assignment), plus the source of an activated or mana ability."""
    ids = [int(m) for m in _ID.findall(option.label)]
    v = option.value
    if isinstance(v, tuple) and len(v) == 3 and v[0] in ("activate", "mana"):
        ids.append(v[1].oid)
    return ids


def hash_feature(feature: str, dim: int = DEFAULT_DIM) -> int:
    return zlib.crc32(feature.encode()) % dim


def encode_state(game, viewer: int, dim: int = DEFAULT_DIM) -> list[int]:
    """Sorted unique feature indices (a multi-hot sparse vector)."""
    return sorted({hash_feature(x, dim) for x in state_features(game, viewer)})


def action_keys(game) -> list[tuple]:
    """Stable, id-free keys of the current options (for policy vocabularies)."""
    return [(game.decision.kind,) + o.key for o in game.legal_options()]
