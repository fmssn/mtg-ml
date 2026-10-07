"""Sparse hashed state features and stable action keys.

Inspired by MageZero's StateEncoder: a state is a set of string features
(hierarchical: a permanent contributes its name, its name+state and coarse
type-level features), hashed into a fixed index space. State and entity
features work from `observe()`, so encoders never see hidden information;
option previews (`option_preview`) read the engine, but only public objects
and the decider's own cards and mana. Reference: docs/features.md.

The hashed state is a set, so anything that matters as a number has to be
spelled out as features: counts of identical objects (`_counted`) and
thermometers for turn, life, hand and graveyard sizes and board power
(`_thermo`). Before 2026-10 duplicates collapsed, so the network could not
tell one Forest from three, and it saw neither the turn nor exact life.
"""

from __future__ import annotations

import re
import zlib

from .engine.view import PLOTTED, observe

DEFAULT_DIM = 1 << 16
# Feature-set versions. A policy is featurized with the version it was trained
# on (`PolicyNet.config["features"]`, absent = 1), so adding features never
# feeds a trained model hashed rows it has not learned.
#   1: the set up to 2026-10-06
#   2: + lethal / readiness, known library positions, skip_untap, stack
#      targets and X, option previews (docs/features.md)
#   3: + `opp:deck:{deck}` (the opponent not on its seat's usual deck), so
#      one network can play several matchups; identical to 2 on jund_blue
FEATURES = 3  # the latest; what new runs train on
FEATURE_VERSIONS = (1, 2, 3)


def check_features(features: int) -> int:
    if features not in FEATURE_VERSIONS:
        raise ValueError(f"unknown feature-set version {features!r} (known: {FEATURE_VERSIONS})")
    return features
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


def state_features(game, viewer: int, features: int = FEATURES) -> list[str]:
    """Global features of what `viewer` can see. Numbers (turn, life, hand
    and graveyard sizes, board power) are thermometers; cards in hand,
    graveyard and exile and the per-type board counts keep their
    multiplicity through `_counted`. Permanents and stack items themselves
    are entities (`entity_features`). `features`: the feature-set version."""
    check_features(features)
    if getattr(game, "NATIVE", False):
        return game.state_features(viewer, features)
    o = observe(game, viewer)
    f = [f"step:{o['step']}", f"active:{o['active']}", f"postboard:{o['match_game'] > 1}"]
    deck = game.deck_names[viewer]
    if deck is not None and features >= 2:  # not the seat's usual deck (match.deck_names)
        f.append(f"self:deck:{deck}")
    opp_deck = game.deck_names[1 - viewer]
    if opp_deck is not None and features >= 3:
        f.append(f"opp:deck:{opp_deck}")
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
        for name in s["exile"]:  # set 1 predates the plotted mark of view._exiled
            raw.append(f"{side}:exile:{name if features >= 2 else name.removesuffix(PLOTTED)}")
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
    if features < 2:
        return f + _counted(raw)
    f += _board_features(o, viewer, game)
    for side in ("self", "opponent"):
        n = o[side]["library_count"]
        for pos, name in o[side]["library_known"]:
            if pos < KNOWN_POS_CAP:
                f.append(f"{side}:known_library:{pos}:{name}")
            elif pos == n - 1:
                f.append(f"{side}:known_library:bottom:{name}")
    return f + _counted(raw)


KNOWN_POS_CAP = 8  # known library cards deeper than this (but not at the bottom) get no position feature


def _ready(p: dict, active: bool) -> bool:
    """Could this creature attack in its controller's current (if active)
    or next turn? Active side: untapped and not summoning sick. Other side:
    sickness wears off and it untaps, unless it is tapped and skips its next
    untap step (Sleep of the Dead)."""
    if active:
        return not p["tapped"] and not p["sick"]
    return not (p["tapped"] and p["skip_untap"] > 0)


def _can_block(blocker: dict, attacker: dict) -> bool:
    """`Game._can_block` on observed permanents."""
    return "flying" not in attacker["keywords"] or "flying" in blocker["keywords"] or "reach" in blocker["keywords"]


def _board_features(o: dict, viewer: int, game) -> list[str]:
    """Lethal and readiness: per side, the power that could attack (`_ready`),
    the part of it no untapped creature of the defender can block, the
    defender's untapped creatures (potential blockers), whether either is
    lethal against the defender's life, and untapped mana sources."""
    f = []
    creatures = [p for p in o["battlefield"] if p["power"] is not None]
    for side, other, p_idx in (("self", "opponent", viewer), ("opponent", "self", 1 - viewer)):
        active = o["active"] == side
        attackers = [p for p in creatures if p["controller"] == side and _ready(p, active)]
        blockers = [p for p in creatures if p["controller"] == other and not p["tapped"]]
        ready = sum(max(p["power"], 0) for p in attackers)
        evasive = sum(max(p["power"], 0) for p in attackers if not any(_can_block(b, p) for b in blockers))
        life = o[other]["life"]
        f += _thermo(f"{side}:ready_power", ready, POWER_STEPS)
        f += _thermo(f"{side}:ready_evasive_power", evasive, POWER_STEPS)
        f += _thermo(f"{side}:potential_blockers", sum(1 for p in creatures if p["controller"] == side and not p["tapped"]))
        if ready > 0 and ready >= life:
            f.append(f"{side}:lethal_on_board")
        if evasive > 0 and evasive >= life:
            f.append(f"{side}:evasive_lethal_on_board")
        f += _thermo(f"{side}:untapped_mana", len(game.mana_sources(p_idx)))
    return f


ENT_STEPS = (*range(1, 9), 10, 12, 15)
MAX_ENTITIES = 64  # permanents and stack items beyond this are left out


def entity_features(game, viewer: int, features: int = FEATURES) -> tuple[list[list[str]], dict[int, int]]:
    """One feature list per object the cards can point at: every permanent
    (battlefield order), then the stack from the top. Returns the lists and
    {oid or stack id: entity index}; the two id spaces are shared, so labels
    like `Tolarian Terror#12` resolve unambiguously. Everything about one
    object stays together, so two Tolarian Terrors, one tapped and damaged,
    are two different entities rather than a bag of shared name features."""
    check_features(features)
    if getattr(game, "NATIVE", False):
        return game.entity_features(viewer, features)
    o = observe(game, viewer)
    ents, index = [], {}
    v2 = features >= 2
    targeted = _targeted_by(o["stack"]) if v2 else {}
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
        if v2 and p["skip_untap"] > 0:
            e.append(f"e:skip_untap:{p['skip_untap']}")
        e += targeted.get(p["oid"], [])
        index[p["oid"]] = len(ents)
        ents.append(e)
    for i, it in enumerate(reversed(o["stack"])):
        index[it["sid"]] = len(ents)
        e = ["e:stack", f"e:name:{it['name']}", f"e:ctrl:{it['controller']}", f"e:stack_pos:{min(i, 3)}", f"e:stack_kind:{it['kind']}"]
        if v2 and it["x"] > 0:
            e += _thermo("e:x", it["x"], ENT_STEPS)
        for label in it["targets"] if v2 else ():
            kind, rel = _parse_target(label)
            e.append(f"e:targets:{kind}:{rel}")
        e += targeted.get(it["sid"], [])
        ents.append(e)
    if len(ents) > MAX_ENTITIES:
        ents = ents[:MAX_ENTITIES]
        index = {k: v for k, v in index.items() if v < MAX_ENTITIES}
    return ents, index


def _parse_target(label: str) -> tuple[str, str]:
    """(kind, relation) of an observed target label: `player 1 (self)`,
    `spell Counterspell#7 (opponent)` or `Swamp#3 (self)`."""
    rel = label.rsplit("(", 1)[1].rstrip(")")
    if label.startswith("player "):
        return "player", rel
    if label.startswith("spell "):
        return "spell", rel
    return "perm", rel


def _targeted_by(stack: list[dict]) -> dict[int, list[str]]:
    """{object id: features} for every permanent or spell a stack item
    targets (stack from the bottom): who targets it and with what."""
    out: dict[int, list[str]] = {}
    for it in stack:
        for label in it["targets"]:
            m = _ID.search(label)
            if m is not None and not label.startswith("player "):
                out.setdefault(int(m.group(1)), []).extend((f"e:targeted_by:{it['controller']}", f"e:targeted_by:{it['controller']}:{it['name']}"))
    return out


def option_object_ids(option) -> list[int]:
    """Ids of the objects an option is about: every `Name#id` in its label
    (attackers, blockers, mana sources, sacrifices, targets, damage
    assignment), plus the source of an activated or mana ability."""
    ids = [int(m) for m in _ID.findall(option.label)]
    v = option.value
    if isinstance(v, tuple) and len(v) == 3 and v[0] in ("activate", "mana"):
        ids.append(v[1].oid)
    return ids


MANA_LEFT_CAP = 8
PREVIEW_COLORS = ("W", "U", "B", "R", "G")


def _ops(effect) -> tuple:
    return getattr(effect, "ops", ()) if effect is not None else ()


def _dies_to(game, c, n: int, deathtouch: bool) -> bool:
    """Would `n` more damage destroy creature `c` (SBA 704.5g/h)?"""
    if n <= 0 or game.has(c, "indestructible"):
        return False
    return deathtouch or c.damage + n >= game.toughness(c)


def _mana_preview(game, player: int, cost, sac_filter, exclude: set[int]) -> list[str]:
    """Mana left after paying `cost`: untapped mana sources plus floating
    mana minus its mana value (payment is a separate decision, so this is an
    estimate), and every colour that could still be produced afterwards
    (exact: `cost` plus one mana of that colour is payable)."""
    from .engine.mana import ManaCost, RemainingCost

    avail = sum(game.players[player].pool.values()) + len(game.mana_sources(player, exclude))
    f = [f"pv:mana_left_after:{min(max(avail - cost.mana_value, 0), MANA_LEFT_CAP)}"]
    for col in PREVIEW_COLORS:
        if game._cost_feasible(player, RemainingCost.of(cost.plus(ManaCost(0, ((col, 1),)))), sac_filter, exclude):
            f.append(f"pv:colors_left:{col}")
    return f


def _kills_preview(game, player: int, ops, source) -> list[str]:
    """Creatures the `damage_each_creature` ops would destroy, per side (an
    op whose amount is X, chosen later, is left out)."""
    dmg: dict[int, int] = {}
    for op in ops:
        if op["op"] != "damage_each_creature" or op.get("x"):
            continue
        without = op.get("without")
        for c in game.battlefield:
            if op.get("whose") == "opponent" and c.controller == player:
                continue
            if game.is_creature(c) and not (without and game.has(c, without)):
                dmg[c.oid] = dmg.get(c.oid, 0) + op["n"]
    if not dmg:
        return []
    deathtouch = game.has(source, "deathtouch")
    kills = {"self": 0, "opp": 0}
    for c in game.battlefield:
        if c.oid in dmg and _dies_to(game, c, dmg[c.oid], deathtouch):
            kills["self" if c.controller == player else "opp"] += 1
    return _thermo("pv:kills_opp", kills["opp"]) + _thermo("pv:kills_self", kills["self"]) + ["pv:kills_none"] * (kills["opp"] + kills["self"] == 0)


def _item_cost(game, item):
    """(cost, sacrifice filter, excluded source oids) of the stack item being
    put on the stack (targets are chosen before costs are paid)."""
    from .engine.mana import ManaCost

    if item.kind == "spell":
        card = item.card
        base = game._mode_cost(card, item.method)
        return base.with_x(item.x).reduced(game._cost_reduction(item.controller, card)), card.face.additional_sac, set()
    src = item.source
    for ab in src.face.abilities:
        if f"{src.name}: {ab.name}" == item.name:
            return ab.cost, ab.sac_other, {src.oid} if ab.tap else set()
    return ManaCost(), None, set()


def _target_preview(game, player: int, ref) -> list[str]:
    """Ward and lethal damage for a target choice of the top stack item."""
    if not game.stack or ref[0] == "stack":
        return []
    item = game.stack[-1]
    f = []
    if ref[0] == "perm":
        c = game.perm(ref[1])
        if c is None:
            return []
        if c.face.ward and c.controller != player:
            f.append(f"pv:target_ward:{c.face.ward}")
            cost, sac, exclude = _item_cost(game, item)
            from .engine.mana import ManaCost, RemainingCost

            if game._cost_feasible(player, RemainingCost.of(cost.plus(ManaCost(c.face.ward))), sac, exclude):
                f.append("pv:ward_payable")
    source = item.card if item.kind == "spell" else item.source
    landfall = game.players[item.controller].landfall_turn == game.turn
    for op in _ops(item.effect):
        if op["op"] != "damage_target" or op.get("index", 0) != len(item.targets):  # the target being chosen
            continue
        n = op["n_landfall"] if "n_landfall" in op and landfall else op["n"]
        if ref[0] == "player":
            lethal = n >= game.players[ref[1]].life
        else:
            lethal = game.is_creature(c) and _dies_to(game, c, n, source is not None and game.has(game.live(source) or source, "deathtouch"))
        if lethal:
            f.append("pv:damage_lethal_to_target")
    return f


def option_preview(game, player: int, i: int, features: int = FEATURES) -> list[str]:
    """Engine-computed effects of taking option `i` of the current decision,
    from the current state without changing it (`pv:` tokens): creatures a
    sweeper ability kills per side, ward and lethal damage on a target, mana
    and colours left after a cast or activation. Feature set 2 and up."""
    if check_features(features) < 2:
        return []
    if getattr(game, "NATIVE", False):
        return game.option_preview(player, i)
    kind = game.decision.kind
    v = game.decision.options[i].value
    if kind == "target" and isinstance(v, tuple):
        return _target_preview(game, player, v)
    if kind != "priority" or not isinstance(v, tuple) or v[0] not in ("cast", "activate"):
        return []
    # A determinized copy re-deals hidden cards under pending options, so an
    # option may name an ability, mode or cost the card no longer has.
    card = v[1]
    if v[0] == "activate":
        if v[2] >= len(card.face.abilities):
            return []
        ab = card.face.abilities[v[2]]
        exclude = {card.oid} if ab.tap else set()
        return _kills_preview(game, player, _ops(ab.effect), card) + _mana_preview(game, player, ab.cost, ab.sac_other, exclude)
    mode, choice = v[2], (v[3] if len(v) > 3 else None)
    d = card.face
    base = game._mode_cost(card, mode)
    if base is None or (choice is not None and choice >= len(d.modes)):
        return []
    effect = d.overload_effect if mode == "overload" else d.effect if choice is None else d.modes[choice].effect
    cost = base.with_x(0).reduced(game._cost_reduction(player, card))
    return _kills_preview(game, player, _ops(effect), card) + _mana_preview(game, player, cost, d.additional_sac, set())


def hash_feature(feature: str, dim: int = DEFAULT_DIM) -> int:
    return zlib.crc32(feature.encode()) % dim


def encode_state(game, viewer: int, dim: int = DEFAULT_DIM, features: int = FEATURES) -> list[int]:
    """Sorted unique feature indices (a multi-hot sparse vector)."""
    return sorted({hash_feature(x, dim) for x in state_features(game, viewer, features)})


def action_keys(game) -> list[tuple]:
    """Stable, id-free keys of the current options (for policy vocabularies)."""
    return [(game.decision.kind,) + o.key for o in game.legal_options()]
