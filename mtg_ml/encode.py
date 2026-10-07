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

from .engine.cards import op_names
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
#   4: + combat relations (who blocks whom, blocked / unblocked attackers,
#      block previews), incoming combat damage, choose_x previews and an X
#      option -> spell pointer, mana colours (sources, hand needs, colour
#      previews on land plays, basic searches and mana payment)
#   5: + cards described by what they do: shape tokens from the card spec on
#      permanents and stack items, the decider's own hand cards as entities,
#      cast / play / plot options pointing at them (docs/features.md)
FEATURES = 5  # the latest; what new runs train on
FEATURE_VERSIONS = (1, 2, 3, 4, 5)


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
    if features >= 4:
        f += _combat_features(game, viewer) + _colour_features(game, viewer)
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


def _attackers(game) -> list:
    """The attacking creatures still on the battlefield, in declaration order."""
    return [c for c in (game.perm(a) for a in game.attackers) if c is not None]


def _blockers_of(game, aoid: int) -> list:
    """The creatures still on the battlefield blocking attacker `aoid`."""
    return [c for c in (game.perm(b) for b, a in game.blocks.items() if a == aoid) if c is not None]


def _unblocked_power(game, skip: int | None = None) -> int:
    """Power of the attackers not blocked (`Game.blocked`: an attacker stays
    blocked when its blockers leave), leaving out attacker `skip`."""
    return sum(max(game.power(c), 0) for c in _attackers(game) if c.oid not in game.blocked and c.oid != skip)


def _combat_features(game, viewer: int) -> list[str]:
    """Set 4, while creatures attack: the attacking side's total and unblocked
    power (from the blocks declared so far), whether the unblocked part is
    lethal to the defender, and the defender's life after it. `_board_features` counts only creatures that
    could still attack, so during combat it misses the attackers."""
    att = _attackers(game)
    if not att:
        return []
    side = "self" if game.active == viewer else "opponent"
    total = sum(max(game.power(c), 0) for c in att)
    unblocked = _unblocked_power(game)
    life = game.players[1 - game.active].life
    f = _thermo(f"{side}:attacking_power", total, POWER_STEPS) + _thermo(f"{side}:unblocked_power", unblocked, POWER_STEPS)
    if unblocked > 0 and unblocked >= life:
        f.append(f"{side}:incoming_lethal")
    other = "opponent" if side == "self" else "self"
    return f + _thermo(f"{other}:life_after_unblocked", life - unblocked, LIFE_STEPS)


def _mana_colors(face) -> tuple[str, ...]:
    """Colours (WUBRG) the card's mana ability makes; every supported permanent has at most one."""
    for ab in face.abilities:
        if ab.mana is not None:
            return tuple(c for c in PREVIEW_COLORS if c in ab.mana)
    return ()


def _colour_counts(game, viewer: int) -> tuple[dict[str, int], set[str]]:
    """({colour: permanents of `viewer` whose mana ability makes it, tapped
    or not}, {colours in the mana costs of the cards in `viewer`'s hand})."""
    sources = dict.fromkeys(PREVIEW_COLORS, 0)
    for c in game.battlefield:
        if c.controller == viewer:
            for col in _mana_colors(c.face):
                sources[col] += 1
    needs = {k for c in game.players[viewer].hand for k, n in c.face.cost.colored if n > 0 and k in sources}
    return sources, needs


def _colour_features(game, viewer: int) -> list[str]:
    sources, needs = _colour_counts(game, viewer)
    f = []
    for col in PREVIEW_COLORS:
        f += _thermo(f"self:sources:{col}", sources[col])
    f += [f"self:hand_needs:{col}" for col in PREVIEW_COLORS if col in needs]
    f += [f"self:hand_missing:{col}" for col in PREVIEW_COLORS if col in needs and sources[col] == 0]
    return f


def _combat_entity(game, c, slots: dict[int, int]) -> list[str]:
    """Set 4 combat relations of permanent `c`. An attacker: blocked or not,
    how many creatures block it, their summed power, and whether that kills
    it. A blocker: the attacker it blocks (name, power and toughness, whether
    the two kill each other one on one). `e:attack_slot:{j}` and
    `e:blocking:slot:{j}` tie a blocker to its attacker by declaration order,
    so blocks on two same-name attackers are told apart."""
    e = []
    if c.oid in slots:
        e.append(f"e:attack_slot:{slots[c.oid]}")
        e.append("e:blocked" if c.oid in game.blocked else "e:unblocked")
        blockers = _blockers_of(game, c.oid)
        e += _thermo("e:blockers", len(blockers), ENT_STEPS)
        power = sum(max(game.power(b), 0) for b in blockers)
        e += _thermo("e:block_power", power, ENT_STEPS)
        if _dies_to(game, c, power, any(game.has(b, "deathtouch") for b in blockers)):
            e.append("e:block_lethal")
    a = game.perm(game.blocks[c.oid]) if c.oid in game.blocks else None
    if a is not None:
        if a.oid in slots:
            e.append(f"e:blocking:slot:{slots[a.oid]}")
        e.append(f"e:blocking:name:{a.name}")
        e += _thermo("e:blocking:power", game.power(a), ENT_STEPS)
        e += _thermo("e:blocking:toughness", game.toughness(a), ENT_STEPS)
        if _dies_to(game, a, game.power(c), game.has(c, "deathtouch")):
            e.append("e:blocking:kills")
        if _dies_to(game, c, game.power(a), game.has(a, "deathtouch")):
            e.append("e:blocking:dies")
    return e


ENT_STEPS = (*range(1, 9), 10, 12, 15)
MAX_ENTITIES = 64  # permanents, stack items and hand cards beyond this are left out (in that order)


def entity_features(game, viewer: int, features: int = FEATURES) -> tuple[list[list[str]], dict[int, int]]:
    """One feature list per object the cards can point at: every permanent
    (battlefield order), then the stack from the top, then (set 5 on) the
    viewer's own hand cards in hand order. Returns the lists and
    {oid or stack id: entity index}; the two id spaces are shared, so labels
    like `Tolarian Terror#12` resolve unambiguously. Everything about one
    object stays together, so two Tolarian Terrors, one tapped and damaged,
    are two different entities rather than a bag of shared name features."""
    check_features(features)
    if getattr(game, "NATIVE", False):
        return game.entity_features(viewer, features)
    o = observe(game, viewer)
    ents, index = [], {}
    v2, v5 = features >= 2, features >= 5
    targeted = _targeted_by(o["stack"]) if v2 else {}
    v4 = features >= 4
    slots = {a: j for j, a in enumerate(game.attackers)} if v4 else {}
    for p, card in zip(o["battlefield"], game.battlefield):
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
        if v4:
            c = game.perm(p["oid"])
            e += [f"e:produces:{col}" for col in _mana_colors(c.face)]
            e += _combat_entity(game, c, slots)
        if v5:
            e += card.face.shape
        index[p["oid"]] = len(ents)
        ents.append(e)
    for i, (it, item) in enumerate(zip(reversed(o["stack"]), reversed(game.stack))):
        index[it["sid"]] = len(ents)
        e = ["e:stack", f"e:name:{it['name']}", f"e:ctrl:{it['controller']}", f"e:stack_pos:{min(i, 3)}", f"e:stack_kind:{it['kind']}"]
        if v2 and it["x"] > 0:
            e += _thermo("e:x", it["x"], ENT_STEPS)
        for label in it["targets"] if v2 else ():
            kind, rel = _parse_target(label)
            e.append(f"e:targets:{kind}:{rel}")
        e += targeted.get(it["sid"], [])
        if v5:
            if item.kind == "spell":
                e += item.card.face.shape
            e += [f"e:res:op:{op}" for op in op_names(_ops(item.effect))]
        ents.append(e)
    if v5:
        _hand_entities(game, viewer, ents, index)
    if len(ents) > MAX_ENTITIES:
        ents = ents[:MAX_ENTITIES]
        index = {k: v for k, v in index.items() if v < MAX_ENTITIES}
    return ents, index


def _hand_entities(game, viewer: int, ents: list, index: dict) -> None:
    """Feature set 5: the viewer's own hand cards as entities (never the
    opponent's: its size stays a state feature), with the card's printed
    types, keywords, P/T and shape tokens, and `e:castable` when the viewer
    is at a priority decision with an option to cast it from hand."""
    d = game.decision
    castable = set()
    if d is not None and d.player == viewer and d.kind == "priority":
        castable = {o.key[1] for o in d.options if o.key[0] == "cast" and o.key[2] == "hand"}
    for c in game.players[viewer].hand:
        f = c.face
        e = ["e:zone:hand", f"e:name:{c.name}", "e:ctrl:self"]
        e += [f"e:type:{t}" for t in sorted(f.types)]
        e += [f"e:kw:{k}" for k in sorted(f.keywords)]
        if f.power is not None:
            e += _thermo("e:power", f.power, ENT_STEPS)
            e += _thermo("e:toughness", f.toughness, ENT_STEPS)
        e += f.shape
        if c.name in castable:
            e.append("e:castable")
        index[c.oid] = len(ents)
        ents.append(e)


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


def option_object_ids(option, game=None, kind: str = "", features: int = 1) -> list[int]:
    """Ids of the objects an option is about: every `Name#id` in its label
    (attackers, blockers, mana sources, sacrifices, targets, damage
    assignment), plus the source of an activated or mana ability and the
    card a cast, land or plot option is about (an entity only while in the
    decider's hand, feature set 5 on); in set 4 also the spell or ability on
    top of the stack for a choose_x option (needs `game` and the decision
    `kind`)."""
    ids = [int(m) for m in _ID.findall(option.label)]
    v = option.value
    if isinstance(v, tuple) and len(v) == 3 and v[0] in ("activate", "mana"):
        ids.append(v[1].oid)
    elif features >= 5 and isinstance(v, tuple) and len(v) >= 2 and v[0] in ("cast", "land", "plot"):
        ids.append(v[1].oid)  # the hand card (other zones' cards are no entities)
    if features >= 4 and kind == "choose_x" and game.stack:
        ids.append(game.stack[-1].sid)
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


def _item_cost(game, item, x: int | None = None):
    """(cost, sacrifice filter, excluded source oids) of the stack item being
    put on the stack (targets are chosen before costs are paid), with X =
    `x` (default: the item's)."""
    from .engine.mana import ManaCost

    if item.kind == "spell":
        card = item.card
        base = game._mode_cost(card, item.method)
        return base.with_x(item.x if x is None else x).reduced(game._cost_reduction(item.controller, card)), card.face.additional_sac, set()
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


def _block_preview(game, player: int, label: str, attacker) -> list[str]:
    """Set 4, a declare_blocker option (blocker: the first id in the label;
    `attacker` None = no block): whether the attacker is already blocked,
    whether it dies to its blockers plus this one, whether it kills this
    blocker, and the unblocked damage left if no further blocks follow."""
    f = []
    skip = None
    if attacker is not None:
        b = game.perm(int(_ID.search(label).group(1)))
        blockers = _blockers_of(game, attacker.oid)
        if attacker.oid in game.blocked:
            f.append("pv:attacker_already_blocked")
        power = sum(max(game.power(x), 0) for x in blockers) + max(game.power(b), 0)
        if _dies_to(game, attacker, power, any(game.has(x, "deathtouch") for x in [*blockers, b])):
            f.append("pv:attacker_dies")
        if _dies_to(game, b, game.power(attacker), game.has(attacker, "deathtouch")):
            f.append("pv:blocker_dies")
        skip = attacker.oid
    left = _unblocked_power(game, skip)
    f += _thermo("pv:unblocked_damage_left", left, POWER_STEPS)
    if left > 0 and left >= game.players[player].life:
        f.append("pv:lethal_left")
    return f


def _x_preview(game, player: int, x: int) -> list[str]:
    """Set 4, a choose_x option: X as a thermometer, whether it is the largest
    X offered, mana and colours left after paying the cost with this X (the
    item on top of the stack), and the power a creature entering with X
    +1/+1 counters would have."""
    f = _thermo("pv:x", x)
    if x == max(o.value for o in game.decision.options):
        f.append("pv:x_is_max")
    if not game.stack:
        return f
    item = game.stack[-1]
    cost, sac, exclude = _item_cost(game, item, x)
    f += _mana_preview(game, player, cost, sac, exclude)
    if item.kind == "spell" and item.card.face.etb_x_counters:
        f += _thermo("pv:enters_power", (item.card.face.power or 0) + x, ENT_STEPS)
    return f


def _adds_colour_preview(game, player: int, face) -> list[str]:
    """Set 4, playing or fetching a land: the colours it makes, and whether
    one of them is needed by the hand and made by nothing the player controls."""
    cols = _mana_colors(face)
    if not cols:
        return []
    sources, needs = _colour_counts(game, player)
    f = [f"pv:adds_color:{col}" for col in cols]
    if any(col in needs and sources[col] == 0 for col in cols):
        f.append("pv:adds_missing_color")
    return f


def _pay_preview(game, player: int, v) -> list[str]:
    """Set 4, a pay_mana option: the colours that could still be produced
    once the rest of the cost is paid after spending this unit (exact, like
    the cast preview's `pv:colors_left`)."""
    from .engine.mana import RemainingCost

    if game.paying is None:
        return []
    rem, sac, exclude = game.paying
    r2 = rem.copy()
    pool, gone, excl = None, set(), set(exclude)
    if v[0] == "pool":
        r2.apply(v[1])
        pool = dict(game.players[player].pool)
        pool[v[1]] -= 1
    else:
        card, color = v[1], v[2]
        r2.apply(color)
        excl.add(card.oid)
        ab = next(a for a in card.face.abilities if a.mana is not None)
        if ab.sac_self:
            gone = {card.oid}
    f = []
    for col in PREVIEW_COLORS:
        r3 = RemainingCost(r2.generic, dict(r2.colored))
        r3.colored[col] = r3.colored.get(col, 0) + 1
        if game._cost_feasible(player, r3, sac, excl, pool, gone):
            f.append(f"pv:colors_left:{col}")
    return f


def _preview_v4(game, player: int, i: int) -> list[str] | None:
    """Set-4 previews of option `i`; None for the decision kinds and options it does not cover."""
    d = game.decision
    o = d.options[i]
    v = o.value
    if d.kind == "declare_blocker":
        return _block_preview(game, player, o.label, v)
    if d.kind == "choose_x":
        return _x_preview(game, player, v)
    if d.kind == "pay_mana":
        return _pay_preview(game, player, v)
    if d.kind == "priority" and isinstance(v, tuple) and v[0] == "land":
        return _adds_colour_preview(game, player, v[1].face)
    if d.kind == "choose_card" and o.key[0] == "search" and v is not None:
        return _adds_colour_preview(game, player, v.face)
    return None


def option_preview(game, player: int, i: int, features: int = FEATURES) -> list[str]:
    """Engine-computed effects of taking option `i` of the current decision,
    from the current state without changing it (`pv:` tokens): creatures a
    sweeper ability kills per side, ward and lethal damage on a target, mana
    and colours left after a cast or activation (set 2 and up); combat
    results of a block, X previews, colours a land adds and colours left
    after a mana payment (set 4)."""
    if check_features(features) < 2:
        return []
    if getattr(game, "NATIVE", False):
        return game.option_preview(player, i, features)
    if features >= 4:
        f = _preview_v4(game, player, i)
        if f is not None:
            return f
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
