"""Card definitions for the supported pool: Jund Wildfire, Mono Blue Terror
and Red Madness (Pauper), plus the tokens they create.

The pool itself is data: `cards.toml` (shared with the Rust port). This
module turns each entry into a `CardDef` and each effect, a list of ops,
into a Python effect function. `OPS` holds the generic ops, `CUSTOM` the
card-specific effects too irregular to be worth an op. Oracle text
snapshot: data/oracle_cards.json. How to add cards: docs/adding-cards.md.
"""

from __future__ import annotations

import inspect
import itertools
import os
import tomllib

from .mana import ManaCost
from .objects import (
    CHOOSE_CARD,
    CHOOSE_MODE,
    EXILE_FROM_GY,
    FREE,
    ORDER,
    YES_NO,
    AbilityDef,
    CardDef,
    Option,
    SpellMode,
    TargetSpec,
    TempEffect,
    TriggerDef,
)

SPEC_PATH = os.path.join(os.path.dirname(__file__), "cards.toml")

CARDS: dict[str, CardDef] = {}
TOKENS: dict[str, CardDef] = {}
FACES: dict[str, CardDef] = {}

M = ManaCost.parse


def is_instant_or_sorcery(c) -> bool:
    return bool(c.face.types & {"Instant", "Sorcery"})


def _is_basic(c) -> bool:
    return "Basic" in c.face.supertypes and "Land" in c.face.types


# ---------------------------------------------------------------------------
# Generic ops: (game, stack item, op spec) -> None or a generator
# ---------------------------------------------------------------------------


def _op_draw(g, item, op):
    """each_controlling: instead each player who controls a permanent with
    this name draws (Bonder's Ornament), the controller first."""
    n = op.get("n_cast_from_graveyard", op["n"]) if item.cast_from == "graveyard" else op["n"]
    if "each_controlling" in op:
        for q in (item.controller, 1 - item.controller):
            if any(c.controller == q and c.name == op["each_controlling"] for c in g.battlefield):
                g.draw(q, n)
        return
    g.draw(item.controller, n)


def _op_mill(g, item, op):
    if op["who"] == "you":
        g.mill(item.controller, op["n"])
        return
    t = g.target(item)
    if t is not None:
        g.mill(t[1], op["n"])


def _color_ok(op, card) -> bool:
    """`if_color`: the effect only does something to an object of that color (Pyroblast)."""
    return "if_color" not in op or op["if_color"] in card.face.colors


def _op_counter_target(g, item, op):
    t = g.target(item)
    if t is not None and _color_ok(op, t.card):
        g.counter(t)


def _op_counter_target_unless_paid(g, item, op):
    t = g.target(item)
    if t is None:
        return
    cost = M(op["cost"])
    paid = yield from g.optional_payment(t.controller, cost, f"{item.name}: pay {cost} or {t.name} is countered")
    if not paid:
        g.counter(t)


def _op_destroy_target(g, item, op):
    t = g.target(item)
    if t is None or not _color_ok(op, t):
        return
    if op.get("mv_is_x") and t.face.mana_value != item.x:
        return
    g.destroy(t)


def _op_bounce_target(g, item, op):
    t = g.target(item)
    if t is not None:
        g._move(t, "hand")


def _op_tap_target(g, item, op):
    t = g.target(item)
    if t is not None:
        t.tapped = True
        t.skip_untap = max(t.skip_untap, op.get("skip_untap", 0))


def _op_grant_target(g, item, op):
    t = g.target(item)
    if t is not None:
        t.temp.append(TempEffect(keywords=frozenset(op["keywords"])))


def _op_create_token(g, item, op):
    for _ in range(op.get("n", 1)):
        g.create_token(item.controller, op["token"])


def _source_power(g, item) -> int:
    """The power of the source: the live permanent, or its last known information."""
    src = _source(item)
    return g.power(g.live(src) or src)


def _op_gain_life(g, item, op):
    """per_storm: n for each spell cast before this one this turn (a storm
    trigger standing in for the copies: Weather the Storm). n_from =
    "source_power": life equal to the source's power (Boulderbranch Golem)."""
    if op.get("n_from") == "source_power":
        g.gain_life(item.controller, _source_power(g, item))
        return
    g.gain_life(item.controller, op["n"] * (item.data["storm"] if op.get("per_storm") else 1))


def _op_lose_life(g, item, op):
    """who: you | opponent | target_player | target_controller (of the targeted permanent)."""
    who = op["who"]
    if who == "you":
        p = item.controller
    elif who == "opponent":
        p = 1 - item.controller
    else:
        t = g.target(item)
        if t is None:
            return
        p = t[1] if who == "target_player" else t.controller
    g.players[p].life -= op["n"]


def _op_counter_on_source(g, item, op):
    live = g.live(item.source)
    if live is not None:
        live.counters += 1


def _source(item):
    """The object dealing an effect's damage: the spell itself, or the
    (last known information of the) source of an ability or trigger."""
    return item.source if item.source is not None else item.card


def _landfall(g, item) -> bool:
    return g.players[item.controller].landfall_turn == g.turn


def _op_damage_target(g, item, op):
    """index: which target (default 0). n_landfall: the amount instead of n
    if a land entered under the controller's control this turn."""
    i = op.get("index", 0)
    if i < len(item.targets) and g.target_legal(item, i):
        n = op["n_landfall"] if "n_landfall" in op and _landfall(g, item) else op["n"]
        g.deal_damage(_source(item), item.targets[i], n)


def _op_damage_target_controller(g, item, op):
    """The source deals n damage to the controller of the targeted permanent (Smash to Smithereens)."""
    t = g.target(item)
    if t is not None:
        g.deal_damage(_source(item), ("player", t.controller), op["n"])


def _op_damage_target_from(g, item, op):
    """Damage to target `index` equal to X (from = "x") or to the power of
    the creature chosen or revealed as the spell was cast (from =
    "chosen_power": Monstrous Emergence; last known information if it left)."""
    i = op.get("index", 0)
    if i < len(item.targets) and g.target_legal(item, i):
        if op["from"] == "chosen_power":
            chosen = item.data["chosen"]
            n = g.power(g.live(chosen) or chosen)
        else:
            n = item.x
        g.deal_damage(_source(item), item.targets[i], n)


def _op_damage_each_opponent(g, item, op):
    """if_discarded_nonland: only if the card discarded to cast it was not a land (Grab the Prize)."""
    if op.get("if_discarded_nonland") and item.data.get("discarded_land", True):
        return
    g.deal_damage(_source(item), ("player", 1 - item.controller), op["n"])


def _op_damage_each_creature(g, item, op):
    """n, or x = true for X; without: a keyword that protects; whose = opponent: only theirs."""
    without = op.get("without")
    n = item.x if op.get("x") else op["n"]
    for c in list(g.battlefield):
        if not g.is_creature(c) or (without and g.has(c, without)):
            continue
        if op.get("whose") == "opponent" and c.controller == item.controller:
            continue
        g.deal_damage(_source(item), ("perm", c.oid), n)


def _op_discard(g, item, op):
    for _ in range(op["n"]):
        if (yield from g.choose_discard(item.controller, item.name)) is None:
            return


def _op_return_to_battlefield(g, item, op):
    """Graveyard trigger: the card returns, if it is still that object in the graveyard."""
    c = item.data["card"]
    if c.zone == "graveyard" and c.oid == item.data["oid"]:
        g.put_onto_battlefield(c, c.owner, tapped=op.get("tapped", False))


def _op_exile_target(g, item, op):
    t = g.target(item)
    if t is not None:
        g._move(t, "exile")


def _op_exile_from_graveyards(g, item, op):
    """Exile up to n cards from any graveyards, chosen one at a time as the
    effect resolves (Faerie Macabre; the engine has no graveyard targets)."""
    p = item.controller
    n = op["n"]
    for i in range(n):
        options = [Option("Exile nothing more", ("exile_any_gy", None), None)]
        for q in (p, 1 - p):
            rel = "self" if q == p else "opponent"
            options += [Option(f"Exile {c.name} ({rel} graveyard)", ("exile_any_gy", rel, c.name), c) for c in g._dedupe_by_name(g.players[q].graveyard)]
        if len(options) == 1:
            return
        c = yield from g.ask(p, EXILE_FROM_GY, f"{item.name}: exile a card from a graveyard ({i + 1}/{n})", options)
        if c is None:
            return
        g._move(c, "exile")


def _op_exile_all_graveyards(g, item, op):
    for q in (0, 1):
        for c in list(g.players[q].graveyard):
            g._move(c, "exile")


def _op_exile_graveyard(g, item, op):
    t = g.target(item)
    if t is None:
        return
    for c in list(g.players[t[1]].graveyard):
        g._move(c, "exile")


def search_filter(op):
    """Library search predicate from an op's supertype / type / subtypes_any / colorless."""
    sup, typ, subs, colorless = op.get("supertype"), op.get("type"), op.get("subtypes_any"), op.get("colorless", False)

    def pred(c) -> bool:
        f = c.face
        if sup and sup not in f.supertypes:
            return False
        if typ and typ not in f.types:
            return False
        if subs and not (f.subtypes & set(subs)):
            return False
        if colorless and f.colors:
            return False
        return True

    return pred


def _op_search_library(g, item, op):
    yield from g.search_library(
        item.controller, search_filter(op), op["dest"], op["what"], tapped=op.get("tapped", False), reveal=op.get("reveal", False)
    )


def _op_optional_payment(g, item, op):
    paid = yield from g.optional_payment(item.controller, M(op["cost"]), op["prompt"])
    if paid:
        yield from run_ops(g, item, op["then"])


def _op_scry(g, item, op):
    yield from g.scry(item.controller, op["n"])


def _op_surveil(g, item, op):
    yield from g.surveil(item.controller)


def _op_dig(g, item, op):
    """Look at the top n, you may put a matching card into your hand (revealed), the rest on the bottom."""
    yield from g.dig(item.controller, op["n"], search_filter(op), op["what"], item.name)


def _op_cascade(g, item, op):
    """Cast trigger: cascade, below the spell's mana value."""
    yield from g.cascade(item.controller, item.source.face.mana_value)


def _op_station(g, item, op):
    """Charge counters on the source equal to the power of the creature tapped to pay the cost."""
    live = g.live(item.source)
    if live is not None:
        live.charge += item.data["tapped_power"]


def _op_attach(g, item, op):
    """Equip (702.6): attach the source, if it is still on the battlefield
    under the ability's controller's control, to the target creature."""
    live = g.live(item.source)
    t = g.target(item)
    if live is None or t is None or live.controller != item.controller:
        return
    live.attached_to = t.oid
    g._log(f"{live.name}#{live.oid} is attached to {t.name}#{t.oid}")


def _op_return_random_from_graveyard(g, item, op):
    """Return a card of this type at random from your graveyard to your hand (Haunted Fengraf)."""
    p = item.controller
    cands = [c for c in g.players[p].graveyard if op["type"] in c.face.types]
    if not cands:
        return
    c = cands[g.rng.randrange(len(cands))]
    g._log(f"p{p} returns {c.name} at random")
    g._move(c, "hand")


def _op_return_from_graveyard(g, item, op):
    """Choose up to n cards of the given types in a graveyard ("you" or
    "any") and return them to their owners' hands. each_type: at most one per
    type (Call Damage Control's modes). Chosen on resolution: targeting in
    graveyards is not modelled."""
    p = item.controller
    types, n, each_type = op["types"], op["n"], op.get("each_type", False)
    owners = (p,) if op["whose"] == "you" else (p, 1 - p)
    used: list[str] = []
    for _ in range(n):
        options = [Option("Return nothing more", ("return_gy", None), None)]
        for q in owners:
            rel = "self" if q == p else "opponent"
            whose = "your" if q == p else "the opponent's"
            for typ in types:
                if each_type and typ in used:
                    continue
                for c in g._dedupe_by_name(c for c in g.players[q].graveyard if typ in c.face.types):
                    options.append(Option(f"Return {c.name} ({typ.lower()}) from {whose} graveyard", ("return_gy", rel, typ.lower(), c.name), (typ, c)))
        choice = yield from g.ask(p, CHOOSE_CARD, f"{item.name}: return a card from a graveyard to its owner's hand", options)
        if choice is None:
            return
        used.append(choice[0])
        g._move(choice[1], "hand")


def _op_explore_target(g, item, op):
    t = g.target(item)
    if t is not None:
        yield from g.explore(t)


def _op_shuffle_into_library(g, item, op):
    """Dies trigger: the card that went to the graveyard is shuffled into its
    owner's library, if it is still that same object in the graveyard."""
    c = item.data["card"]
    if c.zone == "graveyard" and c.oid == item.data["oid"]:
        g._move(c, "library")
        g.shuffle(c.owner)


def _op_custom(g, item, op):
    return CUSTOM[op["fn"]](g, item)


OPS = {
    "draw": _op_draw,
    "mill": _op_mill,
    "counter_target": _op_counter_target,
    "counter_target_unless_paid": _op_counter_target_unless_paid,
    "destroy_target": _op_destroy_target,
    "bounce_target": _op_bounce_target,
    "tap_target": _op_tap_target,
    "grant_target": _op_grant_target,
    "create_token": _op_create_token,
    "gain_life": _op_gain_life,
    "lose_life": _op_lose_life,
    "counter_on_source": _op_counter_on_source,
    "damage_target": _op_damage_target,
    "damage_target_controller": _op_damage_target_controller,
    "damage_target_from": _op_damage_target_from,
    "damage_each_opponent": _op_damage_each_opponent,
    "damage_each_creature": _op_damage_each_creature,
    "discard": _op_discard,
    "return_to_battlefield": _op_return_to_battlefield,
    "exile_graveyard": _op_exile_graveyard,
    "exile_target": _op_exile_target,
    "exile_from_graveyards": _op_exile_from_graveyards,
    "exile_all_graveyards": _op_exile_all_graveyards,
    "search_library": _op_search_library,
    "optional_payment": _op_optional_payment,
    "scry": _op_scry,
    "surveil": _op_surveil,
    "dig": _op_dig,
    "cascade": _op_cascade,
    "station": _op_station,
    "attach": _op_attach,
    "return_random_from_graveyard": _op_return_random_from_graveyard,
    "return_from_graveyard": _op_return_from_graveyard,
    "explore_target": _op_explore_target,
    "shuffle_into_library": _op_shuffle_into_library,
    "custom": _op_custom,
}


def run_ops(g, item, ops):
    for op in ops:
        res = OPS[op["op"]](g, item, op)
        if inspect.isgenerator(res):
            yield from res


# ---------------------------------------------------------------------------
# Custom effects (one card each)
# ---------------------------------------------------------------------------


def _delver_reveal(g, item):
    p = item.controller
    lib = g.players[p].library
    if not lib:
        return
    top = lib[0]
    top.known_to.add(p)
    reveal = yield from g.ask(
        p,
        YES_NO,
        f"Delver of Secrets: reveal {top.name}?",
        [Option("Don't reveal", ("reveal", "no"), False), Option(f"Reveal {top.name}", ("reveal", "yes"), True)],
    )
    if reveal:
        top.known_to = {0, 1}
        live = g.live(item.source)
        if is_instant_or_sorcery(top) and live is not None and not live.transformed:
            live.transformed = True
            g._log(f"{live.defn.name}#{live.oid} transforms into {live.name}")


def _brainstorm(g, item):
    p = item.controller
    g.draw(p, 3)
    for i in range(2):
        if not g.players[p].hand:
            break
        c = yield from g.ask(
            p,
            CHOOSE_CARD,
            f"Brainstorm: put a card from your hand on top of your library ({i + 1}/2, the second one ends on top)",
            g._hand_card_options(p, "put back"),
        )
        g._move(c, "library", position="top", known_to={p})


def _ponder(g, item):
    p = item.controller
    lib = g.players[p].library
    top = lib[:3]
    for c in top:
        c.known_to.add(p)
    if len(top) > 1:
        options = []
        seen = set()
        for perm in itertools.permutations(top):
            names = tuple(c.name for c in perm)
            if names in seen:
                continue
            seen.add(names)
            options.append(Option("Top to bottom: " + ", ".join(names), ("order",) + names, perm))
        order = yield from g.ask(p, ORDER, "Ponder: put the cards back in any order", options)
        lib[: len(top)] = list(order)
    shuffle = yield from g.ask(
        p, YES_NO, "Ponder: shuffle your library?", [Option("Don't shuffle", ("shuffle", "no"), False), Option("Shuffle", ("shuffle", "yes"), True)]
    )
    if shuffle:
        g.shuffle(p)
    g.draw(p)


def _deem_inferior(g, item):
    t = g.target(item)
    if t is None:
        return
    owner = t.owner
    lib = g.players[owner].library
    options = [Option(f"Put {t.name} on the bottom", ("deem", "bottom"), "bottom")]
    if len(lib) >= 2:
        options.insert(0, Option(f"Put {t.name} second from the top", ("deem", "second"), 1))
    where = yield from g.ask(owner, CHOOSE_MODE, f"Deem Inferior: where does {t.name} go?", options)
    g._move(t, "library", position=where, known_to={0, 1})


def _opponent_discards_else_draw(g, item):
    p = item.controller
    opp = 1 - p
    if not g.players[opp].hand:
        g.draw(p)
        return
    c = yield from g.ask(opp, CHOOSE_CARD, f"{item.source.name}: discard a card", g._hand_card_options(opp, "discard"))
    g.discard(c)


def _wildfire(g, item):
    """Destroy target land; its controller may search for a basic land and put it onto the battlefield tapped."""
    t = g.target(item)
    if t is not None:
        controller = t.controller
        g.destroy(t)
        yield from g.search_library(controller, _is_basic, "battlefield", "a basic land card", tapped=True)


def _duress(g, item):
    t = g.target(item)
    if t is None:
        return
    victim = g.players[t[1]]
    for c in victim.hand:
        c.known_to = {0, 1}
    g._log(f"p{victim.idx} reveals {[c.name for c in victim.hand]}")
    cands = g._dedupe_by_name(c for c in victim.hand if not c.face.is_type("Creature") and not c.face.is_type("Land"))
    if not cands:
        return
    c = yield from g.ask(item.controller, CHOOSE_CARD, "Duress: choose a card to discard", [Option(f"Discard {c.name}", ("duress", c.name), c) for c in cands])
    g.discard(c)


def _highway_robbery(g, item):
    """You may discard a card or sacrifice a land. If you do, draw two cards."""
    p = item.controller
    options = [Option("Neither: draw nothing", ("robbery", "none"), None)]
    options += [Option(f"Discard {c.name}", ("robbery", "discard", c.name), ("discard", c)) for c in g._dedupe_by_name(g.players[p].hand)]
    lands = g._dedupe_by_equiv(c for c in g.battlefield if c.controller == p and g.is_land(c))
    options += [Option(f"Sacrifice {c.name}#{c.oid}", ("robbery", "sacrifice", c.name), ("sacrifice", c)) for c in lands]
    choice = yield from g.ask(p, CHOOSE_CARD, "Highway Robbery: discard a card or sacrifice a land to draw two?", options)
    if choice is None:
        return
    if choice[0] == "discard":
        g.discard(choice[1])
    else:
        g.sacrifice(choice[1])
    g.draw(p, 2)


def _relic_exile_one(g, item):
    """Target player exiles a card of their choice from their graveyard."""
    t = g.target(item)
    if t is None or not g.players[t[1]].graveyard:
        return
    options = [Option(f"Exile {c.name}", ("exile_gy", c.name), c) for c in g._dedupe_by_name(g.players[t[1]].graveyard)]
    c = yield from g.ask(t[1], EXILE_FROM_GY, f"{item.name}: exile a card from your graveyard", options)
    g._move(c, "exile")


CUSTOM = {
    "delver_reveal": _delver_reveal,
    "brainstorm": _brainstorm,
    "ponder": _ponder,
    "deem_inferior": _deem_inferior,
    "opponent_discards_else_draw": _opponent_discards_else_draw,
    "wildfire": _wildfire,
    "duress": _duress,
    "highway_robbery": _highway_robbery,
    "relic_exile_one": _relic_exile_one,
}

# ---------------------------------------------------------------------------
# Spec -> CardDef
# ---------------------------------------------------------------------------


def _spells_in_graveyard(g, p) -> int:
    return sum(1 for c in g.players[p].graveyard if is_instant_or_sorcery(c))


def _artifacts_you_control(g, p) -> int:
    return sum(1 for c in g.battlefield if c.controller == p and g.is_artifact(c))


COST_REDUCTIONS = {
    "instants_and_sorceries_in_graveyard": _spells_in_graveyard,
    "artifacts_you_control": _artifacts_you_control,
    "cards_drawn_this_turn": lambda g, p: g.players[p].cards_drawn_this_turn,
}


CAST_FILTERS = {
    "noncreature": lambda card: "Creature" not in card.face.types,
    "instant_or_sorcery": is_instant_or_sorcery,
}


def _condition(spec: dict | None):
    if spec is None:
        return None
    if set(spec) == {"sacrificed_subtype"}:
        sub = spec["sacrificed_subtype"]
        return lambda g, src, lki: sub in lki.face.subtypes
    if set(spec) == {"spell"}:  # you_cast: the kind of spell cast
        flt = CAST_FILTERS[spec["spell"]]
        return lambda g, src, card: flt(card)
    if spec == {"bargained": True}:  # etb: the permanent was cast bargained
        return lambda g, src, method: method == "bargain"
    raise ValueError(f"unsupported trigger condition {spec!r}")


def make_effect(ops: list[dict] | None):
    if not ops:
        return None
    for op in ops:
        if op["op"] not in OPS:
            raise ValueError(f"unknown op {op['op']!r}")
        if op["op"] == "custom" and op["fn"] not in CUSTOM:
            raise ValueError(f"unknown custom effect {op['fn']!r}")
        if op["op"] == "optional_payment":
            make_effect(op["then"])
    fn = lambda g, item: run_ops(g, item, ops)  # noqa: E731
    fn.ops = tuple(ops)  # read by encode.option_preview
    return fn


def _targets(kinds: list[str] | None) -> tuple[TargetSpec, ...]:
    return tuple(TargetSpec(k) for k in kinds or ())


def _mana_amount(spec: dict | None):
    """{ n = 3, if_control = ["Mine", "Power-Plant"] }."""
    if spec is None:
        return None
    if set(spec) != {"n", "if_control"}:
        raise ValueError(f"unsupported mana_amount {spec!r}")
    return spec["n"], tuple(spec["if_control"])


def _ability(a: dict) -> AbilityDef:
    known = {
        "name", "effect", "cost", "tap", "sac_self", "sac_other", "discard_self", "discard_other", "exile_self", "x_target_mv",
        "x_reveal", "zone", "sorcery_speed", "mana", "targets", "mana_amount", "once_per_turn", "tap_other",
    }  # fmt: skip
    if set(a) - known:
        raise ValueError(f"{a.get('name')}: unknown ability fields {sorted(set(a) - known)}")
    if "mana" in a and "cost" in a and M(a["cost"]) != M("{1}"):
        raise ValueError(f"{a['name']}: a mana filter must cost exactly {{1}}")
    if a.get("zone", "battlefield") not in ("battlefield", "hand", "graveyard"):
        raise ValueError(f"{a['name']}: unknown ability zone {a['zone']!r}")
    if a.get("tap_other") not in (None, "creature"):
        raise ValueError(f"{a['name']}: unknown tap_other {a['tap_other']!r}")
    return AbilityDef(
        name=a["name"],
        effect=make_effect(a.get("effect")),
        cost=M(a.get("cost")),
        tap=a.get("tap", False),
        sac_self=a.get("sac_self", False),
        sac_other=a.get("sac_other"),
        discard_self=a.get("discard_self", False),
        discard_other=a.get("discard_other", False),
        exile_self=a.get("exile_self", False),
        x_target_mv=a.get("x_target_mv", 0),
        x_reveal=a.get("x_reveal"),
        zone=a.get("zone", "battlefield"),
        sorcery_speed=a.get("sorcery_speed", False),
        mana=tuple(a["mana"]) if "mana" in a else None,
        targets=_targets(a.get("targets")),
        mana_amount=_mana_amount(a.get("mana_amount")),
        once_per_turn=a.get("once_per_turn", False),
        tap_other=a.get("tap_other"),
    )


def _trigger(t: dict) -> TriggerDef:
    unknown = set(t) - {"name", "event", "effect", "condition", "targets", "optional_targets"}
    if unknown:
        raise ValueError(f"trigger {t.get('name')}: unknown fields {sorted(unknown)}")
    return TriggerDef(
        t["name"], t["event"], make_effect(t["effect"]), _condition(t.get("condition")), _targets(t.get("targets")), t.get("optional_targets", False)
    )


def colors_of(spec: dict) -> frozenset[str]:
    if "colors" in spec:
        return frozenset(spec["colors"].split())
    cost = spec.get("cost")
    if spec.get("devoid") or not cost:
        return frozenset()
    return frozenset(c for c in "WUBRG" if "{%s}" % c in cost or "{%s/P}" % c in cost)


def _land_sac(spec: dict | None) -> tuple[str, int] | None:
    """{ sacrifice = "mountain", n = 2 }: a cost of sacrificing lands instead of mana."""
    if spec is None:
        return None
    if set(spec) != {"sacrifice", "n"} or spec["sacrifice"] != "mountain":
        raise ValueError(f"unsupported land sacrifice cost {spec!r}")
    return spec["sacrifice"], spec["n"]


def _equipped(spec: dict) -> dict:
    """`equipped = { keywords, power, toughness }`: what an Equipment gives the equipped creature."""
    e = spec.get("equipped")
    if e is None:
        return {}
    if set(e) - {"keywords", "power", "toughness"}:
        raise ValueError(f"{spec['name']}: unknown equipped fields {sorted(set(e) - {'keywords', 'power', 'toughness'})}")
    if "Equipment" not in spec.get("subtypes", "").split():
        raise ValueError(f"{spec['name']}: equipped needs the Equipment subtype")
    return {
        "equipped_keywords": frozenset(e.get("keywords", ())),
        "equipped_power": e.get("power", 0),
        "equipped_toughness": e.get("toughness", 0),
    }


def card_def(spec: dict) -> CardDef:
    known = {
        "name", "cost", "types", "subtypes", "supertypes", "text", "devoid", "colors", "power", "toughness", "keywords", "ward",
        "targets", "effect", "additional_sac", "cost_reduction", "flashback", "escape", "escape_exile", "bestow", "enters_tapped",
        "etb_x_counters", "back", "modes", "abilities", "triggers", "madness", "plot", "overload", "overload_effect",
        "additional_discard", "alternative_cost", "flashback_cost", "bargain", "prototype", "prototype_face", "station",
        "additional_choose_creature", "equipped",
    }  # fmt: skip
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"{spec.get('name')}: unknown fields {sorted(unknown)}")
    cr = spec.get("cost_reduction")
    phy = ManaCost.phyrexian(spec.get("cost"))
    return CardDef(
        name=spec["name"],
        cost=M(spec.get("cost")),
        types=frozenset(spec["types"].split()),
        subtypes=frozenset(spec.get("subtypes", "").split()),
        supertypes=frozenset(spec.get("supertypes", "").split()),
        colors=colors_of(spec),
        power=spec.get("power"),
        toughness=spec.get("toughness"),
        keywords=frozenset(spec.get("keywords", ())),
        ward=spec.get("ward", 0),
        text=spec.get("text", ""),
        targets=_targets(spec.get("targets")),
        effect=make_effect(spec.get("effect")),
        additional_sac=spec.get("additional_sac"),
        cost_reduction=COST_REDUCTIONS[cr] if cr else None,
        flashback=M(spec["flashback"]) if "flashback" in spec else FREE if "flashback_cost" in spec else None,
        escape=M(spec["escape"]) if "escape" in spec else None,
        escape_exile=spec.get("escape_exile", 0),
        bestow=M(spec["bestow"]) if "bestow" in spec else None,
        madness=M(spec["madness"]) if "madness" in spec else None,
        plot=M(spec["plot"]) if "plot" in spec else None,
        overload=M(spec["overload"]) if "overload" in spec else None,
        overload_effect=make_effect(spec.get("overload_effect")),
        additional_discard=spec.get("additional_discard", False),
        alternative_sac=_land_sac(spec.get("alternative_cost")),
        flashback_sac=_land_sac(spec.get("flashback_cost")),
        phyrexian_cost=M(spec.get("cost")).minus_colored(phy) if phy.colored else None,
        phyrexian_life=2 * phy.mana_value,
        bargain=spec.get("bargain", False),
        abilities=tuple(_ability(a) for a in spec.get("abilities", ())),
        triggers=tuple(_trigger(t) for t in spec.get("triggers", ())),
        enters_tapped=spec.get("enters_tapped", False),
        etb_x_counters=spec.get("etb_x_counters", False),
        back=FACES[spec["back"]] if "back" in spec else None,
        modes=tuple(SpellMode(m["name"], _targets(m.get("targets")), make_effect(m["effect"])) for m in spec.get("modes", ())),
        prototype=M(spec["prototype"]) if "prototype" in spec else None,
        prototype_face=FACES[spec["prototype_face"]] if "prototype_face" in spec else None,
        station=spec["station"]["n"] if "station" in spec else 0,
        station_keywords=frozenset(spec["station"].get("keywords", ())) if "station" in spec else frozenset(),
        additional_choose_creature=spec.get("additional_choose_creature", False),
        **_equipped(spec),
    )


def load(path: str = SPEC_PATH) -> None:
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    for registry, section in ((FACES, "face"), (CARDS, "card"), (TOKENS, "token")):
        for s in spec.get(section, ()):
            if s["name"] in registry:
                raise ValueError(f"duplicate {section} {s['name']!r}")
            registry[s["name"]] = card_def(s)


load()
