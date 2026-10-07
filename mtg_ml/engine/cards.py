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
    SACRIFICE,
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
    n = op.get("n_cast_from_graveyard", op["n"]) if item.cast_from == "graveyard" else op["n"]
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


def _op_gain_life(g, item, op):
    """per_storm: n for each spell cast before this one this turn (a storm
    trigger standing in for the copies: Weather the Storm). sacrificed_mv:
    instead of n, the mana value of the permanent sacrificed to cast it
    (Reckoner's Bargain)."""
    if op.get("sacrificed_mv"):
        g.gain_life(item.controller, item.data.get("sacrificed_mv", 0))
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


def _metalcraft(g, item) -> bool:
    return _artifacts_you_control(g, item.controller) >= 3


def _op_damage_target(g, item, op):
    """index: which target (default 0). n_landfall: the amount instead of n
    if a land entered under the controller's control this turn. n_metalcraft:
    the amount instead of n if the controller controls three or more
    artifacts as it resolves."""
    i = op.get("index", 0)
    if i < len(item.targets) and g.target_legal(item, i):
        n = op["n"]
        if "n_landfall" in op and _landfall(g, item):
            n = op["n_landfall"]
        if "n_metalcraft" in op and _metalcraft(g, item):
            n = op["n_metalcraft"]
        g.deal_damage(_source(item), item.targets[i], n)


def _op_damage_target_controller(g, item, op):
    """The source deals n damage to the controller of the targeted permanent (Smash to Smithereens)."""
    t = g.target(item)
    if t is not None:
        g.deal_damage(_source(item), ("player", t.controller), op["n"])


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
    """Library search predicate from an op's supertype / type / subtypes_any."""
    sup, typ, subs = op.get("supertype"), op.get("type"), op.get("subtypes_any")

    def pred(c) -> bool:
        f = c.face
        if sup and sup not in f.supertypes:
            return False
        if typ and typ not in f.types:
            return False
        if subs and not (f.subtypes & set(subs)):
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


def _op_counters_on_target(g, item, op):
    """n +1/+1 counters on the target; keywords: keyword counters (a
    lifelink counter), kept until it leaves the battlefield."""
    t = g.target(item)
    if t is not None:
        t.counters += op.get("n", 0)
        if op.get("keywords"):
            t.granted = t.granted | frozenset(op["keywords"])


def _op_animate_target(g, item, op):
    """The target becomes a creature with base power / toughness and gains
    keywords, until it leaves the battlefield (Kenku Artificer)."""
    t = g.target(item)
    if t is not None:
        t.animated = (op["power"], op["toughness"])
        t.granted = t.granted | frozenset(op.get("keywords", ()))


def _op_tap_or_untap_target(g, item, op):
    """You may tap or untap the target (Sewer-veillance Cam)."""
    t = g.target(item)
    if t is None:
        return
    options = [
        Option("Leave it", ("tap_or_untap", "neither"), None),
        Option(f"Tap {t.name}#{t.oid}", ("tap_or_untap", "tap"), True),
        Option(f"Untap {t.name}#{t.oid}", ("tap_or_untap", "untap"), False),
    ]
    choice = yield from g.ask(item.controller, CHOOSE_MODE, f"{item.name}: tap or untap {t.name}#{t.oid}?", options)
    if choice is not None:
        t.tapped = choice


def _op_return_from_graveyard(g, item, op):
    """Return up to n cards of a type from the controller's graveyard to
    their hand, chosen one at a time as it resolves (Blood Fountain)."""
    p = item.controller
    typ, n = op["type"], op["n"]
    for i in range(n):
        cands = g._dedupe_by_name(c for c in g.players[p].graveyard if typ in c.face.types)
        if not cands:
            return
        options = [Option("Stop", ("return_gy", None), None)] + [Option(f"Return {c.name}", ("return_gy", c.name), c) for c in cands]
        c = yield from g.ask(p, CHOOSE_CARD, f"{item.name}: return a {typ.lower()} card from your graveyard to your hand ({i + 1}/{n})", options)
        if c is None:
            return
        g._move(c, "hand", known_to={0, 1})


def _op_opponent_sacrifices(g, item, op):
    """The opponent sacrifices a creature of their choice;
    greatest_power_if_evidence: one with the greatest power among theirs
    if evidence was collected (Extract a Confession)."""
    opp = 1 - item.controller
    cands = [c for c in g.battlefield if c.controller == opp and g.is_creature(c)]
    if not cands:
        return
    if op.get("greatest_power_if_evidence") and item.method == "evidence":
        top = max(g.power(c) for c in cands)
        cands = [c for c in cands if g.power(c) == top]
    options = [Option(f"Sacrifice {c.name}#{c.oid}", ("sacrifice", c.name), c) for c in g._dedupe_by_equiv(cands)]
    c = yield from g.ask(opp, SACRIFICE, f"{item.name}: sacrifice a creature", options)
    g.sacrifice(c)


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
    "explore_target": _op_explore_target,
    "shuffle_into_library": _op_shuffle_into_library,
    "counters_on_target": _op_counters_on_target,
    "animate_target": _op_animate_target,
    "tap_or_untap_target": _op_tap_or_untap_target,
    "return_from_graveyard": _op_return_from_graveyard,
    "opponent_sacrifices": _op_opponent_sacrifices,
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
            raise ValueError(
                f"unknown op {op['op']!r}: add it to OPS (mtg_ml/engine/cards.py) and to Op, parse_ops and Op::name"
                " (native/src/cards.rs) and Eng::run_op (native/src/engine.rs); card_shape picks the name up by itself"
            )
        if op["op"] == "custom" and op["fn"] not in CUSTOM:
            raise ValueError(f"unknown custom effect {op['fn']!r}")
        if op["op"] == "optional_payment":
            make_effect(op["then"])
    fn = lambda g, item: run_ops(g, item, ops)  # noqa: E731
    fn.ops = tuple(ops)  # read by encode.option_preview
    return fn


def _targets(kinds: list[str] | None) -> tuple[TargetSpec, ...]:
    return tuple(TargetSpec(k) for k in kinds or ())


# Spec fields, split by whether `card_shape` turns them into shape tokens
# (feature set 5). A new field goes into exactly one of the two sets of its
# level, here and in native/src/cards.rs (same names), and is parsed in
# card_def / _ability / _trigger and parse_card; `_check_fields` says so.
SHAPE_CARD_FIELDS = frozenset({
    "cost", "colors", "devoid", "cost_reduction", "additional_sac", "additional_discard", "flashback", "escape", "madness", "bestow",
    "plot", "overload", "flashback_cost", "alternative_cost", "ward", "enters_tapped", "etb_x_counters", "back", "targets", "effect",
    "modes", "overload_effect", "abilities", "triggers", "bargain", "collect_evidence",
})  # fmt: skip
# Shown on entities otherwise (name, printed types, keywords, P/T) or not at all.
NON_SHAPE_CARD_FIELDS = frozenset({"name", "types", "subtypes", "supertypes", "text", "power", "toughness", "keywords", "escape_exile"})
SHAPE_ABILITY_FIELDS = frozenset({
    "effect", "cost", "tap", "sac_self", "sac_other", "discard_self", "discard_other", "exile_self", "x_target_mv", "x_reveal", "zone",
    "sorcery_speed", "mana", "targets",
})  # fmt: skip
NON_SHAPE_ABILITY_FIELDS = frozenset({"name"})
SHAPE_TRIGGER_FIELDS = frozenset({"event", "effect", "condition", "targets", "up_to"})
NON_SHAPE_TRIGGER_FIELDS = frozenset({"name"})


def _check_fields(spec: dict, what: str, shape: frozenset, non_shape: frozenset, owner: str) -> None:
    unknown = set(spec) - shape - non_shape
    if unknown:
        w = what.upper()
        raise ValueError(
            f"{owner}: unknown {what} fields {sorted(unknown)}: add each to SHAPE_{w}_FIELDS (if it changes what the card does;"
            f" then also to card_shape) or NON_SHAPE_{w}_FIELDS, in both mtg_ml/engine/cards.py and native/src/cards.rs, and"
            " parse it in card_def / _ability / _trigger and parse_card; docs/adding-cards.md"
        )


def _ability(a: dict) -> AbilityDef:
    _check_fields(a, "ability", SHAPE_ABILITY_FIELDS, NON_SHAPE_ABILITY_FIELDS, a.get("name", "?"))
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
    )


def _trigger(t: dict) -> TriggerDef:
    _check_fields(t, "trigger", SHAPE_TRIGGER_FIELDS, NON_SHAPE_TRIGGER_FIELDS, t.get("name", "?"))
    return TriggerDef(t["name"], t["event"], make_effect(t["effect"]), _condition(t.get("condition")), _targets(t.get("targets")), t.get("up_to", False))


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


SHAPE_MV_STEPS = (1, 2, 3, 4, 5, 6, 7)
SHAPE_AB_MV_CAP = 3
SHAPE_COST_KEYS = ("flashback", "escape", "madness", "bestow", "plot", "overload")


def op_names(ops) -> list[str]:
    """Op names of an effect in order, the ops of an `optional_payment` after it."""
    out = []
    for op in ops or ():
        out.append(op["op"])
        if op["op"] == "optional_payment":
            out += op_names(op["then"])
    return out


SHAPE_N_STEPS = (1, 2, 3, 4)
# Op parameters that change what the op does, each a token `{op}:{key}` when set.
SHAPE_OP_FLAGS = ("n_metalcraft", "sacrificed_mv", "greatest_power_if_evidence")


def _target_tokens(prefix: str, kinds) -> list[str]:
    """`{prefix}{kind}` and `e:target:{kind}` per target kind."""
    return [f"{p}{k}" for k in kinds for p in (prefix, "e:target:")]


def _op_tokens(prefix: str, ops) -> list[str]:
    """`{prefix}{op}` per op in order, `{prefix}{op}:n>={k}` for its amount
    `n` (damage, cards, life), the same under `e:op:` (a spell's, an
    ability's or a trigger's op alike), then an `optional_payment`'s ops."""
    out = []
    for op in ops or ():
        n = op.get("n")
        for p in (prefix, "e:op:"):  # where it happens, and the op wherever it happens
            out.append(p + op["op"])
            if isinstance(n, int) and not isinstance(n, bool):
                out += [f"{p}{op['op']}:n>={k}" for k in SHAPE_N_STEPS if n >= k]
            out += [f"{p}{op['op']}:{k}" for k in SHAPE_OP_FLAGS if op.get(k)]
        if op["op"] == "optional_payment":
            out += _op_tokens(prefix, op["then"])
    return out


def card_shape(spec: dict, d: CardDef) -> tuple[str, ...]:
    """What a card (face or token) does, as entity tokens read from its spec
    rather than its name, so a card the network never saw shares tokens with
    cards that work alike (feature set 5, docs/features.md): mana value and
    colours, cost shapes, the spell's target kinds and ops, each ability's
    zone / costs / mana / targets / ops (`e:ab:`), each trigger's event,
    condition and ops (`e:trig:`), each op with its amount `n` as a
    thermometer (`e:spell:op:damage_target:n>=3`). Deduplicated, first
    occurrence first.
    native/src/cards.rs `card_shape` builds the same list."""
    t = [f"e:mv>={k}" for k in SHAPE_MV_STEPS if d.cost.mana_value >= k]
    if "{X}" in spec.get("cost", ""):
        t.append("e:cost:x")
    if "/P}" in spec.get("cost", ""):
        t.append("e:cost:phyrexian")
    t += [f"e:color:{c}" for c in "WUBRG" if c in d.colors]
    if "cost_reduction" in spec:
        t.append(f"e:cost:reduction:{spec['cost_reduction']}")
    if "additional_sac" in spec:
        t += ["e:cost:additional_sac", f"e:cost:additional_sac:{spec['additional_sac']}"]
    if spec.get("additional_discard"):
        t.append("e:cost:additional_discard")
    if spec.get("bargain"):
        t.append("e:cost:bargain")
    if spec.get("collect_evidence"):
        t.append("e:cost:collect_evidence")
    t += [f"e:cost:{k}" for k in SHAPE_COST_KEYS if k in spec]
    if "flashback_cost" in spec:
        t += ["e:cost:flashback", "e:cost:sac_lands"]
    if "alternative_cost" in spec:
        t += ["e:cost:alternative", "e:cost:sac_lands"]
    if spec.get("ward"):
        t.append("e:ward")
    for k in ("enters_tapped", "etb_x_counters"):
        if spec.get(k):
            t.append(f"e:{k}")
    if "back" in spec:
        t.append("e:transforms")
    t += _target_tokens("e:spell:target:", spec.get("targets", ()))
    t += _op_tokens("e:spell:op:", spec.get("effect"))
    if spec.get("modes"):
        t.append("e:spell:modal")
    for m in spec.get("modes", ()):
        t += _target_tokens("e:spell:target:", m.get("targets", ()))
        t += _op_tokens("e:spell:op:", m["effect"])
    t += _op_tokens("e:spell:op:", spec.get("overload_effect"))
    for a in spec.get("abilities", ()):
        t.append(f"e:ab:zone:{a.get('zone', 'battlefield')}")
        if "mana" in a:
            t += ["e:ab:mana"] + [f"e:ab:mana:{c}" for c in a["mana"]]
        mv = M(a.get("cost")).mana_value
        if mv > 0:
            t.append(f"e:ab:mv:{min(mv, SHAPE_AB_MV_CAP)}")
        for k in ("tap", "sac_self", "sac_other", "discard_self", "discard_other", "exile_self"):
            if a.get(k):
                t.append(f"e:ab:{k}")
        if "sac_other" in a:
            t.append(f"e:ab:sac_other:{a['sac_other']}")
        if a.get("x_target_mv") or "x_reveal" in a:
            t.append("e:ab:x")
        if a.get("sorcery_speed"):
            t.append("e:ab:sorcery_speed")
        t += _target_tokens("e:ab:target:", a.get("targets", ()))
        t += _op_tokens("e:ab:op:", a.get("effect"))
    for tr in spec.get("triggers", ()):
        t.append(f"e:trig:{tr['event']}")
        t += [f"e:trig:cond:{k}:{v if isinstance(v, str) else str(v).lower()}" for k, v in tr.get("condition", {}).items()]
        t += _target_tokens("e:trig:target:", tr.get("targets", ()))
        if tr.get("up_to"):
            t.append("e:trig:up_to")
        t += _op_tokens("e:trig:op:", tr["effect"])
    return tuple(dict.fromkeys(t))


def card_def(spec: dict) -> CardDef:
    _check_fields(spec, "card", SHAPE_CARD_FIELDS, NON_SHAPE_CARD_FIELDS, spec.get("name", "?"))
    cr = spec.get("cost_reduction")
    phy = ManaCost.phyrexian(spec.get("cost"))
    d = CardDef(
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
        collect_evidence=spec.get("collect_evidence", 0),
        abilities=tuple(_ability(a) for a in spec.get("abilities", ())),
        triggers=tuple(_trigger(t) for t in spec.get("triggers", ())),
        enters_tapped=spec.get("enters_tapped", False),
        etb_x_counters=spec.get("etb_x_counters", False),
        back=FACES[spec["back"]] if "back" in spec else None,
        modes=tuple(SpellMode(m["name"], _targets(m.get("targets")), make_effect(m["effect"])) for m in spec.get("modes", ())),
    )
    d.shape = card_shape(spec, d)
    return d


def load(path: str = SPEC_PATH) -> None:
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    for registry, section in ((FACES, "face"), (CARDS, "card"), (TOKENS, "token")):
        for s in spec.get(section, ()):
            if s["name"] in registry:
                raise ValueError(f"duplicate {section} {s['name']!r}")
            registry[s["name"]] = card_def(s)


load()
