"""Card definitions for the supported pool: Jund Wildfire and Mono Blue Terror
(Pauper), plus the tokens they create. Oracle text snapshot: data/oracle_cards.json.
"""

from __future__ import annotations

import dataclasses

from .mana import ManaCost
from .objects import (
    CHOOSE_CARD,
    CHOOSE_MODE,
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

CARDS: dict[str, CardDef] = {}
TOKENS: dict[str, CardDef] = {}

M = ManaCost.parse


def _colors(cost: str | None, devoid: bool = False) -> frozenset[str]:
    if devoid or not cost:
        return frozenset()
    return frozenset(c for c in "WUBRG" if "{%s}" % c in cost)


def card(
    name: str,
    cost: str | None,
    types: str,
    subtypes: str = "",
    supertypes: str = "",
    text: str = "",
    devoid: bool = False,
    registry: dict | None = None,
    **kw,
) -> CardDef:
    d = CardDef(
        name=name,
        cost=M(cost),
        types=frozenset(types.split()),
        subtypes=frozenset(subtypes.split()),
        supertypes=frozenset(supertypes.split()),
        colors=_colors(cost, devoid),
        text=text,
        **kw,
    )
    (CARDS if registry is None else registry)[name] = d
    return d


def is_instant_or_sorcery(c) -> bool:
    return bool(c.face.types & {"Instant", "Sorcery"})


def _spells_in_graveyard(g, p) -> int:
    return sum(1 for c in g.players[p].graveyard if is_instant_or_sorcery(c))


def _mana_ability(*colors: str) -> AbilityDef:
    return AbilityDef(name="add " + "/".join(colors), tap=True, mana=tuple(colors))


T = TargetSpec

# ---------------------------------------------------------------------------
# Lands
# ---------------------------------------------------------------------------

for _name, _sub, _col in (("Island", "Island", "U"), ("Swamp", "Swamp", "B"), ("Mountain", "Mountain", "R"), ("Forest", "Forest", "G")):
    card(_name, None, "Land", _sub, "Basic", text=f"{{T}}: Add {{{_col}}}.", abilities=(_mana_ability(_col),))

card(
    "Drossforge Bridge",
    None,
    "Artifact Land",
    text="Enters tapped. Indestructible. {T}: Add {B} or {R}.",
    keywords=frozenset({"indestructible"}),
    enters_tapped=True,
    abilities=(_mana_ability("B", "R"),),
)
card(
    "Slagwoods Bridge",
    None,
    "Artifact Land",
    text="Enters tapped. Indestructible. {T}: Add {R} or {G}.",
    keywords=frozenset({"indestructible"}),
    enters_tapped=True,
    abilities=(_mana_ability("R", "G"),),
)
card("Vault of Whispers", None, "Artifact Land", text="{T}: Add {B}.", abilities=(_mana_ability("B"),))


def _is_basic(c) -> bool:
    return "Basic" in c.face.supertypes and "Land" in c.face.types


def _twisted_search(g, item):
    yield from g.search_library(
        item.controller,
        lambda c: _is_basic(c) and bool(c.face.subtypes & {"Swamp", "Mountain", "Forest"}),
        "battlefield",
        "a basic Swamp, Mountain, or Forest card",
        tapped=True,
    )


card(
    "Twisted Landscape",
    None,
    "Land",
    text="{T}: Add {C}. {T}, Sacrifice: search for a basic Swamp, Mountain, or Forest, put it onto the battlefield tapped. Cycling {B}{R}{G}.",
    abilities=(
        _mana_ability("C"),
        AbilityDef(name="search for a basic land", tap=True, sac_self=True, effect=_twisted_search),
        AbilityDef(name="cycling {B}{R}{G}", cost=M("{B}{R}{G}"), zone="hand", discard_self=True, effect=lambda g, it: g.draw(it.controller)),
    ),
)

# ---------------------------------------------------------------------------
# Mono Blue Terror
# ---------------------------------------------------------------------------


def _delver_upkeep(g, item):
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


_aberration = card(
    "Insectile Aberration", None, "Creature", "Human Insect", text="Flying", power=3, toughness=2, keywords=frozenset({"flying"}), registry={}
)
_aberration = dataclasses.replace(_aberration, colors=frozenset({"U"}))  # blue colour indicator
card(
    "Delver of Secrets",
    "{U}",
    "Creature",
    "Human Wizard",
    text="Upkeep: look at the top card; you may reveal it; if it's an instant or sorcery, transform.",
    power=1,
    toughness=1,
    triggers=(TriggerDef("look at top card", "your_upkeep", _delver_upkeep),),
    back=_aberration,
)
card(
    "Tolarian Terror",
    "{6}{U}",
    "Creature",
    "Serpent",
    text="Costs {1} less for each instant and sorcery card in your graveyard. Ward {2}.",
    power=5,
    toughness=5,
    ward=2,
    cost_reduction=_spells_in_graveyard,
)
card(
    "Cryptic Serpent",
    "{5}{U}{U}",
    "Creature",
    "Serpent",
    text="Costs {1} less for each instant and sorcery card in your graveyard.",
    power=6,
    toughness=5,
    cost_reduction=_spells_in_graveyard,
)


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


card("Brainstorm", "{U}", "Instant", text="Draw three cards, then put two cards from your hand on top of your library in any order.", effect=_brainstorm)


def _ponder(g, item):
    p = item.controller
    lib = g.players[p].library
    top = lib[:3]
    for c in top:
        c.known_to.add(p)
    if len(top) > 1:
        options = []
        seen = set()
        for perm in _permutations(top):
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


def _permutations(cards):
    import itertools

    return [tuple(x) for x in itertools.permutations(cards)]


card("Ponder", "{U}", "Sorcery", text="Look at the top three cards, put them back in any order. You may shuffle. Draw a card.", effect=_ponder)


def _thought_scour(g, item):
    t = g.target(item)
    if t is not None:
        g.mill(t[1], 2)
    g.draw(item.controller)


card("Thought Scour", "{U}", "Instant", text="Target player mills two cards. Draw a card.", targets=(T("player"),), effect=_thought_scour)


def _mental_note(g, item):
    g.mill(item.controller, 2)
    g.draw(item.controller)


card("Mental Note", "{U}", "Instant", text="Mill two cards. Draw a card.", effect=_mental_note)


def _counterspell(g, item):
    t = g.target(item)
    if t is not None:
        g.counter(t)


card("Counterspell", "{U}{U}", "Instant", text="Counter target spell.", targets=(T("spell"),), effect=_counterspell)


def _force_spike(g, item):
    t = g.target(item)
    if t is None:
        return
    paid = yield from g.optional_payment(t.controller, M("{1}"), f"Force Spike: pay {{1}} or {t.name} is countered")
    if not paid:
        g.counter(t)


card("Force Spike", "{U}", "Instant", text="Counter target spell unless its controller pays {1}.", targets=(T("spell"),), effect=_force_spike)


def _islandcycle(g, item):
    yield from g.search_library(item.controller, lambda c: "Island" in c.face.subtypes, "hand", "an Island card", reveal=True)


card(
    "Lorien Revealed",
    "{3}{U}{U}",
    "Sorcery",
    text="Draw three cards. Islandcycling {1}.",
    effect=lambda g, it: g.draw(it.controller, 3),
    abilities=(AbilityDef(name="islandcycling {1}", cost=M("{1}"), zone="hand", discard_self=True, effect=_islandcycle),),
)


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


card(
    "Deem Inferior",
    "{3}{U}",
    "Sorcery",
    text="Costs {1} less for each card you've drawn this turn. The owner of target nonland permanent puts it second from the top or on the bottom.",
    targets=(T("nonland_permanent"),),
    effect=_deem_inferior,
    cost_reduction=lambda g, p: g.players[p].cards_drawn_this_turn,
)


def _sleep(g, item):
    t = g.target(item)
    if t is not None:
        t.tapped = True
        t.skip_untap += 1


card(
    "Sleep of the Dead",
    "{U}",
    "Sorcery",
    text="Tap target creature. It doesn't untap during its controller's next untap step. Escape {2}{U}, exile three other cards.",
    targets=(T("creature"),),
    effect=_sleep,
    escape=M("{2}{U}"),
    escape_exile=3,
)
card(
    "Plunder the Trollshaws",
    "{1}{U}",
    "Instant",
    text="Draw a card. If cast from a graveyard, draw two instead. Flashback {3}{U}.",
    effect=lambda g, it: g.draw(it.controller, 2 if it.cast_from == "graveyard" else 1),
    flashback=M("{3}{U}"),
)

# ---------------------------------------------------------------------------
# Jund Wildfire
# ---------------------------------------------------------------------------


def _familiar_etb(g, item):
    p = item.controller
    opp = 1 - p
    if not g.players[opp].hand:
        g.draw(p)
        return
    c = yield from g.ask(opp, CHOOSE_CARD, "Refurbished Familiar: discard a card", g._hand_card_options(opp, "discard"))
    g.discard(c)


def _artifacts_you_control(g, p) -> int:
    return sum(1 for c in g.battlefield if c.controller == p and g.is_artifact(c))


card(
    "Refurbished Familiar",
    "{3}{B}",
    "Artifact Creature",
    "Zombie Rat",
    text="Affinity for artifacts. Flying. ETB: each opponent discards a card; for each who can't, you draw.",
    power=2,
    toughness=1,
    keywords=frozenset({"flying"}),
    cost_reduction=_artifacts_you_control,
    triggers=(TriggerDef("opponent discards", "etb", _familiar_etb),),
)


def _chrysalis_cast(g, item):
    g.create_token(item.controller, "Eldrazi Spawn")
    g.create_token(item.controller, "Eldrazi Spawn")


def _add_counter_to_source(g, item):
    live = g.live(item.source)
    if live is not None:
        live.counters += 1


card(
    "Writhing Chrysalis",
    "{2}{R}{G}",
    "Creature",
    "Eldrazi Drone",
    text="Devoid. When you cast this, create two Eldrazi Spawn. Reach. Whenever you sacrifice another Eldrazi, +1/+1 counter.",
    devoid=True,
    power=2,
    toughness=3,
    keywords=frozenset({"reach", "devoid"}),
    triggers=(
        TriggerDef("create two Eldrazi Spawn", "cast", _chrysalis_cast),
        TriggerDef("+1/+1 counter", "you_sacrifice_another", _add_counter_to_source, condition=lambda g, src, lki: "Eldrazi" in lki.face.subtypes),
    ),
)


def _kcs(g, item):
    for c in list(g.battlefield):
        if g.is_creature(c) and not g.has(c, "flying"):
            g.deal_damage(item.source, ("perm", c.oid), 1)


card(
    "Krark-Clan Shaman",
    "{R}",
    "Creature",
    "Goblin Shaman",
    text="Sacrifice an artifact: deals 1 damage to each creature without flying.",
    power=1,
    toughness=1,
    abilities=(AbilityDef(name="1 damage to each creature without flying", sac_other="artifact", effect=_kcs),),
)
card(
    "Nyxborn Hydra",
    "{X}{G}",
    "Enchantment Creature",
    "Hydra",
    text="Bestow {X}{G}{G}. Reach, trample. Enters with X +1/+1 counters. Enchanted creature gets +1/+1 per counter and has reach and trample.",
    power=0,
    toughness=1,
    keywords=frozenset({"reach", "trample"}),
    bestow=M("{X}{G}{G}"),
    etb_x_counters=True,
)


def _wildfire(g, item):
    t = g.target(item)
    if t is not None:
        controller = t.controller
        g.destroy(t)
        yield from g.search_library(controller, _is_basic, "battlefield", "a basic land card", tapped=True)
    g.draw(item.controller)


card(
    "Cleansing Wildfire",
    "{1}{R}",
    "Sorcery",
    text="Destroy target land. Its controller may search for a basic land, put it onto the battlefield tapped. Draw a card.",
    targets=(T("land"),),
    effect=_wildfire,
)


def _destroy_target(g, item):
    t = g.target(item)
    if t is not None:
        g.destroy(t)


card("Cast Down", "{1}{B}", "Instant", text="Destroy target nonlegendary creature.", targets=(T("nonlegendary_creature"),), effect=_destroy_target)


def _offering(g, item):
    g.draw(item.controller, 2)
    g.create_token(item.controller, "Map")


card(
    "Fanatical Offering",
    "{1}{B}",
    "Instant",
    text="As an additional cost, sacrifice an artifact or creature. Draw two cards and create a Map token.",
    additional_sac="artifact_or_creature",
    effect=_offering,
)
card(
    "Eviscerator's Insight",
    "{1}{B}",
    "Instant",
    text="As an additional cost, sacrifice an artifact or creature. Draw two cards. Flashback {4}{B}.",
    additional_sac="artifact_or_creature",
    effect=lambda g, it: g.draw(it.controller, 2),
    flashback=M("{4}{B}"),
)


def _toxin(g, item):
    t = g.target(item)
    if t is None:
        return
    t.temp.append(TempEffect(keywords=frozenset({"deathtouch", "lifelink"})))
    g.create_token(item.controller, "Clue")


card(
    "Toxin Analysis",
    "{B}",
    "Instant",
    text="Target creature gains deathtouch and lifelink until end of turn. Investigate.",
    targets=(T("creature"),),
    effect=_toxin,
)


def _munitions(g, item):
    if g.target_legal(item, 0):
        g.deal_damage(item.source, item.targets[0], 1)


card(
    "Makeshift Munitions",
    "{1}{R}",
    "Enchantment",
    text="{1}, Sacrifice an artifact or creature: 1 damage to any target.",
    abilities=(AbilityDef(name="1 damage to any target", cost=M("{1}"), sac_other="artifact_or_creature", targets=(T("any"),), effect=_munitions),),
)
card(
    "Ichor Wellspring",
    "{2}",
    "Artifact",
    text="When this enters or is put into a graveyard from the battlefield, draw a card.",
    triggers=(
        TriggerDef("draw a card", "etb", lambda g, it: g.draw(it.controller)),
        TriggerDef("draw a card", "to_graveyard_from_battlefield", lambda g, it: g.draw(it.controller)),
    ),
)


def _lembas_etb(g, item):
    yield from g.scry(item.controller, 1)
    g.draw(item.controller)


def _lembas_shuffle(g, item):
    c = item.data["card"]
    if c.zone == "graveyard" and c.oid == item.data["oid"]:
        g._move(c, "library")
        g.shuffle(c.owner)


card(
    "Lembas",
    "{2}",
    "Artifact",
    "Food",
    text="ETB: scry 1, then draw. {2}, {T}, Sacrifice: gain 3 life. When put into a graveyard from the battlefield, its owner shuffles it into their library.",
    abilities=(AbilityDef(name="gain 3 life", cost=M("{2}"), tap=True, sac_self=True, effect=lambda g, it: g.gain_life(it.controller, 3)),),
    triggers=(
        TriggerDef("scry 1, draw", "etb", _lembas_etb),
        TriggerDef("shuffle into library", "to_graveyard_from_battlefield", _lembas_shuffle),
    ),
)


def _spellbomb_exile(g, item):
    t = g.target(item)
    if t is None:
        return
    gy = g.players[t[1]].graveyard
    for c in list(gy):
        g._move(c, "exile")


def _spellbomb_dies(g, item):
    paid = yield from g.optional_payment(item.controller, M("{B}"), "Nihil Spellbomb: pay {B} to draw a card?")
    if paid:
        g.draw(item.controller)


card(
    "Nihil Spellbomb",
    "{1}",
    "Artifact",
    text="{T}, Sacrifice: exile target player's graveyard. When put into a graveyard from the battlefield, you may pay {B}; if you do, draw.",
    abilities=(AbilityDef(name="exile target player's graveyard", tap=True, sac_self=True, targets=(T("player"),), effect=_spellbomb_exile),),
    triggers=(TriggerDef("pay {B}: draw", "to_graveyard_from_battlefield", _spellbomb_dies),),
)
card(
    "Gixian Infiltrator",
    "{1}{B}",
    "Creature",
    "Phyrexian Human",
    text="Whenever you sacrifice another permanent, put a +1/+1 counter on this creature.",
    power=2,
    toughness=1,
    triggers=(TriggerDef("+1/+1 counter", "you_sacrifice_another", _add_counter_to_source),),
)

# ---------------------------------------------------------------------------
# Sideboards (lean, matchup-relevant; see decks.py)
# ---------------------------------------------------------------------------


def _counter_target(g, item):
    t = g.target(item)
    if t is not None:
        g.counter(t)


card(
    "Red Elemental Blast",
    "{R}",
    "Instant",
    text="Choose one - Counter target blue spell; or destroy target blue permanent.",
    modes=(
        SpellMode("counter", (T("blue_spell"),), _counter_target),
        SpellMode("destroy", (T("blue_permanent"),), _destroy_target),
    ),
)
card(
    "Blue Elemental Blast",
    "{U}",
    "Instant",
    text="Choose one - Counter target red spell; or destroy target red permanent.",
    modes=(
        SpellMode("counter", (T("red_spell"),), _counter_target),
        SpellMode("destroy", (T("red_permanent"),), _destroy_target),
    ),
)
card("Go for the Throat", "{1}{B}", "Instant", text="Destroy target creature that isn't an artifact creature.", targets=(T("nonartifact_creature"),), effect=_destroy_target)


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


card(
    "Duress",
    "{B}",
    "Sorcery",
    text="Target opponent reveals their hand. You choose a noncreature, nonland card from it. That player discards that card.",
    targets=(T("opponent"),),
    effect=_duress,
)
card("Dispel", "{U}", "Instant", text="Counter target instant spell.", targets=(T("instant_spell"),), effect=_counter_target)


def _bounce_artifact(g, item):
    t = g.target(item)
    if t is not None:
        g._move(t, "hand")


card(
    "Steel Sabotage",
    "{U}",
    "Instant",
    text="Choose one - Counter target artifact spell; or return target artifact to its owner's hand.",
    modes=(
        SpellMode("counter", (T("artifact_spell"),), _counter_target),
        SpellMode("bounce", (T("artifact"),), _bounce_artifact),
    ),
)

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

card(
    "Eldrazi Spawn",
    None,
    "Creature",
    "Eldrazi Spawn",
    text="Sacrifice this token: Add {C}.",
    power=0,
    toughness=1,
    abilities=(AbilityDef(name="sacrifice: add C", sac_self=True, mana=("C",)),),
    registry=TOKENS,
)
card(
    "Clue",
    None,
    "Artifact",
    "Clue",
    text="{2}, Sacrifice: draw a card.",
    abilities=(AbilityDef(name="draw a card", cost=M("{2}"), sac_self=True, effect=lambda g, it: g.draw(it.controller)),),
    registry=TOKENS,
)


def _map_explore(g, item):
    t = g.target(item)
    if t is not None:
        yield from g.explore(t)


card(
    "Map",
    None,
    "Artifact",
    "Map",
    text="{1}, {T}, Sacrifice: target creature you control explores. Sorcery speed.",
    abilities=(
        AbilityDef(
            name="explore",
            cost=M("{1}"),
            tap=True,
            sac_self=True,
            sorcery_speed=True,
            targets=(T("creature_you_control"),),
            effect=_map_explore,
        ),
    ),
    registry=TOKENS,
)
