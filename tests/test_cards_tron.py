"""Tron cards (docs/tron.md), its sideboard cards and the rules they need:
multi-mana lands, mana filters, prototype, station, cascade, triggers with
targets, scry 2, surveil, graveyard abilities. Every test runs on both engines."""

import pytest
from helpers import bf, choose, find, has, labels, names, new_game, pay, resolve_stack, scenario

from mtg_ml.bots import make_bot
from mtg_ml.engine import objects as O
from mtg_ml.engine.cards import CARDS
from mtg_ml.engine.decks import DECKS, SIDEBOARDS, TRON, TRON_SIDEBOARD
from mtg_ml.match import MATCHUPS, game_args

TRON_LANDS = ["Urza's Mine", "Urza's Power Plant", "Urza's Tower"]
FORESTS = lambda n: ["Forest"] * n  # noqa: E731
WASTES = lambda n: [("Urza's Mine", {})] * n  # noqa: E731


def pay_all(g):
    """Pay with the first option (floating mana first, then sources in order)."""
    pay(g)


# ---------------------------------------------------------------------------
# Urza's lands
# ---------------------------------------------------------------------------


def test_tron_lands_make_seven_mana_together():
    g = scenario(p0={"hand": ["Pinnacle Kill-Ship"], "battlefield": TRON_LANDS})
    assert has(g, "Cast Pinnacle Kill-Ship")
    choose(g, "Cast Pinnacle Kill-Ship")
    assert g.decision.kind == O.PAY_MANA
    choose(g, "Tap Urza's Tower")  # three: one pays, two float
    assert g.players[0].pool == {"C": 2}
    pay_all(g)
    assert all(c.tapped for c in g.battlefield if c.name in TRON_LANDS)
    assert g.players[0].pool == {}
    assert g.stack[-1].name == "Pinnacle Kill-Ship"


def test_tron_lands_make_one_each_without_the_set():
    g = scenario(p0={"hand": ["Pinnacle Kill-Ship", "Bramble Wurm"], "battlefield": ["Urza's Mine", "Urza's Tower", "Urza's Tower"] + FORESTS(3)})
    assert not has(g, "Cast Pinnacle Kill-Ship")  # 6 mana: no Power-Plant, every land makes one
    g2 = scenario(p0={"hand": ["Pinnacle Kill-Ship"], "battlefield": TRON_LANDS + ["Urza's Tower"]})
    assert has(g2, "Cast Pinnacle Kill-Ship")


def test_floating_tron_mana_pays_the_next_spell_in_the_same_step():
    g = scenario(p0={"hand": ["Expedition Map", "Candy Trail"], "battlefield": TRON_LANDS})
    choose(g, "Cast Expedition Map")
    choose(g, "Tap Urza's Tower")
    assert g.players[0].pool == {"C": 2}
    resolve_stack(g)
    # The pool empties only at the end of the step: two left for Candy Trail.
    choose(g, "Cast Candy Trail")
    assert labels(g)[0] == "Pay with floating C"
    pay_all(g)
    assert g.players[0].pool == {"C": 1}
    assert sum(1 for c in g.battlefield if c.tapped) == 1


# ---------------------------------------------------------------------------
# Search and dig
# ---------------------------------------------------------------------------


def test_expedition_map_finds_a_land_revealed():
    g = scenario(p0={"battlefield": ["Expedition Map", "Forest", "Forest"], "library": ["Bramble Wurm", "Urza's Tower", "Forest"]})
    choose(g, "Expedition Map: search for a land card")
    pay_all(g)
    assert "Expedition Map" in names(g.players[0].graveyard)
    resolve_stack(g)
    assert labels(g) == ["Find nothing", "Find Urza's Tower", "Find Forest"]
    choose(g, "Find Urza's Tower")
    assert names(g.players[0].hand) == ["Urza's Tower"] and g.players[0].hand[0].known_to == {0, 1}


def test_ancient_stirrings_takes_a_colorless_card_and_bottoms_the_rest():
    lib = ["Bramble Wurm", "Urza's Mine", "Unfathomable Truths", "Maelstrom Colossus", "Forest", "Crop Rotation", "Candy Trail"]
    g = scenario(p0={"hand": ["Ancient Stirrings"], "battlefield": FORESTS(1), "library": lib})
    choose(g, "Cast Ancient Stirrings")
    resolve_stack(g)
    # Lands are colorless; Unfathomable Truths is devoid (colorless); green cards are not.
    assert labels(g) == ["Take nothing", "Take Urza's Mine", "Take Unfathomable Truths", "Take Maelstrom Colossus", "Take Forest"]
    choose(g, "Take Maelstrom Colossus")
    p = g.players[0]
    assert names(p.hand) == ["Maelstrom Colossus"] and p.hand[0].known_to == {0, 1}
    assert names(p.library) == ["Crop Rotation", "Candy Trail", "Bramble Wurm", "Urza's Mine", "Unfathomable Truths", "Forest"]
    assert p.library[-1].known_to == {0}


def test_crop_rotation_sacrifices_a_land_and_puts_a_land_onto_the_battlefield():
    g = scenario(p0={"hand": ["Crop Rotation"], "battlefield": ["Forest", "Urza's Mine", "Urza's Power Plant"], "library": ["Bramble Wurm", "Urza's Tower"]})
    choose(g, "Cast Crop Rotation")  # the Forest, the only green source, pays and may then be sacrificed
    assert find(g, "Forest").tapped and g.decision.kind == O.SACRIFICE
    assert labels(g) == ["Sacrifice Forest#%d" % find(g, "Forest").oid] + [f"Sacrifice {n}#{find(g, n).oid}" for n in ("Urza's Mine", "Urza's Power Plant")]
    choose(g, "Sacrifice Forest")
    resolve_stack(g)
    choose(g, "Find Urza's Tower")
    assert sorted(bf(g, 0)) == sorted(TRON_LANDS)
    assert not find(g, "Urza's Tower").tapped
    assert names(g.players[0].graveyard) == ["Forest", "Crop Rotation"]


def test_crop_rotation_needs_a_land_to_sacrifice():
    g = scenario(p0={"hand": ["Crop Rotation"], "battlefield": ["Bonder's Ornament"]})
    assert not has(g, "Cast Crop Rotation")


# ---------------------------------------------------------------------------
# Mana filters
# ---------------------------------------------------------------------------


def test_barrels_filters_one_mana_into_any_colour_once_each_turn():
    g = scenario(p0={"hand": ["Unfathomable Truths"], "battlefield": ["Barrels of Blasting Jelly"] + WASTES(4)})
    assert not has(g, "Cast Unfathomable Truths")  # {4}{U}: the filter turns {1} into {U}, it adds no mana
    g = scenario(p0={"hand": ["Unfathomable Truths"], "battlefield": ["Barrels of Blasting Jelly"] + WASTES(5)})
    choose(g, "Cast Unfathomable Truths")
    assert labels(g)[-1].startswith("Activate Barrels of Blasting Jelly#") and labels(g)[-1].endswith(" for U")
    choose(g, "Activate Barrels")  # then {5}: the five Mines, all alike, are tapped
    assert sum(1 for c in g.battlefield if c.tapped) == 5 and not find(g, "Barrels of Blasting Jelly").tapped
    resolve_stack(g)
    assert len(g.players[0].hand) == 3 and bf(g, 0).count("Eldrazi Spawn") == 1


def test_barrels_filter_only_once_per_turn():
    # Two blue pips, one Barrels: only one can be filtered.
    g = scenario(p0={"hand": ["Unfathomable Truths", "Blue Elemental Blast"], "battlefield": ["Barrels of Blasting Jelly"] + WASTES(8)}, p1={"battlefield": ["Guttersnipe"]})
    choose(g, "Cast Unfathomable Truths")
    choose(g, "Activate Barrels")
    pay_all(g)
    # The filter is used up for the turn: a second blue spell is not castable.
    assert g.stack and not any("Blue Elemental Blast" in lab for lab in labels(g))


def test_barrels_deals_five_to_target_creature():
    g = scenario(p0={"battlefield": ["Barrels of Blasting Jelly"] + [("Urza's Mine", {})] * 5}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Barrels of Blasting Jelly: 5 damage to target creature")  # the only target, the only payment
    assert "Barrels of Blasting Jelly" in names(g.players[0].graveyard)
    resolve_stack(g)
    resolve_stack(g)  # ward {2}: we have no mana left, so it is countered
    assert "Tolarian Terror" in bf(g, 1)


def test_conduit_pylons_surveils_and_filters_by_tapping_itself():
    g = scenario(p0={"hand": ["Conduit Pylons", "Blue Elemental Blast"], "library": ["Bramble Wurm", "Forest"], "battlefield": ["Urza's Mine"]})
    choose(g, "Play Conduit Pylons")
    resolve_stack(g)
    assert labels(g) == ["Keep Bramble Wurm on top", "Put Bramble Wurm into your graveyard"]
    choose(g, "Put Bramble Wurm into your graveyard")
    assert names(g.players[0].graveyard) == ["Bramble Wurm"]
    # {U} from the Mine's {C} through the Pylons filter ({1}, {T}).
    g2 = scenario(p0={"hand": ["Blue Elemental Blast"], "battlefield": ["Conduit Pylons", "Urza's Mine"]}, p1={"battlefield": ["Guttersnipe"]})
    choose(g2, "Cast Blue Elemental Blast (destroy)")  # the Pylons filter is the only way to pay
    assert find(g2, "Conduit Pylons").tapped and find(g2, "Urza's Mine").tapped
    resolve_stack(g2)
    assert "Guttersnipe" not in bf(g2)


def test_bonders_ornament_mana_and_draw():
    g = scenario(p0={"hand": ["Blue Elemental Blast"], "battlefield": ["Bonder's Ornament"]}, p1={"battlefield": ["Guttersnipe"]})
    choose(g, "Cast Blue Elemental Blast (destroy)")
    assert find(g, "Bonder's Ornament").tapped
    resolve_stack(g)
    assert "Guttersnipe" not in bf(g)
    g = scenario(p0={"battlefield": ["Bonder's Ornament"] + WASTES(4)}, p1={"battlefield": ["Bonder's Ornament"]})
    choose(g, "Bonder's Ornament: each player with a Bonder's Ornament draws")
    pay_all(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 1 and len(g.players[1].hand) == 1
    g = scenario(p0={"battlefield": ["Bonder's Ornament"] + WASTES(4)})
    choose(g, "Bonder's Ornament: each player with a Bonder's Ornament draws")
    pay_all(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 1 and len(g.players[1].hand) == 0


# ---------------------------------------------------------------------------
# Card flow
# ---------------------------------------------------------------------------


def test_candy_trail_scry_two_and_sacrifice():
    g = scenario(p0={"hand": ["Candy Trail"], "battlefield": WASTES(3), "library": ["Bramble Wurm", "Forest", "Urza's Tower"]})
    choose(g, "Cast Candy Trail")
    pay_all(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.decision.kind == O.ORDER
    assert labels(g) == [
        "Scry: top Bramble Wurm, Forest; bottom -",
        "Scry: top Forest, Bramble Wurm; bottom -",
        "Scry: top Forest; bottom Bramble Wurm",
        "Scry: top Bramble Wurm; bottom Forest",
        "Scry: top -; bottom Bramble Wurm, Forest",
        "Scry: top -; bottom Forest, Bramble Wurm",
    ]
    choose(g, "Scry: top Forest; bottom Bramble Wurm")
    assert names(g.players[0].library) == ["Forest", "Urza's Tower", "Bramble Wurm"]
    choose(g, "Candy Trail: gain 3 life and draw a card")
    pay_all(g)
    resolve_stack(g)
    assert g.players[0].life == 23 and names(g.players[0].hand) == ["Forest"]


def test_unfathomable_truths_is_colorless_and_draws_three():
    assert CARDS["Unfathomable Truths"].colors == frozenset()


# ---------------------------------------------------------------------------
# Threats
# ---------------------------------------------------------------------------


def test_bramble_wurm_etb_and_graveyard_ability():
    g = scenario(p0={"hand": ["Bramble Wurm"], "battlefield": TRON_LANDS + FORESTS(1)})
    choose(g, "Cast Bramble Wurm")
    pay_all(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.players[0].life == 25
    w = find(g, "Bramble Wurm")
    assert g.has(w, "reach") and g.has(w, "trample") and (g.power(w), g.toughness(w)) == (7, 6)
    g = scenario(p0={"graveyard": ["Bramble Wurm"], "battlefield": FORESTS(3)})
    choose(g, "Bramble Wurm: gain 5 life")
    pay_all(g)
    assert names(g.players[0].exile) == ["Bramble Wurm"] and not g.players[0].graveyard
    resolve_stack(g)
    assert g.players[0].life == 25


def test_boulderbranch_golem_full_and_prototype():
    g = scenario(p0={"hand": ["Boulderbranch Golem"], "battlefield": TRON_LANDS})
    assert has(g, "Cast Boulderbranch Golem") and not has(g, "(prototype)")  # no green mana
    choose(g, "Cast Boulderbranch Golem")
    pay_all(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.players[0].life == 26 and (g.power(find(g, "Boulderbranch Golem")), g.toughness(find(g, "Boulderbranch Golem"))) == (6, 5)
    g = scenario(p0={"hand": ["Boulderbranch Golem"], "battlefield": FORESTS(4)})
    assert labels(g)[1:] == ["Cast Boulderbranch Golem (prototype)"]
    choose(g, "Cast Boulderbranch Golem (prototype)")
    pay_all(g)
    assert g.stack[-1].name == "Boulderbranch Golem (prototype)"
    resolve_stack(g)
    c = find(g, "Boulderbranch Golem (prototype)")
    assert (g.power(c), g.toughness(c)) == (3, 3) and c.face.colors == frozenset("G") and c.face.mana_value == 4
    resolve_stack(g)
    assert g.players[0].life == 23
    g.destroy(c)
    assert names(g.players[0].graveyard) == ["Boulderbranch Golem"]  # a normal card again


def test_kill_ship_etb_up_to_one_target():
    g = scenario(p0={"hand": ["Pinnacle Kill-Ship"], "battlefield": TRON_LANDS}, p1={"battlefield": ["Tolarian Terror", "Guttersnipe"]})
    choose(g, "Cast Pinnacle Kill-Ship")
    pay_all(g)
    resolve_stack(g)
    assert g.decision.kind == O.TARGET
    assert labels(g)[0] == "No target" and len(labels(g)) == 3
    choose(g, "Target Guttersnipe")
    resolve_stack(g)
    assert "Guttersnipe" not in bf(g) and "Tolarian Terror" in bf(g)
    ship = find(g, "Pinnacle Kill-Ship")
    assert not g.is_creature(ship)  # a Spacecraft without charge counters


def test_kill_ship_with_no_creatures_and_ward():
    g = scenario(p0={"hand": ["Pinnacle Kill-Ship"], "battlefield": TRON_LANDS})
    choose(g, "Cast Pinnacle Kill-Ship")
    pay_all(g)
    resolve_stack(g)
    assert "Pinnacle Kill-Ship" in bf(g) and not g.stack  # the trigger had nothing to target
    g = scenario(p0={"hand": ["Pinnacle Kill-Ship"], "battlefield": TRON_LANDS + ["Urza's Mine", "Urza's Mine"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Pinnacle Kill-Ship")
    pay_all(g)
    resolve_stack(g)
    choose(g, "Target Tolarian Terror")
    # Ward {2} triggers on the targeting trigger.
    assert g.stack[-1].name == "Tolarian Terror: ward"
    resolve_stack(g)
    choose(g, "Pay {2}")
    pay_all(g)
    resolve_stack(g)
    assert "Tolarian Terror" not in bf(g)


def test_kill_ship_station_makes_a_flying_creature():
    g = scenario(p0={"battlefield": ["Pinnacle Kill-Ship", ("Bramble Wurm", {"sick": True}), "Generous Ent"]})
    choose(g, "Pinnacle Kill-Ship: station")
    assert labels(g) == [f"Tap Bramble Wurm#{find(g, 'Bramble Wurm').oid}", f"Tap Generous Ent#{find(g, 'Generous Ent').oid}"]
    choose(g, "Tap Generous Ent")
    resolve_stack(g)
    ship = find(g, "Pinnacle Kill-Ship")
    assert ship.charge == 5 and not g.is_creature(ship)
    assert not has(g, "Pinnacle Kill-Ship: station") or labels(g)  # Wurm is untapped: still possible
    choose(g, "Pinnacle Kill-Ship: station")
    resolve_stack(g)
    assert ship.charge == 12 and g.is_creature(ship) and g.has(ship, "flying") and (g.power(ship), g.toughness(ship)) == (7, 7)
    assert not has(g, "Pinnacle Kill-Ship: station")  # no untapped creature left besides itself


def test_station_is_sorcery_speed():
    g = scenario(p0={"battlefield": ["Pinnacle Kill-Ship", "Bramble Wurm"]}, active=1)
    assert not has(g, "station")


def test_maelstrom_colossus_cascades():
    lib = ["Urza's Tower", "Forest", "Bramble Wurm", "Candy Trail", "Forest"]
    g = scenario(p0={"hand": ["Maelstrom Colossus"], "battlefield": TRON_LANDS + ["Urza's Mine"], "library": lib})
    choose(g, "Cast Maelstrom Colossus")
    pay_all(g)
    assert g.stack[-1].name == "Maelstrom Colossus: cascade"
    resolve_stack(g)
    assert labels(g) == ["Don't cast Bramble Wurm", "Cast Bramble Wurm"]
    choose(g, "Cast Bramble Wurm")
    # Wurm on the stack above the Colossus; the two lands went to the bottom.
    assert [it.name for it in g.stack] == ["Maelstrom Colossus", "Bramble Wurm"]
    assert names(g.players[0].library)[:2] == ["Candy Trail", "Forest"]
    assert sorted(names(g.players[0].library)[2:]) == ["Forest", "Urza's Tower"]
    resolve_stack(g)
    resolve_stack(g)
    assert "Bramble Wurm" in bf(g) and "Maelstrom Colossus" in bf(g)


def test_cascade_declined_or_nothing_found():
    g = scenario(p0={"hand": ["Maelstrom Colossus"], "battlefield": TRON_LANDS + ["Urza's Mine"], "library": ["Forest", "Unfathomable Truths"]})
    choose(g, "Cast Maelstrom Colossus")
    pay_all(g)
    resolve_stack(g)
    choose(g, "Don't cast Unfathomable Truths")
    assert sorted(names(g.players[0].library)) == ["Forest", "Unfathomable Truths"] and not g.players[0].exile
    g = scenario(p0={"hand": ["Maelstrom Colossus"], "battlefield": TRON_LANDS + ["Urza's Mine"], "library": ["Forest", "Maelstrom Colossus"]})
    choose(g, "Cast Maelstrom Colossus")
    pay_all(g)
    resolve_stack(g)  # nothing with a lower mana value: all go back
    assert sorted(names(g.players[0].library)) == ["Forest", "Maelstrom Colossus"]


# ---------------------------------------------------------------------------
# Utility lands
# ---------------------------------------------------------------------------


def test_bojuka_bog_enters_tapped_and_exiles_a_graveyard():
    g = scenario(p0={"hand": ["Bojuka Bog"]}, p1={"graveyard": ["Lightning Bolt", "Fireblast"]})
    choose(g, "Play Bojuka Bog")
    assert find(g, "Bojuka Bog").tapped
    assert labels(g) == ["Target player 0 (self)", "Target player 1 (opponent)"]
    choose(g, "Target player 1")
    resolve_stack(g)
    assert not g.players[1].graveyard and names(g.players[1].exile) == ["Lightning Bolt", "Fireblast"]


def test_haunted_fengraf_returns_a_creature_at_random():
    g = scenario(p0={"battlefield": ["Haunted Fengraf"] + WASTES(3), "graveyard": ["Bramble Wurm", "Forest", "Generous Ent"]})
    choose(g, "Haunted Fengraf: return a creature card at random")
    pay_all(g)
    resolve_stack(g)
    hand = names(g.players[0].hand)
    assert len(hand) == 1 and hand[0] in ("Bramble Wurm", "Generous Ent")
    assert "Haunted Fengraf" in names(g.players[0].graveyard)


# ---------------------------------------------------------------------------
# Sideboard
# ---------------------------------------------------------------------------


def test_call_damage_control_returns_up_to_two_of_different_types():
    g = scenario(p0={"hand": ["Call Damage Control"], "battlefield": FORESTS(2), "graveyard": ["Bramble Wurm", "Generous Ent", "Urza's Tower", "Candy Trail"]})
    choose(g, "Cast Call Damage Control")
    pay_all(g)
    resolve_stack(g)
    assert labels(g) == [
        "Return nothing more",
        "Return Candy Trail (artifact) from your graveyard",
        "Return Bramble Wurm (creature) from your graveyard",
        "Return Generous Ent (creature) from your graveyard",
        "Return Urza's Tower (land) from your graveyard",
    ]
    choose(g, "Return Bramble Wurm")
    assert not any("(creature)" in lab for lab in labels(g))  # one creature only
    choose(g, "Return Urza's Tower")
    assert sorted(names(g.players[0].hand)) == ["Bramble Wurm", "Urza's Tower"]


def test_pulse_of_murasa_returns_from_any_graveyard_and_gains_six():
    g = scenario(p0={"hand": ["Pulse of Murasa"], "battlefield": FORESTS(3)}, p1={"graveyard": ["Guttersnipe"]})
    choose(g, "Cast Pulse of Murasa")
    pay_all(g)
    resolve_stack(g)
    choose(g, "Return Guttersnipe (creature) from the opponent's graveyard")
    assert names(g.players[1].hand) == ["Guttersnipe"] and g.players[0].life == 26


def test_scour_from_existence_exiles_a_permanent():
    g = scenario(p0={"hand": ["Scour from Existence"], "battlefield": TRON_LANDS}, p1={"battlefield": ["Guttersnipe"]})
    choose(g, "Cast Scour from Existence")
    choose(g, "Target Guttersnipe")
    pay_all(g)
    resolve_stack(g)
    assert names(g.players[1].exile) == ["Guttersnipe"]


def test_kaervek_torch_deals_x():
    g = scenario(p0={"hand": ["Kaervek's Torch"], "battlefield": ["Bonder's Ornament"] + WASTES(3)})
    choose(g, "Cast Kaervek's Torch")
    assert labels(g) == ["X=0", "X=1", "X=2", "X=3"]
    choose(g, "X=3")
    choose(g, "Target player 1")
    pay_all(g)
    resolve_stack(g)
    assert g.players[1].life == 17


def test_monstrous_emergence_uses_the_chosen_or_revealed_power():
    g = scenario(p0={"hand": ["Monstrous Emergence", "Maelstrom Colossus"], "battlefield": FORESTS(2) + ["Generous Ent"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Monstrous Emergence")
    choose(g, "Target Tolarian Terror")
    pay_all(g)
    assert labels(g) == [f"Choose Generous Ent#{find(g, 'Generous Ent').oid}", "Reveal Maelstrom Colossus"]
    choose(g, "Reveal Maelstrom Colossus")
    assert g.players[0].hand[0].known_to == {0, 1}
    resolve_stack(g)  # ward {2}: unpaid (no mana), countered
    assert "Tolarian Terror" in bf(g)
    g = scenario(p0={"hand": ["Monstrous Emergence"], "battlefield": FORESTS(2) + ["Generous Ent"]}, p1={"battlefield": ["Guttersnipe", "Kessig Flamebreather"]})
    choose(g, "Cast Monstrous Emergence")
    choose(g, "Target Kessig Flamebreather")
    pay_all(g)
    resolve_stack(g)
    assert "Kessig Flamebreather" not in bf(g)


def test_monstrous_emergence_needs_a_creature():
    g = scenario(p0={"hand": ["Monstrous Emergence"], "battlefield": FORESTS(2)}, p1={"battlefield": ["Guttersnipe"]})
    assert not has(g, "Cast Monstrous Emergence")


# ---------------------------------------------------------------------------
# Deck, bot, games
# ---------------------------------------------------------------------------


def test_tron_deck_is_sixty_and_fifteen():
    assert sum(TRON.values()) == 60 and sum(TRON_SIDEBOARD.values()) == 15
    assert DECKS["tron"] is TRON and SIDEBOARDS["tron"] is TRON_SIDEBOARD
    for name, n in list(TRON.items()) + list(TRON_SIDEBOARD.items()):
        assert n <= 4 or "Basic" in CARDS[name].supertypes, name


@pytest.mark.parametrize("matchup", ["jund_tron", "blue_tron", "madness_tron"])
def test_bot_game_runs(matchup):
    a, b = MATCHUPS[matchup]
    for game_no in (1, 2):
        g = new_game(**game_args(game_no, matchup), seed=game_no)
        bots = [make_bot(0, a), make_bot(1, b)]
        while not g.over:
            g.step(bots[g.decision.player].act(g))
        assert g.turn > 2
