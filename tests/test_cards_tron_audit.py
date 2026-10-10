"""Tron audit (docs/tron-audit.md): interactions that could make Tron too strong
if the engine got them wrong. Every test runs on both engines."""

import pytest
from helpers import choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario


TRON_LANDS = ["Urza's Mine", "Urza's Power Plant", "Urza's Tower"]


def available(g, p=0):
    """Mana the player can make this instant: the pool plus every untapped land's units."""
    units = sum(g.players[p].pool.values())
    for c, ab in g.mana_sources(p):
        units += g.mana_amount(c, ab)
    return units


def test_assembled_tron_is_exactly_seven_and_needs_all_three_kinds():
    assert available(scenario(p0={"battlefield": TRON_LANDS})) == 7
    assert available(scenario(p0={"battlefield": TRON_LANDS + ["Forest"]})) == 8
    # four lands, two of the same kind: still only the three different ones count together
    assert available(scenario(p0={"battlefield": ["Urza's Mine", "Urza's Mine", "Urza's Tower", "Urza's Tower"]})) == 4
    assert available(scenario(p0={"battlefield": ["Urza's Mine", "Urza's Power Plant"]})) == 2
    # two sets are not a double bonus: each land still makes its own 2/2/3
    assert available(scenario(p0={"battlefield": TRON_LANDS * 2})) == 14


def test_opponents_urza_lands_do_not_complete_my_set():
    g = scenario(p0={"battlefield": ["Urza's Mine", "Urza's Tower"]}, p1={"battlefield": ["Urza's Power Plant"]})
    assert available(g, 0) == 2


def test_tapped_urza_lands_still_complete_the_set_and_losing_one_breaks_it():
    g = scenario(p0={"battlefield": [("Urza's Mine", {"tapped": True}), ("Urza's Power Plant", {"tapped": True}), "Urza's Tower"]})
    assert available(g) == 3  # the Tower makes three because the tapped Mine and Plant are still there
    g = scenario(p0={"battlefield": [("Urza's Mine", {"tapped": True}), "Urza's Tower"]})
    assert available(g) == 1


def test_floating_mana_is_gone_after_the_step():
    g = scenario(p0={"hand": ["Expedition Map"], "battlefield": TRON_LANDS})
    choose(g, "Cast Expedition Map")
    choose(g, "Tap Urza's Tower")
    assert g.players[0].pool == {"C": 2}
    resolve_stack(g)
    while g.step_name == "main1":
        pass_priority(g)
    assert g.players[0].pool == {}


@pytest.mark.python_only
def test_barrels_filter_is_available_again_next_turn():
    g = scenario(p0={"battlefield": ["Barrels of Blasting Jelly", "Forest"]})
    barrels = find(g, "Barrels of Blasting Jelly")
    barrels.mana_used_turn = g.turn
    assert not g.mana_filters(0)
    g.turn += 1
    assert g.mana_filters(0)


def test_crop_rotation_does_not_use_the_land_drop_and_a_fetched_bog_is_tapped():
    g = scenario(p0={"hand": ["Crop Rotation", "Forest"], "battlefield": ["Forest"], "library": ["Bojuka Bog"]})
    choose(g, "Cast Crop Rotation")
    resolve_stack(g)
    choose(g, "Find Bojuka Bog")
    assert find(g, "Bojuka Bog").tapped
    assert g.lands_played == 0  # the fetch is not a land drop


def test_crop_rotation_into_conduit_pylons_surveils():
    g = scenario(p0={"hand": ["Crop Rotation"], "battlefield": ["Forest"], "library": ["Conduit Pylons", "Bramble Wurm"]})
    choose(g, "Cast Crop Rotation")
    resolve_stack(g)
    choose(g, "Find Conduit Pylons")
    resolve_stack(g)
    assert any("Surveil 1" in lab or "Keep" in lab for lab in labels(g))


def test_generous_ent_forestcycling_only_finds_a_forest_card():
    g = scenario(p0={"hand": ["Generous Ent"], "battlefield": ["Urza's Mine"], "library": ["Urza's Tower", "Forest", "Bramble Wurm"]})
    choose(g, "Generous Ent: forestcycling {1}")
    pay(g)
    resolve_stack(g)
    assert labels(g) == ["Find nothing", "Find Forest"]


def test_station_cannot_tap_the_ship_itself_and_six_counters_are_not_a_creature():
    g = scenario(p0={"battlefield": ["Pinnacle Kill-Ship"]})
    assert not has(g, "station")
    g = scenario(p0={"battlefield": ["Pinnacle Kill-Ship", "Boulderbranch Golem"]})
    choose(g, "Pinnacle Kill-Ship: station")
    resolve_stack(g)
    ship = find(g, "Pinnacle Kill-Ship")
    assert ship.charge == 6 and not g.is_creature(ship)


def test_cascade_skips_equal_and_higher_mana_value_and_keeps_the_library_whole():
    lib = ["Maelstrom Colossus", "Forest", "Pinnacle Kill-Ship", "Candy Trail", "Urza's Mine"]
    g = scenario(p0={"hand": ["Maelstrom Colossus"], "battlefield": TRON_LANDS + ["Forest"], "library": lib})
    choose(g, "Cast Maelstrom Colossus")
    pay(g)
    resolve_stack(g)
    # the other Colossus (mv 8) and the land are skipped: Kill-Ship (mv 7) is the first hit
    assert labels(g) == ["Don't cast Pinnacle Kill-Ship", "Cast Pinnacle Kill-Ship"]
    choose(g, "Don't cast Pinnacle Kill-Ship")
    assert sorted(names(g.players[0].library)) == sorted(lib)


def test_candy_trail_and_bramble_wurm_gain_exactly_what_they_say():
    g = scenario(p0={"battlefield": ["Candy Trail"] + [("Urza's Mine", {})] * 2, "library": ["Forest"] * 5, "life": 20})
    choose(g, "Candy Trail: gain 3 life and draw a card")
    pay(g)
    resolve_stack(g)
    assert g.players[0].life == 23 and len(g.players[0].hand) == 1
