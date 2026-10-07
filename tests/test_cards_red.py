"""Red Madness cards, its sideboard and the Blood token."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario, settle

from mtg_ml.engine import objects as O

MOUNTAINS = lambda n: ["Mountain"] * n  # noqa: E731


def test_bolt_any_target_and_flamebreather_guttersnipe_triggers():
    g = scenario(p0={"hand": ["Lightning Bolt"], "battlefield": ["Kessig Flamebreather", "Guttersnipe", "Mountain"]})
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    # Flamebreather (noncreature) and Guttersnipe (instant or sorcery) both trigger.
    assert g.decision.kind == O.ORDER_TRIGGERS
    choose(g, "Guttersnipe")
    resolve_stack(g)
    assert g.players[1].life == 20 - 3 - 1 - 2


def test_creature_spell_does_not_trigger_flamebreather():
    g = scenario(p0={"hand": ["Voldaren Epicure"], "battlefield": ["Kessig Flamebreather", "Guttersnipe", "Mountain"]})
    choose(g, "Cast Voldaren Epicure")
    pay(g)
    assert not g.stack[:-1]  # only the Epicure spell itself
    resolve_stack(g)  # Epicure resolves; its ETB trigger goes on the stack
    resolve_stack(g)
    assert g.players[1].life == 19 and bf(g, 0).count("Blood") == 1


def test_blood_discards_as_cost_and_madness_casts_fiery_temper():
    g = scenario(p0={"hand": ["Fiery Temper"], "battlefield": ["Blood", "Mountain", "Mountain", "Kessig Flamebreather"]})
    choose(g, "Blood: draw a card")
    pay(g)
    # Fiery Temper was the only card: discarded as the cost, into exile.
    assert names(g.players[0].exile) == ["Fiery Temper"] and "Blood" not in bf(g)
    assert g.stack[-1].name == "Fiery Temper: madness"
    resolve_stack(g)
    assert g.decision.kind == O.YES_NO
    assert sorted(labels(g)) == ["Cast Fiery Temper for its madness cost", "Put Fiery Temper into your graveyard"]
    choose(g, "Cast Fiery Temper for its madness cost")
    choose(g, "Target player 1 (opponent)")
    pay(g)  # {R}, not {1}{R}{R}
    assert sum(1 for c in g.battlefield if c.name == "Mountain" and c.tapped) == 2
    resolve_stack(g)
    assert g.players[1].life == 20 - 1 - 3  # Flamebreather trigger and Temper
    assert "Fiery Temper" in names(g.players[0].graveyard)
    assert len(g.players[0].hand) == 1  # Blood's draw


def test_madness_can_be_declined_and_works_for_sorcery_discards():
    g = scenario(p0={"hand": ["Faithless Looting"], "battlefield": ["Mountain"], "library": ["Fiery Temper", "Mountain"] + MOUNTAINS(10)})
    choose(g, "Cast Faithless Looting")
    pay(g)
    resolve_stack(g)
    choose(g, "Discard Fiery Temper")  # then the Mountain, the only card left
    resolve_stack(g)
    # Madness at instant speed with no mana left: only the graveyard option.
    assert names(g.players[0].graveyard) == ["Mountain", "Faithless Looting", "Fiery Temper"]


def test_madness_cast_from_cleanup_discard():
    g = scenario(p0={"hand": ["Fiery Temper"] + MOUNTAINS(7), "battlefield": ["Mountain"]}, step="end")
    pass_priority(g, 2)
    choose(g, "Discard Fiery Temper")
    resolve_stack(g)
    choose(g, "Cast Fiery Temper for its madness cost")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 17


def test_opponent_forced_discard_triggers_madness_for_the_owner():
    g = scenario(p0={"hand": ["Duress"], "battlefield": ["Swamp"]}, p1={"hand": ["Fiery Temper"], "battlefield": ["Mountain"]})
    choose(g, "Cast Duress")
    pay(g)
    resolve_stack(g)  # Duress takes the only noncreature, nonland card
    resolve_stack(g)
    assert g.decision.player == 1 and g.decision.kind == O.YES_NO
    choose(g, "Cast Fiery Temper for its madness cost")
    choose(g, "Target player 0 (opponent)")
    pay(g)
    resolve_stack(g)
    assert g.players[0].life == 17


def test_grab_the_prize_discard_cost_and_damage():
    g = scenario(p0={"hand": ["Grab the Prize", "Lightning Bolt", "Mountain"], "battlefield": MOUNTAINS(2)})
    choose(g, "Cast Grab the Prize")
    pay(g)
    assert sorted(labels(g)) == ["Discard Lightning Bolt", "Discard Mountain"]
    choose(g, "Discard Lightning Bolt")
    resolve_stack(g)
    assert g.players[1].life == 18 and len(g.players[0].hand) == 3
    g = scenario(p0={"hand": ["Grab the Prize", "Mountain"], "battlefield": MOUNTAINS(2)})
    choose(g, "Cast Grab the Prize")
    pay(g)  # the only other card is discarded
    resolve_stack(g)
    assert g.players[1].life == 20  # a land was discarded: no damage
    g = scenario(p0={"hand": ["Grab the Prize"], "battlefield": MOUNTAINS(2)})
    assert not has(g, "Cast Grab the Prize")  # nothing to discard


def test_fireblast_alternative_cost_sacrifices_two_mountains():
    g = scenario(p0={"hand": ["Fireblast"], "battlefield": [("Mountain", {"tapped": True}), "Mountain", "Swamp"]})
    assert [l for l in labels(g) if "Fireblast" in l] == ["Cast Fireblast (alternative)"]
    choose(g, "Cast Fireblast (alternative)")
    choose(g, "Target player 1 (opponent)")
    assert g.decision.kind == O.SACRIFICE and len(labels(g)) == 2  # the tapped or the untapped one
    g.step(0)
    settle(g)  # the second Mountain is the only one left
    resolve_stack(g)
    assert g.players[1].life == 16 and bf(g, 0) == ["Swamp"]
    g = scenario(p0={"hand": ["Fireblast"], "battlefield": ["Mountain", "Swamp"]})
    assert not has(g, "Cast Fireblast")


def test_lava_dart_flashback_sacrifices_a_mountain_and_exiles():
    g = scenario(p0={"graveyard": ["Lava Dart"], "battlefield": ["Mountain"]}, p1={"battlefield": ["Delver of Secrets"]})
    choose(g, "Cast Lava Dart (flashback)")
    choose(g, "Target Delver of Secrets")
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g) and bf(g, 0) == []
    assert names(g.players[0].exile) == ["Lava Dart"] and names(g.players[0].graveyard) == ["Mountain"]


def test_highway_robbery_plot_then_cast_free_next_turn():
    g = scenario(p0={"hand": ["Highway Robbery", "Lightning Bolt"], "battlefield": MOUNTAINS(2)})
    choose(g, "Plot Highway Robbery")
    pay(g)
    assert g.players[0].exile[0].plotted_turn == g.turn and not has(g, "Cast Highway Robbery")
    from mtg_ml.engine.view import observe

    assert observe(g, 1)["opponent"]["exile"] == ["Highway Robbery (plotted)"]
    # Pass to the next own turn.
    while not (g.turn == 3 and g.step_name == "main1"):
        g.step(0)
    choose(g, "Cast Highway Robbery (plotted)")
    resolve_stack(g)
    assert "Discard Lightning Bolt" in labels(g) and "Neither: draw nothing" in labels(g)
    n = len(g.players[0].hand)
    choose(g, "Sacrifice Mountain")
    assert len(g.players[0].hand) == n + 2 and bf(g, 0).count("Mountain") == 1


def test_sneaky_snacker_returns_on_third_draw():
    g = scenario(p0={"hand": ["Faithless Looting"], "graveyard": ["Sneaky Snacker"], "battlefield": ["Mountain"], "drawn": 1})
    choose(g, "Cast Faithless Looting")
    pay(g)
    resolve_stack(g)  # draws cards 2 and 3: Snacker triggers after Looting resolves
    while g.decision.kind == O.CHOOSE_CARD:
        g.step(0)
        settle(g)
    resolve_stack(g)
    s = find(g, "Sneaky Snacker", 0)
    assert s.tapped and "flying" in g.keywords(s)


def test_sneaky_snacker_not_on_second_draw():
    g = scenario(p0={"hand": ["Faithless Looting"], "graveyard": ["Sneaky Snacker"], "battlefield": ["Mountain"]})
    choose(g, "Cast Faithless Looting")
    pay(g)
    resolve_stack(g)
    while g.decision.kind == O.CHOOSE_CARD:
        g.step(0)
        settle(g)
    resolve_stack(g)
    assert "Sneaky Snacker" not in bf(g)


def test_searing_blaze_landfall():
    g = scenario(p0={"hand": ["Searing Blaze", "Mountain"], "battlefield": MOUNTAINS(2)}, p1={"battlefield": ["Writhing Chrysalis"]})
    choose(g, "Play Mountain")
    choose(g, "Cast Searing Blaze")  # targets: the only player with a creature, and the Chrysalis
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 17 and "Writhing Chrysalis" not in bf(g)
    g = scenario(p0={"hand": ["Searing Blaze"], "battlefield": MOUNTAINS(2)}, p1={"battlefield": ["Writhing Chrysalis"]})
    choose(g, "Cast Searing Blaze")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 19 and find(g, "Writhing Chrysalis").damage == 1


def test_electrickery_single_and_overload():
    g = scenario(p0={"hand": ["Electrickery"], "battlefield": MOUNTAINS(2) + ["Voldaren Epicure"]}, p1={"battlefield": ["Eldrazi Spawn", "Krark-Clan Shaman"]})
    assert sorted(l for l in labels(g) if "Electrickery" in l) == ["Cast Electrickery", "Cast Electrickery (overload)"]
    choose(g, "Cast Electrickery (overload)")
    pay(g)
    resolve_stack(g)
    assert bf(g, 1) == [] and "Voldaren Epicure" in bf(g, 0)


def test_gorilla_shaman_x_is_target_mana_value():
    g = scenario(p0={"battlefield": [("Gorilla Shaman", {}), "Mountain", "Mountain", "Mountain"]}, p1={"battlefield": ["Clue", "Ichor Wellspring", "Nihil Spellbomb"]})
    choose(g, "Gorilla Shaman: destroy target noncreature artifact")
    # {X}{X}{1}: Clue (0) costs 1, Spellbomb (1) costs 3, Wellspring (2) costs 5: not affordable.
    assert sorted(labels(g)) == ["Target Clue#%d (opponent)" % find(g, "Clue").oid, "Target Nihil Spellbomb#%d (opponent)" % find(g, "Nihil Spellbomb").oid]
    choose(g, "Target Nihil Spellbomb")
    pay(g)
    resolve_stack(g)
    assert "Nihil Spellbomb" not in bf(g) and all(c.tapped for c in g.battlefield if c.name == "Mountain")


def test_martyr_of_ashes_reveals_red_cards():
    g = scenario(
        p0={"hand": ["Lightning Bolt", "Fireblast", "Mountain"], "battlefield": ["Martyr of Ashes"] + MOUNTAINS(2)},
        p1={"battlefield": ["Gixian Infiltrator", "Refurbished Familiar"]},
    )
    choose(g, "Martyr of Ashes: X damage")
    assert labels(g) == ["X=0", "X=1", "X=2"]
    choose(g, "X=2")
    pay(g)
    choose(g, "Reveal Fireblast")
    resolve_stack(g)
    assert bf(g, 1) == ["Refurbished Familiar"]  # flying: spared
    assert all(c.known_to == {0, 1} for c in g.players[0].hand if c.name != "Mountain")


def test_pyroblast_only_hits_blue():
    g = scenario(p0={"hand": ["Pyroblast"], "battlefield": ["Mountain"]}, p1={"battlefield": ["Delver of Secrets", "Krark-Clan Shaman"]})
    choose(g, "Cast Pyroblast (destroy)")
    choose(g, "Target Krark-Clan Shaman")  # legal target, nothing happens
    pay(g)
    resolve_stack(g)
    assert sorted(bf(g, 1)) == ["Delver of Secrets", "Krark-Clan Shaman"]
    g = scenario(p0={"hand": ["Pyroblast"], "battlefield": ["Mountain"]}, p1={"battlefield": ["Delver of Secrets"]})
    choose(g, "Cast Pyroblast (destroy)")
    choose(g, "Target Delver of Secrets")
    pay(g)
    resolve_stack(g)
    assert bf(g, 1) == []


def test_relic_of_progenitus_both_abilities():
    g = scenario(p0={"battlefield": ["Relic of Progenitus", "Mountain"], "graveyard": ["Lava Dart"]}, p1={"graveyard": ["Ponder", "Brainstorm"]})
    choose(g, "Relic of Progenitus: target player exiles")
    choose(g, "Target player 1 (opponent)")
    resolve_stack(g)
    assert g.decision.player == 1
    choose(g, "Exile Ponder")
    assert names(g.players[1].exile) == ["Ponder"]
    choose(g, "Relic of Progenitus: exile all graveyards")
    pay(g)
    resolve_stack(g)
    assert not g.players[0].graveyard and not g.players[1].graveyard
    assert "Relic of Progenitus" in names(g.players[0].exile) and len(g.players[0].hand) == 1
