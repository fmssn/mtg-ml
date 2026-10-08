"""Jund Wildfire cards and tokens."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.engine import objects as O


def test_familiar_affinity_and_discard():
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Vault of Whispers", "Ichor Wellspring", "Drossforge Bridge"]}, p1={"hand": ["Ponder", "Island"]})
    assert has(g, "Cast Refurbished Familiar")  # {3}{B} - 3 artifacts = {B}
    choose(g, "Cast Refurbished Familiar")
    pay(g)
    resolve_stack(g)  # Familiar resolves; its ETB trigger goes on the stack
    resolve_stack(g)
    assert g.decision.player == 1 and g.decision.kind == O.CHOOSE_CARD
    assert sorted(labels(g)) == ["Discard Island", "Discard Ponder"]
    choose(g, "Discard Ponder")
    assert names(g.players[1].graveyard) == ["Ponder"]


def test_familiar_draws_when_opponent_has_no_cards():
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Swamp"] * 4})
    choose(g, "Cast Refurbished Familiar")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 1


def test_familiar_not_castable_without_affinity_mana():
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Swamp"] * 3})
    assert not has(g, "Cast Refurbished Familiar")


def test_fanatical_offering_tap_then_sacrifice_same_land():
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Vault of Whispers", "Swamp"]})
    assert has(g, "Cast Fanatical Offering")
    choose(g, "Cast Fanatical Offering")
    pay(g)  # the only artifact is tapped for mana and then sacrificed
    assert "Vault of Whispers" not in bf(g) and "Vault of Whispers" in names(g.players[0].graveyard)
    resolve_stack(g)
    assert len(g.players[0].hand) == 2 and "Map" in bf(g)


def test_spawn_cannot_pay_and_be_sacrificed():
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Eldrazi Spawn", "Swamp"]})
    assert not has(g, "Cast Fanatical Offering")


def test_spawn_payment_keeps_a_sacrifice_available():
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Eldrazi Spawn", "Swamp", "Swamp"]})
    choose(g, "Cast Fanatical Offering")
    # Sacrificing the spawn for mana would leave nothing to sacrifice: not offered.
    assert all("Spawn" not in l for l in labels(g)), labels(g)
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Eldrazi Spawn", "Eldrazi Spawn", "Swamp"]})
    choose(g, "Cast Fanatical Offering")
    choose(g, "Sacrifice Eldrazi Spawn")  # one spawn for mana ...
    assert all("Spawn" not in l for l in labels(g)), labels(g)  # ... the other must stay
    pay(g)
    assert "Eldrazi Spawn" not in bf(g)


def test_chrysalis_cast_trigger_survives_counterspell():
    g = scenario(
        p0={"hand": ["Writhing Chrysalis"], "battlefield": ["Mountain", "Forest", "Swamp", "Swamp"]},
        p1={"hand": ["Counterspell"], "battlefield": ["Island", "Island"]},
    )
    choose(g, "Cast Writhing Chrysalis")
    pay(g)
    assert g.stack[-1].kind == "trigger"
    pass_priority(g)
    choose(g, "Cast Counterspell")  # the trigger is not a spell: Chrysalis is the only target
    resolve_stack(g)
    assert bf(g, 0).count("Eldrazi Spawn") == 2
    assert "Writhing Chrysalis" in names(g.players[0].graveyard)


def test_chrysalis_counter_from_sacrificed_spawn_and_floating_mana():
    g = scenario(p0={"battlefield": ["Writhing Chrysalis", "Eldrazi Spawn"]})
    choose(g, "Eldrazi Spawn: sacrifice: add C")
    assert g.players[0].pool == {"C": 1}
    resolve_stack(g)
    assert find(g, "Writhing Chrysalis").counters == 1


def test_gixian_counts_every_sacrifice():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator", "Clue", "Swamp", "Swamp"]})
    choose(g, "Clue: draw a card")
    pay(g)
    resolve_stack(g)
    gix = find(g, "Gixian Infiltrator")
    assert gix.counters == 1 and g.power(gix) == 3
    assert len(g.players[0].hand) == 1


def test_krark_clan_shaman_hits_only_non_flyers():
    g = scenario(
        p0={"battlefield": ["Krark-Clan Shaman", "Refurbished Familiar", "Vault of Whispers"]},
        p1={"battlefield": ["Delver of Secrets", "Tolarian Terror"]},
    )
    choose(g, "Krark-Clan Shaman: 1 damage")
    assert g.decision.kind == O.SACRIFICE  # Familiar or Vault
    choose(g, "Sacrifice Vault of Whispers")
    resolve_stack(g)
    assert sorted(bf(g)) == ["Refurbished Familiar", "Tolarian Terror"]
    assert find(g, "Tolarian Terror").damage == 1


def test_nyxborn_hydra_x_and_bestow():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 4 + ["Gixian Infiltrator"]})
    assert has(g, "Cast Nyxborn Hydra") and has(g, "Cast Nyxborn Hydra (bestow)")
    choose(g, "Cast Nyxborn Hydra (bestow)")
    assert labels(g) == ["X=0", "X=1", "X=2"]
    choose(g, "X=2")
    pay(g)
    resolve_stack(g)
    hydra = find(g, "Nyxborn Hydra")
    gix = find(g, "Gixian Infiltrator")
    assert hydra.attached_to == gix.oid and hydra.counters == 2 and not g.is_creature(hydra)
    assert (g.power(gix), g.toughness(gix)) == (4, 3) and g.has(gix, "trample") and g.has(gix, "reach")
    g.destroy(gix)
    pass_priority(g)  # state-based actions run before the next priority
    hydra = find(g, "Nyxborn Hydra")
    assert hydra.attached_to is None and g.is_creature(hydra) and (g.power(hydra), g.toughness(hydra)) == (2, 3)


def test_nyxborn_hydra_normal_cast_x_range():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 3})
    assert not has(g, "bestow")  # no creature to enchant
    choose(g, "Cast Nyxborn Hydra")
    assert labels(g) == ["X=0", "X=1", "X=2"]


def test_bestow_with_illegal_target_resolves_as_creature():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 3 + ["Gixian Infiltrator"]})
    choose(g, "Cast Nyxborn Hydra (bestow)")
    choose(g, "X=1")
    pay(g)
    g.destroy(find(g, "Gixian Infiltrator"))
    resolve_stack(g)
    hydra = find(g, "Nyxborn Hydra")
    assert g.is_creature(hydra) and hydra.counters == 1


def test_cleansing_wildfire_on_indestructible_bridge():
    g = scenario(p0={"hand": ["Cleansing Wildfire"], "battlefield": ["Drossforge Bridge", "Mountain"], "library": ["Forest", "Swamp", "Ichor Wellspring"]})
    choose(g, "Cast Cleansing Wildfire")
    choose(g, "Target Drossforge Bridge")
    pay(g)
    resolve_stack(g)
    assert sorted(labels(g)) == ["Find Forest", "Find Swamp", "Find nothing"]
    choose(g, "Find Forest")
    assert "Drossforge Bridge" in bf(g)
    forest = find(g, "Forest")
    assert forest.tapped
    assert len(g.players[0].hand) == 1


def test_cleansing_wildfire_opponent_searches():
    g = scenario(p0={"hand": ["Cleansing Wildfire"], "battlefield": ["Mountain", "Mountain"]}, p1={"battlefield": ["Island"]})
    choose(g, "Cast Cleansing Wildfire")
    choose(g, "Target Island")
    pay(g)
    resolve_stack(g)
    assert g.decision.player == 1
    choose(g, "Find Island")
    assert find(g, "Island", 1).tapped and len(g.players[1].graveyard) == 1


def test_eviscerators_insight_flashback():
    g = scenario(p0={"graveyard": ["Eviscerator's Insight"], "battlefield": ["Swamp"] * 5 + ["Ichor Wellspring"]})
    choose(g, "Cast Eviscerator's Insight (flashback)")
    pay(g)
    resolve_stack(g)  # Insight, then the Wellspring trigger
    resolve_stack(g)
    assert len(g.players[0].hand) == 3
    assert names(g.players[0].exile) == ["Eviscerator's Insight"]


def test_toxin_analysis_deathtouch_lifelink_and_clue():
    g = scenario(p0={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Swamp", "Ichor Wellspring"]}, p1={"battlefield": ["Tolarian Terror", "Delver of Secrets"]}, auto_single=False)
    choose(g, "Cast Toxin Analysis")
    choose(g, "Target Krark-Clan Shaman")
    pay(g)
    resolve_stack(g)
    assert "Clue" in bf(g)
    life = g.players[0].life
    choose(g, "Krark-Clan Shaman: 1 damage")
    choose(g, "Sacrifice Ichor Wellspring")
    resolve_stack(g)
    resolve_stack(g)
    # deathtouch kills the 5/5; lifelink gains 1 per creature dealt damage (3)
    assert "Tolarian Terror" not in bf(g) and "Delver of Secrets" not in bf(g)
    assert g.players[0].life == life + 3


def test_damage_ability_uses_sources_current_keywords():
    """Toxin Analysis cast on Krark-Clan Shaman in response to its own ability:
    the source is still on the battlefield, so the ability's damage has the
    deathtouch and lifelink it has now, not the activation-time snapshot's."""
    g = scenario(p0={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Swamp", "Ichor Wellspring"]}, p1={"battlefield": ["Tolarian Terror"]}, auto_single=False)
    choose(g, "Krark-Clan Shaman: 1 damage")  # sacrificing the Wellspring is forced
    choose(g, "Cast Toxin Analysis")
    choose(g, "Target Krark-Clan Shaman")
    pay(g)
    life = g.players[0].life
    resolve_stack(g)
    resolve_stack(g)
    assert "Tolarian Terror" not in bf(g)
    assert g.players[0].life == life + 2  # Shaman and Terror were each dealt 1


def test_makeshift_munitions_can_sacrifice_its_own_target():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Gixian Infiltrator", "Swamp"]})
    choose(g, "Makeshift Munitions: 1 damage")
    choose(g, "Target Gixian Infiltrator")
    assert "Gixian Infiltrator" not in bf(g)  # only artifact/creature: sacrificed as the cost
    resolve_stack(g)  # fizzles
    assert g.players[0].life == 20 and g.players[1].life == 20


def test_makeshift_munitions_to_face():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Clue", "Swamp"]})
    choose(g, "Makeshift Munitions: 1 damage")
    choose(g, "player 1")
    resolve_stack(g)
    assert g.players[1].life == 19


def test_ichor_wellspring_draws_on_etb_and_death():
    g = scenario(p0={"hand": ["Ichor Wellspring"], "battlefield": ["Swamp", "Swamp", "Krark-Clan Shaman"]})
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 1
    choose(g, "Krark-Clan Shaman: 1 damage")
    resolve_stack(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 2


def test_lembas_scry_draw_lifegain_and_shuffle_back():
    g = scenario(p0={"hand": ["Lembas"], "battlefield": ["Swamp"] * 4, "library": ["Cast Down", "Forest", "Swamp"]})
    choose(g, "Cast Lembas")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.decision.kind == O.CHOOSE_MODE
    choose(g, "Put Cast Down on the bottom")
    assert names(g.players[0].hand) == ["Forest"]
    choose(g, "Lembas: gain 3 life")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.players[0].life == 23
    assert "Lembas" in names(g.players[0].library) and not g.players[0].graveyard


def test_nihil_spellbomb_exile_and_pay_to_draw():
    g = scenario(p0={"battlefield": ["Nihil Spellbomb", "Swamp"]}, p1={"graveyard": ["Brainstorm", "Ponder"]})
    choose(g, "Nihil Spellbomb: exile")
    choose(g, "player 1")
    resolve_stack(g)  # dies trigger resolves first
    assert labels(g) == ["Don't pay", "Pay {B}"]
    choose(g, "Pay {B}")
    resolve_stack(g)
    assert len(g.players[0].hand) == 1 and not g.players[1].graveyard and len(g.players[1].exile) == 2


def test_map_explore_land_and_nonland():
    g = scenario(p0={"battlefield": ["Map", "Swamp", "Gixian Infiltrator"], "library": ["Forest", "Cast Down"]})
    choose(g, "Map: explore")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)  # Gixian sacrifice trigger
    assert names(g.players[0].hand) == ["Forest"]
    assert find(g, "Gixian Infiltrator").counters == 1  # from sacrificing the Map, not from exploring
    g = scenario(p0={"battlefield": ["Map", "Swamp", "Krark-Clan Shaman"], "library": ["Cast Down", "Forest"]})
    choose(g, "Map: explore")
    resolve_stack(g)
    choose(g, "Put Cast Down into graveyard")
    assert find(g, "Krark-Clan Shaman").counters == 1 and names(g.players[0].graveyard) == ["Cast Down"]


def test_map_is_sorcery_speed():
    g = scenario(p0={"battlefield": ["Map", "Swamp", "Gixian Infiltrator"]}, step="upkeep")
    assert not has(g, "Map: explore")


def test_twisted_landscape_search_and_cycling():
    g = scenario(p0={"battlefield": ["Twisted Landscape"], "library": ["Island", "Mountain", "Swamp"]})
    choose(g, "Twisted Landscape: search")
    resolve_stack(g)
    assert sorted(labels(g)) == ["Find Mountain", "Find Swamp", "Find nothing"]
    choose(g, "Find Mountain")
    assert find(g, "Mountain").tapped
    g = scenario(p0={"hand": ["Twisted Landscape"], "battlefield": ["Drossforge Bridge", "Slagwoods Bridge", "Forest"]}, step="end")
    choose(g, "Twisted Landscape: cycling")
    pay(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 1 and names(g.players[0].graveyard) == ["Twisted Landscape"]


def test_bridges_enter_tapped():
    g = scenario(p0={"hand": ["Slagwoods Bridge"]})
    choose(g, "Play Slagwoods Bridge")
    assert find(g, "Slagwoods Bridge").tapped
    assert not has(g, "Play")  # one land per turn


# ---------------------------------------------------------------------------
# Audit edge cases
# ---------------------------------------------------------------------------


def test_familiar_affinity_never_touches_black():
    arts = ["Ichor Wellspring", "Nihil Spellbomb", "Lembas", "Vault of Whispers", "Drossforge Bridge"]
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": arts}, p1={"hand": ["Island"]})
    # 5 artifacts: {3}{B} -> {B}, paid with one source
    assert has(g, "Cast Refurbished Familiar")
    choose(g, "Cast Refurbished Familiar")
    pay(g)
    assert sum(1 for c in g.battlefield if c.tapped and c.controller == 0) == 1


def test_affinity_counts_tokens():
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Swamp", "Clue", "Map", "Clue"]}, p1={"hand": ["Island"]})
    assert has(g, "Cast Refurbished Familiar")  # {3}{B} - 3
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Swamp", "Clue", "Map"]}, p1={"hand": ["Island"]})
    assert not has(g, "Cast Refurbished Familiar")


def test_gixian_ignores_its_own_and_opponents_sacrifices():
    g = scenario(p0={"hand": ["Fanatical Offering"], "battlefield": ["Gixian Infiltrator", "Swamp", "Swamp"]})
    choose(g, "Cast Fanatical Offering")
    pay(g)
    resolve_stack(g)  # the only fodder is Gixian itself: no counter trigger for itself
    assert "Gixian Infiltrator" in names(g.players[0].graveyard) and not g.pending
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"battlefield": ["Island"]})
    g.sacrifice(find(g, "Island"))
    assert not g.pending


def test_chrysalis_grows_only_from_eldrazi():
    g = scenario(p0={"battlefield": ["Writhing Chrysalis", "Ichor Wellspring", "Eldrazi Spawn"]})
    ch = find(g, "Writhing Chrysalis")
    g.sacrifice(find(g, "Ichor Wellspring"))
    assert not any("counter" in t.tdef.name for t in g.pending)
    g.pending.clear()
    g.sacrifice(find(g, "Eldrazi Spawn"))
    assert any(t.source.oid == ch.oid and "counter" in t.tdef.name for t in g.pending)


def test_toxin_analysis_wears_off_at_end_of_turn():
    g = scenario(p0={"hand": ["Toxin Analysis"], "battlefield": ["Gixian Infiltrator", "Swamp"]}, step="end")
    choose(g, "Cast Toxin Analysis")
    pay(g)
    resolve_stack(g)
    gix = find(g, "Gixian Infiltrator")
    assert g.has(gix, "deathtouch") and g.has(gix, "lifelink")
    while g.active == 0:
        g.step(0)
    assert not g.has(gix, "deathtouch") and not g.has(gix, "lifelink")


def test_deathtouch_blocker_kills_terror():
    g = scenario(
        p0={"hand": ["Toxin Analysis"], "battlefield": ["Gixian Infiltrator", "Swamp"]},
        p1={"battlefield": [("Tolarian Terror", {"sick": False})]},
        active=1,
        step="declare_attackers",
    )
    while g.decision.kind != O.DECLARE_ATTACKER:
        g.step(0)
    choose(g, "Attack with Tolarian Terror")
    while g.decision.kind != O.DECLARE_BLOCKER:
        g.step(0)
    choose(g, "blocks Tolarian Terror")
    while not (g.decision.kind == O.PRIORITY and g.decision.player == 0):
        g.step(0)
    choose(g, "Cast Toxin Analysis")
    pay(g)
    resolve_stack(g)
    while g.step_name != "combat_damage" or g.stack:
        g.step(0)
    pass_priority(g)
    assert "Tolarian Terror" not in bf(g) and "Gixian Infiltrator" not in bf(g)


def test_krark_clan_shaman_damage_comes_from_shaman():
    # damage is dealt by the Shaman: a deathtouch Shaman kills everything it hits
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Vault of Whispers"]}, p1={"battlefield": ["Cryptic Serpent"]})
    find(g, "Krark-Clan Shaman").temp.append(O.TempEffect(keywords=frozenset({"deathtouch"})))
    choose(g, "Krark-Clan Shaman: 1 damage")
    choose(g, "Sacrifice Vault of Whispers")  # or tap it for {B} first
    resolve_stack(g)
    assert "Cryptic Serpent" not in bf(g)
