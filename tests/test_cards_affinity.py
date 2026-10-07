"""Grixis Affinity cards (rules from their oracle text), its sideboard cards,
deck legality and full games. Runs on both engines (conftest.ENGINE_MODULES)."""

from helpers import bf, choose, find, has, labels, names, new_game, pass_priority, pay, resolve_stack, scenario

from mtg_ml.bots import make_bot
from mtg_ml.engine import objects as O
from mtg_ml.engine.cards import CARDS
from mtg_ml.engine import DECKS, SIDEBOARD_PLANS, SIDEBOARDS, postboard
from mtg_ml.match import MATCHUPS, game_args

SEATS = lambda n: ["Seat of the Synod"] * n  # noqa: E731
FURNACES = lambda n: ["Great Furnace"] * n  # noqa: E731


def tapped(g, player=0) -> int:
    return sum(1 for c in g.battlefield if c.controller == player and c.tapped)


# -- lands --------------------------------------------------------------------


def test_artifact_lands():
    g = scenario(p0={"hand": ["Mistvault Bridge"]})
    choose(g, "Play Mistvault Bridge")
    bridge = find(g, "Mistvault Bridge")
    assert bridge.tapped and g.is_artifact(bridge) and g.is_land(bridge) and g.has(bridge, "indestructible")
    g = scenario(p0={"hand": ["Seat of the Synod"]})
    choose(g, "Play Seat of the Synod")
    seat = find(g, "Seat of the Synod")
    assert not seat.tapped and g.is_artifact(seat) and not g.has(seat, "indestructible")


# -- affinity -------------------------------------------------------------------


def test_myr_enforcer_affinity_for_artifacts():
    # Six artifacts (lands, Wellspring, a Blood token): {7} costs {1}.
    g = scenario(p0={"hand": ["Myr Enforcer"], "battlefield": SEATS(3) + ["Mistvault Bridge", "Ichor Wellspring", "Blood"]})
    choose(g, "Cast Myr Enforcer")
    pay(g)
    assert tapped(g) == 1
    resolve_stack(g)
    e = find(g, "Myr Enforcer")
    assert g.power(e) == 4 and g.toughness(e) == 4 and g.is_artifact(e)


def test_affinity_counts_only_artifacts_you_control():
    # Two artifacts of ours: {5}, with three lands. The opponent's four do not count.
    g = scenario(p0={"hand": ["Myr Enforcer"], "battlefield": SEATS(2) + ["Swamp"]}, p1={"battlefield": SEATS(4)})
    assert not has(g, "Cast Myr Enforcer")
    g = scenario(p0={"hand": ["Myr Enforcer"], "battlefield": SEATS(3) + ["Swamp"]})
    assert has(g, "Cast Myr Enforcer")  # {4} with four lands


def test_thoughtcast_and_utrom_monitor():
    g = scenario(p0={"hand": ["Thoughtcast", "Utrom Monitor"], "battlefield": SEATS(4)})
    choose(g, "Cast Thoughtcast")  # {4}{U} - 4 = {U}
    pay(g)
    resolve_stack(g)
    assert len(g.players[0].hand) == 3 and tapped(g) == 1
    choose(g, "Cast Utrom Monitor")
    pay(g)
    resolve_stack(g)
    m = find(g, "Utrom Monitor")
    assert g.has(m, "flying") and g.power(m) == 3 and tapped(g) == 2


# -- Galvanic Blast -------------------------------------------------------------


def test_galvanic_blast_two_damage_without_metalcraft():
    g = scenario(p0={"hand": ["Galvanic Blast"], "battlefield": ["Great Furnace", "Seat of the Synod"]})
    choose(g, "Cast Galvanic Blast")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 18


def test_galvanic_blast_metalcraft_four_damage():
    g = scenario(p0={"hand": ["Galvanic Blast"], "battlefield": ["Great Furnace", "Seat of the Synod", "Nihil Spellbomb"]})
    choose(g, "Cast Galvanic Blast")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 16


def test_galvanic_blast_metalcraft_is_checked_on_resolution():
    # In response, the Spellbomb is sacrificed: only two artifacts are left.
    g = scenario(p0={"hand": ["Galvanic Blast"], "battlefield": ["Great Furnace", "Seat of the Synod", "Nihil Spellbomb"]})
    choose(g, "Cast Galvanic Blast")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    choose(g, "Nihil Spellbomb: exile target player's graveyard")
    choose(g, "Target player 1 (opponent)")
    resolve_stack(g)  # Spellbomb's ability; its draw trigger asks for {B} (no B source: only "Don't pay")
    resolve_stack(g)
    assert g.players[1].life == 18


# -- Reckoner's Bargain -----------------------------------------------------------


def test_reckoners_bargain_gains_life_equal_to_sacrificed_mana_value():
    g = scenario(p0={"hand": ["Reckoner's Bargain"], "battlefield": ["Vault of Whispers", "Swamp", "Myr Enforcer", "Blood"]})
    choose(g, "Cast Reckoner's Bargain")
    pay(g)
    assert g.decision.kind == O.SACRIFICE
    choose(g, "Sacrifice Myr Enforcer")
    resolve_stack(g)
    assert g.players[0].life == 27 and len(g.players[0].hand) == 2


def test_reckoners_bargain_token_gains_nothing():
    g = scenario(p0={"hand": ["Reckoner's Bargain"], "battlefield": ["Swamp", "Swamp", "Blood"]})
    choose(g, "Cast Reckoner's Bargain")
    pay(g)  # the Blood token is the only artifact or creature: sacrificed by settle()
    resolve_stack(g)
    assert g.players[0].life == 20 and len(g.players[0].hand) == 2 and "Blood" not in bf(g)


# -- Blood Fountain ---------------------------------------------------------------


def test_blood_fountain_creates_blood_and_returns_up_to_two_creatures():
    g = scenario(
        p0={"hand": ["Blood Fountain"], "battlefield": ["Swamp"] * 5, "graveyard": ["Myr Enforcer", "Thoughtcast", "Kenku Artificer", "Myr Enforcer"]}
    )
    choose(g, "Cast Blood Fountain")
    pay(g)
    resolve_stack(g)  # the artifact
    resolve_stack(g)  # its ETB trigger
    assert sorted(bf(g, 0)) == ["Blood", "Blood Fountain"] + ["Swamp"] * 5
    # The Fountain entered this turn, but it is no creature: {T} is fine.
    choose(g, "Blood Fountain: return up to two creature cards")
    pay(g)
    assert "Blood Fountain" not in bf(g)
    resolve_stack(g)
    assert labels(g) == ["Stop", "Return Myr Enforcer", "Return Kenku Artificer"]
    choose(g, "Return Kenku Artificer")
    assert labels(g) == ["Stop", "Return Myr Enforcer"]
    choose(g, "Return Myr Enforcer")
    assert sorted(names(g.players[0].hand)) == ["Kenku Artificer", "Myr Enforcer"]
    assert sorted(names(g.players[0].graveyard)) == ["Blood Fountain", "Myr Enforcer", "Thoughtcast"]


def test_blood_fountain_may_stop_after_one():
    g = scenario(p0={"battlefield": ["Blood Fountain"] + ["Swamp"] * 4, "graveyard": ["Myr Enforcer", "Utrom Monitor"]})
    choose(g, "Blood Fountain: return up to two creature cards")
    pay(g)
    resolve_stack(g)
    choose(g, "Return Utrom Monitor")
    choose(g, "Stop")
    assert names(g.players[0].hand) == ["Utrom Monitor"] and not g.stack


# -- Sewer-veillance Cam ------------------------------------------------------------


def test_cam_has_flash_and_taps_target_creature_on_entering():
    g = scenario(p0={"hand": ["Sewer-veillance Cam"], "battlefield": ["Island"]}, p1={"battlefield": ["Myr Enforcer"]}, active=1)
    pass_priority(g)  # p1 passes in its main phase; p0 gets priority
    choose(g, "Cast Sewer-veillance Cam")
    pay(g)
    resolve_stack(g)
    assert g.stack and g.stack[-1].name == "Sewer-veillance Cam: tap or untap target creature"
    assert g.stack[-1].targets == [("perm", find(g, "Myr Enforcer").oid)]  # the only creature: chosen by settle()
    resolve_stack(g)
    assert g.decision.kind == O.CHOOSE_MODE
    assert [lab.split("#")[0] for lab in labels(g)] == ["Leave it", "Tap Myr Enforcer", "Untap Myr Enforcer"]
    choose(g, "Tap Myr Enforcer")
    assert find(g, "Myr Enforcer").tapped


def test_cam_without_creatures_its_trigger_is_removed():
    g = scenario(p0={"hand": ["Sewer-veillance Cam"], "battlefield": ["Island"]})
    choose(g, "Cast Sewer-veillance Cam")
    pay(g)
    resolve_stack(g)
    assert not g.stack and "Sewer-veillance Cam" in bf(g)


def test_cam_leaves_the_battlefield_trigger_untaps_a_creature():
    g = scenario(p0={"battlefield": ["Sewer-veillance Cam", ("Myr Enforcer", {"tapped": True})] + ["Island"] * 4})
    choose(g, "Sewer-veillance Cam: draw two cards")
    pay(g)
    assert "Sewer-veillance Cam" not in bf(g)
    assert g.stack[-1].name == "Sewer-veillance Cam: tap or untap target creature"  # above the draw
    resolve_stack(g)
    choose(g, "Untap Myr Enforcer")
    assert not find(g, "Myr Enforcer").tapped
    resolve_stack(g)
    assert len(g.players[0].hand) == 2


def test_cam_trigger_targeting_a_ward_creature_triggers_ward():
    g = scenario(p0={"hand": ["Sewer-veillance Cam"], "battlefield": ["Island"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Sewer-veillance Cam")
    pay(g)
    pass_priority(g, 2)  # the Cam resolves; its trigger targets the Terror (the only creature)
    assert [it.name for it in g.stack] == ["Sewer-veillance Cam: tap or untap target creature", "Tolarian Terror: ward"]
    resolve_stack(g)  # p0 has no mana left: only "Don't pay"; the Cam trigger is countered
    assert not g.stack and not find(g, "Tolarian Terror").tapped


# -- Kenku Artificer ----------------------------------------------------------------


def test_kenku_artificer_animates_a_bridge():
    g = scenario(p0={"hand": ["Kenku Artificer"], "battlefield": ["Mistvault Bridge", "Island", "Island", "Ichor Wellspring"]})
    choose(g, "Cast Kenku Artificer")
    pay(g)
    resolve_stack(g)
    assert g.decision.kind == O.TARGET
    assert [lab.split("#")[0] for lab in labels(g)] == ["No target", "Target Mistvault Bridge", "Target Ichor Wellspring"]
    choose(g, "Target Mistvault Bridge")
    resolve_stack(g)
    b = find(g, "Mistvault Bridge")
    assert g.is_creature(b) and g.is_land(b) and g.is_artifact(b)
    assert (g.power(b), g.toughness(b)) == (3, 3) and g.has(b, "flying") and g.has(b, "indestructible")
    assert "Creature" in g.types(b)


def test_kenku_artificer_no_target_and_opponents_artifacts():
    g = scenario(p0={"hand": ["Kenku Artificer"], "battlefield": ["Island"] * 3}, p1={"battlefield": ["Nihil Spellbomb"]})
    choose(g, "Cast Kenku Artificer")
    pay(g)
    resolve_stack(g)
    assert [lab.split("#")[0] for lab in labels(g)] == ["No target", "Target Nihil Spellbomb"]  # any noncreature artifact
    choose(g, "No target")
    resolve_stack(g)
    assert not g.is_creature(find(g, "Nihil Spellbomb"))


def test_kenku_artificer_without_artifacts_trigger_has_no_target():
    g = scenario(p0={"hand": ["Kenku Artificer"], "battlefield": ["Island"] * 3})
    choose(g, "Cast Kenku Artificer")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)  # "No target" was the only option
    assert not g.stack and "Kenku Artificer" in bf(g)


def test_animated_artifact_keeps_counters_until_it_leaves_and_dies_as_a_creature():
    g = scenario(p0={"hand": ["Kenku Artificer"], "battlefield": ["Ichor Wellspring", "Island", "Island", "Island"]}, p1={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]})
    choose(g, "Cast Kenku Artificer")
    pay(g)
    resolve_stack(g)
    choose(g, "Target Ichor Wellspring")
    resolve_stack(g)
    w = find(g, "Ichor Wellspring")
    assert g.power(w) == 3 and w.counters == 3
    pass_priority(g)  # p1 may act
    choose(g, "Cast Lightning Bolt")
    choose(g, "Ichor Wellspring")
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" not in bf(g)  # 3 damage to a 3/3
    resolve_stack(g)  # Wellspring's dies trigger draws
    assert names(g.players[0].graveyard) == ["Ichor Wellspring"]


# -- sideboard -----------------------------------------------------------------------


def test_extract_a_confession_opponent_chooses():
    g = scenario(p0={"hand": ["Extract a Confession"], "battlefield": ["Swamp", "Swamp"]}, p1={"battlefield": ["Myr Enforcer", "Krark-Clan Shaman"]})
    assert not has(g, "(evidence)")  # empty graveyard
    choose(g, "Cast Extract a Confession")
    pay(g)
    resolve_stack(g)
    assert g.decision.player == 1 and g.decision.kind == O.SACRIFICE
    assert [lab.split("#")[0] for lab in labels(g)] == ["Sacrifice Myr Enforcer", "Sacrifice Krark-Clan Shaman"]
    choose(g, "Sacrifice Krark-Clan Shaman")
    assert bf(g, 1) == ["Myr Enforcer"]


def test_extract_a_confession_with_evidence_takes_the_greatest_power():
    g = scenario(
        p0={"hand": ["Extract a Confession"], "battlefield": ["Swamp", "Swamp"], "graveyard": ["Thoughtcast", "Galvanic Blast", "Swamp"]},
        p1={"battlefield": ["Myr Enforcer", "Krark-Clan Shaman"]},
    )
    choose(g, "Cast Extract a Confession (evidence)")
    pay(g)
    assert g.decision.kind == O.EXILE_FROM_GY
    assert labels(g) == ["Exile Thoughtcast", "Exile Galvanic Blast", "Exile Swamp"]
    choose(g, "Exile Swamp")  # mana value 0: still 0/6
    choose(g, "Exile Thoughtcast")  # 5/6: the Blast, the last card, is exiled by settle()
    assert names(g.players[0].exile) == ["Swamp", "Thoughtcast", "Galvanic Blast"]
    resolve_stack(g)  # only the 4/4 is offered, so settle() sacrifices it
    assert bf(g, 1) == ["Krark-Clan Shaman"]


def test_extract_evidence_needs_six_mana_value():
    g = scenario(p0={"hand": ["Extract a Confession"], "battlefield": ["Swamp", "Swamp"], "graveyard": ["Thoughtcast"]}, p1={"battlefield": ["Krark-Clan Shaman"]})
    assert has(g, "Cast Extract a Confession") and not has(g, "(evidence)")


def test_unexpected_fangs_counters_last_beyond_the_turn():
    g = scenario(p0={"hand": ["Unexpected Fangs"], "battlefield": ["Swamp", "Swamp", "Krark-Clan Shaman"]}, step="end")
    choose(g, "Cast Unexpected Fangs")
    pay(g)
    resolve_stack(g)
    s = find(g, "Krark-Clan Shaman")
    assert g.power(s) == 2 and g.has(s, "lifelink")
    pass_priority(g, 2)  # cleanup: until-end-of-turn effects end, counters stay
    s = find(g, "Krark-Clan Shaman")
    assert g.turn == 2 and g.power(s) == 2 and g.has(s, "lifelink") and not s.temp


def test_lifelink_counter_gains_life_on_damage():
    g = scenario(p0={"hand": ["Unexpected Fangs"], "battlefield": ["Swamp", "Swamp", ("Myr Enforcer", {"sick": False})]}, step="main1")
    choose(g, "Cast Unexpected Fangs")
    pay(g)
    resolve_stack(g)
    pass_priority(g, 2)  # to combat
    while g.decision.kind != O.DECLARE_ATTACKER:
        pass_priority(g)
    choose(g, "Attack with Myr Enforcer")  # then "Done" is the only option
    while g.players[1].life == 20:
        pass_priority(g)
    assert g.players[1].life == 15 and g.players[0].life == 25


# -- Black Mage's Rod (Equipment) -----------------------------------------------


def _rod_on_hero(extra_hand=(), extra_bf=(), p1=None):
    g = scenario(p0={"hand": ["Black Mage's Rod", *extra_hand], "battlefield": ["Swamp"] * 2 + list(extra_bf)}, p1=p1 or {})
    choose(g, "Cast Black Mage's Rod")
    pay(g)
    resolve_stack(g)  # the Rod, then its job select trigger
    return g


def test_black_mages_rod_job_select_makes_an_equipped_hero():
    g = _rod_on_hero()
    hero, rod = find(g, "Hero"), find(g, "Black Mage's Rod")
    assert hero.is_token and rod.attached_to == hero.oid
    assert (g.power(hero), g.toughness(hero)) == (2, 1)  # 1/1 Hero, +1/+0
    assert g.is_artifact(rod) and not g.is_creature(rod) and g.is_creature(hero)
    assert g.players[1].life == 20  # casting the Rod itself does not trigger it


def test_black_mages_rod_equipped_creature_pings_on_noncreature_spells():
    g = _rod_on_hero(extra_hand=["Ichor Wellspring", "Gixian Infiltrator"], extra_bf=["Swamp"] * 4)
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    assert [it.name for it in g.stack][-1].startswith("Black Mage's Rod")  # the trigger, above the spell
    resolve_stack(g)
    assert g.players[1].life == 19
    choose(g, "Cast Gixian Infiltrator")  # a creature spell: no trigger
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 19


def test_black_mages_rod_stays_when_the_hero_dies_and_can_be_equipped_again():
    g = _rod_on_hero(extra_hand=["Ichor Wellspring"], extra_bf=["Krark-Clan Shaman"] + ["Swamp"] * 5, p1={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]})
    pass_priority(g)
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target Hero")
    resolve_stack(g)
    rod = find(g, "Black Mage's Rod")
    assert "Hero" not in bf(g) and "Black Mage's Rod" in bf(g) and rod.attached_to is None
    choose(g, "Cast Ichor Wellspring")  # unattached: nothing triggers
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 20
    equip = [lab for lab in labels(g) if lab.startswith("Black Mage's Rod")]
    assert equip == ["Black Mage's Rod: equip"]
    choose(g, "Black Mage's Rod: equip")  # the only target and payment are taken by settle()
    pay(g)
    resolve_stack(g)
    shaman = find(g, "Krark-Clan Shaman")
    assert find(g, "Black Mage's Rod").attached_to == shaman.oid and g.power(shaman) == 2  # 1/1, +1/+0


def test_black_mages_rod_equip_is_sorcery_speed():
    g = _rod_on_hero(extra_bf=["Swamp"] * 3, p1={"hand": ["Lightning Bolt"], "battlefield": ["Mountain"]})
    assert any(lab.startswith("Black Mage's Rod: equip") for lab in labels(g))
    pass_priority(g)
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target player 0")
    assert not any(lab.startswith("Black Mage's Rod") for lab in labels(g))  # p0 responds: not at instant speed


def test_kenku_animated_rod_falls_off_and_stops_triggering():
    g = _rod_on_hero(extra_hand=["Kenku Artificer", "Ichor Wellspring"], extra_bf=["Island"] * 3 + ["Swamp"] * 2)
    choose(g, "Cast Kenku Artificer")
    pay(g)
    resolve_stack(g)
    choose(g, next(lab for lab in labels(g) if lab.startswith("Target Black Mage's Rod")))
    resolve_stack(g)
    rod, hero = find(g, "Black Mage's Rod"), find(g, "Hero")
    assert g.is_creature(rod) and rod.attached_to is None and g.power(hero) == 1  # an Equipment that is a creature cannot equip (CR 301.5c)
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 20


# -- the deck ----------------------------------------------------------------------


def test_deck_legality():
    main, side = DECKS["grixis_affinity"], SIDEBOARDS["grixis_affinity"]
    assert sum(main.values()) == 60 and sum(side.values()) == 15
    for name in set(main) | set(side):
        assert name in CARDS, name
        assert "Basic" in CARDS[name].supertypes or main.get(name, 0) + side.get(name, 0) <= 4, name
    for (deck, opp), plan in SIDEBOARD_PLANS.items():
        if "grixis_affinity" in (deck, opp):
            assert sum(postboard(deck, opp).values()) == 60
    assert {MATCHUPS[m][1] for m in ("jund_affinity", "blue_affinity", "madness_affinity")} == {"grixis_affinity"}


def test_full_bot_games_on_every_affinity_matchup():
    for matchup in ("jund_affinity", "blue_affinity", "madness_affinity"):
        decks = MATCHUPS[matchup]
        for game_no in (1, 2):
            g = new_game(seed=game_no, max_turns=60, **game_args(game_no, matchup))
            bots = [make_bot(0, decks[0]), make_bot(1, decks[1])]
            while not g.over:
                g.step(bots[g.decision.player].act(g))
            assert g.end_reason in ("life", "decking", "turn limit")
