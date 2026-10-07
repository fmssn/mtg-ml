"""Elves cards, its sideboard and the tokens and rules they need (Food,
Treasure, Skeleton, initiative and the Undercity, menace, hexproof,
changeling, omen, collect evidence). Every test runs on both engines."""

import random

from helpers import bf, choose, find, has, labels, names, new_game, pass_priority, pay, resolve_stack, scenario, settle

from mtg_ml.agents import RandomAgent
from mtg_ml.bots import make_bot
from mtg_ml.engine import objects as O
from mtg_ml.engine.view import determinize
from mtg_ml.match import MATCHUPS, game_args
from mtg_ml.engine.decks import DECKS, SIDEBOARD_PLANS, SIDEBOARDS, postboard

FORESTS = lambda n: ["Forest"] * n  # noqa: E731
SWAMPS = lambda n: ["Swamp"] * n  # noqa: E731


def until_own_main(g, turn: int) -> None:
    """Step through (option 0: pass, no attack, no block) to `turn`'s first main phase."""
    while not (g.turn == turn and g.step_name == "main1" and g.decision.kind == O.PRIORITY):
        g.step(0)


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def test_elves_deck_is_legal_and_boards_to_sixty():
    assert sum(DECKS["elves"].values()) == 60 and sum(SIDEBOARDS["elves"].values()) <= 15
    assert all(n <= 4 for name, n in DECKS["elves"].items() if name != "Forest")
    for other in ("jund_wildfire", "mono_blue_terror", "red_madness"):
        assert sum(postboard("elves", other).values()) == 60 and sum(postboard(other, "elves").values()) == 60
        assert ("elves", other) in SIDEBOARD_PLANS and (other, "elves") in SIDEBOARD_PLANS


# ---------------------------------------------------------------------------
# Mana creatures
# ---------------------------------------------------------------------------


def test_summoning_sick_mana_elf_cannot_tap():
    g = scenario(p0={"hand": ["Llanowar Elves"], "battlefield": [("Fyndhorn Elves", {"sick": True})]})
    assert not has(g, "Cast Llanowar Elves")


def test_mana_elf_pays_for_a_spell():
    g = scenario(p0={"hand": ["Llanowar Elves"], "battlefield": ["Elvish Mystic"]})
    choose(g, "Cast Llanowar Elves")
    resolve_stack(g)
    assert find(g, "Elvish Mystic").tapped and find(g, "Llanowar Elves").sick


def test_priest_of_titania_makes_one_green_per_elf_and_the_rest_floats():
    # Elves on the battlefield: Priest, two Llanowar, the opponent's Masked Vandal (changeling) = 4.
    g = scenario(p0={"hand": ["Timberwatch Elf", "Llanowar Elves"], "battlefield": ["Priest of Titania", "Llanowar Elves", "Llanowar Elves"]}, p1={"battlefield": ["Masked Vandal"]})
    choose(g, "Cast Timberwatch Elf")
    choose(g, "Tap Priest of Titania")
    assert g.players[0].pool == {"G": 3}
    pay(g)  # the floating mana pays the rest
    assert g.players[0].pool == {"G": 1}
    assert [c.tapped for c in g.battlefield if c.name == "Llanowar Elves"] == [False, False]
    resolve_stack(g)
    choose(g, "Cast Llanowar Elves")
    choose(g, "Pay with floating G")  # the last floating {G}
    assert g.players[0].pool == {}


def test_priest_counts_for_feasibility():
    # {5}{G}: Priest makes 3 (three Elves), two Forests make the rest... one short.
    g = scenario(p0={"hand": ["Generous Ent"], "battlefield": ["Priest of Titania", "Llanowar Elves", ("Llanowar Elves", {"tapped": True}), "Forest", "Forest"]})
    assert has(g, "Cast Generous Ent")  # 3 + 1 (untapped Llanowar) + 2 = 6
    g = scenario(p0={"hand": ["Generous Ent"], "battlefield": ["Priest of Titania", ("Llanowar Elves", {"tapped": True}), "Forest", "Forest"]})
    assert not has(g, "Cast Generous Ent")  # 2 + 2 = 4


# ---------------------------------------------------------------------------
# Quirion Ranger, Timberwatch Elf
# ---------------------------------------------------------------------------


def test_quirion_ranger_returns_a_forest_to_untap_once_per_turn():
    g = scenario(p0={"battlefield": ["Quirion Ranger", ("Priest of Titania", {"tapped": True}), ("Forest", {"tapped": True}), "Forest"]})
    choose(g, "Quirion Ranger: untap target creature")
    choose(g, "Target Priest of Titania")
    assert g.decision.kind == O.SACRIFICE  # the cost: which Forest goes back
    assert sorted(labels(g))[0].startswith("Return Forest#")
    tapped_forest = next(c for c in g.battlefield if c.name == "Forest" and c.tapped)
    choose(g, f"Return Forest#{tapped_forest.oid}")
    assert names(g.players[0].hand) == ["Forest"]
    resolve_stack(g)
    assert not find(g, "Priest of Titania").tapped
    assert not has(g, "Quirion Ranger: untap target creature")  # once each turn
    choose(g, "Play Forest")  # the land drop is still there


def test_quirion_ranger_needs_a_forest():
    g = scenario(p0={"battlefield": ["Quirion Ranger", "Swamp"]})
    assert not has(g, "Quirion Ranger")


def test_timberwatch_elf_counts_all_elves_including_changelings():
    g = scenario(p0={"battlefield": ["Timberwatch Elf", "Llanowar Elves"]}, p1={"battlefield": ["Masked Vandal", "Writhing Chrysalis"]})
    choose(g, "Timberwatch Elf: target creature gets +X/+X")
    choose(g, "Target Llanowar Elves")
    resolve_stack(g)
    elf = find(g, "Llanowar Elves")
    assert (g.power(elf), g.toughness(elf)) == (4, 4)  # Timberwatch, Llanowar, Vandal
    assert find(g, "Timberwatch Elf").tapped


def test_timberwatch_elf_is_summoning_sick():
    g = scenario(p0={"battlefield": [("Timberwatch Elf", {"sick": True})]})
    assert not has(g, "Timberwatch Elf")


# ---------------------------------------------------------------------------
# Card flow
# ---------------------------------------------------------------------------


def test_winding_way_takes_the_chosen_type_and_mills_the_rest():
    g = scenario(p0={"hand": ["Winding Way"], "battlefield": FORESTS(2), "library": ["Llanowar Elves", "Forest", "Priest of Titania", "Lead the Stampede", "Swamp"]})
    choose(g, "Cast Winding Way")
    pay(g)
    resolve_stack(g)
    assert labels(g) == ["Choose creature", "Choose land"]
    choose(g, "Choose creature")
    assert names(g.players[0].hand) == ["Llanowar Elves", "Priest of Titania"]
    assert names(g.players[0].graveyard) == ["Forest", "Lead the Stampede", "Winding Way"]
    assert names(g.players[0].library) == ["Swamp"]
    assert all(c.known_to == {0, 1} for c in g.players[0].hand)


def test_lead_the_stampede_takes_every_creature_rest_to_the_bottom():
    g = scenario(p0={"hand": ["Lead the Stampede"], "battlefield": FORESTS(3), "library": ["Forest", "Llanowar Elves", "Winding Way", "Avenging Hunter", "Forest", "Swamp"]})
    choose(g, "Cast Lead the Stampede")
    pay(g)
    resolve_stack(g)
    assert names(g.players[0].hand) == ["Llanowar Elves", "Avenging Hunter"]
    assert names(g.players[0].library) == ["Swamp", "Forest", "Winding Way", "Forest"]
    assert 1 not in g.players[0].library[1].known_to and 0 in g.players[0].library[1].known_to


def test_sagu_wildling_omen_finds_a_land_and_shuffles_itself_away():
    g = scenario(p0={"hand": ["Sagu Wildling"], "battlefield": FORESTS(1), "library": ["Forest"] + SWAMPS(5)})
    assert labels(g) == ["Pass priority", "Cast Sagu Wildling (omen)"]
    choose(g, "Cast Sagu Wildling (omen)")
    assert g.stack[-1].name == "Roost Seek"
    resolve_stack(g)
    choose(g, "Find Forest")
    assert names(g.players[0].hand) == ["Forest"]
    assert sorted(names(g.players[0].library)) == ["Sagu Wildling"] + SWAMPS(5)
    assert g.players[0].graveyard == []


def test_sagu_wildling_creature_gains_three():
    g = scenario(p0={"hand": ["Sagu Wildling"], "battlefield": FORESTS(5)})
    choose(g, "Cast Sagu Wildling")
    pay(g)
    resolve_stack(g)
    resolve_stack(g)
    assert g.players[0].life == 23 and g.has(find(g, "Sagu Wildling"), "flying")


def test_countered_omen_goes_to_the_graveyard():
    g = scenario(p0={"hand": ["Sagu Wildling"], "battlefield": FORESTS(1)}, p1={"hand": ["Counterspell"], "battlefield": ["Island", "Island"]})
    choose(g, "Cast Sagu Wildling (omen)")
    pass_priority(g)
    choose(g, "Cast Counterspell")
    pay(g)
    resolve_stack(g)
    assert names(g.players[0].graveyard) == ["Sagu Wildling"]


# ---------------------------------------------------------------------------
# Masked Vandal, Gingerbread Cabin
# ---------------------------------------------------------------------------


def test_masked_vandal_exiles_a_creature_card_to_exile_an_artifact():
    g = scenario(p0={"hand": ["Masked Vandal"], "battlefield": FORESTS(2), "graveyard": ["Llanowar Elves", "Winding Way"]}, p1={"battlefield": ["Ichor Wellspring"]})
    choose(g, "Cast Masked Vandal")
    pay(g)
    resolve_stack(g)  # Vandal resolves; its trigger targets the only artifact
    settle(g)
    assert g.stack[-1].name == "Masked Vandal: exile target artifact or enchantment"
    resolve_stack(g)
    assert labels(g) == ["Exile nothing", "Exile Llanowar Elves"]
    choose(g, "Exile Llanowar Elves")
    assert names(g.players[1].exile) == ["Ichor Wellspring"] and names(g.players[0].exile) == ["Llanowar Elves"]
    assert not g.stack  # exiled, not put into a graveyard: no Wellspring draw


def test_masked_vandal_may_decline_and_needs_a_target():
    g = scenario(p0={"hand": ["Masked Vandal"], "battlefield": FORESTS(2), "graveyard": ["Llanowar Elves"]}, p1={"battlefield": ["Ichor Wellspring"]})
    choose(g, "Cast Masked Vandal")
    pay(g)
    resolve_stack(g)
    settle(g)
    resolve_stack(g)
    choose(g, "Exile nothing")
    assert "Ichor Wellspring" in bf(g, 1) and names(g.players[0].graveyard) == ["Llanowar Elves"]
    # No artifact or enchantment an opponent controls: the trigger is removed.
    g = scenario(p0={"hand": ["Masked Vandal"], "battlefield": FORESTS(2) + ["Food"], "graveyard": ["Llanowar Elves"]})
    choose(g, "Cast Masked Vandal")
    pay(g)
    resolve_stack(g)
    assert not g.stack and g.decision.kind == O.PRIORITY


def test_gingerbread_cabin_enters_tapped_without_three_other_forests():
    g = scenario(p0={"hand": ["Gingerbread Cabin"], "battlefield": FORESTS(2)})
    choose(g, "Play Gingerbread Cabin")
    assert find(g, "Gingerbread Cabin").tapped and not g.stack and "Food" not in bf(g)


def test_gingerbread_cabin_enters_untapped_with_food():
    g = scenario(p0={"hand": ["Gingerbread Cabin"], "battlefield": FORESTS(3)})
    choose(g, "Play Gingerbread Cabin")
    assert not find(g, "Gingerbread Cabin").tapped
    resolve_stack(g)
    assert "Food" in bf(g, 0)


# ---------------------------------------------------------------------------
# Avenging Hunter: initiative and the Undercity
# ---------------------------------------------------------------------------


def test_avenging_hunter_takes_the_initiative_and_enters_the_undercity():
    g = scenario(p0={"hand": ["Avenging Hunter"], "battlefield": FORESTS(5), "library": ["Forest"] + SWAMPS(5)})
    choose(g, "Cast Avenging Hunter")
    pay(g)
    resolve_stack(g)  # Hunter resolves, its ETB trigger goes on the stack
    resolve_stack(g)  # take the initiative: venture into Secret Entrance
    assert g.initiative == 0 and g.players[0].dungeon_room == "Secret Entrance"
    assert g.stack[-1].name == "Undercity: Secret Entrance"
    resolve_stack(g)
    choose(g, "Find Forest")
    assert names(g.players[0].hand) == ["Forest"] and g.players[0].hand[0].known_to == {0, 1}
    assert g.has(find(g, "Avenging Hunter"), "trample")


def _undercity(room: str | None, p0: dict | None = None, p1: dict | None = None):
    """p0 has the initiative with the venture marker in `room`; the game is
    at p1's end step, so passing reaches p0's upkeep venture."""
    g = scenario(p0=p0, p1=p1, active=1, step="end")
    g.initiative = 0
    g.players[0].dungeon_room = room
    pass_priority(g, 2)  # p1's end step; then p0's untap and upkeep
    assert g.turn == 2 and g.step_name == "upkeep"
    assert g.stack[-1].name == "Undercity: venture into Undercity"
    resolve_stack(g)
    return g


def test_upkeep_venture_chooses_the_branch_and_forge_adds_counters():
    g = _undercity("Secret Entrance", p0={"battlefield": ["Llanowar Elves"]})
    assert labels(g) == ["Venture into Forge", "Venture into Lost Well"]
    choose(g, "Venture into Forge")
    assert g.players[0].dungeon_room == "Forge"
    resolve_stack(g)
    assert find(g, "Llanowar Elves").counters == 2


def test_forge_without_a_creature_is_removed():
    g = _undercity("Secret Entrance")
    choose(g, "Venture into Forge")
    assert not g.stack and g.players[0].dungeon_room == "Forge"


def test_trap_makes_target_player_lose_five():
    g = _undercity("Forge")
    choose(g, "Target player 1 (opponent)")
    resolve_stack(g)
    assert g.players[1].life == 15 and g.players[0].dungeon_room == "Trap!"


def test_lost_well_scry_two():
    g = _undercity("Secret Entrance", p0={"library": ["Forest", "Llanowar Elves"] + SWAMPS(5)})
    choose(g, "Venture into Lost Well")
    resolve_stack(g)
    assert g.decision.kind == O.ORDER and len(labels(g)) == 6
    choose(g, "Top: Llanowar Elves; bottom: Forest")
    lib = names(g.players[0].library)
    assert lib[0] == "Llanowar Elves" and lib[-1] == "Forest"


def test_stash_and_catacombs_tokens():
    g = _undercity("Lost Well")
    resolve_stack(g)
    assert bf(g, 0) == ["Treasure"]


def test_throne_of_the_dead_three_puts_a_creature_with_counters_and_hexproof():
    g = _undercity("Archives", p0={"library": ["Forest", "Generous Ent", "Llanowar Elves"] + SWAMPS(10)})
    resolve_stack(g)
    assert labels(g) == ["Put Generous Ent onto the battlefield", "Put Llanowar Elves onto the battlefield"]
    choose(g, "Put Generous Ent onto the battlefield")
    ent = find(g, "Generous Ent")
    assert ent.counters == 3 and g.has(ent, "hexproof") and g.players[0].dungeon_room == "Throne of the Dead Three"
    assert 1 not in next(c for c in g.players[0].library if c.name == "Forest").known_to  # shuffled
    # The opponent cannot target it; its controller can.
    assert ("perm", ent.oid) not in g.target_candidates(O.TargetSpec("creature"), 1)
    assert ("perm", ent.oid) in g.target_candidates(O.TargetSpec("creature"), 0)


def test_venture_after_the_last_room_starts_a_new_undercity():
    g = _undercity("Throne of the Dead Three", p0={"library": ["Forest"] + SWAMPS(5)})
    assert g.players[0].dungeon_room == "Secret Entrance"


def test_combat_damage_to_the_initiative_holder_takes_it():
    g = scenario(p1={"battlefield": ["Llanowar Elves"]}, active=1, step="declare_attackers")
    g.initiative = 0
    choose(g, "Attack with Llanowar Elves")
    while g.initiative != 1:
        g.step(0)
    assert g.players[0].life == 19 and g.players[1].dungeon_room == "Secret Entrance"


# ---------------------------------------------------------------------------
# Tokens: Treasure, Skeleton (menace)
# ---------------------------------------------------------------------------


def test_treasure_pays_any_colour_but_is_not_floated_at_priority():
    g = scenario(p0={"hand": ["Lightning Bolt"], "battlefield": ["Treasure"]})
    assert labels(g) == ["Pass priority", "Cast Lightning Bolt"]
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target player 1 (opponent)")
    pay(g)
    resolve_stack(g)
    assert g.players[1].life == 17 and "Treasure" not in bf(g)


def test_menace_cannot_be_blocked_by_one_creature():
    g = scenario(p0={"battlefield": ["Skeleton"]}, p1={"battlefield": ["Llanowar Elves"]}, step="declare_attackers")
    choose(g, "Attack with Skeleton")
    pass_priority(g, 2)  # "does not block" is the only option and is taken by settle()
    assert g.step_name == "declare_blockers" and g.blocks == {}
    assert any(line.endswith("does not block") for line in g.log) and not has(g, "blocks Skeleton")


def test_menace_lone_block_is_undone():
    g = scenario(p0={"battlefield": ["Skeleton"]}, p1={"battlefield": ["Llanowar Elves", "Elvish Mystic"]}, step="declare_attackers")
    choose(g, "Attack with Skeleton")
    pass_priority(g, 2)
    choose(g, "blocks Skeleton")
    assert has(g, "blocks Skeleton")  # a second blocker is still possible
    choose(g, "does not block")
    assert g.blocks == {}
    while g.step_name != "main2":
        g.step(0)
    assert g.players[1].life == 16


def test_menace_double_block():
    g = scenario(p0={"battlefield": ["Skeleton"]}, p1={"battlefield": ["Llanowar Elves", "Elvish Mystic"]}, step="declare_attackers")
    choose(g, "Attack with Skeleton")
    pass_priority(g, 2)
    choose(g, "blocks Skeleton")
    choose(g, "blocks Skeleton")
    assert len(g.blocks) == 2


# ---------------------------------------------------------------------------
# Sideboard
# ---------------------------------------------------------------------------


def test_monstrous_emergence_reveals_a_creature_for_its_power():
    g = scenario(p0={"hand": ["Monstrous Emergence", "Avenging Hunter"], "battlefield": FORESTS(2) + ["Llanowar Elves"]}, p1={"battlefield": ["Writhing Chrysalis"]})
    choose(g, "Cast Monstrous Emergence")
    choose(g, "Target Writhing Chrysalis")
    pay(g)
    assert g.decision.kind == O.CHOOSE_CARD
    assert [lab.split("#")[0] for lab in labels(g)] == ["Choose Llanowar Elves", "Reveal Avenging Hunter"]
    choose(g, "Reveal Avenging Hunter")
    resolve_stack(g)
    assert "Writhing Chrysalis" not in bf(g)


def test_monstrous_emergence_uses_the_chosen_creatures_current_power():
    g = scenario(p0={"hand": ["Monstrous Emergence"], "battlefield": FORESTS(2) + ["Llanowar Elves"]}, p1={"battlefield": ["Writhing Chrysalis"]})
    choose(g, "Cast Monstrous Emergence")
    choose(g, "Target Writhing Chrysalis")
    pay(g)  # the only choice (the Elf) is taken by settle()
    resolve_stack(g)
    assert find(g, "Writhing Chrysalis").damage == 1


def test_monstrous_emergence_needs_a_creature():
    g = scenario(p0={"hand": ["Monstrous Emergence", "Forest"], "battlefield": FORESTS(2)}, p1={"battlefield": ["Writhing Chrysalis"]})
    assert not has(g, "Cast Monstrous Emergence")


def test_vitu_ghazi_inspector_collects_evidence():
    g = scenario(p0={"hand": ["Vitu-Ghazi Inspector"], "battlefield": FORESTS(2) + ["Llanowar Elves"], "graveyard": ["Generous Ent", "Forest"]})
    assert has(g, "Cast Vitu-Ghazi Inspector (evidence)")
    choose(g, "Cast Vitu-Ghazi Inspector (evidence)")
    pay(g)
    assert g.decision.kind == O.EXILE_FROM_GY and labels(g) == ["Exile Generous Ent", "Exile Forest"]
    choose(g, "Exile Forest")  # 0 of 6: the Ent (the only card left) follows
    resolve_stack(g)
    settle(g)
    choose(g, "Target Llanowar Elves")
    resolve_stack(g)
    assert find(g, "Llanowar Elves").counters == 1 and g.players[0].life == 22
    assert names(g.players[0].exile) == ["Forest", "Generous Ent"]


def test_vitu_ghazi_inspector_without_evidence_has_no_trigger():
    g = scenario(p0={"hand": ["Vitu-Ghazi Inspector"], "battlefield": FORESTS(2), "graveyard": ["Generous Ent"]})
    choose(g, "Cast Vitu-Ghazi Inspector")
    pay(g)
    resolve_stack(g)
    assert not g.stack and g.players[0].life == 20
    g = scenario(p0={"hand": ["Vitu-Ghazi Inspector"], "battlefield": FORESTS(2), "graveyard": ["Winding Way", "Llanowar Elves"]})
    assert not has(g, "(evidence)")  # mana value 3 < 6


def test_spinewoods_paladin_plot_then_cast_free():
    g = scenario(p0={"hand": ["Spinewoods Paladin"], "battlefield": FORESTS(4)})
    choose(g, "Plot Spinewoods Paladin")
    pay(g)
    assert names(g.players[0].exile) == ["Spinewoods Paladin"]
    until_own_main(g, 3)
    choose(g, "Cast Spinewoods Paladin (plotted)")
    resolve_stack(g)
    resolve_stack(g)
    assert g.players[0].life == 23 and g.has(find(g, "Spinewoods Paladin"), "trample")


def test_deglamer_shuffles_an_artifact_into_its_owners_library():
    g = scenario(p0={"hand": ["Deglamer"], "battlefield": FORESTS(2)}, p1={"battlefield": ["Ichor Wellspring"]})
    choose(g, "Cast Deglamer")
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" in names(g.players[1].library) and not g.players[1].hand


# ---------------------------------------------------------------------------
# Generous Ent and Food
# ---------------------------------------------------------------------------


def test_generous_ent_forestcycling_finds_a_forest():
    g = scenario(p0={"hand": ["Generous Ent"], "battlefield": FORESTS(1), "library": ["Swamp", "Forest", "Swamp"]})
    choose(g, "Generous Ent: forestcycling {1}")
    pay(g)
    assert names(g.players[0].graveyard) == ["Generous Ent"]
    resolve_stack(g)
    assert labels(g) == ["Find nothing", "Find Forest"]
    choose(g, "Find Forest")
    assert names(g.players[0].hand) == ["Forest"]
    assert g.players[0].hand[0].known_to == {0, 1}  # revealed


def test_generous_ent_makes_food_and_food_gains_three():
    g = scenario(p0={"hand": ["Generous Ent"], "battlefield": FORESTS(8)})
    choose(g, "Cast Generous Ent")
    pay(g)
    resolve_stack(g)  # the Ent resolves, its ETB trigger goes on the stack
    resolve_stack(g)
    assert "Food" in bf(g, 0) and g.has(find(g, "Generous Ent"), "reach")
    choose(g, "Food: gain 3 life")
    pay(g)
    resolve_stack(g)
    assert g.players[0].life == 23 and "Food" not in bf(g)


# ---------------------------------------------------------------------------
# Bot and full games
# ---------------------------------------------------------------------------


def test_elves_bot_games_finish_against_every_deck():
    for i, matchup in enumerate(("jund_elves", "blue_elves", "madness_elves")):
        for game_no in (1, 2):
            g = new_game(seed=i * 10 + game_no, **game_args(game_no, matchup))
            bots = [make_bot(0, MATCHUPS[matchup][0]), make_bot(1, "elves")]
            while not g.over:
                d = g.decision
                a = bots[d.player].act(g)
                assert 0 <= a < len(d.options)
                g.step(a)
            assert g.end_reason in ("life", "decking", "turn limit")


def test_elves_bot_beats_random():
    wins = 0
    for s in range(12):
        g = new_game(seed=s, **game_args(1, "jund_elves"))
        agents = [RandomAgent(s), make_bot(1, "elves")]
        while not g.over:
            g.step(agents[g.decision.player].act(g))
        wins += g.winner == 1
    assert wins >= 10


def test_elves_bot_does_not_use_hidden_information():
    checked = 0
    for s in range(3):
        g, r = new_game(seed=s, **game_args(1, "blue_elves")), random.Random(s)
        bots = [make_bot(0, "mono_blue_terror"), make_bot(1, "elves")]
        while not g.over and checked < 120:
            d = g.decision
            if d.player == 1 and r.random() < 0.2:
                h = determinize(g, 1, random.Random(1000 * s))
                assert h.legal_options()[make_bot(1, "elves").act(h)].label == g.legal_options()[make_bot(1, "elves").act(g)].label, (d, g.turn)
                checked += 1
            g.step(bots[d.player].act(g))
    assert checked > 40
