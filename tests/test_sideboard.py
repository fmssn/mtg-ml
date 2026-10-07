import pytest
"""Sideboard cards, sideboard plans and best-of-three matches."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.agents import RandomAgent
from mtg_ml.bots import make_bot
from mtg_ml.engine import DECKS, SIDEBOARD_PLANS, SIDEBOARDS, expand, postboard
from mtg_ml.engine import objects as O
from mtg_ml.encode import state_features
from mtg_ml.match import MATCHUPS, MatchResult, deck_names, game_args, match_decks, next_starting_player, parse_matchups, play_match

ISLANDS = lambda n: ["Island"] * n  # noqa: E731
MOUNTAINS = lambda n: ["Mountain"] * n  # noqa: E731


# ---------------------------------------------------------------------------
# Shared sideboard hate (Q3 2026 sideboards)
# ---------------------------------------------------------------------------


def test_hydroblast_counters_and_destroys_only_red():
    g = scenario(p0={"hand": ["Lightning Bolt"], "battlefield": MOUNTAINS(1)}, p1={"hand": ["Hydroblast"], "battlefield": ISLANDS(1) + ["Delver of Secrets"]})
    choose(g, "Cast Lightning Bolt")
    choose(g, "Target player 1")
    pass_priority(g)
    choose(g, "Cast Hydroblast (counter)")
    resolve_stack(g)
    assert g.players[1].life == 20 and "Lightning Bolt" in names(g.players[0].graveyard)
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Refurbished Familiar"]}, p1={"hand": ["Hydroblast"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Hydroblast (destroy)")
    assert any("Refurbished Familiar" in x for x in labels(g))  # any permanent can be targeted
    choose(g, "Target Refurbished Familiar")
    resolve_stack(g)
    assert "Refurbished Familiar" in bf(g)  # not red: nothing happens
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman"]}, p1={"hand": ["Hydroblast"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Hydroblast (destroy)")
    choose(g, "Target Krark-Clan Shaman")
    resolve_stack(g)
    assert "Krark-Clan Shaman" not in bf(g)


def test_annul_counters_artifact_and_enchantment_spells_only():
    g = scenario(p0={"hand": ["Ichor Wellspring", "Cast Down"], "battlefield": ["Swamp"] * 4}, p1={"hand": ["Annul"], "battlefield": ISLANDS(1) + ["Delver of Secrets"]})
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Annul")
    resolve_stack(g)
    assert "Ichor Wellspring" in names(g.players[0].graveyard) and "Ichor Wellspring" not in bf(g)
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 2}, p1={"hand": ["Annul"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Nyxborn Hydra")
    choose(g, "X=1")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Annul")  # an enchantment creature spell
    resolve_stack(g)
    assert "Nyxborn Hydra" in names(g.players[0].graveyard)
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 2}, p1={"hand": ["Annul"], "battlefield": ISLANDS(1) + ["Delver of Secrets"]})
    choose(g, "Cast Cast Down")
    pay(g)
    pass_priority(g)
    assert not has(g, "Cast Annul")


def test_gut_shot_for_red_mana_or_two_life():
    g = scenario(p0={"battlefield": ["Delver of Secrets"]}, p1={"hand": ["Gut Shot"], "battlefield": ISLANDS(2)}, active=1)
    assert has(g, "Cast Gut Shot (phyrexian)") and not any(x == "Cast Gut Shot" for x in labels(g))  # no red mana
    choose(g, "Cast Gut Shot (phyrexian)")
    choose(g, "Target Delver of Secrets")
    assert g.players[1].life == 18 and all(not c.tapped for c in g.battlefield if c.controller == 1)
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g)
    g = scenario(p0={"hand": ["Gut Shot"], "battlefield": MOUNTAINS(1)}, p1={"battlefield": ["Delver of Secrets"]})
    assert has(g, "Cast Gut Shot (phyrexian)")
    choose(g, "Cast Gut Shot")
    choose(g, "Target player 1")
    resolve_stack(g)
    assert g.players[0].life == 20 and g.players[1].life == 19
    g = scenario(p0={"hand": ["Gut Shot"], "life": 1}, p1={})
    assert not has(g, "Cast Gut Shot")  # cannot pay 2 life with 1


def test_troublemaker_ouphe_bargained_exiles_an_artifact():
    opp = {"battlefield": ["Ichor Wellspring", "Island"]}
    g = scenario(p0={"hand": ["Troublemaker Ouphe"], "battlefield": ["Forest", "Forest", "Clue"]}, p1=opp)
    assert has(g, "Cast Troublemaker Ouphe (bargain)")
    choose(g, "Cast Troublemaker Ouphe (bargain)")
    pay(g)  # the Clue is the only thing to sacrifice
    assert "Clue" not in bf(g)
    resolve_stack(g)  # Ouphe resolves; its trigger targets the only opposing artifact
    resolve_stack(g)
    assert "Ichor Wellspring" not in bf(g) and "Ichor Wellspring" in names(g.players[1].exile)
    assert "Troublemaker Ouphe" in bf(g)
    g = scenario(p0={"hand": ["Troublemaker Ouphe"], "battlefield": ["Forest", "Forest", "Clue"]}, p1=opp)
    choose(g, "Cast Troublemaker Ouphe")  # not bargained: no trigger
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" in bf(g) and "Clue" in bf(g) and not g.stack
    g = scenario(p0={"hand": ["Troublemaker Ouphe"], "battlefield": ["Forest", "Forest", "Clue"]}, p1={"battlefield": ["Island"]})
    choose(g, "Cast Troublemaker Ouphe (bargain)")
    pay(g)
    resolve_stack(g)
    assert not g.stack and "Troublemaker Ouphe" in bf(g)  # no target: the trigger is removed
    g = scenario(p0={"hand": ["Troublemaker Ouphe"], "battlefield": ["Forest", "Forest"]}, p1=opp)
    assert has(g, "Cast Troublemaker Ouphe") and not has(g, "(bargain)")  # nothing to sacrifice


def test_breath_weapon_hits_every_creature():
    g = scenario(p0={"hand": ["Breath Weapon"], "battlefield": MOUNTAINS(3) + ["Krark-Clan Shaman", "Writhing Chrysalis"]}, p1={"battlefield": ["Delver of Secrets", "Tolarian Terror"]})
    choose(g, "Cast Breath Weapon")
    pay(g)
    resolve_stack(g)
    assert bf(g, 0) == MOUNTAINS(3) + ["Writhing Chrysalis"] and bf(g, 1) == ["Tolarian Terror"]


def test_faerie_macabre_exiles_up_to_two_cards_from_graveyards():
    g = scenario(p0={"hand": ["Faerie Macabre"]}, p1={"graveyard": ["Brainstorm", "Ponder", "Counterspell"]}, active=1)
    pass_priority(g)
    choose(g, "Faerie Macabre: exile up to two cards from graveyards")
    assert "Faerie Macabre" in names(g.players[0].graveyard)
    resolve_stack(g)
    assert sorted(labels(g)) == ["Exile Brainstorm (opponent graveyard)", "Exile Counterspell (opponent graveyard)", "Exile Faerie Macabre (self graveyard)", "Exile Ponder (opponent graveyard)", "Exile nothing more"]
    choose(g, "Exile Brainstorm")
    choose(g, "Exile Ponder")
    assert names(g.players[1].graveyard) == ["Counterspell"] and sorted(names(g.players[1].exile)) == ["Brainstorm", "Ponder"]
    g = scenario(p0={"hand": ["Faerie Macabre"]}, p1={"graveyard": ["Brainstorm"]}, active=1)
    pass_priority(g)
    choose(g, "Faerie Macabre: exile")
    resolve_stack(g)
    choose(g, "Exile nothing more")
    assert names(g.players[1].graveyard) == ["Brainstorm"]


def test_terminate_destroys_any_creature():
    g = scenario(p0={"hand": ["Terminate"], "battlefield": ["Swamp", "Mountain"]}, p1={"battlefield": ["Cryptic Serpent"]})
    choose(g, "Cast Terminate")
    pay(g)
    resolve_stack(g)
    assert "Cryptic Serpent" not in bf(g)


def test_cast_into_the_fire_modes():
    g = scenario(p0={"hand": ["Cast into the Fire"], "battlefield": MOUNTAINS(2)}, p1={"battlefield": ["Delver of Secrets"]})
    assert has(g, "(one creature)") and not has(g, "(two creatures)") and not has(g, "(exile)")
    g = scenario(p0={"hand": ["Cast into the Fire"], "battlefield": MOUNTAINS(2) + ["Clue"]}, p1={"battlefield": ["Delver of Secrets", "Delver of Secrets"]})
    choose(g, "Cast Cast into the Fire (two creatures)")
    first = labels(g)
    choose(g, first[0])
    assert len(labels(g)) == 1 or labels(g) != first  # the first target is not offered again
    if g.decision.kind == O.TARGET:
        g.step(0)
    pay(g)
    resolve_stack(g)
    assert bf(g, 1) == []
    g = scenario(p0={"hand": ["Cast into the Fire"], "battlefield": MOUNTAINS(2)}, p1={"battlefield": ["Ichor Wellspring", "Delver of Secrets"]})
    choose(g, "Cast Cast into the Fire (exile)")
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" in names(g.players[1].exile) and not g.stack  # exiled: no dies trigger


def test_end_the_festivities_hits_the_opponent_and_their_creatures():
    g = scenario(p0={"hand": ["End the Festivities"], "battlefield": MOUNTAINS(1) + ["Krark-Clan Shaman"]}, p1={"battlefield": ["Delver of Secrets", "Cryptic Serpent"]})
    choose(g, "Cast End the Festivities")
    resolve_stack(g)
    assert g.players[1].life == 19 and g.players[0].life == 20
    assert bf(g, 0) == MOUNTAINS(1) + ["Krark-Clan Shaman"] and bf(g, 1) == ["Cryptic Serpent"]


def test_smash_to_smithereens_and_ancient_grudge():
    g = scenario(p0={"hand": ["Smash to Smithereens"], "battlefield": MOUNTAINS(2)}, p1={"battlefield": ["Refurbished Familiar"]})
    choose(g, "Cast Smash to Smithereens")
    pay(g)
    resolve_stack(g)
    assert "Refurbished Familiar" not in bf(g) and g.players[1].life == 17 and g.players[0].life == 20
    g = scenario(p0={"hand": ["Ancient Grudge"], "battlefield": MOUNTAINS(2) + ["Forest"]}, p1={"battlefield": ["Ichor Wellspring", "Lembas"]})
    choose(g, "Cast Ancient Grudge")
    choose(g, "Target Lembas")
    choose(g, "Tap Mountain")
    pay(g)
    resolve_stack(g)
    assert "Lembas" not in bf(g)
    choose(g, "Cast Ancient Grudge (flashback)")
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" not in bf(g) and "Ancient Grudge" in names(g.players[0].exile)


def test_red_elemental_blast_modes_and_targets():
    g = scenario(p0={"hand": ["Red Elemental Blast"], "battlefield": ["Mountain"]}, p1={"battlefield": ["Delver of Secrets", "Island"]})
    assert has(g, "Cast Red Elemental Blast (destroy)") and not has(g, "(counter)")  # no blue spell on the stack
    choose(g, "Cast Red Elemental Blast (destroy)")
    labels_ = labels(g) if g.decision.kind == O.TARGET else []
    assert all("Island" not in x for x in labels_)  # lands are colourless
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g)


def test_red_elemental_blast_counters_blue_spell_only():
    g = scenario(p0={"hand": ["Red Elemental Blast"], "battlefield": ["Mountain"]}, p1={"hand": ["Ponder"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Ponder")
    pass_priority(g)
    choose(g, "Cast Red Elemental Blast (counter)")
    resolve_stack(g)
    assert "Ponder" in names(g.players[1].graveyard) and len(g.players[1].hand) == 0


def test_insectile_aberration_is_blue_and_chrysalis_is_colourless():
    g = scenario(p0={"hand": ["Red Elemental Blast", "Blue Elemental Blast"], "battlefield": ["Mountain", ("Writhing Chrysalis", {})]}, p1={"battlefield": ["Delver of Secrets"]})
    find(g, "Delver of Secrets").transformed = True
    assert g.target_candidates(O.TargetSpec("blue_permanent"), 0) == [("perm", find(g, "Insectile Aberration").oid)]
    assert g.target_candidates(O.TargetSpec("red_permanent"), 1) == []


def test_red_elemental_blast_on_terror_triggers_ward():
    g = scenario(p0={"hand": ["Red Elemental Blast"], "battlefield": ["Mountain"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Red Elemental Blast (destroy)")
    assert any("ward" in it.name for it in g.stack)


def test_blue_elemental_blast_counters_red_and_destroys_red():
    g = scenario(p0={"hand": ["Cleansing Wildfire"], "battlefield": ["Mountain", "Mountain"]}, p1={"hand": ["Blue Elemental Blast"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Cleansing Wildfire")
    if g.decision.kind == O.TARGET:
        g.step(0)
    pay(g)
    pass_priority(g)
    choose(g, "Cast Blue Elemental Blast (counter)")
    resolve_stack(g)
    assert "Cleansing Wildfire" in names(g.players[0].graveyard)
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Writhing Chrysalis"]}, p1={"hand": ["Blue Elemental Blast"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Blue Elemental Blast (destroy)")  # Chrysalis is devoid: Shaman is the only target
    resolve_stack(g)
    assert "Krark-Clan Shaman" not in bf(g) and "Writhing Chrysalis" in bf(g)


def test_go_for_the_throat_cannot_hit_artifact_creatures():
    g = scenario(p0={"hand": ["Go for the Throat"], "battlefield": ["Swamp", "Swamp", "Refurbished Familiar"]}, p1={"battlefield": ["Delver of Secrets"]})
    choose(g, "Cast Go for the Throat")  # Delver is the only legal target
    pay(g)
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g) and "Refurbished Familiar" in bf(g)


def test_duress_reveals_and_takes_noncreature_nonland():
    g = scenario(p0={"hand": ["Duress"], "battlefield": ["Swamp"]}, p1={"hand": ["Counterspell", "Delver of Secrets", "Island", "Brainstorm"]})
    choose(g, "Cast Duress")
    resolve_stack(g)
    assert sorted(labels(g)) == ["Discard Brainstorm", "Discard Counterspell"]
    assert all(c.known_to == {0, 1} for c in g.players[1].hand)
    choose(g, "Discard Counterspell")
    assert "Counterspell" in names(g.players[1].graveyard) and len(g.players[1].hand) == 3


def test_duress_whiffs_on_creatures_and_lands():
    g = scenario(p0={"hand": ["Duress"], "battlefield": ["Swamp"]}, p1={"hand": ["Delver of Secrets", "Island"]})
    choose(g, "Cast Duress")
    resolve_stack(g)
    assert len(g.players[1].hand) == 2 and g.decision.kind == O.PRIORITY


def test_dispel_counters_instants_only():
    g = scenario(p0={"hand": ["Cast Down", "Ichor Wellspring"], "battlefield": ["Swamp"] * 4}, p1={"hand": ["Dispel"], "battlefield": ["Delver of Secrets", "Island"]})
    choose(g, "Cast Ichor Wellspring")
    pay(g)
    pass_priority(g)
    assert not has(g, "Cast Dispel")  # artifact spell: not an instant
    resolve_stack(g)
    choose(g, "Cast Cast Down")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Dispel")
    resolve_stack(g)
    assert "Delver of Secrets" in bf(g)


def test_steel_sabotage_counters_artifact_or_bounces():
    g = scenario(p0={"hand": ["Refurbished Familiar"], "battlefield": ["Swamp"] * 4}, p1={"hand": ["Steel Sabotage", "Island"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Refurbished Familiar")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Steel Sabotage (counter)")
    resolve_stack(g)
    assert "Refurbished Familiar" in names(g.players[0].graveyard)
    g = scenario(p0={"battlefield": ["Drossforge Bridge", "Clue"]}, p1={"hand": ["Steel Sabotage"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Steel Sabotage (bounce)")
    choose(g, "Target Drossforge Bridge")
    resolve_stack(g)
    assert "Drossforge Bridge" in names(g.players[0].hand)
    g = scenario(p0={"battlefield": ["Clue"]}, p1={"hand": ["Steel Sabotage"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Steel Sabotage (bounce)")
    resolve_stack(g)
    assert "Clue" not in bf(g) and "Clue" not in names(g.players[0].hand)  # tokens cease to exist


def test_postboard_decks_are_legal():
    for d, o in SIDEBOARD_PLANS:
        pb = postboard(d, o)
        assert sum(pb.values()) == 60 and all(n <= 4 or name in ("Island", "Mountain") for name, n in pb.items())
        assert set(pb) - set(DECKS[d]) <= set(SIDEBOARDS[d])
    for a, b in MATCHUPS.values():
        assert (a, b) in SIDEBOARD_PLANS and (b, a) in SIDEBOARD_PLANS
    g1, g2 = match_decks(1), match_decks(2)
    assert "Duress" not in g1[0] and "Duress" in g2[0] and "Dispel" in g2[1]
    g1, g2 = match_decks(1, "jund_madness"), match_decks(2, "jund_madness")
    assert "Fiery Temper" in g1[1] and "Electrickery" not in g1[1] and "Electrickery" in g2[1]
    assert g2[0].count("Weather the Storm") == 3 and g2[0].count("Cleansing Wildfire") == 2 and "Red Elemental Blast" not in g2[0]


def test_deck_names_only_for_non_default_decks():
    assert deck_names("jund_blue") == (None, None)
    assert deck_names("jund_madness") == (None, "red_madness")
    assert deck_names("blue_madness") == ("mono_blue_terror", "red_madness")


def test_opponent_deck_feature_from_set_3(engine):
    """Set 3 names the opponent's deck where it is not its seat's usual one,
    so every seat of every matchup tells its opponent apart; jund_blue is
    unchanged from set 2."""
    from helpers import new_game

    seen = {}
    for m in MATCHUPS:
        g = new_game(seed=1, **game_args(1, m))
        for v in (0, 1):
            f3, f2 = state_features(g, v, 3), state_features(g, v, 2)
            seen[(m, v)] = tuple(sorted(x for x in f3 if x.startswith(("self:deck:", "opp:deck:"))))
            assert [x for x in f3 if not x.startswith("opp:deck:")] == f2
    assert seen[("jund_blue", 0)] == seen[("jund_blue", 1)] == ()
    assert seen[("jund_madness", 0)] == ("opp:deck:red_madness",)
    assert seen[("blue_madness", 1)] == ("opp:deck:mono_blue_terror", "self:deck:red_madness")
    assert len(set(seen.values())) == len(seen) - 1  # only jund_blue's two seats look alike (seat is a feature of its own)


def test_blue_vs_red_bots_play_a_match(engine):
    res = play_match([make_bot(0, "mono_blue_terror"), make_bot(1, "red_madness")], seed=5, engine=engine, matchup="blue_madness")
    assert res.over and len(res.games) >= 2


def test_parse_matchups():
    assert parse_matchups("jund_blue") == [("jund_blue", 1.0)]
    assert parse_matchups("jund_blue:2, jund_madness,blue_madness:0.5") == [("jund_blue", 2.0), ("jund_madness", 1.0), ("blue_madness", 0.5)]
    for bad in ("nope", "jund_blue,jund_blue", "jund_blue:0"):
        with pytest.raises(ValueError):
            parse_matchups(bad)


def test_match_loser_starts_next_game_and_best_of_three():
    assert next_starting_player(0, 0) == 1 and next_starting_player(0, 1) == 0 and next_starting_player(1, None) == 1
    r = MatchResult([(0, 0, "life"), (1, 0, "life")])
    assert r.over and r.winner == 0
    r = MatchResult([(0, 0, "life"), (1, 1, "life")])
    assert not r.over and r.next_game(7)[0] == 3 and r.next_game(7)[1] == 0
    r = MatchResult([(0, 0, "life"), (1, 1, "life"), (0, None, "turn limit")])
    assert r.over and r.winner is None


def test_play_match_runs_with_bots_and_random():
    for s in range(6):
        r = play_match([make_bot(0), RandomAgent(s)] if s % 2 else [make_bot(0), make_bot(1)], seed=s)
        assert r.over and 2 <= len(r.games) <= 3
        for (st, w, _), (st2, _, _) in zip(r.games, r.games[1:]):
            assert st2 == (st if w is None else 1 - w)


# ---------------------------------------------------------------------------
# Plan table (engine/sideboard_plans.toml) and the sideboarding decision step
# ---------------------------------------------------------------------------


def test_every_pair_of_decks_has_a_valid_plan():
    from mtg_ml.engine import PLANS
    from mtg_ml.engine.sideboard import validate_plan

    for d in DECKS:
        for o in DECKS:
            if d != o:
                assert (d, o) in PLANS, f"no sideboard plan for {d} vs {o}"
    for (d, o), plan in PLANS.items():
        validate_plan(plan)
        assert plan.why and plan.swaps > 0
        assert sum(postboard(d, o).values()) == 60


def test_plan_validation_rejects_bad_rows():
    from mtg_ml.engine.sideboard import parse_plans

    row = '[[plan]]\ndeck = "jund_wildfire"\nopponent = "red_madness"\nwhy = "x"\n'
    assert parse_plans(row + 'in = { "Weather the Storm" = 3 }\nout = { "Cleansing Wildfire" = 3 }\n')
    bad = [
        'in = { "Weather the Storm" = 3 }\nout = { "Cleansing Wildfire" = 2 }\n',  # 3 in, 2 out
        'in = { "Weather the Storm" = 9 }\nout = { "Cleansing Wildfire" = 4, "Cast Down" = 4, "Lembas" = 1 }\n',  # more than the sideboard has
        'in = { "Lightning Bolt" = 1 }\nout = { "Cast Down" = 1 }\n',  # not in the sideboard
        'in = { "Weather the Storm" = 1 }\nout = { "Lightning Bolt" = 1 }\n',  # not in the maindeck
        'in = { "Weather the Storm" = 1 }\nout = { "Cast Down" = 1 }\ncolour = "red"\n',  # unknown field
    ]
    for b in bad:
        with pytest.raises(ValueError):
            parse_plans(row + b)
    with pytest.raises(ValueError):
        parse_plans(row.replace("red_madness", "no_such_deck") + "in = {}\nout = {}\n")
    with pytest.raises(ValueError):
        parse_plans((row + "in = {}\nout = {}\n") * 2)  # duplicate pair


def test_plan_matrix_policy_reproduces_the_plan_table(engine):
    """play_match with the default policy plays exactly game_args' decks:
    maindecks in game 1, the table's plan in games 2 and 3."""
    from mtg_ml.engine import plan_for
    from mtg_ml.match import PlanMatrixPolicy, matchup_decks

    for m in MATCHUPS:
        a, b = matchup_decks(m)
        res = play_match([make_bot(0, a), make_bot(1, b)], seed=11, engine=engine, matchup=m)
        assert len(res.plans) == len(res.games) == len(res.seen)
        for n, plans in enumerate(res.plans, 1):
            assert tuple(expand(p.apply()) for p in plans) == match_decks(n, m)
            if n > 1:
                assert plans == (plan_for(a, b), plan_for(b, a))
        h = res.history(0)
        assert [x.game_no for x in h] == list(range(1, len(res.games) + 1))
        assert PlanMatrixPolicy().choose(a, b, 2, h) == plan_for(a, b)
        assert all(x.result in ("win", "loss", "draw") for x in h)
        assert any(x.opponent_seen for x in h)


def test_custom_policy_sees_history_and_is_validated(engine):
    from mtg_ml.engine import SideboardPlan
    from mtg_ml.match import NoSideboardPolicy

    calls = []

    class Recorder:
        def choose(self, deck, opponent, game_no, history):
            calls.append((deck, opponent, game_no, history))
            return SideboardPlan.none(deck, opponent)

    res = play_match([make_bot(0), make_bot(1)], seed=2, engine=engine, policies=(Recorder(), NoSideboardPolicy()))
    assert [c[2] for c in calls] == list(range(2, len(res.games) + 1))
    assert all(len(c[3]) == c[2] - 1 for c in calls)
    assert all(p.swaps == 0 for ps in res.plans for p in ps)

    class Cheater:
        def choose(self, deck, opponent, game_no, history):
            return SideboardPlan(deck, opponent, {"Lightning Bolt": 4}, {"Cast Down": 4}, "illegal")

    with pytest.raises(ValueError):
        play_match([make_bot(0), make_bot(1)], seed=2, engine=engine, policies=(Cheater(), Cheater()))
