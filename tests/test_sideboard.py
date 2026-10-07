"""Sideboard cards, sideboard plans and best-of-three matches."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.agents import RandomAgent
from mtg_ml.bots import make_bot
from mtg_ml.engine import DECKS, SIDEBOARD_PLANS, SIDEBOARDS, postboard
from mtg_ml.engine import objects as O
from mtg_ml.match import MATCHUPS, MatchResult, deck_names, match_decks, next_starting_player, play_match

ISLANDS = lambda n: ["Island"] * n  # noqa: E731


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
    assert "Go for the Throat" in g2[0] and "Red Elemental Blast" not in g2[0]


def test_deck_names_only_for_non_default_decks():
    assert deck_names("jund_blue") == (None, None)
    assert deck_names("jund_madness") == (None, "red_madness")


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
