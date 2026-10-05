"""Mono Blue Terror cards."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.engine import objects as O

ISLANDS = lambda n: ["Island"] * n  # noqa: E731


def test_tolarian_terror_cost_reduction():
    g = scenario(p1={"hand": ["Tolarian Terror"], "battlefield": ISLANDS(4), "graveyard": ["Brainstorm", "Ponder", "Mental Note"]}, active=1)
    assert has(g, "Cast Tolarian Terror")  # {6}{U} - 3 = {3}{U}
    g = scenario(p1={"hand": ["Tolarian Terror"], "battlefield": ISLANDS(3), "graveyard": ["Brainstorm", "Ponder", "Mental Note"]}, active=1)
    assert not has(g, "Cast Tolarian Terror")
    # creature cards in the graveyard do not count
    g = scenario(p1={"hand": ["Cryptic Serpent"], "battlefield": ISLANDS(4), "graveyard": ["Delver of Secrets", "Ponder", "Brainstorm"]}, active=1)
    assert not has(g, "Cast Cryptic Serpent")  # {5}{U}{U} - 2 = {3}{U}{U}


def test_creatures_are_sorcery_speed():
    g = scenario(p1={"hand": ["Delver of Secrets"], "battlefield": ISLANDS(1)}, active=1, step="upkeep", auto_single=False)
    assert not has(g, "Cast Delver")
    g = scenario(p1={"hand": ["Delver of Secrets", "Brainstorm"], "battlefield": ISLANDS(1)}, active=1, step="upkeep", auto_single=False)
    assert has(g, "Cast Brainstorm")


def test_delver_transforms_on_revealed_instant():
    g = scenario(p1={"battlefield": ["Delver of Secrets"], "library": ["Brainstorm"] + ISLANDS(10)}, active=1, step="upkeep")
    # trigger on the stack, p1 has priority; passing resolves it
    resolve_stack(g)
    assert g.decision.kind == O.YES_NO
    choose(g, "Reveal Brainstorm")
    d = find(g, "Insectile Aberration")
    assert g.power(d) == 3 and g.toughness(d) == 2 and g.has(d, "flying")


def test_delver_does_not_transform_on_land_or_unrevealed():
    g = scenario(p1={"battlefield": ["Delver of Secrets"], "library": ISLANDS(10)}, active=1, step="upkeep")
    resolve_stack(g)
    choose(g, "Reveal Island")
    assert "Delver of Secrets" in bf(g)
    g = scenario(p1={"battlefield": ["Delver of Secrets"], "library": ["Ponder"] + ISLANDS(10)}, active=1, step="upkeep")
    resolve_stack(g)
    choose(g, "Don't reveal")
    assert "Delver of Secrets" in bf(g)


def test_brainstorm_puts_two_back_in_order():
    g = scenario(
        p1={"hand": ["Brainstorm", "Counterspell"], "battlefield": ISLANDS(1), "library": ["Ponder", "Delver of Secrets", "Mental Note"] + ISLANDS(5)},
        active=1,
    )
    choose(g, "Cast Brainstorm")
    resolve_stack(g)
    assert g.decision.kind == O.CHOOSE_CARD
    assert sorted(labels(g)) == sorted(["Put back Counterspell", "Put back Ponder", "Put back Delver of Secrets", "Put back Mental Note"])
    choose(g, "Put back Counterspell")
    choose(g, "Put back Delver of Secrets")
    lib = g.players[1].library
    assert names(lib[:2]) == ["Delver of Secrets", "Counterspell"]
    assert all(1 in c.known_to and 0 not in c.known_to for c in lib[:2])
    assert sorted(names(g.players[1].hand)) == ["Mental Note", "Ponder"]


def test_ponder_orders_unique_permutations_and_may_shuffle():
    g = scenario(p1={"hand": ["Ponder"], "battlefield": ISLANDS(1), "library": ["Brainstorm", "Island", "Island", "Delver of Secrets"]}, active=1)
    choose(g, "Cast Ponder")
    resolve_stack(g)
    assert g.decision.kind == O.ORDER
    assert len(g.legal_options()) == 3  # Brainstorm, Island, Island
    choose(g, "Top to bottom: Island, Island, Brainstorm")
    assert g.decision.kind == O.YES_NO
    choose(g, "Don't shuffle")
    assert names(g.players[1].hand) == ["Island"]
    assert names(g.players[1].library) == ["Island", "Brainstorm", "Delver of Secrets"]


def test_thought_scour_targets_any_player():
    g = scenario(p1={"hand": ["Thought Scour"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Thought Scour")
    assert g.decision.kind == O.TARGET
    choose(g, "(opponent)")
    resolve_stack(g)
    assert len(g.players[0].graveyard) == 2 and len(g.players[1].hand) == 1


def test_counterspell_only_targets_spells():
    # an activated ability on the stack is not a legal Counterspell target
    g = scenario(
        p0={"battlefield": ["Krark-Clan Shaman", "Ichor Wellspring"]},
        p1={"hand": ["Counterspell"], "battlefield": ISLANDS(2)},
        auto_single=False,
    )
    choose(g, "Krark-Clan Shaman: 1 damage")
    pass_priority(g)
    assert g.decision.player == 1 and not has(g, "Cast Counterspell")


def test_counterspell_counters():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp", "Swamp"]}, p1={"hand": ["Counterspell"], "battlefield": ["Delver of Secrets"] + ISLANDS(2)})
    choose(g, "Cast Cast Down")
    pass_priority(g)
    choose(g, "Cast Counterspell")
    resolve_stack(g)
    assert "Delver of Secrets" in bf(g) and "Cast Down" in names(g.players[0].graveyard)


def test_force_spike_payment_choice():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 3}, p1={"hand": ["Force Spike"], "battlefield": ["Delver of Secrets"] + ISLANDS(1)})
    choose(g, "Cast Cast Down")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Force Spike")
    resolve_stack(g)
    assert g.decision.player == 0 and g.decision.kind == O.YES_NO
    assert labels(g) == ["Don't pay", "Pay {1}"]
    choose(g, "Pay {1}")
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g)


def test_force_spike_cannot_pay_when_tapped_out():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 2}, p1={"hand": ["Force Spike"], "battlefield": ["Delver of Secrets"] + ISLANDS(1)})
    choose(g, "Cast Cast Down")
    pass_priority(g)
    choose(g, "Cast Force Spike")
    resolve_stack(g)
    assert "Delver of Secrets" in bf(g)  # only "Don't pay" was legal, auto-chosen


def test_lorien_revealed_cast_and_islandcycling():
    g = scenario(p1={"hand": ["Lorien Revealed"], "battlefield": ISLANDS(1), "library": ["Ponder", "Island"]}, active=1)
    assert labels(g) == ["Pass priority", "Lorien Revealed: islandcycling {1}"]
    choose(g, "islandcycling")
    resolve_stack(g)
    assert labels(g) == ["Find nothing", "Find Island"]
    choose(g, "Find Island")
    assert "Island" in names(g.players[1].hand) and "Lorien Revealed" in names(g.players[1].graveyard)
    g = scenario(p1={"hand": ["Lorien Revealed"], "battlefield": ISLANDS(5)}, active=1)
    choose(g, "Cast Lorien Revealed")
    resolve_stack(g)
    assert len(g.players[1].hand) == 3


def test_deem_inferior_cost_and_owner_choice():
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman"]}, p1={"hand": ["Deem Inferior"], "battlefield": ISLANDS(2), "drawn": 2}, active=1)
    assert has(g, "Cast Deem Inferior")  # {3}{U} - 2
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman"]}, p1={"hand": ["Deem Inferior"], "battlefield": ISLANDS(2), "drawn": 1}, active=1)
    assert not has(g, "Cast Deem Inferior")
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman"]}, p1={"hand": ["Deem Inferior"], "battlefield": ISLANDS(2), "drawn": 2}, active=1)
    choose(g, "Cast Deem Inferior")
    resolve_stack(g)
    assert g.decision.player == 0  # the owner chooses
    choose(g, "second from the top")
    lib = g.players[0].library
    assert lib[1].name == "Krark-Clan Shaman" and lib[1].known_to == {0, 1}


def test_deem_inferior_on_token_removes_it():
    g = scenario(p0={"battlefield": ["Map"]}, p1={"hand": ["Deem Inferior"], "battlefield": ISLANDS(4)}, active=1)
    choose(g, "Cast Deem Inferior")
    pay(g)
    resolve_stack(g)
    choose(g, "bottom")
    assert "Map" not in bf(g) and all(c.name != "Map" for c in g.players[0].library)


def test_sleep_of_the_dead_and_escape():
    g = scenario(
        p0={"battlefield": ["Gixian Infiltrator"]},
        p1={"battlefield": ISLANDS(3), "graveyard": ["Sleep of the Dead", "Brainstorm", "Ponder", "Mental Note"]},
        active=1,
    )
    choose(g, "Cast Sleep of the Dead (escape)")
    pay(g)
    assert g.decision.kind == O.EXILE_FROM_GY
    assert sorted(labels(g)) == ["Exile Brainstorm", "Exile Mental Note", "Exile Ponder"]
    choose(g, "Exile Ponder")
    choose(g, "Exile Brainstorm")
    resolve_stack(g)
    gix = find(g, "Gixian Infiltrator")
    assert gix.tapped and gix.skip_untap == 1
    assert names(g.players[1].graveyard) == ["Sleep of the Dead"]  # escape does not exile a sorcery
    assert sorted(names(g.players[1].exile)) == ["Brainstorm", "Mental Note", "Ponder"]
    g.active = 0
    g._untap_step()
    assert gix.tapped and gix.skip_untap == 0
    g._untap_step()
    assert not gix.tapped


def test_escape_needs_three_other_cards():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"battlefield": ISLANDS(3), "graveyard": ["Sleep of the Dead", "Brainstorm", "Ponder"]}, active=1)
    assert not has(g, "escape")


def test_plunder_flashback_draws_two_and_exiles():
    g = scenario(p1={"battlefield": ISLANDS(4), "graveyard": ["Plunder the Trollshaws"]}, active=1, step="end")
    choose(g, "Cast Plunder the Trollshaws (flashback)")
    resolve_stack(g)
    assert len(g.players[1].hand) == 2
    assert names(g.players[1].exile) == ["Plunder the Trollshaws"]


def test_ward_counters_unless_paid():
    # can pay: Cast Down resolves
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 4}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Cast Down")
    pay(g)
    assert g.stack[-1].kind == "trigger" and "ward" in g.stack[-1].name
    resolve_stack(g)
    assert g.decision.player == 0 and labels(g) == ["Don't pay", "Pay {2}"]
    choose(g, "Pay {2}")
    resolve_stack(g)
    assert "Tolarian Terror" not in bf(g)
    # cannot pay: Cast Down is countered
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 2}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Cast Cast Down")
    resolve_stack(g)
    assert "Tolarian Terror" in bf(g) and "Cast Down" in names(g.players[0].graveyard)


def test_ward_does_not_trigger_on_own_spells():
    g = scenario(p1={"hand": ["Sleep of the Dead"], "battlefield": ["Tolarian Terror"] + ISLANDS(1)}, active=1)
    choose(g, "Cast Sleep of the Dead")  # the only creature is targeted automatically
    assert g.stack[-1].targets and not any(it.kind == "trigger" for it in g.stack)


def test_ward_on_abilities():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Ichor Wellspring", "Swamp"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Makeshift Munitions: 1 damage")
    choose(g, "Target Tolarian Terror")
    assert any("ward" in it.name for it in g.stack)
