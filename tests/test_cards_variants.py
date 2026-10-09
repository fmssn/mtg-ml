"""Cards of the most-played real decklists that are not in a fixed decklist
yet: Spell Pierce (Mono Blue Terror), Prophetic Prism and Malevolent Rumble
(Tron), Fiery Cannonade (Tron's sideboard). Every test runs on both engines."""

from helpers import bf, choose, find, has, labels, names, pass_priority, pay, resolve_stack, scenario

from mtg_ml.engine import objects as O

ISLANDS = lambda n: ["Island"] * n  # noqa: E731

# ---------------------------------------------------------------------------
# Spell Pierce
# ---------------------------------------------------------------------------


def test_spell_pierce_counters_unless_two_is_paid():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 4}, p1={"hand": ["Spell Pierce"], "battlefield": ["Delver of Secrets"] + ISLANDS(1)})
    choose(g, "Cast Cast Down")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Spell Pierce")
    resolve_stack(g)
    assert g.decision.player == 0 and g.decision.kind == O.YES_NO
    assert labels(g) == ["Don't pay", "Pay {2}"]
    choose(g, "Pay {2}")
    resolve_stack(g)
    assert "Delver of Secrets" not in bf(g)
    assert names(g.players[1].graveyard) == ["Spell Pierce", "Delver of Secrets"]


def test_spell_pierce_counters_when_its_controller_cannot_pay():
    g = scenario(p0={"hand": ["Cast Down"], "battlefield": ["Swamp"] * 3}, p1={"hand": ["Spell Pierce"], "battlefield": ["Delver of Secrets"] + ISLANDS(1)})
    choose(g, "Cast Cast Down")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Spell Pierce")
    resolve_stack(g)  # one Swamp left: only "Don't pay" is legal
    assert "Delver of Secrets" in bf(g) and "Cast Down" in names(g.players[0].graveyard)


def test_spell_pierce_cannot_target_a_creature_spell():
    g = scenario(p0={"hand": ["Guttersnipe"], "battlefield": ["Mountain"] * 3}, p1={"hand": ["Spell Pierce"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Guttersnipe")
    pay(g)
    pass_priority(g)
    assert g.stack and g.decision.player == 1 and not has(g, "Spell Pierce")


# ---------------------------------------------------------------------------
# Prophetic Prism
# ---------------------------------------------------------------------------


def test_prophetic_prism_draws_a_card_when_it_enters():
    g = scenario(p0={"hand": ["Prophetic Prism"], "battlefield": ["Mountain"] * 2, "library": ["Ponder"] + ["Swamp"] * 5})
    choose(g, "Cast Prophetic Prism")
    pay(g)
    resolve_stack(g)
    assert "Prophetic Prism" in bf(g, 0) and names(g.players[0].hand) == ["Ponder"]


def test_prophetic_prism_filters_one_mana_by_tapping():
    g = scenario(p0={"hand": ["Blue Elemental Blast"], "battlefield": ["Prophetic Prism", "Mountain"]}, p1={"battlefield": ["Guttersnipe"]})
    choose(g, "Cast Blue Elemental Blast (destroy)")  # the Prism filter is the only way to pay {U}
    assert find(g, "Prophetic Prism").tapped and find(g, "Mountain").tapped
    resolve_stack(g)
    assert "Guttersnipe" not in bf(g)


def test_prophetic_prism_filters_only_once():
    g = scenario(p0={"hand": ["Counterspell"], "battlefield": ["Prophetic Prism"] + ["Mountain"] * 4})
    assert not has(g, "Cast Counterspell")  # {U}{U}: one tap, one coloured symbol


# ---------------------------------------------------------------------------
# Malevolent Rumble
# ---------------------------------------------------------------------------

RUMBLE_LIBRARY = ["Ponder", "Forest", "Lightning Bolt", "Guttersnipe", "Island"]


def test_malevolent_rumble_takes_a_permanent_card_and_mills_the_rest():
    g = scenario(p0={"hand": ["Malevolent Rumble"], "battlefield": ["Forest"] * 2, "library": RUMBLE_LIBRARY})
    choose(g, "Cast Malevolent Rumble")
    pay(g)
    resolve_stack(g)
    assert g.decision.kind == O.CHOOSE_CARD
    assert labels(g) == ["Take nothing", "Take Forest", "Take Guttersnipe"]
    assert all(c.known_to == {0, 1} for c in g.players[0].library[:4])
    choose(g, "Take Guttersnipe")
    assert names(g.players[0].hand) == ["Guttersnipe"] and g.players[0].hand[0].known_to == {0, 1}
    assert names(g.players[0].graveyard) == ["Ponder", "Forest", "Lightning Bolt", "Malevolent Rumble"]
    assert names(g.players[0].library) == ["Island"]
    assert bf(g, 0).count("Eldrazi Spawn") == 1


def test_malevolent_rumble_may_take_nothing():
    g = scenario(p0={"hand": ["Malevolent Rumble"], "battlefield": ["Forest"] * 2, "library": RUMBLE_LIBRARY})
    choose(g, "Cast Malevolent Rumble")
    pay(g)
    resolve_stack(g)
    choose(g, "Take nothing")
    assert g.players[0].hand == [] and len(g.players[0].graveyard) == 5
    assert bf(g, 0).count("Eldrazi Spawn") == 1


def test_malevolent_rumble_with_a_short_library():
    g = scenario(p0={"hand": ["Malevolent Rumble"], "battlefield": ["Forest"] * 2, "library": ["Ponder"]})
    choose(g, "Cast Malevolent Rumble")
    pay(g)
    resolve_stack(g)  # only "Take nothing": settled
    assert names(g.players[0].graveyard) == ["Ponder", "Malevolent Rumble"] and g.players[0].library == []
    assert bf(g, 0).count("Eldrazi Spawn") == 1


# ---------------------------------------------------------------------------
# Fiery Cannonade
# ---------------------------------------------------------------------------


def test_fiery_cannonade_deals_two_to_each_non_pirate_creature():
    g = scenario(
        p0={"hand": ["Fiery Cannonade"], "battlefield": ["Mountain"] * 3 + ["Guttersnipe", "Masked Vandal"]},
        p1={"battlefield": ["Delver of Secrets", "Tolarian Terror"]},
    )
    choose(g, "Cast Fiery Cannonade")
    pay(g)
    resolve_stack(g)
    assert "Guttersnipe" not in bf(g) and "Delver of Secrets" not in bf(g)
    assert find(g, "Tolarian Terror").damage == 2
    assert find(g, "Masked Vandal").damage == 0  # changeling: a Pirate
    assert g.players[1].life == 18  # Guttersnipe's cast trigger, not the Cannonade


def test_fiery_cannonade_kill_preview_spares_pirates():
    from mtg_ml.encode import option_preview

    g = scenario(
        p0={"hand": ["Fiery Cannonade"], "battlefield": ["Mountain"] * 3 + ["Masked Vandal", "Krark-Clan Shaman", "Ichor Wellspring"]},
        p1={"battlefield": ["Tolarian Terror"]},
    )
    choose(g, "Krark-Clan Shaman: 1 damage")
    resolve_stack(g)
    assert find(g, "Masked Vandal").damage == 1  # 1/3: two more would kill it, were it not a Pirate
    i = next(i for i, o in enumerate(g.legal_options()) if o.label == "Cast Fiery Cannonade")
    pv = set(option_preview(g, 0, i))
    assert "pv:kills_none" in pv and not any(t.startswith(("pv:kills_self", "pv:kills_opp")) for t in pv)


def test_spell_pierce_can_target_a_bestowed_spell():
    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest"] * 3 + ["Gixian Infiltrator"]}, p1={"hand": ["Spell Pierce"], "battlefield": ISLANDS(1)})
    choose(g, "Cast Nyxborn Hydra (bestow)")
    choose(g, "X=1")
    pay(g)
    pass_priority(g)
    choose(g, "Cast Spell Pierce")  # an Aura spell: the only target
    resolve_stack(g)
    assert "Nyxborn Hydra" in names(g.players[0].graveyard)
