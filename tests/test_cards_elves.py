"""Elves cards, its sideboard and the tokens and rules they need (Food,
initiative and the Undercity, ...). Every test runs on both engines."""

from helpers import bf, choose, find, labels, names, pay, resolve_stack, scenario


FORESTS = lambda n: ["Forest"] * n  # noqa: E731


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
