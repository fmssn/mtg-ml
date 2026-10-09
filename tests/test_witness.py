"""`Game.witnessed(viewer)`: established minimum copies of opponent cards
(belief evidence). Both engines; `make difftest` checks they agree step by step."""

import random

import pytest
from helpers import choose, new_game, pass_priority, pay, resolve_stack, scenario

from mtg_ml.engine import objects as O
from mtg_ml.match import match_decks

pytestmark = pytest.mark.parametrize("engine", ["python", "native"], indirect=True)

ISLANDS = lambda n: ["Island"] * n  # noqa: E731
FORESTS = lambda n: ["Forest"] * n  # noqa: E731
SWAMPS = lambda n: ["Swamp"] * n  # noqa: E731


def test_board_graveyard_and_simultaneous_copies():
    g = scenario(p0={"battlefield": SWAMPS(2)}, p1={"battlefield": ["Delver of Secrets", "Delver of Secrets"] + ISLANDS(3), "graveyard": ["Delver of Secrets", "Ponder"], "hand": ["Counterspell"]})
    assert g.witnessed(0) == {"Delver of Secrets": 3, "Island": 3, "Ponder": 1}  # the hidden Counterspell is not seen
    assert g.witnessed(1) == {"Swamp": 2}
    assert list(g.witnessed(0)) == sorted(g.witnessed(0))


def test_bounce_and_recast_count_once():
    g = scenario(p1={"hand": ["Vapor Snag"], "battlefield": ["Delver of Secrets"] + ISLANDS(2)}, active=1)
    assert g.witnessed(0)["Delver of Secrets"] == 1
    choose(g, "Cast Vapor Snag")  # the only target
    pay(g)
    assert g.witnessed(0) == {"Delver of Secrets": 1, "Island": 2, "Vapor Snag": 1}
    resolve_stack(g)
    assert g.players[1].hand[0].name == "Delver of Secrets"  # known to both: it was seen
    choose(g, "Cast Delver of Secrets")
    pay(g)
    assert g.witnessed(0)["Delver of Secrets"] == 1  # on the stack now, still one card
    resolve_stack(g)
    assert g.witnessed(0) == {"Delver of Secrets": 1, "Island": 2, "Vapor Snag": 1}


def test_transformed_delver_and_its_revealed_card():
    g = scenario(p1={"battlefield": ["Delver of Secrets"], "library": ["Brainstorm"] + ISLANDS(10)}, active=1, step="upkeep")
    resolve_stack(g)
    choose(g, "Reveal Brainstorm")
    assert "Insectile Aberration" in [c.name for c in g.battlefield]
    w = g.witnessed(0)
    assert w["Delver of Secrets"] == 1 and "Insectile Aberration" not in w  # front face name
    assert w["Brainstorm"] == 1  # revealed on top of the library
    pass_priority(g)  # draw step: drawn, still known
    assert g.witnessed(0)["Brainstorm"] == 1


def test_tokens_are_excluded():
    g = scenario(p1={"battlefield": ["Eldrazi Spawn", "Treasure", "Island"]})
    assert g.witnessed(0) == {"Island": 1}


def test_transient_reveal_shuffled_away_is_kept():
    # Throne of the Dead Three reveals ten, puts one onto the battlefield, shuffles the rest away.
    g = scenario(p0={"library": ["Forest", "Generous Ent", "Llanowar Elves"] + SWAMPS(10)}, active=1, step="end")
    g.initiative = 0
    g.players[0].dungeon_room = "Archives"
    pass_priority(g, 2)
    resolve_stack(g)  # venture into Throne of the Dead Three
    resolve_stack(g)
    choose(g, "Put Generous Ent onto the battlefield")
    assert all(1 not in c.known_to for c in g.players[0].library)  # shuffled
    assert g.witnessed(1) == {"Forest": 1, "Generous Ent": 1, "Llanowar Elves": 1, "Swamp": 7}
    assert g.witnessed(0) == {}


def test_cascade_reveals_go_to_the_bottom_hidden():
    lib = ["Urza's Tower", "Forest", "Bramble Wurm", "Candy Trail", "Forest"]
    g = scenario(p0={"hand": ["Maelstrom Colossus"], "battlefield": ["Urza's Mine", "Urza's Power Plant", "Urza's Tower", "Urza's Mine"], "library": lib})
    choose(g, "Cast Maelstrom Colossus")
    pay(g)
    resolve_stack(g)
    choose(g, "Cast Bramble Wurm")
    assert not g.players[0].exile  # the lands are back in the library
    w = g.witnessed(1)
    assert w["Urza's Tower"] == 2 and w["Forest"] == 1 and w["Bramble Wurm"] == 1 and w["Maelstrom Colossus"] == 1
    assert "Candy Trail" not in w


def test_shuffled_into_library_and_hand_reveals():
    g = scenario(p0={"hand": ["Deglamer"], "battlefield": FORESTS(2)}, p1={"battlefield": ["Ichor Wellspring"]})
    choose(g, "Cast Deglamer")
    pay(g)
    resolve_stack(g)
    assert "Ichor Wellspring" not in [c.name for c in g.battlefield]
    assert g.witnessed(0) == {"Ichor Wellspring": 1}
    g = scenario(p0={"hand": ["Duress"], "battlefield": ["Swamp"]}, p1={"hand": ["Island", "Delver of Secrets", "Delver of Secrets"]})
    choose(g, "Cast Duress")
    pay(g)
    resolve_stack(g)
    assert g.witnessed(0) == {"Delver of Secrets": 2, "Island": 1}
    g = scenario(p0={"hand": ["Land Grant", "Llanowar Elves", "Winding Way"], "library": ["Swamp", "Forest", "Swamp"]})
    choose(g, "Cast Land Grant (alternative)")
    assert g.witnessed(1) == {"Land Grant": 1, "Llanowar Elves": 1, "Winding Way": 1}


def _random_game(seed: int):
    g = new_game(match_decks(1 + seed % 2), seed=seed, max_turns=30)
    return g, random.Random(seed)


@pytest.mark.parametrize("seed", range(4))
def test_monotonic_and_survives_copies(seed):
    from mtg_ml.rl.features import featurize

    g, r = _random_game(seed)
    prev = [{}, {}]
    n = 0
    while not g.over:
        for v in (0, 1):
            w = g.witnessed(v)
            assert all(w.get(k, 0) >= c for k, c in prev[v].items()), (n, v)
            prev[v] = w
        if n % 23 == 11:
            before = [g.witnessed(0), g.witnessed(1)]
            featurize(g, g.decision.player)  # simulated option previews run on copies
            assert [g.witnessed(0), g.witnessed(1)] == before
            c, rep = g.copy(), g.fork(replay=True)
            assert [c.witnessed(0), c.witnessed(1)] == before == [rep.witnessed(0), rep.witnessed(1)]
            a = r.randrange(len(g.legal_options()))
            c.step(a)
            g.step(a)
            assert [c.witnessed(0), c.witnessed(1)] == [g.witnessed(0), g.witnessed(1)]
        else:
            g.step(r.randrange(len(g.legal_options())))
        n += 1
    assert sum(prev[0].values()) and sum(prev[1].values())


def test_witnessing_does_not_change_the_game():
    g, r = _random_game(7)
    while not g.over:
        if g.decision.kind != O.PRIORITY:
            g.witnessed(0)
        g.step(r.randrange(len(g.legal_options())))
    h, r = _random_game(7)
    while not h.over:
        h.step(r.randrange(len(h.legal_options())))
    assert (g.actions, g.winner, g.turn, g.rng.getstate()) == (h.actions, h.winner, h.turn, h.rng.getstate())
