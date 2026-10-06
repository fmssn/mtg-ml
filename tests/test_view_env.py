import random

from helpers import choose, new_game, pay, resolve_stack, scenario

from mtg_ml.encode import encode_state, state_features
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, expand
from mtg_ml.engine.view import determinize, observe
from mtg_ml.env import MTGEnv

DECKS = (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))


def advance(g, n, seed=0):
    r = random.Random(seed)
    for _ in range(n):
        if g.over:
            return
        g.step(0 if g.decision.kind == "mulligan" else r.randrange(len(g.legal_options())))


def test_observe_hides_opponent_hand_and_library():
    g = new_game(DECKS, seed=1)
    o = observe(g, 0)
    assert len(o["self"]["hand"]) == 7
    assert o["opponent"]["hand_count"] == 7 and o["opponent"]["hand_known"] == []
    assert o["self"]["library_known"] == [] and o["opponent"]["library_known"] == []


def test_brainstorm_knowledge_is_private():
    g = scenario(p1={"hand": ["Brainstorm", "Ponder"], "battlefield": ["Island"], "library": ["Mental Note"] * 5}, active=1)
    choose(g, "Cast Brainstorm")
    resolve_stack(g)
    choose(g, "Put back Ponder")  # the second card (a Mental Note) is forced
    assert observe(g, 1)["self"]["library_known"][:2] == [(0, "Mental Note"), (1, "Ponder")]
    assert observe(g, 0)["opponent"]["library_known"] == []


def test_determinize_preserves_viewer_information():
    g = new_game(DECKS, seed=5)
    advance(g, 120)
    viewer = g.decision.player
    before = observe(g, viewer)
    d = determinize(g, viewer, random.Random(1))
    assert observe(d, viewer) == before
    # the opponent's hidden hand was actually re-sampled in at least one of several tries
    opp = 1 - viewer
    hands = {tuple(sorted(c.name for c in determinize(g, viewer, random.Random(s)).players[opp].hand)) for s in range(10)}
    assert len(hands) > 1
    # and the determinized game keeps playing
    advance(d, 200, seed=2)


def test_features_and_env_roundtrip():
    env = MTGEnv()
    obs = env.reset(seed=4)
    total = 0
    r = random.Random(0)
    done = False
    while not done:
        assert obs["features"] == sorted(set(obs["features"]))
        assert len(obs["action_keys"]) == len(env.legal_actions())
        obs, rewards, done, info = env.step(r.choice(env.legal_actions()))
        total += 1
    assert sum(rewards) == 0 and total > 10


def test_features_do_not_depend_on_hidden_cards():
    g = new_game(DECKS, seed=9)
    advance(g, 80)
    viewer = g.decision.player
    d = determinize(g, viewer, random.Random(3))
    assert state_features(g, viewer) == state_features(d, viewer)
    assert encode_state(g, viewer) == encode_state(d, viewer)


def test_state_features_keep_counts_turn_and_exact_life():
    """Identical objects stay countable after hashing into a set, and turn,
    life and hand size are thermometers (they all collapsed before 2026-10)."""
    one = scenario(p0={"hand": ["Brainstorm"], "battlefield": ["Swamp"], "life": 19})
    three = scenario(p0={"hand": ["Brainstorm", "Brainstorm"], "battlefield": ["Swamp", "Swamp", ("Swamp", {"tapped": True})], "life": 13})
    f1, f3 = set(state_features(one, 0)), set(state_features(three, 0))
    assert {"self:bf_type:Land#3", "self:bf_type:Land:untapped#2", "self:hand:Brainstorm#2"} <= f3
    assert "self:bf_type:Land#2" not in f1 and "self:bf_type:Land:untapped#3" not in f3
    assert "self:life>=18" in f1 and "self:life>=14" not in f3 and "self:life>=12" in f3
    assert "self:hand_count>=2" in f3 and "self:hand_count>=2" not in f1
    assert any(x.startswith("turn>=") for x in f1)


def test_entities_keep_each_permanent_whole():
    """Each permanent is its own entity, so a tapped and an untapped Swamp
    are told apart, and option labels resolve to entity indices."""
    from mtg_ml.encode import entity_features

    g = scenario(p0={"battlefield": ["Swamp", ("Swamp", {"tapped": True})]}, p1={"battlefield": ["Island"]})
    ents, index = entity_features(g, 0)
    swamps = [e for e in ents if "e:name:Swamp" in e]
    assert len(swamps) == 2 and sum("e:tapped" in e for e in swamps) == 1
    assert all("e:ctrl:self" in e for e in swamps) and any("e:ctrl:opponent" in e for e in ents)
    assert sorted(index.values()) == list(range(len(ents)))
