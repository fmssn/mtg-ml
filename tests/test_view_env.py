import random

from helpers import choose, pay, resolve_stack, scenario

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
    g = Game(DECKS, seed=1)
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
    g = Game(DECKS, seed=5)
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
    g = Game(DECKS, seed=9)
    advance(g, 80)
    viewer = g.decision.player
    d = determinize(g, viewer, random.Random(3))
    assert state_features(g, viewer) == state_features(d, viewer)
    assert encode_state(g, viewer) == encode_state(d, viewer)
