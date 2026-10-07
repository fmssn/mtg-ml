import random

from helpers import choose, new_game, resolve_stack, scenario

from mtg_ml.encode import encode_state, state_features
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, expand
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


def _preview(g, label):
    from mtg_ml.encode import option_preview

    i = next(i for i, o in enumerate(g.legal_options()) if o.label.startswith(label))
    return set(option_preview(g, g.decision.player, i))


def test_lethal_and_readiness_features():
    """Ready power counts creatures that can attack (not tapped, not sick on
    the active side); evasive power the part no untapped defender can block."""
    g = scenario(
        p0={"battlefield": ["Refurbished Familiar", "Tolarian Terror", ("Gixian Infiltrator", {"sick": True}), ("Writhing Chrysalis", {"tapped": True}), "Swamp", ("Swamp", {"tapped": True})]},
        p1={"battlefield": ["Delver of Secrets", ("Tolarian Terror", {"tapped": True}), "Island"], "life": 7},
    )
    f = set(state_features(g, 0))
    assert {"self:ready_power>=7", "self:ready_evasive_power>=2", "self:lethal_on_board", "self:untapped_mana>=1"} <= f
    assert "self:ready_power>=8" not in f and "self:ready_evasive_power>=3" not in f and "self:evasive_lethal_on_board" not in f
    assert "self:untapped_mana>=2" not in f
    assert "self:potential_blockers>=3" in f and "self:potential_blockers>=4" not in f  # the sick Infiltrator can block
    # the inactive side's tapped Terror untaps for its own turn; Delver alone cannot block the Familiar
    assert {"opponent:ready_power>=6", "opponent:potential_blockers>=1", "opponent:untapped_mana>=1"} <= f
    assert "opponent:potential_blockers>=2" not in f and "opponent:lethal_on_board" not in f


def test_evasive_lethal_without_blockers():
    g = scenario(p0={"battlefield": ["Refurbished Familiar", "Krark-Clan Shaman"]}, p1={"battlefield": [("Delver of Secrets", {"tapped": True})], "life": 3})
    f = set(state_features(g, 0))
    assert {"self:ready_evasive_power>=3", "self:evasive_lethal_on_board", "self:lethal_on_board"} <= f


def test_skip_untap_targets_and_x_are_entity_features():
    from mtg_ml.encode import entity_features

    g = scenario(p1={"hand": ["Sleep of the Dead"], "battlefield": ["Island"]}, p0={"battlefield": ["Refurbished Familiar", "Refurbished Familiar"]}, active=1)
    choose(g, "Cast Sleep of the Dead")  # the two Familiars are one target option; paying is forced
    ents, index = entity_features(g, 0)
    stack = next(e for e in ents if "e:stack" in e)
    assert "e:targets:perm:self" in stack
    targeted = [e for e in ents if "e:targeted_by:opponent" in e]
    assert len(targeted) == 1 and "e:targeted_by:opponent:Sleep of the Dead" in targeted[0]
    resolve_stack(g)
    ents, _ = entity_features(g, 0)
    familiars = [e for e in ents if "e:name:Refurbished Familiar" in e]
    assert sum("e:skip_untap:1" in e for e in familiars) == 1  # the locked one is told apart

    g = scenario(p0={"hand": ["Nyxborn Hydra"], "battlefield": ["Forest", "Forest", "Forest"]})
    choose(g, "Cast Nyxborn Hydra")
    choose(g, "X=2")
    ents, _ = entity_features(g, 1)
    stack = next(e for e in ents if "e:stack" in e)
    assert "e:x>=2" in stack and "e:x>=3" not in stack


def test_known_library_positions():
    g = scenario(p1={"hand": ["Brainstorm", "Ponder"], "battlefield": ["Island"], "library": ["Mental Note"] * 5}, active=1)
    choose(g, "Cast Brainstorm")
    resolve_stack(g)
    choose(g, "Put back Ponder")
    f = set(state_features(g, 1))
    assert {"self:known_library:0:Mental Note", "self:known_library:1:Ponder"} <= f
    assert not any(x.startswith("opponent:known_library:") for x in state_features(g, 0))


def test_option_previews_sweeper_kills_and_deathtouch():
    """Krark-Clan Shaman's activation says what it kills; with deathtouch
    (Toxin Analysis) it also kills the 5/5. Flyers are untouched."""
    g = scenario(
        p0={"hand": ["Toxin Analysis"], "battlefield": ["Krark-Clan Shaman", "Refurbished Familiar", "Ichor Wellspring", "Swamp"]},
        p1={"battlefield": ["Delver of Secrets", "Delver of Secrets", "Tolarian Terror"]},
    )
    pv = _preview(g, "Krark-Clan Shaman: 1 damage")
    assert {"pv:kills_opp>=2", "pv:kills_self>=1", "pv:mana_left_after:1", "pv:colors_left:B"} <= pv
    assert "pv:kills_opp>=3" not in pv and "pv:kills_self>=2" not in pv  # Terror survives, Familiar flies
    cast = _preview(g, "Cast Toxin Analysis")
    assert "pv:mana_left_after:0" in cast and not any(t.startswith("pv:colors_left") for t in cast)
    choose(g, "Cast Toxin Analysis")
    choose(g, "Target Krark-Clan Shaman")
    resolve_stack(g)
    assert {"pv:kills_opp>=3", "pv:kills_self>=1"} <= _preview(g, "Krark-Clan Shaman: 1 damage")


def test_option_previews_sweeper_that_kills_nothing():
    g = scenario(p0={"battlefield": [("Krark-Clan Shaman", {"counters": 1}), "Ichor Wellspring"]}, p1={"battlefield": ["Tolarian Terror"]})
    pv = _preview(g, "Krark-Clan Shaman: 1 damage")
    assert "pv:kills_none" in pv and not any(t.startswith(("pv:kills_opp", "pv:kills_self")) for t in pv)


def test_option_previews_on_targets_ward_and_lethal_damage():
    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Clue", "Clue", "Swamp", "Swamp"]}, p1={"battlefield": ["Delver of Secrets", "Tolarian Terror"], "life": 1})
    choose(g, "Makeshift Munitions: 1 damage")
    assert "pv:damage_lethal_to_target" in _preview(g, "Target Delver of Secrets")
    assert "pv:damage_lethal_to_target" in _preview(g, "Target player 1")
    terror = _preview(g, "Target Tolarian Terror")
    assert "pv:target_ward:2" in terror and "pv:damage_lethal_to_target" not in terror
    assert "pv:ward_payable" not in terror  # {1} for the ability + {2} ward > two Swamps

    g = scenario(p0={"battlefield": ["Makeshift Munitions", "Clue", "Swamp", "Swamp", "Swamp"]}, p1={"battlefield": ["Tolarian Terror"]})
    choose(g, "Makeshift Munitions: 1 damage")
    assert {"pv:target_ward:2", "pv:ward_payable"} <= _preview(g, "Target Tolarian Terror")


def test_feature_set_1_reproduces_the_features_before_set_2():
    """Feature set 1 is byte for byte what main produced before set 2 was
    added (digests recorded with that code, `tests/data/features_v1_digests.json`),
    so checkpoints trained on it see exactly their inputs."""
    import hashlib
    import json
    import os

    from mtg_ml.encode import entity_features
    from mtg_ml.match import match_decks
    from mtg_ml.rl.features import featurize

    with open(os.path.join(os.path.dirname(__file__), "data", "features_v1_digests.json")) as f:
        want = json.load(f)
    for seed in range(0, 24, 3):  # a third of the recorded games keeps it quick
        g = new_game(match_decks(1 + seed % 2), seed=seed, max_turns=30)
        r = random.Random(seed)
        h = hashlib.sha256()
        while not g.over:
            p = g.decision.player
            rec = [featurize(g, p, features=1), state_features(g, 0, features=1), state_features(g, 1, features=1), entity_features(g, p, features=1)[0]]
            h.update(json.dumps(rec).encode())
            g.step(r.randrange(len(g.legal_options())))
        assert h.hexdigest() == want[str(seed)], f"seed {seed}"


def test_feature_set_2_adds_to_set_1():
    from mtg_ml.encode import entity_features, option_preview

    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Ichor Wellspring"]}, p1={"battlefield": ["Delver of Secrets"]})
    f1, f2 = state_features(g, 0, features=1), state_features(g, 0, features=2)
    assert set(f1) < set(f2) and not any("ready_power" in x for x in f1)
    i = next(i for i, o in enumerate(g.legal_options()) if o.label.startswith("Krark-Clan Shaman"))
    assert option_preview(g, 0, i, features=1) == [] and option_preview(g, 0, i, features=2)
    assert entity_features(g, 0, features=1)[1] == entity_features(g, 0, features=2)[1]
