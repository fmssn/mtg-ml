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


def test_feature_set_1_has_no_strings_added_after_it():
    """PR #23's `self:deck:` state feature and the plotted mark on exiled
    card names are set 2 only: set 1 stays what set-1 models learned. The
    view text keeps the mark."""
    from mtg_ml.encode import entity_features
    from mtg_ml.engine.view import observe
    from mtg_ml.match import game_args

    seen = set()
    for seed in (0, 3):  # games where Red plots a card
        g = new_game(seed=seed, max_turns=30, **game_args(1, "jund_madness"))
        r = random.Random(seed)
        while not g.over:
            for v in (0, 1):
                v1 = state_features(g, v, features=1) + [t for e in entity_features(g, v, features=1)[0] for t in e]
                assert not any(x.startswith("self:deck:") or "(plotted)" in x for x in v1)
                seen |= {k for x in state_features(g, v, features=2) for k in ("self:deck:red_madness", "(plotted)") if k in x}
                seen |= {"view" for x in observe(g, v)["self"]["exile"] if x.endswith(" (plotted)")}
            g.step(r.randrange(len(g.legal_options())))
    assert seen == {"self:deck:red_madness", "(plotted)", "view"}


def test_feature_set_2_adds_to_set_1():
    from mtg_ml.encode import entity_features, option_preview

    g = scenario(p0={"battlefield": ["Krark-Clan Shaman", "Ichor Wellspring"]}, p1={"battlefield": ["Delver of Secrets"]})
    f1, f2 = state_features(g, 0, features=1), state_features(g, 0, features=2)
    assert set(f1) < set(f2) and not any("ready_power" in x for x in f1)
    i = next(i for i, o in enumerate(g.legal_options()) if o.label.startswith("Krark-Clan Shaman"))
    assert option_preview(g, 0, i, features=1) == [] and option_preview(g, 0, i, features=2)
    assert entity_features(g, 0, features=1)[1] == entity_features(g, 0, features=2)[1]


def _fixtures() -> dict:
    """{(seed, decision): game at that decision} for tests/data/feature_fixtures.json:
    decisions from the r4-control game review (docs/features.md, set 4),
    rebuilt by stepping the recorded actions."""
    import json
    import os

    from mtg_ml.match import game_args

    with open(os.path.join(os.path.dirname(__file__), "data", "feature_fixtures.json")) as f:
        out = {}
        for x in json.load(f):
            g = new_game(**game_args(x["match_game"], x["matchup"]), seed=x["seed"], starting_player=x["starting_player"])
            for a in x["actions"]:
                g.step(a)
            assert [o.label for o in g.legal_options()] == x["options"], (x["seed"], x["decision"])
            out[x["seed"], x["decision"]] = g
    return out


def _option_signature(g, label: str, features: int):
    """What the network sees of one option: its hashed tokens and the hashed
    token set of every entity it points at (`featurize`)."""
    from mtg_ml.rl.features import OPTION_DIM, STATE_DIM, featurize

    state, opts = featurize(g, g.decision.player, features=features)
    ents, cur = [], None
    for t in state:
        if t == STATE_DIM:
            cur = set()
            ents.append(cur)
        elif cur is not None:
            cur.add(t)
    i = [o.label for o in g.legal_options()].index(label)
    return frozenset(t for t in opts[i] if t < OPTION_DIM), sorted(sorted(ents[t - OPTION_DIM]) for t in opts[i] if t >= OPTION_DIM)


# Block decisions where the policy split 0.50 / 0.50 between two same-name
# attackers (one already blocked), and s1010 d359, a second 3-power blocker
# put on a 6/7 that it cannot kill.
BLOCK_FIXTURES = {
    (1000, 153): ("Nyxborn Hydra#218 blocks Tolarian Terror#166", "Nyxborn Hydra#218 blocks Tolarian Terror#180"),
    (1035, 102): ("Eldrazi Spawn#268 blocks Tolarian Terror#247", "Eldrazi Spawn#268 blocks Tolarian Terror#249"),
    (1047, 229): ("Eldrazi Spawn#253 blocks Cryptic Serpent#227", "Eldrazi Spawn#253 blocks Cryptic Serpent#234"),
    (1047, 279): ("Eldrazi Spawn#272 blocks Cryptic Serpent#227", "Eldrazi Spawn#272 blocks Cryptic Serpent#234"),
    (1010, 359): ("Insectile Aberration#259 blocks Writhing Chrysalis#218", "Insectile Aberration#259 blocks Refurbished Familiar#220"),
}


def _pv(g, label: str, features: int = 4) -> set[str]:
    from mtg_ml.encode import option_preview

    labels = [o.label for o in g.legal_options()]
    return set(option_preview(g, g.decision.player, labels.index(label), features))


def test_feature_set_4_tells_block_options_apart():
    games = _fixtures()
    for (seed, d), (blocked, other) in BLOCK_FIXTURES.items():
        g = games[seed, d]
        if seed != 1010:  # same-name attackers: identical inputs in set 3 (the gap set 4 closes)
            assert _option_signature(g, blocked, 3) == _option_signature(g, other, 3), (seed, d)
        assert _option_signature(g, blocked, 4) != _option_signature(g, other, 4), (seed, d)
        assert "pv:attacker_already_blocked" in _pv(g, blocked) and "pv:attacker_already_blocked" not in _pv(g, other)
    # s1010 d359: 3 + 3 power does not kill the 6/7 Chrysalis; blocking the 2/2 Familiar kills it
    g = games[1010, 359]
    assert "pv:attacker_dies" not in _pv(g, BLOCK_FIXTURES[1010, 359][0]) and "pv:attacker_dies" in _pv(g, BLOCK_FIXTURES[1010, 359][1])
    # s1000 d153 at 5 life: stacking the Hydra on the blocked Terror leaves 8 unblocked, lethal
    g = games[1000, 153]
    assert "pv:lethal_left" in _pv(g, BLOCK_FIXTURES[1000, 153][0]) and "pv:lethal_left" not in _pv(g, BLOCK_FIXTURES[1000, 153][1])


def test_feature_set_4_incoming_damage():
    from mtg_ml.encode import entity_features

    games = _fixtures()
    for key in ((1000, 153), (1049, 306)):
        g = games[key]
        f3, f4 = state_features(g, g.decision.player, 3), state_features(g, g.decision.player, 4)
        assert "opponent:incoming_lethal" in f4 and set(f3) < set(f4), key
        e3, e4 = entity_features(g, g.decision.player, 3), entity_features(g, g.decision.player, 4)
        assert e3[1] == e4[1] and all(set(a) <= set(b) for a, b in zip(e3[0], e4[0]))
    # s1011 d216 (cited as lethal in the review): 12 + 4 unblocked against 18 life
    # is not lethal, but it leaves 2 life, and set 4 says so
    f4 = state_features(games[1011, 216], 1, 4)
    assert {"opponent:attacking_power>=15", "opponent:unblocked_power>=15", "self:life_after_unblocked>=2"} <= set(f4)
    assert "opponent:incoming_lethal" not in f4 and "self:life_after_unblocked>=3" not in f4


def test_feature_set_4_x_and_colour_previews():
    games = _fixtures()
    # Nyxborn Hydra, X = 0..7 with 8 sources: an ordered X, the largest X, the mana left and the Hydra's size
    g = games[1003, 250]
    assert "pv:mana_left_after:7" in _pv(g, "X=0") and not any(t.startswith("pv:x") for t in _pv(g, "X=0"))
    assert {"pv:x>=7", "pv:x_is_max", "pv:mana_left_after:0", "pv:enters_power>=7"} <= _pv(g, "X=7")
    assert _pv(g, "X=7", 3) == set()
    # every X option points at the Hydra on the stack (set 4 only)
    assert _option_signature(g, "X=3", 3)[1] == [] and len(_option_signature(g, "X=3", 4)[1]) == 1
    # paying a generic with the only Swamp strands black; a Forest keeps B and R
    g = games[1008, 333]
    assert "pv:colors_left:B" not in _pv(g, "Tap Swamp#191 for B")
    assert {"pv:colors_left:B", "pv:colors_left:R"} <= _pv(g, "Tap Forest#208 for G")
    # basic-land searches: the colour each basic adds, and the one the hand misses
    g = games[1033, 60]
    assert "self:hand_missing:G" in state_features(g, 0, 4)
    assert _pv(g, "Find Forest") == {"pv:adds_color:G", "pv:adds_missing_color"} and _pv(g, "Find Swamp") == {"pv:adds_color:B"}
    assert _pv(games[1048, 71], "Find Mountain") == {"pv:adds_color:R", "pv:adds_missing_color"}


# (matchup, game number, seed) of the games behind tests/data/features_v3_digests.json.
V3_DIGEST_GAMES = [(m, n, s) for s, (m, n) in enumerate((m, n) for m in ("jund_blue", "jund_madness", "blue_madness") for n in (1, 2) for _ in range(3))]


def features_digest(matchup: str, game_no: int, seed: int, features: int) -> str:
    """sha256 over every featurizer output of a random-play game: featurize,
    both seats' state features, entities with their index, and each option's
    preview. `python tests/test_view_env.py` re-records the file."""
    import hashlib
    import json

    from mtg_ml.encode import entity_features, option_preview
    from mtg_ml.match import game_args
    from mtg_ml.rl.features import featurize

    g = new_game(seed=seed, max_turns=30, **game_args(game_no, matchup))
    r = random.Random(seed)
    h = hashlib.sha256()
    while not g.over:
        p = g.decision.player
        ents, index = entity_features(g, p, features)
        previews = [option_preview(g, p, i, features) for i in range(len(g.legal_options()))]
        rec = [featurize(g, p, features=features), state_features(g, 0, features), state_features(g, 1, features), ents, sorted(index.items()), previews]
        h.update(json.dumps(rec).encode())
        g.step(r.randrange(len(g.legal_options())))
    return h.hexdigest()


def test_feature_sets_2_and_3_are_unchanged():
    """Sets 2 and 3 are byte for byte what they were before set 4 was added
    (digests recorded with that code, `tests/data/features_v3_digests.json`),
    across all three matchups, pre- and postboard."""
    import json
    import os

    with open(os.path.join(os.path.dirname(__file__), "data", "features_v3_digests.json")) as f:
        want = json.load(f)
    for m, n, s in V3_DIGEST_GAMES:
        for features in (2, 3):
            assert features_digest(m, n, s, features) == want[f"{m}:{n}:{s}:{features}"], (m, n, s, features)


if __name__ == "__main__":  # re-record tests/data/features_v3_digests.json (only with the pre-set-4 code)
    import json
    import os

    out = {f"{m}:{n}:{s}:{f}": features_digest(m, n, s, f) for m, n, s in V3_DIGEST_GAMES for f in (2, 3)}
    with open(os.path.join(os.path.dirname(__file__), "data", "features_v3_digests.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
