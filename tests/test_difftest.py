"""Differential tests: the Rust port plays exactly the reference engine's
games. The long version is `python -m mtg_ml.difftest fuzz --games 5000`."""

import random

import pytest

from mtg_ml.backend import game_class, native_available
from mtg_ml.difftest import Divergence, first_diff, run_lockstep, snapshot
from mtg_ml.match import match_decks
from mtg_ml.trace import Scenario, new_game, scenarios

pytestmark = pytest.mark.skipif(not native_available(), reason="mtg_ml_native is not built")


@pytest.mark.parametrize("sc", scenarios(36, start=10_000), ids=lambda sc: f"seed{sc.seed}-{'-'.join(sc.agents)}-g{sc.match_game}")
def test_lockstep(sc):
    d = run_lockstep(sc, fork_every=41)
    assert d is None, str(d)


@pytest.mark.parametrize("sc", scenarios(14, start=20_000, auto_mana=True, auto_pass=True), ids=lambda sc: f"seed{sc.seed}-{'-'.join(sc.agents)}-g{sc.match_game}")
def test_lockstep_action_decomposition(sc):
    # Game(auto_mana=True, auto_pass=True): the auto payer and the collapsed passes match too
    d = run_lockstep(sc, fork_every=41)
    assert d is None, str(d)


def test_lockstep_without_auto_single():
    # every forced choice becomes a decision: exercises option lists of length 1
    for seed in range(4):
        py, nat = (game_class(e)(match_decks(1 + seed % 2), seed=seed, auto_single=False, log=True, max_turns=20) for e in ("python", "native"))
        r = random.Random(seed)
        while not py.over:
            assert first_diff(snapshot(py), snapshot(nat)) is None
            a = r.randrange(len(py.legal_options()))
            py.step(a)
            nat.step(a)
        assert py.log == nat.log and (py.winner, py.end_reason) == (nat.winner, nat.end_reason)


def test_harness_reports_divergence():
    sc = Scenario(seed=5, agents=("chaos", "bot"))
    py, nat = new_game(sc, "python"), new_game(Scenario(seed=6, agents=sc.agents), "native")
    assert first_diff(snapshot(py), snapshot(nat)) is not None
    assert "seed 5" in str(Divergence(sc, 3, ".x: python=1 native=2"))


def test_rng_is_cpython_random():
    import mtg_ml_native as n

    from mtg_ml.engine.native import NativeGame

    for seed in (0, 1, 7, -3, 2**31, 2**40 + 5, 123456789012345):
        g = NativeGame(([], []), seed=seed, starting_player=0, setup=lambda g: None)
        assert g.rng.getstate() == random.Random(seed).getstate()
    assert n.loaded_spec()  # the spec the engine runs on


def test_native_runs_the_spec_python_loaded():
    import mtg_ml_native as n

    from mtg_ml.engine.cards import SPEC_PATH

    with open(SPEC_PATH, encoding="utf-8") as f:
        assert n.loaded_spec() == f.read()


def test_search_bot_identical_on_both_engines():
    """SearchBot exercises fork(), determinize() and reseeding the engine RNG."""
    from mtg_ml.bots.search import SearchBot

    picks = {}
    for engine in ("python", "native"):
        g = game_class(engine)(match_decks(1), seed=11)
        bot = SearchBot(0, playouts=2, seed=1)
        out = []
        while not g.over and len(out) < 25:
            a = bot.act(g) if g.decision.player == 0 else 0
            out.append(a)
            g.step(a)
        picks[engine] = out
    assert picks["python"] == picks["native"]


# Set-4 strings that `test_entity_and_preview_strings_identical` must see at
# least once, so the comparison covers every new code path.
SET4_TOKENS = (
    "attacking_power>=",
    "unblocked_power>=",
    ":incoming_lethal",
    ":life_after_unblocked>=",
    "self:sources:",
    "self:hand_needs:",
    "self:hand_missing:",
    "e:produces:",
    "e:attack_slot:",
    "e:blocked",
    "e:unblocked",
    "e:blockers>=",
    "e:block_power>=",
    "e:block_lethal",
    "e:blocking:slot:",
    "e:blocking:name:",
    "e:blocking:power>=",
    "e:blocking:kills",
    "e:blocking:dies",
    "pv:attacker_already_blocked",
    "pv:attacker_dies",
    "pv:blocker_dies",
    "pv:unblocked_damage_left>=",
    "pv:lethal_left",
    "pv:x>=",
    "pv:x_is_max",
    "pv:enters_power>=",
    "pv:adds_color:",
    "pv:adds_missing_color",
    "pv:colors_left:",
)
# Set 6: simulated previews, the stop reasons (game_over: tests/test_sim_previews.py) and the common deltas.
SET6_TOKENS = (
    "pv:sim:skipped",
    "pv:sim:stop:own_decision",
    "pv:sim:stop:opponent_decision",
    "pv:sim:stop:hidden_info",
    "pv:sim:next:self:",
    "pv:sim:next:opponent:",
    "pv:sim:new_turn",
    "pv:sim:step:",
    "pv:sim:self:life-",
    "pv:sim:opponent:life-",
    "pv:sim:self:creatures_lost>=",
    "pv:sim:opponent:creatures_lost>=",
    "lost_power_tier:",
    "pv:sim:self:perms_gained>=",
    "pv:sim:self:tapped>=",
    "pv:sim:opponent:untapped>=",
    "pv:sim:self:hand-",
    "pv:sim:self:graveyard+",
    "pv:sim:self:exile+",
    "pv:sim:stack+",
    "pv:sim:stack-",
    "pv:sim:mana_left>=",
    "pv:sim:color:",
    "pv:sim:gained:",
    "pv:sim:lost:",
    "pv:simp:skipped",
    "pv:simp:stop:own_decision",
    "pv:simp:stop:opponent_decision",
    "pv:simp:stop:hidden_info",
    "pv:simp:stack-",
    "pv:simp:self:creatures_gained>=",
)


def test_entity_and_preview_strings_identical():
    """`featurize` hashes are compared in lockstep; this compares the strings
    behind them (state, entities, option previews) so a mismatch is
    readable, in sets 3 and up, over every matchup. Entities of both
    seats (set 5 adds the viewer's own hand), the ids options point at, and
    the simulated previews of set 6 (`option_previews` too)."""
    from mtg_ml.encode import FEATURE_VERSIONS, entity_features, option_object_ids, option_preview, option_previews, state_features
    from mtg_ml.match import EXPLICIT_ONLY, MATCHUPS, game_args

    first = ("blue_madness", "jund_blue", "jund_madness")  # the token coverage below was found on these (15 games since the fidelity sideboards changed game 2)
    later = sorted(set(MATCHUPS) - set(first) - EXPLICIT_ONLY)
    runs = [(seed, first[seed % 3]) for seed in range(15)] + [(15 + i, later[i % len(later)]) for i in range(2 * len(later))]
    runs += [(100 + i, m) for m in sorted(EXPLICIT_ONLY) for i in range(2)]  # appended, so the runs above are unchanged
    seen = set()
    for seed, matchup in runs:
        args = game_args(1 + seed // 3 % 2, matchup)
        py, nat = (game_class(e)(seed=seed, max_turns=30, **args) for e in ("python", "native"))
        r = random.Random(seed)
        while not py.over:
            p = py.decision.player
            for f in FEATURE_VERSIONS[2:]:
                for v in (0, 1):
                    assert state_features(py, v, f) == state_features(nat, v, f)
                    assert entity_features(py, v, f) == entity_features(nat, v, f), (v, f)
                for i, (o, no) in enumerate(zip(py.legal_options(), nat.legal_options())):
                    assert option_preview(py, p, i, f) == option_preview(nat, p, i, f), o.label
                    assert option_object_ids(o, py, py.decision.kind, f) == option_object_ids(no, nat, nat.decision.kind, f), o.label
                assert option_previews(py, p, f) == option_previews(nat, p, f) == [option_preview(py, p, i, f) for i in range(len(py.legal_options()))]
            strings = state_features(py, p) + [t for e in entity_features(py, p)[0] for t in e]
            strings += [t for i in range(len(py.legal_options())) for t in option_preview(py, p, i)]
            seen |= {k for k in SET4_TOKENS + SET6_TOKENS if any(k in t for t in strings)}
            a = r.randrange(len(py.legal_options()))
            py.step(a)
            nat.step(a)
    assert seen == set(SET4_TOKENS + SET6_TOKENS), sorted(set(SET4_TOKENS + SET6_TOKENS) - seen)


def test_spec_field_sets_identical():
    """Both engines classify the same card-spec fields as shape / non-shape."""
    import mtg_ml_native

    from mtg_ml.engine import cards

    want = {lvl: (getattr(cards, f"SHAPE_{lvl.upper()}_FIELDS"), getattr(cards, f"NON_SHAPE_{lvl.upper()}_FIELDS")) for lvl in ("card", "ability", "trigger")}
    assert {k: (frozenset(a), frozenset(b)) for k, (a, b) in mtg_ml_native.spec_fields().items()} == want


def test_card_shapes_identical():
    """Both engines derive the same shape tokens from cards.toml (set 5)."""
    import mtg_ml_native

    from mtg_ml.engine.cards import CARDS, FACES, TOKENS

    assert mtg_ml_native.card_shapes() == {name: list(d.shape) for reg in (CARDS, FACES, TOKENS) for name, d in reg.items()}


def test_divergence_json_keeps_fork_every():
    d = Divergence(Scenario(seed=5, agents=("bot", "bot")), 3, ".x", [1, 0, 2], fork_every=41)
    assert Divergence.from_json(d.to_json()) == d


def _flaky_bot(monkeypatch):
    """Seat 0's bot picks differently on the native engine from step 4 on."""
    import mtg_ml.difftest as dt

    real = dt.make_agents

    class Flaky:
        def __init__(self, bot):
            self.bot, self.calls = bot, 0

        def act(self, g):
            a = self.bot.act(g)
            if getattr(g, "NATIVE", False) and len(g.actions) >= 4 and len(g.legal_options()) > 1:
                return (a + 1) % len(g.legal_options())
            return a

    monkeypatch.setattr(dt, "make_agents", lambda sc: [Flaky(a) if k == "bot" else a for a, k in zip(real(sc), sc.agents)])


def test_bot_choice_divergence_reproduces_from_its_script(monkeypatch):
    from mtg_ml.difftest import minimize

    _flaky_bot(monkeypatch)
    sc = Scenario(seed=3, agents=("bot", "random"))
    d = run_lockstep(sc)
    assert d is not None and d.diff.startswith("bot choice")
    again = run_lockstep(sc, script=d.actions)  # the script ends where the mismatch is: still found
    assert again is not None and again.diff.startswith("bot choice")
    assert minimize(d).diff.startswith("bot choice")


def test_identical_errors_in_both_engines_are_reported(monkeypatch):
    import mtg_ml.difftest as dt

    real = dt._new_pair

    # Patched on the classes for these two games only: an instance attribute
    # would be carried into the games' copies (set 6 simulates on copies),
    # whose steps would then step the original.
    targets = set()

    def pair(sc):
        games = real(sc)
        for g in games:
            targets.add(id(g))
            cls = type(g)
            if "_unpatched_step" not in cls.__dict__:
                step = cls.step

                def boom(self, a, step=step):
                    if id(self) in targets and len(self.actions) >= 5:
                        raise ValueError("reference bug")
                    return step(self, a)

                monkeypatch.setattr(cls, "_unpatched_step", step, raising=False)
                monkeypatch.setattr(cls, "step", boom)
        return games

    monkeypatch.setattr(dt, "_new_pair", pair)
    d = run_lockstep(Scenario(seed=1, agents=("random", "random")))
    assert d is not None and d.diff.startswith("both raised") and "ValueError: reference bug" in d.diff
