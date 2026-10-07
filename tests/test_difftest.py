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


def test_entity_and_preview_strings_identical():
    """`featurize` hashes are compared in lockstep; this compares the strings
    behind them (entities, option previews) so a mismatch is readable."""
    from mtg_ml.encode import entity_features, option_preview

    for seed in range(6):
        py, nat = (game_class(e)(match_decks(1 + seed % 2), seed=seed, max_turns=30) for e in ("python", "native"))
        r = random.Random(seed)
        while not py.over:
            p = py.decision.player
            assert entity_features(py, p) == entity_features(nat, p)
            for i in range(len(py.legal_options())):
                assert option_preview(py, p, i) == option_preview(nat, p, i), py.legal_options()[i].label
            a = r.randrange(len(py.legal_options()))
            py.step(a)
            nat.step(a)


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

    def pair(sc):
        games = real(sc)
        for g in games:
            step = g.step

            def boom(a, g=g, step=step):
                if len(g.actions) >= 5:
                    raise ValueError("reference bug")
                return step(a)

            g.step = boom
        return games

    monkeypatch.setattr(dt, "_new_pair", pair)
    d = run_lockstep(Scenario(seed=1, agents=("random", "random")))
    assert d is not None and d.diff.startswith("both raised") and "ValueError: reference bug" in d.diff
