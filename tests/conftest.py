"""Engine parametrization: rules, card, view, bot and fuzz tests run against
both the Python reference engine and the Rust port (`MTG_ENGINE=native`).

Tests that poke engine internals the native wrapper does not expose are
marked `@pytest.mark.python_only`. Without a built `mtg_ml_native` the
native variants are skipped.
"""

import pytest

from mtg_ml.backend import ENV_VAR, native_available

ENGINE_MODULES = {"test_correctness_foundations", "test_set7_rl", "test_action_decomposition", "test_audit", "test_cards_blue", "test_cards_jund", "test_cards_red", "test_rules", "test_sideboard", "test_view_env", "test_bots", "test_fuzz", "test_copy", "test_sim_previews", "test_expert_scenarios"}


def pytest_configure(config):
    config.addinivalue_line("markers", "python_only: test reads or mutates reference-engine internals")


def is_engine_module(name: str) -> bool:
    """Engine-parametrized modules: ENGINE_MODULES plus every card module
    (`test_cards_<deck>.py`), so new decks need no edit here."""
    return name in ENGINE_MODULES or name.startswith("test_cards_")


def pytest_generate_tests(metafunc):
    if is_engine_module(metafunc.module.__name__.rsplit(".", 1)[-1]):
        metafunc.parametrize("engine", ["python", "native"], indirect=True)


@pytest.fixture(autouse=True)
def engine(request, monkeypatch):
    name = getattr(request, "param", "python")
    if name == "native":
        if not native_available():
            pytest.skip("mtg_ml_native is not built (cd native && maturin develop --release)")
        if request.node.get_closest_marker("python_only"):
            pytest.skip("reference-engine internals")
    monkeypatch.setenv(ENV_VAR, name)
    return name
