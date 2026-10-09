"""Engine parametrization: rules, card, view, bot and fuzz tests run against
both the Python reference engine and the Rust port (`MTG_ENGINE=native`).

Tests that poke engine internals the native wrapper does not expose are
marked `@pytest.mark.python_only`. Without a built `mtg_ml_native` the
native variants are skipped.
"""

import os
import zlib

import pytest

from mtg_ml.backend import ENV_VAR, native_available

ENGINE_MODULES = {"test_benchmark_engine", "test_benchmark_jund", "test_correctness_foundations", "test_set7_rl", "test_action_decomposition", "test_audit", "test_cards_blue", "test_cards_jund", "test_cards_red", "test_rules", "test_sideboard", "test_view_env", "test_bots", "test_fuzz", "test_copy", "test_sim_previews", "test_expert_scenarios", "test_expert_reconstruction"}


def pytest_configure(config):
    config.addinivalue_line("markers", "python_only: test reads or mutates reference-engine internals")
    config.addinivalue_line("markers", "native: needs the Rust engine (applied automatically, see below)")


def pytest_collection_modifyitems(config, items):
    # CI's native job runs only `-m native`; the python job covers the rest. A test
    # that needs mtg_ml_native must have "native" in its id, live in test_difftest,
    # or carry @pytest.mark.native itself.
    for item in items:
        if "native" in item.nodeid.rsplit("::", 1)[-1] or item.module.__name__.endswith("test_difftest"):
            item.add_marker(pytest.mark.native)
    # PYTEST_SHARD=i/n keeps a stable 1/n of the tests, so CI can run the suite as n jobs.
    if shard := os.environ.get("PYTEST_SHARD"):
        i, n = map(int, shard.split("/"))
        keep, drop = [], []
        for item in items:
            (keep if zlib.crc32(item.nodeid.encode()) % n == i else drop).append(item)
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


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
