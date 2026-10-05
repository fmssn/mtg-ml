"""Engine backend switch.

    MTG_ENGINE=python   the pure-Python reference engine (default)
    MTG_ENGINE=native   the Rust port (`mtg_ml_native`, built with maturin)

Code that creates games calls `game_class()` (or `new_game(...)`) instead of
importing `Game` directly; an explicit `engine=` argument beats the
environment variable. Both engines expose the same API and play identical
games for identical seeds and actions (see `mtg_ml.difftest`).
"""

from __future__ import annotations

import os

ENGINES = ("python", "native")
ENV_VAR = "MTG_ENGINE"


def engine_name(engine: str | None = None) -> str:
    name = (engine or os.environ.get(ENV_VAR) or "python").lower()
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}: expected one of {ENGINES} (set via engine= or ${ENV_VAR})")
    return name


def game_class(engine: str | None = None):
    if engine_name(engine) == "native":
        from .engine.native import NativeGame

        return NativeGame
    from .engine.game import Game

    return Game


def new_game(decks, engine: str | None = None, **kw):
    return game_class(engine)(decks, **kw)


def native_available() -> bool:
    try:
        import mtg_ml_native  # noqa: F401
    except ImportError:
        return False
    return True
