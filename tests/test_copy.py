"""Game.copy(): an exact, independent copy that continues identically.

copy() restores a step-start snapshot and replays the actions since; these
tests hold it to the old replay fork (`fork(replay=True)`): same state, same
features, same log and RNG, at every step of the continuation."""

import random

import pytest
from helpers import new_game, scenario

from mtg_ml.difftest import first_diff, outcome, snapshot
from mtg_ml.engine.view import determinize
from mtg_ml.match import matchup_decks
from mtg_ml.engine.decks import DECKS, expand

MATCHUPS = ("jund_blue", "jund_madness", "blue_madness")


def decks(matchup: str):
    a, b = matchup_decks(matchup)
    return expand(DECKS[a]), expand(DECKS[b])


def same(a, b) -> str | None:
    return first_diff(snapshot(a), snapshot(b)) or first_diff(outcome(a), outcome(b))


def has_snapshot(g) -> bool:
    return g._g.has_snapshot if getattr(g, "NATIVE", False) else g._snap is not None


def lockstep(a, b, rng: random.Random, steps: int) -> None:
    """Step both games with the same random actions, comparing every state."""
    for k in range(steps):
        d = same(a, b)
        assert d is None, f"step {k}: {d}"
        if a.over:
            return
        i = rng.randrange(len(a.legal_options()))
        a.step(i)
        b.step(i)
    assert same(a, b) is None


def play(seed: int, matchup: str, copy_p: float, steps: int, max_decisions: int = 10_000):
    g = new_game(decks(matchup), seed=seed, log=True)
    rng = random.Random(seed)
    n = 0
    while not g.over and n < max_decisions:
        if rng.random() < copy_p:
            lockstep(g.copy(), g.fork(replay=True), random.Random(seed * 1000 + n), steps)
        g.step(rng.randrange(len(g.legal_options())))
        n += 1


@pytest.mark.parametrize("matchup", MATCHUPS)
def test_copy_continues_like_replay(matchup):
    for seed in range(4):
        play(seed, matchup, copy_p=0.1, steps=8)


@pytest.mark.slow
@pytest.mark.parametrize("matchup", MATCHUPS)
def test_copy_continues_like_replay_many(matchup):
    for seed in range(4, 24):
        play(seed, matchup, copy_p=0.1, steps=12)


def test_copy_is_independent():
    g = new_game(decks("jund_blue"), seed=3, log=True)
    rng = random.Random(3)
    for _ in range(60):
        g.step(rng.randrange(len(g.legal_options())))
    before = snapshot(g)
    c = g.copy()
    c2 = g.copy()  # snapshot restore, not the first (replay) copy
    assert has_snapshot(g) and has_snapshot(c2)  # c2 shares g's until its next step
    r = random.Random(4)
    while not c2.over:
        c2.step(r.randrange(len(c2.legal_options())))
    assert first_diff(snapshot(g), before) is None
    # the original, an earlier copy and a replay all still continue alike
    f = g.fork(replay=True)
    seq = random.Random(5)
    while not g.over:
        assert same(g, c) is None and same(g, f) is None
        i = seq.randrange(len(g.legal_options()))
        for x in (g, c, f):
            x.step(i)
    assert same(g, c) is None and same(g, f) is None


def test_copy_of_copy():
    g = new_game(decks("blue_madness"), seed=7, log=True)
    rng = random.Random(7)
    c = g
    for _ in range(5):
        for _ in range(25):
            if c.over:
                break
            c.step(rng.randrange(len(c.legal_options())))
        c = c.copy()
    lockstep(c, c.fork(replay=True), random.Random(8), 20)


def test_fork_defaults_to_copy():
    g = new_game(decks("jund_madness"), seed=11)
    for _ in range(30):
        g.step(0)
    assert not has_snapshot(g)
    f = g.fork()  # first copy replays and turns snapshots on
    assert has_snapshot(g) and has_snapshot(f)
    assert same(g.fork(), g.fork(replay=True)) is None


def test_copy_of_scenario_game():
    """A setup callback and a start step other than untap."""
    g = scenario(
        p0={"hand": ["Lightning Bolt", "Mountain"], "battlefield": ["Mountain", "Kessig Flamebreather"]},
        p1={"battlefield": ["Island", "Delver of Secrets"]},
        step="main1",
    )
    g.copy()
    lockstep(g.copy(), g.fork(replay=True), random.Random(1), 40)


def test_copy_carries_determinization_after_next_step():
    """State edits outside step() drop the snapshot; from the next step on a
    copy carries them (a replay never does)."""
    g = new_game(decks("jund_blue"), seed=5)
    rng = random.Random(5)
    for _ in range(40):
        g.step(rng.randrange(len(g.legal_options())))
    g.copy()
    d = determinize(g, 0, random.Random(1))
    assert not has_snapshot(d)
    d.copy()  # a replay (no snapshot yet), which turns snapshots on for d
    step = d.step_name
    while not d.over and d.step_name == step:
        d.step(0)
    if d.over:
        pytest.skip("game ended")
    assert has_snapshot(d)
    lockstep(d.copy(), d, random.Random(2), 20)
