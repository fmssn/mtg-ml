"""Native `featurize_flat` is a pure function of the game state: its outputs
must not change under speed-ups. Two checks over random games of every
matchup: (1) a digest of all outputs recorded before the optimisations,
(2) set 6/7 features (simulated previews, which copy the game) are equal
whether copies replay decisions from a record or list their options again."""

import hashlib
import random

import pytest

pytest.importorskip("mtg_ml_native")

from mtg_ml.backend import game_class  # noqa: E402
from mtg_ml.match import EXPLICIT_ONLY, MATCHUPS, game_args  # noqa: E402

# sha256 of repr((state, lengths, flat)) for both seats at every decision,
# games 0..N-1 (see `digest`), recorded with the pre-optimisation featurizer
GOLDEN = {
    (7, 12): "183c5cda160b88d51682e9c35c6315a5ed6cff390ac479c3953c74c31aab0e32",
    (6, 12): "f645da96484744b2bdc08dbad28203b5e070faab863fba720dd63b0cc2bb53b2",
    (5, 12): "6618b23cb0a64be151d4e68bf2261cbb10a661e9d900067f1885de253d90008b",
}


def games(n: int):
    pool = sorted(set(MATCHUPS) - EXPLICIT_ONLY)
    for seed in range(n):
        g = game_class("native")(seed=seed, max_turns=30, **game_args(1 + seed // 3 % 2, pool[seed % len(pool)]))
        r = random.Random(seed)
        while not g.over:
            yield g
            g.step(r.randrange(len(g.legal_options())))


def digest(n: int, features: int, replay: bool = True) -> str:
    h = hashlib.sha256()
    for g in games(n):
        g._g.set_decision_replay(replay)
        for p in (g.decision.player, 1 - g.decision.player):
            h.update(repr(g.featurize_flat(p, 4096, 4096, features)).encode())
    return h.hexdigest()


@pytest.mark.parametrize("features,n", sorted(GOLDEN))
def test_features_match_recorded_digest(features, n):
    assert digest(n, features) == GOLDEN[(features, n)]


@pytest.mark.parametrize("features", [6, 7])
def test_decision_replay_equals_relisting(features):
    assert digest(6, features, replay=True) == digest(6, features, replay=False)
