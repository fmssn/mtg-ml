"""Regression checks for mirror statistics and the fresh overnight campaign."""
from types import SimpleNamespace

import pytest

from mtg_ml.match import EXPLICIT_ONLY, MATCHUPS, parse_matchups
from mtg_ml.rl.evaluate import DECK_KEYS, score
from mtg_ml.rl.rollout import BOT, LEARNER
from tools.overnight_campaign import matrix_mix, paired_summary


def test_uniform_ordered_matrix_includes_each_mirror_once():
    mixture = parse_matchups(matrix_mix())
    assert len(mixture) == 21
    assert sum(w for _, w in mixture) == 36
    cells = {}
    for name, weight in mixture:
        a, b = MATCHUPS[name]
        if a == b:
            cells[a, b] = weight / 36
            assert name in EXPLICIT_ONLY
        else:
            cells[a, b] = cells[b, a] = weight / 72
    assert len(cells) == 36
    assert set(cells.values()) == {1 / 36}


@pytest.mark.parametrize('matchup', sorted(EXPLICIT_ONLY))
def test_mirror_score_never_duplicates_games_or_tightens_interval(matchup):
    games = [((LEARNER, BOT), 0), ((BOT, LEARNER), 0), ((LEARNER, BOT), None)]
    result = score(games, matchup)
    deck = DECK_KEYS[MATCHUPS[matchup][0]]
    assert result['all'] == result[deck]
    assert result['all'][2] == 3
    assert result['all'][0] == .5


def test_cross_deck_counts_stay_distinct():
    result = score([((LEARNER, BOT), 0), ((BOT, LEARNER), 0)])
    assert result['jund'][2] == result['blue'][2] == 1
    assert result['all'][2] == 2


def test_matrix_evaluation_balances_physical_seats_and_starting_players():
    from collections import Counter
    from mtg_ml.rl.evaluate import paired_specs
    for matchup in MATCHUPS:
        specs = paired_specs(BOT, 200, seat=0, matchup=matchup, balance_seats=True)
        assert Counter((s.starting_player, s.swap_seats) for s in specs) == {(0, False): 50, (0, True): 50, (1, False): 50, (1, True): 50}
        assert len({s.seed for s in specs}) == 50


def test_joint_bootstrap_preserves_cross_matchup_seed_correlation():
    # Opposite results on each same seed cancel exactly after weighting.
    rows = [dict(weight=2, seed_scores={'1':0., '2':1.}), dict(weight=2, seed_scores={'1':1., '2':0.})]
    result = paired_summary(rows)
    assert result['score'] == .5 and result['ci'] == [.5, .5]
    assert not result['complete'] and result['seed_blocks'] == 2


def test_frequent_evaluation_can_omit_extra_matchups():
    pytest.importorskip('torch')
    from mtg_ml.rl.train import TrainConfig, Trainer
    trainer = object.__new__(Trainer)
    trainer.cfg = TrainConfig(eval_extra_matchups=0)
    trainer.pool = ['parent.pt']
    trainer.blocks, trainer.ladder = (), ()
    trainer.ladder_ratings = ''
    trainer.matchups = parse_matchups(matrix_mix())
    trainer.exploit_seat, trainer.pool_features = None, {}
    evaluation = SimpleNamespace(policy='policy.pt', version=0)
    assert trainer._eval_args(evaluation, 4, 'local')[-1] == ()
    trainer.cfg.eval_extra_matchups = 1
    assert len(trainer._eval_args(evaluation, 4, 'local')[-1]) == 20


def test_extra_mirror_benchmark_runs_once(monkeypatch):
    from mtg_ml.rl import evaluate
    calls = []
    monkeypatch.setattr(evaluate, 'evaluation_features', lambda *_: 7)
    def bench(*args):
        calls.append((args[-2], args[-1]))
        return {'bench/blue_vs_bot': .5}
    monkeypatch.setattr(evaluate, 'benchmark', bench)
    evaluate.evaluate_policy(None, 'policy', 'pool', 0, 1, 0, 0, 8, 0, 100, 'local', blocks=(), extra_matchups=('blue_mirror',))
    assert calls == [('jund_blue', 0), ('blue_mirror', 0)]
