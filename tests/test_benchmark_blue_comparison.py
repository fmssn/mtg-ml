"""Paired diagnostic inference must preserve block correlation and failures."""

from dataclasses import asdict

import pytest

from tools.benchmark_blue import development_manifest, paired_summary, strip_timing
from mtg_ml.benchmark.schedule import episodes


def rows(specs, winners):
    return [dict(**{k:v for k,v in asdict(s).items() if k != "decks"}, status="completed", winner=winners(s)) for s in specs]


def test_shared_block_bootstrap_preserves_opposite_cell_effects():
    specs = episodes(development_manifest(2))
    def winner(s):
        good = (s.block == 0) == (s.cell == "blue_vs_blue")
        return s.learner_seat if good else 1-s.learner_seat
    candidate = rows(specs, winner)
    baseline = rows(specs, lambda s: 1-winner(s))
    result = paired_summary(candidate, baseline, specs, replicates=100)
    assert result["equal_weight_mean_difference"] == 0
    assert result["ci95"] == [0, 0]
    assert result["strength_gate_passed"] is False
    assert result["per_cell"]["blue_vs_blue"]["ci95"] == [-1,1]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "seed", "seat", "error"])
def test_comparison_rejects_incomplete_or_mismatched_rows(problem):
    specs = episodes(development_manifest(2))
    candidate = rows(specs, lambda s: s.learner_seat)
    baseline = rows(specs, lambda s: None)
    if problem == "missing": candidate.pop()
    if problem == "duplicate": candidate.append(candidate[0])
    if problem == "seed": candidate[0]["simulator_seed"] += 1
    if problem == "seat": candidate[0]["learner_seat"] = 1-candidate[0]["learner_seat"]
    if problem == "error": candidate[0]["status"] = "error"
    with pytest.raises(ValueError):
        paired_summary(candidate, baseline, specs, replicates=10)


def test_draw_scores_and_single_block_interval():
    specs = episodes(development_manifest(1))
    result = paired_summary(rows(specs, lambda s:s.learner_seat), rows(specs, lambda s:None), specs, replicates=10)
    assert result["equal_weight_mean_difference"] == .5
    assert result["ci95"] is None and not result["strength_gate_passed"]


def test_worker_comparison_excludes_timing_only():
    a = [{"winner": 0, "elapsed_seconds": .1, "latency_seconds": [.01], "attempted_action": {"index": 1}}]
    b = [{"winner": 0, "elapsed_seconds": .9, "latency_seconds": [.05], "attempted_action": {"index": 1}}]
    assert strip_timing(a) == strip_timing(b)
    b[0]["attempted_action"]["index"] = 0
    assert strip_timing(a) != strip_timing(b)


def test_comparison_rejects_invalid_outcome():
    specs = episodes(development_manifest(2))
    candidate = rows(specs, lambda s:s.learner_seat)
    candidate[0]["winner"] = 7
    with pytest.raises(ValueError, match="invalid outcome"):
        paired_summary(candidate, rows(specs, lambda s:None), specs, replicates=10)
