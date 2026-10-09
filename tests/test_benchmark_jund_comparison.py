"""Arithmetic and provenance checks for the standalone development report."""

from copy import deepcopy

from tools.benchmark_jund import schedule, paired_summary, quantile


def records(blocks=3):
    _, specs = schedule(blocks)
    return [dict(pilot=pilot, cell=s.cell, block=s.block, slot=s.slot, mode=s.mode,
                 simulator_seed=s.simulator_seed, actor_seed=s.actor_seed,
                 starting_player=s.starting_player, learner_seat=s.learner_seat,
                 status="completed", winner=s.learner_seat if pilot == "specialist" else 1-s.learner_seat)
            for pilot in ("specialist", "legacy") for s in specs]


def test_four_slots_and_frozen_equal_opponent_panel():
    manifest, specs = schedule(2)
    assert len(specs) == 16
    assert all(len(deck) == 60 for s in specs for deck in s.decks)
    for cell in ("jund_vs_jund", "jund_vs_blue"):
        block = [s for s in specs if s.cell == cell and s.block == 0]
        assert len({s.simulator_seed for s in block}) == 1
        assert {(s.learner_seat, s.starting_player) for s in block} == {(0, 0), (0, 1), (1, 1), (1, 0)}
        assert all(s.deck_ids[s.learner_seat] == "jund_wildfire" for s in block)
    assert manifest["stream"] == "benchmark-v1/dev"


def test_paired_score_exact_extremes_and_draws():
    rows = records()
    result = paired_summary(rows, 3, replicates=100)
    assert result["mean_difference"] == 1 and result["ci95"] == [1, 1]
    assert result["claim"] == "stronger-on-this-panel"
    for r in rows:
        r["winner"] = None
    result = paired_summary(rows, 3, replicates=100)
    assert result["mean_difference"] == 0 and result["ci95"] == [0, 0]
    assert result["claim"] == "inconclusive"


def test_bootstrap_keeps_all_four_slots_of_a_seed_together():
    rows = records()
    for r in rows:
        r["winner"] = r["learner_seat"] if (r["slot"] < 2) == (r["pilot"] == "specialist") else 1-r["learner_seat"]
    # Every block has paired mean zero; resampling individual games would add
    # spurious variance and produce a nonzero interval width.
    result = paired_summary(rows, 3, replicates=100)
    assert result["ci95"] == [0, 0]


def test_missing_duplicate_or_error_row_prevents_strength_claim():
    rows = records()
    cases = [rows[:-1], rows + [rows[0]]]
    errored = deepcopy(rows)
    errored[0]["status"] = "error"
    cases.append(errored)
    for rows in cases:
        result = paired_summary(rows, 3, replicates=100)
        assert result["status"] == "incomplete" and result["claim"] is None


def test_confirmed_cell_regression_blocks_strength_claim():
    rows = records()
    for r in rows:
        if r["cell"] == "jund_vs_blue":
            r["winner"] = 1-r["winner"]
    result = paired_summary(rows, 3, replicates=100)
    assert result["mean_difference"] == 0 and result["claim"] == "regression-on-this-panel"


def test_seed_seat_start_and_mode_must_match_frozen_schedule():
    for field in ("mode", "simulator_seed", "actor_seed", "learner_seat", "starting_player"):
        rows = records()
        rows[0][field] = "changed" if field == "mode" else rows[0][field] + 1
        assert paired_summary(rows, 3, replicates=10)["status"] == "incomplete"


def test_bootstrap_uses_shared_block_indices_across_cells():
    rows = records()
    for r in rows:
        wins = (r["block"] % 2 == 0) == (r["cell"] == "jund_vs_jund")
        r["winner"] = r["learner_seat"] if wins == (r["pilot"] == "specialist") else 1-r["learner_seat"]
    result = paired_summary(rows, 3, replicates=100)
    assert result["ci95"] == [0, 0]


def test_quantiles_interpolate_and_handle_empty():
    assert quantile([], .95) is None
    assert quantile([1, 3], .5) == 2
