"""Trust-test runner: comparison rule, deck bias, exploitability aggregation, result schema."""

import pytest

from mtg_ml.benchmark import reference as ref
from mtg_ml.benchmark import trust
from mtg_ml.benchmark.jsonio import read_json


def fake_reference(rates=None, n=100, thin=()):
    """All 15 pairings with the given win rates (default 0.5); intervals are Wilson."""
    pairings = {}
    for key, _, _ in trust.pairings():
        p = (rates or {}).get(key, 0.5)
        m = 5 if key in thin else n
        pairings[key] = ref._row(p * m, m)
    return {"decks": list(ref.DECK_ORDER), "pairings": pairings}


def blocks_for(reference, shift=0.0, noise=0.0, n=40):
    """Per-pairing block values around reference + shift (deterministic wobble)."""
    return {key: [min(1.0, max(0.0, p["win_rate"] + shift + noise * (-1) ** i)) for i in range(n)]
            for key, p in reference["pairings"].items()}


def cell(score, lo, hi):
    return {"score": score, "ci95": [lo, hi], "games": 100}


def test_classify_rule_is_inclusive_point_estimate():
    assert trust.classify(0.5, 0.4, 0.6) == "inside"
    assert trust.classify(0.4, 0.4, 0.6) == "inside" and trust.classify(0.6, 0.4, 0.6) == "inside"
    assert trust.classify(0.39, 0.4, 0.6) == "below" and trust.classify(0.61, 0.4, 0.6) == "above"
    assert trust.overlaps([0.3, 0.45], [0.4, 0.6]) and not trust.overlaps([0.1, 0.3], [0.4, 0.6])


def test_thin_pairings_are_reported_but_not_counted():
    reference = fake_reference(thin={"elves|tron"})
    cells = {k: {"score": 0.5, "ci95": [0.45, 0.55], "games": 8} for k, _, _ in trust.pairings()}
    cells["elves|tron"]["score"] = 0.99
    rows = trust.compare_pairings(cells, reference)
    assert len(rows) == 15
    thin = next(r for r in rows if r["pairing"] == "elves|tron")
    assert thin["status"] == "above" and not thin["counted"]
    check = trust.check_per_pairing(rows, 0.7)
    assert check["counted"] == 14 and check["inside"] == 14 and check["passed"]
    bad = trust.check_per_pairing([dict(r, status="below") for r in rows], 0.7)
    assert not bad["passed"] and bad["below"] == 14


def test_per_deck_bias_flags_only_the_biased_deck():
    reference = fake_reference()
    values = blocks_for(reference, noise=0.02)
    for key in values:  # tron wins 10 points too little in every pairing
        a, b = key.split("|")
        if a == "tron":
            values[key] = [v - 0.10 for v in values[key]]
        elif b == "tron":
            values[key] = [v + 0.10 for v in values[key]]
    biases = trust.deck_biases(values, reference, 0.02, 200, 1)
    assert biases["tron"]["flag"] == "under" and biases["tron"]["bias"] == pytest.approx(-0.10)
    assert biases["tron"]["ci95"][1] < -0.02
    # each opponent gains in only one of its five pairings (+2 points on average)
    assert all(biases[d]["flag"] is None for d in biases if d != "tron")
    check = trust.check_per_deck(biases, 0.02)
    assert not check["passed"] and check["flagged"] == ["tron"]


def test_unbiased_model_passes_and_reference_noise_is_in_the_interval():
    reference = fake_reference(n=30)
    biases = trust.deck_biases(blocks_for(reference, noise=0.05), reference, 0.02, 200, 1)
    assert all(b["flag"] is None and b["status"] == "ok" for b in biases.values())
    assert all(b["se_reference"] > 0 for b in biases.values())
    assert trust.check_per_deck(biases, 0.02)["passed"]


def test_bias_excludes_thin_and_handles_single_block():
    reference = fake_reference(thin={"elves|tron"})
    biases = trust.deck_biases(blocks_for(reference, n=1), reference, 0.02, 50, 1)
    assert "elves|tron" not in biases["tron"]["pairings"] and len(biases["tron"]["pairings"]) == 4
    assert biases["tron"]["status"] == "unavailable"
    assert not trust.check_per_deck(biases, 0.02)["passed"]


def test_exploitability_picks_best_member_and_applies_margin():
    pool = {"old1": {"a>b": cell(0.52, 0.45, 0.59), "b>a": cell(0.40, 0.3, 0.5)},
            "old2": {"a>b": cell(0.48, 0.4, 0.56), "b>a": cell(0.62, 0.54, 0.70)}}
    rows, check = trust.exploitability(pool, 0.02)
    by = {r["matchup"]: r for r in rows}
    assert by["a>b"]["member"] == "old1" and by["b>a"]["member"] == "old2"
    assert not by["a>b"]["clearly_exploitable"] and by["b>a"]["clearly_exploitable"]
    assert not check["passed"] and check["clearly_exploitable"] == ["b>a"]
    assert check["max_pool_win_rate"] == 0.62
    assert trust.exploitability({"old1": pool["old1"]}, 0.02)[1]["passed"]
    assert trust.exploitability(pool, 0.05)[1]["passed"]  # lower bound 0.54 is within 0.5 + 0.05


def test_overall_needs_all_three_checks():
    assert trust.overall({c: {"passed": True} for c in trust.CHECKS}) == {"passed": True, "complete": True}
    failed = {"per_pairing": {"passed": True}, "per_deck": {"passed": False}, "exploitability": {"passed": True}}
    assert trust.overall(failed)["passed"] is False
    skipped = {"per_pairing": {"passed": True}, "per_deck": {"passed": True}, "exploitability": {"passed": None}}
    assert trust.overall(skipped) == {"passed": None, "complete": False}


def test_schedule_covers_matrix_with_paired_deals():
    cells = trust.pairings()
    assert len(cells) == 15 and len(trust.ordered_matchups()) == 30
    specs = trust.episodes(trust.manifest(2, ["greedy"], cells))
    assert len(specs) == 15 * 2 * 4
    block = [s for s in specs if s.cell == "jund_wildfire|tron" and s.block == 1]
    assert len({s.simulator_seed for s in block}) == 1
    assert {(s.learner_seat, s.starting_player) for s in block} == {(0, 0), (0, 1), (1, 1), (1, 0)}
    assert all(s.deck_ids[s.learner_seat] == "jund_wildfire" and s.deck_ids[1 - s.learner_seat] == "tron" for s in block)


@pytest.mark.slow
def test_random_agent_end_to_end_schema(tmp_path):
    out = tmp_path / "out"
    trust.main(["--agent", "random", "--contract", "diagnostic", "--games", "4", "--modes", "greedy", "--primary-mode", "greedy",
                "--engine", "python", "--workers", "1", "--replicates", "20", "--out", str(out)])
    result = read_json(out / "trust.json")
    assert result["format"] == "TrustTestResult" and result["schema_version"] == 1
    assert len(result["code_revision"]) == 40 and result["engine"] == "python" and result["parameters"]["games"] == 4
    greedy = result["modes"]["greedy"]
    assert len(greedy["pairings"]) == 15 and set(greedy["decks"]) == set(ref.DECK_ORDER)
    assert greedy["checks"]["exploitability"]["passed"] is None and greedy["checks"]["overall"]["passed"] is None
    assert len(result["game_rows"]["selfplay"]) == 15 * 4
    text = (out / "trust.md").read_text()
    assert "## Pairings" in text and "NOT RUN" in text
