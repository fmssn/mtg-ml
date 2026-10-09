"""Pauper-Research reference matrix: loader validation and interval math (no engine)."""

import copy

import pytest

from mtg_ml.benchmark import reference as ref
from mtg_ml.engine.decks import DECKS


def test_deck_order_matches_engine_decks():
    assert set(ref.DECK_ORDER) == set(DECKS)


def test_wilson_matches_pauper_research_values():
    # Jund Midrange vs Grixis Affinity row of the source matchups.csv (63.5 points in 102 matches).
    lo, hi = ref.wilson_interval(63.5, 102)
    assert lo == pytest.approx(0.5256440474061498, abs=1e-8)
    assert hi == pytest.approx(0.7105582901678075, abs=1e-8)


def test_wilson_edges_and_errors():
    lo, hi = ref.wilson_interval(0, 3)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.5614970356393196, abs=1e-8)
    lo, hi = ref.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and 0.7 < lo < 0.8
    with pytest.raises(ValueError):
        ref.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        ref.wilson_interval(5, 4)


def test_committed_file_loads_and_covers_all_pairs():
    data = ref.load()
    pairs = [k for k, p in data["pairings"].items() if k.split("|")[0] != k.split("|")[1]]
    assert len(pairs) == 15
    assert all(not data["pairings"][k].get("missing") for k in pairs)
    assert data["pairings"]["elves|tron"]["thin"]
    m = ref.matrix(data)
    assert m["tron"]["elves"] == pytest.approx(1 - m["elves"]["tron"])
    assert m["elves"]["elves"] is None


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=99),
    lambda d: d["pairings"].pop("elves|tron"),
    lambda d: d["pairings"]["elves|tron"].update(ci_low=0.0),
    lambda d: d["pairings"]["elves|tron"].update(wins=999),
    lambda d: d["pairings"]["elves|tron"].update(thin=False),
    lambda d: d["source"].pop("commit"),
    lambda d: d.update(decks=list(reversed(d["decks"]))),
])
def test_validate_rejects_corruption(mutate):
    data = copy.deepcopy(ref.load())
    mutate(data)
    with pytest.raises(ValueError):
        ref.validate(data)


def test_build_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "_git", lambda root, *a: "deadbeef")
    src = tmp_path / "pr"
    csv_path = src / ref.SOURCE_CSV
    csv_path.parent.mkdir(parents=True)
    names = ref.ARCHETYPES
    lines = ["archetype,opp_archetype,wins,matches,win_rate,ci_low,ci_high"]
    for a in ref.DECK_ORDER:
        for b in ref.DECK_ORDER:
            if a != b:
                lines.append(f"{names[a]},{names[b]},{3.5 if a < b else 6.5},10,0,0,0")
    csv_path.write_text("\n".join(lines) + "\n")
    data = ref.build(src)
    ref.validate(data)
    assert data == ref.build(src)
    assert data["pairings"]["elves|tron"]["thin"]
    assert data["pairings"]["elves|tron"]["wins"] == 3.5
