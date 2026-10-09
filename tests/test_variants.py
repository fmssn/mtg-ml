"""Deck variants manifest (mtg_ml/engine/variants.py, variants.toml)."""

import os
import random
import sys
import tomllib
from collections import Counter

import pytest

from mtg_ml.engine.cards import CARDS
from mtg_ml.engine.decks import DECKS, SIDEBOARDS
from mtg_ml.engine.sideboard import PLANS, SideboardPlan
from mtg_ml.engine.variants import (
    MANIFEST_PATH, SPLITS, VARIANTS, Variant, assign_splits, parse_manifest, plan_for_variant, sample_variant,
    sampling_prior, stock_plan_valid, variants,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import build_variants  # noqa: E402


def _manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)


def test_stock_decks_are_train_variants():
    for a in DECKS:
        v = VARIANTS[f"{a}:stock"]
        assert v.stock and v.split == "train" and v.archetype == a
        assert dict(v.main) == DECKS[a] and dict(v.sideboard) == SIDEBOARDS[a]
        assert v in variants(a, "train")


def test_every_variant_is_legal_and_implemented():
    for v in VARIANTS.values():
        assert isinstance(v, Variant)
        assert sum(v.main.values()) == 60 and sum(v.sideboard.values()) == 15, v.id
        assert set(v.main) | set(v.sideboard) <= set(CARDS), v.id
        for card, n in v.composition.items():
            assert n <= 4 or "Land" in CARDS[card].types and "Basic" in CARDS[card].supertypes, (v.id, card)
        assert v.weight > 0 and v.split in SPLITS
        assert v.provenance.get("snapshot_commit"), v.id
        assert v.id.startswith(v.archetype + ":")


def test_registered_75_unique_and_duplicates_share_a_split():
    seen = {}
    by_key = {}
    for v in VARIANTS.values():
        assert (v.archetype, v.registered_75) not in seen
        seen[(v.archetype, v.registered_75)] = v.id
        by_key.setdefault((v.archetype, v.split_key), set()).add(v.split)
    assert all(len(s) == 1 for s in by_key.values())
    splits = assign_splits((v.archetype, v.split_key, v.stock) for v in VARIANTS.values())
    assert all(v.split == splits[(v.archetype, v.split_key)] for v in VARIANTS.values())


def test_assign_splits_is_deterministic_and_keeps_duplicates_together():
    keys = [f"{i:064x}" for i in range(10)]
    entries = [("a", k, False) for k in keys] + [("a", keys[3], False), ("a", "f" * 64, True)]
    s1 = assign_splits(entries)
    s2 = assign_splits(list(reversed(entries)))
    assert s1 == s2
    assert s1[("a", "f" * 64)] == "train"
    assert Counter(s1.values()) == Counter({"train": 7, "test": 2, "dev": 2})  # 10 distinct + stock
    assert [s1[("a", k)] for k in keys[:4]] == ["test", "test", "dev", "dev"]  # key order
    # A non-stock duplicate of the stock 75 is train too.
    assert assign_splits([("a", "1" * 64, True), ("a", "1" * 64, False)]) == {("a", "1" * 64): "train"}


def test_sampling_prior_and_sampling():
    for split in SPLITS:
        prior = sampling_prior(split)
        for a, p in prior.items():
            assert abs(sum(p.values()) - 1) < 1e-9
            assert set(p) == {v.id for v in variants(a, split)}
    assert set(sampling_prior("train")) == set(DECKS)
    rng = random.Random(7)
    a = "jund_wildfire"
    draws = Counter(sample_variant(a, rng).id for _ in range(4000))
    prior = sampling_prior("train")[a]
    assert set(draws) <= set(prior)
    top = max(prior, key=prior.get)
    assert abs(draws[top] / 4000 - prior[top]) < 0.03
    assert [sample_variant(a, random.Random(1)).id for _ in range(3)] == [sample_variant(a, random.Random(1)).id] * 3
    for v in (sample_variant(a, rng, split="test") for _ in range(50)):
        assert v.split == "test"


def test_unknown_split_or_empty_split_raises():
    with pytest.raises(ValueError):
        variants("jund_wildfire", "validation")
    empty = [a for a in DECKS if not variants(a, "dev")]
    for a in empty:
        with pytest.raises(ValueError):
            sample_variant(a, random.Random(0), split="dev")


def test_plan_for_every_variant_and_opponent_keeps_the_75():
    for v in VARIANTS.values():
        for opp in DECKS:
            plan = plan_for_variant(v, opp)
            assert isinstance(plan, SideboardPlan)
            after_main, after_side = plan.apply(), plan.side_after()
            assert sum(after_main.values()) == 60 and sum(after_side.values()) == 15
            merged = Counter(after_main) + Counter(after_side)
            assert dict(merged) == v.composition, (v.id, opp)
            stock = PLANS.get((v.archetype, opp))
            if opp in v.plans:
                assert plan.basis == "adjusted" and not stock_plan_valid(v, opp)
            elif stock is not None and stock.swaps:
                assert plan.basis == "stock" and dict(plan.cards_in) == dict(stock.cards_in)
                assert dict(plan.cards_out) == dict(stock.cards_out)


def test_stock_variant_plans_are_the_table():
    for a in DECKS:
        v = VARIANTS[f"{a}:stock"]
        assert not v.plans
        for opp in DECKS:
            stock = PLANS.get((a, opp))
            assert plan_for_variant(v, opp).apply() == (stock.apply() if stock else dict(DECKS[a]))


def test_rejected_candidates_are_recorded_with_reasons():
    m = _manifest()
    rejected = m.get("rejected", [])
    assert rejected
    for r in rejected:
        assert r["reason"] and r["archetype"] in DECKS
        if r["reason"] == "cards not implemented":
            assert r["missing"] and not set(r["missing"]) <= set(CARDS)
    assert m["snapshot"]["commit"] and m["snapshot"]["since"] and m["snapshot"]["until"]


def test_manifest_rejects_bad_rows():
    text = open(MANIFEST_PATH, encoding="utf-8").read()
    with pytest.raises(ValueError, match="cards not implemented|copies|maindeck"):
        parse_manifest(text.replace('"Lembas" = 2', '"Lembas Of Doom" = 2', 1))
    with pytest.raises(ValueError, match="split"):
        parse_manifest(text.replace('id = "red_madness:stock"\narchetype = "red_madness"\nsplit = "train"',
                                    'id = "red_madness:stock"\narchetype = "red_madness"\nsplit = "test"', 1))


def test_aliases_normalise_without_substitution():
    assert build_variants.normalise_list([("Lórien Revealed", 3), ("Lorien Revealed", 1)]) == {"Lorien Revealed": 4}
    assert build_variants.normalise_list([("Spell Pierce", 2)]) == {"Spell Pierce": 2}


def test_adapt_plan_uses_equivalents_and_slot_replacements():
    stock = SideboardPlan("x", "y", {"Red Elemental Blast": 2, "Duress": 1}, {"Lembas": 2, "Toxin Analysis": 1})
    main = {"Lembas": 1, "Gixian Infiltrator": 3, "Swamp": 5}
    stock_main = {"Lembas": 2, "Toxin Analysis": 1, "Gixian Infiltrator": 2, "Swamp": 3}
    side = {"Pyroblast": 2, "Duress": 1}
    cin, cout, note = build_variants.adapt_plan(stock, main, side, stock_main)
    # Pyroblast for REB; Lembas 1 of 2, Toxin missing -> the extra Gixian in its
    # slot (the extra Swamps are lands, never cut); 3 in vs 2 out -> trimmed.
    assert cout == {"Lembas": 1, "Gixian Infiltrator": 1}
    assert cin == {"Pyroblast": 2}
    assert "Pyroblast for Red Elemental Blast" in note and "trimmed to 2" in note
