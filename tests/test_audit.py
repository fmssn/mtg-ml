"""The indistinguishability audit (`mtg_ml.audit`), on both engines."""

import json

import pytest
from helpers import choose, labels, pass_priority, scenario

from mtg_ml.audit import FORMAT, audit_decision, canonical_view, is_blind_spot, main, option_signatures, rebuild, run, successor_views
from mtg_ml.encode import FEATURE_VERSIONS, FEATURES


def _blocks_against_two_terrors():
    """Blue attacks with two identical Tolarian Terrors; Jund has three
    Krark-Clan Shamans and is at its first block decision."""
    g = scenario(p0={"battlefield": ["Krark-Clan Shaman"] * 3}, p1={"battlefield": ["Tolarian Terror"] * 2}, active=1, step="main1")
    pass_priority(g)
    while g.decision.kind != "declare_blocker":
        if g.decision.kind == "declare_attacker":
            choose(g, "Attack with Tolarian Terror" if any("Attack with" in x for x in labels(g)) else "Done declaring attackers")
        else:
            choose(g, "Pass priority")
    return g


def test_second_blocker_on_same_name_attackers_is_a_blind_spot_in_set_3():
    g = _blocks_against_two_terrors()
    # The first Shaman: the engine merges the two (still identical) Terrors.
    assert len(labels(g)) == 2
    choose(g, "blocks Tolarian Terror")
    # The second Shaman: one Terror is blocked, the other is not, but set 3
    # gives both options the same tokens and bit-identical entities
    # (handoff-2026-10-07-feature-set-4.md, P1).
    a, b = (i for i, x in enumerate(labels(g)) if " blocks " in x)
    assert is_blind_spot(g, a, b, features=3) is True
    [res] = audit_decision(g, features=3)
    assert res.options == [a, b] and res.blind and "battlefield" in res.diff


@pytest.mark.parametrize("features", FEATURE_VERSIONS)
def test_genuinely_equivalent_options_are_not_flagged(features):
    g = _blocks_against_two_terrors()
    choose(g, "blocks Tolarian Terror")
    a, b = (i for i, x in enumerate(labels(g)) if " blocks " in x)
    choose(g, labels(g)[b])  # each Terror now blocked by one Shaman
    # The third Shaman: either Terror gives the same game up to renaming.
    a, b = (i for i, x in enumerate(labels(g)) if " blocks " in x)
    assert is_blind_spot(g, a, b, features=features) in (False, None)
    assert not any(r.blind for r in audit_decision(g, features=features))


def test_canonical_view_ignores_ids_but_not_structure():
    g = _blocks_against_two_terrors()
    assert canonical_view(g, 0) == canonical_view(g.fork(), 0)
    choose(g, "blocks Tolarian Terror")
    a, b = (i for i, x in enumerate(labels(g)) if " blocks " in x)
    va, vb = successor_views(g, [a, b])
    assert va != vb
    # Hidden information stays out: the opponent's hand is only a count.
    assert "hand" not in va["opponent"]


def test_signatures_resolve_entity_pointers():
    g = _blocks_against_two_terrors()
    sigs = option_signatures(g, g.decision.player, features=3)
    none, block = sigs
    assert none != block and block[1], "the block option points at the Shaman and the Terror"


def test_run_report_and_rebuild(tmp_path):
    rep = run(games=2, seed=10, agents="bot,bot")
    assert rep["format"] == FORMAT and rep["audited"] == rep["decisions"] > 0
    assert set(rep["by_kind"]) and rep["meta"]["features"] == [FEATURES] * 2
    for ex in rep["examples"]:
        g = rebuild(ex)
        assert g.decision.kind == ex["kind"] and g.decision.player == ex["player"]
        assert [g.decision.options[o["index"]].label for o in ex["options"]] == [o["label"] for o in ex["options"]]
    out = tmp_path / "audit.json"
    main(["--games", "1", "--seed", "3", "--agents", "random,bot", "--features", "2", "--sample-frac", "0.5", "--quiet", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["format"] == FORMAT and data["meta"]["features"] == [2, 2] and data["audited"] <= data["decisions"]
