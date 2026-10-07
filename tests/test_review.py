"""LLM game review: recording, transcripts, seeded faults, counterfactual rollouts.

No network: the LLM step is covered by parsing a canned reply.
"""

from __future__ import annotations

import json

import pytest

from mtg_ml.review import faults as F
from mtg_ml.review.prompt import CAUSES, parse_findings, system_prompt
from mtg_ml.review.record import record_game
from mtg_ml.review.transcript import transcript, turn_chunks
from mtg_ml.review.verify import compare, prefixes, verify


@pytest.fixture(scope="module")
def game():
    return record_game(["bot", "bot"], seed=3, engine="python")


def _choices(rep):
    return [i for i, f in enumerate(rep["frames"]) if f["decision"] and len(f["decision"]["options"]) > 1]


def test_transcript_lists_every_offered_option(game):
    text = transcript(game)
    assert "# CARD TEXT" in text and "# DECKLISTS" in text
    i = next(i for i in _choices(game) if len(game["frames"][i]["decision"]["options"]) >= 3)
    for label in game["frames"][i]["decision"]["options"]:
        assert label in text
    assert f"d{i} " in text


def test_turn_chunks_cover_the_game(game):
    chunks = turn_chunks(game, 20_000)
    assert len(chunks) > 1
    assert chunks[0][0] == 0 and chunks[-1][1] == game["frames"][-1]["state"]["turn"]
    assert all(b[0] == a[1] + 1 for a, b in zip(chunks, chunks[1:]))


def test_parse_findings_normalizes():
    reply = 'sure:\n```json\n{"summary": "s", "findings": [{"decision": "d12", "cause": "nonsense", "better_option": "2"}]}\n```'
    out = parse_findings(reply)
    f = out["findings"][0]
    assert (f["decision"], f["cause"], f["better_option"]) == (12, "unclear", 2)
    assert all(c in system_prompt() for c in CAUSES)


def test_mask_fault_removes_options():
    fs = [F.parse("mask:0:Play ")]
    rep = record_game(["bot", "bot"], seed=5, engine="python", wrap=F.wrapper(fs, 5))
    assert fs[0]["decisions"]
    for f in rep["frames"]:
        d = f["decision"]
        if d and d["player"] == 0:
            assert not any("Play " in o for o in d["options"])


def test_life_tamper_and_score(game):
    rep = json.loads(json.dumps(game))
    f = F.parse("life:1:4:-3")
    F.tamper(rep, f)
    d = f["decisions"][0]
    assert rep["frames"][d]["state"]["players"][1]["life"] == game["frames"][d]["state"]["players"][1]["life"] - 3
    hit = F.score([f], [{"cause": "engine_bug", "decision": d + 10}])
    miss = F.score([f], [{"cause": "undertraining", "decision": d}])
    assert hit[0]["caught"] and not miss[0]["caught"]


def test_rebuild_follows_the_record(game):
    ids = _choices(game)
    seen = [i for i, _, _ in prefixes(game, ids)]
    assert seen == ids


def test_compare_pairs_rollouts(game):
    i = next(i for i in _choices(game) if game["frames"][i]["state"]["turn"] >= 3)
    _, g, agents = next(prefixes(game, [i]))
    d = game["frames"][i]["decision"]
    alt = 1 - d["chosen"] if d["chosen"] < 2 else 0
    r = compare(g, agents, d["player"], d["chosen"], alt, n=2)
    assert r["n"] == 2 and r["verdict"] in ("confirmed", "refuted", "inconclusive")
    # same option on both sides: common random numbers give identical results
    same = compare(g, agents, d["player"], d["chosen"], d["chosen"], n=2)
    assert same["diff"] == 0.0


def test_verify_skips_invalid_options(game):
    i = _choices(game)[0]
    out = verify(game, [{"decision": i, "better_option": 99, "cause": "undertraining"}], n=1)
    assert out[0]["rollout"]["verdict"] == "invalid option"
