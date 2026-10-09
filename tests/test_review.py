"""LLM game review: recording, transcripts, seeded faults, counterfactual rollouts.

No network: the LLM step is covered by parsing a canned reply.
"""

from __future__ import annotations

import json
import re

import pytest

from mtg_ml.review import faults as F
from mtg_ml.review.prompt import CAUSES, observation_sheet, parse_findings, system_prompt
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
    assert len(F.score([f], [{"cause": "engine_bug", "decision": d}, {"cause": "engine_bug", "decision": d + 50}])[0]["findings"]) == 2


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


def test_dotenv_reads_the_nearest_env_upwards(tmp_path, monkeypatch):
    from mtg_ml.review.llm import _dotenv

    (tmp_path / ".env").write_text("# keys\nexport FAKE_REVIEW_KEY='abc'\nOTHER=1\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    monkeypatch.delenv("FAKE_REVIEW_KEY", raising=False)
    assert _dotenv("FAKE_REVIEW_KEY") == "abc"
    monkeypatch.setenv("FAKE_REVIEW_KEY", "env")
    assert _dotenv("FAKE_REVIEW_KEY") == "env"


@pytest.mark.parametrize("features", range(1, 7))
def test_observation_sheet_uses_checkpoint_version(features):
    sheet = observation_sheet({"features": features, "trunk": "entity", "entity_attn": 2, "memory": "none"})
    assert [int(v) for v in re.findall(r"^- Feature set (\d+):", sheet, re.M)] == list(range(1, features + 1))
    assert "2 self-attention layers" in sheet
    assert "no recurrent memory" in sheet
    assert ("pv:sim:*" in sheet) == (features == 6)
    assert ("which attacker a blocker blocks" in sheet) == (features >= 4)
    assert ("own hand cards are entities" in sheet) == (features >= 5)


def test_observation_sheet_legacy_config_and_unknown_replay():
    sheet = observation_sheet({})  # pre-versioned PolicyNet defaults
    assert "feature set 1" in sheet and "trunk=mlp, memory=gru, entity_attn=0" in sheet
    assert "- Feature set 2:" not in sheet
    assert "A GRU carries recurrent state" in sheet
    unknown = observation_sheet()
    assert "configuration unknown" in unknown
    assert "not a claim that this checkpoint uses the latest features" in unknown


def test_record_preserves_checkpoint_config_separately_from_description(game, tmp_path, monkeypatch):
    import mtg_ml.review.record as recording

    torch = pytest.importorskip("torch")
    config = {"features": 3, "trunk": "entity", "memory": "none", "entity_attn": 2}
    checkpoint = tmp_path / "policy.pt"
    torch.save({"config": config}, checkpoint)
    monkeypatch.setattr("mtg_ml.play.make_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(recording, "record", lambda *args, **kwargs: json.loads(json.dumps(game)))
    rep = recording.record_game([f"model:{checkpoint}", "bot"], seed=3)
    meta = rep["meta"]["review"]
    assert meta["policy_configs"] == [config, None]
    assert "features 3" in meta["policies"][0] and "entity_attn 2" in meta["policies"][0]
    assert meta["policies"][1] is None


def test_review_backends_use_each_seats_recorded_config(game, tmp_path, monkeypatch):
    from mtg_ml.review import llm
    from mtg_ml.review.__main__ import review_file

    rep = json.loads(json.dumps(game))
    rep["meta"]["review"]["policy_configs"] = [
        {"features": 3, "trunk": "entity", "entity_attn": 0},
        {"features": 6, "trunk": "entity", "entity_attn": 2},
    ]
    rep["meta"]["review"]["policies"] = ["custom human note", "another custom note"]
    path = tmp_path / "game.json"
    path.write_text(json.dumps(rep))
    review_file(path, "prompt", focus="setup")
    exported = path.with_suffix(".prompt.md").read_text()
    captured = []

    def complete(backend, system, text, **kwargs):
        captured.append(text)
        return '{"summary":"checked", "findings":[]}', {}

    monkeypatch.setattr(llm, "complete", complete)
    review_file(path, "claude", focus="setup")
    assert len(captured) == 1 and captured[0] in exported
    p0, p1 = captured[0].split("# POLICY P1", 1)
    assert "feature set 3" in p0 and "- Feature set 4:" not in p0
    assert "feature set 6" in p1 and "2 self-attention layers" in p1
    # The portable reply importer keeps separate reviewer outputs.
    path.with_suffix(".reply-codex.txt").write_text('{"summary":"checked", "findings":[]}')
    result = review_file(path, "file", model="codex")
    assert result["model"] == "codex"
    assert path.with_suffix(".findings-codex.json").exists()


def test_setup_prompt_for_old_recording_is_explicitly_unknown(game, tmp_path):
    from mtg_ml.review.__main__ import review_file

    rep = json.loads(json.dumps(game))
    del rep["meta"]["review"]["policy_configs"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(rep))
    review_file(path, "prompt", focus="setup")
    assert "configuration unknown" in path.with_suffix(".prompt.md").read_text()
