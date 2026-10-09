import copy
import json

import pytest

from mtg_ml.expert.__main__ import apply_reviews, main, scenario_template
from mtg_ml.expert.artifacts import admissible, evidence, fact, group_split, ref, review, validate_evidence
from mtg_ml.expert.extraction import align, event_metrics, merge_log, parse_vtt
from mtg_ml.expert.interpret import import_response


def test_scrolling_overlap_keeps_repeated_occurrences():
    windows = [{"time": 1, "frame": "f1", "lines": ["A", "B"]},
               {"time": 2, "frame": "f2", "lines": ["B", "B", "C"]},
               {"time": 3, "frame": "f3", "lines": ["B", "C", "D"]}]
    result = merge_log(windows, "video")
    assert [r["value"]["text"] for r in result if r["kind"] == "log"] == ["A", "B", "B", "C", "D"]
    assert all(r["review"]["status"] == "pending" for r in result)


def test_missing_overlap_cut_and_occlusion_are_gaps():
    windows = [{"time": 1, "frame": "f1", "lines": ["A"]},
               {"time": 2, "frame": "f2", "lines": ["B"]},
               {"time": 3, "frame": "f3", "lines": ["B"], "cut": True},
               {"time": 4, "frame": "f4", "lines": [], "occluded": True}]
    result = merge_log(windows, "video")
    assert [r["value"]["reason"] for r in result if r["kind"] == "gap"] == ["missing_log_overlap", "cut", "occluded"]


def test_numeric_changes_are_not_ocr_deduplicated():
    result = merge_log([{"time": 1, "frame": "f1", "lines": ["player loses 1 life from Lightning Bolt"]},
                        {"time": 2, "frame": "f2", "lines": ["player loses 3 life from Lightning Bolt"]}], "video")
    assert len([r for r in result if r["kind"] == "log"]) == 2


def test_rolling_captions_do_not_duplicate_words():
    text = 'WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nwe can cast\n\n00:00:02.000 --> 00:00:04.000\nwe can cast <c>Terror</c>\n\n'
    result = parse_vtt(text, "video", "captions.vtt")
    assert [r["value"]["text"] for r in result] == ["we can cast", "Terror"]
    assert result[1]["refs"][0]["start"] == 2


def test_caption_clip_keeps_original_time():
    text = '00:22:39.000 --> 00:22:41.000\nHello\n\n00:33:00.000 --> 00:33:01.000\nOther round\n'
    result = parse_vtt(text, "video", "captions", 1359, 1930)
    assert len(result) == 1 and result[0]["refs"][0]["start"] == 1359


def test_alignment_uses_cards_and_preserves_beliefs():
    r = fact("narration", "commentary", {"text": "They probably have Counterspell"}, [ref("v", 8, 9, "audio", "speech")])
    actions = [fact("near", "log", {"text": "plays Island"}, [ref("v", 8, 8, "frame", "f1")]),
               fact("related", "log", {"text": "casts Counterspell"}, [ref("v", 12, 12, "frame", "f2")])]
    align([r] + actions, ["Counterspell", "Island"])
    assert r["alignment_candidates"][0]["record_id"] == "related"
    assert r["classification"] == "opponent_prediction" and r["visibility"] == "belief"
    assert not admissible(r, 0, 20)


def test_event_metrics_count_duplicate_errors():
    result = event_metrics(["A", "B"], ["A", "B", "B"])
    assert result["precision"] == 1 and result["recall"] == 2 / 3


def test_review_patch_needs_identity_and_rationale():
    data = evidence({"id": "v"}, [fact("a", "log", {"text": "casts Bolt"}, [ref("v", 1, 1, "frame", "frame")])])
    with pytest.raises(ValueError, match="rationale"):
        apply_reviews(data, [{"id": "a", "review": review("accepted", "reviewer")}])
    result = apply_reviews(data, [{"id": "a", "review": review("accepted", "reviewer", "Verified against frame.")}])
    assert result["records"][0]["review"]["status"] == "accepted"
    assert data["records"][0]["review"]["status"] == "pending"


def test_versions_and_provenance_are_required():
    data = evidence({"id": "v"}, [fact("a", "log", {"text": "A"}, [ref("v", 1, 1, "frame", "frame")])])
    data["version"] = 99
    with pytest.raises(ValueError, match="version"):
        validate_evidence(data)
    data["version"] = 1
    data["records"][0]["refs"][0]["source_id"] = "other"
    with pytest.raises(ValueError, match="unknown source"):
        validate_evidence(data)


def test_template_leaves_unknowns_unfilled():
    data = evidence({"id": "v"})
    spec = scenario_template(data, "v", "scenario", 10, 0)
    assert spec["initial"]["players"][0]["hand"] is None
    assert spec["review"]["status"] == "pending"


def response():
    return {"records": [{"kind": "state", "field": "players.0.hand", "value_json": '["Island"]', "visibility": "self",
                         "classification": "other", "start": 1, "end": 1, "locator": "frame", "uncertainty": ""}]}


def test_model_output_stays_pending_and_cites_supplied_frame():
    records = import_response(response(), "v", 0, [{"frame": "frame", "time": 1}], [], 0, 2)
    assert records[0]["review"]["status"] == "pending"
    assert records[0]["value"] == {"path": "players.0.hand", "value": ["Island"]}
    assert records[0]["refs"][0]["viewer"] == 0


@pytest.mark.parametrize("field,value", [("locator", "invented-frame"), ("start", 0.5), ("end", 2)])
def test_model_cannot_invent_source_references(field, value):
    output = response()
    output["records"][0][field] = value
    with pytest.raises(ValueError):
        import_response(output, "v", 0, [{"frame": "frame", "time": 1}], [], 0, 2)


def test_group_hash_never_splits_variants():
    assert group_split("same-video") == group_split("same-video")
    assert {group_split(str(i)) for i in range(100)} == {"train", "test"}


def test_cli_error_is_actionable_and_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "Wrong"}))
    with pytest.raises(SystemExit) as e:
        main(["align", str(bad), "--out", str(tmp_path / "out.json")])
    assert e.value.code == 2 and "EvidenceGame version" in capsys.readouterr().err


def test_model_predictions_are_never_state_even_when_approved():
    out = response()
    item = out["records"][0]
    item.update(kind="commentary", classification="opponent_prediction", visibility="public", value_json='{"text":"probably Counterspell"}')
    r, = import_response(out, "v", 0, [{"frame": "frame", "time": 1}], [], 0, 2)
    r["review"] = review("accepted", "reviewer", "Verified speech.")
    assert r["visibility"] == "belief" and not admissible(r, 0, 10)


def test_cannot_approve_unknown_patch_fields():
    data = evidence({"id": "v"}, [fact("a", "log", {}, [ref("v", 1, 1, "frame", "f")])])
    with pytest.raises(ValueError, match="unknown review patch"):
        apply_reviews(copy.deepcopy(data), [{"id": "a", "kind": "state"}])
