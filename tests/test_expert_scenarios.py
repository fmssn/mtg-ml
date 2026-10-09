"""Reviewed scenario compilation and knowledge boundaries on both engines."""

import copy

import pytest

from mtg_ml.expert.artifacts import evidence, fact, ref, review
from mtg_ml.expert.learning import demonstrations
from mtg_ml.expert.scenarios import compile_scenario, regression, replay, select, selector_for, verify
from mtg_ml.rl.features import featurize


def fixture():
    players = [dict(life=20, drawn=0, hand=["Lightning Bolt"], hand_count=1, battlefield=["Mountain"], graveyard=[], exile=[], library=["Mountain"] * 10, library_count=10),
               dict(life=3, drawn=0, hand=["Island"], hand_count=1, battlefield=[], graveyard=[], exile=[], library=["Island"] * 10, library_count=10)]
    initial = {"active": 0, "step": "main1", "turn": 8, "lands_played": 1, "spells_cast_this_turn": 0,
               "players": players, "seed": 5, "match_game": 1}
    refs = [ref("fixture", 0, 0, "frame", "synthetic-test-frame", viewer=0)]
    data = evidence({"id": "fixture", "group_id": "fixture", "type": "synthetic_test"})
    bindings, synthetic = {}, {}
    fields = {k: v for k, v in initial.items() if k not in {"players", "seed", "match_game"}}
    fields.update({f"players.{i}.{k}": v for i, player in enumerate(players) for k, v in player.items()})
    for i, (path, value) in enumerate(fields.items()):
        if path.endswith(".library") or path == "players.1.hand":
            synthetic[path] = "Explicit hidden-information test completion; not a real MTGO shuffle."
            continue
        rid = f"state-{i}"
        r = fact(rid, "state", {"path": path, "value": copy.deepcopy(value)}, copy.deepcopy(refs),
                 visibility="self" if path == "players.0.hand" else "public")
        r["review"] = review("accepted", "test", "Synthetic unit-test evidence.")
        data["records"].append(r)
        bindings[path] = [rid]
    spec = {"format": "ExecutableScenario", "version": 1, "id": "bolt-lethal", "group_id": "fixture", "perspective": 0,
            "decision_time": 1, "episode_mode": "reset", "initial": initial, "review": review("accepted", "test", "Synthetic tactical test."),
            "state_facts": bindings, "synthetic_fields": synthetic, "assumptions": ["Synthetic unit-test fixture, not expert footage."],
            "setup_actions": [], "demonstration": [], "preferred": [{"player": 0, "kind": "priority", "key": ["cast", "Lightning Bolt", "hand", "normal"]}]}
    return spec, data


def demonstration_fixture(engine=None):
    spec, data = fixture()
    g, objects = compile_scenario(spec, data, engine)
    chosen = spec["preferred"][0]
    for n in range(20):
        if g.over:
            break
        if n == 0:
            index = select(g, chosen, objects)
        elif g.decision.kind == "target":
            index = next(i for i, o in enumerate(g.legal_options()) if o.key[-2:] == ("player", "opponent"))
        else:
            index = 0
        selector = selector_for(g, index)
        rid = f"action-{n}"
        record = fact(rid, "action", {"selector": selector}, [ref("fixture", 1 + n, 1 + n, "frame", "test-action")], visibility="public")
        record["review"] = review("accepted", "test", "Explicit synthetic test action, not a human label.")
        data["records"].append(record)
        spec["demonstration"].append({"selector": selector, "refs": [rid], "label": g.decision.player == 0})
        g.step(index)
    return spec, data


def test_turn_and_land_history_setup():
    spec, data = fixture()
    spec["initial"]["players"][0]["hand"].append("Mountain")
    spec["initial"]["players"][0]["hand_count"] = 2
    for field in ("hand", "hand_count"):
        path = f"players.0.{field}"
        rid = spec["state_facts"][path][0]
        next(r for r in data["records"] if r["id"] == rid)["value"]["value"] = copy.deepcopy(spec["initial"]["players"][0][field])
    g, _ = compile_scenario(spec, data)
    assert g.turn == 8 and g.lands_played == 1
    assert g.spells_cast_this_turn == 0
    assert not any(o.key[0] == "play_land" for o in g.legal_options())
    # The setup closure is retained by copies/replay forks as on normal games.
    clone = g.copy()
    assert clone.turn == g.turn and clone.lands_played == g.lands_played
    assert featurize(clone, 0) == featurize(g, 0)


def test_compiled_demo_and_regression_continue_legally():
    spec, data = demonstration_fixture()
    episode = demonstrations(spec, data)
    assert episode["episode_mode"] == "reset" and episode["rows"][0]["events"]
    assert any(r["label"] for r in episode["rows"])
    spec["continuation"] = [s["selector"] for s in spec["demonstration"]]
    spec["expected_after"] = {"over": True, "winner": "self", "opponent.life": 0}
    result = regression(spec, data)
    assert result["finish"]["winner"] == "self"


@pytest.mark.parametrize("mutation", ["future", "hindsight", "private", "belief", "wrong_viewer", "pending", "commentary", "contradiction"])
def test_inadmissible_state_is_rejected(mutation):
    spec, data = fixture()
    rid = spec["state_facts"]["players.0.hand"][0]
    r = next(x for x in data["records"] if x["id"] == rid)
    if mutation == "future":
        r["available_at"] = 10
    elif mutation in {"hindsight", "private", "belief"}:
        r["visibility"] = "opponent_private" if mutation == "private" else mutation
    elif mutation == "wrong_viewer":
        r["refs"][0]["viewer"] = 1
    elif mutation == "pending":
        r["review"] = review()
    elif mutation == "commentary":
        r["kind"] = "commentary"
    else:
        r["value"]["value"] = ["Counterspell"]
    with pytest.raises(ValueError):
        compile_scenario(spec, data)


def test_future_reference_cannot_be_backdated():
    spec, data = fixture()
    data["records"][0]["refs"][0]["end"] = 20
    with pytest.raises(ValueError, match="inadmissible"):
        compile_scenario(spec, data)


def test_complete_state_and_review_are_required():
    spec, data = fixture()
    del spec["state_facts"]["players.0.hand"]
    with pytest.raises(ValueError, match="unaccounted"):
        compile_scenario(spec, data)
    spec, data = fixture()
    spec["review"] = review()
    with pytest.raises(ValueError, match="reviewed"):
        compile_scenario(spec, data)


def test_unsupported_cards_are_evidence_only():
    spec, data = fixture()
    spec["initial"]["players"][0]["hand"] = ["Not Implemented Card"]
    with pytest.raises(ValueError, match="unsupported card"):
        compile_scenario(spec, data)


def test_action_requires_reviewed_selector_and_refs():
    spec, data = demonstration_fixture()
    spec["demonstration"][0]["refs"] = []
    with pytest.raises(ValueError, match="references"):
        demonstrations(spec, data)
    spec, data = demonstration_fixture()
    next(r for r in data["records"] if r["kind"] == "action")["value"] = {"text": "Maybe cast Bolt"}
    with pytest.raises(ValueError, match="reviewed engine selector"):
        demonstrations(spec, data)


def test_synthetic_own_hand_cannot_train():
    spec, data = demonstration_fixture()
    del spec["state_facts"]["players.0.hand"]
    spec["synthetic_fields"]["players.0.hand"] = "Invented hand"
    with pytest.raises(ValueError, match="invent"):
        demonstrations(spec, data)


def test_option_key_ambiguity_needs_objects():
    spec, data = fixture()
    spec["initial"]["players"][1]["battlefield"] = [{"name": "Tolarian Terror", "id": "a"}, {"name": "Tolarian Terror", "id": "b", "tapped": True}]
    rid = spec["state_facts"]["players.1.battlefield"][0]
    next(r for r in data["records"] if r["id"] == rid)["value"]["value"] = copy.deepcopy(spec["initial"]["players"][1]["battlefield"])
    g, objects = compile_scenario(spec, data)
    g.step(select(g, spec["preferred"][0], objects))
    targets = [o for o in g.legal_options() if "Tolarian Terror" in o.label]
    assert len(targets) == 2
    selector = {"player": 0, "kind": "target", "key": list(targets[0].key)}
    with pytest.raises(ValueError, match="matches 2"):
        select(g, selector, objects)
    selector["objects"] = ["b"]
    assert "#" + str(objects["b"].oid) in g.legal_options()[select(g, selector, objects)].label


def test_hidden_completion_does_not_change_features():
    spec, data = fixture()
    g, _ = compile_scenario(spec, data)
    spec["initial"]["players"][1]["hand"] = ["Counterspell"]
    spec["initial"]["players"][0]["library"] = ["Lightning Bolt"] * 10
    other, _ = compile_scenario(spec, data)
    assert featurize(g, 0, features=6) == featurize(other, 0, features=6)


def test_group_cannot_separate_video_variants():
    spec, data = fixture()
    spec["group_id"] = "fake-split-group"
    with pytest.raises(ValueError, match="source video group"):
        compile_scenario(spec, data)


def test_replay_is_viewer_compatible_and_hides_synthetic_opponent_hand():
    spec, data = demonstration_fixture()
    result = replay(spec, data)
    assert result["format"] == 1 and result["meta"]["reconstructed"]
    assert result["frames"][0]["state"]["players"][1]["hand"][0]["hidden"]
    assert result["frames"][0]["state"]["players"][1]["hand"][0]["name"] == ""


@pytest.mark.native  # verify() replays on the Rust engine
def test_verification_checks_declared_actions_and_outcome():
    from mtg_ml.backend import native_available

    if not native_available():
        pytest.skip("native engine not built")
    spec, data = demonstration_fixture()
    spec["expected_after"] = {"winner": "self", "opponent.life": 0}
    result = verify(spec, data)
    assert result["identical"] and result["actions_checked"] == len(spec["demonstration"])


def test_cannot_silently_hide_reviewed_opponent_hand_knowledge():
    spec, data = demonstration_fixture()
    path = "players.1.hand"
    del spec["synthetic_fields"][path]
    record = fact("known-hand", "state", {"path": path, "value": ["Island"]},
                  [ref("fixture", 0, 0, "frame", "revealed-hand")], "public")
    record["review"] = review("accepted", "test", "Explicitly revealed opponent hand.")
    data["records"].append(record)
    spec["state_facts"][path] = ["known-hand"]
    with pytest.raises(ValueError, match="hidden-zone knowledge"):
        compile_scenario(spec, data)
