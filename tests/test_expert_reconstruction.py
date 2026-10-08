import json
from pathlib import Path

import pytest

from mtg_ml.expert.artifacts import fact, ref, review
from mtg_ml.expert.log_events import normalize_events, parse_event
from mtg_ml.expert import reconstruction as R
from mtg_ml.expert.reconstruction import checkpoint_template, completions, reconstruct

ROOT = Path(__file__).parent / "data" / "expert"


@pytest.fixture
def native_parity():
    from mtg_ml.backend import native_available

    if not native_available():
        pytest.skip("retained reconstruction paths require Python/native parity")


def fixture():
    evidence = json.loads((ROOT / "bolt-lethal.evidence.json").read_text())
    scenario = json.loads((ROOT / "bolt-lethal.scenario.json").read_text())
    source = evidence["sources"][0]["id"]
    record = fact(
        "observed-bolt",
        "log",
        {"text": "10:41 AM: Alice casts Lightning Bolt targeting Bob."},
        [ref(source, 100, 100, "frame", "bolt.png")],
        visibility="public",
    )
    evidence["records"].append(record)
    spec = {
        "format": "ReconstructionSpec",
        "version": 1,
        "source_id": source,
        "perspective": 0,
        "players": {"Alice": 0, "Bob": 1},
        "games": [
            {
                "id": "g1",
                "start": 0,
                "end": 200,
                "starting_player": 0,
                "review": review("accepted", "test", "Synthetic boundary."),
            }
        ],
        "overrides": {"observed-bolt": {"review": review("accepted", "test", "Synthetic observed cast.")}},
        "checkpoints": [],
        "windows": [
            {
                "id": "bolt",
                "game_id": "g1",
                "start_scenario": scenario,
                "event_ids": ["observed-bolt"],
                "end_assertions": scenario["expected_after"],
            }
        ],
        "budget": {"candidates": 16, "expansions": 500, "actions": 10, "seconds": 10, "escalations": 0},
    }
    return evidence, spec


def test_legal_resolution_with_inferred_payments_and_passes(engine, native_parity):
    evidence, spec = fixture()
    result = reconstruct(evidence, spec, engine)
    window = result["windows"][0]
    assert window["status"] == "matched"
    assert window["paths"][0]["verification"]["identical"]
    assert window["paths"][0]["view"]["opponent"]["life"] == 0
    assert sum(s["observed"] for s in window["paths"][0]["trace"]) == 1
    assert window["exhaustive"] is False
    assert evidence["records"][-1]["review"]["status"] == "pending"


def test_conflicting_endpoint_and_budget_report_unresolved(engine):
    evidence, spec = fixture()
    spec["windows"][0]["end_assertions"]["opponent.life"] = 19
    spec["budget"]["expansions"] = 1
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "unresolved" and result["truncated"]
    assert "budget" in result["reason"]


def test_never_replay_unreviewed_interpretation(engine):
    evidence, spec = fixture()
    spec["overrides"] = {}
    with pytest.raises(ValueError, match="independent review"):
        reconstruct(evidence, spec, engine)


def test_numeric_changes_and_legitimate_repeated_events_survive():
    evidence, spec = fixture()
    spec["windows"] = []
    source = spec["source_id"]
    for i, text in enumerate(["Alice draws a card.", "Alice draws a card.", "Alice draws three cards."]):
        evidence["records"].append(
            fact(
                f"draw-{i}",
                "log",
                {"text": text},
                [ref(source, 110 + i, 110 + i, "frame", str(i))],
                visibility="public",
            )
        )
    events = normalize_events(evidence, spec)[-3:]
    assert [e["event"]["count"] for e in events] == [1, 1, 3]
    assert not any(e.get("duplicate_candidate") for e in events)


def test_reshown_block_is_suggestion_until_independently_reviewed():
    evidence, spec = fixture()
    spec["windows"] = []
    source = spec["source_id"]
    for batch in range(2):
        for i, text in enumerate(["Alice plays Island.", "Bob casts Ponder.", "Bob draws a card."]):
            evidence["records"].append(
                fact(
                    f"b{batch}-{i}",
                    "log",
                    {"text": "10:42 AM: " + text},
                    [ref(source, 110 + batch, 110 + batch, "frame", str(batch))],
                    visibility="public",
                )
            )
    events = normalize_events(evidence, spec)
    assert [e.get("duplicate_candidate") for e in events[-3:]] == ["b0-0", "b0-1", "b0-2"]
    assert all(e["status"] == "proposed" for e in events[-3:])
    spec["overrides"]["b1-0"] = {
        "duplicate_of": "b0-0",
        "review": review("accepted", "test", "Same occurrence shown again."),
    }
    assert normalize_events(evidence, spec)[-3]["status"] == "duplicate"


def test_checkpoint_templates_do_not_mix_old_and_future_facts():
    evidence, spec = fixture()
    spec["windows"] = []
    source = spec["source_id"]
    ids = []
    for i, t in enumerate([100, 120]):
        r = fact(
            f"life-{i}",
            "state",
            {"path": "players.0.life", "value": 20 - i},
            [ref(source, t, t, "frame", str(i))],
            visibility="public",
        )
        r["review"] = review("accepted", "test", "Synthetic visible life.")
        evidence["records"].append(r)
        ids.append(r["id"])
    spec["checkpoints"] = [{"id": "cp", "game_id": "g1", "time": 100, "facts": [ids[0]]}]
    output = checkpoint_template(evidence, spec, "cp", "new")
    assert output["initial"]["players"][0]["life"] == 20 and output["review"]["status"] == "pending"
    spec["checkpoints"][0]["facts"] = [ids[1]]
    with pytest.raises(ValueError, match="inadmissible"):
        checkpoint_template(evidence, spec, "cp", "new")


def test_hidden_completion_variants_replay_from_root(engine, native_parity):
    evidence, spec = fixture()
    spec["windows"][0]["completion_domains"] = [{"path": "players.1.library.0", "candidates": ["Island", "Mountain"]}]
    output = reconstruct(evidence, spec, engine)["windows"][0]
    assert output["status"] == "matched" and output["completion_attempts"] == 2
    assert {p["completion"]["assignments"]["players.1.library.0"] for p in output["paths"]} == {"Island", "Mountain"}
    spec["windows"][0]["completion_domains"][0]["path"] = "players.0.hand.0"
    with pytest.raises(ValueError, match="synthetic hidden"):
        reconstruct(evidence, spec, engine)


def add_event(evidence, spec, rid, text):
    record = fact(
        rid, "log", {"text": text}, [ref(spec["source_id"], 101, 101, "frame", rid + ".png")], visibility="public"
    )
    evidence["records"].append(record)
    spec["overrides"][rid] = {"review": review("accepted", "test", "Reviewed synthetic transition.")}
    spec["windows"][0]["event_ids"].append(rid)


def test_impossible_reviewed_draw_prevents_a_false_match(engine):
    evidence, spec = fixture()
    add_event(evidence, spec, "impossible", "Alice draws 999 cards.")
    spec["budget"].update(actions=12, expansions=500)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "unresolved" and result["required_actions"] == 2
    assert not result["paths"]


def test_wrong_reviewed_target_is_not_compatible(engine):
    evidence, spec = fixture()
    spec["overrides"]["observed-bolt"]["text"] = "Alice casts Lightning Bolt targeting Alice."
    assert reconstruct(evidence, spec, engine)["windows"][0]["status"] == "unresolved"


def test_unsupported_reviewed_event_remains_unchecked(engine):
    evidence, spec = fixture()
    add_event(evidence, spec, "unknown", "Alice does something unavailable.")
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "unresolved" and result["unchecked_event_ids"] == ["unknown"]


def test_checkpoint_filters_replay_at_an_occurrence_boundary(engine):
    evidence, spec = fixture()
    r = fact(
        "wrong-life",
        "state",
        {"path": "players.1.life", "value": 0},
        [ref(spec["source_id"], 101, 101, "frame", "life.png")],
        visibility="public",
    )
    r["review"] = review("accepted", "test", "Contradictory life immediately after cast, before resolution.")
    evidence["records"].append(r)
    spec["checkpoints"] = [
        {
            "id": "cp",
            "game_id": "g1",
            "time": 101,
            "facts": ["wrong-life"],
            "position": {"event_id": "observed-bolt", "side": "after"},
        }
    ]
    spec["windows"][0]["checkpoint_ids"] = ["cp"]
    assert reconstruct(evidence, spec, engine)["windows"][0]["status"] == "unresolved"


def test_card_copy_constraints_prune_hidden_completions():
    from collections import Counter

    evidence, spec = fixture()
    window = spec["windows"][0]
    player = window["start_scenario"]["initial"]["players"][1]
    counts = Counter(
        c if isinstance(c, str) else c["name"]
        for zone in ("hand", "library", "graveyard", "exile", "battlefield")
        for c in player[zone]
    )
    counts["Island"] = 1
    window["source_decklists"] = {"1": dict(counts)}
    window["completion_domains"] = [{"path": f"players.1.library.{i}", "candidates": ["Island"]} for i in range(2)]
    events = [e for e in normalize_events(evidence, spec) if e["id"] == "observed-bolt"]
    assert list(completions(window, events, 64)) == []


def test_conflicting_checkpoint_fields_are_rejected(engine):
    evidence, spec = fixture()
    for i, life in enumerate((20, 19)):
        r = fact(
            f"life-{i}",
            "state",
            {"path": "players.0.life", "value": life},
            [ref(spec["source_id"], 100, 100, "frame", str(i))],
            visibility="public",
        )
        r["review"] = review("accepted", "test", "Synthetic field.")
        evidence["records"].append(r)
    spec["checkpoints"] = [{"id": "cp", "game_id": "g1", "time": 100, "facts": ["life-0", "life-1"]}]
    with pytest.raises(ValueError, match="conflicting"):
        reconstruct(evidence, spec, engine)


def test_reproduction_keeps_seed_assignments_and_event_alignment(engine, native_parity):
    evidence, spec = fixture()
    first = reconstruct(evidence, spec, engine)["windows"][0]["paths"][0]
    second = reconstruct(evidence, spec, engine)["windows"][0]["paths"][0]
    assert first["trace"] == second["trace"] and first["completion"] == second["completion"]
    assert first["seed"] == spec["windows"][0]["start_scenario"]["initial"]["seed"] and first["matched_event_ids"] == [
        "observed-bolt"
    ]


def test_successor_replays_the_retained_prefix_without_resetting(engine, native_parity):
    import copy

    evidence, spec = fixture()
    root = spec["windows"][0]["start_scenario"]
    initial = root["initial"]
    initial["players"][0]["hand"] = ["Lightning Bolt", "Lightning Bolt"]
    initial["players"][0]["hand_count"] = 2
    initial["players"][0]["battlefield"] = ["Mountain", "Mountain"]
    initial["players"][1]["life"] = 10
    for path, ids in root["state_facts"].items():
        from mtg_ml.expert.scenarios import at_path

        for rid in ids:
            next(r for r in evidence["records"] if r["id"] == rid)["value"]["value"] = copy.deepcopy(
                at_path(initial, path)
            )
    spec["windows"][0]["end_assertions"] = {"opponent.life": 7, "self.hand": ["Lightning Bolt"], "stack": []}
    root["expected_after"] = {}
    add_event(evidence, spec, "second-bolt", "Alice casts Lightning Bolt targeting Bob.")
    spec["windows"][0]["event_ids"].remove("second-bolt")
    spec["windows"].append(
        {
            "id": "second",
            "game_id": "g1",
            "start_window_id": "bolt",
            "event_ids": ["second-bolt"],
            "end_assertions": {"opponent.life": 4, "self.hand": [], "stack": []},
        }
    )
    spec["budget"].update(actions=20, expansions=1500)
    result = reconstruct(evidence, spec, engine)
    assert [w["status"] for w in result["windows"]] == ["matched", "matched"]
    child = result["windows"][1]["paths"][0]
    assert child["matched_event_ids"] == ["observed-bolt", "second-bolt"]
    assert child["verification"]["identical"]
    assert sum(s["selector"]["key"][:2] == ["cast", "Lightning Bolt"] for s in child["trace"]) == 2


def test_completions_preserve_named_root_card_identity():
    evidence, spec = fixture()
    window = spec["windows"][0]
    window["start_scenario"]["initial"]["players"][1]["library"][0] = {"name": "Mountain", "id": "hidden-slot"}
    window["completion_domains"] = [{"path": "players.1.library.0", "candidates": ["Island"]}]
    candidate, _ = next(completions(window, normalize_events(evidence, spec), 1))
    assert candidate["initial"]["players"][1]["library"][0] == {"name": "Island", "id": "hidden-slot"}


def test_source_search_reveal_is_not_a_draw(engine):
    from helpers import scenario
    from mtg_ml.expert.transitions import capture, frozen_option, transition

    g = scenario(
        p0={"hand": ["Lorien Revealed"], "battlefield": ["Island", "Island"], "library": ["Island", "Mountain"]}
    )
    facts = []
    for _ in range(12):
        opts = g.legal_options()
        index = next((i for i, o in enumerate(opts) if "islandcycling" in o.label.lower()), 0)
        before = capture(g)
        decision = (g.decision.player, g.decision.kind)
        option = frozen_option(opts[index])
        g.step(index)
        facts += transition(g, before, decision, option)
        if any(f["type"] == "reveal" for f in facts):
            break
    assert any(f["type"] == "cycle" for f in facts)
    assert not any(f["type"] == "draw" for f in facts)


def test_brainstorm_transition_counts_three_cards(engine):
    from helpers import scenario
    from mtg_ml.expert.transitions import capture, frozen_option, transition

    g = scenario(
        p0={
            "hand": ["Brainstorm"],
            "battlefield": ["Island"],
            "library": ["Ponder", "Island", "Counterspell", "Island"],
        }
    )
    facts = []
    for _ in range(12):
        opts = g.legal_options()
        index = next((i for i, o in enumerate(opts) if o.label == "Cast Brainstorm"), 0)
        before = capture(g)
        decision = (g.decision.player, g.decision.kind)
        option = frozen_option(opts[index])
        g.step(index)
        facts += transition(g, before, decision, option)
        if any(f["type"] == "draw" for f in facts):
            break
    assert next(f for f in facts if f["type"] == "draw")["count"] == 3
    assert not any(f["type"] == "reveal" for f in facts)


def rebind(evidence, root):
    import copy
    from mtg_ml.expert.scenarios import at_path

    for path, ids in root["state_facts"].items():
        for rid in ids:
            next(r for r in evidence["records"] if r["id"] == rid)["value"]["value"] = copy.deepcopy(
                at_path(root["initial"], path)
            )


def test_cycling_discard_cost_precedes_activation_and_search(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    p = root["initial"]["players"][0]
    p.update(
        hand=["Lorien Revealed"],
        hand_count=1,
        battlefield=["Island", "Island"],
        library=["Island", "Mountain"],
        library_count=2,
    )
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("cost", "Alice discards Lorien Revealed."),
        ("cycle", "Alice cycles Lorien Revealed."),
        ("reveal", "Alice reveals Island."),
    ]:
        add_event(evidence, spec, rid, text)
    window["end_assertions"] = {
        "self.hand": ["Island"],
        "self.graveyard": ["Lorien Revealed"],
        "self.library_count": 1,
        "stack": [],
    }
    spec["budget"].update(actions=20, expansions=2000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched"
    assert result["paths"][0]["matched_event_ids"] == ["cost", "cycle", "reveal"]


def test_escape_groups_three_exile_choices_before_the_cast(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=[],
        hand_count=0,
        battlefield=["Island"] * 3,
        graveyard=["Sleep of the Dead", "Ponder", "Brainstorm", "Mental Note"],
    )
    root["initial"]["players"][1]["battlefield"] = ["Refurbished Familiar"]
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    add_event(
        evidence, spec, "exile", "Alice exiles Ponder, Brainstorm, and Mental Note with Sleep of the Dead's ability."
    )
    add_event(
        evidence,
        spec,
        "escape",
        "Alice casts Sleep of the Dead from the graveyard for its escape cost targeting Refurbished Familiar.",
    )
    window["end_assertions"] = {
        "self.exile": {"multiset": ["Ponder", "Brainstorm", "Mental Note"]},
        "battlefield.3.name": "Refurbished Familiar",
        "battlefield.3.tapped": True,
        "stack": [],
    }
    spec["budget"].update(actions=30, expansions=4000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]


def test_split_draws_from_different_causes_cannot_match_one_draw():
    from mtg_ml.expert.transitions import advance

    events = [{"id": "draw", "event": {"type": "draw", "player": 0, "count": 2}}]
    cursor, matched, pending = advance(
        events, 0, [{"type": "draw", "player": 0, "count": 1, "cards": ["Island"], "cause": ("stack", 1)}]
    )
    cursor, matched, _ = advance(
        events, cursor, [{"type": "draw", "player": 0, "count": 1, "cards": ["Island"], "cause": ("stack", 2)}], pending
    )
    assert cursor == 0 and not matched


def test_replay_root_cannot_borrow_a_later_game_state(engine):
    evidence, spec = fixture()
    spec["windows"][0]["start_scenario"]["decision_time"] = 201
    with pytest.raises(ValueError, match="outside its game"):
        reconstruct(evidence, spec, engine)


def test_countered_chrysalis_keeps_its_cast_trigger(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=["Writhing Chrysalis"], hand_count=1, battlefield=["Forest", "Forest", "Mountain", "Mountain"]
    )
    root["initial"]["players"][1].update(hand=["Counterspell"], hand_count=1, battlefield=["Island", "Island"])
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("chrysalis", "Alice casts Writhing Chrysalis."),
        ("trigger", "Alice puts a triggered ability from Writhing Chrysalis onto the stack."),
        ("counterspell", "Bob casts Counterspell targeting Writhing Chrysalis."),
        ("tokens", "Alice's Writhing Chrysalis creates two Eldrazi Spawn Tokens."),
    ]:
        add_event(evidence, spec, rid, text)
    window["end_assertions"] = {
        "self.graveyard": ["Writhing Chrysalis"],
        "battlefield.6.name": "Eldrazi Spawn",
        "battlefield.7.name": "Eldrazi Spawn",
        "stack": [],
    }
    spec["budget"].update(candidates=64, actions=30, expansions=10000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]


@pytest.mark.parametrize(
    "text,kind,cards",
    [
        ("Alicecasts Ponder.", "cast", ["Ponder"]),
        ("Alice mills Lórien Revealed and Dispel.", "mill", ["Lorien Revealed", "Dispel"]),
        (
            "Bob's Writhing Chrysalis creates two Eldrazi Spawn Tokens.",
            "token",
            ["Writhing Chrysalis", "Eldrazi Spawn"],
        ),
        ("Eldrazi Spawn Token blocks Cryptic Serpent.", "block", ["Eldrazi Spawn", "Cryptic Serpent"]),
        ("Turn2: Alice", "turn", []),
        ("Alice draws three cards with Brainstorm.", "draw", ["Brainstorm"]),
    ],
)
def test_pilot_grammar(text, kind, cards):
    event = parse_event(text, {"Alice": 0, "Bob": 1})
    assert event["type"] == kind and event["cards"] == cards


def test_fetch_sacrifice_and_shuffle_replay_in_order(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=["Crop Rotation"],
        hand_count=1,
        battlefield=["Forest", "Forest"],
        library=["Swamp", "Island", "Swamp", "Island", "Swamp", "Island"],
        library_count=6,
    )
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("sac", "Alice sacrifices Forest."),
        ("rotation", "Alice casts Crop Rotation."),
        ("shuffle", "Alice shuffles her library."),
    ]:
        add_event(evidence, spec, rid, text)
    window["end_assertions"] = {
        "self.graveyard": ["Forest", "Crop Rotation"],
        "self.library_count": 5,
        "battlefield.1.name": "Swamp",
        "stack": [],
    }
    spec["budget"].update(actions=24, expansions=4000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]
    # The additional cost is logged before the cast, and fetching a land is a
    # shuffle even though the library no longer holds the same cards.
    assert result["paths"][0]["matched_event_ids"] == ["sac", "rotation", "shuffle"]


def ponder_window(evidence, spec):
    """A Ponder cast that keeps its optional shuffle, as MTGO logs it."""
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=["Ponder"],
        hand_count=1,
        battlefield=["Island"],
        library=["Swamp", "Island", "Swamp", "Island", "Swamp", "Island"],
        library_count=6,
    )
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("ponder", "Alice casts Ponder."),
        ("use", "Alice chooses to use the shuffle option of Ponder."),
        ("shuffle", "Alice shuffles her library."),
        ("draw", "Alice draws a card."),
    ]:
        add_event(evidence, spec, rid, text)
    window["end_assertions"] = {"self.graveyard": ["Ponder"], "self.library_count": 5, "stack": []}
    return window


def test_optional_shuffle_choice_matches_the_engine_obligation(engine, native_parity):
    evidence, spec = fixture()
    ponder_window(evidence, spec)
    spec["budget"].update(candidates=4, actions=24, expansions=6000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]
    assert result["paths"][0]["matched_event_ids"] == ["ponder", "use", "shuffle", "draw"]
    # The verification receipt counts the decisions whose features were
    # compared, so it cannot claim a check the lockstep replay never ran.
    verification = result["paths"][0]["verification"]
    assert verification["features_checked"] >= verification["actions_checked"] > 0


def test_one_completion_can_retain_several_action_paths(engine, native_parity):
    evidence, spec = fixture()
    ponder_window(evidence, spec)
    spec["budget"].update(candidates=6, actions=24, expansions=20000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    # Ponder's reordering is never logged, so several legal traces satisfy the
    # same evidence from one completion. They are retained, not collapsed.
    assert result["status"] == "matched" and result["completion_attempts"] == 1
    traces = {json.dumps([step["selector"] for step in p["trace"]], sort_keys=True) for p in result["paths"]}
    assert len(traces) == len(result["paths"]) > 1


def test_combat_declarations_replay_through_damage(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(hand=[], hand_count=0, battlefield=["Cryptic Serpent"])
    root["initial"]["players"][1].update(life=20, battlefield=["Refurbished Familiar"])
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("attack", "Bob is being attacked by Cryptic Serpent."),
        ("block", "Refurbished Familiar blocks Cryptic Serpent."),
    ]:
        add_event(evidence, spec, rid, text)
    window["end_assertions"] = {
        "opponent.graveyard": ["Refurbished Familiar"],
        "opponent.life": 20,
        "battlefield.0.name": "Cryptic Serpent",
        "battlefield.0.damage": 2,
    }
    spec["budget"].update(actions=24, expansions=4000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]
    assert result["paths"][0]["matched_event_ids"] == ["attack", "block"]


def test_counterspell_chain_resolves_in_stack_order(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=["Deep Analysis", "Counterspell"], hand_count=2, battlefield=["Island"] * 6
    )
    root["initial"]["players"][1].update(hand=["Counterspell"], hand_count=1, battlefield=["Island"] * 2)
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("analysis", "Alice casts Deep Analysis targeting Alice."),
        ("counter", "Bob casts Counterspell targeting Deep Analysis."),
        ("recounter", "Alice casts Counterspell targeting Counterspell."),
        ("draw", "Alice draws two cards."),
    ]:
        add_event(evidence, spec, rid, text)
    # Countering the counter lets the original spell resolve.
    window["end_assertions"] = {
        "self.graveyard": {"multiset": ["Counterspell", "Deep Analysis"]},
        "opponent.graveyard": ["Counterspell"],
        "self.cards_drawn_this_turn": 2,
        "stack": [],
    }
    spec["budget"].update(actions=40, expansions=20000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]
    assert result["paths"][0]["matched_event_ids"] == ["analysis", "counter", "recounter", "draw"]


def test_brainstorm_window_replays_one_three_card_draw(engine, native_parity):
    evidence, spec = fixture()
    window = spec["windows"][0]
    root = window["start_scenario"]
    root["initial"]["players"][0].update(
        hand=["Brainstorm"],
        hand_count=1,
        battlefield=["Island"],
        library=["Swamp", "Island", "Swamp", "Island"],
        library_count=4,
    )
    root["expected_after"] = {}
    rebind(evidence, root)
    window["event_ids"] = []
    for rid, text in [
        ("brainstorm", "Alice casts Brainstorm."),
        ("draw", "Alice draws three cards with Brainstorm."),
    ]:
        add_event(evidence, spec, rid, text)
    # Three drawn, two put back: the hand grows by one and the library by -1.
    window["end_assertions"] = {"self.graveyard": ["Brainstorm"], "self.library_count": 3, "stack": []}
    spec["budget"].update(candidates=4, actions=24, expansions=6000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "matched" and result["paths"][0]["verification"]["identical"]
    assert result["paths"][0]["matched_event_ids"] == ["brainstorm", "draw"]


def test_each_budget_reports_the_limit_it_exhausted(engine, monkeypatch):
    """An unresolved window says whether to raise a limit or bring evidence."""
    evidence, spec = fixture()
    spec["windows"][0]["end_assertions"]["opponent.life"] = 19
    spec["budget"].update(actions=1, expansions=10000)
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "unresolved" and result["truncated"]
    assert result["limits_hit"] == ["actions"] and "actions" in result["reason"]

    evidence, spec = fixture()
    spec["windows"][0]["end_assertions"]["opponent.life"] = 19
    spec["budget"].update(seconds=1, expansions=10**6, actions=200)
    clock = iter(0.4 * n for n in range(10**6))
    monkeypatch.setattr(R.time, "monotonic", lambda: next(clock))
    result = reconstruct(evidence, spec, engine)["windows"][0]
    assert result["status"] == "unresolved" and result["limits_hit"] == ["seconds"]
    assert "seconds" in result["reason"]


def test_aliases_must_be_text_substitutions(engine):
    evidence, spec = fixture()
    spec["aliases"] = {"saidin raken": "Alice"}
    assert reconstruct(evidence, spec, engine)["windows"][0]["status"] == "matched"
    spec["aliases"] = {"": "Alice"}
    with pytest.raises(ValueError, match="nonempty"):
        reconstruct(evidence, spec, engine)
    spec["aliases"] = ["saidin raken"]
    with pytest.raises(ValueError, match="aliases must be a mapping"):
        reconstruct(evidence, spec, engine)
