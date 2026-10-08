"""Compile reviewed specs by setup + legal actions, never by state injection."""

from __future__ import annotations

import copy
import math
import re

from ..backend import game_class
from ..encode import option_object_ids
from ..engine.cards import CARDS, TOKENS
from ..engine.game import STEPS
from ..engine.view import observe
from .artifacts import VERSION, admissible, require, validate_evidence, validate_review


def canonical(value):
    return [canonical(x) for x in value] if isinstance(value, (tuple, list)) else value


def select(g, selector: dict, objects: dict) -> int:
    """Exact key or exact label, optionally restricted to named setup objects.

    Keys intentionally omit IDs. Distinct copies need an object reference;
    ambiguous keys/labels never silently choose the first option.
    """
    require(g.decision is not None, "action requested after game over")
    require(selector.get("player") == g.decision.player and selector.get("kind") == g.decision.kind,
            f"decision mismatch: expected {selector.get('player')}/{selector.get('kind')}, got {g.decision}")
    require("key" in selector or "label" in selector, "selector needs a key or label")
    require(set(selector) <= {"player", "kind", "key", "label", "objects"}, "unknown selector field")
    hits = []
    for i, opt in enumerate(g.legal_options()):
        if "key" in selector and canonical(opt.key) != selector["key"]:
            continue
        if "label" in selector and opt.label != selector["label"]:
            continue
        if "objects" in selector:
            require(all(x in objects for x in selector["objects"]), "unknown setup object reference")
            wanted = [objects[x].oid for x in selector["objects"]]
            ids = option_object_ids(opt, g, g.decision.kind, features=6)
            if sorted(ids) != sorted(wanted):
                continue
        hits.append(i)
    require(len(hits) == 1, f"selector matches {len(hits)} legal actions: {selector}; offered {[o.label for o in g.legal_options()]}")
    return hits[0]


def _validate_initial(initial: dict) -> None:
    require(set(initial) <= {"players", "active", "step", "seed", "match_game", "deck_names", "turn", "lands_played", "spells_cast_this_turn"}, "unsupported initial state field")
    require(initial.get("active") in (0, 1) and initial.get("step") in STEPS, "initial active player and step required")
    require(isinstance(initial.get("players"), list) and len(initial["players"]) == 2, "two explicit players required")
    require(isinstance(initial.get("seed", 0), int) and initial.get("seed", 0) >= 0, "seed must be a nonnegative integer")
    require(initial.get("match_game", 1) in (1, 2, 3), "invalid match game")
    for field, default in (("turn", 1), ("lands_played", 0), ("spells_cast_this_turn", 0)):
        require(isinstance(initial.get(field, default), int) and initial.get(field, default) >= (1 if field == "turn" else 0), f"invalid {field}")
    ids = set()
    for player in initial["players"]:
        require(set(player) <= {"hand", "library", "graveyard", "exile", "battlefield", "life", "drawn", "hand_count", "library_count"}, "unsupported player state field")
        require(all(z in player for z in ("hand", "library", "graveyard", "exile", "battlefield", "life", "drawn", "hand_count", "library_count")), "all player zones, visible counts, life and drawn count must be explicit")
        require(isinstance(player["life"], int) and isinstance(player["drawn"], int) and player["drawn"] >= 0, "invalid player life or drawn count")
        for zone in ("hand", "library", "graveyard", "exile", "battlefield"):
            require(isinstance(player[zone], list), f"{zone} must be a list")
            for entry in player[zone]:
                card = {"name": entry} if isinstance(entry, str) else entry
                require(isinstance(card, dict) and card.get("name") in CARDS.keys() | TOKENS.keys(), f"unsupported card {card}")
                require(set(card) <= {"name", "id", "tapped", "sick", "counters"}, "unsupported card state (use legal setup actions)")
                if "id" in card:
                    require(bool(card["id"]) and card["id"] not in ids, "duplicate setup object id")
                    ids.add(card["id"])
                require(zone == "battlefield" or not (set(card) & {"tapped", "sick", "counters"}), "permanent flags require battlefield")
                for flag in ("tapped", "sick"):
                    require(flag not in card or isinstance(card[flag], bool), f"{flag} must be boolean")
                require(isinstance(card.get("counters", 0), int) and card.get("counters", 0) >= 0, "invalid counters")
        require(player["hand_count"] == len(player["hand"]) and player["library_count"] == len(player["library"]), "hidden completion changed visible zone counts")


def validate_scenario(spec: dict, evidence: dict) -> dict:
    validate_evidence(evidence)
    require(spec.get("format") == "ExecutableScenario" and spec.get("version") == VERSION, "unsupported ExecutableScenario version")
    require(bool(spec.get("id")) and bool(spec.get("group_id")), "scenario and leakage group ids required")
    validate_review(spec["review"])
    require(spec["review"]["status"] == "accepted", "scenario must be reviewed before compilation")
    require(spec.get("perspective") in (0, 1), "scenario perspective required")
    require(spec.get("episode_mode") == "reset", "v1 scenarios explicitly reset memory at the target position")
    require(bool(spec.get("assumptions")), "scenario assumptions (including normalized opening turn) required")
    _validate_initial(spec["initial"])
    records = {r["id"]: r for r in evidence["records"]}
    time = spec["decision_time"]
    p = spec["perspective"]
    require(isinstance(time, (int, float)) and math.isfinite(time) and time >= 0, "decision timestamp required")
    bindings = spec.get("state_facts", {})
    synthetic = spec.get("synthetic_fields", {})
    require(isinstance(bindings, dict) and isinstance(synthetic, dict), "state_facts and synthetic_fields must be maps")
    required = {"active", "step", "turn", "lands_played", "spells_cast_this_turn"} | {f"players.{i}.{k}" for i in (0, 1) for k in spec["initial"]["players"][i]}
    require(required >= bindings.keys() | synthetic.keys(), "unknown state field binding")
    require(required <= bindings.keys() | synthetic.keys(), f"unaccounted state fields: {sorted(required - bindings.keys() - synthetic.keys())}")
    require(not bindings.keys() & synthetic.keys(), "state field cannot be both observed and synthetic")
    for path, ids in bindings.items():
        require(path in required and bool(ids), f"invalid state binding {path}")
        for rid in ids:
            require(rid in records and admissible(records[rid], p, time), f"inadmissible state fact {rid}")
            record = records[rid]
            require(record["kind"] == "state" and record["value"].get("path") == path, f"{rid} does not attest state field {path}")
            require(record["value"].get("value") == at_path(spec["initial"], path), f"state fact {rid} contradicts initial {path}")
            for r in record["refs"]:
                source = next(s for s in evidence["sources"] if s["id"] == r["source_id"])
                require(spec["group_id"] == source.get("group_id", source["id"]), "scenario group must match its source video group")
    for path, rationale in synthetic.items():
        require(path in required and isinstance(rationale, str) and bool(rationale), "synthetic fields need explicit rationale")
    steps = spec.get("demonstration", [])
    require(steps or spec.get("preferred") or spec.get("continuation"), "scenario needs a demonstration, preferred actions or a regression continuation")
    if steps:
        # Opponent hand/library completions are hidden, but the player's own
        # hand must come from the video for an expert action to be meaningful.
        require(f"players.{p}.hand" not in synthetic, "expert demonstrations cannot invent the player's hand")
        hidden_fields = {f"players.{p}.library", f"players.{1-p}.library", f"players.{1-p}.hand"}
        require(synthetic.keys() <= hidden_fields, "expert demonstrations cannot invent public state")
        for step in steps:
            if not step.get("label", False):
                require(bool(step.get("refs")), "history actions need evidence references")
            for rid in step.get("refs", []):
                require(rid in records and admissible(records[rid], p, time, action=True), f"inadmissible action fact {rid}")
                require(records[rid]["kind"] in {"log", "action"}, "action label must reference an observed action")
                require(records[rid]["value"].get("selector") == step["selector"], f"action {rid} lacks the reviewed engine selector")
                for r in records[rid]["refs"]:
                    source = next(s for s in evidence["sources"] if s["id"] == r["source_id"])
                    require(spec["group_id"] == source.get("group_id", source["id"]), "action group must match its source video group")
            require(bool(step.get("refs")), "demonstrated actions need evidence references")
    return spec


def compile_scenario(spec: dict, evidence: dict, engine: str | None = None):
    validate_scenario(spec, evidence)
    initial = copy.deepcopy(spec["initial"])
    objects = {}

    def setup(g):
        g.turn = initial.get("turn", 1) - 1  # _main increments at its first turn
        g.lands_played = initial.get("lands_played", 0)
        g.spells_cast_this_turn = initial.get("spells_cast_this_turn", 0)
        for seat, player in enumerate(initial["players"]):
            for zone in ("hand", "library", "graveyard", "exile", "battlefield"):
                for entry in player[zone]:
                    card = {"name": entry} if isinstance(entry, str) else dict(entry)
                    name, oid = card.pop("name"), card.pop("id", None)
                    obj = g.add_card(name, seat, zone, **card)
                    if oid:
                        objects[oid] = obj
            g.players[seat].life = player["life"]
            g.players[seat].cards_drawn_this_turn = player["drawn"]

    g = game_class(engine)(([], []), seed=initial.get("seed", 0), starting_player=initial["active"],
                           setup=setup, start_step=initial["step"], auto_single=False, log=True,
                           match_game=initial.get("match_game", 1), deck_names=initial.get("deck_names"),
                           max_turns=initial.get("turn", 1) + 100)
    for selector in spec.get("setup_actions", []):
        g.step(select(g, selector, objects))
    require(g.decision is not None and g.decision.player == spec["perspective"], "compiled position is not the reviewed player's decision")
    if "expected_view" in spec:
        assert_paths(observe(g, spec["perspective"]), spec["expected_view"])
    return g, objects


def at_path(value, path: str):
    for bit in path.split("."):
        value = value[int(bit)] if isinstance(value, list) else value[bit]
    return value


def assert_paths(value: dict, assertions: dict) -> None:
    for path, expected in assertions.items():
        actual = at_path(value, path)
        require(actual == expected, f"{path}: expected {expected!r}, got {actual!r}")


def regression(spec: dict, evidence: dict, engine: str | None = None) -> dict:
    g, objects = compile_scenario(spec, evidence, engine)
    start = observe(g, spec["perspective"])
    preferred = [select(g, s, objects) for s in spec.get("preferred", [])]
    for selector in spec.get("continuation", []):
        g.step(select(g, selector, objects))
    finish = observe(g, spec["perspective"])
    require(bool(spec.get("expected_after")) or bool(preferred), "regression needs assertions or preferred actions")
    assert_paths(finish, spec.get("expected_after", {}))
    return {"scenario_id": spec["id"], "start": start, "finish": finish, "preferred_indices": preferred}


def selector_for(g, index: int) -> dict:
    opt = g.legal_options()[index]
    # Exact labels retain IDs, for local inspection only. Durable importers
    # should attach named objects instead (especially for distinct copies).
    return {"player": g.decision.player, "kind": g.decision.kind, "key": canonical(opt.key), "label": opt.label}


def normalized_label(label: str) -> str:
    return re.sub(r"#\d+", "#object", label)
