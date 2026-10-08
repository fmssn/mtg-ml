"""Bounded legal replay between evidence checkpoints, with explicit completions."""

from __future__ import annotations

from collections import Counter
import copy
import time

from ..engine.view import observe
from ..engine.cards import CARDS
from .artifacts import VERSION, admissible, finite, require, validate_evidence, validate_review
from .log_events import normalize_events
from .scenarios import at_path, canonical, compile_scenario, selector_for
from .transitions import CHECKABLE, advance, capture, frozen_option, transition

DEFAULTS = {"candidates": 64, "expansions": 50000, "actions": 200, "seconds": 30, "escalations": 2}


def matches(view, assertions):
    for path, expected in assertions.items():
        try:
            actual = at_path(view, path)
        except (KeyError, IndexError, TypeError):
            return False
        if isinstance(expected, dict) and set(expected) == {"multiset"}:
            if Counter(map(repr, actual)) != Counter(map(repr, expected["multiset"])):
                return False
        elif canonical(actual) != expected:
            return False
    return True


def checkpoint_state(g):
    """Only comparable scenario fields; hidden completions remain separate."""
    state = {
        "active": g.active,
        "step": g.step_name,
        "turn": g.turn,
        "lands_played": g.lands_played,
        "spells_cast_this_turn": g.spells_cast_this_turn,
        "players": [],
    }
    for seat, p in enumerate(g.players):
        state["players"].append(
            {
                "life": p.life,
                "drawn": p.cards_drawn_this_turn,
                "hand": [c.name for c in p.hand],
                "library": [c.name for c in p.library],
                "hand_count": len(p.hand),
                "library_count": len(p.library),
                "graveyard": [c.name for c in p.graveyard],
                "exile": [c.name for c in p.exile],
                "battlefield": [
                    {"name": c.name, "tapped": c.tapped, "sick": c.sick, "counters": c.counters}
                    for c in g.battlefield
                    if c.controller == seat
                ],
            }
        )
    return state


def _gates(window, evidence, events):
    records = {r["id"]: r for r in evidence["records"]}
    order = {e["id"]: i for i, e in enumerate(events)}
    gates = []
    for cp in window.get("checkpoints", []):
        if "position" in cp:
            pos = cp["position"]
            require(pos["event_id"] in order, "checkpoint position outside replay window")
            cursor = order[pos["event_id"]] + int(pos["side"] == "after")
        else:
            require(
                cp["time"] == window["start_scenario"]["decision_time"],
                "non-root replay checkpoints need occurrence positions",
            )
            cursor = 0
        assertions = {}
        for rid in cp["facts"]:
            r = records[rid]
            path = r["value"]["path"]
            value = r["value"]["value"]
            if isinstance(value, list) and not path.endswith("library"):
                value = {"multiset": value}
            assertions[path] = value
        gates.append({"cursor": cursor, "id": cp["id"], "assertions": assertions, "kind": "state"})
    return gates


def _check_gates(g, gates, previous, cursor, perspective):
    for gate in gates:
        if previous < gate["cursor"] < cursor:
            return False
        if gate["cursor"] == cursor and previous < cursor:
            state = observe(g, perspective) if gate["kind"] == "view" else checkpoint_state(g)
            if not matches(state, gate["assertions"]):
                return False
    return True


def validate_spec(spec, evidence):
    validate_evidence(evidence)
    require(
        spec.get("format") == "ReconstructionSpec" and spec.get("version") == VERSION,
        "unsupported ReconstructionSpec version",
    )
    require(spec.get("source_id") in {s["id"] for s in evidence["sources"]}, "unknown reconstruction source")
    require(spec.get("perspective") in (0, 1), "perspective required")
    require(
        isinstance(spec.get("players"), dict) and set(spec["players"].values()) == {0, 1},
        "explicit two-player mapping required",
    )
    previous = -1
    game_ids = set()
    for g in spec.get("games", []):
        require(g.get("id") and g["id"] not in game_ids, "duplicate/empty game id")
        require(
            previous <= finite(g["start"], "game start") < finite(g["end"], "game end"), "overlapping/unordered games"
        )
        require(g.get("starting_player") in (0, 1), "game starting player required")
        validate_review(g["review"])
        require(g["review"]["status"] == "accepted", "game boundaries must be independently reviewed")
        previous = g["end"]
        game_ids.add(g["id"])
    require(game_ids, "game boundaries required")
    records = {r["id"]: r for r in evidence["records"]}
    for rid, patch in spec.get("overrides", {}).items():
        require(rid in records, "correction references missing evidence")
        validate_review(patch.get("review", {"status": "pending"}))
        if patch.get("status") in {"duplicate", "rejected"} or patch.get("duplicate_of"):
            require(patch["review"]["status"] == "accepted", "occurrence removal requires explicit review")
    ids = set()
    for cp in spec.get("checkpoints", []):
        require(cp.get("id") and cp["id"] not in ids and cp["game_id"] in game_ids, "invalid checkpoint identity")
        ids.add(cp["id"])
        finite(cp["time"], "checkpoint time")
        game = next(g for g in spec["games"] if g["id"] == cp["game_id"])
        require(game["start"] <= cp["time"] < game["end"], "checkpoint outside game")
        require(cp.get("facts"), "checkpoint needs source facts")
        for rid in cp["facts"]:
            require(
                rid in records
                and records[rid]["kind"] == "state"
                and admissible(records[rid], spec["perspective"], cp["time"]),
                "inadmissible checkpoint fact",
            )
            require(
                all(ref["source_id"] == spec["source_id"] for ref in records[rid]["refs"]), "checkpoint source mismatch"
            )
        facts = [records[rid] for rid in cp["facts"]]
        paths = [r["value"]["path"] for r in facts]
        require(len(paths) == len(set(paths)), "conflicting/repeated checkpoint field")
        if "position" in cp:
            position = cp["position"]
            require(
                set(position) == {"event_id", "side"}
                and position["event_id"] in records
                and position["side"] in {"before", "after"},
                "invalid checkpoint position",
            )
    window_ids = set()
    for w in spec.get("windows", []):
        require(w.get("id") and w["id"] not in window_ids and w["game_id"] in game_ids, "invalid window identity")
        window_ids.add(w["id"])
        require(w.get("event_ids"), "window needs observed events")
        require(set(w["event_ids"]) <= records.keys(), "window references missing events")
        require(w.get("end_assertions"), "window needs endpoint constraints")
        if w.get("start_window_id"):
            parent = next((x for x in spec["windows"] if x["id"] == w["start_window_id"]), None)
            require(
                parent is not None and parent["id"] != w["id"] and parent["game_id"] == w["game_id"],
                "invalid predecessor window",
            )
            require(not w.get("start_scenario"), "chained window cannot overwrite predecessor state")
        else:
            require(
                w.get("start_scenario", {}).get("perspective") == spec["perspective"], "window perspective mismatch"
            )
            game = next(g for g in spec["games"] if g["id"] == w["game_id"])
            decision_time = finite(w["start_scenario"].get("decision_time"), "window decision time")
            require(game["start"] <= decision_time < game["end"], "replay root outside its game")
        for cp_id in w.get("checkpoint_ids", []):
            require(cp_id in ids, "unknown window checkpoint")
    for game_id, decks in spec.get("source_decklists", {}).items():
        require(game_id in game_ids and isinstance(decks, dict), "invalid source decklist game")
        for seat, counts in decks.items():
            require(seat in {"0", "1"} and isinstance(counts, dict) and bool(counts), "invalid source decklist seat")
            require(
                all(
                    name in CARDS and isinstance(n, int) and not isinstance(n, bool) and n > 0
                    for name, n in counts.items()
                ),
                "invalid source card counts",
            )
    budget = {**DEFAULTS, **spec.get("budget", {})}
    require(set(budget) == set(DEFAULTS), "unknown search budget field")
    for k, v in budget.items():
        require(
            isinstance(v, int) and not isinstance(v, bool) and v >= (0 if k == "escalations" else 1),
            f"invalid budget {k}",
        )
    require(budget["escalations"] <= 2, "at most two automatic escalations")
    return budget


def _requirements(events):
    result = []
    for e in events:
        v = e["event"]
        t = v["type"]
        p = v.get("player")
        if t in {"cast", "play_land", "activate", "cycle"}:
            kind = "activate" if t == "cycle" else t
            prefix = [kind, v["card"]]
            if t == "cast":
                prefix += ["graveyard" if v["method"] in {"flashback", "escape"} else "hand", v["method"]]
            result.append(
                {
                    "player": p,
                    "kind": "priority",
                    "prefix": prefix,
                    "event_id": e["id"],
                    "targets": v.get("targets", []),
                    "target_player": v.get("target_player"),
                    "mode": v.get("mode"),
                }
            )
        elif t == "attack":
            result.append(
                {
                    "player": p,
                    "kind": "declare_attacker",
                    "prefix": ["attack"],
                    "cards": v["cards"],
                    "event_id": e["id"],
                }
            )
        elif t == "block" and v.get("targets"):
            result.append(
                {
                    "player": p,
                    "kind": "declare_blocker",
                    "prefix": ["block", v["card"], v["targets"][0]],
                    "event_id": e["id"],
                }
            )
        elif t == "choice" and "Ponder" in v["cards"]:
            result.append(
                {"player": p, "kind": "yes_no", "prefix": ["shuffle", "yes" if v["yes"] else "no"], "event_id": e["id"]}
            )
    return result


def _options(g, req, obligation):
    d = g.decision
    if d is None:
        return []
    result = []
    for i, o in enumerate(g.legal_options()):
        key = canonical(o.key)
        observed = (
            req is not None
            and req["player"] in (None, d.player)
            and d.kind == req["kind"]
            and key[: len(req["prefix"])] == req["prefix"]
        )
        if observed and req.get("cards") and key[1] is not None and key[1] not in req["cards"]:
            observed = False
        if observed and req.get("mode") and req["mode"].rstrip(".") not in o.label:
            observed = False
        if d.kind == "priority" and not observed and key[0] not in {"pass", "mana"}:
            continue
        if d.kind == "declare_attacker" and not observed and key != ["attack", None]:
            continue
        if d.kind == "declare_blocker" and not observed and key[-1] is not None:
            continue
        if d.kind == "target" and obligation:
            if obligation.get("target_player") is not None:
                if not o.label.startswith(f"Target player {obligation['target_player']} "):
                    continue
            elif obligation.get("targets") and not any(n in o.label for n in obligation["targets"]):
                continue
        result.append((i, observed))
    # Observed action first, then stable engine option order. No strategy bot involved.
    return sorted(result, key=lambda x: not x[1])


def _set(value, path, new):
    bits = path.split(".")
    obj = value
    for bit in bits[:-1]:
        obj = obj[int(bit)] if isinstance(obj, list) else obj[bit]
    if isinstance(obj, list):
        old = obj[int(bits[-1])]
        obj[int(bits[-1])] = {**old, "name": new} if isinstance(old, dict) and "name" in old else new
    else:
        obj[bits[-1]] = new


def completions(window, events, limit, stats=None):
    """Finite, evidence-directed candidates. Bounds never imply exhaustive hidden-state search."""
    base = copy.deepcopy(window["start_scenario"])
    domains = copy.deepcopy(window.get("completion_domains", []))
    p = base["perspective"]
    hidden = base["synthetic_fields"]
    # Explicit domains and all compatible anonymous hand slots are branched;
    # observed cards are not assigned to slots by event position.
    names = [
        e["event"]["card"]
        for e in events
        if e["event"].get("player") == 1 - p
        and e["event"]["type"] in {"play_land", "cast", "cycle"}
        and e["event"].get("method", "normal") == "normal"
    ]
    if names and f"players.{1 - p}.hand" in hidden:
        hand = base["initial"]["players"][1 - p]["hand"]
        for i in range(len(hand)):
            path = f"players.{1 - p}.hand.{i}"
            if not any(d["path"] == path for d in domains):
                domains.append(
                    {
                        "path": path,
                        "candidates": list(
                            dict.fromkeys(names + [hand[i] if isinstance(hand[i], str) else hand[i]["name"]])
                        ),
                        "anonymous_hand": True,
                    }
                )
    # New endpoint hand identities constrain anonymous top cards; do not backdate knowledge.
    expected = window["end_assertions"].get("self.hand")
    if expected is not None and f"players.{p}.library" in hidden and not domains:
        endpoint = expected.get("multiset") if isinstance(expected, dict) else expected
        opening = Counter(c if isinstance(c, str) else c["name"] for c in base["initial"]["players"][p]["hand"])
        for e in events:
            v = e["event"]
            if v.get("player") == p and v["type"] in {"cast", "play_land", "cycle"} and opening[v["card"]]:
                opening[v["card"]] -= 1
        new = list((Counter(endpoint) - opening).elements())
        milled = [
            n
            for e in events
            if e["event"]["type"] == "mill" and e["event"].get("player") == p
            for n in e["event"]["cards"]
        ]
        candidates = sorted(set(new + milled))
        for i in range(min(len(new) + len(milled), 3, len(base["initial"]["players"][p]["library"]))):
            domains.append({"path": f"players.{p}.library.{i}", "candidates": candidates})
    for d in domains:
        field = ".".join(d["path"].split(".")[:3])
        require(
            field in hidden and field in {f"players.{p}.library", f"players.{1 - p}.library", f"players.{1 - p}.hand"},
            "completion domain must address an explicitly synthetic hidden zone",
        )
        require(d.get("candidates"), "empty completion domain")
        at_path(base["initial"], d["path"])
    require(len({d["path"] for d in domains}) == len(domains), "repeated completion domain")
    decks = window.get("source_decklists", {})

    def known_counts(candidate, seat, assignments):
        result = Counter()
        for zone in ("hand", "library", "graveyard", "exile", "battlefield"):
            synthetic = f"players.{seat}.{zone}" in hidden
            for i, c in enumerate(candidate["initial"]["players"][int(seat)][zone]):
                if synthetic and f"players.{seat}.{zone}.{i}" not in assignments:
                    continue
                name = c if isinstance(c, str) else c["name"]
                if name in CARDS:
                    result[name] += 1
        return result

    # Evidence-driven recursion checks each partial assignment, rather than
    # enumerating a Cartesian product and discovering copy violations later.
    produced = 0
    visits = 0
    visit_limit = max(limit * 100, 1000)

    def walk(index, candidate, assignments):
        nonlocal produced, visits
        if produced >= limit or visits >= visit_limit:
            return
        visits += 1
        if any(known_counts(candidate, seat, assignments) - Counter(counts) for seat, counts in decks.items()):
            return
        if index < len(domains):
            domain = domains[index]
            for value in domain["candidates"]:
                if (
                    domain.get("anonymous_hand")
                    and index
                    and domains[index - 1].get("anonymous_hand")
                    and domain["candidates"] == domains[index - 1]["candidates"]
                ):
                    prior = assignments[domains[index - 1]["path"]]
                    # Only anonymous, unbound root slots are exchangeable.
                    # Explicit object/domain constraints retain their identity.
                    if domain["candidates"].index(value) < domain["candidates"].index(prior):
                        continue
                require(value in CARDS, "unsupported completion card")
                child = copy.deepcopy(candidate)
                _set(child["initial"], domain["path"], value)
                yield from walk(index + 1, child, {**assignments, domain["path"]: value})
            return
        opp = Counter(c if isinstance(c, str) else c["name"] for c in candidate["initial"]["players"][1 - p]["hand"])
        # Future draws may supply cards not in the opening hand; require only
        # observed opponent cards before the first recorded opponent draw.
        required = []
        for e in events:
            v = e["event"]
            if v.get("player") != 1 - p:
                continue
            if v["type"] == "draw":
                break
            if v["type"] in {"cast", "play_land", "cycle"} and v.get("method", "normal") == "normal":
                required.append(v["card"])
        if Counter(required) - opp and f"players.{1 - p}.hand" in hidden:
            return
        for seat, counts in decks.items():
            player = candidate["initial"]["players"][int(seat)]
            remaining = list((Counter(counts) - known_counts(candidate, seat, assignments)).elements())
            slots = [
                f"players.{seat}.{zone}.{i}"
                for zone in ("hand", "library", "graveyard", "exile", "battlefield")
                if f"players.{seat}.{zone}" in hidden
                for i in range(len(player[zone]))
                if f"players.{seat}.{zone}.{i}" not in assignments
            ]
            if len(slots) > len(remaining):
                return
            for path, name in zip(slots, remaining):
                _set(candidate["initial"], path, name)
                assignments[path] = name
        produced += 1
        yield (
            candidate,
            {
                "assignments": assignments,
                "retrospective": bool(domains),
                "assumption": "Explicit hidden completions and RNG reproduce a compatible path, not the recorded shuffle.",
                "domain_visits": visits,
                "completion_limit": limit,
            },
        )

    for candidate, completion in walk(0, base, {}):
        yield candidate, completion
    if stats is not None:
        stats.update(
            domain_visits=visits,
            completion_candidates=produced,
            candidate_limit_hit=produced >= limit,
            domain_limit_hit=visits >= visit_limit,
        )


def verify_path(scenario, evidence, trace, assertions, events=None):
    games = [compile_scenario(scenario, evidence, engine)[0] for engine in ("python", "native")]
    from ..difftest import snapshot

    cursor = 0
    pending = []
    for step in [*trace, None]:
        require(snapshot(games[0]) == snapshot(games[1]), "reconstruction engine divergence")
        if step is not None:
            facts = []
            for g in games:
                opts = g.legal_options()
                i = step["index"]
                require(
                    i < len(opts) and selector_for(g, i) == step["selector"], "reconstruction trace no longer resolves"
                )
                before = capture(g)
                decision = (g.decision.player, g.decision.kind)
                option = frozen_option(opts[i])
                g.step(i)
                facts.append(transition(g, before, decision, option))
            require(facts[0] == facts[1], "reconstruction transition divergence")
            if events is not None:
                cursor, matched, pending = advance(events, cursor, facts[0], pending)
                require(matched == step["matched_event_ids"], "reconstruction event alignment changed")
    if events is not None:
        require(cursor == len(events), "reconstruction left unchecked events")
    require(matches(observe(games[0], scenario["perspective"]), assertions), "reconstruction endpoint contradicted")
    return {"identical": True, "actions_checked": len(trace), "features_checked": True, "engines": ["python", "native"]}


def _search(window, evidence, events, budget, engine):
    started = time.monotonic()
    expansions = 0
    truncated = False
    paths = []
    attempts = 0
    best = 0
    unchecked = [e["id"] for e in events if e["event"]["type"] not in CHECKABLE]
    if unchecked:
        return {
            "status": "unresolved",
            "paths": [],
            "expansions": 0,
            "completion_attempts": 0,
            "actions_matched": 0,
            "required_actions": len(events),
            "wall_seconds": 0,
            "truncated": False,
            "exhaustive": False,
            "unchecked_event_ids": unchecked,
            "reason": "Unsupported transition constraints: " + ", ".join(unchecked),
        }
    gates = _gates(window, evidence, events)
    completion_stats = {}
    prefixes = window.get("prefixes", [{"trace": [], "assertions": {}}])
    if not prefixes:
        return {
            "status": "unresolved",
            "paths": [],
            "expansions": 0,
            "completion_attempts": 0,
            "actions_matched": 0,
            "required_actions": len(events),
            "wall_seconds": 0,
            "truncated": False,
            "exhaustive": False,
            "reason": "Predecessor window has no compatible paths.",
        }
    for root, completion in completions(window, events, budget["candidates"], completion_stats):
        if expansions >= budget["expansions"] or time.monotonic() - started >= budget["seconds"]:
            truncated = True
            break
        attempts += 1
        base_paths = len(paths)
        g, _ = compile_scenario(root, evidence, engine)
        if not _check_gates(g, gates, -1, 0, root["perspective"]):
            continue
        frontier = []
        for prefix in prefixes:
            if time.monotonic() - started >= budget["seconds"]:
                truncated = True
                break
            candidate = g.copy()
            cursor = 0
            pending = []
            trace = []
            valid = True
            for recorded in prefix["trace"]:
                if expansions >= budget["expansions"] or time.monotonic() - started >= budget["seconds"]:
                    truncated = True
                    valid = False
                    break
                if candidate.decision is None:
                    valid = False
                    break
                hits = [
                    i
                    for i in range(len(candidate.legal_options()))
                    if selector_for(candidate, i) == recorded["selector"]
                ]
                if len(hits) != 1:
                    valid = False
                    break
                index = hits[0]
                before = capture(candidate)
                decision = (candidate.decision.player, candidate.decision.kind)
                option = frozen_option(candidate.legal_options()[index])
                candidate.step(index)
                expansions += 1
                new_cursor, matched, pending = advance(
                    events, cursor, transition(candidate, before, decision, option), pending
                )
                if matched != recorded["matched_event_ids"] or not _check_gates(
                    candidate, gates, cursor, new_cursor, root["perspective"]
                ):
                    valid = False
                    break
                cursor = new_cursor
                trace.append({**recorded, "index": index})
            if valid and matches(observe(candidate, root["perspective"]), prefix["assertions"]):
                frontier.append((candidate, cursor, trace, None, pending))
        for depth in range(budget["actions"] + 1):
            next_frontier = []
            for game, cursor, trace, obligation, pending in frontier:
                best = max(best, cursor)
                if cursor == len(events) and matches(observe(game, root["perspective"]), window["end_assertions"]):
                    paths.append(
                        {
                            "scenario": root,
                            "completion": completion,
                            "trace": trace,
                            "view": observe(game, root["perspective"]),
                            "seed": root["initial"].get("seed", 0),
                            "matched_event_ids": [e["id"] for e in events],
                            "checkpoint_agreement": [{"checkpoint_id": gate["id"], "agrees": True} for gate in gates],
                            "verification": verify_path(root, evidence, trace, window["end_assertions"], events),
                        }
                    )
                    if len(paths) >= budget["candidates"]:
                        truncated = True
                        break
                    continue
                if len(trace) >= budget["actions"]:
                    truncated = True
                    continue
                # MTGO can log a discard/exile cost before the corresponding
                # activation/cast. Guide with the next action, while retaining
                # the original cursor and checking every intervening outcome.
                requirements = next((_requirements([event]) for event in events[cursor:] if _requirements([event])), [])
                req = requirements[0] if requirements else None
                for index, observed in _options(game, req, obligation):
                    if expansions >= budget["expansions"] or time.monotonic() - started >= budget["seconds"]:
                        truncated = True
                        break
                    child = game.copy()
                    selector = selector_for(child, index)
                    before = capture(child)
                    decision = (child.decision.player, child.decision.kind)
                    option = frozen_option(child.legal_options()[index])
                    child.step(index)
                    expansions += 1
                    new_cursor, matched, new_pending = advance(
                        events, cursor, transition(child, before, decision, option), pending
                    )
                    if not _check_gates(child, gates, cursor, new_cursor, root["perspective"]):
                        continue
                    new_obligation = (
                        req if observed and (req.get("targets") or req.get("target_player") is not None) else obligation
                    )
                    if child.decision is None or child.decision.kind == "priority":
                        new_obligation = None
                    step = {
                        "index": index,
                        "selector": selector,
                        "observed": observed,
                        "event_id": req["event_id"] if observed else None,
                        "matched_event_ids": matched,
                        "assumption": None if observed else "Unrecorded legal engine choice.",
                    }
                    next_frontier.append((child, new_cursor, trace + [step], new_obligation, new_pending))
                if expansions >= budget["expansions"] or time.monotonic() - started >= budget["seconds"]:
                    break
            if (
                len(paths) > base_paths
                or expansions >= budget["expansions"]
                or time.monotonic() - started >= budget["seconds"]
            ):
                break
            # Beam pruning is reported. Never use an observable digest to merge hidden states/coroutines.
            next_frontier.sort(key=lambda item: -item[1])
            if len(next_frontier) > budget["candidates"]:
                truncated = True
            frontier = next_frontier[: budget["candidates"]]
            if not frontier:
                break
        if (
            len(paths) >= budget["candidates"]
            or expansions >= budget["expansions"]
            or time.monotonic() - started >= budget["seconds"]
        ):
            break
    truncated = (
        truncated
        or completion_stats.get("candidate_limit_hit", False)
        or completion_stats.get("domain_limit_hit", False)
    )
    return {
        "status": "matched" if paths else "unresolved",
        "paths": paths,
        "expansions": expansions,
        "completion_attempts": attempts,
        "completion_stats": completion_stats,
        "actions_matched": best,
        "required_actions": len(events),
        "wall_seconds": time.monotonic() - started,
        "truncated": truncated,
        "exhaustive": False,
        "reason": None if paths else "No compatible path within the explored completion/search budget.",
    }


def reconstruct(evidence, spec, engine="native", on_progress=None):
    budget = validate_spec(spec, evidence)
    started = time.monotonic()
    events = normalize_events(evidence, spec)
    by_id = {e["id"]: e for e in events}
    windows = []
    issues = []
    for e in events:
        if e["status"] in {"outside_game", "duplicate", "rejected"}:
            continue
        if e["status"] == "gap" and e.get("recovery", {}).get("status") == "reviewed":
            continue
        if e["status"] == "gap" or e["event"]["type"] == "unsupported" or e.get("duplicate_candidate"):
            issues.append(
                {
                    "record_id": e["id"],
                    "game_id": e["game_id"],
                    "kind": "missing_observation" if e["status"] == "gap" else "extraction_error",
                    "detail": "Inspect source frames and apply a reviewed correction.",
                    "refs": e["refs"],
                }
            )
    expanded = {}
    checkpoint_by_id = {c["id"]: c for c in spec.get("checkpoints", [])}
    for original in spec.get("windows", []):
        w = copy.deepcopy(original)
        if w.get("start_window_id"):
            require(w["start_window_id"] in expanded, "predecessor windows must precede successors")
            parent = expanded[w["start_window_id"]]
            w["start_scenario"] = copy.deepcopy(parent["start_scenario"])
            w["event_ids"] = parent["event_ids"] + w["event_ids"]
            require(len(w["event_ids"]) == len(set(w["event_ids"])), "chained windows repeat event occurrences")
            w["completion_domains"] = parent.get("completion_domains", []) + w.get("completion_domains", [])
            w["checkpoint_ids"] = list(dict.fromkeys(parent.get("checkpoint_ids", []) + w.get("checkpoint_ids", [])))
            parent_result = next(result for result in windows if result["id"] == parent["id"])
            w["prefixes"] = [
                {"trace": path["trace"], "assertions": parent["end_assertions"]} for path in parent_result["paths"]
            ]
        w["source_decklists"] = spec.get("source_decklists", {}).get(w["game_id"], {})
        w["checkpoints"] = [checkpoint_by_id[c] for c in w.get("checkpoint_ids", [])]
        expanded[w["id"]] = w
        selected = [by_id[rid] for rid in w["event_ids"]]
        require(
            all(
                e["game_id"] == w["game_id"] and e["status"] not in {"duplicate", "rejected", "outside_game", "gap"}
                for e in selected
            ),
            "window includes removed/outside/gap occurrence",
        )
        # Only independently reviewed interpretations are hard replay constraints.
        require(all(e["review"]["status"] == "accepted" for e in selected), "window events need independent review")
        attempts = []
        for stage in range(budget["escalations"] + 1):
            limits = {
                **budget,
                "candidates": min(budget["candidates"] * (2**stage), 256),
                "expansions": min(budget["expansions"] * (2**stage), 200000),
                "seconds": min(budget["seconds"] * (2**stage), 120),
            }
            result = _search(w, evidence, selected, limits, engine)
            attempts.append({k: v for k, v in result.items() if k != "paths"})
            if on_progress:
                on_progress(
                    {
                        "window_id": w["id"],
                        "game_id": w["game_id"],
                        "stage": stage,
                        "limits": limits,
                        "result": attempts[-1],
                    }
                )
            if result["paths"]:
                break
        result.update(
            id=w["id"],
            game_id=w["game_id"],
            event_ids=w["event_ids"],
            checkpoint_ids=w.get("checkpoint_ids", []),
            attempts=attempts,
        )
        windows.append(result)
        if not result["paths"]:
            issues.append(
                {
                    "window_id": w["id"],
                    "game_id": w["game_id"],
                    "kind": "search_limit" if result["truncated"] else "hidden_completion",
                    "detail": result["reason"],
                    "next_action": "Inspect neighboring frames, add a checkpoint or completion domain, and retry.",
                }
            )
    records = {r["id"]: r for r in evidence["records"]}
    checkpoints = []
    for cp in spec.get("checkpoints", []):
        facts = [records[r] for r in cp["facts"]]
        checkpoints.append(
            {
                **cp,
                "state": {r["value"]["path"]: r["value"]["value"] for r in facts},
                "refs": [ref for r in facts for ref in r["refs"]],
                "review_status": "accepted",
                "knowledge_mode": "as_of",
            }
        )
    checked = {rid for window in windows if window["status"] == "matched" for rid in window["event_ids"]}
    contextual = {"opening", "starting_player", "skip_draw", "lobby", "concede", "result"}
    for e in events:
        e["replay_coverage"] = (
            "verified"
            if e["id"] in checked
            else "removed"
            if e["status"] in {"duplicate", "rejected", "outside_game"}
            else "boundary"
            if e["event"]["type"] in contextual
            else "evidence_only"
        )
    uncovered = [e["id"] for e in events if e["replay_coverage"] == "evidence_only" and e["event"]["type"] != "gap"]
    return {
        "format": "ReconstructionTimeline",
        "version": VERSION,
        "source_id": spec["source_id"],
        "perspective": spec["perspective"],
        "spec": spec,
        "sources": evidence["sources"],
        "games": spec["games"],
        "events": events,
        "checkpoints": checkpoints,
        "windows": windows,
        "issues": issues,
        "metrics": {
            "log_occurrences": sum(e["event"]["type"] != "gap" for e in events),
            "gaps": sum(e["event"]["type"] == "gap" for e in events),
            "duplicate_occurrences": sum(e["status"] == "duplicate" for e in events),
            "reviewed_events": sum(e["review"]["status"] == "accepted" and e["event"]["type"] != "gap" for e in events),
            "verified_event_ids": sorted(checked),
            "evidence_only_event_ids": uncovered,
            "reviewed_gaps": sum(
                e["event"]["type"] == "gap" and e.get("recovery", {}).get("status") == "reviewed" for e in events
            ),
            "checkpoint_count": len(checkpoints),
            "matched_windows": sum(w["status"] == "matched" for w in windows),
            "unresolved_windows": sum(w["status"] == "unresolved" for w in windows),
            "wall_seconds": time.monotonic() - started,
        },
    }


def checkpoint_template(evidence, spec, checkpoint_id, scenario_id):
    """Snapshot-scoped facts avoid treating an older checkpoint as a current field binding."""
    validate_spec(spec, evidence)
    cp = next((c for c in spec["checkpoints"] if c["id"] == checkpoint_id), None)
    require(cp is not None, "unknown checkpoint")
    from .__main__ import scenario_template

    selected = {**evidence, "records": [r for r in evidence["records"] if r["id"] in cp["facts"]]}
    result = scenario_template(selected, spec["source_id"], scenario_id, cp["time"], spec["perspective"])
    result["reconstruction"] = {"checkpoint_id": checkpoint_id, "game_id": cp["game_id"], "knowledge_mode": "as_of"}
    return result
