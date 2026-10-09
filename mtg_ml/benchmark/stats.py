"""The one implementation of benchmark statistics: pure Python, deterministic, no numpy.

Game scores are resampled by whole four-slot blocks with one index list shared by every
cell; puzzle scores are resampled by whole leakage groups. Every function raises on
technical errors or missing rows, so a failure can never raise a score.
"""

from collections import Counter
import math
import random

from .artifacts import FAIR, require
from .schedule import bootstrap_seed

INTERVAL_METHOD = "paired-block-bootstrap-v1"
MIN_GROUPS_DECK = 20  # independent leakage groups before a puzzle interval is reported
MIN_GROUPS_OVERALL = MIN_GROUPS_DECK
MARGIN = 0.03
SATURATED, FLOOR = 0.95, 0.05
Z_ALPHA, Z_POWER = 1.96, 0.84
CONSTANTS = {"min_groups_deck": MIN_GROUPS_DECK, "min_groups_overall": MIN_GROUPS_OVERALL, "margin": MARGIN,
             "saturated": SATURATED, "floor": FLOOR, "interval_method": INTERVAL_METHOD}
SLOTS = 4


def percentile(values, q):
    """Linear interpolation between order statistics, q in [0, 1]."""
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def score(row):
    """win = 1, draw = 0.5, loss = 0; a row that did not complete has no score."""
    require(row["status"] == "completed", "game_rows.status", "technical errors cannot enter scores")
    if row["winner"] is None:
        return 0.5
    return 1.0 if row["winner"] == row["learner_seat"] else 0.0


def identity(row):
    return row["mode"], row["cell"], row["block"], row["slot"]


def block_scores(rows, plan):
    """{mode: {cell: [block value, ...]}}, blocks in order; strict join against the plan."""
    expected = {identity(p) for p in plan}
    require(len(expected) == len(plan) and bool(expected), "planned_game_rows", "empty or duplicate plan")
    seen = Counter(identity(r) for r in rows)
    require(set(seen) == expected and all(n == 1 for n in seen.values()), "game_rows", "missing, duplicate or unexpected rows")
    slots = {}
    for r in rows:
        mode, cell, block, slot = identity(r)
        slots.setdefault((mode, cell, block), {})[slot] = score(r)
    out = {}
    for (mode, cell, block), by_slot in sorted(slots.items()):
        require(len(by_slot) == SLOTS, "game_rows", "a block needs all four slots")
        out.setdefault(mode, {}).setdefault(cell, []).append(sum(by_slot.values()) / SLOTS)
    for mode, cells in out.items():
        require(len({len(v) for v in cells.values()}) == 1, "game_rows", f"{mode}: cells differ in block count")
    return out


def _interval(samples, confidence):
    tail = (1 - confidence) / 2
    return [percentile(samples, tail), percentile(samples, 1 - tail)]


def game_intervals(block_values, replicates, seed, confidence=0.95):
    """Per-cell and equal-weight cell-mean intervals, one shared index list per replicate."""
    cells = sorted(block_values)
    require(bool(cells), "block_values", "no cells")
    n = len(block_values[cells[0]])
    require(all(len(block_values[c]) == n for c in cells), "block_values", "cells differ in block count")
    means = {c: sum(block_values[c]) / n for c in cells}
    mean = sum(means.values()) / len(cells)
    out = {"blocks": n, "replicates": replicates, "confidence": confidence, "seed": seed,
           "cells": {c: {"mean": means[c], "ci95": None} for c in cells}, "mean": {"mean": mean, "ci95": None}}
    if n < 2:
        out["status"], out["reason"] = "unavailable", "fewer than 2 blocks"
        return out
    rng = random.Random(seed)
    by_cell = {c: [] for c in cells}
    overall = []
    for _ in range(replicates):
        indices = rng.choices(range(n), k=n)
        total = 0.0
        for c in cells:
            values = block_values[c]
            m = sum(values[i] for i in indices) / n
            by_cell[c].append(m)
            total += m
        overall.append(total / len(cells))
    for c in cells:
        out["cells"][c]["ci95"] = _interval(by_cell[c], confidence)
    out["mean"]["ci95"] = _interval(overall, confidence)
    out["status"] = "ok"
    return out


def paired_game_differences(baseline, candidate, plan):
    """{mode: {cell: [candidate - baseline per block]}} from two complete row sets."""
    a, b = block_scores(baseline, plan), block_scores(candidate, plan)
    return {mode: {cell: [y - x for x, y in zip(a[mode][cell], b[mode][cell])] for cell in a[mode]} for mode in a}


def leakage_groups(puzzles):
    """Union-find over template/source ids; {group id: sorted puzzle ids}. A group is named
    by its smallest puzzle id. Puzzles are mappings with id, template_id and source_group_id."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = set()
    for p in puzzles:
        ids.add(p["id"])
        a, b = find(("template", p["template_id"])), find(("source", p["source_group_id"]))
        parent[a] = b
    members = {}
    for p in puzzles:
        members.setdefault(find(("template", p["template_id"])), []).append(p["id"])
    groups = {}
    for ids_ in members.values():
        ids_ = sorted(set(ids_))
        groups[ids_[0]] = ids_
    return dict(sorted(groups.items()))


def puzzle_values(rows):
    """{puzzle: value}: a repetition succeeds only if every case does; repetitions are averaged
    per puzzle and every puzzle weighs the same. Raises on any technical error."""
    require(all(r["status"] != "error" for r in rows), "puzzle_rows", "technical errors cannot enter scores")
    cases = {}
    for r in rows:
        cases.setdefault((r["puzzle"], r["repetition"]), []).append(r["status"] == "success")
    reps = {}
    for (puzzle, _), ok in cases.items():
        reps.setdefault(puzzle, []).append(all(ok))
    return {p: sum(v) / len(v) for p, v in sorted(reps.items())}


def puzzle_meta(rows):
    """{puzzle: {deck, category, template_id, source_group_id}} from planned or result rows."""
    return {r["puzzle"]: {"id": r["puzzle"], "deck": r["deck"], "category": r["category"],
                          "template_id": r["template_id"], "source_group_id": r["source_group_id"]} for r in rows}


def paired_puzzle_differences(baseline, candidate):
    require(set(baseline) == set(candidate), "puzzle_values", "arms cover different puzzles")
    return {p: candidate[p] - baseline[p] for p in sorted(baseline)}


def _unavailable(reason):
    return {"status": "unavailable", "reason": reason, "ci95": None}


def puzzle_intervals(values, meta, replicates, seed, confidence=0.95):
    """Overall and per-deck intervals over whole leakage groups. `values` is per puzzle (a
    single arm or a paired difference); `meta` maps puzzle to its puzzle_meta entry. Category
    results are descriptive only."""
    require(set(values) <= set(meta), "puzzle_values", "puzzle without metadata")
    groups = leakage_groups([meta[p] for p in values])
    decks = sorted({meta[p]["deck"] for p in values})
    by_deck = {d: sorted(p for p in values if meta[p]["deck"] == d) for d in decks}
    deck_groups = {d: sum(any(meta[p]["deck"] == d for p in ids) for ids in groups.values()) for d in decks}
    out = {"puzzles": len(values), "groups": len(groups), "mean": sum(values.values()) / len(values) if values else None,
           "decks": {d: {"puzzles": len(by_deck[d]), "groups": deck_groups[d], "mean": sum(values[p] for p in by_deck[d]) / len(by_deck[d])}
                     for d in decks}}
    wanted = {"overall": (None, MIN_GROUPS_OVERALL, len(groups))}
    wanted.update({d: (d, MIN_GROUPS_DECK, deck_groups[d]) for d in decks})
    results = {}
    for name, (deck, minimum, have) in wanted.items():
        if not values or have < minimum:
            results[name] = _unavailable(f"{have} leakage groups, minimum {minimum}")
    if len(results) < len(wanted):
        rng = random.Random(seed)
        keys = list(groups)
        drawn = {name: [] for name in wanted if name not in results}
        for _ in range(replicates):
            picked = [p for g in rng.choices(keys, k=len(keys)) for p in groups[g]]
            for name in drawn:
                subset = [values[p] for p in picked if name == "overall" or meta[p]["deck"] == name]
                if subset:
                    drawn[name].append(sum(subset) / len(subset))
        for name, samples in drawn.items():
            results[name] = {"status": "ok", "ci95": _interval(samples, confidence)} if samples else _unavailable("no resamples")
    out.update(overall_ci95=results["overall"]["ci95"], overall_status=results["overall"]["status"])
    if "reason" in results["overall"]:
        out["overall_reason"] = results["overall"]["reason"]
    for d in decks:
        out["decks"][d].update(status=results[d]["status"], ci95=results[d]["ci95"])
        if "reason" in results[d]:
            out["decks"][d]["reason"] = results[d]["reason"]
    out["replicates"], out["confidence"], out["seed"] = replicates, confidence, seed
    return out


def categories(values, meta):
    """Descriptive per-category means; never an interval."""
    out = {}
    for p, v in values.items():
        out.setdefault(meta[p]["category"], []).append(v)
    return {c: {"puzzles": len(v), "mean": sum(v) / len(v)} for c, v in sorted(out.items())}


def claim(per_cell_ci, mean_ci, margin=MARGIN):
    """The release gate: stronger / regression / inconclusive."""
    intervals = list(per_cell_ci.values()) + [mean_ci]
    if not per_cell_ci or any(ci is None for ci in intervals):
        return "inconclusive"
    if any(ci[1] < -margin for ci in per_cell_ci.values()) or mean_ci[1] < 0:
        return "regression"
    if mean_ci[0] > 0 and all(ci[0] > -margin for ci in per_cell_ci.values()):
        return "stronger"
    return "inconclusive"


def flags(value):
    """Saturated or floor cells cannot carry a headline claim."""
    if value is None:
        return None
    return "saturated" if value >= SATURATED else "floor" if value <= FLOOR else None


def blocks_for_effect(var_d, delta):
    """Blocks needed to detect a mean paired difference delta at 80% power, alpha .05."""
    require(delta > 0 and var_d >= 0, "blocks_for_effect", "positive effect required")
    return math.ceil(round((Z_ALPHA + Z_POWER) ** 2 * var_d / delta**2, 9))


def variance(values):
    """Sample variance of block differences (n - 1 denominator)."""
    require(len(values) > 1, "variance", "at least two blocks required")
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def seeds(result):
    """Bootstrap settings recorded in a result: game and puzzle seeds per mode."""
    b = result["bootstrap"]
    return {m: {"games": bootstrap_seed(b["stream"], b["suite"], m), "puzzles": bootstrap_seed(b["stream"], b["suite"] + "/puzzles", m)}
            for m in result["modes"]}


def aggregate(result):
    """Per-mode aggregates recomputed from the raw rows; all null unless the run is complete."""
    empty = {"fair": None, "cells": None, "game_score": None, "game_ci95": None, "game_interval": None,
             "puzzle_success": None, "puzzle_ci95": None, "puzzles": None}
    if result["status"] != "complete":
        return {m: dict(empty) for m in result["modes"]}
    b = result["bootstrap"]
    full = result["panel"] == "full"
    blocks = block_scores(result["game_rows"], result["planned_game_rows"]) if result["planned_game_rows"] else {}
    seed = seeds(result)
    out = {}
    for mode in result["modes"]:
        a = dict(empty, fair=result["candidate"]["information_contract"] == FAIR)
        if mode in blocks:
            iv = game_intervals(blocks[mode], b["replicates"], seed[mode]["games"], b["confidence"])
            cells = {}
            for cell in sorted(blocks[mode]):
                rows = [r for r in result["game_rows"] if r["mode"] == mode and r["cell"] == cell]
                outcome = Counter("draw" if r["winner"] is None else "win" if r["winner"] == r["learner_seat"] else "loss" for r in rows)
                cells[cell] = {"games": len(rows), "wins": outcome["win"], "draws": outcome["draw"], "losses": outcome["loss"],
                               "score": iv["cells"][cell]["mean"], "ci95": iv["cells"][cell]["ci95"], "flag": flags(iv["cells"][cell]["mean"])}
            a["cells"] = cells
            a["game_interval"] = {k: iv[k] for k in ("status", "blocks", "replicates", "confidence", "seed") if k in iv}
            if "reason" in iv:
                a["game_interval"]["reason"] = iv["reason"]
            if full:
                a["game_score"], a["game_ci95"] = iv["mean"]["mean"], iv["mean"]["ci95"]
        planned = [p for p in result["planned_puzzle_rows"] if p["mode"] == mode]
        if planned:
            rows = [r for r in result["puzzle_rows"] if r["mode"] == mode]
            values = puzzle_values(rows)
            meta = puzzle_meta(planned)
            require(set(values) == set(meta), "puzzle_rows", "missing puzzles")
            iv = puzzle_intervals(values, meta, b["replicates"], seed[mode]["puzzles"], b["confidence"])
            a["puzzle_success"], a["puzzle_ci95"] = iv["mean"], iv["overall_ci95"]
            iv["categories"] = categories(values, meta)
            a["puzzles"] = iv
        out[mode] = a
    return out


def compare(baseline, candidate):
    """Paired comparison of two complete, compatible results (see release.compare_results)."""
    require(baseline["modes"] == candidate["modes"] and baseline["selected_cells"] == candidate["selected_cells"], "compare", "modes/cells differ")
    b = baseline["bootstrap"]
    plan = baseline["planned_game_rows"]
    diffs = paired_game_differences(baseline["game_rows"], candidate["game_rows"], plan)
    out = {}
    for mode in baseline["modes"]:
        entry = {}
        if mode in diffs:
            seed = bootstrap_seed(b["stream"], b["suite"] + "/compare", mode)
            iv = game_intervals(diffs[mode], b["replicates"], seed, b["confidence"])
            a, c = block_scores(baseline["game_rows"], plan)[mode], block_scores(candidate["game_rows"], plan)[mode]
            entry["cells"] = {cell: {"baseline": sum(a[cell]) / len(a[cell]), "candidate": sum(c[cell]) / len(c[cell]),
                                     "difference": iv["cells"][cell]["mean"], "ci95": iv["cells"][cell]["ci95"],
                                     "flags": {"baseline": flags(sum(a[cell]) / len(a[cell])), "candidate": flags(sum(c[cell]) / len(c[cell]))},
                                     "variance": variance(diffs[mode][cell]) if iv["blocks"] > 1 else None} for cell in sorted(diffs[mode])}
            entry["mean_difference"], entry["mean_ci95"] = iv["mean"]["mean"], iv["mean"]["ci95"]
            entry["interval"] = {k: iv[k] for k in ("status", "blocks", "replicates", "confidence", "seed")}
            entry["claim"] = claim({c: v["ci95"] for c, v in entry["cells"].items()}, iv["mean"]["ci95"])
            entry["saturated_cells"] = sorted(c for c, v in entry["cells"].items() if v["flags"]["baseline"] or v["flags"]["candidate"])
        planned = [p for p in baseline["planned_puzzle_rows"] if p["mode"] == mode]
        if planned:
            meta = puzzle_meta(planned)
            va = puzzle_values([r for r in baseline["puzzle_rows"] if r["mode"] == mode])
            vb = puzzle_values([r for r in candidate["puzzle_rows"] if r["mode"] == mode])
            seed = bootstrap_seed(b["stream"], b["suite"] + "/compare/puzzles", mode)
            entry["puzzles"] = puzzle_intervals(paired_puzzle_differences(va, vb), meta, b["replicates"], seed, b["confidence"])
        out[mode] = entry
    return out
