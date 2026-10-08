"""Result bookkeeping and exclusive writes; interval estimation comes later."""

from collections import Counter
import platform
import sys
import math
import os
from pathlib import Path
import copy

from .artifacts import FAIR, DIAGNOSTIC, MODES, CELLS, header, integer, identifier, sha, require, load, reference, load_manifest
from .jsonio import write_json
from .schedule import episodes, puzzle_plan


def game_identity(r):
    return r["mode"], r["cell"], r["block"], r["slot"]


def puzzle_identity(r):
    return r["mode"], r["puzzle"], r["case"], r["repetition"]


def _game_summary(rows):
    valid = [r for r in rows if r["status"] == "completed"]
    outcomes = Counter("draw" if r["winner"] is None else "win" if r["winner"] == r["learner_seat"] else "loss" for r in valid)
    latency = sorted(v for r in valid for v in r.get("latency_seconds", []))
    n = len(valid)
    return {"games": n, "errors": len(rows) - n, "wins": outcomes["win"], "draws": outcomes["draw"], "losses": outcomes["loss"],
            "score": (outcomes["win"] + 0.5 * outcomes["draw"]) / n if n else None,
            "termination_reasons": dict(Counter(r["reason"] for r in valid)),
            "latency_median": (latency[(len(latency) - 1) // 2] + latency[len(latency) // 2]) / 2 if latency else None,
            "latency_p95": latency[min(len(latency) - 1, math.ceil(0.95 * len(latency)) - 1)] if latency else None}


def validate_result(d):
    header(d, "BenchmarkResult")
    require(d.get("status") in {"complete", "incomplete", "invalid"}, "status", "unsupported run status")
    for refname in ("manifest",):
        identifier(d[refname].get("path"), refname + ".path")
        sha(d[refname].get("sha256"), refname + ".sha256")
    candidate = d["candidate"]
    identifier(candidate.get("checkpoint"), "candidate.checkpoint")
    sha(candidate.get("sha256"), "candidate.sha256")
    integer(candidate.get("features"), "candidate.features")
    require(candidate.get("information_contract") in {FAIR, DIAGNOSTIC}, "candidate.information_contract", "unsupported contract")
    from ..encode import information_contract
    actual = information_contract(candidate["features"])
    require(candidate.get("recorded_information_contract") == actual, "candidate.recorded_information_contract", "encoder contract mismatch")
    require(candidate["information_contract"] != FAIR or actual == "hidden_list", "candidate", "privileged features cannot enter fair result")
    sha(d.get("code_revision"), "code_revision", True)
    require(d.get("engine") in {"python", "native"}, "engine", "unsupported engine")
    if d["engine"] == "native":
        sha(d.get("native_build_revision"), "native_build_revision", True)
    require(isinstance(d.get("runtime"), dict) and all(d["runtime"].get(k) for k in ("python", "platform", "machine")), "runtime", "runtime/hardware provenance required")
    require(d.get("panel") in {"full", "partial"}, "panel", "panel required")
    require(isinstance(d.get("settings"), dict), "settings", "settings required")
    require(isinstance(d.get("game_rows"), list) and isinstance(d.get("puzzle_rows"), list), "rows", "game/case rows required")
    planned_games = {game_identity(r): r for r in d["planned_game_rows"]}
    expected_games = set(planned_games)
    expected_puzzles = {puzzle_identity(r) for r in d["planned_puzzle_rows"]}
    require(len(expected_games) == len(d["planned_game_rows"]) and len(expected_puzzles) == len(d["planned_puzzle_rows"]), "planned_rows", "duplicate identity")
    seen_games, seen_puzzles = set(), set()
    for r in d["game_rows"]:
        key = game_identity(r)
        require(key in expected_games and key not in seen_games, "game_rows", "unexpected/duplicate identity")
        seen_games.add(key)
        require(r["mode"] in MODES and r["cell"] in CELLS and r["status"] in {"completed", "error"}, "game_rows", "invalid row")
        integer(r["block"], "game_rows.block", 0)
        integer(r["slot"], "game_rows.slot", 0, 3)
        integer(r["simulator_seed"], "game_rows.simulator_seed", 0, 2**31 - 1)
        integer(r["actor_seed"], "game_rows.actor_seed", 0, 2**31 - 1)
        require(type(r["learner_seat"]) is int and r["learner_seat"] in (0, 1), "game_rows.learner_seat", "invalid seat")
        require(r["winner"] is None or type(r["winner"]) is int and r["winner"] in (0, 1), "game_rows.winner", "invalid winner")
        integer(r["decisions"], "game_rows.decisions", 0)
        integer(r["turns"], "game_rows.turns", 0)
        identifier(r.get("reason"), "game_rows.reason")
        planned = planned_games[key]
        for k in ("simulator_seed", "actor_seed", "learner_seat", "starting_player", "deck_ids"):
            require(r[k] == planned[k], "game_rows." + k, "planned mapping/seed mismatch")
    for r in d["puzzle_rows"]:
        key = puzzle_identity(r)
        require(key in expected_puzzles and key not in seen_puzzles, "puzzle_rows", "unexpected/duplicate identity")
        seen_puzzles.add(key)
        require(r["status"] in {"success", "failure", "error"}, "puzzle_rows.status", "invalid status")
        integer(r["repetition"], "puzzle_rows.repetition", 0)
        integer(r["decisions"], "puzzle_rows.decisions", 0, 64)
        planned = next(p for p in d["planned_puzzle_rows"] if puzzle_identity(p) == key)
        require(all(r.get(k) == v for k, v in planned.items()), "puzzle_rows", "planned case/seed mismatch")
        require(type(r.get("objective_met")) is bool, "puzzle_rows.objective_met", "boolean required")
        require(r["status"] != "success" or r["objective_met"], "puzzle_rows", "success requires objective")
    require(set(d["counts"]) == set(d["modes"]), "counts", "mode mismatch")
    for mode, c in d["counts"].items():
        rows = [r for r in d["game_rows"] if r["mode"] == mode]
        require(c["planned_games"] == sum(p[0] == mode for p in expected_games), "counts", "planned game mismatch")
        require(c["attempted_games"] == len(rows) and c["completed_games"] == sum(r["status"] == "completed" for r in rows), "counts", "completed game mismatch")
        require(c["technical_game_errors"] == sum(r["status"] == "error" for r in rows), "counts", "game error count mismatch")
        puzzles = {(r["puzzle"], r["repetition"]) for r in d["planned_puzzle_rows"] if r["mode"] == mode}
        completed = sum(all(any(puzzle_identity(r) == puzzle_identity(p) and r["status"] != "error" for r in d["puzzle_rows"])
                            for p in d["planned_puzzle_rows"] if p["mode"] == mode and (p["puzzle"], p["repetition"]) == key) for key in puzzles)
        require(c["planned_puzzles"] == len(puzzles) and c["completed_puzzles"] == completed, "counts", "puzzle count mismatch")
        require(c["technical_case_errors"] == sum(r["mode"] == mode and r["status"] == "error" for r in d["puzzle_rows"]), "counts", "case error count mismatch")
    all_done = seen_games == expected_games and seen_puzzles == expected_puzzles and all(r["status"] == "completed" for r in d["game_rows"]) and all(r["status"] != "error" for r in d["puzzle_rows"])
    require(d["status"] != "complete" or all_done, "status", "technical errors or missing rows cannot be complete")
    eligible = d["status"] == "complete" and d["panel"] == "full" and candidate["information_contract"] == FAIR
    for aggregate in d["aggregate"].values():
        require(aggregate.get("ci95") is None, "aggregate.ci95", "intervals not implemented in foundation")
        if not eligible:
            require(all(v is None for v in aggregate.values()), "aggregate", "partial/privileged/invalid/incomplete primary scores must be null")


def build_result(manifest, candidate, game_rows, puzzle_rows=(), *, engine="python", code_revision,
                 native_build_revision=None, cells=None, modes=None, planned_puzzles=None, invalid=False, workers=1, runtime=None):
    """Derive planned cases; the later tactical runner supplies actual rows."""
    from dataclasses import asdict
    from .jsonio import canonical_bytes
    import json
    from ..encode import information_contract
    candidate = dict(candidate)
    candidate.setdefault("recorded_information_contract", information_contract(candidate["features"]))
    integer(workers, "workers")
    runtime = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "hostname": platform.node(),
               "processor": platform.processor(), "cpu_count": os.cpu_count(), "workers": workers,
               "torch_threads": sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None,
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), **(runtime or {})}
    specs = episodes(manifest, cells, modes)
    expected_puzzles = puzzle_plan(manifest, cells, modes)
    require(planned_puzzles is None or list(planned_puzzles) == expected_puzzles, "planned_puzzles", "manifest plan mismatch")
    planned_puzzles = expected_puzzles
    planned = [{k: v for k, v in asdict(s).items() if k != "decks"} for s in specs]
    # Normalize tuples to the JSON representation also used by loaded rows.
    planned, game_rows, puzzle_rows = json.loads(canonical_bytes([planned, list(game_rows), list(puzzle_rows)]))
    game_rows.sort(key=game_identity)
    puzzle_rows.sort(key=puzzle_identity)
    modes = list(manifest.data["modes"] if modes is None else modes)
    counts, summaries = {}, []
    for mode in modes:
        rows = [r for r in game_rows if r["mode"] == mode]
        puzzles = {(r["puzzle"], r["repetition"]) for r in planned_puzzles if r["mode"] == mode}
        completed = sum(all(any(puzzle_identity(r) == puzzle_identity(p) and r["status"] != "error" for r in puzzle_rows)
                            for p in planned_puzzles if p["mode"] == mode and (p["puzzle"], p["repetition"]) == key) for key in puzzles)
        counts[mode] = dict(planned_games=sum(s.mode == mode for s in specs), attempted_games=len(rows),
                            completed_games=sum(r["status"] == "completed" for r in rows), technical_game_errors=sum(r["status"] == "error" for r in rows),
                            planned_puzzles=len(puzzles), completed_puzzles=completed,
                            technical_case_errors=sum(r["mode"] == mode and r["status"] == "error" for r in puzzle_rows))
        for cell in sorted({s.cell for s in specs}):
            group = [r for r in rows if r["cell"] == cell]
            summaries.append(dict(mode=mode, cell=cell, **_game_summary(group)))
            for seat in (0, 1):
                for play in (True, False):
                    subset = [r for r in group if r["learner_seat"] == seat and (r["starting_player"] == seat) == play]
                    summaries.append(dict(mode=mode, cell=cell, seat=seat, play=play, **_game_summary(subset)))
    done = len(game_rows) == len(specs) and all(r["status"] == "completed" for r in game_rows) and len(puzzle_rows) == len(planned_puzzles) and all(r["status"] != "error" for r in puzzle_rows)
    d = dict(format="BenchmarkResult", version=1, manifest={"path": str(manifest.path), "sha256": manifest.sha256}, candidate=candidate,
             engine=engine, code_revision=code_revision, native_build_revision=native_build_revision,
             runtime=runtime,
             settings=manifest.data["engine_settings"], modes=modes, selected_cells=sorted({s.cell for s in specs}),
             panel="full" if {s.cell for s in specs} == set(CELLS) else "partial",
             status="invalid" if invalid else "complete" if done else "incomplete", counts=counts,
             planned_game_rows=planned, planned_puzzle_rows=list(planned_puzzles), game_rows=game_rows, puzzle_rows=puzzle_rows,
             partial_cells=summaries, interval_method="unavailable-foundation", aggregate={m: {"game_score": None, "puzzle_success": None, "ci95": None} for m in modes})
    validate_result(d)
    return d


def load_result(path):
    artifact = load(path, validate_result)
    manifest = load_manifest(reference(artifact.path, artifact.data["manifest"], "manifest"))
    _validate_plan(artifact.data, manifest)
    for row in artifact.data["game_rows"] + artifact.data["puzzle_rows"]:
        if "replay_reference" in row:
            reference(artifact.path, row["replay_reference"], "replay_reference")
    return artifact


def write_result(path, data):
    stored = copy.deepcopy(data)
    path = Path(path).resolve()
    manifest_path = Path(stored["manifest"]["path"])
    if manifest_path.is_absolute():
        stored["manifest"]["path"] = os.path.relpath(manifest_path, path.parent)
    manifest = load_manifest(reference(path, stored["manifest"], "manifest"))
    _validate_plan(stored, manifest)
    validate_result(stored)
    for row in stored["game_rows"] + stored["puzzle_rows"]:
        if "replay_reference" in row:
            reference(path, row["replay_reference"], "replay_reference")
    write_json(path, stored)
    return stored


def _validate_plan(data, manifest):
    import json
    from dataclasses import asdict
    from .jsonio import canonical_bytes
    specs = episodes(manifest, data["selected_cells"], data["modes"])
    planned = [{k: v for k, v in asdict(s).items() if k != "decks"} for s in specs]
    require(data["planned_game_rows"] == json.loads(canonical_bytes(planned)), "planned_game_rows", "manifest schedule mismatch")
    require(data["planned_puzzle_rows"] == puzzle_plan(manifest, data["selected_cells"], data["modes"]), "planned_puzzle_rows", "manifest cases mismatch")
    require(data["settings"] == manifest.data["engine_settings"], "settings", "manifest mismatch")
    full = set(data["selected_cells"]) == set(CELLS)
    require(data["panel"] == ("full" if full else "partial"), "panel", "selection mismatch")
