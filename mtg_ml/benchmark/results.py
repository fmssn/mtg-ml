"""Result and comparison bookkeeping, exclusive writes, and aggregates verified against the raw rows."""

from collections import Counter
import platform
import sys
import math
import os
from pathlib import Path
import copy
import json
import importlib.metadata

from .artifacts import FAIR, DIAGNOSTIC, DECKS, MODES, CELLS, header, integer, identifier, sha, require, load, reference, load_manifest
from .jsonio import canonical_bytes, write_json
from .schedule import episodes, puzzle_plan
from . import stats


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
    validate_candidate(d["candidate"])
    sha(d.get("code_revision"), "code_revision", True)
    require(d.get("engine") in {"python", "native"}, "engine", "unsupported engine")
    if d["engine"] == "native":
        sha(d.get("native_build_revision"), "native_build_revision", True)
    require(isinstance(d.get("runtime"), dict) and all(d["runtime"].get(k) for k in ("python", "platform", "machine")), "runtime", "runtime/hardware provenance required")
    require(isinstance(d["runtime"].get("command"), list) and bool(d["runtime"]["command"]) and all(isinstance(a, str) for a in d["runtime"]["command"]), "runtime.command", "argv required")
    require(isinstance(d["runtime"].get("dependencies"), dict), "runtime.dependencies", "dependency versions required")
    require(d["engine"] != "native" or isinstance(d["runtime"].get("native_build"), dict), "runtime.native_build", "native build identity required")
    b = d.get("bootstrap")
    require(isinstance(b, dict) and set(b) == {"replicates", "confidence", "stream", "suite"}, "bootstrap", "recorded bootstrap settings required")
    integer(b["replicates"], "bootstrap.replicates")
    require(type(b["confidence"]) in (float, int) and 0 < b["confidence"] < 1, "bootstrap.confidence", "invalid confidence")
    identifier(b["stream"], "bootstrap.stream")
    identifier(b["suite"], "bootstrap.suite")
    require(d.get("interval_method") == stats.INTERVAL_METHOD and d.get("statistics") == stats.CONSTANTS, "interval_method", "unsupported interval method or constants")
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
    require(set(d["aggregate"]) == set(d["modes"]), "aggregate", "mode mismatch")
    # Every reported score and interval is recomputed from the raw rows and must match exactly;
    # an incomplete or invalid run reports nothing.
    recomputed = json.loads(canonical_bytes(stats.aggregate(d)))
    require(d["aggregate"] == recomputed, "aggregate", "reported scores/intervals do not match the raw rows")


def validate_candidate(candidate):
    """checkpoint (the default, for records without `kind`), agent or calibration oracle."""
    require(isinstance(candidate, dict), "candidate", "object required")
    kind = candidate.get("kind", "checkpoint")
    require(candidate.get("information_contract") in {FAIR, DIAGNOSTIC}, "candidate.information_contract", "unsupported contract")
    if kind == "checkpoint":
        identifier(candidate.get("checkpoint"), "candidate.checkpoint")
        sha(candidate.get("sha256"), "candidate.sha256")
        integer(candidate.get("features"), "candidate.features")
        from ..encode import information_contract
        actual = information_contract(candidate["features"])
        require(candidate.get("recorded_information_contract") == actual, "candidate.recorded_information_contract", "encoder contract mismatch")
        require(candidate["information_contract"] != FAIR or actual == "hidden_list", "candidate", "privileged features cannot enter fair result")
    elif kind == "agent":
        for k in ("id", "deck"):
            identifier(candidate.get(k), "candidate." + k)
        require(candidate["deck"] in DECKS, "candidate.deck", "unknown deck")
        sha(candidate.get("source_revision"), "candidate.source_revision", True)
        integer(candidate.get("rules_revision"), "candidate.rules_revision")
        sha(candidate.get("parameters_sha256"), "candidate.parameters_sha256")
        require(candidate.get("agent_kind") in {"specialist", "legacy", "synthetic"}, "candidate.agent_kind", "unsupported agent kind")
        require(candidate["agent_kind"] != "legacy" or candidate["information_contract"] == DIAGNOSTIC, "candidate", "legacy agents are diagnostic")
    elif kind == "calibration":
        identifier(candidate.get("id"), "candidate.id")
        require(candidate["information_contract"] == DIAGNOSTIC, "candidate", "calibration oracles are diagnostic")
    else:
        require(False, "candidate.kind", "unsupported candidate kind")


def bootstrap_settings(manifest):
    d = manifest.data
    return {"replicates": d["bootstrap"]["replicates"], "confidence": d["bootstrap"]["confidence"], "stream": d["stream"], "suite": d["id"]}


def candidate_name(candidate):
    kind = candidate.get("kind", "checkpoint")
    return candidate["checkpoint"] if kind == "checkpoint" else candidate["id"]


def dependencies():
    out = {}
    for name in ("torch", "numpy", "mtg_ml_native"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_result(manifest, candidate, game_rows, puzzle_rows=(), *, engine="python", code_revision,
                 native_build_revision=None, cells=None, modes=None, planned_puzzles=None, invalid=False, workers=1, runtime=None,
                 command=None, native_build=None):
    """Derive planned cases; the later tactical runner supplies actual rows."""
    from dataclasses import asdict
    from ..encode import information_contract
    candidate = dict(candidate)
    if candidate.get("kind", "checkpoint") == "checkpoint":
        candidate.setdefault("recorded_information_contract", information_contract(candidate["features"]))
    integer(workers, "workers")
    runtime = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "hostname": platform.node(),
               "processor": platform.processor(), "cpu_count": os.cpu_count(), "workers": workers,
               "torch_threads": sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None,
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "command": list(sys.argv if command is None else command),
               "dependencies": dependencies(), "native_build": native_build, **(runtime or {})}
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
             partial_cells=summaries, interval_method=stats.INTERVAL_METHOD, statistics=stats.CONSTANTS,
             bootstrap=bootstrap_settings(manifest))
    d["aggregate"] = json.loads(canonical_bytes(stats.aggregate(d)))
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
    from dataclasses import asdict
    specs = episodes(manifest, data["selected_cells"], data["modes"])
    planned = [{k: v for k, v in asdict(s).items() if k != "decks"} for s in specs]
    require(data["planned_game_rows"] == json.loads(canonical_bytes(planned)), "planned_game_rows", "manifest schedule mismatch")
    require(data["planned_puzzle_rows"] == puzzle_plan(manifest, data["selected_cells"], data["modes"]), "planned_puzzle_rows", "manifest cases mismatch")
    require(data["settings"] == manifest.data["engine_settings"], "settings", "manifest mismatch")
    require(data["bootstrap"] == bootstrap_settings(manifest), "bootstrap", "manifest mismatch")
    full = set(data["selected_cells"]) == set(CELLS)
    require(data["panel"] == ("full" if full else "partial"), "panel", "selection mismatch")


def _latency(data):
    out = {}
    for mode in data["modes"]:
        values = sorted(v for r in data["game_rows"] if r["mode"] == mode for v in r.get("latency_seconds", []))
        out[mode] = {"decisions": len(values),
                     "median": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2 if values else None,
                     "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else None}
    return out


def _machine(data):
    r = data["runtime"]
    return {k: r.get(k) for k in ("machine", "processor", "workers")}


def comparison_body(baseline, candidate):
    """Everything in a comparison that derives from the two results; refuses incompatible pairs."""
    a, b = baseline.data, candidate.data
    for name, d in (("baseline", a), ("candidate", b)):
        require(d["status"] == "complete", name, "only complete results can be compared")
    require(a["manifest"]["sha256"] == b["manifest"]["sha256"], "manifest", "different manifests")
    for k in ("selected_cells", "modes", "engine", "code_revision", "native_build_revision", "panel", "settings", "bootstrap"):
        require(a[k] == b[k], k, f"results differ in {k}")
    if a["engine"] == "native":
        require(a["runtime"]["native_build"]["source_sha256"] == b["runtime"]["native_build"]["source_sha256"], "native_build", "different native builds")
    fair = a["candidate"]["information_contract"] == FAIR
    require(fair == (b["candidate"]["information_contract"] == FAIR), "information_contract", "fair and diagnostic results cannot be compared")
    return dict(engine=a["engine"], code_revision=a["code_revision"], native_build_revision=a["native_build_revision"], modes=a["modes"],
                selected_cells=a["selected_cells"], panel=a["panel"], fair=fair, statistics=stats.CONSTANTS, interval_method=stats.INTERVAL_METHOD,
                bootstrap=a["bootstrap"], results=json.loads(canonical_bytes(stats.compare(a, b))),
                latency={"baseline": _latency(a), "candidate": _latency(b), "machines": {"baseline": _machine(a), "candidate": _machine(b)},
                         "latency_comparable": _machine(a) == _machine(b)})


def build_comparison(baseline, candidate, out):
    out = Path(out).resolve()
    d = dict(format="BenchmarkComparison", version=1, manifest={"sha256": baseline.data["manifest"]["sha256"], "id": baseline.data["bootstrap"]["suite"]},
             **{name: {"path": os.path.relpath(art.path, out.parent), "sha256": art.sha256, "candidate": candidate_name(art.data["candidate"])}
                for name, art in (("baseline", baseline), ("candidate", candidate))},
             **comparison_body(baseline, candidate),
             runtime={"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "command": list(sys.argv),
                      "dependencies": dependencies()})
    return json.loads(canonical_bytes(d))


def validate_comparison(d):
    header(d, "BenchmarkComparison")
    for k in ("baseline", "candidate"):
        identifier(d[k].get("path"), k + ".path")
        sha(d[k].get("sha256"), k + ".sha256")
    sha(d["manifest"].get("sha256"), "manifest.sha256")
    require(set(d["results"]) == set(d["modes"]) and d["interval_method"] == stats.INTERVAL_METHOD and d["statistics"] == stats.CONSTANTS, "results", "mode/method mismatch")
    for mode, entry in d["results"].items():
        require(entry.get("claim") in {"stronger", "regression", "inconclusive", None}, "results.claim", "unsupported claim")
    require(type(d["latency"]["latency_comparable"]) is bool, "latency", "comparability flag required")


def load_comparison(path):
    """Reloads both results (re-verifying their aggregates) and recomputes the whole comparison."""
    artifact = load(path, validate_comparison)
    results = {k: load_result(reference(artifact.path, artifact.data[k], k)) for k in ("baseline", "candidate")}
    try:
        body = json.loads(canonical_bytes(comparison_body(results["baseline"], results["candidate"])))
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{artifact.path}: {e}") from e
    stored = {k: artifact.data[k] for k in body}
    if stored != body:
        raise ValueError(f"{artifact.path}: comparison does not match its results ({[k for k in body if stored[k] != body[k]]})")
    return artifact


def write_comparison(path, data):
    path = Path(path).resolve()
    validate_comparison(data)
    write_json(path, data)
    return data
