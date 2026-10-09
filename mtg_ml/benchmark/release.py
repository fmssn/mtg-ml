"""Release tooling behind `python -m mtg_ml.benchmark`: run, compare, turn-limit sensitivity
and calibration controls. Statistics live in stats.py; every result is re-verified on load.

Games run in chunks of one mode x cell x 100 blocks, each written exclusively to
`<out>.parts/`. A technical error stops the run after its chunk and leaves an
`incomplete` result; error rows are never retried, replaced or scored.
"""

from dataclasses import dataclass
from functools import partial
import copy
import os
from pathlib import Path
import platform
import sys

from .adapters import CheckpointAdapter, LegacyAdapter, ScriptedAdapter
from .artifacts import DIAGNOSTIC, FAIR, load_manifest, reference, require
from .jsonio import digest, file_digest, read_json, write_json
from .results import (build_comparison, build_result, dependencies, load_result, write_comparison, write_result)
from .runner import run_episodes
from .schedule import actor_seed, bootstrap_seed, episodes
from . import stats
from .tactics import SPECIALISTS, ResponseFactory, SelectorFactory, pass_learner, run_puzzles, specialist_registry
from .validation import ROOT, validate

CHUNK_BLOCKS = 100
LEGACY = {"legacy-jund": "jund_wildfire", "legacy-blue": "mono_blue_terror"}
TIMING = {"elapsed_seconds", "latency_seconds"}


class LegacyFactory:
    """Picklable learner/opponent factory for the legacy scripted bots (diagnostic)."""

    def __init__(self, deck):
        self.deck = deck

    def __call__(self, seat, mode):
        from ..bots import make_bot
        return LegacyAdapter(partial(make_bot, deck=self.deck), seat)

    def __eq__(self, other):
        return isinstance(other, LegacyFactory) and other.deck == self.deck

    def __hash__(self):
        return hash(self.deck)


class CheckpointLearner:
    """Picklable factory; one adapter per worker process, seat and mode."""
    _cache = {}

    def __init__(self, path, contract):
        self.path, self.contract = path, contract

    def __call__(self, seat, mode):
        key = self.path, self.contract, seat, mode
        if key not in self._cache:
            self._cache[key] = CheckpointAdapter(self.path, seat, mode, self.contract)
        return self._cache[key]


@dataclass(frozen=True)
class Learner:
    """What is evaluated: the result's candidate record plus its game and puzzle factories.
    `cells` restricts a scripted agent to its own deck; `learner_for` picks puzzle learners per case."""
    record: dict
    games: object
    puzzles: object
    cells: tuple | None = None
    learner_for: object = None


def checkpoint_learner(path, contract=FAIR):
    adapter = CheckpointAdapter(path, 0, "greedy", contract)  # admission check before any attempt
    factory = CheckpointLearner(str(path), contract)
    record = dict(kind="checkpoint", checkpoint=str(path), sha256=adapter.sha256, features=adapter.features, information_contract=contract)
    return Learner(record, factory, factory)


def agent_learner(name, manifest, registry):
    """A registered specialist or a legacy bot, restricted to its own deck's cells."""
    revision = manifest.data["freeze"]["code_revision"]
    if name in LEGACY:
        deck = LEGACY[name]
        sources = {p.name: file_digest(p) for p in sorted((ROOT / "mtg_ml" / "bots").glob("*.py"))}
        record = dict(kind="agent", id=name, deck=deck, source_revision=revision, rules_revision=1, agent_kind="legacy",
                      parameters_sha256=digest({"factory": "mtg_ml.bots.make_bot", "deck": deck, "sources": sources}), information_contract=DIAGNOSTIC)
        factory = LegacyFactory(deck)
    else:
        require(name in SPECIALISTS, "agent", f"unknown agent {name}; use one of {list(SPECIALISTS) + list(LEGACY)}")
        m = registry.metadata(name)
        record = dict(kind="agent", id=name, deck=m.deck, source_revision=m.source_revision, rules_revision=m.rules_revision,
                      agent_kind=m.kind, parameters_sha256=m.parameters_sha256, information_contract=m.information_contract)
        factory = ResponseFactory(registry, name)
    own = tuple(c["id"] for c in manifest.data["cells"] if c["learner_deck"] == record["deck"])
    return Learner(record, {record["deck"]: factory}, factory, own)


def _rel(path, parent):
    return os.path.relpath(Path(path).resolve(), Path(parent).resolve())


def _chunks(specs):
    out = {}
    for s in specs:
        out.setdefault((s.mode, s.cell, s.block // CHUNK_BLOCKS * CHUNK_BLOCKS), []).append(s)
    return dict(sorted(out.items()))


def _run_key(manifest, record, engine, revision, native):
    return digest({"manifest": manifest.sha256, "candidate": record, "engine": engine, "code_revision": revision, "native": native})


def play_games(manifest, learner, registry, *, engine, workers, cells, modes, parts, key, resume):
    """Rows for the planned games, chunk by chunk. Stops after the first chunk with an error.
    Returns (rows, resumed chunk names). Chunk files are exclusive and never rewritten."""
    parts = Path(parts)
    require(resume or not parts.exists(), "parts", f"{parts} exists; use --resume to reuse its clean chunks")
    specs = episodes(manifest, cells, modes)
    opponents = {b["id"]: ResponseFactory(registry, b["id"]) for b in manifest.data["bots"]}
    rows, resumed = [], []
    for (mode, cell, offset), group in _chunks(specs).items():
        path = parts / f"{mode}-{cell}-{offset}.json"
        identities = sorted([list(s.identity) for s in group])
        if path.exists():
            chunk = read_json(path)
            require(chunk.get("format") == "BenchmarkChunk" and chunk.get("run_key") == key, str(path), "chunk belongs to a different run")
            require(sorted([r["mode"], r["cell"], r["block"], r["slot"]] for r in chunk["rows"]) == identities, str(path), "chunk rows do not match the plan")
            if all(r["status"] == "completed" for r in chunk["rows"]):
                resumed.append(path.name)
        else:
            chunk = {"format": "BenchmarkChunk", "version": 1, "run_key": key, "mode": mode, "cell": cell, "offset": offset,
                     "rows": run_episodes(group, manifest.data["engine_settings"], learner.games, opponents, engine, workers)}
            write_json(path, chunk)
        rows += chunk["rows"]
        if any(r["status"] == "error" for r in chunk["rows"]):
            break
    return rows, resumed


def _failure_replays(manifest, learner, registry, rows, engine, out):
    """Replay each error episode with recording; the reference lands on the error row."""
    opponents = {b["id"]: ResponseFactory(registry, b["id"]) for b in manifest.data["bots"]}
    specs = {s.identity: s for s in episodes(manifest)}
    folder = out.parent / (out.stem + ".failures")
    for row in rows:
        if row["status"] != "error":
            continue
        key = (row["mode"], row["cell"], row["block"], row["slot"])
        replay = run_episodes([specs[key]], manifest.data["engine_settings"], learner.games, opponents, engine, 1, record=True)[0]
        path = folder / ("-".join(map(str, key)) + ".json")
        write_json(path, {"format": "BenchmarkFailureReplay", "version": 1, "identity": list(key),
                          "error": {k: replay.get(k) for k in ("reason", "error_type", "attempted_action", "decisions", "turns")},
                          "replay": replay.get("replay")})
        row["replay_reference"] = {"path": _rel(path, out.parent), "sha256": file_digest(path)}


def _admit(manifest_path, engines, report=None):
    """Validate the freeze and reviewed lines, unless a report for this manifest's content already exists."""
    report = validate(manifest_path, tuple(engines)) if report is None else report
    require(report["status"] == "complete", "validate", "; ".join(report["errors"]))
    manifest = load_manifest(manifest_path)
    require({b["id"] for b in manifest.data["bots"]} <= set(SPECIALISTS), "manifest", "a release manifest pins the two specialists")
    return manifest, specialist_registry(manifest.data["freeze"]["code_revision"]), report


def _select(learner, cells):
    if learner.cells is None:
        return cells
    cells = list(learner.cells if cells is None else cells)
    require(set(cells) <= set(learner.cells), "cells", f"{learner.record['id']} plays only cells {list(learner.cells)}")
    return cells


def run_benchmark(manifest_path, learner, *, engine="python", workers=1, cells=None, modes=None, out, resume=False, command=None, admitted=None):
    """Run games and puzzles, write one BenchmarkResult. The result is `incomplete` after any technical error.
    `admitted` is the validation report of an identical-content manifest (calibration reuses its own)."""
    out = Path(out).resolve()
    require(not out.exists(), "out", f"{out} exists; results are never overwritten")
    manifest, registry, report = _admit(manifest_path, (engine,), admitted)
    revision = manifest.data["freeze"]["code_revision"]
    cells = _select(learner, cells)
    native = report.get("native_build")
    key = _run_key(manifest, learner.record, engine, revision, native and native["source_sha256"])
    rows, resumed = play_games(manifest, learner, registry, engine=engine, workers=workers, cells=cells, modes=modes,
                               parts=out.parent / (out.stem + ".parts"), key=key, resume=resume)
    errors = [r for r in rows if r["status"] == "error"]
    puzzle_rows = []
    if errors:
        _failure_replays(manifest, learner, registry, rows, engine, out)
    elif len(rows) == len(episodes(manifest, cells, modes)):
        puzzle_rows = run_puzzles(manifest, learner.puzzles, registry, engine=engine, cells=cells, modes=modes, workers=workers, learner_for=learner.learner_for)
    result = build_result(manifest, learner.record, rows, puzzle_rows, engine=engine, code_revision=revision,
                          native_build_revision=revision if engine == "native" else None, cells=cells, modes=modes, workers=workers,
                          runtime={"resumed_chunks": resumed, "run_key": key}, command=command, native_build=native)
    return write_result(out, result)


def compare_results(baseline_path, candidate_path, out):
    """Paired comparison of two complete, compatible results; raises ValueError otherwise."""
    baseline, candidate = load_result(baseline_path), load_result(candidate_path)
    data = build_comparison(baseline, candidate, out)
    write_comparison(out, data)
    return data


def _variant(manifest, cap, folder):
    data = copy.deepcopy(manifest.data)
    data["engine_settings"]["max_turns"] = cap
    data["id"] = f"{manifest.data['id']}-t{cap}"
    data["puzzles"] = dict(data["puzzles"], path=_rel(reference(manifest.path, manifest.data["puzzles"], "puzzles"), folder))
    path = Path(folder) / f"manifest-t{cap}.json"
    write_json(path, data)
    return load_manifest(path)


def _cap_share(rows):
    return {mode: {cell: sum(r["reason"] == "turn limit" for r in rows if (r["mode"], r["cell"]) == (mode, cell)) /
                   sum((r["mode"], r["cell"]) == (mode, cell) for r in rows)
                   for cell in sorted({r["cell"] for r in rows if r["mode"] == mode})} for mode in sorted({r["mode"] for r in rows})}


def turn_limit_sensitivity(manifest_path, learner, caps=(100, 200, 400), margin=0.005, *, engine="python", workers=1, cells=None,
                           modes=None, out_dir, command=None):
    """Rerun the same seeded games under longer turn caps; the benchmark is only a faithful test
    of the game when the shorter cap's scores stay within +-margin of the longer ones."""
    folder = Path(out_dir).resolve()
    require(len(caps) >= 2 and list(caps) == sorted(set(caps)), "caps", "ascending distinct caps required")
    base, registry, report = _admit(manifest_path, (engine,))
    revision = base.data["freeze"]["code_revision"]
    cells = _select(learner, cells)
    native = report.get("native_build")
    runs, files = {}, {}
    for cap in caps:
        variant = _variant(base, cap, folder)
        key = _run_key(variant, learner.record, engine, revision, native and native["source_sha256"])
        rows, _ = play_games(variant, learner, registry, engine=engine, workers=workers, cells=cells, modes=modes,
                             parts=folder / f"t{cap}.parts", key=key, resume=False)
        path = folder / f"games-t{cap}.json"
        write_json(path, {"format": "BenchmarkGameRows", "version": 1, "run_key": key, "max_turns": cap, "rows": rows})
        runs[cap] = rows
        files[cap] = {"manifest": {"path": _rel(variant.path, folder), "sha256": variant.sha256}, "rows": {"path": _rel(path, folder), "sha256": file_digest(path)}}
        if any(r["status"] == "error" for r in rows):
            return _write_sensitivity(folder, base, learner, caps, margin, files, runs, None, engine, native, command, error=True)
    plan = [{"mode": s.mode, "cell": s.cell, "block": s.block, "slot": s.slot} for s in episodes(base, cells, modes)]
    return _write_sensitivity(folder, base, learner, caps, margin, files, runs, plan, engine, native, command)


def sensitivity_body(base_manifest, caps, margin, runs, plan):
    """Gate and shares recomputed from the stored rows."""
    b = base_manifest.data["bootstrap"]
    out = {"caps": {}, "comparisons": {}}
    for cap in caps:
        out["caps"][str(cap)] = {"turn_limit_draw_share": _cap_share(runs[cap])}
    passed = True
    for cap in caps[1:]:
        diffs = stats.paired_game_differences(runs[caps[0]], runs[cap], plan)
        entry = {}
        for mode, cells in diffs.items():
            iv = stats.game_intervals(cells, b["replicates"], bootstrap_seed(base_manifest.data["stream"], f"{base_manifest.data['id']}/turn-limit-t{cap}", mode), b["confidence"])
            cis = [c["ci95"] for c in iv["cells"].values()] + [iv["mean"]["ci95"]]
            ok = all(ci is not None and ci[0] >= -margin and ci[1] <= margin for ci in cis)
            passed = passed and ok
            entry[mode] = {"passed": ok, "cells": iv["cells"], "mean": iv["mean"], "interval": {k: iv[k] for k in ("status", "blocks", "seed")}}
        out["comparisons"][f"{caps[0]}-vs-{cap}"] = entry
    out["status"] = "passed" if passed else "restricted-game"
    return out


def _write_sensitivity(folder, base, learner, caps, margin, files, runs, plan, engine, native, command, error=False):
    data = {"format": "BenchmarkTurnLimitSensitivity", "version": 1, "manifest": {"path": _rel(base.path, folder), "sha256": base.sha256},
            "candidate": learner.record, "engine": engine, "native_build": native, "caps": list(caps), "margin": margin, "files": {str(k): v for k, v in files.items()},
            "statistics": stats.CONSTANTS, "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                                                      "command": list(sys.argv if command is None else command), "dependencies": dependencies()}}
    if error:
        data.update(status="incomplete", reason="technical errors; sensitivity is not evaluated")
    else:
        data.update(sensitivity_body(base, list(caps), margin, runs, plan))
    write_json(folder / "turn-limit.json", data)
    return data


def verify_sensitivity(path):
    """Recompute a sensitivity report from its stored rows."""
    path = Path(path)
    data = read_json(path)
    require(data.get("format") == "BenchmarkTurnLimitSensitivity", str(path), "unsupported format")
    if data["status"] == "incomplete":
        return data
    manifest = load_manifest(reference(path, data["manifest"], "manifest"))
    runs = {}
    for cap in data["caps"]:
        f = data["files"][str(cap)]
        reference(path, f["manifest"], "manifest")
        runs[cap] = read_json(reference(path, f["rows"], "rows"))["rows"]
    plan = [{k: r[k] for k in ("mode", "cell", "block", "slot")} for r in runs[data["caps"][0]]]
    body = sensitivity_body(manifest, data["caps"], data["margin"], runs, plan)
    require(all(data[k] == body[k] for k in body), str(path), "sensitivity does not match its rows")
    return data


# -- calibration -----------------------------------------------------------------------------

def witness_lines(case):
    return [w for w in case.get("witnesses", []) if w.get("learner_only")]


@dataclass(frozen=True)
class LinePlayer:
    """`learner_for` playing the index-th reviewed witness (clipped) or mistake of each case."""
    kind: str
    index: int

    def __call__(self, plan, puzzle, case):
        lines = witness_lines(case) if self.kind == "witness" else case.get("mistakes", [])
        if not lines or (self.kind == "mistake" and self.index >= len(lines)):
            return pass_learner
        return SelectorFactory(tuple(lines[min(self.index, len(lines) - 1)]["actions"]))


class FaultAdapter(ScriptedAdapter):
    """Calibration-only wrapper that misbehaves on demand; never registered."""

    def __init__(self, agent, seat, fault, armed):
        super().__init__(agent, seat)
        self.fault, self.armed, self.active = fault, armed, False

    def reset(self, own_deck, actor_seed=0):
        self.active = self.fault in {"raise", "illegal"} and actor_seed in self.armed
        super().reset(own_deck, actor_seed)

    def act(self, game):
        if self.fault == "raise" and self.active:
            raise RuntimeError("injected exception")
        if self.fault == "illegal" and self.active:
            return len(game.legal_options()) + 3
        if self.fault == "losing" and game.players[self.seat].life < game.players[1 - self.seat].life:
            raise RuntimeError("injected exception while behind on life")
        return super().act(game)


@dataclass(frozen=True)
class FaultFactory:
    inner: ResponseFactory
    fault: str
    armed: frozenset = frozenset()

    def __call__(self, seat, mode):
        adapter = self.inner(seat, mode)
        return FaultAdapter(adapter.agent, seat, self.fault, self.armed)


def specialist_panel(registry):
    return {m.deck: ResponseFactory(registry, m.id) for m in (registry.metadata(n) for n in SPECIALISTS)}


@dataclass(frozen=True)
class DeckSpecialists:
    """`learner_for`: each puzzle is attempted by the specialist of the puzzle's deck."""
    registry: object

    def __call__(self, plan, puzzle, case):
        deck = puzzle.data["deck"]
        return ResponseFactory(self.registry, next(n for n in SPECIALISTS if self.registry.metadata(n).deck == deck))


def calibration_record(name, contract=DIAGNOSTIC):
    return {"kind": "calibration", "id": name, "information_contract": contract}


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


def _rows_equal(a, b):
    return _strip(a) == _strip(b)


def _derived(manifest, folder, name, blocks):
    data = copy.deepcopy(manifest.data)
    data["blocks_per_cell"] = min(blocks, data["blocks_per_cell"])
    data["id"] = f"{data['id']}-{name}"
    data["puzzles"] = dict(data["puzzles"], path=_rel(reference(manifest.path, manifest.data["puzzles"], "puzzles"), folder))
    path = Path(folder) / f"manifest-{name}.json"
    write_json(path, data)
    return path


def _puzzle_runs(manifest, registry, engines, learner_for, count, learner=pass_learner):
    return {e: [run_puzzles(manifest, learner, registry, engine=e, modes=["greedy"], learner_for=learner_for(i)) for i in range(max(count, 1))] for e in engines}


def _scored(rows, problems, label):
    """Puzzle values for the rows without technical errors; each error becomes a problem, never a score."""
    for r in rows:
        if r["status"] == "error":
            problems.append(f"{label}: {r['puzzle']}/{r['case']} technical error {r['reason']}: {r.get('error', '')}")
    errored = {r["puzzle"] for r in rows if r["status"] == "error"}
    return stats.puzzle_values([r for r in rows if r["puzzle"] not in errored])


def _trace(rows):
    return [{k: r[k] for k in ("status", "reason", "decisions", "trace_sha256")} for r in rows]


def _agree(runs, engines):
    return all(_trace(runs[e]) == _trace(runs[engines[0]]) for e in engines)


def _check_lines(manifest, registry, engines):
    """The three tactical controls: reviewed witnesses pass, declared mistakes fail as declared,
    and passing through solves nothing, with identical traces on every engine."""
    from .tactics import _plan_puzzles
    puzzles = _plan_puzzles(manifest)
    cases = [(p.data["id"], c) for p in puzzles.values() for c in p.data["cases"]]
    n_witness = max([len(witness_lines(c)) for _, c in cases] + [0])
    n_mistake = max([len(c.get("mistakes", [])) for _, c in cases] + [0])
    witness = _puzzle_runs(manifest, registry, engines, lambda i: LinePlayer("witness", i), n_witness)
    mistake = _puzzle_runs(manifest, registry, engines, lambda i: LinePlayer("mistake", i), n_mistake)
    passing = {e: run_puzzles(manifest, pass_learner, registry, engine=e, modes=["greedy"]) for e in engines}
    w = {"id": "witnesses-pass", "problems": [f"{p}/{c['id']}: no learner-only witness" for p, c in cases if not witness_lines(c)],
         "witness_lines": sum(len(witness_lines(c)) for _, c in cases), "engines": list(engines)}
    for i in range(n_witness):
        for e in engines:
            w["problems"] += [f"witness {i} on {e}: {p} scores {v}, expected 1" for p, v in _scored(witness[e][i], w["problems"], f"witness {i} on {e}").items() if v != 1.0]
        if not _agree({e: witness[e][i] for e in engines}, engines):
            w["problems"].append(f"witness {i}: engines disagree")
    m = {"id": "mistakes-fail", "problems": [], "mistake_lines": sum(len(c.get("mistakes", [])) for _, c in cases), "engines": list(engines)}
    for j in range(n_mistake):
        declared = {(p, c["id"]): c["mistakes"][j]["expect"] for p, c in cases if j < len(c.get("mistakes", []))}
        for e in engines:
            for r in mistake[e][j]:
                want = declared.get((r["puzzle"], r["case"]))
                if want is not None and r["reason"] != want:
                    m["problems"].append(f"mistake {j} on {e}: {r['puzzle']}/{r['case']} classified {r['reason']}, expected {want}")
            scored = _scored([r for r in mistake[e][j] if any(k[0] == r["puzzle"] for k in declared)], m["problems"], f"mistake {j} on {e}")
            m["problems"] += [f"mistake {j} on {e}: {p} scores {v}, expected 0" for p, v in scored.items() if v != 0.0]
        if not _agree({e: mistake[e][j] for e in engines}, engines):
            m["problems"].append(f"mistake {j}: engines disagree")
    t = {"id": "passing-through-fails", "problems": [], "engines": list(engines), "reviewed_pass_lines": []}
    # Passing through must not solve a puzzle, unless passing is itself one of the case's reviewed
    # witnesses (a resource-preservation puzzle whose right answer is to do nothing).
    reviewed = {(r["puzzle"], r["case"]): set() for r in passing[engines[0]]}
    for run in witness[engines[0]][:n_witness]:
        for r in run:
            reviewed[(r["puzzle"], r["case"])].add(r["trace_sha256"])
    for e in engines:
        for p, v in _scored(passing[e], t["problems"], f"passing through on {e}").items():
            if v == 0.0:
                continue
            by_pass = [r for r in passing[e] if r["puzzle"] == p]
            if all(r["trace_sha256"] in reviewed[(p, r["case"])] for r in by_pass):
                if p not in t["reviewed_pass_lines"]:
                    t["reviewed_pass_lines"].append(p)
            else:
                t["problems"].append(f"passing through solves {p} on {e} with no reviewed witness that passes: corpus defect")
    if not _agree(passing, engines):
        t["problems"].append("passing learner: engines disagree")
    for check in (w, m, t):
        check["passed"] = not check["problems"]
    return [w, m, t]


def _zero(entry):
    return entry["difference"] == 0.0 and entry["ci95"] == [0.0, 0.0]


def _check_self_compare(path, work, engine, workers, blocks, learner, report):
    out = {"id": "self-compare", "passed": False, "problems": []}
    small = _derived(load_manifest(path), work, "selfcompare", blocks)
    a = run_benchmark(small, learner, engine=engine, workers=1, out=work / "selfcompare-w1.json", command=["calibrate", "self-compare", "workers=1"], admitted=report)
    b = run_benchmark(small, learner, engine=engine, workers=workers, out=work / "selfcompare-wN.json", command=["calibrate", "self-compare", f"workers={workers}"], admitted=report)
    out["problems"] += [f"{x['status']} result" for x in (a, b) if x["status"] != "complete"]
    if not out["problems"]:
        if not (_rows_equal(a["game_rows"], b["game_rows"]) and a["puzzle_rows"] == b["puzzle_rows"]):
            out["problems"].append(f"rows differ between workers=1 and workers={workers}")
        comparison = compare_results(work / "selfcompare-w1.json", work / "selfcompare-wN.json", work / "selfcompare-comparison.json")
        for mode, entry in comparison["results"].items():
            bad = [c for c, v in entry["cells"].items() if not _zero(v)]
            out["problems"] += [f"{mode}/{c}: nonzero difference" for c in bad]
            if entry["mean_ci95"] != [0.0, 0.0] or entry["mean_difference"] != 0.0:
                out["problems"].append(f"{mode}: nonzero mean difference")
            if "puzzles" in entry and entry["puzzles"]["mean"] != 0.0:
                out["problems"].append(f"{mode}: nonzero puzzle difference")
    out.update(engine=engine, workers=[1, workers], blocks=blocks)
    out["passed"] = not out["problems"]
    return out


def _check_engine_games(path, work, registry, engines, blocks):
    out = {"id": "engine-games", "passed": False, "problems": []}
    if len(engines) < 2:
        out["problems"].append("needs both engines")
        return out
    small = load_manifest(_derived(load_manifest(path), work, "engines", blocks))
    learner = Learner(calibration_record("specialist-panel"), specialist_panel(registry), None)
    runs = {}
    for e in engines:
        key = digest({"calibration": "engine-games", "engine": e})
        runs[e], _ = play_games(small, learner, registry, engine=e, workers=1, cells=None, modes=None, parts=work / f"engines-{e}.parts", key=key, resume=False)
    errors = [r for e in engines for r in runs[e] if r["status"] == "error"]
    out["problems"] += [f"error row {r['cell']}/{r['block']}/{r['slot']}: {r['reason']}" for r in errors]
    out["problems"] += [f"{e} rows differ from {engines[0]}" for e in engines[1:] if not _rows_equal(runs[e], runs[engines[0]])]
    out.update(engines=list(engines), games=len(runs[engines[0]]))
    out["passed"] = not out["problems"]
    return out


def _check_faults(path, work, registry, engine, blocks, report):
    out = {"id": "errors-cannot-inflate", "passed": False, "problems": [], "faults": {}}
    manifest = load_manifest(path)
    small = _derived(manifest, work, "faults", blocks)
    sm = load_manifest(small)
    armed = frozenset(actor_seed(sm.data["stream"], c["id"], 0, 0, m) for c in sm.data["cells"] for m in sm.data["modes"])
    panel = specialist_panel(registry)

    def learner(name, fault):
        return Learner(calibration_record(f"fault-{name}"), {d: FaultFactory(f, fault, armed) for d, f in panel.items()}, pass_learner,
                       learner_for=DeckSpecialists(registry))

    control = run_benchmark(small, learner("control", "none"), engine=engine, out=work / "fault-control.json", command=["calibrate", "fault-control"], admitted=report)
    if control["status"] != "complete":
        out["problems"].append(f"control run is {control['status']}")
    for name, fault in (("exception", "raise"), ("illegal-index", "illegal"), ("exception-when-losing", "losing")):
        faulty = run_benchmark(small, learner(name, fault), engine=engine, out=work / f"fault-{name}.json", command=["calibrate", f"fault-{name}"], admitted=report)
        errors = sum(c["technical_game_errors"] for c in faulty["counts"].values())
        problems = []
        if faulty["status"] != "incomplete" or not errors:
            problems.append("injection did not produce an incomplete result")
        if any(v is not None for a in faulty["aggregate"].values() for v in a.values()):
            problems.append("aggregates are not null")
        try:
            compare_results(work / "fault-control.json", work / f"fault-{name}.json", work / f"fault-{name}-comparison.json")
            problems.append("compare accepted the incomplete run")
        except ValueError:
            pass
        out["faults"][name] = {"status": faulty["status"], "errors": errors, "passed": not problems}
        out["problems"] += [f"{name}: {p}" for p in problems]
    out["passed"] = not out["problems"]
    return out


def calibrate(manifest_path, engines, out, checkpoint=None, *, workers=2, blocks=4, command=None):
    """Run the six controls and write BenchmarkCalibration; status `passed` only if every check passed."""
    out = Path(out).resolve()
    require(not out.exists(), "out", f"{out} exists; results are never overwritten")
    engines = list(engines)
    manifest, registry, report = _admit(manifest_path, engines)
    work = out.parent / (out.stem + ".work")
    if checkpoint:
        try:
            learner = checkpoint_learner(checkpoint, FAIR)
        except ValueError:
            learner = checkpoint_learner(checkpoint, DIAGNOSTIC)
    else:
        learner = agent_learner(SPECIALISTS[0], manifest, registry)
    native = report.get("native_build")
    checks = []
    controls = (("tactics", lambda: _check_lines(manifest, registry, engines)),
                ("self-compare", lambda: [_check_self_compare(manifest_path, work, engines[0], workers, blocks, learner, report)]),
                ("engine-games", lambda: [_check_engine_games(manifest_path, work, registry, engines, blocks)]),
                ("errors-cannot-inflate", lambda: [_check_faults(manifest_path, work, registry, engines[0], blocks, report)]))
    for name, run in controls:
        try:
            checks += run()
        except Exception as e:  # a crashed control is a failed control, not a skipped one
            checks.append({"id": name, "passed": False, "problems": [f"{type(e).__name__}: {e}"]})
    data = {"format": "BenchmarkCalibration", "version": 1, "manifest": {"path": _rel(manifest_path, out.parent), "sha256": manifest.sha256},
            "engines": engines, "code_revision": manifest.data["freeze"]["code_revision"], "native_build": native,
            "candidate": learner.record, "checks": checks, "status": "passed" if all(c["passed"] for c in checks) else "failed",
            "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "workers": workers,
                        "command": list(sys.argv if command is None else command), "dependencies": dependencies()}}
    write_json(out, data)
    return data
