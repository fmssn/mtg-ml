"""Artifact admission and reviewed witness checks, without agent attempts."""

from pathlib import Path
import subprocess
import hashlib

from ..backend import native_available
from ..engine.cards import SPEC_PATH
from ..engine.view import observe
from ..expert.scenarios import compile_scenario, select, at_path
from ..replay import visible_events
from .artifacts import load_manifest, load_puzzle, load, validate_bundle, reference, require
from .jsonio import file_digest, read_json, digest
from .views import inputs, thaw

ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATHS = ("mtg_ml", "native")


def check_freeze(manifest):
    freeze = manifest.data["freeze"]
    require(freeze["card_spec_sha256"] == file_digest(SPEC_PATH), "freeze.card_spec_sha256", "installed card spec mismatch")
    rev = freeze["code_revision"]
    proc = subprocess.run(["git", "cat-file", "-t", rev], cwd=ROOT, capture_output=True, text=True)
    require(proc.returncode == 0 and proc.stdout.strip() == "commit", "freeze.code_revision", "commit unavailable locally")
    proc = subprocess.run(["git", "diff", "--name-only", rev, "--", *FREEZE_PATHS], cwd=ROOT, capture_output=True, text=True)
    require(proc.returncode == 0 and not proc.stdout.strip(), "freeze.code_revision", "checkout differs from freeze: " + proc.stdout.strip())
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *FREEZE_PATHS], cwd=ROOT, capture_output=True, text=True)
    require(untracked.returncode == 0 and not untracked.stdout.strip(), "freeze.code_revision", "untracked runtime source")


def native_build():
    require(native_available(), "engines.native", "native engine unavailable")
    import mtg_ml_native
    require(hasattr(mtg_ml_native, "build_sources"), "engines.native", "native extension lacks build identity; run make native")
    compiled = dict(mtg_ml_native.build_sources())
    current = {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / "native/src").glob("*.rs")}
    current.update({str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / "native/Cargo.toml", ROOT / "native/Cargo.lock")})
    require(compiled == current, "engines.native", "installed binary does not match checkout; run make native")
    return {"source_sha256": digest({k: hashlib.sha256(v.encode()).hexdigest() for k, v in compiled.items()}),
            "extension": str(mtg_ml_native.__file__)}


def boundary(game, perspective, stop):
    if game.over:
        return True
    return stop["kind"] == "decision_boundary" and game.turn == stop["turn"] and game.step_name == stop["step"] and \
        game.active == (perspective if stop["active"] == "self" else 1 - perspective) and not game.stack and \
        game.decision is not None and game.decision.player == perspective and game.decision.kind == "priority" and not game.pending


def objective(game, perspective, clauses, objects):
    obs = observe(game, perspective)
    for clause in clauses:
        if "object" in clause:
            name = clause["object"]
            require(name in objects, "objective.object", f"unknown binding {name}")
            card = objects[name]
            # uid is physical identity and survives zone movement; oid does not.
            candidates = game.battlefield if clause["zone"] == "battlefield" else [c for p in game.players for c in getattr(p, clause["zone"])]
            actual = any(c.uid == card.uid for c in candidates)
            if actual != clause["present"]:
                return False
        else:
            value = at_path(obs, clause["path"])
            target, op = clause["value"], clause["op"]
            if not {"eq": lambda: value == target, "gte": lambda: value >= target, "lte": lambda: value <= target,
                    "contains": lambda: target in value}[op]():
                return False
    return True


def _verify_case(puzzle, case, decks, engines):
    scenario_path = reference(puzzle.path, case["scenario"], "cases.scenario")
    evidence_path = reference(puzzle.path, case["evidence"], "cases.evidence")
    scenario, evidence = read_json(scenario_path), read_json(evidence_path)
    witnesses = case.get("witnesses")
    if witnesses is None:
        actions = [s["selector"] for s in scenario.get("demonstration", [])] or scenario.get("continuation", [])
        require(bool(actions), "cases.witnesses", "successful full-line witness required")
        witnesses = [{"actions": actions, "note": "Reviewed scenario continuation"}]
    checked = []
    for witness in witnesses:
        actions = witness["actions"]
        require(0 < len(actions) <= puzzle.data["max_decisions"], "witness", "decision cap exceeded")
        traces = []
        for engine in engines:
            g, objects = compile_scenario(scenario, evidence, engine)
            viewer = scenario["perspective"]
            # Reset-mode setup has no fabricated recurrent history. Registration
            # is supplied separately to the scripted input builder.
            own = decks[puzzle.data["deck"]]["cards"]
            if puzzle.data.get("acceptable_first_actions"):
                for selector in puzzle.data["acceptable_first_actions"]:
                    select(g, selector, objects)
            trace, seen = [], len(g.log)
            reached = False
            initial_view = observe(g, viewer)
            for n, selector in enumerate(actions):
                require(not reached, "witness", "continues beyond first stopping boundary")
                decider = g.decision.player
                registration = own if decider == viewer else {}
                view, legal = inputs(g, decider, registration)
                trace.append({"state": thaw(view.state), "context": thaw(view.context), "cards": thaw(view.cards),
                              "actions": [{"index": a.index, "kind": a.kind, "key": thaw(a.key), "data": thaw(a.data), "label": a.label} for a in legal]})
                index = select(g, selector, objects)
                g.step(index)
                trace[-1]["chosen"] = index
                trace[-1]["events"] = visible_events(g.log[seen:], viewer)
                seen = len(g.log)
                reached = boundary(g, viewer, puzzle.data["stop"])
                if reached:
                    require(objective(g, viewer, puzzle.data["objective"]["all"], objects), "witness.objective", "reviewed line does not achieve goal")
                require(n + 1 <= puzzle.data["max_decisions"], "witness", "cap exceeded")
            require(reached, "witness.stop", "line does not reach stopping boundary")
            traces.append({"initial": initial_view, "trace": trace, "finish": observe(g, viewer)})
        require(all(t == traces[0] for t in traces), "parity", "engine inputs/actions/events/outcomes diverged")
        checked.append({"actions": len(actions), "trace_sha256": digest(traces[0])})
    return {"case": case["id"], "witnesses": checked}


def validate(manifest_path, engines=("python", "native"), registry=None):
    report = {"format": "BenchmarkValidation", "version": 1, "status": "invalid", "manifest": {"path": str(manifest_path)},
              "engines": list(engines), "puzzles": [], "errors": []}
    try:
        require(bool(engines) and len(set(engines)) == len(engines) and set(engines) <= {"python", "native"}, "engines", "invalid selection")
        manifest = load_manifest(manifest_path)
        report["manifest"]["sha256"] = manifest.sha256
        check_freeze(manifest)
        if "native" in engines:
            report["native_build"] = native_build()
        if registry is None:
            from . import REGISTRY
            registry = REGISTRY
        d = manifest.data
        for bot in d["bots"]:
            registry.verify(bot, d["information_contract"])
        bundle_path = reference(manifest.path, d["puzzles"], "puzzles")
        bundle = load(bundle_path, validate_bundle)
        puzzles = [load_puzzle(reference(bundle.path, ref, "bundle.puzzles")) for ref in bundle.data["puzzles"]]
        require(len({p.data["id"] for p in puzzles}) == len(puzzles), "bundle.puzzles", "duplicate puzzle id")
        # Connected source/template groups must not straddle dev/final.
        groups = []
        for p in puzzles:
            groups.append(({("template", p.data["template_id"]), ("source", p.data["source_group_id"])}, {p.data["split"]}))
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i][0] & groups[j][0]:
                        groups[i] = (groups[i][0] | groups[j][0], groups[i][1] | groups[j][1])
                        groups.pop(j)
                        changed = True
                        break
                if changed:
                    break
        require(all(len(splits) == 1 for _, splits in groups), "bundle.split", "intersecting leakage groups cross splits")
        bots = {b["id"]: b for b in d["bots"]}
        for puzzle in puzzles:
            expected_split = "dev" if d["split"] == "power-pilot" else d["split"]
            require(puzzle.data["split"] == expected_split, "puzzle.split", "manifest split mismatch")
            cases = []
            for case in puzzle.data["cases"]:
                response = case["response_policy"]
                require(response["id"] in bots and bots[response["id"]]["parameters_sha256"] == response["parameters_sha256"], "response_policy", "freeze mismatch")
                cases.append(_verify_case(puzzle, case, d["decks"], engines))
            report["puzzles"].append({"id": puzzle.data["id"], "cases": cases})
        report["status"] = "complete"
        report["scope"] = "foundation-validation; response-policy branch coverage and release corpus certification deferred"
    except (ValueError, KeyError, TypeError, AttributeError, OSError, IndexError) as e:
        report["errors"].append(str(e))
    return report
