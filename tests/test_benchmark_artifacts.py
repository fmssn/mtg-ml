import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest

from benchmark_fixtures import fixture
from mtg_ml.benchmark.artifacts import load_manifest, load_puzzle, reference
from mtg_ml.benchmark.jsonio import canonical_bytes, digest, file_digest, read_json, write_json
from mtg_ml.benchmark.schedule import episodes, simulator_seed, actor_seed, puzzle_seed, bootstrap_seed
from mtg_ml.benchmark.results import build_result, validate_result, write_result, load_result


def test_canonical_identity_and_reference_bytes(tmp_path):
    a = {"ü": [3, 2], "a": 1}
    assert canonical_bytes(a) == '{"a":1,"ü":[3,2]}'.encode()
    assert digest(a) == digest({"a": 1, "ü": [3, 2]})
    assert digest(a) != digest({"a": 1, "ü": [2, 3]})
    p = tmp_path / "pretty.json"
    write_json(p, a)
    assert file_digest(p) != digest(a)
    assert reference(tmp_path / "parent.json", {"path": "pretty.json", "sha256": file_digest(p)}, "test") == p
    with pytest.raises(ValueError, match="hash mismatch"):
        reference(tmp_path / "parent.json", {"path": "pretty.json", "sha256": digest(a)}, "test")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x": NaN}', '{"x": Infinity}', '{"x":1e999}'])
def test_bad_json_rejected(tmp_path, text):
    p = tmp_path / "bad.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        read_json(p)


def assessment(name, supersedes=()):
    return {"id": name, "exposed": {"checkpoints": ["a" * 64], "puzzle_groups": ["synthetic-jund-removal"]}, "supersedes": list(supersedes)}


@pytest.mark.parametrize("change", ["version", "bool_version", "count", "bool_count", "deck_hash", "bundle_hash", "cards", "cell", "duplicate", "mode", "stream", "missing",
                                    "final_no_assessment", "final_stream", "dev_assessment", "exposed_hash"])
def test_manifest_errors_name_artifact(tmp_path, change):
    m, _ = fixture(tmp_path, puzzles=False)
    d = copy.deepcopy(m.data)
    if change == "version": d["version"] = 2
    if change == "bool_version": d["version"] = True
    if change == "count": d["blocks_per_cell"] = 0
    if change == "bool_count": d["blocks_per_cell"] = True
    if change == "deck_hash": d["decks"]["jund_wildfire"]["sha256"] = "0" * 64
    if change == "bundle_hash": d["freeze"]["deck_bundle_sha256"] = "0" * 64
    if change == "cards": d["decks"]["jund_wildfire"]["cards"]["Swamp"] = 4
    if change == "cell": d["cells"][0]["learner_deck"] = "mono_blue_terror"
    if change == "duplicate": d["bots"].append(d["bots"][0])
    if change == "mode": d["modes"] = ["sampled", "sampled"]
    if change == "stream": d["stream"] = "benchmark-v1/final"
    if change == "missing": del d["freeze"]
    if change == "final_no_assessment": d.update(split="final", stream="benchmark-v1/final")
    if change == "final_stream": d.update(split="final", stream="benchmark-v1/final/a2", assessment=assessment("a1"))
    if change == "dev_assessment": d["assessment"] = assessment("a1")
    if change == "exposed_hash":
        d.update(split="final", stream="benchmark-v1/final/a1", assessment=assessment("a1"))
        d["assessment"]["exposed"]["checkpoints"] = ["not-a-hash"]
    p = tmp_path / "bad.json"
    write_json(p, d)
    with pytest.raises(ValueError, match="bad.json"):
        load_manifest(p)


@pytest.mark.parametrize("change", ["review", "split", "mode", "cap", "objective", "boundary", "forced"])
def test_puzzle_errors(tmp_path, change):
    fixture(tmp_path)
    d = read_json(tmp_path / "puzzle.json")
    if change == "review": d["review"]["status"] = "pending"
    if change == "split": d["split"] = "test"
    if change == "mode": d["episode_mode"] = "history"
    if change == "cap": d["max_decisions"] = 65
    if change == "objective": d["objective"] = {"all": [{"path": "opponent.library_known", "op": "eq", "value": []}]}
    if change == "boundary": d["stop"]["require_empty_stack"] = False
    if change == "forced": d["claim"] = "forced"
    p = tmp_path / "bad-puzzle.json"
    write_json(p, d)
    with pytest.raises(ValueError, match="bad-puzzle.json"):
        load_puzzle(p)


def test_exact_seed_tags_and_slots(tmp_path):
    m, _ = fixture(tmp_path, puzzles=False)
    specs = episodes(m)
    assert len(specs) == 32
    assert simulator_seed("benchmark-v1/dev", "jund_vs_blue", 0) == int.from_bytes(hashlib.sha256(b'["benchmark-v1/dev","jund_vs_blue",0]').digest()[:8], "big") % 2**31
    for cell in {s.cell for s in specs}:
        rows = [s for s in specs if s.cell == cell and s.mode == "sampled"]
        assert [(s.learner_seat, s.starting_player) for s in rows] == [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert len({s.simulator_seed for s in rows}) == 1
        assert len({s.actor_seed for s in rows}) == 4
        for s in rows:
            expected = "jund_wildfire" if cell.startswith("jund") else "mono_blue_terror"
            assert s.deck_ids[s.learner_seat] == expected
            assert s.decks[s.learner_seat] == tuple(sorted(s.decks[s.learner_seat]))
    assert simulator_seed(m.data["stream"], "jund_vs_blue", 0) != actor_seed(m.data["stream"], "jund_vs_blue", 0, 0, "sampled")
    assert puzzle_seed(m.data["stream"], "p", 0, "sampled") == puzzle_seed(m.data["stream"], "p", 0, "sampled")
    assert bootstrap_seed(m.data["stream"], "s", "greedy") != puzzle_seed(m.data["stream"], "s", 0, "greedy")


def test_final_assessment_id_enters_every_seed(tmp_path):
    m, _ = fixture(tmp_path, puzzles=False)
    seeds = []
    for name, prior in (("a1", ()), ("a2", ("a1",))):
        d = copy.deepcopy(m.data)
        d.update(split="final", stream=f"benchmark-v1/final/{name}", assessment=assessment(name, prior))
        p = tmp_path / f"final-{name}.json"
        write_json(p, d)
        seeds.append({(s.cell, s.block): (s.simulator_seed, s.actor_seed) for s in episodes(load_manifest(p))})
    assert seeds[0].keys() == seeds[1].keys()
    assert all(seeds[0][k][0] != seeds[1][k][0] and seeds[0][k][1] != seeds[1][k][1] for k in seeds[0])
    assert seeds[0][("jund_vs_blue", 0)][0] != simulator_seed("benchmark-v1/dev", "jund_vs_blue", 0)


def test_atomic_exclusive_publish(tmp_path):
    path = tmp_path / "result.json"
    def publish(i):
        try:
            write_json(path, {"writer": i})
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(publish, range(8))) == 1
    assert 0 <= read_json(path)["writer"] < 8
    assert not list(tmp_path.glob(".benchmark-*"))
    with pytest.raises(ValueError):
        write_json(tmp_path / "nonfinite.json", {"x": float("nan")})
    assert not (tmp_path / "nonfinite.json").exists()


def candidate():
    return {"checkpoint": "synthetic-not-a-checkpoint", "sha256": "a" * 64, "features": 7, "information_contract": "own-list-hidden-opponent-v1"}


def test_result_counts_duplicates_errors_and_roundtrip(tmp_path):
    m, _ = fixture(tmp_path, puzzles=False)
    from dataclasses import asdict
    rows = [{k: v for k, v in asdict(s).items() if k != "decks"} | dict(status="completed", winner=None, reason="turn_limit", decisions=20, turns=2)
            for s in episodes(m, modes=["greedy"])]
    result = build_result(m, candidate(), rows, code_revision=m.data["freeze"]["code_revision"], modes=["greedy"])
    assert result["status"] == "complete"
    assert result["partial_cells"][0]["score"] == 0.5
    claimed = copy.deepcopy(result)
    claimed["aggregate"]["greedy"]["game_score"] = 0.5
    with pytest.raises(ValueError, match="deferred"):
        validate_result(claimed)
    stored = write_result(tmp_path / "result.json", result)
    assert load_result(tmp_path / "result.json").data == stored
    for mutation in ("duplicate", "count", "seed", "missing", "error"):
        bad = copy.deepcopy(result)
        if mutation == "duplicate": bad["game_rows"].append(bad["game_rows"][0])
        if mutation == "count": bad["counts"]["greedy"]["completed_games"] -= 1
        if mutation == "seed": bad["game_rows"][0]["actor_seed"] += 1
        if mutation == "missing": bad["game_rows"].pop()
        if mutation == "error": bad["game_rows"][0]["status"] = "error"
        with pytest.raises(ValueError):
            validate_result(bad)
    missing = build_result(m, candidate(), rows[:-1], code_revision=m.data["freeze"]["code_revision"], modes=["greedy"])
    assert missing["status"] == "incomplete" and all(v is None for v in missing["aggregate"]["greedy"].values())
    missing["aggregate"]["greedy"]["game_score"] = 0.5
    with pytest.raises(ValueError): validate_result(missing)


def test_cli_invalid_report_and_no_overwrite(tmp_path):
    from mtg_ml.benchmark.__main__ import main
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit) as e:
        main(["validate", "--manifest", str(tmp_path / "missing.json"), "--out", str(out)])
    assert e.value.code == 1 and read_json(out)["status"] == "invalid"
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(["validate", "--manifest", "missing.json", "--out", str(out)])
    assert out.read_bytes() == before


def test_registry_freeze_and_reserved_identities(tmp_path):
    from mtg_ml.benchmark import AgentMetadata, AgentRegistry, FAIR, DIAGNOSTIC
    from benchmark_fixtures import PassAgent, revision
    registry = AgentRegistry()
    reserved = AgentMetadata("benchmark-jund@1", "jund_wildfire", revision(), 1, digest({}), FAIR, "synthetic")
    with pytest.raises(ValueError, match="reserved"):
        registry.register(reserved, PassAgent, {})
    legacy = AgentMetadata("legacy-jund@1", "jund_wildfire", revision(), 1, digest({}), FAIR, "legacy")
    with pytest.raises(ValueError, match="diagnostic"):
        registry.register(legacy, PassAgent, {})
    legacy = __import__("dataclasses").replace(legacy, information_contract=DIAGNOSTIC)
    registry.register(legacy, PassAgent, {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register(legacy, PassAgent, {})
    m, r = fixture(tmp_path, puzzles=False)
    bot = copy.deepcopy(m.data["bots"][0])
    bot["parameters_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="freeze mismatch"):
        r.verify(bot, FAIR)
    mutable = {"threshold": 1}
    meta = AgentMetadata("synthetic-params@1", "jund_wildfire", revision(), 1, digest(mutable), FAIR, "synthetic")
    registry.register(meta, PassAgent, mutable)
    mutable["threshold"] = 2
    with pytest.raises(ValueError, match="parameters changed"):
        registry.create(meta.id)


def test_validation_fails_missing_components_and_freeze(tmp_path, monkeypatch):
    from mtg_ml.benchmark.validation import validate
    m, r = fixture(tmp_path, puzzles=False)
    wrong = copy.deepcopy(m.data)
    wrong["freeze"]["card_spec_sha256"] = "0" * 64
    p = tmp_path / "wrong.json"
    write_json(p, wrong)
    report = validate(p, engines=("python",), registry=r)
    assert report["status"] == "invalid" and "card spec mismatch" in report["errors"][0]
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    monkeypatch.setattr("mtg_ml.benchmark.validation.native_available", lambda: False)
    report = validate(m.path, registry=r)
    assert report["status"] == "invalid" and "native engine unavailable" in report["errors"][0]
    report = validate(m.path, engines=("python",))
    assert report["status"] == "invalid" and "missing agent" in report["errors"][0]
    (tmp_path / "bundle.json").unlink()
    report = validate(m.path, engines=("python",), registry=r)
    assert report["status"] == "invalid" and "missing file" in report["errors"][0]


def test_planned_cases_strategic_failure_and_missing_completion(tmp_path):
    from dataclasses import asdict
    from mtg_ml.benchmark import puzzle_plan
    m, _ = fixture(tmp_path)
    cells, modes = ["jund_vs_blue"], ["greedy"]
    games = [{k: v for k, v in asdict(s).items() if k != "decks"} | dict(status="completed", winner=None, reason="turn_limit", decisions=20, turns=2)
             for s in episodes(m, cells, modes)]
    kw = dict(cells=cells, modes=modes, code_revision=m.data["freeze"]["code_revision"])
    missing = build_result(m, candidate(), games, **kw)
    assert missing["status"] == "incomplete" and missing["counts"]["greedy"]["planned_puzzles"] == 1
    rows = [r | dict(status="failure", objective_met=False, first_action_correct=True, decisions=64, reason="horizon_exhausted") for r in puzzle_plan(m, cells, modes)]
    complete = build_result(m, candidate(), games, rows, **kw)
    assert complete["status"] == "complete" and complete["panel"] == "partial"
    assert complete["counts"]["greedy"]["completed_puzzles"] == 1
    rows[0]["status"] = "error"
    incomplete = build_result(m, candidate(), games, rows, **kw)
    assert incomplete["status"] == "incomplete" and incomplete["counts"]["greedy"]["technical_case_errors"] == 1
    rows[0]["status"] = "success"
    with pytest.raises(ValueError, match="objective"):
        build_result(m, candidate(), games, rows, **kw)


def test_result_manifest_and_plan_are_reverified(tmp_path):
    m, _ = fixture(tmp_path, puzzles=False)
    result = build_result(m, candidate(), [], code_revision=m.data["freeze"]["code_revision"])
    stored = write_result(tmp_path / "result.json", result)
    stored["planned_game_rows"][0]["simulator_seed"] += 1
    with pytest.raises(ValueError, match="manifest schedule mismatch"):
        write_result(tmp_path / "false-plan.json", stored)
    m.path.write_text(m.path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_result(tmp_path / "result.json")


def test_cli_success_uses_explicit_registered_adapters(tmp_path, monkeypatch):
    from mtg_ml.benchmark.__main__ import main
    m, r = fixture(tmp_path)
    monkeypatch.setattr("mtg_ml.benchmark.REGISTRY", r)
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    out = tmp_path / "validation.json"
    main(["validate", "--manifest", str(m.path), "--engines", "python", "--out", str(out)])
    assert read_json(out)["status"] == "complete"
