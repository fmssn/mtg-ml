"""Frozen benchmark release tooling: statistics, verified aggregates, run/compare/resume,
the objective loophole, engine agreement, fault injection, calibration and archives.

Tests that need an engine parametrize it themselves (the conftest `engine` fixture is indirect).
Game-playing tests use the reviewed development corpus with the specialists at a pinned fake
revision and a two-block panel; the freeze check is patched out as in test_benchmark_tactics.
"""

import copy
import json
import math
from pathlib import Path
import sys

import pytest

from benchmark_fixtures import fixture
from mtg_ml.benchmark import build_result, episodes
from mtg_ml.benchmark import stats
from mtg_ml.benchmark.__main__ import main
from mtg_ml.benchmark.artifacts import load_manifest
from mtg_ml.benchmark.jsonio import file_digest, read_json, write_json
from mtg_ml.benchmark.results import load_result, validate_result, write_result
from mtg_ml.benchmark.tactics import SelectorFactory, attempt_case, ResponseFactory

ROOT = Path(__file__).resolve().parents[1]
ENGINES = pytest.mark.parametrize("engine", ["python", "native"], indirect=True)
REVISION = "a" * 40


# -- statistics ---------------------------------------------------------------------------------

def test_percentile_interpolates():
    assert stats.percentile([0, 1, 2, 3], 0.5) == 1.5
    assert stats.percentile([5], 0.9) == 5
    assert stats.percentile([1, 2, 3, 4, 5], 0.25) == 2


def plan_and_rows(outcomes, cells=("a", "b"), mode="greedy"):
    """Rows for blocks of four slots; `outcomes[cell]` is one win/draw/loss letter string per block."""
    plan, rows = [], []
    for cell in cells:
        for block, letters in enumerate(outcomes[cell]):
            for slot, letter in enumerate(letters):
                key = dict(mode=mode, cell=cell, block=block, slot=slot)
                plan.append(key)
                rows.append(dict(key, status="completed", learner_seat=0, winner={"w": 0, "l": 1, "d": None}[letter]))
    return plan, rows


def test_block_scores_are_strict_and_four_slot_means():
    plan, rows = plan_and_rows({"a": ["wwll", "wddd"], "b": ["wwww", "llll"]})
    assert stats.block_scores(rows, plan) == {"greedy": {"a": [0.5, 0.625], "b": [1.0, 0.0]}}
    for bad in (rows[1:], rows + [rows[0]], [dict(rows[0], status="error")] + rows[1:], [dict(rows[0], block=9)] + rows[1:]):
        with pytest.raises(ValueError):
            stats.block_scores(bad, plan)


def test_intervals_share_indices_across_cells():
    # Anti-correlated cells: any shared resample leaves the equal-weight mean at exactly 0.5.
    iv = stats.game_intervals({"a": [0.0, 1.0], "b": [1.0, 0.0]}, 500, 7)
    assert iv["mean"]["ci95"] == [0.5, 0.5] and iv["cells"]["a"]["ci95"] == [0.0, 1.0]
    # Identical cells resample identically, so the mean interval is each cell's interval.
    same = stats.game_intervals({"a": [0.0, 1.0, 0.5], "b": [0.0, 1.0, 0.5]}, 500, 7)
    assert same["mean"]["ci95"] == same["cells"]["a"]["ci95"] == same["cells"]["b"]["ci95"]
    assert stats.game_intervals({"a": [0.0, 1.0, 0.5]}, 500, 7) == stats.game_intervals({"a": [0.0, 1.0, 0.5]}, 500, 7)


def test_self_difference_is_exactly_zero_and_single_block_is_unavailable():
    plan, rows = plan_and_rows({"a": ["wwll", "wddd", "llld"], "b": ["wwww", "llll", "wlwl"]})
    diffs = stats.paired_game_differences(rows, rows, plan)["greedy"]
    iv = stats.game_intervals(diffs, 200, 3)
    assert iv["mean"] == {"mean": 0.0, "ci95": [0.0, 0.0]} and all(c == {"mean": 0.0, "ci95": [0.0, 0.0]} for c in iv["cells"].values())
    one = stats.game_intervals({"a": [0.5]}, 200, 3)
    assert one["mean"]["ci95"] is None and one["status"] == "unavailable" and "fewer than 2" in one["reason"]


def test_paired_differences_are_candidate_minus_baseline():
    plan, base = plan_and_rows({"a": ["llll", "llll"], "b": ["wwww", "wwww"]})
    _, cand = plan_and_rows({"a": ["wwww", "dddd"], "b": ["wwww", "llll"]})
    assert stats.paired_game_differences(base, cand, plan) == {"greedy": {"a": [1.0, 0.5], "b": [0.0, -1.0]}}


def test_claim_truth_table():
    claim = stats.claim
    assert claim({"a": [0.01, 0.1], "b": [-0.02, 0.05]}, [0.01, 0.08]) == "stronger"
    # A cell whose lower bound sits at -3pp or below has not ruled out a regression.
    assert claim({"a": [0.01, 0.1], "b": [-0.03, 0.05]}, [0.01, 0.08]) == "inconclusive"
    assert claim({"a": [0.01, 0.1], "b": [-0.10, 0.02]}, [0.01, 0.08]) == "inconclusive"
    assert claim({"a": [0.0, 0.1]}, [0.0, 0.08]) == "inconclusive"  # mean lower bound must exceed zero
    assert claim({"a": [-0.2, -0.031], "b": [0.0, 0.1]}, [-0.05, 0.05]) == "regression"
    assert claim({"a": [-0.1, 0.1]}, [-0.1, -0.001]) == "regression"
    assert claim({"a": None}, None) == "inconclusive"


def test_flags_and_effect_sizing():
    assert [stats.flags(v) for v in (0.95, 0.96, 0.5, 0.05, 0.0)] == ["saturated", "saturated", None, "floor", "floor"]
    assert stats.blocks_for_effect(0.10, 0.02) == 1960
    assert stats.blocks_for_effect(0.10, 0.03) == math.ceil(7.84 * 0.1 / 0.0009)


def puzzle(pid, template, source, deck="jund_wildfire", category="combat"):
    return dict(id=pid, template_id=template, source_group_id=source, deck=deck, category=category)


def test_leakage_group_carries_all_its_puzzles_transitively():
    groups = stats.leakage_groups([puzzle("a", "t1", "s1"), puzzle("b", "t1", "s2"), puzzle("c", "t2", "s2"), puzzle("d", "t3", "s3"), puzzle("e", "t4", "s1")])
    assert groups == {"a": ["a", "b", "c", "e"], "d": ["d"]}


def test_puzzle_values_need_every_case_and_reject_errors():
    def row(p, case, rep, status):
        return dict(puzzle=p, case=case, repetition=rep, status=status)
    rows = [row("p", "c1", 0, "success"), row("p", "c2", 0, "failure"), row("p", "c1", 1, "success"), row("p", "c2", 1, "success"), row("q", "c1", 0, "success")]
    assert stats.puzzle_values(rows) == {"p": 0.5, "q": 1.0}
    with pytest.raises(ValueError, match="technical errors"):
        stats.puzzle_values(rows + [row("q", "c2", 0, "error")])


def test_puzzle_intervals_resample_groups_and_need_enough_of_them():
    few = {f"p{i}": float(i % 2) for i in range(7)}
    meta = {p: puzzle(p, f"t{p}", f"s{p}") for p in few}
    small = stats.puzzle_intervals(few, meta, 100, 1)
    assert small["overall_status"] == "unavailable" and small["overall_ci95"] is None and "7 leakage groups" in small["overall_reason"]
    assert small["decks"]["jund_wildfire"]["status"] == "unavailable"
    many = {f"p{i}": float(i % 2) for i in range(stats.MIN_GROUPS_DECK)}
    meta = {p: puzzle(p, f"t{p}", f"s{p}") for p in many}
    big = stats.puzzle_intervals(many, meta, 200, 1)
    assert big["overall_status"] == "ok" and 0 < big["overall_ci95"][0] < big["mean"] < big["overall_ci95"][1] < 1
    # Twenty puzzles in a single leakage group are one independent unit.
    one_group = {p: puzzle(p, "same", f"s{p}") for p in many}
    assert stats.puzzle_intervals(many, one_group, 50, 1)["overall_status"] == "unavailable"
    assert stats.categories(many, meta) == {"combat": {"puzzles": 20, "mean": 0.5}}


# -- results: aggregates are recomputed from rows ----------------------------------------------

def candidate():
    return {"checkpoint": "synthetic.pt", "sha256": "0" * 64, "features": 7, "information_contract": "own-list-hidden-opponent-v1"}


def synthetic_result(tmp_path, blocks=3, **kw):
    m, _ = fixture(tmp_path, puzzles=False)
    data = copy.deepcopy(m.data)
    data["blocks_per_cell"] = blocks
    path = tmp_path / "blocks.json"
    write_json(path, data)
    m = load_manifest(path)
    rows = []
    for i, s in enumerate(episodes(m, modes=["greedy"])):
        winner = None if i % 5 == 0 else s.learner_seat if i % 3 else 1 - s.learner_seat
        rows.append({k: v for k, v in s.__dict__.items() if k != "decks"} | dict(status="completed", winner=winner, reason="x", decisions=3, turns=2, latency_seconds=[0.1 * (i % 4 + 1)]))
    return m, rows, build_result(m, candidate(), rows, code_revision=REVISION, modes=["greedy"], **kw)


def test_aggregates_are_real_and_verified(tmp_path):
    m, rows, result = synthetic_result(tmp_path)
    a = result["aggregate"]["greedy"]
    assert result["status"] == "complete" and result["interval_method"] == "paired-block-bootstrap-v1"
    assert a["fair"] and a["game_score"] is not None and len(a["game_ci95"]) == 2 and set(a["cells"]) == set(c["id"] for c in m.data["cells"])
    assert a["puzzle_success"] is None  # no puzzles planned
    assert sum(c["games"] for c in a["cells"].values()) == len(rows)
    assert result["runtime"]["command"] and "dependencies" in result["runtime"]
    stored = write_result(tmp_path / "r.json", result)
    assert load_result(tmp_path / "r.json").data == stored
    for tamper in (lambda d: d["aggregate"]["greedy"].update(game_score=a["game_score"] + 0.01),
                   lambda d: d["aggregate"]["greedy"]["cells"]["jund_vs_blue"].update(ci95=[0.0, 1.0]),
                   lambda d: d["aggregate"]["greedy"]["cells"]["jund_vs_blue"].update(flag="floor"),
                   lambda d: d["game_rows"][0].update(winner=1 - d["game_rows"][0]["winner"] if d["game_rows"][0]["winner"] is not None else 0)):
        bad = copy.deepcopy(result)
        tamper(bad)
        with pytest.raises(ValueError):
            validate_result(bad)


def test_incomplete_runs_report_nothing_and_old_candidates_load(tmp_path):
    m, rows, _ = synthetic_result(tmp_path)
    incomplete = build_result(m, candidate(), rows[:-1], code_revision=REVISION, modes=["greedy"])
    assert incomplete["status"] == "incomplete" and all(v is None for v in incomplete["aggregate"]["greedy"].values())
    claimed = copy.deepcopy(incomplete)
    claimed["aggregate"]["greedy"]["game_score"] = 0.9
    with pytest.raises(ValueError):
        validate_result(claimed)
    errored = [dict(rows[0], status="error")] + rows[1:]
    result = build_result(m, candidate(), errored, code_revision=REVISION, modes=["greedy"])
    assert result["status"] == "incomplete" and result["aggregate"]["greedy"]["cells"] is None
    assert "kind" not in candidate()  # a record without `kind` is a checkpoint
    legacy_marked = dict(candidate(), kind="agent")
    with pytest.raises(ValueError):
        build_result(m, legacy_marked, rows, code_revision=REVISION, modes=["greedy"])


def test_candidate_kinds(tmp_path):
    m, rows, _ = synthetic_result(tmp_path)
    agent = dict(kind="agent", id="legacy-jund", deck="jund_wildfire", source_revision=REVISION, rules_revision=1, agent_kind="legacy",
                 parameters_sha256="1" * 64, information_contract="privileged-diagnostic")
    result = build_result(m, agent, rows, code_revision=REVISION, modes=["greedy"])
    assert result["aggregate"]["greedy"]["fair"] is False
    with pytest.raises(ValueError, match="diagnostic"):
        build_result(m, dict(agent, information_contract="own-list-hidden-opponent-v1"), rows, code_revision=REVISION, modes=["greedy"])
    oracle = dict(kind="calibration", id="oracle", information_contract="privileged-diagnostic")
    assert build_result(m, oracle, rows, code_revision=REVISION, modes=["greedy"])["status"] == "complete"


# -- the objective loophole ---------------------------------------------------------------------

CAST = {"player": 0, "kind": "priority", "key": ["cast", "Cast Down", "hand", "normal"]}
TARGET = {"player": 0, "kind": "target", "key": ["target", "nonlegendary_creature", "perm", "opponent", "Delver of Secrets"]}
PAY = {"player": 0, "kind": "pay_mana", "key": ["pay", "source", "Swamp", "B"]}


def tactics_setup(tmp_path, edit_puzzle=None, edit_scenario=None):
    from test_benchmark_tactics import setup
    return setup(tmp_path, edit_puzzle, edit_scenario)


def attempt(p, case, reg, learner, engine):
    from test_benchmark_tactics import DECKS
    return attempt_case(p, case, DECKS, learner, ResponseFactory(reg, case["response_policy"]["id"]), mode="greedy", engine=engine)


@ENGINES
def test_reaching_the_goal_and_then_losing_the_game_fails(tmp_path, engine):
    def decked(scenario):
        scenario["initial"]["players"][0]["library"] = []
        scenario["initial"]["players"][0]["library_count"] = 0

    def to_the_end(puzzle):
        puzzle["stop"] = {"kind": "terminal"}
        puzzle["cases"][0].pop("witnesses")
        puzzle.pop("acceptable_first_actions")
    _, reg, p, case = tactics_setup(tmp_path, to_the_end, decked)
    row = attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)
    assert (row["status"], row["reason"], row["game_lost"], row["objective_met"]) == ("failure", "objective_failed", True, False), row
    # The same objective is met by a line whose game is still running at the boundary.
    _, reg, p, case = tactics_setup(tmp_path / "ok")
    ok = attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)
    assert (ok["status"], ok["game_lost"]) == ("success", False)


@ENGINES
def test_violated_life_clause_fails(tmp_path, engine):
    def life(puzzle):
        puzzle["objective"]["all"].append({"path": "self.life", "op": "gte", "value": 20})
        puzzle["cases"][0].pop("witnesses")

    def hurt(scenario):
        scenario["initial"]["players"][0]["life"] = 19
    _, reg, p, case = tactics_setup(tmp_path, life, hurt)
    row = attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)
    assert (row["status"], row["reason"]) == ("failure", "objective_failed")


# -- running, comparing, resuming ---------------------------------------------------------------

@pytest.fixture
def dev(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    from puzzle_manifest import manifest
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    # The reviewed lines were verified by the corpus tests; these tests exercise the runner.
    monkeypatch.setattr("mtg_ml.benchmark.release.validate", lambda path, engines: {"status": "complete", "errors": [], "native_build": None})
    path = tmp_path / "dev.json"
    write_json(path, manifest(path, revision=REVISION, blocks=2, modes=("greedy",)))
    return path


def run(dev, out, *extra, agent="benchmark-jund@1"):
    main(["run", "--manifest", str(dev), "--agent", agent, "--out", str(out), *extra])


@pytest.mark.slow
def test_run_compare_self_and_refusals(dev, tmp_path):
    run(dev, tmp_path / "a.json")
    run(dev, tmp_path / "b.json", "--workers", "2")
    a = load_result(tmp_path / "a.json").data
    assert a["status"] == "complete" and a["panel"] == "partial" and a["selected_cells"] == ["jund_vs_blue", "jund_vs_jund"]
    assert a["aggregate"]["greedy"]["game_score"] is None and a["aggregate"]["greedy"]["puzzles"]["overall_status"] == "unavailable"
    assert (tmp_path / "a.parts").is_dir() and a["runtime"]["workers"] == 1
    main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json")])
    c = read_json(tmp_path / "c.json")["results"]["greedy"]
    assert all(v["difference"] == 0.0 and v["ci95"] == [0.0, 0.0] for v in c["cells"].values()) and c["puzzles"]["mean"] == 0.0
    assert read_json(tmp_path / "c.json")["latency"]["latency_comparable"] is False  # different worker counts
    # Never overwritten.
    before = (tmp_path / "c.json").read_bytes()
    with pytest.raises(SystemExit):
        main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit):
        run(dev, tmp_path / "a.json")
    assert (tmp_path / "c.json").read_bytes() == before
    # A fair result and a diagnostic one cannot be compared, nor can different cell selections.
    run(dev, tmp_path / "legacy.json", "--cells", "jund_vs_blue,jund_vs_jund", agent="legacy-jund")
    for other in ("legacy.json",):
        with pytest.raises(SystemExit):
            main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / other), "--out", str(tmp_path / "x.json")])
    run(dev, tmp_path / "one.json", "--cells", "jund_vs_blue")
    with pytest.raises(SystemExit):
        main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "one.json"), "--out", str(tmp_path / "y.json")])
    # A different manifest (here: another block count) is refused.
    sys.path.insert(0, str(ROOT / "tools"))
    from puzzle_manifest import manifest
    other = tmp_path / "other.json"
    write_json(other, manifest(other, revision=REVISION, blocks=3, modes=("greedy",)))
    run(other, tmp_path / "three.json")
    with pytest.raises(SystemExit):
        main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "three.json"), "--out", str(tmp_path / "z.json")])
    assert not (tmp_path / "x.json").exists() and not (tmp_path / "y.json").exists() and not (tmp_path / "z.json").exists()


@pytest.mark.slow
def test_resume_reuses_only_clean_chunks_and_never_retries_errors(dev, tmp_path):
    run(dev, tmp_path / "a.json")
    original = load_result(tmp_path / "a.json").data
    # Without --resume an existing parts directory is refused.
    (tmp_path / "a.json").unlink()
    with pytest.raises(SystemExit):
        run(dev, tmp_path / "a.json")
    run(dev, tmp_path / "a.json", "--resume")
    resumed = load_result(tmp_path / "a.json").data
    assert len(resumed["runtime"]["resumed_chunks"]) == 2 and resumed["game_rows"] == original["game_rows"]
    # A chunk written by another candidate is not this run's.
    (tmp_path / "a.json").unlink()
    with pytest.raises(SystemExit):
        run(dev, tmp_path / "a.json", "--resume", agent="legacy-jund")
    # An error row in a chunk is kept as evidence: not reused, not rerun, nothing scored.
    chunk = sorted((tmp_path / "a.parts").glob("greedy-jund_vs_blue-*.json"))[0]
    data = read_json(chunk)
    data["rows"][0].update(status="error", reason="injected", winner=None)
    chunk.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        run(dev, tmp_path / "a.json", "--resume")
    stored = read_json(tmp_path / "a.json")
    assert stored["status"] == "incomplete" and stored["runtime"]["resumed_chunks"] == [] or len(stored["runtime"]["resumed_chunks"]) <= 1
    assert all(v is None for v in stored["aggregate"]["greedy"].values())
    assert stored["counts"]["greedy"]["technical_game_errors"] == 1 and stored["puzzle_rows"] == []
    assert [r["reason"] for r in stored["game_rows"] if r["status"] == "error"] == ["injected"]


# -- engines agree ------------------------------------------------------------------------------

def test_engines_agree_on_games_native(dev):
    pytest.importorskip("mtg_ml_native")
    from mtg_ml.backend import native_available
    if not native_available():
        pytest.skip("mtg_ml_native is not built")
    from mtg_ml.benchmark.release import _rows_equal, play_games, specialist_panel, Learner, calibration_record
    from mtg_ml.benchmark.tactics import specialist_registry
    manifest = load_manifest(dev)
    registry = specialist_registry(REVISION)
    learner = Learner(calibration_record("panel"), specialist_panel(registry), None)
    rows = {e: play_games(manifest, learner, registry, engine=e, workers=1, cells=None, modes=None, parts=dev.parent / f"{e}.parts", key=e, resume=False)[0]
            for e in ("python", "native")}
    assert len(rows["python"]) == 4 * 2 * 4 and not any(r["status"] == "error" for r in rows["python"])
    assert _rows_equal(rows["python"], rows["native"])


def test_corpus_lines_agree_native_and_pass_calibration_checks(dev):
    from mtg_ml.backend import native_available
    from mtg_ml.benchmark.release import _check_lines
    from mtg_ml.benchmark.tactics import specialist_registry
    engines = ["python", "native"] if native_available() else ["python"]
    checks = _check_lines(load_manifest(dev), specialist_registry(REVISION), engines)
    assert [c["id"] for c in checks] == ["witnesses-pass", "mistakes-fail", "passing-through-fails"]
    assert all(c["passed"] for c in checks), [c["problems"] for c in checks]
    assert checks[0]["witness_lines"] == 9 and checks[1]["mistake_lines"] == 11
    assert checks[2]["reviewed_pass_lines"] == ["blue-sleep-redundant"]


def test_passing_through_is_a_defect_unless_passing_is_a_reviewed_line(tmp_path):
    from mtg_ml.benchmark.release import _check_lines

    def keep_delver(puzzle):
        puzzle["objective"] = {"all": [{"object": "delver", "zone": "battlefield", "present": True}]}
        puzzle["cases"][0]["witnesses"] = [{"actions": [CAST, TARGET, PAY, PAY], "note": "not a pass", "learner_only": True}]
        puzzle["cases"][0]["mistakes"] = []
        puzzle.pop("acceptable_first_actions")
    m, reg, _, _ = tactics_setup(tmp_path, keep_delver)
    _, _, defect = _check_lines(m, reg, ["python"])
    assert not defect["passed"] and "corpus defect" in defect["problems"][0] and defect["reviewed_pass_lines"] == []
    # The same goal with a reviewed witness that passes is a legitimate "do nothing" puzzle.

    def passes(puzzle):
        keep_delver(puzzle)
        puzzle["cases"][0]["witnesses"] = [{"actions": [{"player": 0, "kind": "priority", "key": ["pass"]}], "note": "Pass.", "learner_only": True}]
    m, reg, _, _ = tactics_setup(tmp_path / "ok", passes)
    _, _, fine = _check_lines(m, reg, ["python"])
    assert fine["passed"] and fine["reviewed_pass_lines"] == ["synthetic-removal"]


# -- fault injection ----------------------------------------------------------------------------

@pytest.mark.slow
def test_injected_faults_leave_incomplete_runs_with_null_scores(dev, tmp_path):
    from mtg_ml.benchmark.release import _check_faults
    from mtg_ml.benchmark.tactics import specialist_registry
    out = _check_faults(dev, tmp_path / "work", specialist_registry(REVISION), "python", 2)
    assert out["passed"], out["problems"]
    assert set(out["faults"]) == {"exception", "illegal-index", "exception-when-losing"}
    for name, fault in out["faults"].items():
        result = read_json(tmp_path / "work" / f"fault-{name}.json")
        assert fault["status"] == "incomplete" and fault["errors"] > 0
        assert result["status"] == "incomplete" and all(v is None for a in result["aggregate"].values() for v in a.values())
        assert not (tmp_path / "work" / f"fault-{name}-comparison.json").exists()
        replays = [r for r in result["game_rows"] if r["status"] == "error"]
        assert replays and all("replay_reference" in r for r in replays)
    assert read_json(tmp_path / "work" / "fault-control.json")["status"] == "complete"


# -- calibration and archive --------------------------------------------------------------------

@pytest.mark.slow
def test_calibrate_passes_on_the_dev_corpus_and_archives(dev, tmp_path):
    from mtg_ml.backend import native_available
    engines = "python,native" if native_available() else "python"
    cal = tmp_path / "cal.json"
    main(["calibrate", "--manifest", str(dev), "--engines", engines, "--workers", "2", "--blocks", "2", "--out", str(cal)])
    report = read_json(cal)
    assert report["status"] == "passed" and [c["id"] for c in report["checks"]] == [
        "witnesses-pass", "mistakes-fail", "passing-through-fails", "self-compare", "engine-games" if "," in engines else "engine-games", "errors-cannot-inflate"]
    assert all(c["passed"] for c in report["checks"] if c["id"] != "engine-games" or "," in engines)
    run(dev, tmp_path / "a.json")
    out = tmp_path / "archive"
    main(["archive", "build", "--out", str(out), "--manifest", str(dev), "--result", str(tmp_path / "a.json"), "--calibration", str(cal)])
    assert (out / "SHA256SUMS").is_file() and (out / "environment" / "pip-freeze.txt").is_file() and (out / "commands.txt").is_file()
    main(["archive", "verify", str(out)])
    with pytest.raises(SystemExit):
        main(["archive", "build", "--out", str(out), "--manifest", str(dev), "--result", str(tmp_path / "a.json")])
    stored = next((out / "artifacts").rglob("a.json"))
    stored.write_text(stored.read_text().replace('"status": "complete"', '"status": "complete" ', 1))
    with pytest.raises(SystemExit):
        main(["archive", "verify", str(out)])


@pytest.mark.slow
def test_archive_round_trip_recomputes_comparisons(dev, tmp_path):
    from mtg_ml.benchmark import archive
    run(dev, tmp_path / "a.json")
    run(dev, tmp_path / "b.json")
    main(["compare", "--baseline", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json")])
    out = archive.build(tmp_path / "bundle", manifest=dev, results=[tmp_path / "a.json", tmp_path / "b.json"], comparisons=[tmp_path / "c.json"])
    assert archive.verify(out)["files"] > 10
    # A comparison edited to look better, with its hash list repaired, is still caught by recomputation.
    comparison = next((out / "artifacts").rglob("c.json"))
    data = read_json(comparison)
    data["results"]["greedy"]["mean_difference"] = 0.2
    comparison.write_text(json.dumps(data))
    sums = out / "SHA256SUMS"
    sums.write_text("\n".join(line if not line.endswith("c.json") else f"{file_digest(comparison)}  {line.split('  ', 1)[1]}" for line in sums.read_text().splitlines()) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        archive.verify(out)


def test_puzzle_manifest_options(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from puzzle_manifest import manifest
    pilot = manifest(tmp_path / "p.json", revision=REVISION, blocks=400, split="power-pilot", id="pilot-1")
    assert (pilot["split"], pilot["stream"], pilot["id"], pilot["blocks_per_cell"]) == ("power-pilot", "benchmark-v1/power-pilot", "pilot-1", 400)
    assert manifest(tmp_path / "d.json", revision=REVISION)["stream"] == "benchmark-v1/dev"
