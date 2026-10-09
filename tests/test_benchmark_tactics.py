"""Agent-driven tactical puzzle attempts on both engines (PR 4 runner).

The synthetic fixture is the plan's Cast Down vs Delver position; the reviewed
development corpus lives in benchmark/puzzles/dev.
"""

from pathlib import Path

import pytest

from benchmark_fixtures import fixture, registry as synthetic_registry
from mtg_ml.benchmark import ScriptedAdapter, build_result, episodes, load_puzzle
from mtg_ml.benchmark.jsonio import file_digest, read_json, write_json
from mtg_ml.benchmark.tactics import (RandomFactory, ResponseFactory, SelectorFactory, attempt_case, pass_learner,
                                      run_puzzles, specialist_registry, summarize, verify_lines)
from mtg_ml.benchmark.validation import validate
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR

# Explicit engine parametrization (the conftest `engine` fixture is indirect).
pytestmark = pytest.mark.parametrize("engine", ["python", "native"], indirect=True)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmark" / "puzzles" / "dev"
DECKS = {n: {"cards": dict(c)} for n, c in (("jund_wildfire", JUND_WILDFIRE), ("mono_blue_terror", MONO_BLUE_TERROR))}
CAST = {"player": 0, "kind": "priority", "key": ["cast", "Cast Down", "hand", "normal"]}
TARGET = {"player": 0, "kind": "target", "key": ["target", "nonlegendary_creature", "perm", "opponent", "Delver of Secrets"]}
PAY = {"player": 0, "kind": "pay_mana", "key": ["pay", "source", "Swamp", "B"]}
PASS = {"player": 0, "kind": "priority", "key": ["pass"]}


def setup(tmp_path, edit_puzzle=None, edit_scenario=None):
    """The synthetic fixture, optionally edited, with its hash chain rewritten."""
    m, reg = fixture(tmp_path)
    for name, edit in (("scenario.json", edit_scenario), ("puzzle.json", edit_puzzle)):
        if edit is None:
            continue
        data = read_json(tmp_path / name)
        edit(data)
        (tmp_path / name).unlink()
        write_json(tmp_path / name, data)
    puzzle = read_json(tmp_path / "puzzle.json")
    puzzle["cases"][0]["scenario"]["sha256"] = file_digest(tmp_path / "scenario.json")
    (tmp_path / "puzzle.json").unlink()
    write_json(tmp_path / "puzzle.json", puzzle)
    bundle = read_json(tmp_path / "bundle.json")
    bundle["puzzles"][0]["sha256"] = file_digest(tmp_path / "puzzle.json")
    (tmp_path / "bundle.json").unlink()
    write_json(tmp_path / "bundle.json", bundle)
    m.data["puzzles"]["sha256"] = file_digest(tmp_path / "bundle.json")
    m.path.unlink()
    write_json(m.path, m.data)
    p = load_puzzle(tmp_path / "puzzle.json")
    return m, reg, p, p.data["cases"][0]


def attempt(p, case, reg, learner, engine, **kw):
    return attempt_case(p, case, DECKS, learner, ResponseFactory(reg, case["response_policy"]["id"]), mode="greedy", engine=engine, **kw)


def test_outcome_classification(tmp_path, engine):
    _, reg, p, case = setup(tmp_path)
    ok = attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)
    assert (ok["status"], ok["reason"], ok["objective_met"], ok["first_action_correct"]) == ("success", "objective_met", True, True)
    # Equivalent by outcome: removing Delver in the beginning-of-combat step instead.
    later = attempt(p, case, reg, SelectorFactory((PASS, CAST, TARGET, PAY, PAY)), engine)
    assert later["reason"] == "objective_met" and later["first_action_correct"] is False
    assert later["trace_sha256"] != ok["trace_sha256"]
    passed = attempt(p, case, reg, pass_learner, engine)
    assert (passed["status"], passed["reason"]) == ("failure", "objective_failed")
    short = attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine, max_decisions=3)
    assert (short["status"], short["reason"], short["decisions"]) == ("failure", "horizon_exhausted", 3)
    # Deterministic: an identical line repeats exactly.
    assert attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)["trace_sha256"] == ok["trace_sha256"]


class Illegal:
    def reset(self, own_deck, actor_seed=0):
        pass

    def act(self, game):
        return 99


class Boom:
    def reset(self, own_deck):
        pass

    def observe(self, event):
        pass

    def choose(self, view, actions):
        raise RuntimeError("bot crashed")


class BoomResponse(ResponseFactory):
    def __call__(self, seat, mode):
        return ScriptedAdapter(Boom(), seat)


def test_technical_errors_are_not_attempts(tmp_path, engine):
    _, reg, p, case = setup(tmp_path)
    assert attempt(p, case, reg, lambda seat, mode: Illegal(), engine)["reason"] == "illegal_action"
    broken = attempt(p, case, reg, SelectorFactory(({"player": 0, "kind": "priority", "key": ["cast", "Bolt"]},)), engine)
    assert (broken["status"], broken["reason"]) == ("error", "agent_exception")
    crash = attempt_case(p, case, DECKS, pass_learner, BoomResponse(reg, "synthetic-blue@1"), mode="greedy", engine=engine)
    assert (crash["status"], crash["reason"]) == ("error", "response_policy_error") and "bot crashed" in crash["error"]
    assert not crash["objective_met"]


def test_goal_true_initially_is_not_credited_if_lost(tmp_path, engine):
    def keep_delver(puzzle):
        puzzle["objective"] = {"all": [{"object": "delver", "zone": "battlefield", "present": True}]}
        puzzle["cases"][0].pop("witnesses")
    _, reg, p, case = setup(tmp_path, keep_delver)
    assert attempt(p, case, reg, SelectorFactory((CAST, TARGET, PAY, PAY)), engine)["reason"] == "objective_failed"
    assert attempt(p, case, reg, pass_learner, engine)["reason"] == "objective_met"


def test_missing_history_and_ambiguous_selectors_are_rejected(tmp_path, engine, monkeypatch):
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    def brainstorm(scenario):
        me = scenario["initial"]["players"][0]
        me["hand"] = ["Cast Down", "Brainstorm"]
        me["hand_count"] = 2
        me["battlefield"] = ["Swamp", "Swamp", "Island"]
        me["library"] = ["Swamp"] * 6
        me["library_count"] = 6
        scenario["setup_actions"] = [{"player": 0, "kind": "priority", "key": ["cast", "Brainstorm", "hand", "normal"]},
                                     {"player": 0, "kind": "pay_mana", "key": ["pay", "source", "Island", "U"]},
                                     {"player": 0, "kind": "priority", "key": ["pass"]},
                                     {"player": 1, "kind": "priority", "key": ["pass"]},
                                     {"player": 0, "kind": "choose_card", "key": ["put back", "Swamp"]},
                                     {"player": 0, "kind": "choose_card", "key": ["put back", "Swamp"]}]
    (tmp_path / "history").mkdir()
    m, reg, p, case = setup(tmp_path / "history", edit_scenario=brainstorm)
    row = attempt(p, case, reg, pass_learner, engine)
    assert row["reason"] == "engine_error" and "missing history" in row["error"]
    report = validate(m.path, engines=(engine,), registry=reg)
    assert report["status"] == "invalid" and "missing history" in report["errors"][0]

    def two_delvers(scenario):
        scenario["initial"]["players"][1]["battlefield"].append({"name": "Delver of Secrets", "id": "delver-2", "sick": False, "tapped": True})
    (tmp_path / "ambiguous").mkdir()
    _, reg, p, case = setup(tmp_path / "ambiguous", edit_scenario=two_delvers)
    row = attempt(p, case, reg, SelectorFactory((CAST, TARGET)), engine)
    assert row["reason"] == "agent_exception" and "matches 2" in row["error"]
    # Named bindings disambiguate; the objective names the untapped Delver.
    for oid, reason in (("delver", "objective_met"), ("delver-2", "objective_failed")):
        line = (CAST, dict(TARGET, objects=[oid]), PAY, PAY)
        assert attempt(p, case, reg, SelectorFactory(line), engine)["reason"] == reason


def test_validation_runs_learner_lines_mistakes_and_probes(tmp_path, engine, monkeypatch):
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)

    def lines(puzzle):
        c = puzzle["cases"][0]
        c["witnesses"].append({"actions": [PASS, CAST, TARGET, PAY, PAY], "note": "Remove Delver in combat.", "learner_only": True})
        c["mistakes"] = [{"actions": [PASS], "note": "Pass the turn.", "expect": "objective_failed"}]
    m, reg, _, _ = setup(tmp_path, lines)
    report = validate(m.path, engines=(engine,), registry=reg)
    assert report["status"] == "complete", report["errors"]
    tactics = report["puzzles"][0]["cases"][0]["tactics"]
    assert [(x["kind"], x["reason"]) for x in tactics["lines"]] == [("witness", "objective_met"), ("mistake", "objective_failed")]
    assert [x["learner"] for x in tactics["probes"]] == ["pass", "random-0", "random-1", "random-2", "random-3"]

    def wrong_expectation(puzzle):
        puzzle["cases"][0]["mistakes"] = [{"actions": [CAST, TARGET, PAY, PAY], "note": "Not a mistake.", "expect": "objective_failed"}]
    (tmp_path / "bad").mkdir()
    m, reg, _, _ = setup(tmp_path / "bad", wrong_expectation)
    report = validate(m.path, engines=(engine,), registry=reg)
    assert report["status"] == "invalid" and "classified objective_met" in report["errors"][0]


def test_run_puzzles_rows_feed_results(tmp_path, engine):
    m, reg, _, _ = setup(tmp_path)
    cells, modes = ["jund_vs_blue"], ["greedy"]
    rows = run_puzzles(m, SelectorFactory((CAST, TARGET, PAY, PAY)), reg, engine=engine, cells=cells, modes=modes)
    assert [(r["puzzle"], r["status"]) for r in rows] == [("synthetic-removal", "success")]
    games = [{k: v for k, v in s.__dict__.items() if k != "decks"} | dict(status="completed", winner=None, reason="turn_limit", decisions=1, turns=1)
             for s in episodes(m, cells, modes)]
    kw = dict(cells=cells, modes=modes, code_revision=m.data["freeze"]["code_revision"])
    assert build_result(m, candidate(), games, rows, **kw)["status"] == "complete"
    assert summarize(rows)["greedy"]["puzzle_success"] == 1.0
    bad = run_puzzles(m, lambda seat, mode: Illegal(), reg, engine=engine, cells=cells, modes=modes)
    assert build_result(m, candidate(), games, bad, **kw)["status"] == "incomplete"
    summary = summarize(bad)["greedy"]
    assert summary["puzzle_success"] is None and summary["technical_case_errors"] == 1


def candidate():
    return {"checkpoint": "synthetic.pt", "sha256": "0" * 64, "features": 7, "information_contract": "own-list-hidden-opponent-v1"}


@pytest.mark.slow
def test_checkpoint_attempts_reset_memory_and_repeat_across_workers(tmp_path, engine):
    pytest.importorskip("torch")
    from test_benchmark_engine import checkpoint
    from mtg_ml.benchmark.__main__ import CheckpointLearner
    m, reg, _, _ = setup(tmp_path)
    learner = CheckpointLearner(str(checkpoint(tmp_path)), "own-list-hidden-opponent-v1")
    for mode in ("greedy", "sampled"):
        one = run_puzzles(m, learner, reg, engine=engine, cells=["jund_vs_blue"], modes=[mode])
        again = run_puzzles(m, learner, reg, engine=engine, cells=["jund_vs_blue"], modes=[mode])
        two = run_puzzles(m, learner, synthetic_registry(), engine=engine, cells=["jund_vs_blue"], modes=[mode], workers=2)
        strip = [{k: r[k] for k in ("status", "reason", "decisions", "trace_sha256")} for r in one]
        assert all(r["status"] != "error" for r in one)
        assert strip == [{k: r[k] for k in ("status", "reason", "decisions", "trace_sha256")} for r in again]
        assert strip == [{k: r[k] for k in ("status", "reason", "decisions", "trace_sha256")} for r in two]


def corpus_puzzles():
    bundle = read_json(CORPUS / "bundle.json")
    return [load_puzzle(CORPUS / ref["path"]) for ref in bundle["puzzles"]]


def test_corpus_smoke(engine):
    p = next(p for p in corpus_puzzles() if p.data["id"] == "jund-color-sequencing")
    report = verify_lines(p, p.data["cases"][0], DECKS, specialist_registry("a" * 40), (engine,), robustness=1)
    assert [x["reason"] for x in report["lines"]] == ["objective_met", "objective_met", "objective_failed"]


def test_corpus_is_current_and_within_decklists(engine):
    import subprocess
    import sys
    from collections import Counter
    if engine == "native":
        pytest.skip("artifact check is engine independent")
    subprocess.run([sys.executable, str(ROOT / "tools" / "puzzle_corpus.py"), "--check"], check=True)
    for p in corpus_puzzles():
        scenario = read_json(CORPUS / p.data["id"] / "scenario.json")
        learner = p.data["deck"]
        other = "jund_wildfire" if learner == "mono_blue_terror" else "mono_blue_terror"
        for seat, deck in ((scenario["perspective"], learner), (1 - scenario["perspective"], other)):
            zones = scenario["initial"]["players"][seat]
            used = Counter(c if isinstance(c, str) else c["name"]
                           for z in ("hand", "library", "graveyard", "exile", "battlefield") for c in zones[z])
            over = {n: k for n, k in used.items() if k > DECKS[deck]["cards"].get(n, 0)}
            assert not over, (p.data["id"], deck, over)


def test_corpus_lines_and_probes(engine):
    reg = specialist_registry("a" * 40)
    puzzles = corpus_puzzles()
    assert len(puzzles) == 7 and {p.data["split"] for p in puzzles} == {"dev"}
    for p in puzzles:
        for case in p.data["cases"]:
            verify_lines(p, case, DECKS, reg, (engine,))


def test_random_probe_is_seeded(tmp_path, engine):
    _, reg, p, case = setup(tmp_path)
    a = attempt(p, case, reg, RandomFactory(7), engine)
    b = attempt(p, case, reg, RandomFactory(7), engine)
    assert a["trace_sha256"] == b["trace_sha256"] and a["status"] != "error"


def test_cli_puzzles_report_and_no_overwrite(tmp_path, engine, monkeypatch):
    import sys
    from mtg_ml.benchmark.__main__ import main
    sys.path.insert(0, str(ROOT / "tools"))
    from puzzle_manifest import manifest
    monkeypatch.setattr("mtg_ml.benchmark.validation.check_freeze", lambda m: None)
    path = tmp_path / "dev.json"
    write_json(path, manifest(path, revision="a" * 40))
    out = tmp_path / "report.json"
    main(["puzzles", "--manifest", str(path), "--agent", "benchmark-jund@1", "--engine", engine, "--modes", "greedy", "--out", str(out)])
    report = read_json(out)
    assert report["status"] == "complete" and {r["deck"] for r in report["rows"]} == {"jund_wildfire"}
    assert report["summary"]["greedy"]["scored_puzzles"] == 3
    with pytest.raises(SystemExit):
        main(["puzzles", "--manifest", str(path), "--agent", "benchmark-jund@1", "--engine", engine, "--modes", "greedy", "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["puzzles", "--manifest", str(path), "--agent", "benchmark-jund@1", "--cells", "blue_vs_jund", "--out", str(tmp_path / "x.json")])
