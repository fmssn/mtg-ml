"""Training tracker: discovery, incremental metrics reading and ledger loading (no GPU, no ssh)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "apps" / "training-dashboard"
spec = importlib.util.spec_from_file_location("tracker", APP / "tracker.py")
tracker = importlib.util.module_from_spec(spec)
sys.modules["tracker"] = tracker
spec.loader.exec_module(tracker)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def row(i, **extra):
    return dict(iteration=i, games=100, games_total=100 * i, decisions=5000, wall_s=10.0, elapsed_s=10.0 * i, entropy=1.0 / i, **extra)


def make_run(path, n=5, evals=()):
    path.mkdir(parents=True)
    rows = [row(i) for i in range(1, n + 1)]
    for i in evals:
        rows[i - 1].update({"bench/jund_vs_bot": 0.6, "bench/jund_vs_bot_ci": [0.5, 0.7], "bench/jund_vs_bot_greedy": 0.7,
                            "ladder/elo": 12.5, "ladder/elo_se": 3.0})
    write_rows(path / "metrics.jsonl", rows)
    return path


def test_reader_is_incremental_and_ignores_partial_line(tmp_path):
    run = make_run(tmp_path / "r", 3)
    reader = tracker.RunReader()
    assert len(reader.read(run / "metrics.jsonl")[0]) == 3
    with open(run / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row(4)) + "\n" + '{"iteration": 5, "ga')
    rows, _ = reader.read(run / "metrics.jsonl")
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4]
    with open(run / "metrics.jsonl", "a") as f:
        f.write('mes": 100}\n')
    assert [r["iteration"] for r in reader.read(run / "metrics.jsonl")[0]][-1] == 5


def test_reader_rereads_after_trainer_rewrites_file(tmp_path):
    run = make_run(tmp_path / "r", 4)
    reader = tracker.RunReader()
    assert reader.read(run / "metrics.jsonl")[1] == []
    rows = [row(i) for i in range(1, 5)]
    rows[1]["ladder/elo"] = 7.0
    tmp = run / "m.tmp"
    write_rows(tmp, rows)
    os.replace(tmp, run / "metrics.jsonl")  # new inode, as the trainer does
    _, evals = reader.read(run / "metrics.jsonl")
    assert [e["iteration"] for e in evals] == [2] and evals[0]["elo"] == 7.0


def test_evaluation_splits_sampled_and_greedy():
    ev = tracker.evaluation({"iteration": 3, "bench/jund_vs_bot": 0.6, "bench/jund_vs_bot_n": 1000,
                             "bench/jund_vs_bot_greedy": 0.7, "ladder/elo": 5.0})
    assert ev["bench"] == {"jund_vs_bot": 0.6} and ev["greedy"] == {"jund_vs_bot": 0.7} and ev["elo"] == 5.0
    assert tracker.evaluation({"iteration": 3, "entropy": 1.0}) is None


def test_discovers_plain_runs_and_campaign_arms(tmp_path):
    make_run(tmp_path / "runs" / "plain", 5, evals=(3,))
    camp = tmp_path / "campaigns" / "c1"
    arm = make_run(camp / "lr1", 4)
    (camp / "campaign.json").write_text(json.dumps([{"name": "lr1", "arm": "lr1", "run": str(arm), "target_games": 400,
                                                     "command": ["python", "-m", "mtg_ml.rl.train", "--ppo-lr", "1e-4"], "code_ref": "abc"}]))
    (tmp_path / "runs" / "empty").mkdir()
    state = tracker.build_state([str(tmp_path / "runs" / "*"), str(tmp_path / "campaigns" / "*")], active={}, gpus=[])
    by = {r["name"]: r for r in state["runs"]}
    assert set(by) == {"plain", "lr1"}
    assert by["lr1"]["campaign"] == "c1" and by["lr1"]["flags"]["ppo-lr"] == "1e-4"
    assert by["lr1"]["status"] == "finished" and by["lr1"]["games"] == 400
    assert by["plain"]["evals"][0]["elo"] == 12.5 and by["plain"]["status"] in ("running", "stopped")
    assert state["campaigns"][0]["runs"] == ["c1/lr1"]


def test_active_pid_marks_running_and_old_idle_runs_are_hidden(tmp_path):
    live = make_run(tmp_path / "live", 3)
    old = make_run(tmp_path / "old", 3)
    past = 1_000_000
    os.utime(old / "metrics.jsonl", (past, past))
    os.utime(live / "metrics.jsonl", (past, past))
    state = tracker.build_state([str(tmp_path / "*")], active={str(live.resolve()): (42, ["x", "--total-games", "1000"])}, gpus=[])
    assert [r["name"] for r in state["runs"]] == ["live"] and state["hidden"] == 1
    assert state["runs"][0]["status"] == "running" and state["runs"][0]["pid"] == 42
    assert state["runs"][0]["target_games"] == 1000 and state["runs"][0]["eta_s"] > 0
    # explicitly named directories are always shown
    state = tracker.build_state([], explicit=[str(old)], active={}, gpus=[])
    assert [r["name"] for r in state["runs"]] == ["old"]


def test_state_is_strict_json_and_downsampled(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    rows = [row(i) for i in range(1, 1001)]
    rows[5]["entropy"] = float("nan")
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    state = tracker.build_state([str(run)], active={}, gpus=[])
    assert len(state["runs"][0]["series"]) <= tracker.MAX_POINTS + 1
    json.dumps(tracker.clean(state), allow_nan=False)


def test_ledger_loading_normalizes_and_skips_bad_lines(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join([
        json.dumps({"id": "a", "date": "2026-10-06", "parent": None, "games": 100, "verdict": "parent",
                    "results": {"bench_sampled_2000": 0.65, "bench_sampled_1000_old_engine": 0.6, "bench_greedy_2000": 0.7, "elo_L1": 103.2}}),
        "not json",
        json.dumps({"id": "b", "parent": "a", "verdict": "reject (no gain)", "results": None}),
        json.dumps({"no_id": 1}),
    ]))
    rows = tracker.load_ledger(p)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["elo"] == 103.2 and rows[0]["sampled"] == 0.65 and rows[0]["greedy"] == 0.7
    assert rows[1]["elo"] is None and rows[1]["verdict_class"] == "bad"
    assert tracker.load_ledger(tmp_path / "missing.jsonl") == []


def test_repo_ledger_loads():
    rows = tracker.load_ledger(Path(__file__).resolve().parents[1] / "docs" / "experiments" / "ledger.jsonl")
    assert rows and all(r["id"] for r in rows)
