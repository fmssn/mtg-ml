"""Pipelined trainer: the rollout of iteration k+1 runs during the update of
iteration k with the weights of iteration k, workers load versioned
weights-only policy files, and latest.pt is written in the background.
`--pipeline 0` keeps the old sequential order."""

import json
import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.rl import train as train_mod  # noqa: E402
from mtg_ml.rl.train import Trainer, parse_args  # noqa: E402

TINY = ["--hidden", "16", "--games-per-iter", "4", "--workers", "2", "--max-turns", "6"]
NO_EVAL = ["--eval-every", "0"]
SMALL_EVAL = ["--eval-games", "4", "--eval-bo3-matches", "2", "--bench-games", "4", "--bench-bo3-matches", "2"]
TIMING = {"rollout_s", "update_s", "wait_s", "wall_s", "decisions_per_s"}


def _cfg(run, *args):
    return parse_args(["--run", str(run), *TINY, *args])


def _rows(run) -> list[dict]:
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("pipeline", [0, 1])
def test_smoke_and_resume(tmp_path, monkeypatch, pipeline):
    run = tmp_path / "run"
    cfg = _cfg(run, "--pipeline", str(pipeline), "--iterations", "4", "--snapshot-every", "2", "--eval-every", "2", *SMALL_EVAL)
    threads, update, update_threads = torch.get_num_threads(), train_mod.ppo_update, []

    def counting_update(*args, **kwargs):
        update_threads.append(torch.get_num_threads())
        return update(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_cpus", lambda: 3)  # the 2 workers leave one core to the update
    monkeypatch.setattr(train_mod, "ppo_update", counting_update)
    Trainer(cfg).train()
    assert set(update_threads) == {1 if pipeline else threads} and torch.get_num_threads() == threads
    rows = _rows(run)
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4] and [r["games_total"] for r in rows] == [4, 8, 12, 16]
    assert all(TIMING | {"policy_lag"} <= r.keys() for r in rows)
    assert [r["policy_lag"] for r in rows] == ([0, 1, 1, 1] if pipeline else [0, 0, 0, 0])
    assert pipeline or all(r["wait_s"] == 0 for r in rows)
    assert "eval/random/jund" in rows[1] and rows[3]["bench/jund_vs_bot_n"] == 4
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00002.pt", "v00003.pt", "v00004.pt"]
    assert sorted(p.name for p in (run / "pool").iterdir()) == ["iter_00000.pt", "iter_00002.pt", "iter_00004.pt"]
    for p in [*(run / "policy").iterdir(), *(run / "pool").iterdir()]:
        assert torch.load(p, weights_only=False).keys() == {"config", "model"}
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert ck["iteration"] == 4 and ck["games_total"] == 16 and ck["optim"]["state"]
    newest = torch.load(run / "policy" / "v00004.pt", weights_only=False)["model"]
    assert all(torch.equal(ck["model"][k], v) for k, v in newest.items())

    cfg.iterations = 6
    t = Trainer(cfg)
    assert (t.iteration, t.games_total) == (4, 16)
    t.train()
    rows = _rows(run)
    assert [(r["iteration"], r["games_total"], r["policy_lag"]) for r in rows[4:]] == [(5, 20, 0), (6, 24, int(pipeline))]
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00004.pt", "v00005.pt", "v00006.pt"]


def test_pipelined_with_inference_server(tmp_path):
    """The server loads the versioned policy files by (path, version) too."""
    cfg = _cfg(tmp_path / "run", "--inference", "server", "--server-device", "cpu", "--iterations", "3", "--eval-every", "3", *SMALL_EVAL)
    Trainer(cfg).train()
    rows = _rows(tmp_path / "run")
    assert [(r["games_total"], r["policy_lag"]) for r in rows] == [(4, 0), (8, 1), (12, 1)]
    assert rows[-1]["decisions"] > 0 and "eval/bot/jund" in rows[-1]


def test_failed_update_terminates_the_rollout_in_flight(tmp_path, monkeypatch):
    """An exception while the next rollout runs stops the workers instead of
    waiting on them, and keeps the last complete checkpoint."""
    before = set(mp.active_children())
    t = Trainer(_cfg(tmp_path / "run", "--iterations", "3", *NO_EVAL))

    def fail(*args, **kwargs):
        raise RuntimeError("update failed")

    monkeypatch.setattr(train_mod, "ppo_update", fail)
    with pytest.raises(RuntimeError, match="update failed"):
        t.train()
    assert not set(mp.active_children()) - before
    assert torch.load(tmp_path / "run" / "latest.pt", weights_only=False)["iteration"] == 0


@pytest.fixture
def in_process(monkeypatch):
    """Rollouts and evaluations in one thread of this process, on one torch
    thread, so that a run is reproducible (worker processes start from random
    torch seeds, and multithreaded CPU kernels sum in varying order). Yields
    the rollouts started, in order: train or eval, (policy file, version), games."""
    started = []

    def record(jobs):
        (policy,) = {(os.path.basename(j.learner_path), j.learner_version) for j in jobs}
        games = sorted((g.seed, tuple(map(os.path.basename, g.seats)), g.match_game) for j in jobs for g in j.games)
        started.append(SimpleNamespace(train=jobs[0].record, policy=policy, games=games))

    class Pool(ThreadPool):  # no `claim`, so rollout.play takes the fixed-split `map` path
        def map(self, func, jobs, *args, **kwargs):
            record(jobs)
            return super().map(func, jobs, *args, **kwargs)

    monkeypatch.setattr(train_mod, "create_pool", lambda workers, inference="local", server=None: Pool(1))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield started
    torch.set_num_threads(threads)


def test_sequential_runs_are_reproducible(tmp_path, in_process):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        Trainer(_cfg(run, "--pipeline", "0", "--iterations", "3", "--max-turns", "20", *NO_EVAL)).train()
    a, b = ([json.dumps({k: v for k, v in r.items() if k not in TIMING}, sort_keys=True) for r in _rows(run)] for run in runs)
    assert a == b and len(a) == 3
    wa, wb = (torch.load(run / "latest.pt", weights_only=False)["model"] for run in runs)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


@pytest.mark.parametrize("pipeline", [0, 1])
def test_rollouts_play_the_newest_weights_or_one_update_behind(tmp_path, in_process, pipeline):
    Trainer(_cfg(tmp_path / "run", "--pipeline", str(pipeline), "--iterations", "3", "--eval-every", "2",
                 "--eval-games", "4", "--eval-bo3-matches", "0", "--bench-games", "0", "--bench-bo3-matches", "0")).train()
    train = [r.policy for r in in_process if r.train]
    if pipeline:  # the games of iteration k+1 are started before the update of iteration k
        assert train == [("v00000.pt", 1), ("v00000.pt", 1), ("v00001.pt", 2)]
    else:
        assert train == [("v00000.pt", 1), ("v00001.pt", 2), ("v00002.pt", 3)]
    assert {r.policy for r in in_process if not r.train} == {("v00002.pt", 3)}  # evaluation after iteration 2


def test_resume_replays_the_games_of_a_lost_rollout(tmp_path, in_process, monkeypatch):
    """The checkpoint of iteration k keeps the rng from before the games of
    iteration k were drawn, although with pipelining they are drawn (and
    played) during the update before it: a restarted run plays the same games."""
    cfg = _cfg(tmp_path / "run", "--iterations", "3", *NO_EVAL)
    update, calls = train_mod.ppo_update, []

    def update_until_the_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("killed")
        return update(*args, **kwargs)

    monkeypatch.setattr(train_mod, "ppo_update", update_until_the_third)
    with pytest.raises(RuntimeError, match="killed"):
        Trainer(cfg).train()
    lost = in_process[-1]
    t = Trainer(cfg)
    assert t.iteration == 2
    t.train()
    again = in_process[-1]
    assert lost.policy == ("v00001.pt", 2) and again.policy == ("v00002.pt", 3)
    assert again.games == lost.games and len(lost.games) == 4
