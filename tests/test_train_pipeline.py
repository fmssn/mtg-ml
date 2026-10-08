"""Pipelined trainer: the rollout of iteration k+1 runs during the update of
iteration k with the weights of iteration k, workers load versioned
weights-only policy files, latest.pt is written in the background every
`--checkpoint-every` iterations, the pool lives in a collector process that
hands the merged result over in shared memory, and the evaluation runs in a
process of its own, merged into the metrics rows when it ends.
`--pipeline 0` keeps the old sequential order."""

import gc
import json
import multiprocessing as mp
import os
import threading
import time
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.rl import collect  # noqa: E402
from mtg_ml.rl import train as train_mod  # noqa: E402
from mtg_ml.rl.collect import PoolThread, SharedResult, cpu_layout, parse_cpus, release  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import LEARNER, RANDOM, GameSpec, Job, run_job  # noqa: E402
from mtg_ml.rl.train import Trainer, parse_args  # noqa: E402

TINY = ["--hidden", "16", "--games-per-iter", "4", "--workers", "2", "--max-turns", "6"]
NO_EVAL = ["--eval-every", "0"]
SMALL_EVAL = ["--eval-games", "4", "--eval-bo3-matches", "2", "--bench-games", "4", "--bench-greedy-games", "4", "--bench-bo3-matches", "2", "--eval-workers", "1"]
TIMING = {"rollout_s", "queue_s", "update_s", "ppo_s", "publish_s", "wait_s", "wall_s", "decisions_per_s", "eval_s", "eval_lag"}
IN_PROCESS = ["--collector", "thread", "--eval-process", "0"]


@pytest.fixture(autouse=True)
def _free_native_garbage():
    """Earlier tests leave native games in reference cycles. The native objects
    are unsendable, so collect them here on the main thread before a test
    starts threads; a GC pass on one of those threads would free them there."""
    gc.collect()


def _cfg(run, *args):
    return parse_args(["--run", str(run), *TINY, *args])


def _rows(run, name="metrics.jsonl") -> list[dict]:
    return [json.loads(line) for line in (run / name).read_text().splitlines()]


@pytest.mark.parametrize("pipeline", [0, 1, 2])
def test_smoke_and_resume(tmp_path, monkeypatch, pipeline):
    run = tmp_path / "run"
    cfg = _cfg(run, "--pipeline", str(pipeline), "--iterations", "4", "--snapshot-every", "2", "--eval-every", "2", *SMALL_EVAL)
    threads, update, update_threads = torch.get_num_threads(), train_mod.ppo_update, []

    def counting_update(*args, **kwargs):
        update_threads.append(torch.get_num_threads())
        return update(*args, **kwargs)

    cpus = train_mod.available_cpus()[:3]
    monkeypatch.setattr(train_mod, "available_cpus", lambda: cpus)  # the 2 workers leave one core to the update
    monkeypatch.setattr(train_mod, "ppo_update", counting_update)
    before = set(mp.active_children())
    Trainer(cfg).train()
    assert set(update_threads) == {1 if pipeline else threads} and torch.get_num_threads() == threads
    assert not set(mp.active_children()) - before
    rows = _rows(run)
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4] and [r["games_total"] for r in rows] == [4, 8, 12, 16]
    assert all({"rollout_s", "update_s", "wait_s", "wall_s", "policy_lag"} <= r.keys() for r in rows)
    assert [r["policy_lag"] for r in rows] == ([0, 1, 2, 2] if pipeline == 2 else [0, 1, 1, 1] if pipeline else [0, 0, 0, 0])
    assert pipeline or all(r["wait_s"] == 0 for r in rows)
    # the evaluations, merged into the rows of the iterations they evaluated
    assert "eval/random/jund" in rows[1] and rows[3]["bench/jund_vs_bot_n"] == 4 and rows[1]["eval_lag"] >= 0
    assert not any(k.startswith(("eval", "bench")) for r in (rows[0], rows[2]) for k in r)
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00002.pt", "v00003.pt", "v00004.pt"]
    assert sorted(p.name for p in (run / "pool").iterdir()) == ["iter_00000.pt", "iter_00002.pt", "iter_00004.pt"]
    for p in [*(run / "policy").iterdir(), *(run / "pool").iterdir()]:
        assert torch.load(p, weights_only=False).keys() == {"config", "model"}
    ck = torch.load(run / "latest.pt", weights_only=False)  # written at the end, although --checkpoint-every is 10
    assert ck["iteration"] == 4 and ck["games_total"] == 16 and ck["optim"]["state"]
    newest = torch.load(run / "policy" / "v00004.pt", weights_only=False)["model"]
    assert all(torch.equal(ck["model"][k], v) for k, v in newest.items())

    cfg.iterations = 6
    t = Trainer(cfg)
    assert (t.iteration, t.games_total) == (4, 16)
    t.train()
    rows = _rows(run)
    assert [(r["iteration"], r["games_total"], r["policy_lag"]) for r in rows[4:]] == [(5, 20, 0), (6, 24, int(bool(pipeline)))]
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00004.pt", "v00005.pt", "v00006.pt"]


def test_pipelined_with_inference_server(tmp_path):
    """The server loads the versioned policy files by (path, version) too;
    it lives in the collector process with the pool."""
    cfg = _cfg(tmp_path / "run", "--inference", "server", "--server-device", "cpu", "--iterations", "3", "--eval-every", "3", *SMALL_EVAL)
    Trainer(cfg).train()
    rows = _rows(tmp_path / "run")
    assert [(r["games_total"], r["policy_lag"]) for r in rows] == [(4, 0), (8, 1), (12, 1)]
    assert rows[-1]["decisions"] > 0 and "eval/bot/jund" in rows[-1]


def test_failed_update_terminates_the_rollout_in_flight(tmp_path, monkeypatch):
    """An exception while the next rollout runs stops the collector and the
    evaluation instead of waiting on them, and keeps the last complete checkpoint."""
    before = set(mp.active_children())
    t = Trainer(_cfg(tmp_path / "run", "--iterations", "3", "--eval-every", "1", *SMALL_EVAL))
    update, calls = train_mod.ppo_update, []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("update failed")
        return update(*args, **kwargs)

    monkeypatch.setattr(train_mod, "ppo_update", fail_second)
    with pytest.raises(RuntimeError, match="update failed"):
        t.train()
    assert not set(mp.active_children()) - before
    assert torch.load(tmp_path / "run" / "latest.pt", weights_only=False)["iteration"] == 0


def test_collector_serialization_failure_is_reported():
    collector = collect.PoolProcess(1)
    try:
        pending = collector.call(lambda pool: None)
        with pytest.raises(RuntimeError, match="submission failed"):
            pending.wait()
    finally:
        collector.terminate()


def test_shared_result_round_trip(tmp_path):
    """The merged result parked by the collector and mapped by the trainer
    trains exactly like the result itself."""
    net = PolicyNet(hidden=16)
    path = str(tmp_path / "p.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    specs = [GameSpec(s, (LEARNER, LEARNER)) for s in range(3)] + [GameSpec(9, (LEARNER, RANDOM))]
    res = run_job(Job(specs, path, 1, shaping=0.1, max_turns=8))
    got = SharedResult(res).attach()
    assert len(got.samples) == len(res.samples) and list(got.samples) == list(res.samples)
    assert list(got.actions) == res.actions and list(got.kinds) == res.kinds and got.lengths == res.lengths and got.games == res.games
    assert torch.allclose(torch.tensor(list(got.logps)), torch.tensor(res.logps))
    stats = []
    for data in (res, got):
        n = PolicyNet(hidden=16)
        n.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(n.parameters(), lr=1e-3)
        stats.append((ppo_update(n, opt, data, PPOConfig(epochs=2, minibatch=64), gen=torch.Generator().manual_seed(0)), n.state_dict()))
    assert stats[0][0] == stats[1][0]
    assert all(torch.equal(stats[0][1][k], stats[1][1][k]) for k in stats[0][1])
    release(got)
    assert len(got.samples) == 0 and not collect._UNMAPPED


def test_cpu_layout(monkeypatch):
    monkeypatch.setattr(collect, "gpu_numa_cpus", lambda device: (2, tuple(range(32, 48))))
    lay = cpu_layout(27, "cuda", avail=range(32, 60))  # the box: GPU 4 on node 2 (cores 32-47), taskset 32-59
    assert lay.trainer == (32,) and lay.workers == tuple(range(33, 60)) and lay.shared_eval and lay.evaluator == lay.workers
    assert lay.collector == tuple(range(48, 60))  # off the trainer's node
    lay = cpu_layout(23, "cuda", avail=range(32, 60))
    assert lay.trainer == (32,) and lay.evaluator == (33, 34, 35, 36) and lay.workers == tuple(range(37, 60)) and not lay.shared_eval
    lay = cpu_layout(20, "cuda", server=True, evaluation=False, avail=range(28, 60))  # workers off the GPU's node first
    assert lay.server == (33,) and lay.workers == (*range(28, 32), *range(44, 60)) and lay.trainer == (32, *range(34, 44))
    lay = cpu_layout(4, "cuda", trainer_cpus="40-41", worker_cpus="50-53", eval_cpus="54", avail=range(32, 60))
    assert (lay.trainer, lay.workers, lay.evaluator) == ((40, 41), (50, 51, 52, 53), (54,))
    lay = cpu_layout(8, avail=range(4))  # oversubscribed: workers unpinned
    assert lay.trainer and lay.workers == ()
    assert parse_cpus("32-34,40") == (32, 33, 34, 40) and collect.fmt_cpus((32, 33, 34, 40)) == "32-34,40"


def test_training_layout_reserves_every_gpu_process(monkeypatch):
    monkeypatch.setattr(train_mod, "available_cpus", lambda: tuple(range(16)))
    cfg = train_mod.TrainConfig(workers=4, device="cuda:0", learner_devices="cuda:0,cuda:1", learner_cpus="0;1",
                               inference="server", server_devices="cuda:2,cuda:3", server_cpus="2;3")
    lay, lc, sc = train_mod.training_cpu_layout(cfg, False)
    assert lc == ((0,), (1,)) and sc == ((2,), (3,))
    assert lay.trainer == (0,) and lay.server == (2, 3)
    assert not set(lay.workers) & {0, 1, 2, 3}
    cfg.worker_cpus = "1,4-6"
    with pytest.raises(ValueError, match="overlap"):
        train_mod.training_cpu_layout(cfg, False)


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

    monkeypatch.setattr(train_mod, "create_pool", lambda workers, inference="local", server=None, **kw: Pool(1))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield started
    torch.set_num_threads(threads)


def test_sequential_runs_are_reproducible(tmp_path, in_process):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        Trainer(_cfg(run, "--pipeline", "0", "--iterations", "3", "--max-turns", "20", *IN_PROCESS, *NO_EVAL)).train()
    a, b = ([json.dumps({k: v for k, v in r.items() if k not in TIMING}, sort_keys=True) for r in _rows(run)] for run in runs)
    assert a == b and len(a) == 3
    wa, wb = (torch.load(run / "latest.pt", weights_only=False)["model"] for run in runs)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


@pytest.mark.parametrize("pipeline", [0, 1])
def test_rollouts_play_the_newest_weights_or_one_update_behind(tmp_path, in_process, pipeline):
    Trainer(_cfg(tmp_path / "run", "--pipeline", str(pipeline), "--iterations", "3", "--eval-every", "2", *IN_PROCESS,
                 "--eval-games", "4", "--eval-bo3-matches", "0", "--bench-games", "0", "--bench-greedy-games", "0", "--bench-bo3-matches", "0")).train()
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
    cfg = _cfg(tmp_path / "run", "--iterations", "3", "--checkpoint-every", "1", *IN_PROCESS, *NO_EVAL)
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


def test_lag2_durable_checkpoint_pins_and_exact_pfsp_resume(tmp_path, in_process, monkeypatch):
    run = tmp_path / "run"
    cfg = _cfg(run, "--pipeline", "2", "--iterations", "6", "--checkpoint-every", "2", "--snapshot-every", "1",
               "--pool-sampling", "pfsp", "--self-play-frac", "0", "--matchup", "jund_blue,jund_madness", *IN_PROCESS, *NO_EVAL)
    real_save = train_mod._save

    def fail_successor(obj, path):
        if path.endswith("latest.pt") and obj["iteration"] == 4:
            raise RuntimeError("checkpoint disk failure")
        return real_save(obj, path)

    monkeypatch.setattr(train_mod, "_save", fail_successor)
    with pytest.raises(RuntimeError, match="checkpoint disk failure"):
        Trainer(cfg).train()
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert ck["iteration"] == 2
    queued = ck["pending_rollouts"]
    assert [d["it"] for d in queued] == [2, 3]
    assert all(os.path.exists(d["job"].learner_path) for d in queued)
    assert os.path.basename(queued[0]["job"].learner_path) == "v00000.pt"  # older than KEEP_POLICIES
    before = [r for r in in_process if r.train]
    monkeypatch.setattr(train_mod, "_save", real_save)
    n = len(in_process)
    Trainer(cfg).train()
    replayed = [r for r in in_process[n:] if r.train]
    assert [(r.policy, r.games) for r in replayed[:2]] == [(r.policy, r.games) for r in before[2:4]]
    assert torch.load(run / "latest.pt", weights_only=False)["pending_rollouts"] == []


@pytest.mark.parametrize("pipeline", [0, 1, 2])
def test_checkpoint_every_and_resume_after_a_crash(tmp_path, in_process, monkeypatch, pipeline):
    """latest.pt only every --checkpoint-every iterations: a run killed in
    the update of iteration 5 restarts from iteration 4, drops the rows and
    the snapshots of the lost iterations, and then plays the same games as a
    run that was never interrupted (unpipelined also the same updates: the
    checkpoint keeps torch's global rng, which samples actions and shuffles
    minibatches)."""
    args = ["--pipeline", str(pipeline), "--iterations", "6", "--checkpoint-every", "2", "--snapshot-every", "3", "--max-turns", "20", *IN_PROCESS, *NO_EVAL]
    Trainer(_cfg(tmp_path / "ref", *args)).train()
    ref_games = [r.games for r in in_process if r.train]
    assert len(ref_games) == 6
    saves, real_save = [], train_mod._save
    monkeypatch.setattr(train_mod, "_save", lambda obj, path: saves.append(os.path.basename(path)) or real_save(obj, path))
    update, calls = train_mod.ppo_update, []

    def killed_in_the_fifth(*a, **kw):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("killed")
        return update(*a, **kw)

    run = tmp_path / "run"
    monkeypatch.setattr(train_mod, "ppo_update", killed_in_the_fifth)
    with pytest.raises(RuntimeError, match="killed"):
        Trainer(_cfg(run, *args)).train()
    assert saves.count("latest.pt") == 3  # iterations 0, 2, 4
    assert torch.load(run / "latest.pt", weights_only=False)["iteration"] == 4
    monkeypatch.setattr(train_mod, "ppo_update", update)
    for stale in ("pool/iter_00006.pt", "policy/v00005.pt", "policy/v00006.pt", "policy/v00007.pt"):  # as if from lost iterations
        torch.save({"stale": True}, run / stale)
    with open(run / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"iteration": 5, "lost": True}) + "\n")
    t = Trainer(_cfg(run, *args))
    assert (t.iteration, t.games_total) == (4, 16) and not (run / "pool" / "iter_00006.pt").exists()
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00002.pt", "v00003.pt", "v00004.pt"]
    assert [r["iteration"] for r in _rows(run, "metrics-dropped.jsonl")] == [5]
    n = len(in_process)
    t.train()
    assert [r.games for r in in_process[n:] if r.train] == ref_games[4:]
    strip = lambda r: {k: v for k, v in r.items() if k not in TIMING}  # noqa: E731
    rows, ref = _rows(run), _rows(tmp_path / "ref")
    if not pipeline:  # pipelined, the rollout thread and the update draw from torch's rng in varying order
        assert [strip(r) for r in rows] == [strip(r) for r in ref]
        wa, wb = (torch.load(r / "latest.pt", weights_only=False)["model"] for r in (run, tmp_path / "ref"))
        assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert [(r["iteration"], r["games_total"], r["pool_size"]) for r in rows] == [(r["iteration"], r["games_total"], r["pool_size"]) for r in ref]
    assert torch.load(run / "latest.pt", weights_only=False)["iteration"] == 6
    assert sorted(p.name for p in (run / "pool").iterdir()) == ["iter_00000.pt", "iter_00003.pt", "iter_00006.pt"]


def test_evaluation_never_blocks_the_loop(tmp_path, in_process, monkeypatch):
    """With --eval-process 1 the loop goes on while an evaluation runs; the
    policy file it reads stays until it is done although newer ones are
    published; a newer due evaluation waits and replaces an older waiting
    one; results land in the rows of the iterations they evaluated."""
    run = tmp_path / "run"
    first = threading.Event()

    def fake_eval(pool, policy, pool0, version, *args):
        if version == 2:  # the evaluation of iteration 1 runs until 5 more iterations have been written
            first.set()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and (not (run / "metrics.jsonl").exists() or len((run / "metrics.jsonl").read_text().splitlines()) < 5):
                time.sleep(0.01)
        return {"eval/version": version, "eval/policy_exists": os.path.exists(policy)}

    monkeypatch.setattr(train_mod, "PoolProcess", lambda *a, **kw: PoolThread(ThreadPool(1)))
    monkeypatch.setattr(train_mod, "evaluate_policy", fake_eval)
    Trainer(_cfg(run, "--iterations", "7", "--eval-every", "1", "--collector", "thread")).train()
    rows = {r["iteration"]: r for r in _rows(run)}
    assert first.is_set() and len(rows) == 7
    assert rows[1]["eval/version"] == 2 and rows[1]["eval/policy_exists"] and rows[1]["eval_lag"] >= 4
    evaluated = [k for k, r in rows.items() if "eval/version" in r]
    assert all(rows[k]["eval/version"] == k + 1 and rows[k]["eval/policy_exists"] for k in evaluated)
    assert 2 not in evaluated and 7 in evaluated  # 2 was replaced while 1 ran; the last one is waited for at the end
    assert sorted(p.name for p in (run / "policy").iterdir()) == ["v00005.pt", "v00006.pt", "v00007.pt"]


def test_amend_merges_into_the_newest_row(tmp_path):
    path = str(tmp_path / "m.jsonl")
    for r in ({"iteration": 1, "a": 1}, {"iteration": 2, "a": 2}, {"iteration": 3, "a": 3}):
        train_mod._append(path, r)
    train_mod._amend(path, 2, {"eval/x": 0.5})
    train_mod._amend(path, 9, {"eval/x": 0.1})
    rows = [json.loads(ln) for ln in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows == [{"iteration": 1, "a": 1}, {"iteration": 2, "a": 2, "eval/x": 0.5}, {"iteration": 3, "a": 3}, {"iteration": 9, "eval/x": 0.1}]


def test_resume_ignores_and_removes_partial_writes(tmp_path, in_process):
    """A run killed inside `_save` leaves `*.tmp` files; the pool must not
    pick one up as its newest snapshot."""
    run = tmp_path / "run"
    cfg = _cfg(run, "--iterations", "1", *IN_PROCESS, *NO_EVAL)
    Trainer(cfg).train()
    tmps = [run / "pool" / "iter_00009.pt.tmp", run / "policy" / "v00009.pt.tmp", run / "latest.pt.tmp"]
    for f in tmps:
        f.write_bytes(b"partial")
    t = Trainer(cfg)
    assert [os.path.basename(p) for p in t.pool] == ["iter_00000.pt"]
    assert not any(f.exists() for f in tmps)
    t.train()


def test_resume_restores_torch_rng_and_keeps_the_ppo_lr(tmp_path, in_process):
    run = tmp_path / "run"
    Trainer(_cfg(run, "--iterations", "1", *IN_PROCESS, *NO_EVAL)).train()
    ck = torch.load(run / "latest.pt", weights_only=False)
    torch.manual_seed(12345)  # what a resumed run must not continue from
    t = Trainer(_cfg(run, "--iterations", "1", "--ppo-lr", "1e-5", *IN_PROCESS, *NO_EVAL))
    assert torch.equal(torch.get_rng_state(), ck["torch_rng"])
    assert all(g["lr"] == 1e-5 for g in t.opt.param_groups)
    t.train()


def test_training_seeds_and_metrics(tmp_path, in_process):
    """Training games never reuse an evaluation or benchmark seed; win_vs_pool
    counts only games against pool snapshots, win_vs_bot those against the bot."""
    from mtg_ml.match import game_seed
    from mtg_ml.rl.evaluate import EVAL_SEED

    run = tmp_path / "run"
    t = Trainer(_cfg(run, "--iterations", "2", "--games-per-iter", "8", "--self-play-frac", "0", "--bot-frac", "0.5", *IN_PROCESS, *NO_EVAL))
    t.train()
    seeds = [g[0] for r in in_process if r.train for g in r.games]
    assert min(seeds) >= train_mod.TRAIN_SEED_BASE > game_seed(EVAL_SEED + 10**6, 3)
    assert all(sp.seed >= train_mod.TRAIN_SEED_BASE for it in (9, 39) for sp in t._train_specs(it))  # it=9 used to hit EVAL_SEED + s
    assert all({"win_vs_pool", "win_vs_bot"} <= r.keys() for r in _rows(run))


def test_request_ints_for_sizes_the_server_slot_from_the_games_per_request():
    from mtg_ml.rl.train import request_ints_for

    assert request_ints_for(2048, 13) == 1 << 22  # the overnight run: 79 decisions per request, 1.3M ints seen
    assert request_ints_for(256, 63) == 1 << 18  # many workers: the floor
    assert request_ints_for(2048, 1, groups=1) >= 2048 * (1 << 15)


def test_new_deck_against_a_fixed_opponent_from_its_weights(tmp_path):
    """--matchup with exploiter mode: the learner plays only Red Madness
    (seat 1 of jund_madness) against a frozen checkpoint, starting from its weights."""
    src = tmp_path / "src"
    Trainer(_cfg(src, "--iterations", "1", *NO_EVAL, *IN_PROCESS)).train()
    jund = str(src / "pool" / "iter_00000.pt")
    run = tmp_path / "red"
    cfg = _cfg(run, "--iterations", "2", "--matchup", "jund_madness", "--exploit", jund, "--exploit-deck", "red", "--hidden", "32",
               *IN_PROCESS, *SMALL_EVAL, "--eval-every", "2")  # fmt: skip
    t = Trainer(cfg)
    init = torch.load(jund, weights_only=False)
    assert t.net.config["hidden"] == init["config"]["hidden"] == 16  # the checkpoint's network, not --hidden 32
    assert all(torch.equal(a, b) for a, b in zip(t.net.state_dict().values(), init["model"].values()))
    specs = t._train_specs(0)
    assert all(sp.seats == (os.path.abspath(jund), LEARNER) and sp.matchup == "jund_madness" for sp in specs)
    t.train()
    rows = _rows(run)
    assert [r["iteration"] for r in rows] == [1, 2] and "win_vs_main" in rows[0]
    assert "eval/opponent/red" in rows[1] and "eval/random/red" in rows[1] and "eval/random/jund" not in rows[1]
    assert rows[1]["bench/red_vs_bot_n"] == 4
    with pytest.raises(ValueError):
        Trainer(_cfg(tmp_path / "bad", "--exploit", jund, "--exploit-deck", "red"))  # no red deck in jund_blue


def test_matchup_mix_draws_each_game_and_benchmarks_every_seat(tmp_path):
    """--matchup a,b,c: each training game's matchup is drawn by weight; the
    first matchup gets the full evaluation, the others a benchmark per seat."""
    run = tmp_path / "mix"
    t = Trainer(_cfg(run, "--iterations", "1", "--matchup", "jund_blue,jund_madness,blue_madness", *IN_PROCESS, *SMALL_EVAL, "--eval-every", "1"))
    specs = t._train_specs(0)
    assert {sp.matchup for sp in specs} == {"jund_blue", "jund_madness", "blue_madness"}
    t.spec_matchup.clear()
    t.train()
    row = _rows(run)[0]
    assert abs(sum(row[f"matchup_share/{m}"] for m in ("jund_blue", "jund_madness", "blue_madness")) - 1) < 1e-9
    assert "bench/jund_vs_bot" in row and "eval/random/blue" in row
    for key in ("jund_madness/jund", "jund_madness/red", "blue_madness/blue", "blue_madness/red"):
        assert f"bench/{key}_vs_bot_n" in row
    assert not t.spec_matchup  # every game's entry consumed
    single = Trainer(_cfg(tmp_path / "one", "--iterations", "1", *IN_PROCESS, *NO_EVAL))
    assert {sp.matchup for sp in single._train_specs(0)} == {"jund_blue"} and not single.spec_matchup
    with pytest.raises(ValueError):
        Trainer(_cfg(tmp_path / "bad", "--matchup", "jund_blue,jund_madness", "--exploit", str(run / "pool" / "iter_00000.pt")))


def test_restore_torch_rng_accepts_states_moved_by_map_location():
    from mtg_ml.rl.train import restore_torch_rng

    state = torch.get_rng_state()
    a = torch.rand(3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # what torch.load(latest.pt, map_location=cfg.device) hands back: the ByteTensor on the training device
    restore_torch_rng({"torch_rng": state.to(dev), "cuda_rng": [s.to(dev) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None})
    assert torch.equal(torch.rand(3), a)
    restore_torch_rng({})  # older checkpoints: nothing to restore
