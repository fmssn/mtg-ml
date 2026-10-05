"""Central inference server: rollouts through the server record exactly what
the network computes when it replays the recorded trajectories from scratch
(sequence mode, as in the PPO update). That checks the server-side hidden
state table (slots, fresh flags), the cross-worker batching and the scatter
of results back to the right decision."""

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.backend import native_available  # noqa: E402
from mtg_ml.rl.inference import InferenceServer, ServerConfig  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, RANDOM, GameSpec, Job, run_job, split_games  # noqa: E402


def _ckpt(tmp_path, memory="gru", seed=0):
    torch.manual_seed(seed)
    net = PolicyNet(hidden=32, memory=memory)
    path = str(tmp_path / f"net_{memory}_{seed}.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


def _replay_matches(net, res, tol=1e-4):
    net.eval()
    with torch.no_grad():
        logits, values, _ = net(collate(res.samples), lengths=res.lengths)
    logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(res.actions)[:, None])[:, 0]
    rec_values = torch.tensor(res.returns) - torch.tensor(res.advantages)
    assert torch.allclose(logp, torch.tensor(res.logps), atol=tol), (logp - torch.tensor(res.logps)).abs().max()
    assert torch.allclose(values, rec_values, atol=tol), (values - rec_values).abs().max()


@pytest.mark.parametrize("memory", ["gru", "none"])
def test_server_rollouts_replay_exactly(tmp_path, memory):
    net, path = _ckpt(tmp_path, memory)
    _, pool_path = _ckpt(tmp_path, memory, seed=1)
    specs = [GameSpec(seed=s, seats=seats, match_game=1 + s % 2) for s, seats in enumerate(
        [(LEARNER, LEARNER), (LEARNER, pool_path), (pool_path, LEARNER), (LEARNER, BOT), (RANDOM, LEARNER), (LEARNER, LEARNER)] * 3
    )]  # fmt: skip
    engine = "native" if native_available() else "python"
    srv = InferenceServer(3, ServerConfig(device="cpu", max_wait_ms=2.0, threads=2))
    try:
        with srv.pool() as procs:
            results = procs.map(run_job, [Job(c, path, 1, max_turns=30, engine=engine, inference="server") for c in split_games(specs, 3)])
        stats = srv.stats()
    finally:
        srv.close()
    assert sum(len(r.games) for r in results) == len(specs)
    assert stats["rows"] > 0 and stats["batches"] < stats["requests"]  # requests from several workers were batched together
    for r in results:
        assert r.actions
        _replay_matches(net, r)


def test_local_rollouts_replay_exactly(tmp_path):
    net, path = _ckpt(tmp_path)
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER)) for s in range(4)]
    _replay_matches(net, run_job(Job(specs, path, 1, max_turns=30)))


def test_server_workers_do_not_import_torch(tmp_path):
    """Workers only play and featurize; the network lives in the server."""
    from mtg_ml.rl.inference import torch_loaded

    _, path = _ckpt(tmp_path)
    srv = InferenceServer(1, ServerConfig(device="cpu", threads=1))
    try:
        with srv.pool() as procs:
            procs.map(run_job, [Job([GameSpec(seed=1, seats=(LEARNER, LEARNER))], path, 1, max_turns=10, inference="server")])
            loaded = procs.apply(torch_loaded)
    finally:
        srv.close()
    assert not loaded
