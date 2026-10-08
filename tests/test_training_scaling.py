"""Attention serving, fixed policy residency and precision controls."""
from array import array

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.rl.features import OPTION_DIM, STATE_DIM  # noqa: E402
from mtg_ml.rl.inference import GREEDY_FLAG, InferenceClient, InferenceServer, ServerConfig, _Server  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate_packed  # noqa: E402
from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, Result, _batch, residency_waves, run_specs  # noqa: E402
from mtg_ml.rl.samples import PackedSamples  # noqa: E402


def _decision(entities=2):
    state = [1] + [t for k in range(entities) for t in (STATE_DIM, 2 + k)]
    opts = [7, OPTION_DIM] if entities else [7, 9]
    return tuple(array("i", x) for x in (state, [1, 1], opts, [10]))


def _save(tmp_path, k, attention=1):
    torch.manual_seed(k)
    net = PolicyNet(hidden=16, memory="none", trunk="entity", entity_attn=attention)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    path = str(tmp_path / f"p{k}.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return path, net


def test_mixed_attention_batch_keeps_fast_rows_stacked(tmp_path):
    paths, nets = zip(_save(tmp_path, 0), _save(tmp_path, 1, attention=2))
    srv = _Server(ServerConfig(device="cpu", threads=1, compile=False, graphs=False), 1)
    cl = InferenceClient(0, None, None, srv.names, srv.layout)
    try:
        keys = [(p, 0) for p in paths]
        cl.ids = {k: srv.register_key(k) for k in keys}
        decisions = [_decision(0), _decision(2), _decision(5)]
        rows = [(keys[k % 2], k, 1 | GREEDY_FLAG, *d) for k, d in enumerate(decisions)]
        # Client rows are grouped by policy; output order follows request order.
        rows.sort(key=lambda r: r[0])
        h = cl.submit(0, rows)
        srv.process(srv.pending())
        srv.drain()
        actions, logps, values = cl.collect(h)
        assert srv.stats["legacy"] == 1 and srv.stats["padded_rows"] > 0
        for row, a, lp, v in zip(rows, actions, logps, values):
            net = nets[keys.index(row[0])]
            with torch.no_grad():
                logits, val, _ = net(_batch(torch, [row[3:]])[0])
            assert a == logits.argmax(-1).item()
            assert abs(lp - torch.log_softmax(logits, -1)[0, a].item()) < 1e-5
            assert abs(v - val.item()) < 1e-5
    finally:
        cl.close()
        srv.close()


def test_resident_slots_are_stable_and_old_ids_invalidated(tmp_path):
    paths = [_save(tmp_path, k)[0] for k in range(4)]
    srv = _Server(ServerConfig(device="cpu", threads=1, resident_limit=2), 1)
    try:
        learner = (paths[0], 1)
        srv.select_policies([learner, (paths[1], 0)])
        stack, address = srv.stack, srv.stack.tables["option_emb.weight"].data_ptr()
        first_generation = int(srv.c[3])
        for p in paths[2:]:
            srv.select_policies([learner, (p, 0)])
            assert srv.stack is stack and stack.capacity == 2
            assert stack.tables["option_emb.weight"].data_ptr() == address
            assert len(srv.ids) == 2 and set(srv.ids) == {learner, (p, 0)}
        assert srv.c[3] > first_generation
        with pytest.raises(ValueError, match="not selected"):
            srv.register_key((paths[1], 0))
        with pytest.raises(ValueError, match="exceed"):
            srv.select_policies([learner, (paths[1], 0), (paths[2], 0)])
    finally:
        srv.close()


def test_residency_waves_preserve_every_sampled_game():
    specs = [GameSpec(k, (LEARNER, f"opp{k % 5}")) for k in range(17)]
    waves = list(residency_waves(specs, 3))
    assert len(waves) == 3
    assert sorted(s.seed for wave in waves for s in wave) == list(range(17))
    assert all(len({p for s in wave for p in s.seats}) <= 3 for wave in waves)
    with pytest.raises(ValueError, match="one game's"):
        list(residency_waves(specs, 1))


def test_packed_trajectory_reordering_preserves_tokens():
    ps = PackedSamples()
    for k in range(6):
        ps.append(_decision(k))
    got = ps.take_ranges([(3, 6), (0, 3)])
    assert list(got) == [ps[i] for i in (3, 4, 5, 0, 1, 2)]


def _training_data(net, lengths=(1, 7, 2)):
    data = Result(lengths=list(lengths))
    for k in range(sum(lengths)):
        data.samples.append(_decision(k % 4))
    with torch.no_grad():
        logits, values, _ = net(collate_packed(data.samples), lengths=list(lengths))
    data.actions = [k % 2 for k in range(sum(lengths))]
    data.logps = torch.log_softmax(logits, -1).gather(1, torch.tensor(data.actions)[:, None])[:, 0].tolist()
    data.advantages = [(k % 5 - 2) * 0.2 for k in range(sum(lengths))]
    data.returns = (values + torch.tensor(data.advantages)).tolist()
    data.kinds = [1] * sum(lengths)
    return data


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_selective_bf16_keeps_optimizer_and_probability_math_fp32(mode):
    from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update

    net = PolicyNet(hidden=16, trunk="entity", entity_attn=1)
    data = _training_data(net)
    cfg = PPOConfig(epochs=1, minibatch=4, precision="bf16", target_kl=None)
    opt = make_optimizer(net.parameters(), cfg.lr, "cpu")
    stats = ppo_update(net, opt, data, cfg, mode=mode)
    assert all(torch.isfinite(torch.tensor(stats[k])) for k in ("pg_loss", "v_loss", "entropy", "approx_kl"))
    assert all(p.dtype == torch.float32 for p in net.parameters())
    assert all(s["exp_avg"].dtype == torch.float32 for s in opt.state.values())


@pytest.mark.slow
@pytest.mark.parametrize("target_kl", [None, -1.0])
def test_distributed_unequal_and_empty_shards_match_single_gpu(target_kl):
    import copy

    from mtg_ml.rl.distributed import DistributedLearner
    from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update

    torch.manual_seed(7)
    reference = PolicyNet(hidden=8, trunk="entity", entity_attn=1)
    net = copy.deepcopy(reference)
    data = _training_data(reference)
    cfg = PPOConfig(epochs=2, minibatch=3, capture=0, target_kl=target_kl)
    opt0, opt1 = [make_optimizer(n.parameters(), cfg.lr, "cpu") for n in (reference, net)]
    gen = lambda: torch.Generator().manual_seed(9)  # noqa: E731
    expected = ppo_update(reference, opt0, data, cfg, gen=gen(), mode="eager")
    learner = DistributedLearner(net, opt1, cfg, ("cpu", "cpu", "cpu"))
    try:
        got = learner.update(net, opt1, data, cfg, cfg.lr, gen=gen())
    finally:
        learner.close()
    assert got["updates"] == expected["updates"] and got["early_stop"] == expected["early_stop"]
    for key in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "nt_frac", "nt_entropy"):
        assert got[key] == pytest.approx(expected[key], abs=1e-5), key
    for (name, p), (_, q) in zip(reference.named_parameters(), net.named_parameters()):
        assert torch.allclose(p, q, atol=2e-5), (name, (p - q).abs().max())
        if p in opt0.state:
            assert opt1.state[q]["step"] == opt0.state[p]["step"]
            assert torch.allclose(opt1.state[q]["exp_avg"], opt0.state[p]["exp_avg"], atol=1e-6)


@pytest.mark.slow
def test_bounded_rollouts_across_multiple_servers(tmp_path):
    paths = [_save(tmp_path, k)[0] for k in range(5)]
    specs = [GameSpec(k, (LEARNER, paths[1 + k % 4])) for k in range(8)]
    srv = InferenceServer(2, ServerConfig(device="cpu", devices=("cpu", "cpu"), resident_limit=2, threads=1))
    try:
        with srv.pool() as pool:
            res = run_specs(pool, specs, Job([], paths[0], 1, max_turns=5, inference="server"), 2)
        stats = srv.stats()
        assert len(res.games) == len(specs)
        assert res.trajectory_ids == sorted(res.trajectory_ids)
        assert all(s["policies"] <= 2 for s in stats["servers"])
        assert stats["legacy"] == 0
    finally:
        srv.close()
