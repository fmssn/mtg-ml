"""Entity self-attention (`PolicyNet(entity_attn=N)`, `--entity-attn N`).

entity_attn=0 is the model as it was (config, weights, outputs). New
attention blocks start as the identity, so a checkpoint without them loads
into a model with them (`load_partial`, `--init`) and gives the same
outputs. With non-zero attention weights, the mask-based, precomputed and
padded (CUDA graph) paths agree, samples never attend to each other, and a
checkpoint round trip, the trainer's --init and the inference server (on its
eager per-policy path) all keep the layers."""

import copy
import math
from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.backend import native_available  # noqa: E402
from mtg_ml.rl.features import OPTION_DIM, STATE_DIM  # noqa: E402
from mtg_ml.rl.model import VALUE_NETS, PolicyNet, collate, collate_packed, load_partial, packed_tensors, pad_fits, pad_split, padded_batch, split, structure  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update, trajectory_minibatches  # noqa: E402
from mtg_ml.rl.rollout import LEARNER, RANDOM, GameSpec, Job, load_net, run_job  # noqa: E402
from mtg_ml.rl.stacked import PolicyStack  # noqa: E402
from mtg_ml.rl.train import Trainer, parse_args  # noqa: E402

ENGINE = "native" if native_available() else "python"
ATTN = ".state.attn."


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Learner decisions of a few short games, with entities and pointers."""
    torch.manual_seed(0)
    net = PolicyNet(hidden=16)
    path = str(tmp_path_factory.mktemp("entity_attn") / "net.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    specs = [GameSpec(s, (LEARNER, LEARNER) if s % 2 else (RANDOM, LEARNER), match_game=1 + s % 2) for s in range(10)]
    res = run_job(Job(specs, path, 1, shaping=0.5, max_turns=20, engine=ENGINE))
    assert len(res.lengths) >= 14 and STATE_DIM in res.samples.s_idx and max(res.samples.o_idx) >= OPTION_DIM
    return res


def _net(seed=0, **kw) -> PolicyNet:
    """An entity-trunk net with every weight non-zero (value heads, attention
    output projections), so that each layer matters."""
    torch.manual_seed(seed)
    net = PolicyNet(hidden=32, trunk="entity", value_hidden=48, **kw)
    with torch.no_grad():
        for p in net.parameters():
            p += 0.05 * torch.randn_like(p)
    return net


def _rows(data, n=4):
    lens = data.lengths[:n]
    return collate_packed(data.samples, range(sum(lens))), lens


def _finite(logits):
    return logits[torch.isfinite(logits)]


def test_default_model_is_unchanged():
    """entity_attn=0: same config (no new key), same weights; entity_attn=N
    only adds attention weights, and needs the entity trunk."""
    for vn in VALUE_NETS:
        torch.manual_seed(0)
        old = PolicyNet(hidden=32, trunk="entity", value_net=vn)
        torch.manual_seed(0)
        zero = PolicyNet(hidden=32, trunk="entity", value_net=vn, entity_attn=0)
        assert "entity_attn" not in zero.config and zero.config == old.config
        assert list(old.state_dict()) == list(zero.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(old.state_dict().values(), zero.state_dict().values()))
        attn = PolicyNet(hidden=32, trunk="entity", value_net=vn, entity_attn=2)
        assert attn.config["entity_attn"] == 2
        new = set(attn.state_dict()) - set(old.state_dict())
        assert set(old.state_dict()) <= set(attn.state_dict()) and new and all(ATTN in k for k in new)
        assert any(k.startswith("value_core.") for k in new) == (vn == "separate")
    with pytest.raises(ValueError):
        PolicyNet(hidden=32, trunk="mlp", entity_attn=1)


@pytest.mark.parametrize("value_net", VALUE_NETS)
@pytest.mark.parametrize("memory", ["gru", "none"])
def test_checkpoint_without_attention_loads_as_identity(data, value_net, memory):
    """A checkpoint without attention in a model with 2 layers: only the
    attention weights are new, and logits, values and hidden states are the
    checkpoint model's, in step mode (mask-based paths) and in sequence mode
    (precomputed structure)."""
    old = _net(memory=memory, value_net=value_net)
    torch.manual_seed(5)
    new = PolicyNet(hidden=32, trunk="entity", value_hidden=48, memory=memory, value_net=value_net, entity_attn=2)
    added = load_partial(new, old.state_dict())
    assert set(added) == set(new.state_dict()) - set(old.state_dict()) and added
    b, lens = _rows(data)
    hidden = torch.randn(b.n_opts.shape[0], old.state_size)
    with torch.no_grad():
        for batch, kw in ((b, {"hidden": hidden}), (structure(b), {"lengths": lens})):
            (l0, v0, h0), (l1, v1, h1) = old(batch, **kw), new(batch, **kw)
            assert torch.equal(torch.isinf(l0), torch.isinf(l1))
            assert torch.allclose(_finite(l0), _finite(l1), atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)
            assert (h0 is None) == (h1 is None) and (h0 is None or torch.allclose(h0, h1, atol=1e-5))
    with pytest.raises(ValueError):  # a checkpoint lacking anything else does not fit
        load_partial(copy.deepcopy(new), {k: v for k, v in old.state_dict().items() if "scorer" not in k})


@pytest.mark.parametrize("value_net", VALUE_NETS)
def test_attention_paths_agree(data, value_net):
    """With attention active: the mask-based path, the precomputed structure
    and a padded piece (padded rows without entities, padded entities at
    position -1, the static width of the padded shape) give the same logits,
    values and gradients."""
    net = _net(memory="none", value_net=value_net, entity_attn=2)
    b, lens = _rows(data, 6)
    outs = []
    for batch in (b, structure(b)):
        net.zero_grad()
        logits, values, _ = net(batch, lengths=lens)
        (_finite(logits).sum() + values.sum()).backward()
        outs.append((logits.detach(), values.detach(), [p.grad.clone() for p in net.parameters()]))
    (l0, v0, g0), (l1, v1, g1) = outs
    assert torch.allclose(_finite(l0), _finite(l1), atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(g0, g1))
    attn_grads = [p.grad for n, p in net.named_parameters() if ATTN in n]
    assert attn_grads and all(torch.isfinite(g).all() for g in attn_grads) and any(g.abs().sum() > 0 for g in attn_grads)

    # padded pieces: memory "none", so no GRU layout is needed
    big = structure(collate_packed(data.samples))
    bounds = list(accumulate(data.lengths[:3], initial=0)) + [len(data.actions)]
    fields, sizes, counts = pad_split(big, bounds, ent_width=True)
    assert pad_fits(sizes, counts) and sizes["entw"] >= max(counts["entw"])
    for k, piece in enumerate(split(big, bounds)):
        f = {name: t[k] for name, t in fields.items()}
        R = counts["row"][k]
        assert (f["e_pos"][counts["ent"][k] :] == -1).all() and (f["e_pos"][: counts["ent"][k]] >= 0).all()
        with torch.no_grad():
            want_l, want_v, _ = net(piece, lengths=[R])
            got_l, got_v, _ = net(padded_batch(f, sizes["entw"]), lengths=[sizes["row"]], max_options=want_l.shape[1])
        assert torch.isfinite(got_v).all() and torch.allclose(got_v[:R], want_v, atol=1e-5)
        assert torch.equal(torch.isinf(got_l[:R]), torch.isinf(want_l)) and torch.allclose(_finite(got_l[:R]), _finite(want_l), atol=1e-5)


def test_samples_do_not_attend_to_each_other(data):
    """A decision's outputs do not depend on the other decisions of the batch."""
    net = _net(memory="none", entity_attn=1).eval()
    b, lens = _rows(data, 6)
    n = b.n_opts.shape[0]
    with torch.no_grad():
        full, fv, _ = net(b)
        for lo, hi in ((0, 1), (3, 9), (n - 5, n)):
            part, pv, _ = net(collate_packed(data.samples, range(lo, hi)))
            w = part.shape[1]
            assert torch.allclose(_finite(part), _finite(full[lo:hi, :w]), atol=1e-5) and torch.allclose(pv, fv[lo:hi], atol=1e-5)


@pytest.mark.parametrize("value_net,memory,epochs", [("shared", "gru", 2), ("separate", "none", 1)])
def test_padded_update_matches_the_eager_one(data, value_net, memory, epochs):
    """The padded path (what the CUDA graphs replay), run eagerly on the
    CPU, with attention: same statistics and weights as the unpadded one."""
    data = replace(data, logps=[lp + 0.3 * math.sin(i) for i, lp in enumerate(data.logps)])
    cfg = PPOConfig(epochs=epochs, minibatch=128, target_kl=None)
    nets = [_net(memory=memory, value_net=value_net, entity_attn=1)]
    nets.append(copy.deepcopy(nets[0]))
    stats = [ppo_update(n, make_optimizer(n.parameters(), cfg.lr, "cpu"), data, cfg, gen=torch.Generator().manual_seed(3), mode=m) for n, m in zip(nets, ("padded", "eager"))]
    assert stats[0]["updates"] == stats[1]["updates"] > 2
    for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac"):
        assert stats[0][k] == pytest.approx(stats[1][k], rel=1e-5, abs=1e-7), k
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(*(n.parameters() for n in nets)))
    before = _net(memory=memory, value_net=value_net, entity_attn=1)
    assert any(not torch.equal(p, q) for (n, p), q in zip(nets[0].named_parameters(), before.parameters()) if ATTN in n)  # attention trains


def test_split_pieces_keep_an_entity_width_bound(data):
    order, chunks = trajectory_minibatches(data.lengths, 150, torch.Generator().manual_seed(0))
    bounds = list(accumulate((sum(c) for c in chunks), initial=0))
    big = structure(collate_packed(packed_tensors(data.samples), order))
    for b, lo, hi in zip(split(big, bounds), bounds, bounds[1:]):
        direct = structure(collate_packed(data.samples, order[lo:hi]))
        assert b.ent_width == big.ent_width >= direct.ent_width > 0


def test_checkpoint_round_trip(tmp_path, data):
    """The config records entity_attn; `load_net` (workers, server,
    evaluation) rebuilds the layers and gives the same outputs."""
    net = _net(entity_attn=2).eval()
    path = str(tmp_path / "attn.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    loaded = load_net(path)
    assert loaded.config == net.config and loaded.config["entity_attn"] == 2
    b, _ = _rows(data)
    with torch.no_grad():
        (l0, v0, h0), (l1, v1, h1) = net(b), loaded(b)
    assert torch.equal(l0, l1) and torch.equal(v0, v1) and torch.equal(h0, h1)
    assert not PolicyStack.supports(net.config) and PolicyStack.supports(_net().config)


def test_trainer_init_from_a_checkpoint_without_attention(tmp_path, data):
    old = _net()
    path = str(tmp_path / "latest.pt")
    torch.save({"config": old.config, "model": old.state_dict(), "iteration": 7}, path)
    torch.manual_seed(1)
    new = PolicyNet(hidden=32, trunk="entity", value_hidden=48, entity_attn=1)
    Trainer._init_from(SimpleNamespace(cfg=SimpleNamespace(device="cpu"), net=new), path)
    b, _ = _rows(data)
    with torch.no_grad():
        (l0, v0, _), (l1, v1, _) = old(b), new(b)
    assert torch.allclose(_finite(l0), _finite(l1), atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)
    attn_path = str(tmp_path / "attn.pt")
    torch.save({"config": new.config, "model": new.state_dict()}, attn_path)
    for bad, src in ((PolicyNet(hidden=64, trunk="entity", entity_attn=1), path), (PolicyNet(hidden=32, trunk="entity", value_hidden=48), attn_path)):  # another width; fewer layers than the checkpoint
        with pytest.raises(ValueError):
            Trainer._init_from(SimpleNamespace(cfg=SimpleNamespace(device="cpu"), net=bad), src)


def test_flags():
    cfg = parse_args(["--trunk", "entity", "--entity-attn", "2", "--init", "runs/x/latest.pt", "--hidden", "192"])
    assert (cfg.entity_attn, cfg.init, cfg.hidden) == (2, "runs/x/latest.pt", 192)
    assert parse_args([]).entity_attn == 0 and parse_args([]).init == ""


@pytest.mark.slow
def test_server_serves_attention_policies(tmp_path):
    """The inference server runs attention policies on its eager per-policy
    path: rollouts replay exactly."""
    from mtg_ml.rl.inference import InferenceServer, ServerConfig
    from mtg_ml.rl.rollout import split_games

    net = _net(entity_attn=1).eval()
    path = str(tmp_path / "attn.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER) if s % 2 else (RANDOM, LEARNER)) for s in range(6)]
    srv = InferenceServer(2, ServerConfig(device="cpu", max_wait_ms=2.0, min_rows=16, threads=2))
    try:
        with srv.pool() as procs:
            results = procs.map(run_job, [Job(c, path, 1, max_turns=20, engine=ENGINE, inference="server") for c in split_games(specs, 2)])
        stats = srv.stats()
    finally:
        srv.close()
    assert stats["legacy"] > 0
    for r in results:
        with torch.no_grad():
            logits, values, _ = net(collate(r.samples), lengths=r.lengths)
        logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(r.actions)[:, None])[:, 0]
        assert torch.allclose(logp, torch.tensor(r.logps), atol=1e-4)
        assert torch.allclose(values, torch.tensor(r.returns) - torch.tensor(r.advantages), atol=1e-4)
