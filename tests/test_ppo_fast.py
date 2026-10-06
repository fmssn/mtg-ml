"""The sync-free PPO update computes what the straightforward one did:
minibatches sliced out of an epoch (`model.structure` + `model.split`) equal
`collate_packed` of the same decisions, the network gives the same outputs
with the precomputed entity structure as with its mask-based paths, the GRU
sequence layouts equal packing with `pack_padded_sequence`, and one epoch of
`ppo_update` gives the statistics and weights of the pre-change
implementation, frozen below as the reference. Runs on CUDA too when present."""

import copy
import math
from dataclasses import fields, replace
from itertools import accumulate

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence  # noqa: E402

from mtg_ml.backend import native_available  # noqa: E402
from mtg_ml.rl.features import OPTION_DIM, STATE_DIM  # noqa: E402
from mtg_ml.rl.model import MEMORY_KINDS, TRUNKS, VALUE_NETS, Batch, PolicyNet, _Core, _sequence_layout, collate_packed, masked_entropy, packed_tensors, split, structure  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update, trajectory_minibatches  # noqa: E402
from mtg_ml.rl.rollout import LEARNER, RANDOM, GameSpec, Job, run_job  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SHARED = ("s_idx", "s_off", "e_idx", "e_off", "o_idx", "o_off", "o_row", "o_pos", "n_opts")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Learner decisions of a few short games, with entities and pointers
    (life shaping, so that drawn games still have advantages)."""
    torch.manual_seed(0)
    net = PolicyNet(hidden=16)
    path = str(tmp_path_factory.mktemp("ppo_fast") / "net.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    specs = [GameSpec(s, (LEARNER, LEARNER) if s % 2 else (RANDOM, LEARNER), match_game=1 + s % 2) for s in range(10)]
    res = run_job(Job(specs, path, 1, shaping=0.5, max_turns=20, engine="native" if native_available() else "python"))
    assert len(res.lengths) >= 14 and STATE_DIM in res.samples.s_idx and max(res.samples.o_idx) >= OPTION_DIM
    assert torch.tensor(res.advantages).std() > 0
    return res


# -- the pre-change implementation (reference) ----------------------------------


def _old_memory(self, x, hidden, lengths):
    if self.memory == "none":
        return self.mix(x), None
    seqs = list(torch.split(x, lengths))
    packed = pack_padded_sequence(pad_sequence(seqs, batch_first=True), torch.tensor(lengths), batch_first=True, enforce_sorted=False)
    out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
    return torch.cat([out[i, :n] for i, n in enumerate(lengths)]), None


def _old_minibatches(lengths, size, gen=None):
    starts, acc = [], 0
    for n in lengths:
        starts.append(acc)
        acc += n
    out, idx, lens = [], [], []
    for t in torch.randperm(len(lengths), generator=gen).tolist():
        idx.extend(range(starts[t], starts[t] + lengths[t]))
        lens.append(lengths[t])
        if len(idx) >= size:
            out.append((idx, lens))
            idx, lens = [], []
    if idx:
        out.append((idx, lens))
    return out


def _old_ppo_update(net, opt, data, cfg, gen=None):
    n = len(data.actions)
    actions = torch.tensor(data.actions, dtype=torch.long)
    old_logp = torch.tensor(data.logps, dtype=torch.float32)
    adv = torch.tensor(data.advantages, dtype=torch.float32)
    ret = torch.tensor(data.returns, dtype=torch.float32)
    var = ret.var().item()
    explained_var = float("nan") if n < 2 or var == 0 else 1 - adv.var().item() / var
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    net.train()
    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    steps, stop = 0, False
    for _ in range(cfg.epochs):
        epoch_kl = []
        for idx, lens in _old_minibatches(data.lengths, cfg.minibatch, gen):
            b = collate_packed(data.samples, idx)
            it = torch.tensor(idx)
            a, olp, ad, rt = (x[it] for x in (actions, old_logp, adv, ret))
            logits, values, _ = net(b, lengths=lens)
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a[:, None]).squeeze(1)
            ratio = (logp - olp).exp()
            pg_loss = -torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad).mean()
            v_loss = 0.5 * (values - rt).pow(2).mean()
            ent = masked_entropy(logits).mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - (logp - olp)).mean().item()
                epoch_kl.append(kl)
                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["approx_kl"] += kl
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            steps += 1
        if cfg.target_kl is not None and sum(epoch_kl) / len(epoch_kl) > cfg.target_kl:
            stop = True
            break
    out = {k: v / max(steps, 1) for k, v in stats.items()}
    out["updates"] = steps
    out["early_stop"] = stop
    out["explained_var"] = explained_var
    return out


# -- tests ------------------------------------------------------------------------


def test_trajectory_minibatches_keep_the_reference_order(data):
    for seed in range(3):
        order, chunks = trajectory_minibatches(data.lengths, 100, torch.Generator().manual_seed(seed))
        ref = _old_minibatches(data.lengths, 100, torch.Generator().manual_seed(seed))
        assert chunks == [lens for _, lens in ref]
        assert order.tolist() == [i for idx, _ in ref for i in idx]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_minibatches_equal_collate(data, device, seed):
    """Every minibatch cut out of the epoch equals the reference collate of
    its decisions, and its structure equals `structure` of that collate."""
    order, chunks = trajectory_minibatches(data.lengths, 150, torch.Generator().manual_seed(seed))
    bounds = list(accumulate((sum(c) for c in chunks), initial=0))
    pieces = split(structure(collate_packed(packed_tensors(data.samples, device), order.to(device))), bounds)
    assert len(pieces) == len(chunks) > 2
    for b, lo, hi in zip(pieces, bounds, bounds[1:]):
        ref = collate_packed(data.samples, order[lo:hi])
        direct = structure(ref)
        assert direct.p_opt.numel() and direct.e_row.numel()
        for f in fields(Batch):
            want = getattr(ref if f.name in SHARED else direct, f.name)
            assert torch.equal(getattr(b, f.name).cpu(), want), f.name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("memory", MEMORY_KINDS)
@pytest.mark.parametrize("value_net", VALUE_NETS)
@pytest.mark.parametrize("trunk", TRUNKS)
def test_precomputed_structure_gives_the_same_outputs(data, device, trunk, value_net, memory):
    """Sequence mode, with the structure (no masks) and without (mask-based
    paths): same logits, values and gradients."""
    lens = data.lengths[:4]
    b = collate_packed(data.samples, range(sum(lens))).to(device)
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, memory=memory, trunk=trunk, value_net=value_net, value_hidden=48).to(device)
    outs = []
    for batch in (b, structure(b)):
        net.zero_grad()
        logits, values, _ = net(batch, lengths=lens)
        finite = torch.isfinite(logits)
        (logits[finite].sum() + values.sum()).backward()
        outs.append((logits.detach(), values.detach(), [p.grad.clone() for p in net.parameters() if p.grad is not None]))
    (l0, v0, g0), (l1, v1, g1) = outs
    assert torch.equal(torch.isinf(l0), torch.isinf(l1))
    assert torch.allclose(l0[torch.isfinite(l0)], l1[torch.isfinite(l1)], atol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert len(g0) == len(g1) and all(torch.allclose(x, y, atol=1e-5) for x, y in zip(g0, g1))


def test_packed_layout_is_pack_padded_sequence():
    lengths = [5, 1, 9, 3, 9, 2, 1]
    rows = torch.arange(sum(lengths)).float()[:, None]
    ref = pack_padded_sequence(pad_sequence(list(torch.split(rows, lengths)), batch_first=True), torch.tensor(lengths), batch_first=True, enforce_sorted=False)
    idx, sizes = _sequence_layout(lengths, torch.device("cpu"), packed=True)
    assert torch.equal(sizes, ref.batch_sizes)
    for blk_idx, blk_ref in zip(idx.split(sizes.tolist()), ref.data[:, 0].long().split(sizes.tolist())):  # trajectories of equal length may swap
        assert sorted(blk_idx.tolist()) == sorted(blk_ref.tolist())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden", [32, 160])  # on CUDA: padded up to 128, packed above
def test_gru_sequence_mode_matches_pack_padded_sequence(device, hidden):
    lengths = [5, 1, 9, 3, 9, 2, 1]
    torch.manual_seed(0)
    core = _Core(hidden, "gru", "mlp", 64, 64).to(device)
    x = torch.randn(sum(lengths), 2 * hidden, device=device)
    w = torch.randn(sum(lengths), hidden, device=device)
    outs = []
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):  # TF32 alone would put them ~1e-4 apart
        for memory in (_Core._memory, _old_memory):
            xg = x.clone().requires_grad_()
            core.zero_grad()
            y, _ = memory(core, xg, None, lengths)
            (y * w).sum().backward()
            outs.append((y.detach(), xg.grad, [p.grad.clone() for p in core.gru.parameters()]))
    (y0, d0, g0), (y1, d1, g1) = outs  # cuDNN's padded and packed kernels sum in different orders: ~1e-5 apart
    assert torch.allclose(y0, y1, atol=5e-5) and torch.allclose(d0, d1, atol=5e-5)
    assert all(torch.allclose(a, b, rtol=1e-4, atol=5e-4) for a, b in zip(g0, g1))


@pytest.mark.parametrize(
    "trunk,value_net,memory,epochs,target_kl",
    [("entity", "shared", "gru", 1, None), ("mlp", "shared", "gru", 1, None), ("mlp", "separate", "gru", 1, None), ("entity", "separate", "none", 1, None), ("entity", "shared", "gru", 3, 0.0)],
)
def test_ppo_update_matches_the_reference(data, monkeypatch, trunk, value_net, memory, epochs, target_kl):
    """Same minibatch order (generator), same statistics and weights; with
    target_kl=0 both stop after the first epoch."""
    data = replace(data, logps=[lp + 0.3 * math.sin(i) for i, lp in enumerate(data.logps)])  # ratios away from 1: clipping and KL matter
    cfg = PPOConfig(epochs=epochs, minibatch=128, target_kl=target_kl)
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, memory=memory, trunk=trunk, value_net=value_net)
    ref_net = copy.deepcopy(net)
    stats = ppo_update(net, make_optimizer(net.parameters(), cfg.lr, "cpu"), data, cfg, gen=torch.Generator().manual_seed(3))
    monkeypatch.setattr(_Core, "_memory", _old_memory)
    ref = _old_ppo_update(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg.lr, eps=1e-5), data, cfg, gen=torch.Generator().manual_seed(3))
    assert stats["updates"] == ref["updates"] > 2 and stats["early_stop"] == ref["early_stop"] == (target_kl is not None)
    assert ref["clip_frac"] > 0.1 and ref["approx_kl"] > 1e-3
    for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "explained_var"):
        assert stats[k] == pytest.approx(ref[k], rel=1e-5, abs=1e-7), k
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(net.parameters(), ref_net.parameters()))


def test_make_optimizer_fuses_on_cuda_only():
    p = [nn.Parameter(torch.zeros(3))]
    assert not make_optimizer(p, 1e-3, "cpu").param_groups[0]["fused"]
    if torch.cuda.is_available():
        assert make_optimizer([nn.Parameter(torch.zeros(3, device="cuda"))], 1e-3, "cuda").param_groups[0]["fused"]
