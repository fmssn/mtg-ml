"""The belief head of feature set 8 (model/data side): evidence travels with
the recorded samples (packing, pickling, ranges, collate, padded pieces,
the inference server's requests), the head reads nothing but that
evidence, its targets stay aligned with their decisions through PPO's
minibatch reordering and padding, its loss trains only the belief branch,
PPO never reaches it, upgrading a set-7 checkpoint changes nothing, and
local, stacked and server inference agree."""

import math
import pickle
import random
from array import array

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.rl import ppo, stacked  # noqa: E402
from mtg_ml.rl.belief import MAX_BASIC, BeliefSpec  # noqa: E402
from mtg_ml.rl.features import OPTION_DIM, STATE_DIM  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate, collate_packed, load_partial, pad_split, padded_batch, split, step_batch, structure  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, belief_losses, make_optimizer, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import Result  # noqa: E402
from mtg_ml.rl.samples import PackedSamples  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SPEC = BeliefSpec.default()
V, A = len(SPEC.vocab), len(SPEC.archetypes)
NB = SPEC.basic.index(False)  # a nonbasic card
BASIC = SPEC.basic.index(True)  # a basic land


def _net(seed=0, features=8, proj=0.0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("hidden", 16)
    net = PolicyNet(features=features, belief=SPEC if features >= 8 else None, **kw)
    if proj and features >= 8:
        with torch.no_grad():
            net.belief_proj.weight.normal_(0, proj)
            net.belief_proj.bias.normal_(0, proj)
    return net


def _decision(rng, evidence=None, entities=False):
    """A synthetic flat sample (state, option lengths, option tokens, events[, evidence])."""
    state = [rng.randrange(1, STATE_DIM) for _ in range(rng.randint(2, 8))]
    n_ent = rng.randint(0, 3) if entities else 0
    for _ in range(n_ent):
        state += [STATE_DIM] + [rng.randrange(1, STATE_DIM) for _ in range(rng.randint(1, 4))]
    n_o = rng.randint(1, 4)
    o_len, o_flat = [], []
    for _ in range(n_o):
        toks = [rng.randrange(OPTION_DIM) for _ in range(rng.randint(1, 3))] + ([OPTION_DIM + rng.randrange(n_ent)] if n_ent and rng.random() < 0.5 else [])
        o_len.append(len(toks))
        o_flat += toks
    x = (array("i", state), array("i", o_len), array("i", o_flat), array("i", [rng.randrange(OPTION_DIM) for _ in range(rng.randint(0, 4))]))
    return x if evidence is None else x + (array("i", evidence),)


def _random_evidence(rng, n=None):
    out = []
    for _ in range(rng.randint(0, 6) if n is None else n):
        card = rng.randrange(V)
        out += [card, rng.randrange(3), rng.randint(1, MAX_BASIC if SPEC.basic[card] else 4)]
    return out


def _data(n_traj=14, seed=0, entities=False) -> Result:
    """Trajectories whose every decision's first evidence triple names the
    trajectory's targets: (archetype, slot 0, copies of card NB), then noise.
    Alignment checks read the targets back from the evidence."""
    rng = random.Random(seed)
    res = Result()
    targets = []
    for _ in range(n_traj):
        arch = rng.randrange(A)
        counts = array("i", [0]) * V
        counts[NB], counts[BASIC] = rng.randint(1, 4), rng.randint(0, 20)
        L = rng.randint(2, 9)
        for _ in range(L):
            x = _decision(rng, [arch, 0, counts[NB]] + _random_evidence(rng), entities)
            res.samples.append(x)
            n_o = len(x[1])
            res.actions.append(rng.randrange(n_o))
            res.logps.append(-math.log(n_o))
            res.advantages.append(rng.gauss(0, 1))
            res.returns.append(rng.gauss(0, 1))
            res.kinds.append(0)
        res.lengths.append(L)
        targets.append((arch, counts))
    res.belief_targets = targets
    return res


# -- packed samples ------------------------------------------------------------------


def test_packed_samples_carry_evidence():
    rng = random.Random(0)
    xs = [_decision(rng, _random_evidence(rng, k)) for k in (2, 0, 1)] + [_decision(rng)]  # a 4-tuple append: no evidence
    ps = PackedSamples()
    for x in xs:
        ps.append(x)
    want = [list(x[4]) if len(x) > 4 else [] for x in xs]
    assert [ps.evidence(i) for i in range(len(xs))] == want and ps.evidence(-1) == []
    assert ps[0][0] == list(xs[0][0]) and len(ps[0]) == 3  # list access unchanged
    back = pickle.loads(pickle.dumps(ps))
    assert [back.evidence(i) for i in range(4)] == want and list(back) == list(ps)
    part = ps.take_ranges([(2, 4), (0, 1)])
    assert [part.evidence(i) for i in range(3)] == [want[2], want[3], want[0]] and list(part) == [ps[2], ps[3], ps[0]]
    both = PackedSamples()
    both.extend(ps)
    both += part
    assert [both.evidence(i) for i in range(7)] == want + [want[2], want[3], want[0]]
    with pytest.raises(ValueError):
        PackedSamples().append(xs[0][:4] + (array("i", [1, 2]),))


def test_old_pickles_load_with_empty_evidence():
    rng = random.Random(1)
    ps = PackedSamples()
    for _ in range(3):
        ps.append(_decision(rng))
    old = PackedSamples.__new__(PackedSamples)
    old.__setstate__(ps.__getstate__()[:7])  # what a pickle from before evidence holds
    assert list(old) == list(ps) and [old.evidence(i) for i in range(3)] == [[]] * 3
    old.append(_decision(rng, [5, 1, 2]))
    assert old.evidence(3) == [5, 1, 2]


def test_collate_paths_agree_on_evidence():
    rng = random.Random(2)
    xs = [_decision(rng, _random_evidence(rng)) for _ in range(9)]
    ps = PackedSamples()
    for x in xs:
        ps.append(x)
    ref = collate([(list(x[0]), [list(x[2][sum(x[1][:k]) : sum(x[1][: k + 1])]) for k in range(len(x[1]))], list(x[3]), list(x[4])) for x in xs])
    for b in (collate(ps), step_batch(xs)[0], collate_packed(ps, list(range(9)), evidence=True)):
        for f in ("v_card", "v_slot", "v_cnt", "v_off"):
            assert torch.equal(getattr(b, f).long(), getattr(ref, f)), f
    sub = collate_packed(ps, [4, 1], evidence=True)
    assert sub.v_card.tolist() == [xs[4][4][k] for k in range(0, len(xs[4][4]), 3)] + [xs[1][4][k] for k in range(0, len(xs[1][4]), 3)]
    assert collate_packed(ps).v_off is None  # networks without a belief head never see it


# -- the head and its information boundary ---------------------------------------------


def test_feature_set_8_requires_a_belief_head():
    with pytest.raises(ValueError):
        PolicyNet(hidden=16, features=8)
    with pytest.raises(ValueError):
        PolicyNet(hidden=16, features=7, belief=SPEC)
    net = _net()
    assert net.belief == SPEC and net.config["belief"] == SPEC.to_config() and net.config["belief_coef"] == {"archetype": 0.1, "counts": 0.1}
    assert PolicyNet(**net.config).config == net.config


@pytest.mark.parametrize("trunk", ["mlp", "entity"])
def test_identical_evidence_gives_identical_beliefs(trunk):
    """Whatever else differs (state, events, options, hidden state: the
    unseen cards, the archetype the simulator dealt, sideboard choices all
    reach the network only through those), the same evidence gives the same
    belief predictions; different evidence gives different ones."""
    net = _net(trunk=trunk, proj=0.1).eval()
    rng = random.Random(3)
    ev = [_random_evidence(rng, 4) for _ in range(5)]
    a = [_decision(rng, e, entities=True) for e in ev]
    b = [_decision(rng, e, entities=True) for e in ev]
    with torch.no_grad():
        oa = net(step_batch(a)[0], torch.randn(5, net.state_size), aux=True)[3]
        ob = net(step_batch(b)[0], None, aux=True)[3]
        oc = net(step_batch([x[:4] + (array("i", _random_evidence(rng, 4)),) for x in a])[0], aux=True)[3]
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)
    assert not torch.equal(oa.arch, oc.arch)
    assert oa.arch.shape == (5, A) and oa.nonbasic.shape[-1] == 5 and oa.basic.shape[-1] == MAX_BASIC + 1


def test_belief_reads_only_evidence_fields():
    """The encoder's input is the evidence alone: a batch stripped of every
    other field still gives the same predictions."""
    net = _net()
    rng = random.Random(4)
    b = step_batch([_decision(rng, _random_evidence(rng)) for _ in range(6)])[0]
    only = type(b)(*([None] * 9), v_card=b.v_card, v_slot=b.v_slot, v_cnt=b.v_cnt, v_off=b.v_off)
    with torch.no_grad():
        full = net(b, aux=True)[3]
        alone = net.belief_net(only, 6)
    assert all(torch.equal(x, y) for x, y in zip(full, alone))


def test_gradients_stay_on_their_side():
    """PPO-side losses (logits, values) give no gradient to the belief
    branch (its predictions enter the policy detached); the belief loss
    gives gradients to the belief branch only."""
    net = _net(proj=0.2)
    data = _data(4)
    b = collate(data.samples)
    logits, values, _, bel = net(b, lengths=data.lengths, aux=True)
    (torch.log_softmax(logits, -1).nan_to_num(0, 0, 0).sum() + values.sum()).backward()
    for name, p in net.named_parameters():
        if name.startswith("belief_net."):
            assert p.grad is None or not p.grad.any(), name
    assert net.belief_proj.weight.grad.abs().sum() > 0  # the projection itself is policy
    net.zero_grad(set_to_none=True)
    bel = net(b, lengths=data.lengths, aux=True)[3]
    tid = torch.repeat_interleave(torch.arange(4), torch.tensor(data.lengths))
    arch = torch.tensor([t[0] for t in data.belief_targets])[tid]
    counts = torch.tensor([list(t[1]) for t in data.belief_targets])[tid]
    al, cl, _ = belief_losses(net, bel, arch, counts, lambda x: x.mean())
    (al + cl).backward()
    got = {name for name, p in net.named_parameters() if p.grad is not None and p.grad.any()}
    assert got and all(n.startswith("belief_net.") for n in got), got


# -- checkpoints ----------------------------------------------------------------------


@pytest.mark.parametrize("trunk,value_net", [("mlp", "shared"), ("entity", "separate")])
def test_zero_projection_upgrade_reproduces_the_set7_parent(trunk, value_net, tmp_path):
    from mtg_ml.rl.rollout import load_net

    parent = _net(features=7, trunk=trunk, value_net=value_net, value_hidden=24).eval()
    child = _net(seed=5, trunk=trunk, value_net=value_net, value_hidden=24).eval()
    new = load_partial(child, parent.state_dict())
    assert new and all(k.startswith(("belief_net.", "belief_proj.")) for k in new)
    data = _data(5, entities=trunk == "entity")
    b = collate(data.samples)
    with torch.no_grad():
        want = parent(b, lengths=data.lengths)
        got = child(b, lengths=data.lengths)
        decs = [_tuple(data.samples, i) for i in range(5)]
        h = torch.randn(5, parent.state_size)
        step = (parent(step_batch(decs)[0], h), child(step_batch(decs)[0], h))
    assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])
    for x, y in zip(*step):
        assert torch.equal(x, y)
    # a saved belief checkpoint loads (meta device + assigned state) and computes the same
    path = str(tmp_path / "b.pt")
    torch.save({"config": child.config, "model": child.state_dict()}, path)
    with torch.no_grad():
        again = load_net(path)(b, lengths=data.lengths, aux=True)
        mine = child(b, lengths=data.lengths, aux=True)
    assert torch.equal(again[0], mine[0]) and all(torch.equal(x, y) for x, y in zip(again[3], mine[3]))


def _tuple(ps, i):
    """Sample i of a PackedSamples as a flat 5-tuple."""
    state, opts, events = ps[i]
    return (array("i", state), array("i", [len(o) for o in opts]), array("i", [t for o in opts for t in o]), array("i", events), array("i", ps.evidence(i)))


# -- PPO: target alignment, padding, learning --------------------------------------------


def _spy(monkeypatch, seen):
    """Wrap ppo._losses: check that every real row's targets are the ones its
    evidence names (first triple: archetype, slot 0, copies of NB)."""
    real = ppo._losses

    def check(net, cfg, b, lengths, width, a, olp, ad, rt, w=None, n=None, kind=None, belief=None):
        traj, arch_t, counts_t = belief
        rows = int(n.item()) if n is not None else traj.shape[0]
        first = b.v_off[:rows].long()
        assert torch.equal(b.v_card[first].long().cpu(), arch_t[traj[:rows]].cpu())
        assert torch.equal(b.v_cnt[first].long().cpu(), counts_t[traj[:rows], NB].cpu())
        assert (b.v_slot[first] == 0).all()
        seen.append(rows)
        return real(net, cfg, b, lengths, width, a, olp, ad, rt, w, n, kind, belief)

    monkeypatch.setattr(ppo, "_losses", check)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["eager", "padded"])
@pytest.mark.parametrize("trunk", ["mlp", "entity"])
def test_targets_stay_aligned_through_minibatches(monkeypatch, device, mode, trunk):
    data = _data(30, seed=6, entities=trunk == "entity")
    net = _net(trunk=trunk).to(device)
    opt = make_optimizer(net.parameters(), 1e-3, device)
    seen = []
    _spy(monkeypatch, seen)
    out = ppo_update(net, opt, data, PPOConfig(epochs=2, minibatch=40, target_kl=None), device=device, gen=torch.Generator().manual_seed(0), mode=mode)
    assert sum(seen) == 2 * len(data.actions) and len(seen) > 4
    assert {"belief/arch_loss", "belief/count_loss", "belief/arch_acc", "belief/count_acc", "belief/count_mae"} <= set(out)
    assert out["belief/arch_loss"] > 0 and out["belief/count_loss"] > 0


@pytest.mark.parametrize("trunk", ["mlp", "entity"])
def test_padded_update_matches_the_eager_one(trunk):
    """Same minibatches eagerly and padded to static shapes: the same belief
    statistics and the same weights (padding adds nothing)."""
    data = _data(20, seed=7, entities=trunk == "entity")
    outs, nets = [], []
    for mode in ("eager", "padded"):
        net = _net(trunk=trunk, proj=0.05)
        opt = make_optimizer(net.parameters(), 1e-3, "cpu")
        outs.append(ppo_update(net, opt, data, PPOConfig(epochs=2, minibatch=30, target_kl=None), gen=torch.Generator().manual_seed(1), mode=mode))
        nets.append(net)
    for k in ppo.BELIEF_STATS:
        assert outs[0][f"belief/{k}"] == pytest.approx(outs[1][f"belief/{k}"], rel=1e-4, abs=1e-6), k
    for (name, a), b in zip(nets[0].named_parameters(), nets[1].parameters()):
        assert torch.allclose(a, b, atol=1e-5), name


def test_padded_pieces_hold_the_evidence():
    data = _data(12, seed=8, entities=True)
    order = torch.randperm(len(data.actions), generator=torch.Generator().manual_seed(0))
    big = structure(collate_packed(data.samples, order, evidence=True))
    bounds = [0, 17, 40, len(data.actions)]
    fields, sizes, counts = pad_split(big, bounds)
    net = _net(trunk="entity")
    for k, piece in enumerate(split(big, bounds)):
        n, t = counts["row"][k], counts["v"][k]
        for name in ("v_card", "v_slot", "v_cnt"):
            assert torch.equal(fields[name][k][:t], getattr(piece, name).long())
        off = fields["v_off"][k]
        assert torch.equal(off[:n], piece.v_off.long()) and (off[n:] >= t).all() and off[-1] <= sizes["v"]
        with torch.no_grad():
            want = net.belief_net(piece, n)
            got = net.belief_net(padded_batch({f: x[k] for f, x in fields.items()}), sizes["row"])
        assert all(torch.allclose(x[:n], y, atol=1e-6) for x, y in zip(got, want) if y is not None)


def test_ppo_refuses_a_belief_network_without_targets():
    data = _data(3)
    data.belief_targets = []
    net = _net()
    with pytest.raises(ValueError, match="belief target"):
        ppo_update(net, make_optimizer(net.parameters(), 1e-3, "cpu"), data, PPOConfig(epochs=1))


def test_belief_head_learns_distinguishing_evidence():
    """A small fixture: the archetype shows in which card was seen; the
    belief loss falls and the predictions follow the evidence."""
    torch.manual_seed(0)
    net = _net(hidden=32)
    rng = random.Random(9)
    xs, arch, counts = [], [], []
    for _ in range(96):
        a = rng.randrange(A)
        c = [0] * V
        c[a + 10] = rng.randint(1, 4)
        xs.append(_decision(rng, [a + 10, rng.randrange(3), rng.randint(1, c[a + 10])]))
        arch.append(a)
        counts.append(c)
    b = step_batch(xs)[0]
    arch, counts = torch.tensor(arch), torch.tensor(counts)
    opt = torch.optim.Adam(net.belief_net.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        al, cl, _ = belief_losses(net, net.belief_net(b, len(xs)), arch, counts, lambda x: x.mean())
        opt.zero_grad()
        (al + cl).backward()
        opt.step()
        losses.append(float((al + cl).detach()))
    assert losses[-1] < 0.5 * losses[0]
    with torch.no_grad():
        acc = (net.belief_net(b, len(xs)).arch.argmax(-1) == arch).float().mean()
    assert acc > 0.9


# -- inference: local, stacked, server -----------------------------------------------------


def _belief_nets(trunk, memory, k=3):
    nets = []
    for s in range(k):
        net = _net(seed=s, hidden=32, trunk=trunk, memory=memory, proj=0.1).eval()
        with torch.no_grad():
            for p in net.parameters():
                p += 0.02 * torch.randn_like(p)
        nets.append(net)
    return nets


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trunk,memory", [("entity", "gru"), ("mlp", "none")])
def test_stacked_forward_matches_local_with_beliefs(device, trunk, memory):
    rng = random.Random(10)
    decs = [_decision(rng, _random_evidence(rng), entities=trunk == "entity") for _ in range(40)]
    nets = _belief_nets(trunk, memory)
    slots = [2, 0, 1]
    stack = stacked.PolicyStack(nets[0].config, 3, device)
    for s, n in zip(slots, nets):
        stack.load(s, n)
    pols = sorted(rng.choices(slots, k=len(decs)))
    fresh = [i % 3 == 0 for i in range(len(decs))]
    table = torch.randn(len(decs) + 1, nets[0].state_size)
    x = stacked.step_input(decs, pols, list(range(len(decs))), fresh, device=device, pad=2)
    with torch.no_grad():
        act, logp, val, hn, bel = stacked.step_forward(stack, x, table.clone().to(device) if memory == "gru" else None, write_hidden=False, belief=True)
        for s, net in zip(slots, nets):
            rows = [i for i, p in enumerate(pols) if p == s]
            batch, width = step_batch([decs[i] for i in rows])
            h = torch.where(torch.tensor([fresh[i] for i in rows])[:, None], 0.0, table[rows]) if memory == "gru" else None
            logits, v, hn_ref, ref = net(batch, h, max_options=width, aux=True)
            idx = torch.tensor(rows)
            assert torch.allclose(torch.log_softmax(logits, -1).gather(1, act[idx].cpu()[:, None])[:, 0], logp[idx].cpu(), atol=1e-5)
            assert torch.allclose(v, val[idx].cpu(), atol=1e-5)
            for mine, want in zip(bel, ref):
                assert torch.allclose(mine[idx].cpu(), want, atol=1e-5)
            if hn_ref is not None:
                assert torch.allclose(hn[idx].cpu(), hn_ref, atol=1e-5)


@pytest.mark.parametrize("trunk", ["entity", "transformer"])
def test_server_requests_carry_evidence(tmp_path, trunk):
    """In-process server (no worker processes): rows with evidence, two
    belief policies, greedy; the replies equal local forwards. The entity
    trunk runs stacked, the transformer per policy (`_legacy`)."""
    from mtg_ml.rl.inference import GREEDY_FLAG, InferenceClient, ServerConfig, _Server

    nets = _belief_nets(trunk, "gru", 2)
    keys = []
    for k, net in enumerate(nets):
        path = str(tmp_path / f"b{k}.pt")
        torch.save({"config": net.config, "model": net.state_dict()}, path)
        keys.append((path, 0))
    srv = _Server(ServerConfig(device="cpu", threads=1, compile=False, graphs=False), 1)
    try:
        ids = {key: srv.register_key(key) for key in keys}
        cl = InferenceClient(0, None, None, srv.names, srv.layout)
        cl.ids = dict(ids)
        rng = random.Random(11)
        decs = [_decision(rng, _random_evidence(rng), entities=trunk == "entity") for _ in range(24)]
        pol = sorted(rng.randrange(2) for _ in decs)
        rows = [(keys[p], i, 1 | GREEDY_FLAG, *d) for i, (p, d) in enumerate(zip(pol, decs))]
        h = cl.submit(0, rows)
        srv.process(srv.pending())
        srv.drain()
        acts, logps, values = cl.collect(h)
        assert (srv.stats["legacy"] > 0) == (trunk == "transformer")
        with torch.no_grad():
            for p, net in enumerate(nets):
                idx = [i for i in range(len(decs)) if pol[i] == p]
                logits, v, _ = net(step_batch([decs[i] for i in idx])[0], None)
                assert [acts[i] for i in idx] == logits.argmax(-1).tolist()
                assert torch.allclose(v, torch.tensor([values[i] for i in idx]), atol=1e-5)
                # and the evidence mattered: without it the policy differs
                bare, _, _ = net(step_batch([decs[i][:4] for i in idx])[0], None)
                assert not torch.allclose(bare, logits)
        cl.close()
    finally:
        srv.close()
