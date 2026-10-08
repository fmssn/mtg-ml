"""Central inference server: rollouts through the server record exactly what
the network computes when it replays the recorded trajectories from scratch
(sequence mode, as in the PPO update). That checks the shared-memory
transport, the server-side hidden state table (slots, fresh flags), the
cross-worker batching, the stacked multi-policy forward and the scatter of
results back to the right decision. The stacked step forward is also
checked directly against per-policy `PolicyNet` forwards, and its entity
structure against `model.structure`."""

import random
from array import array

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.backend import game_class, native_available  # noqa: E402
from mtg_ml.match import match_decks  # noqa: E402
from mtg_ml.rl import stacked  # noqa: E402
from mtg_ml.rl.features import OPTION_DIM, STATE_DIM, encode_event_hashes, featurize_flat  # noqa: E402
from mtg_ml.rl.inference import InferenceServer, ServerConfig  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate, structure  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, RANDOM, GameSpec, Job, _batch, run_job, split_games  # noqa: E402

ENGINE = "native" if native_available() else "python"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _ckpt(tmp_path, memory="gru", seed=0, trunk="mlp", value_net="shared"):
    torch.manual_seed(seed)
    net = PolicyNet(hidden=32, memory=memory, trunk=trunk, value_net=value_net, value_hidden=48)
    path = str(tmp_path / f"net_{memory}_{trunk}_{value_net}_{seed}.pt")
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


VARIANTS = [("gru", "mlp", "shared"), ("none", "mlp", "shared"), ("gru", "mlp", "separate"), ("gru", "transformer", "separate"), ("none", "transformer", "shared"), ("gru", "entity", "shared"), ("none", "entity", "separate")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("memory,trunk,value_net", VARIANTS)
def test_server_rollouts_replay_exactly(tmp_path, memory, trunk, value_net, device):
    net, path = _ckpt(tmp_path, memory, trunk=trunk, value_net=value_net)
    _, pool_path = _ckpt(tmp_path, memory, seed=1, trunk=trunk, value_net=value_net)
    specs = [GameSpec(seed=s, seats=seats, match_game=1 + s % 2) for s, seats in enumerate(
        [(LEARNER, LEARNER), (LEARNER, pool_path), (pool_path, LEARNER), (LEARNER, BOT), (RANDOM, LEARNER), (LEARNER, LEARNER)] * 3
    )]  # fmt: skip
    srv = InferenceServer(3, ServerConfig(device=device, max_wait_ms=2.0, min_rows=64, threads=2))
    try:
        with srv.pool() as procs:
            results = procs.map(run_job, [Job(c, path, 1, max_turns=30, engine=ENGINE, inference="server") for c in split_games(specs, 3)])
        stats = srv.stats()
    finally:
        srv.close()
    assert sum(len(r.games) for r in results) == len(specs)
    assert stats["rows"] > 0 and stats["batches"] < stats["requests"]  # requests from several workers were batched together
    assert (stats["legacy"] > 0) == (trunk == "transformer")  # mlp and entity run stacked
    for r in results:
        assert r.actions
        _replay_matches(net, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("memory,trunk,value_net", [("gru", "entity", "shared"), ("gru", "mlp", "separate")])
@pytest.mark.parametrize("compile_", [False, True])
def test_cuda_server_variants_replay_exactly(tmp_path, memory, trunk, value_net, compile_):
    """The CUDA paths: graphs, with and without torch.compile, and the eager forward."""
    net, path = _ckpt(tmp_path, memory, trunk=trunk, value_net=value_net)
    _, pool_path = _ckpt(tmp_path, memory, seed=1, trunk=trunk, value_net=value_net)
    specs = [GameSpec(seed=s, seats=seats) for s, seats in enumerate([(LEARNER, LEARNER), (LEARNER, pool_path), (pool_path, LEARNER)] * 4)]
    for graphs in (True, False) if not compile_ else (True,):
        srv = InferenceServer(3, ServerConfig(device="cuda", threads=2, compile=compile_, graphs=graphs))
        try:
            with srv.pool() as procs:
                results = procs.map(run_job, [Job(c, path, 1, max_turns=30, engine=ENGINE, inference="server") for c in split_games(specs, 3)])
            stats = srv.stats()
        finally:
            srv.close()
        assert (stats["graphs"] > 0) == graphs and stats["legacy"] == 0
        for r in results:
            _replay_matches(net, r)


def test_server_shards_workers_over_devices(tmp_path):
    """Two servers (here both on the CPU), workers split between them."""
    net, path = _ckpt(tmp_path, "gru", trunk="entity")
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER)) for s in range(8)]
    srv = InferenceServer(4, ServerConfig(device="cpu", threads=1), devices=["cpu", "cpu"])
    try:
        with srv.pool() as procs:
            results = procs.map(run_job, [Job(c, path, 1, max_turns=20, engine=ENGINE, inference="server") for c in split_games(specs, 4)])
        stats = srv.stats()
    finally:
        srv.close()
    assert len(stats["servers"]) == 2 and all(s["rows"] > 0 for s in stats["servers"])
    for r in results:
        _replay_matches(net, r)


@pytest.mark.parametrize("device", DEVICES)
def test_local_rollouts_replay_exactly_smoke(tmp_path, device):
    net, path = _ckpt(tmp_path, "gru", trunk="entity")
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER)) for s in range(4)]
    _replay_matches(net, run_job(Job(specs, path, 1, max_turns=30)))


@pytest.mark.parametrize("memory,trunk,value_net", VARIANTS)
def test_local_rollouts_replay_exactly(tmp_path, memory, trunk, value_net):
    net, path = _ckpt(tmp_path, memory, trunk=trunk, value_net=value_net)
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


def test_old_checkpoints_still_load():
    """Checkpoints from before the policy/value core split (top-level state_emb, trunk, gru...)."""
    net = PolicyNet(hidden=16)
    old = {k.replace("policy_core.", ""): v for k, v in net.state_dict().items()}
    fresh = PolicyNet(hidden=16)
    fresh.load_state_dict(old)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), fresh.state_dict().values()))


# ---------------------------------------------------------------------------
# The stacked step forward (rl/stacked.py)
# ---------------------------------------------------------------------------


def _decisions(n=60, seed=0):
    """Real featurized decisions (entities, pointers) with random events."""
    G = game_class(ENGINE)
    out, r, s = [], random.Random(seed), 0
    while len(out) < n:
        g = G(match_decks(1 + s % 2), seed=s, match_game=1 + s % 2)
        while not g.over and len(out) < n:
            st, ol, of = featurize_flat(g, g.decision.player)
            out.append(tuple(array("i", x) for x in (st, ol, of, encode_event_hashes([r.randrange(OPTION_DIM) for _ in range(r.randrange(6))]))))
            g.step(r.randrange(len(ol)))
        s += 1
    return out


STACKABLE = [("gru", "entity", "shared"), ("none", "entity", "separate"), ("gru", "entity", "separate"), ("gru", "mlp", "separate"), ("none", "mlp", "shared")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("memory,trunk,value_net", STACKABLE)
def test_stacked_forward_matches_per_policy_forwards(memory, trunk, value_net, device):
    """One stacked forward over rows of three policies (sorted by policy, two
    padded rows) gives each row what its own PolicyNet gives in step mode:
    log-prob of the sampled option, value, new hidden state."""
    decs = _decisions()
    slots = [3, 0, 2]  # stack slots of the three nets (capacity 4, slot 1 empty)
    nets = []
    for k in range(3):
        torch.manual_seed(k)
        net = PolicyNet(hidden=32, memory=memory, trunk=trunk, value_net=value_net, value_hidden=48).eval()
        with torch.no_grad():
            for p in net.parameters():
                p += 0.02 * torch.randn_like(p)  # non-zero value heads
        nets.append(net)
    stack = stacked.PolicyStack(nets[0].config, 4, device)
    for s, n in zip(slots, nets):
        stack.load(s, n)
    pols = sorted(random.Random(1).choices(slots, k=len(decs)))
    fresh = [i % 3 == 0 for i in range(len(decs))]
    table = torch.randn(len(decs) + 1, nets[0].state_size)
    dev_table = table.clone().to(device)  # updated in place
    x = stacked.step_input(decs, pols, list(range(len(decs))), fresh, device=device, pad=2)
    with torch.no_grad():
        act, logp, val = stacked.step_forward(stack, x, dev_table if memory == "gru" else None)
        act, logp, val, dev_table = act.cpu(), logp.cpu(), val.cpu(), dev_table.cpu()
        for s, net in zip(slots, nets):
            rows = [i for i, p in enumerate(pols) if p == s]
            batch, width = _batch(torch, [decs[i] for i in rows])
            h = None
            if memory == "gru":
                h = torch.where(torch.tensor([fresh[i] for i in rows])[:, None], 0.0, table[rows])
            logits, v, hn = net(batch, h, max_options=width)
            a = act[rows]
            assert (a < batch.n_opts).all()
            assert torch.allclose(torch.log_softmax(logits, -1).gather(1, a[:, None])[:, 0], logp[rows], atol=1e-5)
            assert torch.allclose(v, val[rows], atol=1e-5)
            if hn is not None:
                assert torch.allclose(hn, dev_table[rows], atol=1e-5)


def test_stacked_sampling_follows_the_policy():
    """Gumbel-max draws match the softmax: one decision drawn 4000 times."""
    decs = _decisions(40)
    d = max(decs, key=lambda x: len(x[1]))
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, memory="none", trunk="entity").eval()
    stack = stacked.PolicyStack(net.config, 1, "cpu")
    stack.load(0, net)
    n = 4000
    x = stacked.step_input([d] * n, [0] * n, list(range(n)), [True] * n)
    with torch.no_grad():
        act, logp, _ = stacked.step_forward(stack, x, None)
        logits, _, _ = net(_batch(torch, [d])[0], None)
    p = torch.softmax(logits[0], -1)
    freq = torch.bincount(act[:n], minlength=len(p)).float() / n
    assert (freq - p).abs().max() < 0.03, (freq, p)
    assert torch.allclose(logp[:n], torch.log(p)[act[:n]], atol=1e-5)


def test_step_structure_matches_model_structure():
    """The fixed-shape structure of the step forward (separators and pointers
    looked up as zero rows, no compaction) puts the same tokens in each bag
    as `model.structure`, and points options at the same entities."""
    decs = _decisions(50, seed=3)
    torch.manual_seed(0)
    net = PolicyNet(hidden=16, memory="none", trunk="entity")
    stack = stacked.PolicyStack(net.config, 1, "cpu")
    stack.load(0, net)
    x = stacked.step_input(decs, [0] * len(decs), list(range(len(decs))), [True] * len(decs), pad=3, n_ent=sum(list(d[0]).count(STATE_DIM) for d in decs) + 7)
    st = stacked.structure(stack, x)
    table = torch.randn(STATE_DIM + 1, 16, dtype=torch.float64)
    table[-1] = 0  # the zero row
    mine = torch.nn.functional.embedding_bag(st["s_emb"], table, st["bag_off"], mode="sum")
    ref = structure(_batch(torch, decs)[0])
    theirs = torch.nn.functional.embedding_bag(ref.st_idx.long(), table, ref.bag_off.long(), mode="sum")
    R, E = len(decs), ref.e_row.shape[0]
    assert torch.allclose(mine[st["g_bag"][:R]], theirs[ref.g_bag.long()])
    assert torch.allclose(mine[st["e_bag"][:E]], theirs[ref.e_bag.long()])
    assert torch.equal(st["e_row"][:E], ref.e_row.long())
    # pointers: (option, entity) pairs
    is_ptr = st["ptr_w"].bool()
    mine_ptr = sorted(zip(x.ot_opt[is_ptr].tolist(), st["ptr_ent"][is_ptr].tolist()))
    assert mine_ptr == sorted(zip(ref.p_opt.tolist(), ref.p_ent.tolist())) and mine_ptr
    # padded entities and tokens land in padded rows and bags only
    assert (st["e_row"][E:] == x.pol.shape[0] - 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_grouped_linear_kernel_matches_per_policy_matmuls():
    torch.manual_seed(0)
    P, M, K, N = 5, 777, 96, 80
    w, b = torch.randn(P, N, K, device="cuda"), torch.randn(P, N, device="cuda")
    pol = torch.randint(0, P, (M,), device="cuda").sort().values
    x = torch.randn(M, K, device="cuda")
    ref = torch.bmm(x[:, None, :].double(), w[pol].double().transpose(1, 2))[:, 0] + b[pol].double()
    assert (stacked.grouped_linear(x, w, b, pol).double() - ref).abs().max() < 1e-4


def test_worker_ids_are_never_shared():
    """A replacement pool worker takes a dead worker's id, never a live one's."""
    import multiprocessing as mp
    import os

    from mtg_ml.rl import inference

    ids = mp.get_context("spawn").Array("i", 3)
    dead = mp.get_context("spawn").Process(target=os.getpid)
    dead.start()
    dead.join()
    ids.get_obj()[:] = [os.getppid(), dead.pid, 0]  # a live process, a dead one, a free slot
    assert inference._claim_id(ids) == 1
    assert inference._claim_id(ids) == 1  # this process already holds it
    ids.get_obj()[1] = os.getppid()
    assert inference._claim_id(ids) == 2
    ids.get_obj()[2] = os.getppid()
    with pytest.raises(RuntimeError, match="no free inference worker id"):
        inference._claim_id(ids)


def _client(srv):
    from mtg_ml.rl.inference import InferenceClient

    first, n, layout, names, req_q, resp_qs = srv.shards[0]
    return InferenceClient(0, req_q, resp_qs[0], names, layout)


def test_oversized_request_is_split_into_chunks(tmp_path):
    """A request larger than the slot (ServerConfig.request_ints) goes out
    in chunks that fit and comes back whole: the same greedy actions and
    values as through a slot that holds it at once. A single decision
    larger than the slot is a clear error."""
    from mtg_ml.rl.inference import GREEDY_FLAG, REC

    _, path = _ckpt(tmp_path, "none", trunk="entity")
    xs = _decisions(40)
    rows = [((path, 0), k, 1 | GREEDY_FLAG, *x) for k, x in enumerate(xs)]
    sizes = [REC + len(x[0]) + len(x[1]) + len(x[2]) + len(x[3]) for x in xs]
    small = 2 * max(sizes)
    assert sum(sizes) > 4 * small  # at least five chunks
    out = {}
    for ints in (1 << 18, small):
        srv = InferenceServer(1, ServerConfig(device="cpu", threads=1, request_ints=ints, compile=False, graphs=False))
        try:
            cl = _client(srv)
            h = cl.submit(0, rows)
            assert bool(h[-1]) == (ints == small)  # the small slot leaves chunks for collect to send
            out[ints] = cl.collect(h)
            if ints == small:
                big = ((path, 0), 0, 1, xs[0][0], array("i", [1] * small), array("i", [1] * small), xs[0][3])
                with pytest.raises(ValueError, match="one decision of .* does not fit"):
                    cl.submit(0, [big])
                assert cl.collect(cl.submit(0, rows[:3]))[0] == out[1 << 18][0][:3]  # the slot still works
            cl.close()
        finally:
            srv.close()
    (a0, _, v0), (a1, _, v1) = out[1 << 18], out[small]
    assert len(a1) == len(rows) and a0 == a1
    assert torch.allclose(torch.tensor(v0), torch.tensor(v1), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("reaped", [True, False])
def test_worker_notices_a_killed_server(tmp_path, monkeypatch, reaped):
    """A server that dies without a Python exception (SIGKILL, OOM) makes
    waiting workers raise instead of spinning forever: by its pid once
    reaped, by its heartbeat while it is a zombie."""
    import os
    import signal

    from mtg_ml.rl import inference

    monkeypatch.setattr(inference, "SERVER_TIMEOUT_S", 2.0)
    _, path = _ckpt(tmp_path, "none")
    srv = InferenceServer(1, ServerConfig(device="cpu", threads=1))
    try:
        cl = _client(srv)
        g = game_class(ENGINE)(match_decks(1), seed=0)
        state, ol, of = featurize_flat(g, g.decision.player)
        row = ((path, 0), 0, 1, array("i", state), array("i", ol), array("i", of), array("i", encode_event_hashes([])))
        cl.collect(cl.submit(0, [row]))  # the server works
        os.kill(srv.proc.pid, signal.SIGKILL)
        if reaped:
            srv.proc.join()
        h = cl.submit(0, [row])
        with pytest.raises(RuntimeError, match="died|heartbeat"):
            cl.collect(h)
        with pytest.raises(RuntimeError, match="died|heartbeat"):
            cl._id(("other.pt", 0))
        cl.close()
    finally:
        srv.close()


def test_stack_takes_policies_of_another_feature_set():
    """The feature-set version has no weights: a set-3 learner and set-1/2
    pool snapshots share one stack (else every pool policy ran on the slow
    per-policy path)."""
    old, new = PolicyNet(hidden=16, trunk="entity"), PolicyNet(hidden=16, trunk="entity", features=3)
    assert old.config != new.config and stacked.stack_config(old.config) == stacked.stack_config(new.config)
    stack = stacked.PolicyStack(new.config, 2, "cpu")
    stack.load(0, old)
    stack.load(1, new)
    with pytest.raises(ValueError):
        stack.load(0, PolicyNet(hidden=32, trunk="entity"))
