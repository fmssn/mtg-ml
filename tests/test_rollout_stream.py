"""Shared game queue (`rollout.run_specs`) and the vectorised local evaluator.

Streaming must play every spec exactly once whatever the game lengths, and
record exactly what the network computes when PPO replays the trajectories
(hidden-state slots are reused across games, so a stale state would show).
The evaluator must give what the old per-row evaluator gave."""

import collections
import gc
import multiprocessing as mp
import os
from array import array

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.backend import game_class, native_available  # noqa: E402
from mtg_ml.match import match_decks  # noqa: E402
from mtg_ml.rl import rollout  # noqa: E402
from mtg_ml.rl.features import encode_event_hashes, featurize_flat  # noqa: E402
from mtg_ml.rl.inference import InferenceServer, ServerConfig, _Server  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, RANDOM, GameSpec, Job, create_pool, load_policy, run_job, run_specs  # noqa: E402
from mtg_ml.rl.samples import PackedSamples  # noqa: E402

ENGINE = "native" if native_available() else "python"


def _ckpt(tmp_path, memory="gru", seed=0, trunk="mlp"):
    torch.manual_seed(seed)
    net = PolicyNet(hidden=32, memory=memory, trunk=trunk)
    path = str(tmp_path / f"net_{memory}_{trunk}_{seed}.pt")
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


def _specs(pool_path, n=24):
    seats = [(LEARNER, LEARNER), (LEARNER, pool_path), (pool_path, LEARNER), (LEARNER, BOT), (RANDOM, LEARNER), (BOT, RANDOM)]
    return [GameSpec(seed=100 + s, seats=seats[s % len(seats)], match_game=1 + s % 3, starting_player=s % 2 if s % 4 else None) for s in range(n)]


def test_streaming_plays_every_spec_once(tmp_path):
    _, path = _ckpt(tmp_path)
    _, pool_path = _ckpt(tmp_path, seed=1)
    specs = _specs(pool_path, 30)
    with create_pool(3) as procs:
        res = run_specs(procs, specs, Job([], path, 1, max_turns=40, engine=ENGINE), 3, inflight=2)
        again = run_specs(procs, specs[:5], Job([], path, 2, max_turns=40, engine=ENGINE), 3, inflight=4)  # counters reset per call
    assert collections.Counter((g[5], g[0]) for g in res.games) == collections.Counter((sp.seed, sp.seats) for sp in specs)
    assert sorted(g[5] for g in again.games) == [sp.seed for sp in specs[:5]]
    assert len(res.timing) == 3 and len({g[4] for g in res.games}) > 5  # one entry per job; unequal game lengths
    assert len(res.lengths) == sum(sp.seats.count(LEARNER) for sp in specs)
    # scripted-only games are deterministic: same outcome as a plain job (match game, starting player respected)
    scripted = [sp for sp in specs if LEARNER not in sp.seats]
    plain = {g[5]: g for g in run_job(Job(scripted, path, 1, max_turns=40, engine=ENGINE)).games}
    for g in res.games:
        if g[5] in plain:
            assert g == plain[g[5]]


def test_streaming_needs_a_shared_counter(tmp_path):
    _, path = _ckpt(tmp_path)
    with pytest.raises(RuntimeError, match="create_pool"):
        run_job(Job([GameSpec(1, (LEARNER, LEARNER))], path, 1, inflight=1))


@pytest.mark.parametrize("inference", ["local", "server"])
@pytest.mark.parametrize("memory", ["gru", "none"])
@pytest.mark.parametrize("trunk", ["mlp", "entity"])
def test_streaming_rollouts_replay_exactly(tmp_path, inference, memory, trunk):
    net, path = _ckpt(tmp_path, memory, trunk=trunk)
    _, pool_path = _ckpt(tmp_path, memory, seed=1, trunk=trunk)
    srv = InferenceServer(3, ServerConfig(device="cpu", max_wait_ms=2.0, threads=2)) if inference == "server" else None
    try:
        with create_pool(3, inference, srv) as procs:
            res = run_specs(procs, _specs(pool_path), Job([], path, 1, max_turns=30, engine=ENGINE, inference=inference, groups=2), 3, inflight=3)
    finally:
        if srv is not None:
            srv.close()
    assert len(res.games) == 24 and res.actions
    _replay_matches(net, res)


class _PerRowEvaluator:
    """The local evaluator before vectorisation (frozen copy): per-row hidden
    states in a dict keyed by (game, seat), PackedSamples + collate per
    policy, per-row int()/float()."""

    def __init__(self, learner_path, version):
        self.learner = load_policy(learner_path, version)
        self.hidden = {}

    def submit(self, items):
        acts_out, logp_out, val_out = [0] * len(items), [0.0] * len(items), [0.0] * len(items)
        by_pol = {}
        for k, it in enumerate(items):
            by_pol.setdefault(it[0], []).append(k)
        with torch.no_grad():
            for pol, ks in by_pol.items():
                net = self.learner if pol == LEARNER else load_policy(pol)
                hidden = None
                if net.memory != "none":
                    hs = [self.hidden.get((items[k][1], items[k][2])) for k in ks]
                    hidden = torch.stack([net.initial_state(1)[0] if h is None else h for h in hs])
                ps = PackedSamples()
                ps.extend(items[k][3] for k in ks)
                logits, values, hn = net(collate(ps), hidden)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
                for r, k in enumerate(ks):
                    if hn is not None:
                        self.hidden[(items[k][1], items[k][2])] = hn[r]
                    acts_out[k], logp_out[k], val_out[k] = int(acts[r]), float(logps[r]), float(values[r])
        return acts_out, logp_out, val_out


@pytest.mark.parametrize("trunk", ["mlp", "entity"])
def test_vectorised_evaluator_matches_per_row(tmp_path, trunk):
    _, path = _ckpt(tmp_path, trunk=trunk)
    _, pool_path = _ckpt(tmp_path, seed=1, trunk=trunk)
    Game = game_class(ENGINE)
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER) if s % 2 else (pool_path, LEARNER)) for s in range(6)]
    games = [Game(match_decks(1), seed=sp.seed, max_turns=20) for sp in specs]
    new = rollout._LocalEvaluator(Job(specs, path, 1), slots=2 * len(specs))
    old = _PerRowEvaluator(path, 1)
    seen, steps = set(), 0
    while steps < 60 and not all(g.over for g in games):
        old_items, new_items = [], []
        for i, g in enumerate(games):
            if g.over:
                continue
            p = g.decision.player
            st, ol, of = featurize_flat(g, p)
            ev = encode_event_hashes([i, steps])
            pol = specs[i].seats[p]
            old_items.append((pol, i, p, (st, ol, of, ev)))
            new_items.append((pol, 2 * i + p, (i, p) not in seen, tuple(array("i", x) for x in (st, ol, of, ev))))
            seen.add((i, p))
        order = sorted(range(len(new_items)), key=lambda k: new_items[k][0])  # the evaluator takes items sorted by policy
        old_items, new_items = [old_items[k] for k in order], [new_items[k] for k in order]
        torch.manual_seed(steps)
        a_old, lp_old, v_old = old.submit(old_items)
        torch.manual_seed(steps)
        a_new, lp_new, v_new = new.collect(new.submit(0, new_items))
        assert a_new == a_old and lp_new == pytest.approx(lp_old, abs=1e-6) and v_new == pytest.approx(v_old, abs=1e-6)
        for it, a in zip(new_items, a_new):
            games[it[1] // 2].step(a)
        steps += 1
    assert steps > 10


def test_new_learner_version_evicts_older_ones_whatever_the_path(tmp_path):
    _, v1 = _ckpt(tmp_path, seed=0)
    _, frozen = _ckpt(tmp_path, seed=1)
    _, v2 = _ckpt(tmp_path, seed=2)
    rollout._MODELS.clear()
    load_policy(v1, 1)
    load_policy(frozen)
    load_policy(v2, 2)
    assert (v1, 1) not in rollout._MODELS and (frozen, 0) in rollout._MODELS and (v2, 2) in rollout._MODELS
    threads = torch.get_num_threads()
    srv = _Server(ServerConfig(device="cpu", threads=1), 1)
    torch.set_num_threads(threads)
    for key in ((v1, 1), (frozen, 0), (v2, 2)):
        srv.model(key)
    assert set(srv.models) == {(frozen, 0), (v2, 2)}


def test_loaded_policy_has_the_saved_weights(tmp_path):
    net, path = _ckpt(tmp_path, trunk="entity")
    loaded = rollout.load_net(path)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), loaded.state_dict().values()))
    assert not loaded.training


def test_worker_profile_hook(tmp_path, monkeypatch):
    _, path = _ckpt(tmp_path)
    prof = tmp_path / "w.prof"
    monkeypatch.setenv("MTG_WORKER_PROFILE", str(prof))
    monkeypatch.setattr(rollout, "_PROFILE", None)  # restored after the test
    run_job(Job([GameSpec(3, (LEARNER, RANDOM))], path, 1, max_turns=4, engine=ENGINE))
    assert os.path.exists(f"{prof}.{os.getpid()}")


def test_matchup_bounds_group_specs_by_opponent_network():
    specs = sorted([GameSpec(s, seats) for s, seats in enumerate([("a", LEARNER), (LEARNER, LEARNER), (LEARNER, "b"), (LEARNER, "a"), (BOT, LEARNER)])], key=rollout._matchup)
    assert [rollout._matchup(sp) for sp in specs] == [(), (), ("a",), ("a",), ("b",)]
    assert rollout._matchup_bounds(specs) == [0, 2, 4, 5]
    assert rollout._matchup_bounds(specs[2:]) == [0, 0, 2, 3]  # run 0 (learner-only) is always there


def test_claims_stick_to_one_opponent(tmp_path, monkeypatch):
    """With per-matchup counters a worker takes its current opponent's games,
    then learner-only games, then the opponent with the most games left."""
    specs = sorted([GameSpec(s, (LEARNER, LEARNER)) for s in range(4)] + [GameSpec(10 + s, (LEARNER, "a")) for s in range(3)] + [GameSpec(20 + s, (LEARNER, "b")) for s in range(5)], key=rollout._matchup)
    claim = mp.get_context("spawn").Array("i", rollout.MAX_MATCHUPS)
    claim.get_obj()[:3] = rollout._matchup_bounds(specs)[:-1]
    monkeypatch.setattr(rollout, "_CLAIM", claim)
    take = rollout._claimer(Job(specs, "unused", 1, inflight=4))
    got = [[rollout._matchup(specs[i]) for i in take(n)] for n in (3, 4, 4, 4)]
    assert got[0] == [("b",)] * 3  # starts on the matchup with most games left
    assert got[1] == [("b",)] * 2 + [()] * 2  # then learner-only games
    assert got[2] == [()] * 2 + [("a",)] * 2  # then switches to the next opponent
    assert got[3] == [("a",)] and take(1) == []


def test_failed_job_collects_requests_in_flight(monkeypatch):
    """A job that raises with several groups' requests in flight collects
    them all before the error leaves it (the next job reuses their areas)."""
    submitted, collected = [], []

    class FakeEvaluator:
        def __init__(self, job, slots):
            pass

        def submit(self, group, items):
            submitted.append((group, len(submitted)))
            return (submitted[-1], len(items))

        def collect(self, handle):
            collected.append(handle[0])
            return [0] * handle[1], [0.0] * handle[1], [0.0] * handle[1]

    real_step, calls = rollout._step, [0]

    def failing_step(g, seats, a):
        calls[0] += 1
        if calls[0] > 5 and len(submitted) >= 2:
            raise RuntimeError("boom")
        real_step(g, seats, a)

    monkeypatch.setattr(rollout, "_ServerEvaluator", FakeEvaluator)
    monkeypatch.setattr(rollout, "_step", failing_step)
    specs = [GameSpec(seed=s, seats=(LEARNER, LEARNER)) for s in range(8)]
    with pytest.raises(RuntimeError, match="boom"):
        run_job(Job(specs, "unused.pt", 1, max_turns=20, engine=ENGINE, inference="server", groups=4))
    assert len(submitted) >= 2 and set(collected) == set(submitted)  # every request collected (some twice: harmless)
    # The traceback keeps the job's native games alive in a reference cycle;
    # free them here, not in a later test's training thread (they are unsendable).
    gc.collect()
