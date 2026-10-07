import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, expand  # noqa: E402
from mtg_ml.rl.features import STATE_DIM, encode_events, event_tokens, featurize  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate, masked_entropy  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, RANDOM, GameSpec, Job, run_job  # noqa: E402
from mtg_ml.rl.train import TrainConfig, Trainer, parse_args  # noqa: E402

DECKS = (expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR))


def _positions(n=40, seed=0):
    g, r, out = Game(DECKS, seed=seed), random.Random(seed), []
    while not g.over and len(out) < n:
        state, opts = featurize(g, g.decision.player)
        out.append((state, opts, encode_events([])))
        g.step(r.randrange(len(g.legal_options())))
    return out


def test_featurize_one_entry_per_legal_option():
    g = Game(DECKS, seed=3)
    state, opts = featurize(g, g.decision.player)
    glob = state[: state.index(STATE_DIM)]  # then one segment per entity: from set 5 on the decider's hand
    assert glob == sorted(set(glob)) and state.count(STATE_DIM) == 7
    assert featurize(g, g.decision.player, features=4)[0] == glob  # no permanents or stack yet
    assert opts and all(opts)
    assert len(opts) == len(g.legal_options())


def test_logits_masked_to_legal_options():
    xs = _positions()
    net = PolicyNet(hidden=16)
    logits, values, hidden = net(collate(xs))
    assert values.shape == (len(xs),) and hidden.shape == (len(xs), 16)
    for row, (_, opts, _) in enumerate(xs):
        assert torch.isfinite(logits[row, : len(opts)]).all()
        assert torch.isinf(logits[row, len(opts) :]).all()
    p = torch.softmax(logits, -1)
    assert torch.allclose(p.sum(-1), torch.ones(len(xs)))
    ent = masked_entropy(logits)
    assert torch.isfinite(ent).all()
    for row, (_, opts, _) in enumerate(xs):
        assert ent[row] <= math.log(len(opts)) + 1e-5


def test_hidden_opponent_choices_are_not_leaked():
    assert event_tokens("choose_card", ("put_back", "Lightning Bolt"), mine=False) == ["opp>choose_card"]
    assert any("Lightning Bolt" in t for t in event_tokens("choose_card", ("put_back", "Lightning Bolt"), mine=True))
    assert any("Delver of Secrets" in t for t in event_tokens("priority", ("cast", "Delver of Secrets", "hand", "normal"), mine=False))


def test_memory_changes_the_policy():
    """Same position, different history -> different outputs (GRU state is used)."""
    (state, opts, ev), = _positions(1)
    net = PolicyNet(hidden=16)
    b = collate([(state, opts, ev)])
    l0, _, h1 = net(b)
    l1, _, _ = net(b, hidden=h1 * 0 + 1.0)
    assert not torch.allclose(l0, l1)


@pytest.fixture(params=["gru", "none"])
def learner_ckpt(tmp_path, request):
    net = PolicyNet(hidden=16, memory=request.param)
    path = str(tmp_path / "learner.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


def test_rollout_records_learner_seats_only(learner_ckpt):
    net, path = learner_ckpt
    specs = [GameSpec(1, (LEARNER, LEARNER)), GameSpec(2, (LEARNER, RANDOM)), GameSpec(3, (RANDOM, LEARNER))]
    res = run_job(Job(specs, path, 1, shaping=0.1, max_turns=8))
    assert len(res.games) == 3
    n = len(res.actions)
    assert n > 0 and len(res.samples) == len(res.logps) == len(res.advantages) == len(res.returns) == n
    assert sum(res.lengths) == n and len(res.lengths) == 4  # one trajectory per learner seat
    for (state, opts, events), a in zip(res.samples, res.actions):
        assert 0 <= a < len(opts) and events
    # replaying whole trajectories (training mode) reproduces the rollout log-probs
    logits, _, _ = net(collate(res.samples), lengths=res.lengths)
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(res.actions)[:, None]).squeeze(1)
    assert torch.allclose(lp, torch.tensor(res.logps), atol=1e-4)


def test_ppo_update_runs_and_changes_weights(learner_ckpt):
    net, path = learner_ckpt
    res = run_job(Job([GameSpec(5, (LEARNER, LEARNER))], path, 1, max_turns=6))
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = ppo_update(net, opt, res, PPOConfig(epochs=2, minibatch=64, target_kl=None))
    assert all(math.isfinite(v) for k, v in stats.items() if isinstance(v, float) and k != "explained_var")
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_trainer_smoke_and_resume(tmp_path):
    cfg = parse_args(["--run", str(tmp_path / "run"), "--iterations", "2", "--games-per-iter", "4", "--workers", "2",
                      "--hidden", "16", "--max-turns", "6", "--snapshot-every", "1", "--eval-every", "2", "--eval-games", "4",
                      "--eval-bo3-matches", "2", "--bench-games", "4", "--bench-greedy-games", "4", "--bench-bo3-matches", "2"])
    assert isinstance(cfg, TrainConfig)
    Trainer(cfg).train()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "eval/random/jund" in lines[-1]
    last = json.loads(lines[-1])
    assert last["bench/jund_vs_bot_n"] == 4 and last["bench/jund_vs_bot_greedy_n"] == 4 and "pay_mana_share" in last and "nt_frac" in last and last["bench/jund_vs_bot_bo3_n"] == 2 and last["games_total"] == 8
    assert len(list((tmp_path / "run" / "pool").iterdir())) == 3
    cfg.iterations = 3
    t = Trainer(cfg)
    assert t.iteration == 2
    t.train()
    assert len((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()) == 3


def test_trainer_stops_at_total_games(tmp_path):
    cfg = parse_args(["--run", str(tmp_path / "run"), "--iterations", "10", "--total-games", "6", "--games-per-iter", "4",
                      "--workers", "2", "--hidden", "16", "--max-turns", "6", "--eval-every", "0"])
    t = Trainer(cfg)
    t.train()
    assert t.iteration == 2 and t.games_total == 8
    assert Trainer(cfg).games_total == 8  # restored from the checkpoint


class _InProcess:
    map = staticmethod(lambda f, xs: list(map(f, xs)))


def test_benchmark_keeps_learner_on_jund(learner_ckpt):
    from mtg_ml.rl.evaluate import benchmark, paired_specs

    specs = paired_specs(BOT, 4, seat=0)
    assert all(sp.seats == (LEARNER, BOT) for sp in specs) and [sp.starting_player for sp in specs] == [0, 1, 0, 1]
    _, path = learner_ckpt
    res = benchmark(_InProcess(), path, games=4, bo3_matches=2, n_jobs=2, version=1, max_turns=8)
    assert res["bench/jund_vs_bot_n"] == 4 and res["bench/jund_vs_bot_bo3_n"] == 2


def test_benchmark_with_action_decomposition(learner_ckpt):
    from mtg_ml.rl.evaluate import benchmark

    _, path = learner_ckpt
    res = benchmark(_InProcess(), path, games=4, bo3_matches=0, n_jobs=2, version=1, max_turns=8, auto_mana=True, auto_pass=True)
    assert res["bench/jund_vs_bot_n"] == 4


def test_bo3_eval_plays_two_or_three_games_per_match(learner_ckpt):
    from mtg_ml.rl.evaluate import head_to_head_bo3

    _, path = learner_ckpt
    res = head_to_head_bo3(_InProcess(), path, RANDOM, matches=4, n_jobs=2, version=1, max_turns=8)
    assert res["all"][2] == 4 and res["jund"][2] == 2 and res["blue"][2] == 2


def test_entity_trunk_scores_options_through_their_entities():
    """Options that point at different entities get different inputs, and
    the pointer projection receives gradient."""
    from mtg_ml.rl.features import OPTION_DIM, featurize
    from mtg_ml.rl.model import collate

    torch.manual_seed(0)
    net = PolicyNet(hidden=16, trunk="entity")
    g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=5)
    rng = random.Random(0)
    samples = []
    while not g.over and len(samples) < 200:
        st, opts = featurize(g, g.decision.player)
        if any(t >= OPTION_DIM for o in opts for t in o):
            samples.append((st, opts, []))
        g.step(rng.randrange(len(g.legal_options())))
    assert samples, "no option pointed at an entity"
    logits, values, _ = net(collate(samples))
    assert torch.isfinite(values).all() and torch.isfinite(logits.max(-1).values).all()
    logits[torch.isfinite(logits)].sum().backward()
    assert net.pointer.weight.grad.abs().sum() > 0


def _ckpt(tmp_path, name, **kw):
    net = PolicyNet(hidden=16, **kw)
    path = str(tmp_path / name)
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


def test_feature_set_version_travels_with_the_model(tmp_path):
    """`features` is in the config only when not 1, so configs written before
    it existed are unchanged and mean feature set 1; it survives a save."""
    from mtg_ml.encode import FEATURES
    from mtg_ml.rl.rollout import checkpoint_config, load_net, policy_features

    old, old_path = _ckpt(tmp_path, "old.pt")
    new, new_path = _ckpt(tmp_path, "new.pt", features=2)
    assert "features" not in old.config and old.features == 1 and new.config["features"] == 2
    assert checkpoint_config(new_path) == new.config and checkpoint_config(old_path) == old.config
    assert (policy_features(old_path), policy_features(new_path)) == (1, 2)
    assert load_net(new_path).features == 2 and load_net(old_path).features == 1
    assert FEATURES == 5 and TrainConfig().features == 0  # 0: new runs train on the latest set (--init / resume: the source's)
    with pytest.raises(ValueError):
        PolicyNet(hidden=16, features=6)


def test_each_seat_is_featurized_with_its_own_policys_version(tmp_path, monkeypatch):
    """A learner on feature set 2 against a pool snapshot on set 1: every
    decision is featurized in the deciding seat's version, and what the
    learner records is exactly that."""
    from mtg_ml.rl import rollout

    _, new_path = _ckpt(tmp_path, "new.pt", features=2)
    _, old_path = _ckpt(tmp_path, "old.pt")
    calls = []
    real = rollout.featurize_flat

    def spy(g, p, features):
        calls.append((id(g), p, features))
        return real(g, p, features=features)

    monkeypatch.setattr(rollout, "featurize_flat", spy)
    specs = [GameSpec(1, (LEARNER, old_path)), GameSpec(2, (old_path, LEARNER))]
    res = run_job(Job(specs, new_path, 1, max_turns=10))
    per_game: dict = {}
    for gid, p, f in calls:
        per_game.setdefault(gid, set()).add((p, f))
    assert sorted(sorted(v) for v in per_game.values()) == [[(0, 1), (1, 2)], [(0, 2), (1, 1)]]
    assert len(res.actions) == sum(1 for _, _, f in calls if f == 2)  # the learner records its own (set 2) decisions


def test_model_agent_uses_its_networks_feature_set(tmp_path, monkeypatch):
    from mtg_ml.rl import agent as agent_mod

    _, old_path = _ckpt(tmp_path, "old.pt")
    seen = []
    real = agent_mod.featurize_flat
    monkeypatch.setattr(agent_mod, "featurize_flat", lambda g, p, features: seen.append(features) or real(g, p, features=features))
    g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=3)
    a = agent_mod.ModelAgent(old_path, seat=g.decision.player)
    a.act(g)
    assert seen == [1]


def test_job_features_override_a_checkpoints_version(tmp_path, monkeypatch):
    """`Job.features` (evaluate / ladder `--features`) featurizes a seat in a
    given version whatever its checkpoint records; the others keep theirs."""
    from mtg_ml.rl import rollout

    _, a = _ckpt(tmp_path, "a.pt")
    _, b = _ckpt(tmp_path, "b.pt")
    calls = []
    real = rollout.featurize_flat
    monkeypatch.setattr(rollout, "featurize_flat", lambda g, p, features: calls.append((p, features)) or real(g, p, features=features))
    run_job(Job([GameSpec(1, (LEARNER, b))], a, 1, record=False, max_turns=6, features={LEARNER: 2}))
    assert {p: f for p, f in calls} == {0: 2, 1: 1}
    calls.clear()
    run_job(Job([GameSpec(1, (LEARNER, b))], a, 1, record=False, max_turns=6, features={b: 2}))
    assert {p: f for p, f in calls} == {0: 1, 1: 2}


def test_model_agent_features_override(tmp_path, monkeypatch):
    from mtg_ml.rl import agent as agent_mod

    _, old_path = _ckpt(tmp_path, "old.pt")
    seen = []
    real = agent_mod.featurize_flat
    monkeypatch.setattr(agent_mod, "featurize_flat", lambda g, p, features: seen.append(features) or real(g, p, features=features))
    g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=3)
    agent_mod.ModelAgent(old_path, seat=g.decision.player, features=2).act(g)
    assert seen == [2]


def _stamp():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("stamp_features", os.path.join(os.path.dirname(__file__), "..", "tools", "stamp_features.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stamp_features_writes_a_copy(tmp_path):
    """tools/stamp_features.py: a copy of a policy file or a full checkpoint
    with `features` in its config, weights and everything else unchanged; the
    source is never modified and an existing destination never overwritten."""
    from mtg_ml.rl.rollout import checkpoint_config, load_net, policy_features

    stamp = _stamp()
    net, policy = _ckpt(tmp_path, "policy.pt")
    full = str(tmp_path / "latest.pt")
    torch.save({"config": net.config, "model": net.state_dict(), "optim": {"state": {}}, "iteration": 7}, full)
    before = {p: open(p, "rb").read() for p in (policy, full)}
    for src in (policy, full):
        dst = src.replace(".pt", ".f2.pt")
        assert stamp.stamp(src, dst, 2) == {**net.config, "features": 2}
        assert checkpoint_config(dst) == {**net.config, "features": 2} and policy_features(dst) == 2
        assert all(torch.equal(x, y) for x, y in zip(load_net(dst).state_dict().values(), net.state_dict().values()))
        with pytest.raises(FileExistsError):
            stamp.stamp(src, dst, 2)
        with pytest.raises(FileExistsError):
            stamp.stamp(src, src, 2)
    assert torch.load(str(tmp_path / "latest.f2.pt"), weights_only=False)["iteration"] == 7
    assert {p: open(p, "rb").read() for p in (policy, full)} == before
    with pytest.raises(ValueError):  # it already records set 2
        stamp.stamp(str(tmp_path / "policy.f2.pt"), str(tmp_path / "back.pt"), 1)
    assert not (tmp_path / "back.pt").exists()
    with pytest.raises(SystemExit):
        stamp.main([policy, str(tmp_path / "policy.f2.pt"), "--features", "2"])
