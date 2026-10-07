"""Training knobs from docs/next-actions.md: value clamp and bound, learning
rate anneal (tensor lr, no recapture), per-game-turn discounting, the bot
seat, PFSP pool sampling and exploiter mode."""

import math
import os
import random
from array import array
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from test_train_pipeline import IN_PROCESS, NO_EVAL, _cfg, _free_native_garbage, _rows, in_process  # noqa: E402, F401

from mtg_ml.backend import game_class  # noqa: E402
from mtg_ml.encode import FEATURES  # noqa: E402
from mtg_ml.rl import stacked  # noqa: E402
from mtg_ml.rl.features import featurize_flat  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, _graphs_for, _StepGraphs, get_lr, load_optimizer_state, make_optimizer, ppo_update, set_lr  # noqa: E402
from mtg_ml.match import match_decks  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, GameSpec, Job, Result, Trajectory, _batch, _finish, run_job  # noqa: E402
from mtg_ml.rl.train import Trainer  # noqa: E402

# -- GAE: value clamp and per-turn discounting ---------------------------------


def _traj(values, turns=None, potentials=None):
    n = len(values)
    return Trajectory(samples=[(array("i"), array("i", [1]), array("i", [0]), array("i"))] * n, actions=[0] * n, logps=[0.0] * n, values=list(values), potentials=list(potentials or [0.0] * n), turns=list(turns or []))


def _gae(rewards, values, gam, lam):
    """Reference GAE with per-step discounts."""
    n, adv, last = len(rewards), [0.0] * len(rewards), 0.0
    for t in reversed(range(n)):
        nv = values[t + 1] if t + 1 < n else 0.0
        last = rewards[t] + gam[t] * nv - values[t] + gam[t] * lam[t] * last
        adv[t] = last
    return adv


def _run(traj, outcome, **job):
    out = Result()
    _finish(traj, outcome, Job([], "", 0, **job), out)
    return out


def test_finish_clamps_values_before_the_deltas():
    raw = [0.2, 1.41, -1.12, 0.9]
    traj = _traj(raw)
    out = _run(traj, 1.0, gamma=0.99, lam=0.95)
    clamped = [0.2, 1.0, -1.0, 0.9]
    exp = _gae([0, 0, 0, 1.0], clamped, [0.99] * 4, [0.95] * 4)
    assert out.advantages == pytest.approx(exp)
    assert out.returns == pytest.approx([a + v for a, v in zip(exp, clamped)])
    assert traj.values == raw  # the raw values stay for logging
    off = _run(_traj(raw), 1.0, gamma=0.99, lam=0.95, value_clamp=0)
    assert off.advantages == pytest.approx(_gae([0, 0, 0, 1.0], raw, [0.99] * 4, [0.95] * 4))


def test_value_clamp_leaves_room_for_shaping():
    out = _run(_traj([1.1, 0.5]), 1.0, gamma=1.0, lam=1.0, shaping=0.2)  # bound 1.2: nothing clamped
    assert out.returns[1] == pytest.approx(1.0) and out.advantages[0] == pytest.approx(1.0 - 1.1)


def test_per_turn_discount_between_decisions():
    turns = [1, 1, 1, 2, 4, 4]
    values = [0.1, 0.2, -0.3, 0.4, 0.5, 0.6]
    out = _run(_traj(values, turns), -1.0, gamma=0.5, lam=0.5, gamma_turn=0.9, lam_turn=0.8)
    gam = [1, 1, 0.9, 0.81, 1, 1]  # 0.9 ** turns elapsed to the next decision; the last one bootstraps from 0
    lam = [1, 1, 0.8, 0.64, 1, 1]
    assert out.advantages == pytest.approx(_gae([0, 0, 0, 0, 0, -1.0], values, gam, lam))
    # gamma per turn, lambda per decision
    out = _run(_traj(values, turns), -1.0, gamma=0.5, lam=0.5, gamma_turn=0.9)
    assert out.advantages == pytest.approx(_gae([0, 0, 0, 0, 0, -1.0], values, gam, [0.5] * 6))


def test_per_turn_equals_per_decision_with_one_decision_per_turn():
    values = [0.3, -0.2, 0.7, 0.1]
    pots = [0.1, -0.2, 0.05, 0.4]
    a = _run(_traj(values, [1, 2, 3, 4], pots), 1.0, gamma_turn=0.97, lam_turn=0.9, shaping=0.2)
    b = _run(_traj(values, [], pots), 1.0, gamma=0.97, lam=0.9, shaping=0.2)
    assert a.advantages == pytest.approx(b.advantages) and a.returns == pytest.approx(b.returns)


def test_per_turn_shaping_uses_the_same_discount():
    turns, pots = [1, 1, 3], [0.1, 0.3, -0.2]
    out = _run(_traj([0.0, 0.0, 0.0], turns, pots), 0.0, gamma_turn=0.9, lam_turn=1.0, shaping=1.0, value_clamp=0)
    gam = [1.0, 0.81, 1.0]
    rewards = [gam[0] * pots[1] - pots[0], gam[1] * pots[2] - pots[1], 0.0 - pots[2]]
    assert out.advantages == pytest.approx(_gae(rewards, [0, 0, 0], gam, [1, 1, 1]))


def test_per_turn_needs_the_turns():
    with pytest.raises(ValueError):
        _run(_traj([0.0, 0.0]), 1.0, gamma_turn=0.97)


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)  # the games below depend on the weights
    net = PolicyNet(hidden=16)
    path = str(tmp_path / "learner.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return path


def test_rollout_records_the_turn_of_each_decision(ckpt, monkeypatch):
    from mtg_ml.rl import rollout

    seen = []
    finish = rollout._finish
    monkeypatch.setattr(rollout, "_finish", lambda traj, *a: (seen.append(list(traj.turns)), finish(traj, *a)))
    res = run_job(Job([GameSpec(3, (LEARNER, LEARNER)), GameSpec(4, (LEARNER, BOT))], ckpt, 1, max_turns=8, gamma_turn=0.97, lam_turn=0.9))
    assert len(seen) == 3 and [len(t) for t in seen] == res.lengths
    assert all(t == sorted(t) for t in seen) and any(t[-1] > t[0] for t in seen)
    assert max(max(t) for t in seen) <= max(g[3] for g in res.games)


def test_bot_games_record_the_learner_seat_only(ckpt):
    res = run_job(Job([GameSpec(s, (LEARNER, BOT)) for s in (1, 2)], ckpt, 1, max_turns=8))
    assert len(res.lengths) == 2 and sum(res.lengths) == len(res.actions)
    assert sum(res.lengths) < sum(g[4] for g in res.games)  # the bot's decisions are not recorded


# -- value bound ---------------------------------------------------------------


def test_tanh_value_bound():
    torch.manual_seed(0)
    net = PolicyNet(hidden=16, value_net="separate")
    assert "value_bound" not in net.config  # configs of unbounded nets are unchanged
    bounded = PolicyNet(**net.config, value_bound="tanh")
    assert bounded.config["value_bound"] == "tanh" and PolicyNet(**bounded.config).value_bound == "tanh"
    bounded.load_state_dict(net.state_dict())  # no weights of its own
    with torch.no_grad():
        for n in (net, bounded):
            n.value_head[-1].bias.fill_(3.0)
    g = game_class("python")(match_decks(1), seed=1)
    p = g.decision.player
    st, ol, of = featurize_flat(g, p)
    x = (array("i", st), array("i", ol), array("i", of), array("i"))
    b, w = _batch(torch, [x])
    with torch.no_grad():
        v_raw = net(b, max_options=w)[1]
        v = bounded(b, max_options=w)[1]
    assert v_raw.item() == pytest.approx(3.0, abs=1e-5) and v.item() == pytest.approx(math.tanh(3.0), abs=1e-5)
    # the stacked (inference server) forward bounds it too
    stack = stacked.PolicyStack(bounded.config, 2, "cpu")
    stack.load(0, bounded.eval())
    xi = stacked.step_input([x], [0], [0], [True], device="cpu")
    with torch.no_grad():
        _, _, val = stacked.step_forward(stack, xi, torch.zeros(2, bounded.state_size))
    assert val[0].item() == pytest.approx(math.tanh(3.0), abs=1e-5)


# -- learning rate ---------------------------------------------------------------


def test_tensor_lr_is_set_in_place_and_keeps_the_graph_fingerprint():
    net, cfg = PolicyNet(hidden=16), PPOConfig()
    opt = make_optimizer(net.parameters(), 3e-4, "cpu", tensor_lr=True)
    lr = opt.param_groups[0]["lr"]
    assert torch.is_tensor(lr)
    fp = _StepGraphs.fingerprint_of(net, opt, cfg)
    set_lr(opt, 1e-4)
    assert opt.param_groups[0]["lr"] is lr and get_lr(opt) == pytest.approx(1e-4)
    assert _StepGraphs.fingerprint_of(net, opt, cfg) == fp  # captured graphs stay valid
    load_optimizer_state(opt, make_optimizer(net.parameters(), 3e-4, "cpu").state_dict())  # an old float-lr checkpoint
    assert opt.param_groups[0]["lr"] is lr
    plain = make_optimizer(net.parameters(), 3e-4, "cpu")
    fp = _StepGraphs.fingerprint_of(net, plain, cfg)
    set_lr(plain, 1e-4)
    assert plain.param_groups[0]["lr"] == 1e-4 and _StepGraphs.fingerprint_of(net, plain, cfg) != fp  # a float lr: recapture


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tensor_lr_update_equals_the_float_lr_update(ckpt, mode):
    res = run_job(Job([GameSpec(5, (LEARNER, LEARNER))], ckpt, 1, max_turns=6))
    nets = []
    for tensor_lr in (False, True):
        torch.manual_seed(0)
        net = PolicyNet(hidden=16)
        opt = make_optimizer(net.parameters(), 1.0, "cpu", tensor_lr=tensor_lr)
        set_lr(opt, 1e-3)
        ppo_update(net, opt, res, PPOConfig(epochs=2, minibatch=64, target_kl=None), gen=torch.Generator().manual_seed(0), mode=mode)
        nets.append(net)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(nets[0].parameters(), nets[1].parameters()))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_annealed_lr_does_not_recapture(ckpt):
    res = run_job(Job([GameSpec(s, (LEARNER, LEARNER)) for s in (5, 6)], ckpt, 1, max_turns=6))
    net = PolicyNet(hidden=16).cuda()
    opt = make_optimizer(net.parameters(), 3e-4, "cuda", tensor_lr=True)
    cfg = PPOConfig(epochs=1, minibatch=64, target_kl=None, capture=1)
    ppo_update(net, opt, res, cfg, device="cuda", gen=torch.Generator().manual_seed(0))  # Adam's first step runs eagerly
    ppo_update(net, opt, res, cfg, device="cuda", gen=torch.Generator().manual_seed(0))
    graphs = _graphs_for(net, opt, cfg)
    captures = graphs.captures
    before = [p.detach().clone() for p in net.parameters()]
    set_lr(opt, 0.0)
    ppo_update(net, opt, res, cfg, device="cuda", gen=torch.Generator().manual_seed(0))
    assert _graphs_for(net, opt, cfg) is graphs and graphs.captures == captures
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))  # the replays read the new lr


def _bare_trainer(*args):
    t = Trainer.__new__(Trainer)
    t.cfg = _cfg("unused", *args)
    t.rng, t.pool, t.pfsp, t.games_total, t.lr_origin = random.Random(0), [], {}, 0, 0
    return t


def test_lr_schedules():
    t = _bare_trainer("--lr-anneal-games", "1000", "--ppo-lr", "3e-4", "--ppo-lr-final", "3e-5")
    t.lr_origin = 200
    for games, lr in ((0, 3e-4), (200, 3e-4), (700, 1.65e-4), (1200, 3e-5), (5000, 3e-5)):
        t.games_total = games
        assert t._lr() == pytest.approx(lr)
    t.cfg.lr_schedule = "cosine"
    t.games_total = 450
    assert t._lr() == pytest.approx(3e-5 + 2.7e-4 * 0.5 * (1 + math.cos(math.pi / 4)))
    t.cfg.lr_anneal_games = 0
    assert t._lr() == 3e-4


def test_trainer_anneals_and_resumes_the_schedule(tmp_path, in_process):  # noqa: F811
    run = tmp_path / "run"
    args = ["--iterations", "2", "--lr-anneal-games", "8", "--ppo-lr-final", "1e-4", *IN_PROCESS, *NO_EVAL]
    Trainer(_cfg(run, *args)).train()
    assert [r["lr"] for r in _rows(run)] == pytest.approx([3e-4, 2e-4])  # 4 games per iteration
    t = Trainer(_cfg(run, *args[2:], "--iterations", "3"))
    assert t.lr_origin == 0 and torch.is_tensor(t.opt.param_groups[0]["lr"]) and get_lr(t.opt) == pytest.approx(1e-4)
    t.train()
    assert _rows(run)[-1]["lr"] == pytest.approx(1e-4)


# -- game mix: bot seat, PFSP, exploiter -----------------------------------------


def test_bot_seat_jund_puts_the_learner_on_jund():
    t = _bare_trainer("--self-play-frac", "0", "--bot-frac", "1", "--bot-seat", "jund", "--games-per-iter", "64")
    assert {sp.seats for sp in t._train_specs(0)} == {(LEARNER, BOT)}
    t = _bare_trainer("--self-play-frac", "0", "--bot-frac", "1", "--games-per-iter", "64")
    assert {sp.seats for sp in t._train_specs(0)} == {(LEARNER, BOT), (BOT, LEARNER)}


def test_pfsp_tracks_win_rates_and_prefers_opponents_that_win():
    t = _bare_trainer("--self-play-frac", "0", "--pool-recent-frac", "0", "--pool-sampling", "pfsp", "--games-per-iter", "4000", "--pfsp-ema", "0.5")
    t.pool = ["run/pool/iter_00000.pt", "run/pool/iter_00010.pt", "run/pool/iter_00020.pt"]
    a, b, c = t.pool
    games = [((LEARNER, a), 0), ((a, LEARNER), 1), ((LEARNER, b), 1), ((LEARNER, b), None), ((LEARNER, LEARNER), 0), (("other.pt", LEARNER), 0)]
    t._update_pfsp([(seats, w, None, 0, 0, 0) for seats, w in games])
    assert t.pfsp == pytest.approx({"iter_00000.pt": 0.875, "iter_00010.pt": 0.25 * 0.5 + 0.5 * 0.5})  # b: 0.5 -> 0.25 -> 0.375
    assert t._pfsp_weights() == pytest.approx([0.125**2, 0.625**2, 0.25])  # unplayed c: 0.5
    n = Counter(next(s for s in sp.seats if s != LEARNER) for sp in t._train_specs(0))
    assert n[a] < n[c] < n[b]
    t.pfsp = {os.path.basename(p): 1.0 for p in t.pool}
    assert t._pfsp_weights() is None  # all zero: uniform


def test_exploiter_trains_one_deck_against_the_frozen_main_policy(tmp_path, in_process):  # noqa: F811
    main = tmp_path / "main.pt"
    torch.manual_seed(7)
    net = PolicyNet(hidden=16)
    torch.save({"config": net.config, "model": net.state_dict()}, main)
    run = tmp_path / "exploit"
    t = Trainer(_cfg(run, "--iterations", "2", "--exploit", str(main), "--exploit-deck", "blue", "--hidden", "32", *IN_PROCESS, *NO_EVAL))
    assert t.net.config == net.config  # the main policy's architecture, weights and feature set, whatever --hidden says
    assert all(torch.equal(a, b) for a, b in zip(t.net.state_dict().values(), net.state_dict().values()))
    t.train()
    train = [g for r in in_process if r.train for g in r.games]
    assert train and all(seats == ("main.pt", LEARNER) for _, seats, _ in train)
    rows = _rows(run)
    assert all(0 <= r["win_vs_main"] <= 1 for r in rows)


def test_init_starts_a_fresh_run_from_a_checkpoint(tmp_path, in_process):  # noqa: F811
    src = tmp_path / "src.pt"
    torch.manual_seed(3)
    net = PolicyNet(hidden=16)
    torch.save({"config": net.config, "model": net.state_dict()}, src)
    t = Trainer(_cfg(tmp_path / "run", "--iterations", "1", "--init", str(src), "--value-bound", "tanh", *IN_PROCESS, *NO_EVAL))
    assert t.net.config == {**net.config, "value_bound": "tanh"}  # and the source's feature set (1)
    assert all(torch.equal(a, b) for a, b in zip(t.net.state_dict().values(), net.state_dict().values()))
    t.train()


def test_init_takes_features_only_when_given(tmp_path):
    src = tmp_path / "src.pt"
    net = PolicyNet(hidden=16, features=2)
    torch.save({"config": net.config, "model": net.state_dict()}, src)
    assert Trainer(_cfg(tmp_path / "a", "--init", str(src), *IN_PROCESS, *NO_EVAL)).net.features == 2
    assert Trainer(_cfg(tmp_path / "b", "--init", str(src), "--features", "1", *IN_PROCESS, *NO_EVAL)).net.features == 1
    assert Trainer(_cfg(tmp_path / "c", *IN_PROCESS, *NO_EVAL)).net.features == FEATURES


def test_resume_with_features_stamps_the_run(tmp_path, in_process):  # noqa: F811
    """A run trained on set 2 before checkpoints recorded it, resumed with
    --features 2: the learner and the run's own unstamped pool snapshots play
    in set 2, and the version is in every file written from then on."""
    from mtg_ml.rl.rollout import checkpoint_config

    run = tmp_path / "run"
    Trainer(_cfg(run, "--iterations", "1", "--features", "1", *IN_PROCESS, *NO_EVAL)).train()
    assert "features" not in checkpoint_config(str(run / "latest.pt"))
    t = Trainer(_cfg(run, "--iterations", "2", "--features", "2", *IN_PROCESS, *NO_EVAL))
    assert t.net.features == 2 and t.pool_features == {p: 2 for p in t.pool}
    t.train()
    assert checkpoint_config(str(run / "latest.pt"))["features"] == 2
    t = Trainer(_cfg(run, "--iterations", "3", *IN_PROCESS, *NO_EVAL))  # without the flag: keeps the stamped version
    assert t.net.features == 2 and t.pool_features == {p: 2 for p in t.pool if "features" not in checkpoint_config(p)}

