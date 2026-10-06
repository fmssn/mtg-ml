import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR, Game, expand  # noqa: E402
from mtg_ml.rl.features import encode_events, event_tokens, featurize  # noqa: E402
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
    assert state == sorted(set(state))
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
                      "--eval-bo3-matches", "2", "--bench-games", "4", "--bench-bo3-matches", "2"])
    assert isinstance(cfg, TrainConfig)
    Trainer(cfg).train()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "eval/random/jund" in lines[-1]
    last = json.loads(lines[-1])
    assert last["bench/jund_vs_bot_n"] == 4 and last["bench/jund_vs_bot_bo3_n"] == 2 and last["games_total"] == 8
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

    specs = paired_specs(BOT, 4, jund_only=True)
    assert all(sp.seats == (LEARNER, BOT) for sp in specs) and [sp.starting_player for sp in specs] == [0, 1, 0, 1]
    _, path = learner_ckpt
    res = benchmark(_InProcess(), path, games=4, bo3_matches=2, n_jobs=2, version=1, max_turns=8)
    assert res["bench/jund_vs_bot_n"] == 4 and res["bench/jund_vs_bot_bo3_n"] == 2


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
