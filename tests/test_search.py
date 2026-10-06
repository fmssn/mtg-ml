"""Own-turn search (rl/search.py): finds a deep combo the policy prior hides,
plugs into rollouts as distillation targets, and PPO trains on them."""

from __future__ import annotations

import math

import pytest

from mtg_ml.rl.ppo import PPOConfig, ppo_update
from mtg_ml.rl.rollout import BOT, LEARNER, GameSpec, Job, run_job
from mtg_ml.rl.search import NetEvaluator, SearchConfig, eligible, search

from helpers import choose, scenario

torch = pytest.importorskip("torch")
from mtg_ml.rl.model import PolicyNet  # noqa: E402

import random  # noqa: E402

TERROR_TAPPED_OUT = {"battlefield": [("Tolarian Terror", {"tapped": True})] + [("Island", {"tapped": True})] * 5}
SHAMAN_TOXIN = {"hand": ["Krark-Clan Shaman", "Toxin Analysis"], "battlefield": ["Swamp", "Slagwoods Bridge", "Ichor Wellspring"]}
SWEEP = "Krark-Clan Shaman: 1 damage to each creature without flying"


class _BoardOnly:
    """A critic that only knows whether the opponent still has a creature, and
    a flat policy: what a value head looks like before the combo was ever seen
    through to the end. Argmax auto-play takes option 0 (pass / first payment)."""

    def __call__(self, game, player, hidden, events):
        n = len(game.decision.options)
        alive = any(c.controller == 1 and game.is_creature(c) for c in game.battlefield)
        v = -0.5 if alive else 1.0
        return [0.0] * n, v if player == 0 else -v, None


def test_search_finds_the_shaman_sweep():
    g = scenario(SHAMAN_TOXIN, TERROR_TAPPED_OUT)
    cfg = SearchConfig(budget=200, max_depth=8, determinize=False)
    assert eligible(g, 0, cfg)
    res = search(g, 0, _BoardOnly(), cfg, random.Random(1))
    labels = [o.label for o in g.decision.options]
    assert res.value == 1.0 and res.root_value == -0.5
    assert res.line[0] == labels[res.action]
    assert any(s.startswith("Target Krark-Clan Shaman") for s in res.line), res.line
    assert SWEEP in res.line, res.line
    assert not any(s.startswith("Sacrifice") for s in res.line)  # the sacrifice cost is the policy's call, not a node
    assert len(res.policy) == len(labels) and math.isclose(sum(res.policy), 1.0, abs_tol=1e-6)
    assert res.policy[res.action] == max(res.policy)
    # the improved policy keeps mass off a line that cannot reach the sweep: Toxin first
    assert res.policy[labels.index("Cast Toxin Analysis")] < res.policy[res.action]
    assert len(g.actions) == 0  # the game itself is untouched


def test_search_with_the_sweep_unavailable_prefers_nothing_in_particular():
    # No artifact to sacrifice: the sweep can never happen, so every line is worth -0.5.
    g = scenario({"hand": ["Krark-Clan Shaman", "Toxin Analysis"], "battlefield": ["Swamp", "Mountain"]}, TERROR_TAPPED_OUT)
    res = search(g, 0, _BoardOnly(), SearchConfig(budget=40, determinize=False), random.Random(2))
    assert res.value == -0.5 and SWEEP not in res.line


def test_eligibility_is_own_turn_main_phase_with_a_real_choice():
    cfg = SearchConfig()
    g = scenario(SHAMAN_TOXIN, TERROR_TAPPED_OUT)
    assert eligible(g, 0, cfg) and not eligible(g, 1, cfg)
    only_land = scenario({"hand": ["Swamp"], "battlefield": ["Swamp"]}, {})
    assert not eligible(only_land, 0, cfg)  # pass or play a land: nothing to search
    theirs = scenario(TERROR_TAPPED_OUT, SHAMAN_TOXIN, active=1)
    assert not eligible(theirs, 0, cfg) and eligible(theirs, 1, cfg)
    g = scenario(SHAMAN_TOXIN, TERROR_TAPPED_OUT)
    choose(g, "Cast Krark-Clan Shaman")
    while g.decision.kind == "pay_mana":
        g.step(0)
    assert g.stack and not eligible(g, 0, cfg)  # holding priority over our own spell is not searched


def test_search_with_a_network_evaluator_returns_a_valid_policy():
    net = PolicyNet(hidden=16, trunk="entity")
    g = scenario(SHAMAN_TOXIN, TERROR_TAPPED_OUT)
    res = search(g, 0, NetEvaluator(net), SearchConfig(budget=6, max_depth=3), random.Random(3))
    n = len(g.decision.options)
    assert 0 <= res.action < n and len(res.policy) == n and math.isclose(sum(res.policy), 1.0, abs_tol=1e-6)
    assert res.expansions >= 1 and res.evals >= res.expansions and res.line


@pytest.fixture(params=["gru", "none"])
def learner_ckpt(tmp_path, request):
    net = PolicyNet(hidden=16, memory=request.param, trunk="entity")
    path = str(tmp_path / "learner.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


def test_rollout_records_search_targets_and_ppo_distills(learner_ckpt):
    net, path = learner_ckpt
    cfg = SearchConfig(budget=4, max_depth=2, max_evals=30)
    specs = [GameSpec(5, (LEARNER, LEARNER)), GameSpec(6, (LEARNER, BOT))]
    res = run_job(Job(specs, path, 1, max_turns=12, search=cfg))
    n = len(res.actions)
    assert n and len(res.target_len) == n
    searched = [i for i, k in enumerate(res.target_len) if k]
    assert searched, "no eligible decision was searched in 12 turns"
    assert sum(res.target_len) == len(res.targets)
    pos = 0
    for i, k in enumerate(res.target_len):
        if k:
            assert k == len(res.samples[i][1])
            assert math.isclose(sum(res.targets[pos : pos + k]), 1.0, abs_tol=1e-5)
            pos += k
    # the recorded log-prob is the behaviour policy's, for searched actions too
    from mtg_ml.rl.model import collate

    logits, _, _ = net(collate(res.samples), lengths=res.lengths)
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(res.actions)[:, None]).squeeze(1)
    assert torch.allclose(lp, torch.tensor(res.logps), atol=1e-4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = ppo_update(net, opt, res, PPOConfig(epochs=2, minibatch=64, target_kl=None))
    assert stats["searched"] == len(searched)
    assert math.isfinite(stats["distill_loss"]) and stats["distill_loss"] > 0


def test_search_rejects_server_inference(learner_ckpt):
    _, path = learner_ckpt
    with pytest.raises(ValueError, match="local"):
        run_job(Job([GameSpec(1, (LEARNER, LEARNER))], path, 1, inference="server", search=SearchConfig(budget=2)))
