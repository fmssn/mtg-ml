"""tools/diag_interference.py: the gradient it sums is the full-batch PPO gradient,
the global advantage transform is exact, the cosine tables read noise ceilings
right, and a tiny end-to-end run works."""

import math
import os
import random
import sys
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import diag_interference as D  # noqa: E402

from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.rollout import LEARNER  # noqa: E402

ENGINE = "python"  # the reference engine: the tool is engine-agnostic and a few games are cheap


class InProcess:
    """A pool stand-in: `play` falls back to `map`."""

    map = staticmethod(map)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    net = PolicyNet(hidden=16)
    path = str(tmp_path_factory.mktemp("diag") / "net.pt")
    torch.save({"config": net.config, "model": net.state_dict()}, path)
    return net, path


@pytest.fixture(scope="module")
def data(ckpt):
    _, path = ckpt
    specs = D.make_specs("mono_blue_terror", path, 8, 1000, random.Random(1))
    return D.collect(InProcess(), 1, specs, path, ENGINE, 12)


def test_pairings_cover_the_training_mix():
    for deck in D.DECKS:
        table = D.pairings(deck)
        assert len(table) == 6  # five opponents and the mirror
        assert sum(w for _, _, w in table) == 11
        for matchup, pos, w in table:
            decks = D.MATCHUPS[matchup]
            assert decks[pos] == deck and (w == D.MIRROR_WEIGHT) == (decks[0] == decks[1])


def test_specs_record_only_the_learner_on_its_deck(ckpt):
    _, path = ckpt
    for s in D.make_specs("tron", path, 30, 0, random.Random(0)):
        pos = 0 if s.seats[0] == LEARNER else 1
        assert s.seats[1 - pos] == path and D.MATCHUPS[s.matchup][pos] == "tron"


def test_groups_cover_every_parameter():
    for kw in ({}, {"trunk": "entity", "entity_attn": 1, "features": 7}, {"value_net": "separate", "memory": "none"}):
        net = PolicyNet(hidden=16, **kw)
        sl = D.group_slices(net)
        assert sum(e - s for ranges in sl.values() for s, e in ranges) == sum(p.numel() for p in net.parameters())
        assert "value_head" in sl and "gru" in sl or kw.get("memory") == "none"


def test_summed_gradient_is_the_full_batch_gradient(ckpt, data):
    net, _ = ckpt
    cfg = D.make_config()
    many, rows_many, _ = D.chunk_gradient(net, data, replace(cfg, minibatch=40), seed=3)
    assert len(data.lengths) > 2 and rows_many == len(data.actions)
    one, rows_one, _ = D.chunk_gradient(net, data, replace(cfg, minibatch=10**9))
    assert rows_one == rows_many
    a, b = many / rows_many, one / rows_one
    assert a.norm() > 0
    assert (a - b).norm() < 1e-3 * a.norm()
    assert all(p.grad is None for p in net.parameters())  # nothing left behind, no weights moved


def test_global_affine_is_exact(ckpt, data):
    net, _ = ckpt
    cfg = D.make_config()
    mean, std = D.advantage_moments(data)
    plain, rows, _ = D.chunk_gradient(net, data, cfg)
    same, _, _ = D.chunk_gradient(net, data, cfg, D.global_affine(mean, std, mean, std))
    assert (plain - same).norm() < 1e-4 * plain.norm()
    # the transform takes the batch-normalised advantages to (a - g_mean) / g_std
    scale, shift = D.global_affine(mean, std, mean + 0.1, 2 * std)
    adv = torch.tensor(data.advantages, dtype=torch.float32)
    local = (adv - mean) / (std + 1e-8)
    assert torch.allclose(local * scale + shift, (adv - mean - 0.1) / (2 * std + 1e-8), atol=1e-5)


def test_cosine_tables_noise_ceiling():
    g = torch.Generator().manual_seed(0)
    dim = 400
    slices = {"gru": [(0, 200)], "value_head": [(200, 400)]}
    s_a, s_b = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
    s_b2 = s_a + 0.0
    grads = {}
    for name, sig in (("a", s_a), ("b", s_b), ("c", s_b2)):  # a, c: the same signal; b: an independent one
        grads[name] = [((sig + 2 * torch.randn(dim, generator=g)) * 10, 10) for _ in range(4)]
    for level in ("half", "chunk"):
        t = D.cosine_tables(grads, slices, level)["gru"]
        ceiling = t["within"]["a"][0]
        assert 0.1 < ceiling < 0.9
        assert abs(t["cross"][("a", "b")][0]) < 0.2  # independent signals
        assert t["cross"][("a", "c")][0] == pytest.approx(ceiling, abs=0.15)  # same signal: cross equals the ceiling
        assert t["corrected"][("a", "c")][0] == pytest.approx(1.0, abs=0.2)
        assert abs(t["corrected"][("a", "b")][0]) < 0.4


def test_end_to_end(ckpt, tmp_path):
    net, path = ckpt
    decks = ("jund_wildfire", "mono_blue_terror")
    out = D.run(net, path, InProcess(), 1, decks, 2, 6, ("global", "per_deck"), ENGINE, 10, cache=str(tmp_path), minibatch=64, log=lambda *_: None)
    assert os.listdir(tmp_path)
    for d in decks:
        s = out["stats"][d]
        assert s["games"] == 12 and s["decisions"] > 0 and s["ppo"]["v_loss"] >= 0
    assert sum(out["batch_share"]["decisions"].values()) == pytest.approx(1)
    for mode in ("global", "per_deck"):
        t = out["cosines"][mode]["half"]["gru"]
        assert -1.0001 <= t["cross"]["jund_wildfire|mono_blue_terror"][0] <= 1.0001
        assert all(math.isfinite(out["norms"][mode][d]["whole"]) for d in decks)
        assert sum(out["dominance"][mode]["whole"][d]["g2_share"] for d in decks) == pytest.approx(1, abs=1e-3)
    md = D.scale_markdown(out) + D.markdown(out)
    assert "jund" in md and "blue" in md
    # a second run reads the cache and gets the same games
    again = D.run(net, path, InProcess(), 1, decks, 2, 6, ("global",), ENGINE, 10, cache=str(tmp_path), minibatch=64, log=lambda *_: None)
    assert again["stats"][decks[0]]["decisions"] == out["stats"][decks[0]]["decisions"]
