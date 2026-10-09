"""Learning smoke for the feature-set-8 belief head (docs/belief-head.md).

Plays best-of-three matches (an untrained set-8 learner against the
scripted bots, seats and starts mixed) on `engine.variants` lists, trains
only the belief branch on the recorded evidence with its supervised loss,
and reports held-out archetype log loss, accuracy and calibration (ECE,
10 bins) and per-card count error on the dev and test splits, by game of
the match, against the training split's archetype prior. It is a check
that the head learns from witnessed evidence, not a gameplay result.

    python tools/belief_smoke.py --train-matches 300 --eval-matches 120 --workers 8 --engine native --out .context/belief-smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from mtg_ml.engine.variants import sample_variant, sampling_prior  # noqa: E402
from mtg_ml.match import EXPLICIT_ONLY, MATCHUPS, matchup_decks  # noqa: E402
from mtg_ml.rl.belief import BeliefSpec  # noqa: E402
from mtg_ml.rl.model import PolicyNet, collate_packed  # noqa: E402
from mtg_ml.rl.ppo import belief_losses  # noqa: E402
from mtg_ml.rl.rollout import BOT, LEARNER, GameSpec, Job, create_pool, play  # noqa: E402


def specs(n: int, split: str, seed: int) -> list[GameSpec]:
    rng = random.Random(seed)
    prior = sampling_prior(split)  # held-out splits lack some archetypes: only matchups both of whose decks have one
    names = [m for m in MATCHUPS if m not in EXPLICIT_ONLY and all(d in prior for d in matchup_decks(m))]
    out = []
    for k in range(n):
        m = rng.choice(names)
        variants = tuple(sample_variant(d, rng, split).id for d in matchup_decks(m))
        seats = (LEARNER, BOT) if rng.random() < 0.5 else (BOT, LEARNER)
        out.append(GameSpec(seed + k, seats, matchup=m, bo3=True, variants=variants, swap_seats=rng.random() < 0.5))
    return out


def rows(res):
    """Per recorded decision: (archetype target, counts target, game of the match)."""
    arch, counts, game = [], [], []
    for (a, c), n in zip(res.belief_targets, res.lengths):
        arch += [a] * n
        counts += [list(c)] * n
    for i in range(len(res.samples)):
        ev = res.samples.evidence(i)
        game.append(1 + max((ev[j + 1] for j in range(0, len(ev), 3)), default=0))
    return torch.tensor(arch), torch.tensor(counts), torch.tensor(game)


def predict(net, res, idx=None):
    b = collate_packed(res.samples, idx, evidence=True)
    rows_n = int(b.s_off.shape[0])
    with torch.no_grad():
        return net.belief_net(b, rows_n)


def report(net, res, prior):
    arch, counts, game = rows(res)
    out = predict(net, res)
    ce_a, ce_c, st = belief_losses(net, out, arch, counts, lambda x: x.mean())
    probs = torch.softmax(out.arch.float(), -1)
    conf, pred = probs.max(-1)
    correct = (pred == arch).float()
    ece = 0.0
    for lo in [i / 10 for i in range(10)]:
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            ece += float(m.float().mean()) * abs(float(conf[m].mean()) - float(correct[m].mean()))
    prior_ll = float(-torch.log(prior[arch]).mean())
    by_game = {}
    for g in sorted(set(game.tolist())):
        m = game == g
        by_game[f"game{g}"] = {"decisions": int(m.sum()), "arch_logloss": round(float(-torch.log(probs[m].gather(1, arch[m, None])).mean()), 4),
                               "arch_acc": round(float(correct[m].mean()), 4)}
    return {"decisions": len(arch), "arch_logloss": round(float(ce_a), 4), "prior_logloss": round(prior_ll, 4), "arch_acc": round(float(correct.mean()), 4),
            "ece": round(ece, 4), "count_ce_per_card": round(float(ce_c), 4), "count_mae_per_card": round(float(st[4]), 4), "by_game": by_game}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-matches", type=int, default=200)
    ap.add_argument("--eval-matches", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", default="python")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    net = PolicyNet(hidden=a.hidden, features=8, belief=BeliefSpec.default())
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.pt")
        torch.save({"config": net.config, "model": net.state_dict()}, path)
        pool = create_pool(a.workers)
        try:
            job = Job([], path, 1, record=True, max_turns=a.max_turns, engine=a.engine)
            data = {s: play(pool, specs(n, s, 1_000_000 * (i + 1) + a.seed), job, a.workers)
                    for i, (s, n) in enumerate((("train", a.train_matches), ("dev", a.eval_matches), ("test", a.eval_matches)))}
        finally:
            pool.close()
            pool.join()
    t_play = time.time() - t0
    arch_tr, counts_tr, _ = rows(data["train"])
    prior = torch.bincount(arch_tr, minlength=len(net.belief.archetypes)).float() + 1
    prior /= prior.sum()
    before = {s: report(net, data[s], prior) for s in ("dev", "test")}
    opt = torch.optim.Adam(net.belief_net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(a.seed)
    n = len(arch_tr)
    losses = []
    for step in range(a.steps):
        idx = torch.randint(n, (min(a.batch, n),), generator=gen).sort().values
        out = net.belief_net(collate_packed(data["train"].samples, idx, evidence=True), len(idx))
        la, lc, _ = belief_losses(net, out, arch_tr[idx], counts_tr[idx], lambda x: x.mean())
        loss = la + lc
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    # distinguishing evidence moves the prediction
    probe = {}
    spec_b = net.belief
    from mtg_ml.knowledge import EMPTY
    from mtg_ml.rl.samples import PackedSamples

    for name, seen in (("nothing", {}), ("Lightning Bolt x2, Mountain x3", {"Lightning Bolt": 2, "Mountain": 3}),
                       ("Island x3, Counterspell", {"Island": 3, "Counterspell": 1}), ("Urza's Tower, Expedition Map", {"Urza's Tower": 1, "Expedition Map": 1})):
        ps = PackedSamples()
        ps.append(([], [1], [0], [], spec_b.evidence(EMPTY, seen)))
        with torch.no_grad():
            p = torch.softmax(net.belief_net(collate_packed(ps, evidence=True), 1).arch[0], -1)
        probe[name] = {k: round(float(v), 3) for k, v in zip(spec_b.archetypes, p)}
    after = {s: report(net, data[s], prior) for s in ("dev", "test")}
    res = {"machine": os.uname().nodename, "engine": a.engine, "args": vars(a), "play_s": round(t_play, 1),
           "matches": {s: len(data[s].matches) for s in data}, "games": {s: len(data[s].games) for s in data},
           "train_decisions": n, "loss_first_50": round(sum(losses[:50]) / 50, 4), "loss_last_50": round(sum(losses[-50:]) / 50, 4),
           "before": before, "after": after, "probe": probe}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if not math.isfinite(res["loss_last_50"]) or res["loss_last_50"] >= res["loss_first_50"]:
        sys.exit("belief loss did not decrease")


if __name__ == "__main__":
    main()
