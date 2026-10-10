"""Multi-deck interference diagnostics for a PPO checkpoint (no training).

    python tools/diag_interference.py --checkpoint policy.pt --games-per-chunk 500 \
        --workers 4 --out diag.json --md diag.md

Question: when one network plays six decks, do the decks' PPO gradients pull in
different directions (interference / negative transfer), or does one deck simply
dominate the batch (scale mismatch)? Two measurements, both at the checkpoint's
own weights, no optimizer step:

1. Gradient conflict. For each deck the learner plays that deck in `--chunks`
   independent chunks of `--games-per-chunk` games: the deck's normal pairings
   (every matchup of the r7 training mix that seats it, mirror at half weight),
   the other seat played by the same checkpoint (self-play; only the learner's
   seat is recorded, so every decision is the deck's), sampled play, 20% of the
   games postboard. The PPO loss gradient of each chunk is the one `ppo_update`
   computes at the first minibatch step (policy + value + entropy with
   `PPOConfig`'s coefficients), summed over minibatches with their row counts so
   it is the full-batch mean gradient. Chunks are paired into two halves in every
   balanced way (3 ways for 4 chunks). Per parameter group:
     within(A)    cosine of two halves of deck A: the noise ceiling (for
                  independent noise it equals |s|^2 / (|s|^2 + |n|^2) of a half)
     cross(A, B)  mean cosine of a half of A and a half of B
     corrected    cross / sqrt(within(A) * within(B)): the cosine of the noise-free
                  gradients under independent noise (can leave [-1, 1] by chance)
   A cross cosine is only meaningful next to that ceiling: a cross cosine below
   the ceiling is not yet a conflict.

2. Scale mismatch. Per deck: raw advantages, returns (= the value targets: the
   value loss regresses on them), the clamped value predictions, decisions per
   game, turns, the deck's share of a training batch, the gradient norm. The
   trainer normalises advantages once per batch over every deck; `--norm global`
   reproduces that (each deck's advantages are transformed with the statistics of
   the six-deck training mix, weighted by batch share), `per_deck` is what a
   specialist's batch sees. Both gradients are reported by default.

The loss is not duplicated: gradients come from `ppo.ppo_update` itself
(eager, one epoch, clipping off) with a stand-in optimizer that adds up the
gradient of each step instead of applying it; `ppo._losses` is wrapped only to
read the row count of the step and, for `--norm global`, to apply the affine
advantage change."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from mtg_ml.backend import ENV_VAR, engine_name  # noqa: E402
from mtg_ml.match import MATCHUPS  # noqa: E402
from mtg_ml.rl import ppo  # noqa: E402
from mtg_ml.rl.model import PolicyNet  # noqa: E402
from mtg_ml.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, Result, create_pool, play  # noqa: E402

DECKS = ("jund_wildfire", "mono_blue_terror", "red_madness", "grixis_affinity", "elves", "tron")
SHORT = {"jund_wildfire": "jund", "mono_blue_terror": "blue", "red_madness": "red", "grixis_affinity": "affinity", "elves": "elves", "tron": "tron"}
POSTBOARD_FRAC = 0.2  # the r7 runs' --postboard-frac
MIRROR_WEIGHT = 1.0  # r7 mix: every cross-deck matchup 2, every mirror 1
CROSS_WEIGHT = 2.0
# A group's parameters by name prefix, first match wins. `trunk` is the encoder
# and trunk without the hashed tables; the tables are their own group because a
# deck touches few of their rows.
GROUPS = (
    ("emb_tables", ("policy_core.state.emb.", "policy_core.event_emb.", "option_emb.", "policy_core.state_emb.", "value_core.state.emb.", "value_core.event_emb.")),
    ("gru", ("policy_core.gru.", "policy_core.mix.", "value_core.gru.", "value_core.mix.")),
    ("trunk", ("policy_core.", "value_core.")),
    ("option_scorer", ("option_mlp.", "pointer.", "scorer.")),
    ("value_head", ("value_head.",)),
    ("belief", ("belief_net.", "belief_proj.")),
)
GROUP_NAMES = tuple(g for g, _ in GROUPS)
AGGREGATES = {"embedding+trunk": ("emb_tables", "trunk")}  # the shared representation, as one number


# -- the games -----------------------------------------------------------------


def pairings(deck: str) -> list[tuple[str, int, float]]:
    """(matchup, the learner's canonical position, weight) of every matchup of
    the training mix that seats `deck`."""
    out = []
    for name, decks in MATCHUPS.items():
        if deck in decks:
            mirror = decks[0] == decks[1]
            out.append((name, decks.index(deck), MIRROR_WEIGHT if mirror else CROSS_WEIGHT))
    return out


def make_specs(deck: str, checkpoint: str, games: int, seed: int, rng: random.Random) -> list[GameSpec]:
    """`games` games of the learner on `deck` against the checkpoint itself,
    matchups by weight, seat swap and the sideboard draw as in `Trainer._train_specs`."""
    table = pairings(deck)
    specs = []
    for k in range(games):
        matchup, pos, _ = rng.choices(table, [w for _, _, w in table])[0]
        seats = (LEARNER, checkpoint) if pos == 0 else (checkpoint, LEARNER)
        game_no = 2 if rng.random() < POSTBOARD_FRAC else 1
        specs.append(GameSpec(seed=seed + k, seats=seats, match_game=game_no, matchup=matchup, swap_seats=rng.random() < 0.5))
    return specs


def collect(pool, workers: int, specs: list[GameSpec], checkpoint: str, engine: str | None, max_turns: int) -> Result:
    job = Job([], checkpoint, 1, record=True, shaping=0.0, max_turns=max_turns, engine=engine)  # shaping is annealed to 0 long before r7 ends
    return play(pool, specs, job, workers)


# -- gradients -----------------------------------------------------------------


def group_of(name: str) -> str:
    for group, prefixes in GROUPS:
        if name.startswith(prefixes):
            return group
    raise ValueError(f"parameter {name!r} is in no group")


def group_slices(net: PolicyNet) -> dict[str, list[tuple[int, int]]]:
    """Per group, the (start, end) of its parameters in the flat gradient vector."""
    out, at = {g: [] for g in GROUP_NAMES}, 0
    for name, p in net.named_parameters():
        out[group_of(name)].append((at, at + p.numel()))
        at += p.numel()
    return {g: s for g, s in out.items() if s}


class _GradSum:
    """Stands in for the optimizer of `ppo_update` (eager mode): each step's
    gradient is added to `total` with the step's row count as weight instead of
    being applied, so `total / rows` is the mean gradient over every row."""

    param_groups = [{}]

    def __init__(self, net: PolicyNet):
        self.net = net
        self.total = torch.zeros(sum(p.numel() for p in net.parameters()), dtype=torch.float64)
        self.rows = 0
        self.pending = 0  # rows of the step being taken, set by `_losses`' wrapper

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.net.parameters():
            p.grad = None

    def step(self) -> None:
        at = 0
        for p in self.net.parameters():
            n = p.numel()
            if p.grad is not None:
                self.total[at : at + n] += p.grad.reshape(-1).double() * self.pending
            at += n
        self.rows += self.pending


@contextmanager
def _wrapped_losses(acc: _GradSum, affine: tuple[float, float] | None):
    """`ppo._losses` that tells `acc` the row count and, with `affine` = (scale,
    shift), replaces the batch-normalised advantages by ad * scale + shift."""
    real = ppo._losses

    def losses(net, cfg, b, lengths, width, a, olp, ad, rt, *rest, **kw):
        acc.pending = len(a)
        if affine is not None:
            ad = ad * affine[0] + affine[1]
        return real(net, cfg, b, lengths, width, a, olp, ad, rt, *rest, **kw)

    ppo._losses = losses
    try:
        yield
    finally:
        ppo._losses = real


def advantage_moments(res: Result) -> tuple[float, float]:
    """The mean and std `ppo_update` normalises this data by (float32, unbiased)."""
    adv = torch.tensor(res.advantages, dtype=torch.float32)
    return adv.mean().item(), adv.std().item() if len(adv) > 1 else 0.0


def global_affine(mean: float, std: float, g_mean: float, g_std: float) -> tuple[float, float]:
    """(scale, shift) taking `ppo_update`'s own normalisation, (a - mean) / (std + 1e-8),
    to the one of the whole training batch, (a - g_mean) / (g_std + 1e-8)."""
    return (std + 1e-8) / (g_std + 1e-8), (mean - g_mean) / (g_std + 1e-8)


def make_config(**overrides) -> PPOConfig:
    """The trainer's coefficients; one epoch, no gradient clipping, no early stop, FP32."""
    return replace(PPOConfig(**overrides), epochs=1, max_grad_norm=float("inf"), target_kl=None, precision="fp32")


def chunk_gradient(net: PolicyNet, res: Result, cfg: PPOConfig, affine: tuple[float, float] | None = None, seed: int = 0) -> tuple[torch.Tensor, int, dict]:
    """(sum of the per-row gradients over the batch as float64, rows, ppo_update's statistics).
    Divide by rows for the mean. The minibatches are shuffled with `seed`
    (irrelevant for the sum: every row is in exactly one minibatch)."""
    acc = _GradSum(net)
    gen = torch.Generator().manual_seed(seed)
    with _wrapped_losses(acc, affine):
        stats = ppo_update(net, acc, res, cfg, mode="eager", gen=gen)
    acc.zero_grad()
    return acc.total, acc.rows, stats


def group_vector(vec: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([vec[s:e] for s, e in slices])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    return float("nan") if na == 0 or nb == 0 else (a @ b).item() / (na * nb)


def balanced_splits(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every way to put n chunks (n even) into two halves, each once."""
    idx = tuple(range(n))
    return [(h, tuple(i for i in idx if i not in h)) for h in itertools.combinations(idx, n // 2) if 0 in h]


def mean_std(xs) -> tuple[float, float]:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def pieces(chunks: list[tuple[torch.Tensor, int]], level: str) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    """The mean gradients to compare, and the index pairs that are two independent
    samples of the same deck. level "half": both halves of every balanced split of
    the chunks (pieces 2s, 2s + 1 are split s); "chunk": the single chunks, every pair."""
    n = len(chunks)
    if level == "half":
        mean = lambda ids: sum(chunks[i][0] for i in ids) / sum(chunks[i][1] for i in ids)  # noqa: E731
        vecs = [mean(h) for split in balanced_splits(n) for h in split]
        return vecs, [(2 * s, 2 * s + 1) for s in range(len(vecs) // 2)]
    return [c[0] / c[1] for c in chunks], list(itertools.combinations(range(n), 2))


def cosine_tables(grads: dict[str, list[tuple[torch.Tensor, int]]], slices: dict[str, list[tuple[int, int]]], level: str = "half") -> dict:
    """{group: {"within": {deck: (mean, sd)}, "cross": {(a, b): (mean, sd)}, "corrected": {(a, b): (mean, sd)}}}
    from per-deck chunk gradients (`pieces`). within: the cosine of two
    independent samples of one deck, over the splits (or chunk pairs); cross: the
    cosine of a piece of A and a piece of B, over all piece pairs; corrected:
    cross / sqrt(within(A) * within(B)) (per cross entry). The sd is the spread
    over those splits and pairs, not a confidence interval: they share chunks."""
    decks = list(grads)
    groups = dict(slices)
    for agg, members in AGGREGATES.items():
        if all(m in slices for m in members):
            groups[agg] = [s for m in members for s in slices[m]]
    out = {}
    for group, sl in groups.items():
        unit, idx = {}, {}
        for d in decks:
            vecs, idx[d] = pieces(grads[d], level)
            m = torch.stack([group_vector(v, sl) for v in vecs])
            unit[d] = m / m.norm(dim=1, keepdim=True).clamp(min=1e-30)
        within = {}
        for d in decks:
            gram = unit[d] @ unit[d].T
            within[d] = mean_std([gram[i, j].item() for i, j in idx[d]])
        cross, corrected = {}, {}
        for a, b in itertools.combinations(decks, 2):
            c = (unit[a] @ unit[b].T).flatten().tolist()
            cross[(a, b)] = mean_std(c)
            ra, rb = within[a][0], within[b][0]
            corrected[(a, b)] = mean_std([x / math.sqrt(ra * rb) for x in c]) if ra > 0 and rb > 0 else (float("nan"), float("nan"))
        out[group] = {"within": within, "cross": cross, "corrected": corrected}
    return out


def deck_mean(chunks: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
    return sum(c[0] for c in chunks) / sum(c[1] for c in chunks)


def norms(grads: dict[str, list[tuple[torch.Tensor, int]]], slices: dict[str, list[tuple[int, int]]]) -> dict:
    """{deck: {group or "whole": norm of the deck's mean gradient (all chunks)}}."""
    out = {}
    for d, chunks in grads.items():
        g = deck_mean(chunks)
        out[d] = {"whole": g.norm().item(), **{grp: group_vector(g, sl).norm().item() for grp, sl in slices.items()}}
    return out


def dominance(grads: dict[str, list[tuple[torch.Tensor, int]]], shares: dict[str, float], slices: dict[str, list[tuple[int, int]]]) -> dict:
    """The training batch's mean gradient is G = sum_d share_d * g_d (g_d: the deck's mean
    gradient). Per deck: the norm contribution share_d * |g_d| as a fraction of the sum,
    its part of |G|^2, share_d * <g_d, G> / |G|^2 (sums to 1; negative when the deck's
    gradient points against the batch's), and cos(g_d, G)."""
    mean = {d: deck_mean(ch) for d, ch in grads.items()}
    out = {}
    for name, sl in (("whole", None), *slices.items()):
        vec = {d: (g if sl is None else group_vector(g, sl)) for d, g in mean.items()}
        G = sum(shares[d] * vec[d] for d in vec)
        tot = sum(shares[d] * vec[d].norm().item() for d in vec)
        g2 = (G @ G).item()
        out[name] = {d: {"norm_share": shares[d] * vec[d].norm().item() / tot if tot else float("nan"),
                         "g2_share": shares[d] * (vec[d] @ G).item() / g2 if g2 else float("nan"),
                         "cos_with_batch": cosine(vec[d], G)} for d in vec}
    return out


# -- scale statistics ----------------------------------------------------------


def moments(xs) -> tuple[float, float]:
    t = xs.double() if torch.is_tensor(xs) else torch.tensor(xs, dtype=torch.float64)
    return (t.mean().item(), t.std().item() if len(t) > 1 else 0.0) if len(t) else (float("nan"), float("nan"))


def deck_stats(results: list[Result], specs: list[list[GameSpec]]) -> dict:
    """Scale statistics of one deck's chunks, pooled. `by_matchup`: the recorded
    decisions per trajectory of each matchup it played."""
    matchup_of = {s.seed: s.matchup for sp in specs for s in sp}
    adv = torch.tensor([a for r in results for a in r.advantages], dtype=torch.float64)
    ret = torch.tensor([a for r in results for a in r.returns], dtype=torch.float64)
    val = ret - adv  # the (clamped) value predictions GAE used
    lengths = [n for r in results for n in r.lengths]
    games = [g for r in results for g in r.games]
    by = {}
    for seed, n in zip([t[0] for r in results for t in r.trajectory_ids], lengths):
        by.setdefault(matchup_of[seed], []).append(n)
    wins = []
    for seats, winner, _reason, _turns, _dec, _seed in games:
        pos = 0 if seats[0] == LEARNER else 1
        wins.append(1.0 if winner == pos else 0.5 if winner is None else 0.0)
    return {
        "games": len(games), "decisions": int(sum(lengths)),
        "adv_mean_sd": moments(adv), "ret_mean_sd": moments(ret), "value_mean_sd": moments(val), "adv_abs_mean": adv.abs().mean().item(),
        "explained_var": 1 - adv.var().item() / ret.var().item() if ret.var() > 0 else float("nan"),
        "decisions_per_game": sum(lengths) / len(games), "turns_mean_sd": moments([g[3] for g in games]), "win_rate": sum(wins) / len(wins),
        "by_matchup": {m: {"trajectories": len(ns), "decisions_per_trajectory": sum(ns) / len(ns)} for m, ns in by.items()},
    }


def batch_shares(stats: dict[str, dict]) -> dict:
    """Each deck's share of a training batch of the r7 mix (every cross-deck matchup
    weight 2, each mirror 1, a mirror seats the deck twice): of the recorded
    trajectories and of the recorded decisions. A matchup's decisions per
    trajectory come from the chunks where that deck played it."""
    traj, dec = {d: 0.0 for d in stats}, {d: 0.0 for d in stats}
    for matchup, decks in MATCHUPS.items():
        w = MIRROR_WEIGHT if decks[0] == decks[1] else CROSS_WEIGHT
        for d in set(decks):
            per = stats[d]["by_matchup"].get(matchup, {}).get("decisions_per_trajectory")
            if per is not None:
                traj[d] += w * decks.count(d)
                dec[d] += w * decks.count(d) * per
    ts, ds = sum(traj.values()), sum(dec.values())
    return {"trajectories": {d: v / ts for d, v in traj.items()}, "decisions": {d: v / ds for d, v in dec.items()}}


def mixture_moments(stats: dict[str, dict], shares: dict[str, float]) -> tuple[float, float]:
    """Mean and std of the advantages of a batch that mixes the decks with `shares`."""
    mean = sum(shares[d] * stats[d]["adv_mean_sd"][0] for d in stats)
    var = sum(shares[d] * (stats[d]["adv_mean_sd"][1] ** 2 + (stats[d]["adv_mean_sd"][0] - mean) ** 2) for d in stats)
    return mean, math.sqrt(var)


# -- the run -------------------------------------------------------------------


def load_checkpoint(path: str) -> PolicyNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyNet(**ck["config"])
    net.load_state_dict(ck["model"])
    return net


def run(net: PolicyNet, checkpoint: str, pool, workers: int, decks, chunks: int, games: int, norm: tuple[str, ...], engine: str | None = None,
        max_turns: int = 100, seed: int = 0, cache: str | None = None, minibatch: int = 2048, threads: int = 1, log=print) -> dict:
    """Collect, then measure; returns the tables as one JSON-able dict."""
    cfg = replace(make_config(), minibatch=minibatch)
    results, specs = {d: [] for d in decks}, {d: [] for d in decks}
    t0 = time.time()
    for di, d in enumerate(decks):
        for c in range(chunks):
            base = 50_000_000 + (di * 100 + c) * 100_000 + seed * 7919
            sp = make_specs(d, checkpoint, games, base, random.Random(base))
            path = os.path.join(cache, f"{d}-{c}-{games}-{seed}.pkl") if cache else None
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    res = pickle.load(f)
            else:
                res = collect(pool, workers, sp, checkpoint, engine, max_turns)
                if path:
                    os.makedirs(cache, exist_ok=True)
                    with open(path, "wb") as f:
                        pickle.dump(res, f)
            specs[d].append(sp)
            results[d].append(res)
            log(f"collected {d} chunk {c}: {len(res.games)} games, {len(res.actions)} decisions ({time.time() - t0:.0f}s)")
    stats = {d: deck_stats(results[d], specs[d]) for d in decks}
    shares = batch_shares(stats) if len(decks) == len(DECKS) else {"decisions": {d: 1 / len(decks) for d in decks}, "trajectories": {d: 1 / len(decks) for d in decks}}
    g_mean, g_std = mixture_moments(stats, shares["decisions"])
    slices = group_slices(net)
    torch.set_num_threads(threads)
    out = {"decks": list(decks), "stats": stats, "batch_share": shares, "global_advantage": {"mean": g_mean, "std": g_std},
           "chunks": chunks, "games_per_chunk": games, "cosines": {}, "norms": {}, "dominance": {}}
    ppo_stats = {d: [] for d in decks}
    for mode in norm:
        grads = {d: [] for d in decks}
        for d in decks:
            for c, res in enumerate(results[d]):
                cm, cs = advantage_moments(res)  # `ppo_update` normalises by these; the affine goes from there to the mix's
                total, rows, st = chunk_gradient(net, res, cfg, global_affine(cm, cs, g_mean, g_std) if mode == "global" else None, seed=seed + c)
                grads[d].append((total.float(), rows))
                ppo_stats[d].append(st)
            log(f"gradients ({mode}) {d}: done ({time.time() - t0:.0f}s)")
        out["cosines"][mode] = {lvl: _jsonable(cosine_tables(grads, slices, lvl)) for lvl in ("half", "chunk")}
        out["norms"][mode] = norms(grads, slices)
        out["dominance"][mode] = dominance(grads, shares["decisions"], slices)
        del grads
    for d in decks:  # the loss values come from the last mode's pass
        ps = ppo_stats[d]
        stats[d]["ppo"] = {k: sum(s[k] for s in ps) / len(ps) for k in ("pg_loss", "v_loss", "entropy", "nt_entropy", "nt_frac", "explained_var") if k in ps[0]}
    return out


def _jsonable(tables: dict) -> dict:
    return {g: {"within": t["within"], "cross": {f"{a}|{b}": v for (a, b), v in t["cross"].items()}, "corrected": {f"{a}|{b}": v for (a, b), v in t["corrected"].items()}} for g, t in tables.items()}


# -- the report ----------------------------------------------------------------


def _f(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def markdown(out: dict, level: str = "half") -> str:
    decks = out["decks"]
    lines = []
    for mode, by_level in out["cosines"].items():
        lines.append(f"### Gradient cosines, `{mode}` advantage normalisation, {level} level\n")
        for group, t in by_level[level].items():
            lines.append(f"**{group}**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).\n")
            lines.append("| | " + " | ".join(SHORT[d] for d in decks) + " |")
            lines.append("|---|" + "---|" * len(decks))
            for i, a in enumerate(decks):
                row = []
                for j, b in enumerate(decks):
                    if i == j:
                        m, sd = t["within"][a]
                        row.append(f"**{_f(m)}** ±{_f(sd, 2)}")
                    elif i < j:
                        row.append(_f(t["cross"][f"{a}|{b}"][0]))
                    else:
                        row.append(_f(t["corrected"][f"{b}|{a}"][0], 2))
                lines.append(f"| {SHORT[a]} | " + " | ".join(row) + " |")
            lines.append("")
    return "\n".join(lines)


def scale_markdown(out: dict) -> str:
    decks = out["decks"]
    st, sh = out["stats"], out["batch_share"]
    lines = ["| deck | games | dec/game | turns/game | batch share (dec.) | batch share (traj.) | adv mean ± sd | return mean ± sd | value mean ± sd | v_loss | expl. var | win rate |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in decks:
        s = st[d]
        ms = lambda k: f"{s[k][0]:+.3f} ± {s[k][1]:.3f}"  # noqa: E731
        lines.append(f"| {SHORT[d]} | {s['games']} | {s['decisions_per_game']:.0f} | {s['turns_mean_sd'][0]:.1f} ± {s['turns_mean_sd'][1]:.1f} | {sh['decisions'][d]:.3f} | {sh['trajectories'][d]:.3f} | {ms('adv_mean_sd')} | {ms('ret_mean_sd')} | {ms('value_mean_sd')} | {_f(s.get('ppo', {}).get('v_loss'), 4)} | {_f(s['explained_var'], 2)} | {s['win_rate']:.3f} |")
    ga = out["global_advantage"]
    lines += ["", f"Training-batch advantage statistics (decks mixed by decision share): mean {ga['mean']:+.4f}, sd {ga['std']:.4f}.", ""]
    for mode, per in out["norms"].items():
        groups = list(per[decks[0]])
        lines += [f"Norm of each deck's mean loss gradient, `{mode}` normalisation:", "", "| deck | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        lines += [f"| {SHORT[d]} | " + " | ".join(f"{per[d][g]:.4g}" for g in groups) + " |" for d in decks]
        dom = out["dominance"][mode]["whole"]
        lines += ["", f"Who drives the batch gradient, `{mode}` (G = sum of share x deck gradient): norm share = share x |g| / sum, G2 share = share x <g, G> / |G|^2.", "",
                  "| deck | batch share | norm share | G2 share | cos(g_deck, G) |", "|---|---|---|---|---|"]
        lines += [f"| {SHORT[d]} | {sh['decisions'][d]:.3f} | {dom[d]['norm_share']:.3f} | {dom[d]['g2_share']:.3f} | {dom[d]['cos_with_batch']:.3f} |" for d in decks]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decks", default=",".join(DECKS))
    ap.add_argument("--chunks", type=int, default=4, help="independent chunks per deck (even; two halves are formed from them)")
    ap.add_argument("--games-per-chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="torch threads of the gradient phase")
    ap.add_argument("--engine", default="native")
    ap.add_argument("--norm", default="global,per_deck", help="advantage normalisation(s): global (the trainer's: one per batch of all decks), per_deck")
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="", help="directory for the collected rollouts (reused when the same chunk exists)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    args = ap.parse_args(argv)
    if args.chunks % 2 or args.chunks < 2:
        ap.error("--chunks must be even and at least 2")
    decks = tuple(args.decks.split(","))
    for d in decks:
        if d not in DECKS:
            ap.error(f"unknown deck {d!r}: {DECKS}")
    os.environ[ENV_VAR] = args.engine
    net = load_checkpoint(args.checkpoint)
    net.train()
    pool = create_pool(args.workers)
    t0 = time.time()
    torch.set_num_threads(1)  # rollouts first (the workers hold the cores), then the gradients on `--threads`
    out = run(net, os.path.abspath(args.checkpoint), pool, args.workers, decks, args.chunks, args.games_per_chunk, tuple(args.norm.split(",")), args.engine, args.max_turns, args.seed,
              args.cache or None, args.minibatch, args.threads)
    pool.close()
    cfg = make_config()
    out["meta"] = {"checkpoint": os.path.abspath(args.checkpoint), "engine": engine_name(), "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                   "threads": args.threads, "seconds": time.time() - t0, "config": net.config, "minibatch": args.minibatch,
                   "ppo": {"clip": cfg.clip, "vf_coef": cfg.vf_coef, "ent_coef": cfg.ent_coef}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    md = scale_markdown(out) + "\n\n" + markdown(out, "half")
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
