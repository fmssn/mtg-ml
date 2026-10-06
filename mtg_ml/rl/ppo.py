"""Clipped PPO update over recorded decisions (masked categorical policy).

Minibatches are sets of whole trajectories (about `minibatch` decisions
each), so the recurrent core is replayed from the start of every game with
the current weights instead of reusing stale hidden states.

The network is small and a minibatch is ~2k decisions, so a step costs
kernel launches, not arithmetic, and any wait for the device stalls the
whole pipeline. So the loop never waits: the decisions go to the device once
per update; each epoch lays them out in minibatch order and derives the
entity structure in one go (`model.structure`, which does wait), minibatches
are slices of that (`model.split`), and the statistics are summed on the
device and read once per epoch.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import nn

from .model import PolicyNet, _excl, _segments, collate_packed, masked_entropy, packed_tensors, split, structure
from .rollout import Result

STATS = ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "distill_loss")


@dataclass
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 2048
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03  # stop the epoch loop early past this
    # Searched decisions (rl/search.py) are trained towards the search's policy
    # (cross-entropy) instead of the policy-gradient term, whose action they did
    # not sample from the policy; they still train the value head.
    distill_coef: float = 1.0  # the distillation term is a mean over all rows, so ~0.5% searched rows weigh ~0.5% of a one-hot cross-entropy
    # Searched decisions: the value target is (1 - coef) * return + coef * the search's
    # value of the action taken (its line to the horizon). The game outcome alone leaves
    # the critic blind to what a state is worth with the right line; the searched value
    # propagates the resolved board (the sweep gone through) back to the decision.
    search_value_coef: float = 0.5


def make_optimizer(params, lr: float, device) -> torch.optim.Optimizer:
    """Adam with eps 1e-5; on CUDA fused (one kernel for all parameters
    instead of a few per tensor), elsewhere plain. `Optimizer.load_state_dict`
    restores the `fused` flag from the checkpoint's param groups, so after
    resuming callers must set `group["fused"]` again, and for a checkpoint
    of an unfused optimizer also move every `state["step"]` to its
    parameter's device (plain Adam keeps it on the CPU; fused Adam fails on that)."""
    return torch.optim.Adam(params, lr=lr, eps=1e-5, fused=torch.device(device).type == "cuda")


def load_optimizer_state(opt: torch.optim.Optimizer, state_dict: dict) -> None:
    """`opt.load_state_dict` for an optimizer from `make_optimizer`: keeps
    this optimizer's `fused` flag (the checkpoint's groups bring their own,
    None for plain Adam) and, when fused, moves every `step` to its
    parameter's device (plain Adam keeps it on the CPU, which fused Adam
    rejects). The steps are the same either way."""
    fused = opt.param_groups[0].get("fused")
    opt.load_state_dict(state_dict)
    for g in opt.param_groups:
        g["fused"] = fused
        if fused:
            for p in g["params"]:
                st = opt.state.get(p)
                if st and "step" in st:
                    st["step"] = st["step"].to(p.device, torch.float32)


def trajectory_minibatches(lengths: list[int], size: int, gen=None) -> tuple[torch.Tensor, list[list[int]]]:
    """Shuffle trajectories and group them into chunks of >= size steps.
    Returns the decision order (the shuffled trajectories laid end to end)
    and the trajectory lengths of each chunk; chunks are consecutive in that
    order."""
    L = torch.tensor(lengths, dtype=torch.long)
    perm = torch.randperm(len(lengths), generator=gen)
    Lp = L[perm]
    chunks, cur, acc = [], [], 0
    for n in Lp.tolist():
        cur.append(n)
        acc += n
        if acc >= size:
            chunks.append(cur)
            cur, acc = [], 0
    if cur:
        chunks.append(cur)
    order = torch.repeat_interleave((L.cumsum(0) - L)[perm] - (Lp.cumsum(0) - Lp), Lp) + torch.arange(int(L.sum()))
    return order, chunks


def ppo_update(net: PolicyNet, opt: torch.optim.Optimizer, data: Result, cfg: PPOConfig, device="cpu", gen=None) -> dict:
    n = len(data.actions)
    floats = lambda xs: torch.frombuffer(array("f", xs), dtype=torch.float32)  # noqa: E731 - 3x faster than torch.tensor(list)
    old_logp, adv, ret = floats(data.logps), floats(data.advantages), floats(data.returns)
    var = ret.var().item()
    explained_var = float("nan") if n < 2 or var == 0 else 1 - adv.var().item() / var  # values = ret - adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    sv = floats(getattr(data, "search_values", None) or [float("nan")] * n) if n else torch.zeros(0)
    valued = ~torch.isnan(sv)
    search_valued = int(valued.sum())
    if search_valued and cfg.search_value_coef:
        ret = torch.where(valued, (1 - cfg.search_value_coef) * ret + cfg.search_value_coef * sv.nan_to_num(), ret)
    net.train()
    dev = torch.device(device)
    samples = packed_tensors(data.samples, dev)
    n_opts = torch.frombuffer(data.samples.n_opts, dtype=torch.int32)  # on the CPU: logit widths without asking the device
    actions = torch.frombuffer(array("q", data.actions), dtype=torch.long).to(dev)
    recorded = torch.stack([old_logp, adv, ret]).to(dev)
    # distillation targets of searched decisions, kept flat (a dense n x widest-decision
    # tensor reached 25 GB: attack declarations can have thousands of options) and
    # expanded per minibatch to that minibatch's logit width
    t_len = torch.frombuffer(array("i", list(getattr(data, "target_len", ())) or [0] * n), dtype=torch.int32).long() if n else torch.zeros(0, dtype=torch.long)
    searched = int((t_len > 0).sum())
    t_vals = floats(data.targets).to(dev) if searched else torch.zeros(0, device=dev)
    t_len, t_start = t_len.to(dev), _excl(t_len.to(dev))
    pg_mask = (t_len == 0).float()

    def dense_targets(r: torch.Tensor, width: int) -> torch.Tensor:
        """(len(r), width) target rows for decisions r, zero where there is none."""
        lens = t_len[r]
        t = torch.zeros(len(r), width, device=dev)
        if int(lens.sum()):
            vals, off = _segments(t_vals, t_start[r], lens)
            rr = torch.repeat_interleave(torch.arange(len(r), device=dev), lens)
            cols = torch.arange(int(lens.sum()), device=dev) - torch.repeat_interleave(off, lens)
            t[rr, cols] = vals
        return t
    totals = [0.0] * len(STATS)
    steps = 0
    stop = False
    for _ in range(cfg.epochs):
        order, chunks = trajectory_minibatches(data.lengths, cfg.minibatch, gen)
        bounds = list(accumulate((sum(c) for c in chunks), initial=0))
        rows = order.to(dev)
        batches = split(structure(collate_packed(samples, rows), net.config["state_dim"], net.config["option_dim"]), bounds)
        acts, rec, width = actions[rows], recorded[:, rows], n_opts[order]
        pgm = pg_mask[rows]
        acc = torch.zeros(len(STATS), dtype=torch.float64, device=dev)
        for b, lens, lo, hi in zip(batches, chunks, bounds, bounds[1:]):
            a, (olp, ad, rt) = acts[lo:hi], rec[:, lo:hi]
            logits, values, _ = net(b, lengths=lens, max_options=int(width[lo:hi].max()))
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a[:, None]).squeeze(1)
            ratio = (logp - olp).exp()
            m = pgm[lo:hi]
            pg = -torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad)
            pg_loss = (pg * m).sum() / m.sum().clamp(min=1)
            v_loss = 0.5 * (values - rt).pow(2).mean()
            ent = masked_entropy(logits).mean()
            t = dense_targets(rows[lo:hi], logits.shape[1]) if searched else torch.zeros_like(logp_all)
            distill = -(t * logp_all.masked_fill(torch.isinf(logits), 0.0)).sum(-1)
            # averaged over the whole minibatch, not the searched rows: a handful of
            # sharp targets must not pull as hard as a full batch of policy gradients
            d_loss = (distill * (1 - m)).sum() / len(m)
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent + cfg.distill_coef * d_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                # KL and clip fraction over the policy-gradient rows only: a searched
                # action can have a behaviour probability of 1e-7, and its ratio is not optimized
                mean_m = lambda x: (x * m).sum() / m.sum().clamp(min=1)  # noqa: E731
                kl = mean_m((ratio - 1) - (logp - olp))
                acc += torch.stack([pg_loss, v_loss, ent, kl, mean_m(((ratio - 1).abs() > cfg.clip).float()), d_loss])
        steps += len(chunks)
        epoch = acc.tolist()  # read once per epoch (target_kl), not per step
        totals = [t + x for t, x in zip(totals, epoch)]
        if cfg.target_kl is not None and epoch[STATS.index("approx_kl")] / len(chunks) > cfg.target_kl:
            stop = True
            break
    out = {k: v / max(steps, 1) for k, v in zip(STATS, totals)}
    out["updates"] = steps
    out["early_stop"] = stop
    out["explained_var"] = explained_var
    out["searched"] = searched
    out["search_valued"] = search_valued
    return out
