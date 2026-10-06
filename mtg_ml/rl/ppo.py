"""Clipped PPO update over recorded decisions (masked categorical policy).

Minibatches are sets of whole trajectories (about `minibatch` decisions
each), so the recurrent core is replayed from the start of every game with
the current weights instead of reusing stale hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import PolicyNet, collate, collate_packed, masked_entropy
from .samples import PackedSamples
from .rollout import Result


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


def trajectory_minibatches(lengths: list[int], size: int, gen=None) -> list[tuple[list[int], list[int]]]:
    """Shuffle trajectories and group them into (step indices, lengths) chunks of >= size steps."""
    starts, acc = [], 0
    for n in lengths:
        starts.append(acc)
        acc += n
    out, idx, lens = [], [], []
    for t in torch.randperm(len(lengths), generator=gen).tolist():
        idx.extend(range(starts[t], starts[t] + lengths[t]))
        lens.append(lengths[t])
        if len(idx) >= size:
            out.append((idx, lens))
            idx, lens = [], []
    if idx:
        out.append((idx, lens))
    return out


def ppo_update(net: PolicyNet, opt: torch.optim.Optimizer, data: Result, cfg: PPOConfig, device="cpu", gen=None) -> dict:
    n = len(data.actions)
    actions = torch.tensor(data.actions, dtype=torch.long)
    old_logp = torch.tensor(data.logps, dtype=torch.float32)
    adv = torch.tensor(data.advantages, dtype=torch.float32)
    ret = torch.tensor(data.returns, dtype=torch.float32)
    var = ret.var().item()
    explained_var = float("nan") if n < 2 or var == 0 else 1 - adv.var().item() / var  # values = ret - adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    net.train()
    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    steps = 0
    stop = False
    for _ in range(cfg.epochs):
        epoch_kl = []
        for idx, lens in trajectory_minibatches(data.lengths, cfg.minibatch, gen):
            b = (collate_packed(data.samples, idx) if isinstance(data.samples, PackedSamples) else collate([data.samples[i] for i in idx])).to(device)
            it = torch.tensor(idx)
            a, olp, ad, rt = (x[it].to(device) for x in (actions, old_logp, adv, ret))
            logits, values, _ = net(b, lengths=lens)
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a[:, None]).squeeze(1)
            ratio = (logp - olp).exp()
            pg_loss = -torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad).mean()
            v_loss = 0.5 * (values - rt).pow(2).mean()
            ent = masked_entropy(logits).mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - (logp - olp)).mean().item()
                epoch_kl.append(kl)
                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["approx_kl"] += kl
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            steps += 1
        if cfg.target_kl is not None and sum(epoch_kl) / len(epoch_kl) > cfg.target_kl:
            stop = True
            break
    out = {k: v / max(steps, 1) for k, v in stats.items()}
    out["updates"] = steps
    out["early_stop"] = stop
    out["explained_var"] = explained_var
    return out
