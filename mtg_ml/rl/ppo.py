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

On CUDA the step itself is captured: every minibatch of an epoch is padded
to one set of shapes (`model.pad_split`, padded rows weighted 0), and the
whole step (forward, losses, backward, gradient clipping, fused Adam) is a
CUDA graph replayed per minibatch (`_StepGraphs`), one launch instead of
~400. The padded path runs eagerly too (mode="padded"), which is how the
tests check it on the CPU.

Belief head (feature set 8): the loss adds coef_arch * CE(archetype) +
coef_counts * CE(copies) averaged over the vocabulary cards, against the
per-trajectory targets `Result.belief_targets` ((archetype index, counts
array) per trajectory, aligned with `Result.lengths`), which every decision
of the trajectory shares. Rows look their targets up by trajectory: eagerly
in per-update tables, in padded pieces in per-piece tables (fixed shapes).
The policy reads the belief predictions detached, so PPO gradients never
reach the belief branch and the belief loss trains only it.
"""

from __future__ import annotations

import math
import time
import weakref
from array import array
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import nn

from .belief import MAX_BASIC, MAX_NONBASIC
from .model import DEFAULT_BELIEF_COEF, PAD_EVIDENCE, PAD_FIELDS, PolicyNet, SequenceLayout, bucket, collate_packed, packed_tensors, pad_fits, pad_sequences, pad_sizes, pad_split, padded_batch, split, structure
from .rollout import KINDS, Result

STATS = ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")  # minibatch means
# After them, per decision kind (`rollout.KINDS`), sums over the non-trivial
# decisions only: their count, entropy and approx KL (`_losses`).
KIND_STATS = ("n", "entropy", "approx_kl")
# Then the belief head's minibatch means (zero without one): its two losses,
# archetype accuracy, exact-count accuracy and mean absolute error of the
# expected copies, both averaged over the vocabulary cards.
BELIEF_STATS = ("arch_loss", "count_loss", "arch_acc", "count_acc", "count_mae")
N_STATS = len(STATS) + len(KINDS) * len(KIND_STATS) + len(BELIEF_STATS)
TRIVIAL_P = 0.99  # a decision whose behaviour probability of the taken option is at least this is trivial
_LOG_TRIVIAL_P = math.log(TRIVIAL_P)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    lr_final: float = 3e-5  # where the learning rate anneals to when the trainer's --lr-anneal-games is set
    epochs: int = 4
    minibatch: int = 2048
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03  # stop the epoch loop early past this
    capture: int = 2  # on CUDA: 1 every step a CUDA graph replay over padded minibatches, 2 also the forward and losses compiled by Inductor (~15% faster, ~10-30 s of compiling per process), 0 plain eager steps
    precision: str = "fp32"  # selective BF16 dense/attention; optimizer and probability math stay FP32
    gru_precision: str = "fp32"  # "bf16": the GRU of the update runs in BF16 (cuDNN, ~2x faster at h256); the rollout's GRU stays FP32. Changes the math slightly: only with a learning A/B
    belief_coef: tuple[float, float] | None = None  # (archetype, counts) belief loss coefficients; None: the network's config `belief_coef`


def make_optimizer(params, lr: float, device, tensor_lr: bool = False) -> torch.optim.Optimizer:
    """Adam with eps 1e-5; on CUDA fused (one kernel for all parameters
    instead of a few per tensor), elsewhere plain. `tensor_lr`: the learning
    rate is a one-element tensor on `device` that `set_lr` changes in place,
    so the CUDA graphs of the update (which read it from device memory) stay
    valid while it anneals; a float lr is baked into each graph and a new
    value means recapturing them all. `Optimizer.load_state_dict`
    restores the `fused` flag from the checkpoint's param groups, so after
    resuming callers must set `group["fused"]` again, and for a checkpoint
    of an unfused optimizer also move every `state["step"]` to its
    parameter's device (plain Adam keeps it on the CPU; fused Adam fails on that)."""
    if tensor_lr:
        lr = torch.tensor(float(lr), dtype=torch.float32, device=device)
    return torch.optim.Adam(params, lr=lr, eps=1e-5, fused=torch.device(device).type == "cuda")


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate of every param group: in place for a tensor lr
    (`make_optimizer(tensor_lr=True)`; captured step graphs keep replaying),
    else by replacing the float (the step graphs are then recaptured)."""
    for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
            g["lr"].fill_(lr)
        else:
            g["lr"] = lr


def get_lr(opt: torch.optim.Optimizer) -> float:
    lr = opt.param_groups[0]["lr"]
    return float(lr.item()) if torch.is_tensor(lr) else float(lr)


def load_optimizer_state(opt: torch.optim.Optimizer, state_dict: dict) -> None:
    """`opt.load_state_dict` for an optimizer from `make_optimizer`: keeps
    this optimizer's `fused` flag (the checkpoint's groups bring their own,
    None for plain Adam) and, when fused, moves every `step` to its
    parameter's device (plain Adam keeps it on the CPU, which fused Adam
    rejects). The steps are the same either way. Also keeps this optimizer's
    learning rates: the checkpoint's groups would bring the lr it was saved
    with, silently overriding a changed --ppo-lr on resume. A tensor lr stays
    the same tensor object (its value is the caller's to set: `set_lr`)."""
    fused = opt.param_groups[0].get("fused")
    lrs = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict(state_dict)
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr
        if "initial_lr" in g:
            g["initial_lr"] = lr
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


def _belief_coef(net: PolicyNet, cfg: PPOConfig) -> tuple[float, float]:
    if cfg.belief_coef is not None:
        return tuple(float(x) for x in cfg.belief_coef)
    c = dict(DEFAULT_BELIEF_COEF, **net.config.get("belief_coef", {}))
    return float(c["archetype"]), float(c["counts"])


def belief_losses(net: PolicyNet, out, arch, counts, mean):
    """(archetype CE, count CE averaged over the vocabulary cards) per row
    reduced by `mean`, and the BELIEF_STATS (detached), of predictions `out`
    (`model.BeliefOut`) against per-row targets arch (B,) and counts (B, V)."""
    bn = net.belief_net
    alog = torch.log_softmax(out.arch.float(), -1)
    arch_ce = -alog.gather(1, arch[:, None])[:, 0]
    t_nb = counts.index_select(1, bn.nonbasic_idx)
    nlog = torch.log_softmax(out.nonbasic.float(), -1)
    ce = -nlog.gather(2, t_nb[:, :, None])[:, :, 0].sum(1)
    hit = (out.nonbasic.argmax(-1) == t_nb).float().sum(1)
    err = ((nlog.exp() * torch.arange(MAX_NONBASIC + 1, device=nlog.device)).sum(-1) - t_nb).abs().sum(1)
    if out.basic is not None:
        t_b = counts.index_select(1, bn.basic_idx)
        blog = torch.log_softmax(out.basic.float(), -1)
        ce = ce - blog.gather(2, t_b[:, :, None])[:, :, 0].sum(1)
        hit = hit + (out.basic.argmax(-1) == t_b).float().sum(1)
        err = err + ((blog.exp() * torch.arange(MAX_BASIC + 1, device=blog.device)).sum(-1) - t_b).abs().sum(1)
    V = counts.shape[1]
    arch_loss, count_loss = mean(arch_ce), mean(ce / V)
    with torch.no_grad():
        stats = torch.stack([arch_loss, count_loss, mean((alog.argmax(-1) == arch).float()), mean(hit / V), mean(err / V)])
    return arch_loss, count_loss, stats


def _losses(net: PolicyNet, cfg: PPOConfig, b, lengths, width: int, a, olp, ad, rt, w=None, n=None, kind=None, belief=None):
    """The PPO loss of one minibatch and its statistics (N_STATS,), detached:
    the STATS means, then per decision kind (`kind`, ids into KINDS; None:
    all "other") the KIND_STATS sums over the non-trivial decisions. With
    row weights `w` (0 for padded rows) and their sum `n`: means over the
    real rows only.

    Non-trivial: the behaviour policy gave the taken option less than
    TRIVIAL_P. That is p_max < TRIVIAL_P except for the rare draws of a
    <1% option from a near-certain decision, which count as non-trivial;
    the rollout records only the taken option's log-prob, so this is the
    test it allows for free. Forced moves (one option) never count.

    belief: (trajectory of each row, archetype per trajectory, counts per
    trajectory (trajectories, V)): the targets of the belief loss, for a
    network with a belief head."""
    if cfg.precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    device_type = ad.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
        logits, values, _, bel = net(b, lengths=lengths, max_options=width, aux=True)
    logits, values = logits.float(), values.float()
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(1, a[:, None]).squeeze(1)
    ratio = (logp - olp).exp()
    mean = (lambda x: x.mean()) if w is None else (lambda x: (w * x).sum() / n)  # noqa: E731
    pg_loss = -mean(torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad))
    v_loss = 0.5 * mean((values - rt).pow(2))
    row_ent = -(logp_all.exp() * logp_all.masked_fill(torch.isinf(logits), 0.0)).sum(-1)  # masked_entropy, sharing the log_softmax
    ent = mean(row_ent)
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    b_stats = None
    if bel is not None:
        if belief is None:
            raise ValueError("a network with a belief head needs belief targets (Result.belief_targets)")
        traj, arch_t, counts_t = belief
        ca, cc = _belief_coef(net, cfg)
        arch_loss, count_loss, b_stats = belief_losses(net, bel, arch_t[traj], counts_t[traj], mean)
        loss = loss + ca * arch_loss + cc * count_loss
    with torch.no_grad():
        row_kl = (ratio - 1) - (logp - olp)
        kl = mean(row_kl)
        base = torch.stack([pg_loss, v_loss, ent, kl, mean(((ratio - 1).abs() > cfg.clip).float())])
        nt = (olp < _LOG_TRIVIAL_P).to(row_kl.dtype)
        if w is not None:
            nt = nt * w
        per = torch.stack([nt, nt * row_ent.detach(), nt * row_kl], 1)
        if kind is None:
            kind = torch.zeros_like(a)
        by_kind = per.new_zeros(len(KINDS), len(KIND_STATS)).index_add_(0, kind, per)
        stats = torch.cat([base, by_kind.view(-1), base.new_zeros(len(BELIEF_STATS)) if b_stats is None else b_stats.to(base.dtype)])
    return loss, stats


def _step(net: PolicyNet, opt: torch.optim.Optimizer, cfg: PPOConfig, loss: torch.Tensor, fold_clip: bool = False) -> None:
    """Backward, gradient clipping, Adam. fold_clip, with fused Adam: the
    clipping factor goes to Adam as `grad_scale` (which it divides the
    gradients by), saving clip_grad_norm_'s pass over every gradient; equal
    up to rounding."""
    loss.backward()
    if fold_clip and opt.param_groups[0].get("fused"):
        norm = nn.utils.get_total_norm([p.grad for p in net.parameters() if p.grad is not None])
        opt.grad_scale = ((norm + 1e-6) / cfg.max_grad_norm).clamp(min=1.0)
        try:
            opt.step()
        finally:
            del opt.grad_scale
    else:
        nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        opt.step()


# -- padded minibatches (static shapes) and CUDA graphs of the step ---------------


TRANSPOSED_BAGS = True  # padded path: embedding weight gradients from per-epoch transposes (model._TransposedBag)
FOLD_CLIP = True  # padded path: gradient clipping folded into fused Adam (`_step`)


def _inputs(shapes: dict) -> list[tuple[str, str, int]]:
    """The inputs of a padded minibatch: (name, "i" for int64 or "f" for
    float32, size); the transposed token lists are those with a u_ size."""
    R, keys = shapes["row"], [k for k in ("st", "e", "ot") if f"u_{k}" in shapes]
    out = [(name, "i", shapes[lvl]) for name, (lvl, _, _) in PAD_FIELDS.items()]
    if "v" in shapes:  # belief evidence and targets: per row its trajectory in the piece's tables
        out += [(name, "i", shapes[lvl]) for name, (lvl, _, _) in PAD_EVIDENCE.items()]
        out += [("b_traj", "i", R), ("b_arch", "i", shapes["traj"]), ("b_counts", "i", shapes["traj"] * shapes["vocab"])]
    out += [(f"t_{k}_{x}", "i", shapes[k] if x == "bag" else shapes[f"u_{k}"]) for k in keys for x in ("bag", "off", "uid")]
    out += [(name, "i", R) for name in ("pos_in", "pos_out", "action", "kind")] + [(name, "f", R) for name in ("old_logp", "adv", "ret", "weight")]
    out += [("e_pos", "i", shapes["ent"])] if "entw" in shapes else []  # entity attention
    return out + [("n", "f", 1)] + ([("t_e_w", "f", shapes["e"])] if "e" in keys else [])


def _layout(shapes: dict) -> dict:
    """Where each input of a padded minibatch sits in its flat int64 and
    float32 buffers: name -> (buffer, start, size)."""
    out, at = {}, {"i": 0, "f": 0}
    for name, buf, size in _inputs(shapes):
        out[name] = (buf, at[buf], size)
        at[buf] += size
    return out


def _padded_losses(net, cfg, shapes, gru, ints, flts):
    """`_losses` of the padded minibatch in the flat buffers `ints`, `flts`,
    its GRU batch of shape gru = (sequences, steps)."""
    v = {k: (ints if buf == "i" else flts).narrow(0, at, size) for k, (buf, at, size) in _layout(shapes).items()}
    seq = SequenceLayout(v["pos_in"], v["pos_out"], *gru)
    belief = (v["b_traj"], v["b_arch"], v["b_counts"].view(shapes["traj"], shapes["vocab"])) if "v" in shapes else None
    return _losses(net, cfg, padded_batch(v, shapes.get("entw")), seq, shapes["width"], v["action"], v["old_logp"], v["adv"], v["ret"], v["weight"], v["n"][0], v["kind"], belief)


def _need(counts: dict, extra: dict) -> dict:
    """Padded shapes for an epoch's pieces: `pad_sizes` per level, plus the
    logit width (and, with belief targets, the trajectories per piece and
    the vocabulary size)."""
    out = pad_sizes(counts) | {"width": extra["width"]}
    if "traj" in extra:
        out |= {"traj": bucket(extra["traj"] + 1), "vocab": extra["vocab"]}
    return out


def _padded_epoch(big, bounds, chunks, acts, rec, width, choose, transpose=None, choose_gru=None, kinds=None, ent_width=False, belief=None):
    """The minibatches of an epoch padded to one set of shapes, packed as
    (pieces, ·) int64 and float32 tensors in `_layout` order, and the GRU
    batch of each, (trajectories, longest), rounded to 4 sizes per octave:
    the GRU's time is ~linear in both. `choose(counts, extra)` picks the
    shapes (`_need` or a cached one they fit), `choose_gru(shapes, gru)`
    may pick a bigger GRU batch; `transpose` and `ent_width` (entity
    attention) go to `pad_split`. belief: (trajectory id of each decision
    in epoch order (CPU), archetype per trajectory, counts per trajectory
    (trajectories, V)), for per-piece target tables."""
    dev = acts.device
    extra = {"width": int(width.max())}
    if belief is not None:
        extra |= {"traj": max(len(c) for c in chunks), "vocab": belief[2].shape[1]}
    fields, shapes, counts = pad_split(big, bounds, lambda counts: choose(counts, extra), transpose, ent_width)
    M, R = len(chunks), shapes["row"]
    gru = [(bucket(len(c), 4), bucket(max(c), 4)) for c in chunks]
    if choose_gru:
        gru = [choose_gru(shapes, g) for g in gru]
    pos_in, pos_out = pad_sequences(chunks, R, gru)
    n = torch.tensor(counts["row"], dtype=torch.long)
    piece = torch.repeat_interleave(torch.arange(M), n)
    dest = (piece * R + torch.arange(int(n.sum())) - torch.tensor(bounds[:-1])[piece]).to(dev)
    padded = lambda x: x.new_zeros(M * R).index_copy(0, dest, x).view(M, R)  # noqa: E731
    kinds = torch.zeros_like(acts) if kinds is None else kinds
    fields.update(pos_in=pos_in.to(dev), pos_out=pos_out.to(dev), action=padded(acts), kind=padded(kinds), old_logp=padded(rec[0]), adv=padded(rec[1]), ret=padded(rec[2]))
    fields.update(weight=padded(torch.ones_like(rec[0])), n=n.to(dev, torch.float32)[:, None])
    if belief is not None:
        tid, arch_t, counts_t = belief
        TP, V = shapes["traj"], shapes["vocab"]
        k = torch.tensor([len(c) for c in chunks], dtype=torch.long)
        lens = torch.tensor([x for c in chunks for x in c], dtype=torch.long)
        first = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
        ids = tid[first]  # each trajectory of the epoch, in order
        tp = torch.repeat_interleave(torch.arange(M), k)
        slot = tp * TP + torch.arange(len(lens)) - (k.cumsum(0) - k)[tp]  # its place in its piece's table
        local = (slot - tp * TP).to(dev)
        fields["b_traj"] = padded(torch.repeat_interleave(local, lens.to(dev), output_size=int(lens.sum())))
        slot, ids = slot.to(dev), ids.to(dev)
        fields["b_arch"] = arch_t.new_zeros(M * TP).index_copy(0, slot, arch_t[ids]).view(M, TP)
        fields["b_counts"] = counts_t.new_zeros(M * TP, V).index_copy(0, slot, counts_t[ids]).view(M, TP * V)
    spec = _inputs(shapes)
    ints = torch.cat([fields[name] for name, buf, _ in spec if buf == "i"], 1)
    flts = torch.cat([fields[name] for name, buf, _ in spec if buf == "f"], 1)
    return shapes, gru, ints, flts


class _StepGraphs:
    """CUDA graphs of the whole training step of one (network, optimizer):
    forward, losses, backward, gradient clipping, Adam and the statistics
    sum; one graph per padded shape and GRU batch, replayed with each
    minibatch copied into its static input buffers. A new shape is captured
    once (after a warm-up forward and backward on a side stream, as CUDA
    graphs need); epochs whose minibatches fit a captured shape reuse its
    graphs. The graphs hold the addresses of the weights, Adam state and the
    hyperparameters, so `fingerprint` covers these, and `_graphs_for` drops
    the graphs when it changes (a reloaded optimizer state, a new float
    learning rate). A tensor learning rate enters by its address, not its
    value: the graphs read it, so `set_lr` changes it without a recapture."""

    MAX_SHAPES = 3  # each with a graph per GRU batch (~16)
    MARGIN = 1.25  # a new shape has room for 25% more than the epoch that needed it: a shape is ~16-50 captures (seconds), and games grow over a long run (1.05 recaptured every few iterations: 1.7 -> 6 s per update)

    def __init__(self, net, opt, cfg):
        self.fingerprint = _StepGraphs.fingerprint_of(net, opt, cfg)
        self.pools: dict[tuple, tuple] = {}  # shapes -> memory pool, shared by its graphs (they never run concurrently); dropped with them
        self.stream = torch.cuda.Stream()  # one for every warm-up and capture: the allocator caches blocks per stream, so a new stream per capture strands its warm-up's memory
        self.graphs: dict[tuple, dict[tuple, tuple]] = {}  # shapes -> GRU batch -> (graph, static ints, static floats)
        self.acc = torch.zeros(N_STATS, dtype=torch.float64, device=next(net.parameters()).device)
        self.captures, self.capture_s = 0, 0.0

    @staticmethod
    def fingerprint_of(net, opt, cfg) -> tuple:
        params = [p for g in opt.param_groups for p in g["params"]]
        state = tuple(t.data_ptr() for p in params for t in opt.state.get(p, {}).values() if torch.is_tensor(t))
        lr = lambda x: ("tensor", x.data_ptr(), x.device) if torch.is_tensor(x) else x  # noqa: E731
        groups = tuple((lr(g["lr"]), g["betas"], g["eps"], g["weight_decay"], g.get("amsgrad"), g.get("maximize")) for g in opt.param_groups)
        coef = _belief_coef(net, cfg) if getattr(net, "belief_net", None) is not None else None
        return (id(net), tuple(p.data_ptr() for p in params), state, groups, cfg.clip, cfg.vf_coef, cfg.ent_coef, cfg.max_grad_norm, cfg.capture, cfg.precision, cfg.gru_precision, coef)

    def choose(self, counts: dict, extra: dict) -> dict:
        """The smallest captured shape these pieces fit, else a new one with
        some room (MARGIN) that also fits every earlier epoch (the
        elementwise max), so that the shapes settle within a few epochs."""
        fit = [dict(k) for k in self.graphs if pad_fits(dict(k), counts) and all(dict(k)[x] >= v for x, v in extra.items())]
        if fit:
            return min(fit, key=lambda s: (s["row"], s["st"]))
        roomy = {k: [int(x * self.MARGIN) for x in v] if k != "opt" else v for k, v in counts.items()}
        new = _need(roomy, extra)
        grown = {k: max([v] + [dict(c).get(k, 0) for c in self.graphs]) for k, v in new.items()}
        grown["bag"] = grown["row"] + grown["ent"]
        grown["opt"] = max(grown["opt"], bucket(max(o + grown["row"] - r for o, r in zip(counts["opt"], counts["row"]))))
        return grown

    def choose_gru(self, shapes: dict, gru: tuple) -> tuple:
        """A captured GRU batch that fits `gru` = (trajectories, steps) and
        costs at most 15% more (the GRU takes ~steps * (1.9 + 0.06 *
        trajectories) us on an H100), else `gru`: rare combinations would
        otherwise each cost a capture."""
        cost = lambda g: g[1] * (1.9 + 0.06 * g[0])  # noqa: E731
        have = [g for g in self.graphs.get(tuple(sorted(shapes.items())), {}) if g[0] >= gru[0] and g[1] >= gru[1]]
        best = min(have, key=cost, default=None)
        return best if best is not None and cost(best) <= 1.15 * cost(gru) else gru

    def run(self, net, opt, cfg, shapes: dict, gru: list[tuple], ints: torch.Tensor, flts: torch.Tensor) -> None:
        """One step per minibatch (row) of ints, flts; statistics summed into `acc`."""
        key = tuple(sorted(shapes.items()))
        if key not in self.graphs:
            if len(self.graphs) >= self.MAX_SHAPES:
                old = next(iter(self.graphs))
                del self.graphs[old], self.pools[old]
            self.graphs[key], self.pools[key] = {}, torch.cuda.graph_pool_handle()
        graphs = self.graphs[key]
        for m in range(ints.shape[0]):
            if gru[m] not in graphs:
                if any(p not in opt.state for g in opt.param_groups for p in g["params"]):  # Adam allocates its state at the first step: not in a graph
                    opt.zero_grad(set_to_none=True)
                    loss, stats = _padded_losses(net, cfg, shapes, gru[m], ints[m], flts[m])
                    _step(net, opt, cfg, loss, FOLD_CLIP)
                    self.acc += stats
                    del loss, stats  # a live autograd graph would break the capture
                    self.fingerprint = _StepGraphs.fingerprint_of(net, opt, cfg)
                    continue
                graphs[gru[m]] = self._capture(net, opt, cfg, shapes, gru[m], ints[m], flts[m], self.pools[key])
            graph, si, sf = graphs[gru[m]]
            si.copy_(ints[m])
            sf.copy_(flts[m])
            graph.replay()

    def _capture(self, net, opt, cfg, shapes, gru, ints, flts, pool) -> tuple:
        t = time.perf_counter()
        losses = _compiled_losses() if cfg.capture >= 2 else _padded_losses  # compiles in the warm-up, outside the capture
        si, sf = ints.clone(), flts.clone()
        _drop_entities(net)
        side, main = self.stream, torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):  # lazy initialisation (cuDNN, cuBLAS) must not happen while capturing
            opt.zero_grad(set_to_none=True)
            loss, _ = losses(net, cfg, shapes, gru, si, sf)
            loss.backward()
            del loss
        main.wait_stream(side)
        opt.zero_grad(set_to_none=True)  # the graph allocates the gradients, so replays overwrite them instead of accumulating
        _drop_entities(net)
        capturable = [g.get("capturable", False) for g in opt.param_groups]
        for g in opt.param_groups:
            g["capturable"] = True  # fused Adam keeps its step counts on the device either way; this only lifts the capture check
        graph = torch.cuda.CUDAGraph()
        side.wait_stream(main)
        try:  # not `torch.cuda.graph`, which synchronizes and empties the allocator's cache on entry: ~100 ms per capture
            with torch.cuda.stream(side):
                _clear_cublas_workspaces()  # so that the capture allocates its own, in its pool
                graph.capture_begin(pool=pool)
                try:
                    loss, stats = losses(net, cfg, shapes, gru, si, sf)
                    _step(net, opt, cfg, loss, FOLD_CLIP)
                    self.acc += stats
                    del loss, stats
                finally:
                    graph.capture_end()
                    _clear_cublas_workspaces()  # ... and eager matmuls never share it
            main.wait_stream(side)
        finally:
            for g, c in zip(opt.param_groups, capturable):
                g["capturable"] = c
            _drop_entities(net)
        self.captures += 1
        self.capture_s += time.perf_counter() - t
        return graph, si, sf


def _clear_cublas_workspaces() -> None:
    """cuBLAS keeps a workspace per (handle, stream), allocated on first use
    and released by `torch.cuda.empty_cache` (and the allocator's recovery
    from an out-of-memory). A graph captured while one exists bakes in its
    address, and replays after it is released write to freed memory
    (illegal-address errors in the trainer). Dropping the workspaces right
    before a capture makes the capture allocate one inside the graph's pool,
    owned by the graph; dropping them right after keeps eager code off it.
    (Inductor's CUDA graph trees do the same.)"""
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()


_COMPILED = []


def _compiled_losses():
    """`_padded_losses` compiled by Inductor (fuses the ~100 small kernels of
    the losses and their backward); captured inside the step graphs."""
    if not _COMPILED:
        _COMPILED.append(torch.compile(_padded_losses))
    return _COMPILED[0]


def set_gru_precision(net: PolicyNet, precision: str) -> None:
    """Run the GRUs of `net` in sequence mode (the update) in `precision` ("fp32" or "bf16"); step mode
    (rollouts) is unaffected."""
    if precision not in ("fp32", "bf16"):
        raise ValueError("gru_precision must be fp32 or bf16")
    for m in net.modules():
        if hasattr(m, "seq_bf16"):
            m.seq_bf16 = precision == "bf16"


def _drop_entities(net: PolicyNet) -> None:
    """The cores keep the entity vectors of their last forward (for the
    pointer scores); a capture must not start with stale autograd references."""
    for m in net.modules():
        if hasattr(m, "entities"):
            m.entities = None


_GRAPHS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # optimizer -> _StepGraphs


def _graphs_for(net, opt, cfg) -> _StepGraphs:
    g = _GRAPHS.get(opt)
    if g is None or g.fingerprint != _StepGraphs.fingerprint_of(net, opt, cfg):
        g = _GRAPHS[opt] = _StepGraphs(net, opt, cfg)
    return g


def ppo_update(net: PolicyNet, opt: torch.optim.Optimizer, data: Result, cfg: PPOConfig, device="cpu", gen=None, mode: str | None = None) -> dict:
    """`cfg.epochs` PPO epochs over `data`. Returns the STATS (means over
    every step), `updates`, `early_stop`, `explained_var`, the approx KL of
    the first and the last epoch run (`approx_kl_first`, `approx_kl_last`),
    and over the non-trivial decisions (`_losses`): their share `nt_frac`,
    `nt_entropy`, `nt_approx_kl` (all epochs), `nt_approx_kl_first` /
    `_last`, and per decision kind with any: `kind/<kind>/share` (of the
    non-trivial decisions), `kind/<kind>/entropy`, `kind/<kind>/approx_kl`,
    and with a belief head `belief/<BELIEF_STATS>` (means over every step),
    trained against `data.belief_targets`.
    mode: "eager" (each minibatch as
    it is), "padded" (minibatches padded to static shapes, run eagerly: the
    math of the captured path, so it is testable on the CPU) or "graph"
    (padded, every step a CUDA graph replay). Default: "graph" on CUDA when
    `cfg.capture` and the trunk is not "transformer" (whose encoder waits
    for the device), else "eager"."""
    n = len(data.actions)
    if n == 0:  # nothing to learn from (and no minibatches to average over)
        return {**{k: 0.0 for k in STATS}, "updates": 0, "early_stop": False, "explained_var": float("nan")}  # no breakdowns
    floats = lambda xs: torch.frombuffer(array("f", xs), dtype=torch.float32)  # noqa: E731 - 3x faster than torch.tensor(list)
    old_logp, adv, ret = floats(data.logps), floats(data.advantages), floats(data.returns)
    var = ret.var().item() if n > 1 else 0.0
    explained_var = float("nan") if var == 0 else 1 - adv.var().item() / var  # values = ret - adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if n > 1 else adv - adv.mean()  # std of one sample is NaN
    net.train()
    set_gru_precision(net, cfg.gru_precision)
    dev = torch.device(device)
    if mode is None:
        mode = "graph" if dev.type == "cuda" and cfg.capture and net.config["trunk"] != "transformer" else "eager"
    if mode not in ("eager", "padded", "graph") or (mode == "graph" and dev.type != "cuda"):
        raise ValueError(f"mode {mode!r} on {dev}")
    graphs = _graphs_for(net, opt, cfg) if mode == "graph" else None
    captures = graphs.captures if graphs else 0
    sd, od = net.config["state_dim"], net.config["option_dim"]
    transpose = {"st": (sd, "bag_off" if net.config["trunk"] == "entity" else "st_off"), "e": (od, "e_off"), "ot": (od, "ot_off")} if TRANSPOSED_BAGS else None
    samples = packed_tensors(data.samples, dev)
    n_opts = torch.frombuffer(data.samples.n_opts, dtype=torch.int32)  # on the CPU: logit widths without asking the device
    actions = torch.frombuffer(array("q", data.actions), dtype=torch.long).to(dev)
    recorded = torch.stack([old_logp, adv, ret]).to(dev)
    kinds = data.kinds if len(data.kinds) == n else array("b", bytes(n))  # Results built by hand may lack them: all "other"
    kinds = torch.frombuffer(kinds if isinstance(kinds, array) and kinds.typecode == "b" else array("b", kinds), dtype=torch.int8).to(dev, torch.long)
    has_belief = getattr(net, "belief_net", None) is not None
    belief = None
    if has_belief:
        targets = getattr(data, "belief_targets", None) or []
        if len(targets) != len(data.lengths):
            raise ValueError(f"a network with a belief head needs one belief target per trajectory: {len(targets)} for {len(data.lengths)}")
        V = net.belief_net.n_vocab
        arch_t = torch.tensor([int(t[0]) for t in targets], dtype=torch.long)
        counts_t = torch.frombuffer(array("i", (c for t in targets for c in t[1])), dtype=torch.int32).long().view(len(targets), V) if targets else torch.zeros(0, V, dtype=torch.long)
        tid = torch.repeat_interleave(torch.arange(len(targets)), torch.tensor(data.lengths, dtype=torch.long))  # trajectory of each decision (CPU)
        tables = (arch_t.to(dev), counts_t.to(dev))
    totals = [0.0] * N_STATS
    steps = 0
    stop = False
    epoch_kl: list[tuple[float, float]] = []  # (approx KL, non-trivial approx KL) per epoch
    for _ in range(cfg.epochs):
        order, chunks = trajectory_minibatches(data.lengths, cfg.minibatch, gen)
        bounds = list(accumulate((sum(c) for c in chunks), initial=0))
        rows = order.to(dev)
        big = structure(collate_packed(samples, rows, evidence=has_belief), sd, od)
        if has_belief:
            tid_e = tid[order]
            belief = (tid_e.to(dev), *tables)
        acts, rec, width, knd = actions[rows], recorded[:, rows], n_opts[order], kinds[rows]
        acc = graphs.acc if graphs else torch.zeros(N_STATS, dtype=torch.float64, device=dev)
        if mode == "eager":
            for b, lens, lo, hi in zip(split(big, bounds), chunks, bounds, bounds[1:]):
                opt.zero_grad(set_to_none=True)
                loss, stats = _losses(net, cfg, b, lens, int(width[lo:hi].max()), acts[lo:hi], *rec[:, lo:hi], kind=knd[lo:hi], belief=belief and (belief[0][lo:hi], *tables))
                _step(net, opt, cfg, loss)
                acc += stats
        else:
            shapes, gru, ints, flts = _padded_epoch(
                big, bounds, chunks, acts, rec, width, graphs.choose if graphs else _need, transpose, graphs and graphs.choose_gru,
                kinds=knd, ent_width=bool(net.config.get("entity_attn")), belief=(tid_e, *tables) if has_belief else None,
            )
            del big
            if graphs:
                graphs.run(net, opt, cfg, shapes, gru, ints, flts)
            else:
                for m in range(len(chunks)):
                    opt.zero_grad(set_to_none=True)
                    loss, stats = _padded_losses(net, cfg, shapes, gru[m], ints[m], flts[m])
                    _step(net, opt, cfg, loss, FOLD_CLIP)
                    acc += stats
        steps += len(chunks)
        epoch = acc.tolist()  # read once per epoch (target_kl), not per step
        acc.zero_()
        totals = [t + x for t, x in zip(totals, epoch)]
        nt_n, _, nt_kl = _by_kind(epoch).sum(0).tolist()
        epoch_kl.append((epoch[STATS.index("approx_kl")] / len(chunks), nt_kl / nt_n if nt_n else float("nan")))
        if cfg.target_kl is not None and epoch[STATS.index("approx_kl")] / len(chunks) > cfg.target_kl:
            stop = True
            break
    if graphs:
        opt.zero_grad(set_to_none=True)  # the gradients live in the graphs' pool
        if graphs.captures > captures:  # the warm-ups leave cached blocks of new sizes behind: ~1 GiB per capture at minibatch 8192
            torch.cuda.empty_cache()
    out = {k: v / max(steps, 1) for k, v in zip(STATS, totals)}
    out["updates"] = steps
    if graphs:
        out["captures"] = graphs.captures - captures
        out["graph_shapes"] = len(graphs.graphs)
    out["early_stop"] = stop
    out["explained_var"] = explained_var
    out["approx_kl_first"], out["nt_approx_kl_first"] = epoch_kl[0]
    out["approx_kl_last"], out["nt_approx_kl_last"] = epoch_kl[-1]
    by_kind = _by_kind(totals)
    nt_n, nt_ent, nt_kl = by_kind.sum(0).tolist()
    nan = float("nan")
    out["nt_frac"] = nt_n / (n * len(epoch_kl))
    out["nt_entropy"] = nt_ent / nt_n if nt_n else nan
    out["nt_approx_kl"] = nt_kl / nt_n if nt_n else nan
    for name, (k_n, k_ent, k_kl) in zip(KINDS, by_kind.tolist()):
        if k_n:
            out.update({f"kind/{name}/share": k_n / nt_n, f"kind/{name}/entropy": k_ent / k_n, f"kind/{name}/approx_kl": k_kl / k_n})
    if has_belief:
        out.update({f"belief/{k}": v / max(steps, 1) for k, v in zip(BELIEF_STATS, totals[N_STATS - len(BELIEF_STATS) :])})
    return out


def _by_kind(stats: list[float]) -> torch.Tensor:
    """The per-kind sums of a statistics vector as a (kinds, KIND_STATS) tensor."""
    return torch.tensor(stats[len(STATS) : len(STATS) + len(KINDS) * len(KIND_STATS)], dtype=torch.float64).view(len(KINDS), len(KIND_STATS))
