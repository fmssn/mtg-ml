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
"""

from __future__ import annotations

import time
import weakref
from array import array
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import nn

from .model import PAD_FIELDS, PolicyNet, SequenceLayout, bucket, collate_packed, masked_entropy, packed_tensors, pad_fits, pad_sequences, pad_sizes, pad_split, padded_batch, split, structure
from .rollout import Result

STATS = ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")


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
    capture: int = 1  # on CUDA, every step a CUDA graph replay over padded minibatches (0: plain eager steps)


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


def _losses(net: PolicyNet, cfg: PPOConfig, b, lengths, width: int, a, olp, ad, rt, w=None, n=None):
    """The PPO loss of one minibatch and its statistics (5,), detached. With
    row weights `w` (0 for padded rows) and their sum `n`: means over the
    real rows only."""
    logits, values, _ = net(b, lengths=lengths, max_options=width)
    logp = torch.log_softmax(logits, dim=-1).gather(1, a[:, None]).squeeze(1)
    ratio = (logp - olp).exp()
    mean = (lambda x: x.mean()) if w is None else (lambda x: (w * x).sum() / n)  # noqa: E731
    pg_loss = -mean(torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad))
    v_loss = 0.5 * mean((values - rt).pow(2))
    ent = mean(masked_entropy(logits))
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    with torch.no_grad():
        kl = mean((ratio - 1) - (logp - olp))
        stats = torch.stack([pg_loss, v_loss, ent, kl, mean(((ratio - 1).abs() > cfg.clip).float())])
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
    out += [(f"t_{k}_{x}", "i", shapes[k] if x == "bag" else shapes[f"u_{k}"]) for k in keys for x in ("bag", "off", "uid")]
    out += [(name, "i", R) for name in ("pos_in", "pos_out", "action")] + [(name, "f", R) for name in ("old_logp", "adv", "ret", "weight")]
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
    return _losses(net, cfg, padded_batch(v), seq, shapes["width"], v["action"], v["old_logp"], v["adv"], v["ret"], v["weight"], v["n"][0])


def _need(counts: dict, extra: dict) -> dict:
    """Padded shapes for an epoch's pieces: `pad_sizes` per level, plus the
    logit width."""
    return pad_sizes(counts) | {"width": extra["width"]}


def _padded_epoch(big, bounds, chunks, acts, rec, width, choose, transpose=None):
    """The minibatches of an epoch padded to one set of shapes, packed as
    (pieces, ·) int64 and float32 tensors in `_layout` order, and the GRU
    batch of each, (trajectories, longest), rounded to 4 sizes per octave:
    the GRU's time is ~linear in both. `choose(counts, extra)` picks the
    shapes (`_need` or a cached one they fit); `transpose` goes to
    `pad_split`."""
    dev = acts.device
    extra = {"width": int(width.max())}
    fields, shapes, counts = pad_split(big, bounds, lambda counts: choose(counts, extra), transpose)
    M, R = len(chunks), shapes["row"]
    gru = [(bucket(len(c), 4), bucket(max(c), 4)) for c in chunks]
    pos_in, pos_out = pad_sequences(chunks, R, gru)
    n = torch.tensor(counts["row"], dtype=torch.long)
    piece = torch.repeat_interleave(torch.arange(M), n)
    dest = (piece * R + torch.arange(int(n.sum())) - torch.tensor(bounds[:-1])[piece]).to(dev)
    padded = lambda x: x.new_zeros(M * R).index_copy(0, dest, x).view(M, R)  # noqa: E731
    fields.update(pos_in=pos_in.to(dev), pos_out=pos_out.to(dev), action=padded(acts), old_logp=padded(rec[0]), adv=padded(rec[1]), ret=padded(rec[2]))
    fields.update(weight=padded(torch.ones_like(rec[0])), n=n.to(dev, torch.float32)[:, None])
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
    the graphs when it changes (a reloaded optimizer state, a new learning
    rate)."""

    MAX_SHAPES = 3  # each with a graph per GRU batch (~20)

    def __init__(self, net, opt, cfg):
        self.fingerprint = _StepGraphs.fingerprint_of(net, opt, cfg)
        self.pool = torch.cuda.graph_pool_handle()  # shared: the graphs never run concurrently
        self.graphs: dict[tuple, dict[tuple, tuple]] = {}  # shapes -> GRU batch -> (graph, static ints, static floats)
        self.acc = torch.zeros(len(STATS), dtype=torch.float64, device=next(net.parameters()).device)
        self.captures, self.capture_s = 0, 0.0

    @staticmethod
    def fingerprint_of(net, opt, cfg) -> tuple:
        params = [p for g in opt.param_groups for p in g["params"]]
        state = tuple(t.data_ptr() for p in params for t in opt.state.get(p, {}).values() if torch.is_tensor(t))
        groups = tuple((g["lr"], g["betas"], g["eps"], g["weight_decay"], g.get("amsgrad"), g.get("maximize")) for g in opt.param_groups)
        return (id(net), tuple(p.data_ptr() for p in params), state, groups, cfg.clip, cfg.vf_coef, cfg.ent_coef, cfg.max_grad_norm)

    def choose(self, counts: dict, extra: dict) -> dict:
        """The smallest captured shape these pieces fit, else a new one."""
        fit = [dict(k) for k in self.graphs if pad_fits(dict(k), counts) and all(dict(k)[x] >= v for x, v in extra.items())]
        return min(fit, key=lambda s: (s["row"], s["st"])) if fit else _need(counts, extra)

    def run(self, net, opt, cfg, shapes: dict, gru: list[tuple], ints: torch.Tensor, flts: torch.Tensor) -> None:
        """One step per minibatch (row) of ints, flts; statistics summed into `acc`."""
        key = tuple(sorted(shapes.items()))
        if key not in self.graphs:
            if len(self.graphs) >= self.MAX_SHAPES:
                del self.graphs[next(iter(self.graphs))]
            self.graphs[key] = {}
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
                graphs[gru[m]] = self._capture(net, opt, cfg, shapes, gru[m], ints[m], flts[m])
            graph, si, sf = graphs[gru[m]]
            si.copy_(ints[m])
            sf.copy_(flts[m])
            graph.replay()

    def _capture(self, net, opt, cfg, shapes, gru, ints, flts) -> tuple:
        t = time.perf_counter()
        si, sf = ints.clone(), flts.clone()
        _drop_entities(net)
        side, main = torch.cuda.Stream(), torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):  # lazy initialisation (cuDNN, cuBLAS) must not happen while capturing
            opt.zero_grad(set_to_none=True)
            loss, _ = _padded_losses(net, cfg, shapes, gru, si, sf)
            loss.backward()
            del loss
        main.wait_stream(side)
        opt.zero_grad(set_to_none=True)  # the graph allocates the gradients, so replays overwrite them instead of accumulating
        _drop_entities(net)
        capturable = [g.get("capturable", False) for g in opt.param_groups]
        for g in opt.param_groups:
            g["capturable"] = True  # fused Adam keeps its step counts on the device either way; this only lifts the capture check
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, pool=self.pool):
                loss, stats = _padded_losses(net, cfg, shapes, gru, si, sf)
                _step(net, opt, cfg, loss, FOLD_CLIP)
                self.acc += stats
                del loss, stats
        finally:
            for g, c in zip(opt.param_groups, capturable):
                g["capturable"] = c
            _drop_entities(net)
        self.captures += 1
        self.capture_s += time.perf_counter() - t
        return graph, si, sf


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
    """`cfg.epochs` PPO epochs over `data`. mode: "eager" (each minibatch as
    it is), "padded" (minibatches padded to static shapes, run eagerly: the
    math of the captured path, so it is testable on the CPU) or "graph"
    (padded, every step a CUDA graph replay). Default: "graph" on CUDA when
    `cfg.capture` and the trunk is not "transformer" (whose encoder waits
    for the device), else "eager"."""
    n = len(data.actions)
    floats = lambda xs: torch.frombuffer(array("f", xs), dtype=torch.float32)  # noqa: E731 - 3x faster than torch.tensor(list)
    old_logp, adv, ret = floats(data.logps), floats(data.advantages), floats(data.returns)
    var = ret.var().item()
    explained_var = float("nan") if n < 2 or var == 0 else 1 - adv.var().item() / var  # values = ret - adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    net.train()
    dev = torch.device(device)
    if mode is None:
        mode = "graph" if dev.type == "cuda" and cfg.capture and net.config["trunk"] != "transformer" else "eager"
    if mode not in ("eager", "padded", "graph") or (mode == "graph" and dev.type != "cuda"):
        raise ValueError(f"mode {mode!r} on {dev}")
    graphs = _graphs_for(net, opt, cfg) if mode == "graph" else None
    sd, od = net.config["state_dim"], net.config["option_dim"]
    transpose = {"st": (sd, "bag_off" if net.config["trunk"] == "entity" else "st_off"), "e": (od, "e_off"), "ot": (od, "ot_off")} if TRANSPOSED_BAGS else None
    samples = packed_tensors(data.samples, dev)
    n_opts = torch.frombuffer(data.samples.n_opts, dtype=torch.int32)  # on the CPU: logit widths without asking the device
    actions = torch.frombuffer(array("q", data.actions), dtype=torch.long).to(dev)
    recorded = torch.stack([old_logp, adv, ret]).to(dev)
    totals = [0.0] * len(STATS)
    steps = 0
    stop = False
    for _ in range(cfg.epochs):
        order, chunks = trajectory_minibatches(data.lengths, cfg.minibatch, gen)
        bounds = list(accumulate((sum(c) for c in chunks), initial=0))
        rows = order.to(dev)
        big = structure(collate_packed(samples, rows), sd, od)
        acts, rec, width = actions[rows], recorded[:, rows], n_opts[order]
        acc = graphs.acc if graphs else torch.zeros(len(STATS), dtype=torch.float64, device=dev)
        if mode == "eager":
            for b, lens, lo, hi in zip(split(big, bounds), chunks, bounds, bounds[1:]):
                opt.zero_grad(set_to_none=True)
                loss, stats = _losses(net, cfg, b, lens, int(width[lo:hi].max()), acts[lo:hi], *rec[:, lo:hi])
                _step(net, opt, cfg, loss)
                acc += stats
        else:
            shapes, gru, ints, flts = _padded_epoch(big, bounds, chunks, acts, rec, width, graphs.choose if graphs else _need, transpose)
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
        if cfg.target_kl is not None and epoch[STATS.index("approx_kl")] / len(chunks) > cfg.target_kl:
            stop = True
            break
    if graphs:
        opt.zero_grad(set_to_none=True)  # the gradients live in the graphs' pool
    out = {k: v / max(steps, 1) for k, v in zip(STATS, totals)}
    out["updates"] = steps
    out["early_stop"] = stop
    out["explained_var"] = explained_var
    return out

