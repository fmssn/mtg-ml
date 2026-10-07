"""Step-mode forward of many policies in one pass: the inference server's fast path.

The server keeps every policy it serves (opponent-pool checkpoints and the
learner, all of one config) in a `PolicyStack`:

* each embedding table is one big table with a block of rows per policy
  slot, plus one zero row at the end that entity separators, entity
  pointers and padding look up (so they add nothing to a bag, which is
  what removing them does in `PolicyNet`);
* each dense layer is a (slots, out, in) weight. The rows of a batch are
  sorted by policy, hence so are its entities and options, and a dense
  layer is one grouped matmul over contiguous runs (`grouped_linear`: a
  Triton kernel on CUDA, a loop over the runs elsewhere).

`step_forward` takes a `StepInput`: the step-mode arrays of a batch padded
to fixed sizes (every size is known on the host), derives the entity
structure with fixed-shape ops (no host syncs, no boolean indexing), runs
the network, samples one option per decision (Gumbel-max) with its
log-prob, and updates the hidden-state table in place. So all of it can be
captured in one CUDA graph per padded shape.

Padding conventions (the caller guarantees them): the last row is padding
(R_p > rows), and so is the last option (N_p > options); padded tokens,
options and entities belong to the last row; padded rows have the policy
of the last real row (the order stays sorted), zero lengths and the dummy
hidden slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

try:  # the CUDA kernel; CPU (and CUDA without Triton) use the loop
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the install
    triton = None

STACKABLE_TRUNKS = ("mlp", "entity")


# ---------------------------------------------------------------------------
# Grouped linear layer
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _grouped_linear_kernel(X, W, Bias, Y, POL, M, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Y[i] = X[i] @ W[POL[i]] (+ Bias[POL[i]]) for rows sorted by POL, W
        (slots, K, N): a tile of rows loops over the few policies it spans,
        loading only the rows of the current one (fp32, IEEE products: no
        TF32)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        valid = rm < M
        pol = tl.load(POL + rm, mask=valid, other=0)
        lo = tl.min(tl.where(valid, pol, 1 << 30))
        hi = tl.max(tl.where(valid, pol, -1))
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for q in range(lo, hi + 1):
            sel = valid & (pol == q)
            for k0 in range(0, K, BK):
                kk = k0 + rk
                a = tl.load(X + rm[:, None] * K + kk[None, :], mask=sel[:, None] & (kk[None, :] < K), other=0.0)
                b = tl.load(W + q * N * K + kk[:, None] * N + rn[None, :], mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
                acc += tl.dot(a, b, input_precision="ieee")
            if HAS_BIAS:
                bias = tl.load(Bias + q * N + rn, mask=rn < N, other=0.0)
                acc += tl.where(sel[:, None], bias[None, :], 0.0)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc, mask=valid[:, None] & (rn[None, :] < N))


if triton is not None:
    # An opaque op for torch.compile: the compiled forward fuses around it.

    @torch.library.custom_op("mtg::grouped_linear", mutates_args=())
    def _grouped_linear_op(x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor | None, pol: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, K = x.shape
        N = wt.shape[2]
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
        # Tiles measured on an H100 (h128, 26 policies): small batches need many programs.
        BM, BN, BK, warps = (16, 32, 32, 2) if M < 4096 else (32, 128, 32, 4)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _grouped_linear_kernel[grid](x, wt, wt if b is None else b, y, pol, M, N, K, b is not None, BM, BN, BK, num_warps=warps)
        return y

    @_grouped_linear_op.register_fake
    def _(x, wt, b, pol):
        return x.new_empty(x.shape[0], wt.shape[2])


def grouped_linear(x: torch.Tensor, w, b: torch.Tensor | None, pol: torch.Tensor) -> torch.Tensor:
    """y[i] = x[i] @ w[pol[i]].T + b[pol[i]] for `pol` non-decreasing; w is
    (slots, out, in) as nn.Linear stores it, or (that, its (slots, in, out)
    transpose) as `PolicyStack.lin` holds it; b (slots, out)."""
    w, wt = w if isinstance(w, tuple) else (w, None)
    M, K = x.shape
    N = w.shape[1]
    if N == 1:  # a dot product per row: gather the weight row
        y = (x * w[pol, 0]).sum(1, keepdim=True)
        return y if b is None else y + b[pol]
    if x.is_cuda and triton is not None:
        if wt is None:
            wt = w.transpose(1, 2).contiguous()
        return torch.ops.mtg.grouped_linear(x, wt, b, pol)
    # Host loop over the runs of equal policy (syncs once on CUDA; fine on the CPU).
    y = x.new_empty(M, N)
    vals, counts = torch.unique_consecutive(pol, return_counts=True)
    start = 0
    for q, n in zip(vals.tolist(), counts.tolist()):
        y[start : start + n] = F.linear(x[start : start + n], w[q], None if b is None else b[q])
        start += n
    return y


# ---------------------------------------------------------------------------
# Stacked weights
# ---------------------------------------------------------------------------


def _table_dims(config: dict) -> dict:
    """Embedding tables of a PolicyNet state dict: name -> rows per policy."""
    sd, od = config["state_dim"], config["option_dim"]
    cores = ["policy_core"] + (["value_core"] if config["value_net"] == "separate" else [])
    out = {"option_emb.weight": od}
    for c in cores:
        out[f"{c}.state.emb.weight" if config["trunk"] == "entity" else f"{c}.state_emb.weight"] = sd
        out[f"{c}.event_emb.weight"] = od
    return out


class PolicyStack:
    """The weights of up to `capacity` policies of one config, stacked.
    Slot k holds one policy; `load(k, net)` copies a PolicyNet in place
    (CUDA graphs captured over the stack stay valid)."""

    def __init__(self, config: dict, capacity: int, device):
        if not PolicyStack.supports(config):
            raise ValueError(f"config {config} is not stackable (trunks {STACKABLE_TRUNKS}, no entity attention)")
        from .model import PolicyNet

        self.config = dict(config)
        self.capacity = capacity
        self.device = torch.device(device)
        with torch.device("meta"):
            shapes = {k: v.shape for k, v in PolicyNet(**config).state_dict().items()}
        self.dims = _table_dims(config)
        self.tables: dict[str, torch.Tensor] = {}
        self.dense: dict[str, torch.Tensor] = {}
        self.wt: dict[str, torch.Tensor] = {}  # (slots, in, out) copies of the matrices: the kernel reads them along `out`
        for name, shape in shapes.items():
            if name in self.dims:  # rows of slot k: [k * dim, (k + 1) * dim); the last row stays zero
                self.tables[name] = torch.zeros(capacity * self.dims[name] + 1, shape[1], device=self.device)
            else:
                self.dense[name] = torch.zeros(capacity, *shape, device=self.device)
                if len(shape) == 2 and shape[0] > 1:
                    self.wt[name] = torch.zeros(capacity, shape[1], shape[0], device=self.device)
        # what the forward reads: matrices as (w, transposed w), the rest as is
        self.lin = {k: (v, self.wt[k]) if k in self.wt else v for k, v in self.dense.items()}

    @staticmethod
    def supports(config: dict) -> bool:
        """Entity self-attention needs a (rows, entities per row) layout whose
        width the server's padded shapes do not carry yet: those policies
        run on the server's eager per-policy path."""
        return config.get("trunk") in STACKABLE_TRUNKS and not config.get("entity_attn")

    def load(self, slot: int, net) -> None:
        if net.config != self.config:
            raise ValueError("a stack holds policies of one config")
        sd = net.state_dict()
        with torch.no_grad():
            for name, t in self.tables.items():
                d = self.dims[name]
                t[slot * d : (slot + 1) * d].copy_(sd[name], non_blocking=True)
            for name, t in self.dense.items():
                t[slot].copy_(sd[name], non_blocking=True)
            for name, t in self.wt.items():
                t[slot].copy_(self.dense[name][slot].t())

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in [*self.tables.values(), *self.dense.values()])


# ---------------------------------------------------------------------------
# The forward pass
# ---------------------------------------------------------------------------


@dataclass
class StepInput:
    """A padded step-mode batch, rows sorted by policy (see module doc).
    Every tensor is int64 or bool on the stack's device."""

    pol: torch.Tensor  # (R_p,) policy slot per row, non-decreasing
    gslot: torch.Tensor  # (R_p,) row of the hidden-state table
    fresh: torch.Tensor  # (R_p,) bool: start of a sequence (zero state)
    s_len: torch.Tensor  # (R_p,) state tokens per row
    e_len: torch.Tensor  # (R_p,) event tokens per row
    n_opt: torch.Tensor  # (R_p,) options per row
    s_tok: torch.Tensor  # (S_p,) state tokens (separator = state_dim)
    s_row: torch.Tensor  # (S_p,) row of each token
    s_valid: torch.Tensor  # (S_p,) bool
    ev_tok: torch.Tensor  # (Ev_p,)
    ev_valid: torch.Tensor  # (Ev_p,) bool
    o_len: torch.Tensor  # (N_p,) tokens per option
    o_row: torch.Tensor  # (N_p,) row of each option
    ot_tok: torch.Tensor  # (O_p,) option tokens (pointer = option_dim + entity)
    ot_opt: torch.Tensor  # (O_p,) option of each token
    ot_valid: torch.Tensor  # (O_p,) bool
    n_ent: int  # E_p: padded entity count (>= entities of the batch)


def excl_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, 0) - x


def segments(lengths: torch.Tensor, size: int):
    """For positions 0..size-1 laid out as consecutive segments of
    `lengths`: (segment of each position, offset within it). Positions past
    the end fall in the last segment (the caller's padding)."""
    ends = torch.cumsum(lengths, 0)
    pos = torch.arange(size, device=lengths.device)
    seg = torch.searchsorted(ends, pos, right=True).clamp_(max=lengths.shape[0] - 1)
    return seg, pos - (ends - lengths)[seg]


PAD_CHUNK = 64


def _sum_into(out: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """out[idx[i]] += src[i] along dim 0, with atomics (scatter_add: compiled
    index_add becomes a sort-based index_put, ~1 ms per call)."""
    if src.dim() == 2:
        idx = idx.unsqueeze(1).expand(-1, src.shape[1])
    return out.scatter_add_(0, idx, src)


def _bags(bag: torch.Tensor, total: torch.Tensor, n_bags: int):
    """Offsets for an embedding bag whose first `total` tokens go to `bag`
    (sorted) and whose padding (the tail) goes to extra bags of PAD_CHUNK
    tokens after the n_bags real ones: one bag with all the padding would be
    summed by one thread block, milliseconds at 100k tokens. The ids are
    sorted, so each offset is a binary search (no atomics)."""
    size = bag.shape[0]
    pad = torch.arange(size, device=bag.device) - total
    bag = torch.where(pad < 0, bag, n_bags + pad.clamp(min=0) // PAD_CHUNK)
    return torch.searchsorted(bag, torch.arange(n_bags + -(-size // PAD_CHUNK), device=bag.device))


def _bag_sum(idx, table, off, n_bags: int, mode: str = "sum"):
    return F.embedding_bag(idx, table, off, mode=mode)[:n_bags]


def _core(stack: PolicyStack, c: str, x: StepInput, st: dict, hidden: torch.Tensor | None):
    """One _Core (policy or value) on the batch: (c (R_p, H), new hidden or None)."""
    T, D = stack.tables, stack.lin
    pol = x.pol
    if stack.config["trunk"] == "entity":
        bags = _bag_sum(st["s_emb"], T[f"{c}.state.emb.weight"], st["bag_off"], st["n_bags"])
        e_pol = pol[st["e_row"]]
        ents = torch.relu(grouped_linear(torch.relu(bags.index_select(0, st["e_bag"])), D[f"{c}.state.ent.1.weight"], D[f"{c}.state.ent.1.bias"], e_pol))
        E_p = ents.shape[0]
        g = bags.index_select(0, st["g_bag"]) + _bag_sum(torch.arange(E_p, device=ents.device), ents, st["ent_off"], pol.shape[0])
    else:
        ents = None
        g = _bag_sum(st["s_emb"], T[f"{c}.state_emb.weight"], st["s_off"], pol.shape[0])
    h = torch.relu(grouped_linear(torch.relu(g), D[f"{c}.trunk.1.weight"], D[f"{c}.trunk.1.bias"], pol))
    s = torch.relu(grouped_linear(h, D[f"{c}.trunk.3.weight"], D[f"{c}.trunk.3.bias"], pol))
    e = _bag_sum(st["ev_emb"], T[f"{c}.event_emb.weight"], st["ev_off"], pol.shape[0], mode="mean")
    xe = torch.cat([s, e], 1)
    if stack.config["memory"] == "none":
        return s + torch.relu(grouped_linear(xe, D[f"{c}.mix.0.weight"], D[f"{c}.mix.0.bias"], pol)), None, ents
    H = s.shape[1]
    gi = grouped_linear(xe, D[f"{c}.gru.weight_ih_l0"], D[f"{c}.gru.bias_ih_l0"], pol)
    gh = grouped_linear(hidden, D[f"{c}.gru.weight_hh_l0"], D[f"{c}.gru.bias_hh_l0"], pol)
    r = torch.sigmoid(gi[:, :H] + gh[:, :H])
    z = torch.sigmoid(gi[:, H : 2 * H] + gh[:, H : 2 * H])
    n = torch.tanh(gi[:, 2 * H :] + r * gh[:, 2 * H :])
    hn = (1 - z) * n + z * hidden
    return s + hn, hn, ents


def structure(stack: PolicyStack, x: StepInput) -> dict:
    """Embedding indices and bag layout of the batch, fixed shapes only."""
    cfg = stack.config
    sd, od = cfg["state_dim"], cfg["option_dim"]
    cap = stack.capacity
    R_p, E_p = x.pol.shape[0], x.n_ent
    dev = x.pol.device
    st: dict = {}
    sep = (x.s_tok == sd) & x.s_valid
    zero_s, zero_o = cap * sd, cap * od  # the zero rows
    st["s_emb"] = torch.where(x.s_valid & ~sep, x.s_tok + x.pol[x.s_row] * sd, zero_s)
    if cfg["trunk"] == "entity":
        # Bags in token order: a row's global features, then each of its
        # entities (opened by a separator, which looks up the zero row).
        csum = torch.cumsum(sep.long(), 0)
        s_start = excl_cumsum(x.s_len)
        cx = torch.cat([csum.new_zeros(1), csum])
        n_ent = cx[s_start + x.s_len] - cx[s_start]  # separators per row
        ent_base = excl_cumsum(n_ent)
        base_t = ent_base[x.s_row]
        bag = torch.where(csum > base_t, x.s_row + csum, x.s_row + base_t)
        st["bag_off"] = _bags(bag, x.s_len.sum(), R_p + E_p)
        st["n_bags"] = R_p + E_p
        e_row, _ = segments(n_ent, E_p)
        st["e_row"] = e_row
        st["e_bag"] = e_row + torch.arange(E_p, device=dev) + 1
        st["g_bag"] = torch.arange(R_p, device=dev) + ent_base
        st["ent_off"] = _bags(e_row, n_ent.sum(), R_p)  # each row's entities, as bags over the entity vectors
        st["ent_base"] = ent_base
    else:
        st["s_off"] = _bags(x.s_row, x.s_len.sum(), R_p)
    ev_row = segments(x.e_len, x.ev_tok.shape[0])[0]
    st["ev_emb"] = torch.where(x.ev_valid, x.ev_tok + x.pol[ev_row] * od, zero_o)
    st["ev_off"] = _bags(ev_row, x.e_len.sum(), R_p)
    ptr = x.ot_tok >= od
    t_row = x.o_row[x.ot_opt]
    st["o_emb"] = torch.where(x.ot_valid & ~ptr, x.ot_tok + x.pol[t_row] * od, zero_o)
    st["o_off"] = _bags(x.ot_opt, x.o_len.sum(), x.o_len.shape[0])
    if cfg["trunk"] == "entity":  # pointer tokens: bags over the entity vectors, weight 1 (other tokens 0)
        is_ptr = x.ot_valid & ptr
        st["ptr_w"] = is_ptr.float()
        st["ptr_ent"] = torch.where(is_ptr, st["ent_base"][t_row] + x.ot_tok - od, 0).clamp_(max=E_p - 1)
    return st


def step_forward(stack: PolicyStack, x: StepInput, hidden_table: torch.Tensor | None, noise: torch.Tensor | None = None, write_hidden: bool = True):
    """Forward + sampling of a padded batch. Returns (action, log-prob,
    value), each (R_p,); writes the new hidden states into `hidden_table`
    at `x.gslot` (write_hidden=False: returns them as a fourth item, None
    without memory, for the caller to write). `noise` (N_p,) uniform in
    (0, 1) for the Gumbel-max draw (default: drawn here)."""
    cfg = stack.config
    D = stack.lin
    pol = x.pol
    st = structure(stack, x)
    H = cfg["hidden"]
    hp = hv = None
    if cfg["memory"] == "gru":
        h0 = torch.where(x.fresh.unsqueeze(1), 0.0, hidden_table.index_select(0, x.gslot))
        hp, hv = h0[:, :H], h0[:, H:]
    c, hn, ents = _core(stack, "policy_core", x, st, hp)
    if cfg["value_net"] == "separate":
        cv, hvn, _ = _core(stack, "value_core", x, st, hv)
        v = torch.relu(grouped_linear(cv, D["value_head.0.weight"], D["value_head.0.bias"], pol))
        values = grouped_linear(v, D["value_head.2.weight"], D["value_head.2.bias"], pol)[:, 0]
        if hn is not None:
            hn = torch.cat([hn, hvn], 1)
    else:
        values = grouped_linear(c, D["value_head.weight"], D["value_head.bias"], pol)[:, 0]
    if hn is not None:
        if write_hidden:
            hidden_table.index_copy_(0, x.gslot, hn)
    # options
    o_pol = pol[x.o_row]
    a = _bag_sum(st["o_emb"], stack.tables["option_emb.weight"], st["o_off"], x.o_len.shape[0])
    if cfg["trunk"] == "entity":  # pointer(sum of entities) = sum of pointer(entity): Linear without bias
        pe = F.embedding_bag(st["ptr_ent"], ents, st["o_off"], mode="sum", per_sample_weights=st["ptr_w"])[: a.shape[0]]
        a = a + grouped_linear(pe, D["pointer.weight"], None, o_pol)
    a = torch.relu(grouped_linear(torch.relu(a), D["option_mlp.1.weight"], D["option_mlp.1.bias"], o_pol))
    cr = c.index_select(0, x.o_row)
    hs = torch.relu(grouped_linear(torch.cat([cr, a, cr * a], 1), D["scorer.0.weight"], D["scorer.0.bias"], o_pol))
    scores = grouped_linear(hs, D["scorer.2.weight"], D["scorer.2.bias"], o_pol)[:, 0]
    # sample per row: Gumbel-max, log-prob = score - logsumexp over the row's options
    R_p, N_p = pol.shape[0], scores.shape[0]
    neg = torch.full((R_p,), float("-inf"), device=scores.device)
    m = neg.scatter_reduce(0, x.o_row, scores, "amax")
    lse = m + torch.log(_sum_into(torch.zeros_like(m), x.o_row, torch.exp(scores - m[x.o_row])))
    if noise is None:
        noise = torch.rand(N_p, device=scores.device)
    key = scores - torch.log(-torch.log(noise.clamp(min=1e-30)))
    km = neg.scatter_reduce(0, x.o_row, key, "amax")
    first = excl_cumsum(x.n_opt)
    pos = torch.arange(N_p, device=scores.device) - first[x.o_row]
    big = torch.full((R_p,), N_p, dtype=torch.long, device=scores.device)
    act = big.scatter_reduce(0, x.o_row, torch.where(key == km[x.o_row], pos, N_p), "amin")
    chosen = (first + act).clamp_(max=N_p - 1)
    logp = scores.index_select(0, chosen) - lse
    return (act, logp, values) if write_hidden else (act, logp, values, hn)


# ---------------------------------------------------------------------------
# Building a StepInput from decisions on the host (tests, reference)
# ---------------------------------------------------------------------------


def step_input(decisions: list, pols: list[int], gslots: list[int], fresh: list[bool], device="cpu", pad: int = 1, n_ent: int | None = None) -> StepInput:
    """A StepInput from decisions (state, option lengths, option tokens,
    events) whose `pols` are non-decreasing, with `pad` padded rows (>= 1)
    and one padded option. Padded rows use hidden slot gslots[-1] + 1."""
    from .features import STATE_DIM

    t = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
    s_len = [len(d[0]) for d in decisions] + [0] * pad
    e_len = [len(d[3]) for d in decisions] + [0] * pad
    n_opt = [len(d[1]) for d in decisions] + [0] * pad
    n_opt[-1] = 1  # the padded option belongs to the last row
    o_len = [n for d in decisions for n in d[1]] + [0]
    s_tok = t([v for d in decisions for v in d[0]])
    ev_tok = t([v for d in decisions for v in d[3]])
    ot_tok = t([v for d in decisions for v in d[2]])
    s_row, _ = segments(t(s_len), len(s_tok))
    o_row, _ = segments(t(n_opt), len(o_len))
    ot_opt, _ = segments(t(o_len), len(ot_tok))
    ents = sum(list(d[0]).count(STATE_DIM) for d in decisions)
    return StepInput(
        pol=t(list(pols) + [pols[-1]] * pad),
        gslot=t(list(gslots) + [max(gslots) + 1] * pad),
        fresh=t(list(fresh) + [True] * pad).bool(),
        s_len=t(s_len), e_len=t(e_len), n_opt=t(n_opt),
        s_tok=s_tok, s_row=s_row, s_valid=torch.ones_like(s_tok, dtype=torch.bool),
        ev_tok=ev_tok, ev_valid=torch.ones_like(ev_tok, dtype=torch.bool),
        o_len=t(o_len), o_row=o_row,
        ot_tok=ot_tok, ot_opt=ot_opt, ot_valid=torch.ones_like(ot_tok, dtype=torch.bool),
        n_ent=ents + 1 if n_ent is None else n_ent,
    )  # fmt: skip


__all__ = ["PolicyStack", "StepInput", "grouped_linear", "step_forward", "step_input", "structure", "segments"]
