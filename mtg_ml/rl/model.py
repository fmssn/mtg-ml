"""Recurrent policy/value network over hashed state features and scored options.

    state indices  --EmbeddingBag(sum)--> MLP --> s                (B, H)
    event indices  --EmbeddingBag(mean)------> e                  (B, H)
    memory         z, h' = GRU([s, e], h)       (memory="gru")
                   z     = MLP([s, e])          (memory="none")
    core           c = s + z
    option indices --EmbeddingBag(sum)--> MLP --> a                (N, H)
    logit(option)  = MLP([c[row], a, c[row] * a])                 (N,)
    value          = Linear(c)                                    (B,)

The recurrent state runs over one player's own decisions in one game. Events
are what that player saw happen since their previous decision: their own
last choice and the opponent's public actions (`features.event_tokens`).
That is how memory reaches back to earlier turns and how a target or
payment decision knows which spell it belongs to.

Two calling modes:
  * step mode (rollouts): every row is one step of a different sequence,
    `hidden` is (B, H) and the new hidden state is returned;
  * sequence mode (training): rows are whole trajectories laid end to end,
    `lengths` gives their sizes, hidden states start at zero.

Logits are scattered into a (B, max_options) matrix and padding is masked
to -inf, so a softmax over a row is a distribution over exactly the legal
options of that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .features import OPTION_DIM, STATE_DIM
from .samples import PackedSamples

MEMORY_KINDS = ("gru", "none")


@dataclass
class Batch:
    s_idx: torch.Tensor  # flat state feature indices
    s_off: torch.Tensor  # (B,) offsets into s_idx
    e_idx: torch.Tensor  # flat event indices
    e_off: torch.Tensor  # (B,) offsets into e_idx
    o_idx: torch.Tensor  # flat option token indices
    o_off: torch.Tensor  # (N,) offsets into o_idx
    o_row: torch.Tensor  # (N,) decision each option belongs to
    o_pos: torch.Tensor  # (N,) position of the option within its decision
    n_opts: torch.Tensor  # (B,)

    def to(self, device) -> "Batch":
        return Batch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def collate(samples) -> Batch:
    """samples: (state indices, [option indices, ...], event indices), or a
    `PackedSamples` (then all of it; see `collate_packed` for subsets)."""
    if isinstance(samples, PackedSamples):
        return collate_packed(samples)
    s_idx, s_off, e_idx, e_off, o_idx, o_off, o_row, o_pos, n_opts = [], [], [], [], [], [], [], [], []
    for row, (state, opts, events) in enumerate(samples):
        s_off.append(len(s_idx))
        s_idx.extend(state)
        e_off.append(len(e_idx))
        e_idx.extend(events)
        n_opts.append(len(opts))
        for pos, toks in enumerate(opts):
            o_off.append(len(o_idx))
            o_idx.extend(toks)
            o_row.append(row)
            o_pos.append(pos)
    t = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    return Batch(t(s_idx), t(s_off), t(e_idx), t(e_off), t(o_idx), t(o_off), t(o_row), t(o_pos), t(n_opts))


def _excl(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    if x.numel() > 1:
        torch.cumsum(x[:-1], 0, out=out[1:])
    return out


def _segments(flat: torch.Tensor, start: torch.Tensor, length: torch.Tensor):
    """Concatenate flat[start[i] : start[i] + length[i]]; returns (values, offsets)."""
    off = _excl(length)
    idx = torch.repeat_interleave(start - off, length) + torch.arange(int(length.sum()))
    return flat[idx], off


def _packed_tensors(ps: PackedSamples) -> dict:
    if ps._cache is None:
        t = {}
        for f in ("s_len", "e_len", "n_opts", "o_len", "s_idx", "e_idx", "o_idx"):
            a = getattr(ps, f)
            t[f] = torch.frombuffer(a, dtype=torch.int32).long() if len(a) else torch.zeros(0, dtype=torch.long)
        t["s_start"], t["e_start"], t["opt_start"], t["o_start"] = _excl(t["s_len"]), _excl(t["e_len"]), _excl(t["n_opts"]), _excl(t["o_len"])
        ps._cache = t
    return ps._cache


def collate_packed(ps: PackedSamples, idx=None) -> Batch:
    """Batch of samples `idx` (default: all, in order) of a PackedSamples."""
    t = _packed_tensors(ps)
    if idx is None:
        no = t["n_opts"]
        s_v, s_off = t["s_idx"], _excl(t["s_len"])
        e_v, e_off = t["e_idx"], _excl(t["e_len"])
        o_v, o_off = t["o_idx"], _excl(t["o_len"])
    else:
        rows = torch.as_tensor(idx, dtype=torch.long)
        no = t["n_opts"][rows]
        s_v, s_off = _segments(t["s_idx"], t["s_start"][rows], t["s_len"][rows])
        e_v, e_off = _segments(t["e_idx"], t["e_start"][rows], t["e_len"][rows])
        opt_ids = torch.repeat_interleave(t["opt_start"][rows] - _excl(no), no) + torch.arange(int(no.sum()))
        o_v, o_off = _segments(t["o_idx"], t["o_start"][opt_ids], t["o_len"][opt_ids])
    o_row = torch.repeat_interleave(torch.arange(no.shape[0]), no)
    o_pos = torch.arange(o_row.shape[0]) - torch.repeat_interleave(_excl(no), no)
    return Batch(s_v, s_off, e_v, e_off, o_v, o_off, o_row, o_pos, no.clone())


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 128, memory: str = "gru", state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM):
        super().__init__()
        if memory not in MEMORY_KINDS:
            raise ValueError(f"memory must be one of {MEMORY_KINDS}")
        self.config = {"hidden": hidden, "memory": memory, "state_dim": state_dim, "option_dim": option_dim}
        self.hidden = hidden
        self.memory = memory
        self.state_emb = nn.EmbeddingBag(state_dim, hidden, mode="sum")
        self.event_emb = nn.EmbeddingBag(option_dim, hidden, mode="mean")
        self.option_emb = nn.EmbeddingBag(option_dim, hidden, mode="sum")
        self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if memory == "gru":
            self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        else:
            self.mix = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        self.option_mlp = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.state_emb.weight, std=0.05)
        nn.init.normal_(self.event_emb.weight, std=0.05)
        nn.init.normal_(self.option_emb.weight, std=0.05)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def initial_state(self, n: int = 1) -> torch.Tensor:
        return torch.zeros(n, self.hidden)

    def _memory(self, x: torch.Tensor, hidden, lengths):
        if self.memory == "none":
            return self.mix(x), None
        if lengths is None:  # step mode
            h0 = self.initial_state(x.shape[0]).to(x) if hidden is None else hidden
            out, hn = self.gru(x[:, None], h0[None])
            return out[:, 0], hn[0]
        seqs = list(torch.split(x, lengths))
        packed = pack_padded_sequence(pad_sequence(seqs, batch_first=True), torch.tensor(lengths), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return torch.cat([out[i, :n] for i, n in enumerate(lengths)]), None

    def forward(self, b: Batch, hidden: torch.Tensor | None = None, lengths: list[int] | None = None, max_options: int | None = None):
        """Returns (logits (B, max_opts), values (B,), new hidden (B, H) or None).
        `max_options` (= n_opts.max()) can be passed to avoid a device sync."""
        s = self.trunk(self.state_emb(b.s_idx, b.s_off))
        e = self.event_emb(b.e_idx, b.e_off)
        z, hn = self._memory(torch.cat([s, e], dim=-1), hidden, lengths)
        c = s + z
        a = self.option_mlp(self.option_emb(b.o_idx, b.o_off))
        cr = c[b.o_row]
        scores = self.scorer(torch.cat([cr, a, cr * a], dim=-1)).squeeze(-1)
        width = int(b.n_opts.max()) if max_options is None else max_options
        logits = torch.full((c.shape[0], width), float("-inf"), device=c.device, dtype=scores.dtype)
        logits[b.o_row, b.o_pos] = scores
        return logits, self.value_head(c).squeeze(-1), hn


def masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp.masked_fill(torch.isinf(logits), 0.0)).sum(-1)
