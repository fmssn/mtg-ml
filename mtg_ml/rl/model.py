"""Recurrent policy/value network over hashed state features and scored options.

    state indices  --EmbeddingBag(sum)--> MLP --> s                (B, H)
                   (trunk="entity": global features summed, plus each
                    entity's features summed -> MLP -> entity vector,
                    entity vectors summed; then the MLP)
    event indices  --EmbeddingBag(mean)------> e                  (B, H)
    memory         z, h' = GRU([s, e], h)       (memory="gru")
                   z     = MLP([s, e])          (memory="none")
    core           c = s + z
    option indices --EmbeddingBag(sum)--> MLP --> a                (N, H)
                   (trunk="entity": plus a projection of the vector of
                    every entity the option points at)
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


TRUNKS = ("mlp", "transformer", "entity")


def _lengths(off: torch.Tensor, total: int) -> torch.Tensor:
    return torch.diff(off, append=torch.tensor([total], device=off.device, dtype=off.dtype)).long()


def _keep(idx: torch.Tensor, off: torch.Tensor, keep: torch.Tensor):
    """(idx, off) with the tokens where `keep` is False removed."""
    rows = torch.repeat_interleave(torch.arange(off.shape[0], device=idx.device), _lengths(off, idx.shape[0]))
    n = torch.zeros(off.shape[0], dtype=torch.long, device=idx.device).index_add_(0, rows[keep], torch.ones_like(rows[keep]))
    return idx[keep], _excl(n)


class EntityEncoder(nn.Module):
    """State = global features, then entity segments each opened by the
    separator `dim` (rl/features.py). Global features are summed; each
    entity's features are summed and passed through an MLP; entity vectors
    are summed into the state (a deep-sets encoder, so counts survive) and
    kept so options can point at them."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.sep = dim
        self.emb = nn.Embedding(dim, hidden)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.ent = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, idx: torch.Tensor, off: torch.Tensor):
        """Returns (state sum (B, H), entity vectors (E, H), first entity row of each sample (B,))."""
        B, dev = off.shape[0], idx.device
        idx = idx.long()
        rows = torch.repeat_interleave(torch.arange(B, device=dev), _lengths(off, idx.shape[0]))
        sep = idx == self.sep
        csum = torch.cumsum(sep.long(), 0)
        seg = csum - (csum - sep.long())[off.long()][rows]  # entity number within the sample, 0 = global
        n_ent = torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, rows, sep.long())
        base = _excl(n_ent)
        glob = ~sep & (seg == 0)
        inent = ~sep & (seg > 0)
        ones = lambda m: torch.ones_like(m, dtype=torch.long)  # noqa: E731
        g_n = torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, rows[glob], ones(rows[glob]))
        g = nn.functional.embedding_bag(idx[glob], self.emb.weight, _excl(g_n), mode="sum")
        E = int(n_ent.sum())
        ent_of = base[rows[inent]] + seg[inent] - 1  # tokens are already grouped by entity
        e_n = torch.zeros(E, dtype=torch.long, device=dev).index_add_(0, ent_of, ones(ent_of))
        ents = self.ent(nn.functional.embedding_bag(idx[inent], self.emb.weight, _excl(e_n), mode="sum"))
        owner = torch.repeat_interleave(torch.arange(B, device=dev), n_ent)
        return g.index_add(0, owner, ents), ents, base
VALUE_NETS = ("shared", "separate")


class TokenEncoder(nn.Module):
    """Transformer over the active state features of a decision (each hashed
    feature is a token, as in MageZero), masked mean-pooled to one vector."""

    def __init__(self, dim: int, hidden: int, layers: int = 2, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(dim, hidden)
        nn.init.normal_(self.emb.weight, std=0.05)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, idx: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        B = off.shape[0]
        lengths = torch.diff(off, append=torch.tensor([idx.shape[0]], device=off.device, dtype=off.dtype)).long()
        rows = torch.repeat_interleave(torch.arange(B, device=idx.device), lengths)
        pos = torch.arange(idx.shape[0], device=idx.device) - torch.repeat_interleave(off.long(), lengths)
        width = max(int(lengths.max()) if B else 1, 1)
        tok = torch.zeros(B, width, self.emb.embedding_dim, device=idx.device, dtype=self.emb.weight.dtype)
        tok[rows, pos] = self.emb(idx.long())
        pad = torch.ones(B, width, dtype=torch.bool, device=idx.device)
        pad[rows, pos] = False
        h = self.enc(tok, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        return self.out((h * keep).sum(1) / keep.sum(1).clamp(min=1))


class _Core(nn.Module):
    """State + events -> c, with optional recurrent memory over a player's decisions."""

    def __init__(self, hidden: int, memory: str, trunk: str, state_dim: int, option_dim: int):
        super().__init__()
        self.hidden = hidden
        self.memory = memory
        self.sep = state_dim
        self.entities = None  # (entity vectors, first row per sample) of the last forward, trunk="entity"
        if trunk == "transformer":
            self.state = TokenEncoder(state_dim, hidden)
        elif trunk == "entity":
            self.state = EntityEncoder(state_dim, hidden)
            self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        else:
            emb = nn.EmbeddingBag(state_dim, hidden, mode="sum")
            nn.init.normal_(emb.weight, std=0.05)
            self.state_emb = emb
            self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.trunk_kind = trunk
        self.event_emb = nn.EmbeddingBag(option_dim, hidden, mode="mean")
        nn.init.normal_(self.event_emb.weight, std=0.05)
        if memory == "gru":
            self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        else:
            self.mix = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())

    def _memory(self, x: torch.Tensor, hidden, lengths):
        if self.memory == "none":
            return self.mix(x), None
        if lengths is None:  # step mode
            h0 = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=x.dtype) if hidden is None else hidden
            out, hn = self.gru(x[:, None], h0[None])
            return out[:, 0], hn[0]
        seqs = list(torch.split(x, lengths))
        packed = pack_padded_sequence(pad_sequence(seqs, batch_first=True), torch.tensor(lengths), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        return torch.cat([out[i, :n] for i, n in enumerate(lengths)]), None

    def forward(self, b: Batch, hidden, lengths):
        if self.trunk_kind == "entity":
            g, ents, base = self.state(b.s_idx, b.s_off)
            s = self.trunk(g)
            self.entities = (ents, base)
        else:  # these trunks treat entity features as part of the flat bag
            idx, off = _keep(b.s_idx, b.s_off, b.s_idx != self.sep)
            s = self.state(idx, off) if self.trunk_kind == "transformer" else self.trunk(self.state_emb(idx, off))
        e = self.event_emb(b.e_idx, b.e_off)
        z, hn = self._memory(torch.cat([s, e], dim=-1), hidden, lengths)
        return s + z, hn


class PolicyNet(nn.Module):
    """hidden: policy width. trunk: "mlp" (EmbeddingBag + MLP),
    "transformer" (2 layers over the active state features) or "entity"
    (deep sets over permanents and stack items; options point at entities). value_net:
    "shared" (a linear value head on the policy core) or "separate" (its own
    embeddings, trunk and memory of width value_hidden, no shared gradients;
    Andrychowicz et al. 2020 found separate, wider value networks better).
    The recurrent state of a separate value net is appended to the policy's,
    so `state_size` = hidden (+ value_hidden)."""

    def __init__(
        self,
        hidden: int = 128,
        memory: str = "gru",
        state_dim: int = STATE_DIM,
        option_dim: int = OPTION_DIM,
        trunk: str = "mlp",
        value_net: str = "shared",
        value_hidden: int = 0,
    ):
        super().__init__()
        if memory not in MEMORY_KINDS or trunk not in TRUNKS or value_net not in VALUE_NETS:
            raise ValueError(f"memory in {MEMORY_KINDS}, trunk in {TRUNKS}, value_net in {VALUE_NETS}")
        value_hidden = value_hidden or hidden
        self.config = {"hidden": hidden, "memory": memory, "state_dim": state_dim, "option_dim": option_dim, "trunk": trunk, "value_net": value_net, "value_hidden": value_hidden}
        self.hidden = hidden
        self.memory = memory
        self.value_net = value_net
        self.policy_core = _Core(hidden, memory, trunk, state_dim, option_dim)
        self.option_emb = nn.EmbeddingBag(option_dim, hidden, mode="sum")
        nn.init.normal_(self.option_emb.weight, std=0.05)
        self.option_mlp = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.option_dim = option_dim
        self.pointer = nn.Linear(hidden, hidden, bias=False) if trunk == "entity" else None
        self.scorer = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if value_net == "separate":
            self.value_core = _Core(value_hidden, memory, trunk, state_dim, option_dim)
            self.value_head = nn.Sequential(nn.Linear(value_hidden, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))
            last = self.value_head[-1]
        else:
            self.value_head = nn.Linear(hidden, 1)
            last = self.value_head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.value_hidden = value_hidden if value_net == "separate" else 0
        self.state_size = hidden + self.value_hidden

    def initial_state(self, n: int = 1) -> torch.Tensor:
        return torch.zeros(n, self.state_size)

    def forward(self, b: Batch, hidden: torch.Tensor | None = None, lengths: list[int] | None = None, max_options: int | None = None):
        """Returns (logits (B, max_opts), values (B,), new hidden (B, state_size) or None).
        `max_options` (= n_opts.max()) can be passed to avoid a device sync."""
        hp = hv = None
        if hidden is not None:
            hp, hv = hidden[:, : self.hidden].contiguous(), hidden[:, self.hidden :].contiguous()
        c, hn = self.policy_core(b, hp, lengths)
        if self.value_net == "separate":
            cv, hvn = self.value_core(b, hv, lengths)
            values = self.value_head(cv).squeeze(-1)
            if hn is not None:
                hn = torch.cat([hn, hvn], dim=-1)
        else:
            values = self.value_head(c).squeeze(-1)
        a = self.option_mlp(self._options(b))
        cr = c[b.o_row]
        scores = self.scorer(torch.cat([cr, a, cr * a], dim=-1)).squeeze(-1)
        width = int(b.n_opts.max()) if max_options is None else max_options
        logits = torch.full((c.shape[0], width), float("-inf"), device=c.device, dtype=scores.dtype)
        logits[b.o_row, b.o_pos] = scores
        return logits, values, hn

    def _options(self, b: Batch) -> torch.Tensor:
        """Summed option token embeddings, plus the projected vectors of the
        entities an option points at (tokens >= option_dim)."""
        ptr = b.o_idx >= self.option_dim
        idx, off = _keep(b.o_idx, b.o_off, ~ptr)
        a = self.option_emb(idx, off)
        if self.pointer is not None and self.policy_core.entities is not None:
            ents, base = self.policy_core.entities
            opt = torch.repeat_interleave(torch.arange(b.o_off.shape[0], device=a.device), _lengths(b.o_off, b.o_idx.shape[0]))[ptr]
            k = b.o_idx[ptr].long() - self.option_dim
            a = a.index_add(0, opt, self.pointer(ents[base[b.o_row[opt]] + k]))
        return a

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts checkpoints from before the core refactor (state_emb, trunk,
        event_emb, gru/mix at the top level)."""
        old = ("state_emb.", "trunk.", "event_emb.", "gru.", "mix.")
        if any(k.startswith(old) for k in state_dict):
            state_dict = {("policy_core." + k if k.startswith(old) else k): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp.masked_fill(torch.isinf(logits), 0.0)).sum(-1)
