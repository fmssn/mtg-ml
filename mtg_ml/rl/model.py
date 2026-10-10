"""Recurrent policy/value network over hashed state features and scored options.

    state indices  --EmbeddingBag(sum)--> MLP --> s                (B, H)
                   (trunk="entity": global features summed, plus each
                    entity's features summed -> MLP -> entity vector,
                    entity_attn > 0: N pre-norm self-attention layers
                    over each sample's entity vectors; entity vectors
                    summed; then the MLP)
    event indices  --EmbeddingBag(mean)------> e                  (B, H)
    memory         z, h' = GRU([s, e], h)       (memory="gru")
                   z     = MLP([s, e])          (memory="none")
    core           c = s + z
                   (feature set 8: + belief_proj([archetype probs,
                    expected counts / cap]).detach(), zero at init)
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

Belief head (feature set 8, `PolicyNet(belief=BeliefSpec)`): `BeliefNet`
reads only the decision's witnessed-card evidence (`Batch.v_*`, flat
(card, slot, copies) triples of `rl.belief`), never the state, events or
options: per triple card + slot + count embeddings -> MLP, summed per
decision -> MLP -> archetype logits and per-card count logits (0..4 copies,
basic lands 0..75). The policy core gets a zero-initialised projection of
the detached (archetype probabilities, expected counts), so PPO gradients
never reach the belief branch and the supervised belief loss (`ppo.py`)
trains only it. `forward(..., aux=True)` returns the predictions as a
fourth item (`BeliefOut`, None without a belief head); the default return
contract is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from ..encode import BELIEF_FEATURES, check_features
from .belief import MAX_BASIC, MAX_NONBASIC, SLOTS, BeliefSpec
from .features import OPTION_DIM, STATE_DIM
from .samples import FIELDS, PackedSamples

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
    # The entity structure, precomputed by `structure` for training. When these
    # are None (rollouts, the inference server) the modules derive it with
    # boolean masks, each of which waits for the device.
    st_idx: torch.Tensor | None = None  # state tokens without the entity separators
    st_off: torch.Tensor | None = None  # (B,) offsets into st_idx
    bag_off: torch.Tensor | None = None  # (B + E,) bags of st_idx: each sample's global features, then each of its entities
    g_bag: torch.Tensor | None = None  # (B,) bag of each sample's global features
    e_bag: torch.Tensor | None = None  # (E,) bag of each entity
    e_row: torch.Tensor | None = None  # (E,) sample of each entity
    ot_idx: torch.Tensor | None = None  # option tokens without the entity pointers
    ot_off: torch.Tensor | None = None  # (N,) offsets into ot_idx
    p_opt: torch.Tensor | None = None  # (P,) option of each entity pointer
    p_ent: torch.Tensor | None = None  # (P,) entity it points at (row of the batch's entity vectors)
    # Padded training pieces only (`pad_split`): per token list ("st", "e",
    # "ot"), its tokens sorted by index, for the embedding weight gradient
    # (`_bag`): (bag of each sorted token, segment offsets, index of each
    # segment, mean weights or None).
    bag_t: dict | None = None
    # For entity self-attention (`EntityAttention`): the position of each
    # entity within its sample (-1: a padded entity), and an upper bound of the
    # entities per sample (`structure`: the epoch's maximum; padded pieces: the
    # padded shape's), which fixes the attention's (samples, width) layout.
    e_pos: torch.Tensor | None = None  # (E,)
    ent_width: int | None = None
    # Belief evidence (feature set 8): one (card, slot, copies) triple per
    # entry, level "v"; None: no evidence (the belief head then sees none).
    v_card: torch.Tensor | None = None  # (T,) belief vocabulary index
    v_slot: torch.Tensor | None = None  # (T,) 0 = current game, g = finished game g of the match
    v_cnt: torch.Tensor | None = None  # (T,) established copies
    v_off: torch.Tensor | None = None  # (B,) offsets into the triples

    def to(self, device) -> "Batch":
        return Batch(*(x.to(device) if torch.is_tensor(x) else x for x in (getattr(self, f) for f in self.__dataclass_fields__)))


def collate(samples) -> Batch:
    """samples: (state indices, [option indices, ...], event indices[, belief
    evidence triples]), or a `PackedSamples` (then all of it; see
    `collate_packed` for subsets). The batch carries evidence fields (empty
    for samples without)."""
    if isinstance(samples, PackedSamples):
        return collate_packed(samples, evidence=True)
    s_idx, s_off, e_idx, e_off, o_idx, o_off, o_row, o_pos, n_opts = [], [], [], [], [], [], [], [], []
    v_idx, v_off = [], []
    for row, (state, opts, events, *ev) in enumerate(samples):
        v_off.append(len(v_idx) // 3)
        v_idx.extend(ev[0] if ev else ())
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
    v = t(v_idx).view(-1, 3)
    return Batch(t(s_idx), t(s_off), t(e_idx), t(e_off), t(o_idx), t(o_off), t(o_row), t(o_pos), t(n_opts), v_card=v[:, 0], v_slot=v[:, 1], v_cnt=v[:, 2], v_off=t(v_off))


def step_batch(xs: list) -> tuple[Batch, int]:
    """Step-mode input of decisions xs = (state, option lengths, option
    tokens, events[, belief evidence]) as int32 arrays (`rollout._batch`
    plus the evidence fields). Returns (Batch, most options of a decision)."""
    ps = PackedSamples()
    for x in xs:
        ps.append(x)
    return collate_packed(ps, evidence=True), max(ps.n_opts)


def _excl(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    if x.numel() > 1:
        torch.cumsum(x[:-1], 0, out=out[1:])
    return out


def _segments(flat: torch.Tensor, start: torch.Tensor, length: torch.Tensor):
    """Concatenate flat[start[i] : start[i] + length[i]]; returns (values, offsets)."""
    off = _excl(length)
    idx = torch.repeat_interleave(start - off, length) + torch.arange(int(length.sum()), device=flat.device)
    return flat[idx], off


def packed_tensors(ps: PackedSamples, device=None) -> dict:
    """The fields of `ps` as long tensors plus the start of every segment:
    cached on the CPU, or built on `device` from the int32 arrays (half the
    bytes to copy). `collate_packed` takes either."""
    if device is None or torch.device(device).type == "cpu":
        if ps._cache is None:
            ps._cache = _with_starts({f: _ints(getattr(ps, f)).long() for f in FIELDS})
        return ps._cache
    return _with_starts({f: _ints(getattr(ps, f)).to(device).long() for f in FIELDS})


def _ints(a) -> torch.Tensor:
    return torch.frombuffer(a, dtype=torch.int32) if len(a) else torch.zeros(0, dtype=torch.int32)


def _with_starts(t: dict) -> dict:
    t["s_start"], t["e_start"], t["opt_start"], t["o_start"] = _excl(t["s_len"]), _excl(t["e_len"]), _excl(t["n_opts"]), _excl(t["o_len"])
    if "v_len" in t:
        t["v_start"] = _excl(t["v_len"])
    return t


def collate_packed(ps: PackedSamples | dict, idx=None, evidence: bool = False) -> Batch:
    """Batch of samples `idx` (default: all, in order) of a PackedSamples, or
    of its `packed_tensors` on a device (then built there). evidence: also
    the belief evidence fields (default None: what networks without a
    belief head read; `collate` and `step_batch` include them)."""
    t = ps if isinstance(ps, dict) else packed_tensors(ps)
    ev = {}
    if evidence and "v_len" in t:
        if idx is None:
            v_v, v_off = t["v_idx"], _excl(t["v_len"])
        else:
            rows_ = torch.as_tensor(idx, dtype=torch.long, device=t["n_opts"].device)
            v_v, v_off = _segments(t["v_idx"], t["v_start"][rows_], t["v_len"][rows_])
        v = v_v.view(-1, 3)
        ev = {"v_card": v[:, 0], "v_slot": v[:, 1], "v_cnt": v[:, 2], "v_off": torch.div(v_off, 3, rounding_mode="floor")}
    dev = t["n_opts"].device
    if idx is None:
        no = t["n_opts"]
        s_v, s_off = t["s_idx"], _excl(t["s_len"])
        e_v, e_off = t["e_idx"], _excl(t["e_len"])
        o_v, o_off = t["o_idx"], _excl(t["o_len"])
    else:
        rows = torch.as_tensor(idx, dtype=torch.long, device=dev)
        no = t["n_opts"][rows]
        s_v, s_off = _segments(t["s_idx"], t["s_start"][rows], t["s_len"][rows])
        e_v, e_off = _segments(t["e_idx"], t["e_start"][rows], t["e_len"][rows])
        opt_ids = torch.repeat_interleave(t["opt_start"][rows] - _excl(no), no) + torch.arange(int(no.sum()), device=dev)
        o_v, o_off = _segments(t["o_idx"], t["o_start"][opt_ids], t["o_len"][opt_ids])
    o_row = torch.repeat_interleave(torch.arange(no.shape[0], device=dev), no)
    o_pos = torch.arange(o_row.shape[0], device=dev) - torch.repeat_interleave(_excl(no), no)
    return Batch(s_v, s_off, e_v, e_off, o_v, o_off, o_row, o_pos, no.clone(), **ev)


def structure(b: Batch, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM) -> Batch:
    """`b` with the entity structure filled in: the state tokens split at the
    separators (`state_dim`) into each sample's global bag and entity bags,
    and the option tokens split into hashed tokens and entity pointers
    (`option_dim + k`). The same derivation as the mask-based paths of
    `EntityEncoder` and `PolicyNet._options`, with the same host syncs, so it
    is meant for a whole epoch at once; `split` then cuts minibatches out of
    it without any."""
    dev = b.s_idx.device
    B, N = b.s_off.shape[0], b.o_off.shape[0]
    s_idx, o_idx = b.s_idx.long(), b.o_idx.long()
    s_len = _lengths(b.s_off, s_idx.shape[0])
    rows = torch.repeat_interleave(torch.arange(B, device=dev), s_len)
    sep = (s_idx == state_dim).long()
    csum = torch.cumsum(sep, 0)
    seg = csum - (csum - sep)[b.s_off.long()][rows]  # entity number within the sample, 0 = global
    n_ent = torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, rows, sep)
    keep = sep == 0
    g_bag = torch.arange(B, device=dev) + _excl(n_ent)  # bags in token order: a sample's global features, then its entities
    bag = (g_bag[rows] + seg)[keep]
    E, width = torch.stack([n_ent.sum(), n_ent.max()]).tolist() if B else (0, 0)  # one wait for both
    e_row = torch.repeat_interleave(torch.arange(B, device=dev), n_ent, output_size=E)
    ptr = o_idx >= option_dim
    opt = torch.repeat_interleave(torch.arange(N, device=dev), _lengths(b.o_off, o_idx.shape[0]))
    p_opt = opt[ptr]
    return replace(
        b,
        st_idx=s_idx[keep],
        st_off=_excl(s_len - n_ent),
        bag_off=_excl(torch.zeros(B + E, dtype=torch.long, device=dev).index_add_(0, bag, torch.ones_like(bag))),
        g_bag=g_bag,
        e_bag=torch.arange(E, device=dev) + e_row + 1,
        e_row=e_row,
        ot_idx=o_idx[~ptr],
        ot_off=_excl(torch.zeros(N, dtype=torch.long, device=dev).index_add_(0, opt, (~ptr).long())),
        p_opt=p_opt,
        p_ent=_excl(n_ent)[b.o_row[p_opt]] + o_idx[ptr] - option_dim,
        e_pos=torch.arange(E, device=dev) - _excl(n_ent)[e_row],
        ent_width=width,
    )


# Batch fields: the level they run over (decisions, state tokens, options...)
# and the level their values are positions in (None: plain values). `split`
# slices every field to a range of decisions and re-bases the positions.
_FIELD_LEVELS = {
    "s_idx": ("s", None), "s_off": ("row", "s"), "e_idx": ("e", None), "e_off": ("row", "e"),
    "o_idx": ("o", None), "o_off": ("opt", "o"), "o_row": ("opt", "row"), "o_pos": ("opt", None), "n_opts": ("row", None),
    "st_idx": ("st", None), "st_off": ("row", "st"), "bag_off": ("bag", "st"), "g_bag": ("row", "bag"), "e_bag": ("ent", "bag"), "e_row": ("ent", "row"),
    "ot_idx": ("ot", None), "ot_off": ("opt", "ot"), "p_opt": ("ptr", "opt"), "p_ent": ("ptr", "ent"), "e_pos": ("ent", None),
    "v_card": ("v", None), "v_slot": ("v", None), "v_cnt": ("v", None), "v_off": ("row", "v"),
}  # fmt: skip
EVIDENCE_FIELDS = ("v_card", "v_slot", "v_cnt", "v_off")


def _level_sizes(b: Batch) -> dict:
    """Per decision: how many items of each level it owns."""
    B, dev = b.n_opts.shape[0], b.n_opts.device
    per_row = lambda rows, x: torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, rows, x)  # noqa: E731
    n_ent = per_row(b.e_row, torch.ones_like(b.e_row))
    out = {
        "row": torch.ones_like(b.n_opts), "s": _lengths(b.s_off, b.s_idx.shape[0]), "e": _lengths(b.e_off, b.e_idx.shape[0]),
        "opt": b.n_opts, "o": per_row(b.o_row, _lengths(b.o_off, b.o_idx.shape[0])), "st": _lengths(b.st_off, b.st_idx.shape[0]),
        "bag": n_ent + 1, "ent": n_ent, "ot": per_row(b.o_row, _lengths(b.ot_off, b.ot_idx.shape[0])), "ptr": per_row(b.o_row[b.p_opt], torch.ones_like(b.p_opt)),
    }  # fmt: skip
    if b.v_off is not None:
        out["v"] = _lengths(b.v_off, b.v_card.shape[0])
    return out


def _piece_starts(b: Batch, bounds: list[int]) -> tuple[dict, torch.Tensor]:
    """(level -> row of `at`, at): where the pieces [bounds[k], bounds[k + 1])
    of a `structure`d batch start at every level, (levels, pieces + 1)."""
    sizes = _level_sizes(b)
    cum = torch.stack(list(sizes.values())).cumsum(1)  # (levels, B): a scan along dim 0 of (B, levels) is ~100x slower on CUDA
    at = torch.cat([cum.new_zeros(len(sizes), 1), cum], 1)[:, torch.tensor(bounds, device=cum.device)]  # (levels, pieces + 1) starts
    return {k: i for i, k in enumerate(sizes)}, at


def split(b: Batch, bounds: list[int]) -> list[Batch]:
    """The decisions [bounds[k], bounds[k + 1]) (all of them) of a
    `structure`d batch as batches of their own: slices, with positions
    re-based to start at 0 (one subtraction per field for all pieces). Waits
    for the device here, for the sizes; using the pieces never does (each
    keeps the whole batch's `ent_width`, a bound for every piece)."""
    level, at = _piece_starts(b, bounds)
    n = at[:, 1:] - at[:, :-1]
    counts = n.tolist()
    fields = {}
    for name, (lvl, ref) in _FIELD_LEVELS.items():
        x = getattr(b, name)
        if x is None:  # no evidence
            continue
        if ref is not None:
            x = x - torch.repeat_interleave(at[level[ref], :-1], n[level[lvl]], output_size=x.shape[0])
        fields[name] = x.split(counts[level[lvl]])
    return [Batch(**{name: parts[k] for name, parts in fields.items()}, ent_width=b.ent_width) for k in range(len(bounds) - 1)]


# -- padded pieces, for a training step captured in a CUDA graph -----------------
#
# A graph replays fixed shapes, so every minibatch of an epoch is padded to the
# same size at every level. Padding never reaches a real decision: padded
# rows, options, entities, pointers and bags point only at padding (in turn,
# so that no index_add piles onto one row: atomics on one address serialise),
# there is always at least one padded row, entity and bag, every padded row
# gets an option, so its logits are finite, and the loss weighs padded rows
# with 0, so they add exactly 0 to every gradient. The padded tokens of the
# flat token lists are spread evenly over the padded bags (rows, options) and
# over 64 indices, for the same reason.

PAD_TOKENS = 64
TRANSPOSE_CHUNK = 32  # tokens per bag of the transposed embedding gradient (`_TransposedBag`)
PAD_FIELDS = {  # the fields the structured forward reads: (level, level its values are positions in, value at padded positions)
    "st_idx": ("st", None, "token"), "st_off": ("row", "st", "spread:st"), "bag_off": ("bag", "st", "spread:st"),
    "g_bag": ("row", "bag", "cycle:bag"), "e_bag": ("ent", "bag", "cycle:bag"), "e_row": ("ent", "row", "cycle:row"),
    "e_idx": ("e", None, "token"), "e_off": ("row", "e", "spread:e"), "ot_idx": ("ot", None, "token"), "ot_off": ("opt", "ot", "spread:ot"),
    "o_row": ("opt", "row", "cycle:row"), "o_pos": ("opt", None, "zero"), "p_opt": ("ptr", "opt", "cycle:opt"), "p_ent": ("ptr", "ent", "cycle:ent"),
}  # fmt: skip
PAD_EVIDENCE = {  # belief evidence (`pad_split` of a batch that carries it): padded triples are (0, 0, 0) in padded rows
    "v_card": ("v", None, "zero"), "v_slot": ("v", None, "zero"), "v_cnt": ("v", None, "zero"), "v_off": ("row", "v", "spread:v"),
}  # fmt: skip
_PADDED_LEVELS = ("st", "e", "ot", "ptr", "v")  # token levels padded to at least one more than any piece holds


def bucket(x: int, per_octave: int = 8) -> int:
    """x rounded up to one of `per_octave` sizes per power of two (8: at
    most 14% more), so that padded sizes repeat across epochs and a captured
    graph is reused."""
    q = 1 << max(int(x).bit_length() - 1 - (per_octave.bit_length() - 1), 0)
    return -(-int(x) // q) * q


def pad_sizes(counts: dict[str, list[int]]) -> dict[str, int]:
    """Padded size of every level for pieces with these item counts (per
    level, per piece): room for every piece plus at least one padded row,
    entity and bag, and an option for every padded row."""
    row, ent = bucket(max(counts["row"]) + 1), bucket(max(counts["ent"]) + 1)
    opt = bucket(max(o + row - r for o, r in zip(counts["opt"], counts["row"])))
    entw = {"entw": bucket(max(max(counts["entw"]), 1), 4)} if "entw" in counts else {}  # entity attention: entities per sample
    return {"row": row, "ent": ent, "bag": row + ent, "opt": opt, **entw, **{k: bucket(max(counts[k]) + 1) for k in counts if k in _PADDED_LEVELS or k.startswith("u_")}}


def pad_fits(sizes: dict[str, int], counts: dict[str, list[int]]) -> bool:
    """Whether pieces with these counts fit padded `sizes` (of other pieces)."""
    return (
        sizes["bag"] == sizes["row"] + sizes["ent"]
        and all(sizes.get(k, 0) > max(counts[k]) for k in counts if k in ("row", "ent") + _PADDED_LEVELS or k.startswith("u_"))
        and sizes["opt"] >= max(o + sizes["row"] - r for o, r in zip(counts["opt"], counts["row"]))
        and ("entw" not in counts or sizes.get("entw", 0) >= max(counts["entw"]))
    )


def pad_split(b: Batch, bounds: list[int], sizes=pad_sizes, transpose: dict | None = None, ent_width: bool = False) -> tuple[dict, dict, dict]:
    """The pieces of `split(b, bounds)`, padded to `sizes(counts)` (counts:
    level -> items per piece): {field: (pieces, size of its level)} for the
    fields of PAD_FIELDS, positions re-based to each piece. Also returns the
    sizes and the counts. Waits for the counts; a few kernels per field for
    all pieces.

    transpose: {token list: (vocabulary size, its bag offsets field)}, of
    "st" (bag_off or st_off), "e" (e_off), "ot" (ot_off): also the inputs of
    `_TransposedBag` for these lists, fields t_<list>_bag (level of the
    tokens), t_<list>_off and t_<list>_uid (level u_<list>: chunks of one
    index per piece, counted in counts), and t_e_w (float, mean weights).

    ent_width (entity attention): also field e_pos (level ent; -1 at padded
    entities) and counts["entw"], the most entities of a sample per piece,
    which `sizes` turns into sizes["entw"].

    A batch with belief evidence (`b.v_off`) also gets the PAD_EVIDENCE
    fields (level "v")."""
    level, at = _piece_starts(b, bounds)
    M, dev = len(bounds) - 1, at.device
    piece = torch.arange(M, device=dev)
    n = at[:, 1:] - at[:, :-1]
    cnt = {k: n[i] for k, i in level.items()}  # level -> items per piece, on the device
    start = {k: at[i] for k, i in level.items()}
    lists, extra = {}, {}
    for key, (vocab, off_name) in (transpose or {}).items():  # tokens sorted by (piece, index): one sort for the epoch
        tok_lvl = {"st": "st", "e": "e", "ot": "ot"}[key]
        bag_lvl = PAD_FIELDS[off_name][0]
        idx, off = getattr(b, f"{key}_idx").long(), getattr(b, off_name)
        T = idx.shape[0]
        lens = _lengths(off, T)
        bag = torch.repeat_interleave(torch.arange(off.shape[0], device=dev), lens, output_size=T)
        own = torch.repeat_interleave(piece, cnt[tok_lvl], output_size=T)
        kt = torch.int32 if M * (vocab + 1) < 2**31 else torch.long  # radix sort: half the passes
        skey, perm = torch.sort(own.to(kt) * (vocab + 1) + idx.to(kt), stable=True)
        new = torch.ones(T, dtype=torch.bool, device=dev)
        new[1:] = skey[1:] != skey[:-1]
        pos = torch.arange(T, device=dev)
        seg_at = pos[new]
        new |= (pos - seg_at[torch.cumsum(new, 0) - 1]) % TRANSPOSE_CHUNK == 0  # chunks of at most TRANSPOSE_CHUNK tokens
        seg_own = own[new]
        u = torch.zeros(M, dtype=torch.long, device=dev).index_add_(0, seg_own, torch.ones_like(seg_own))
        cnt[f"u_{key}"], start[f"u_{key}"] = u, torch.cat([u.new_zeros(1), u.cumsum(0)])
        sbag = bag[perm]
        lists[key] = (tok_lvl, bag_lvl, vocab, sbag - start[bag_lvl][own], (torch.arange(T, device=dev) - start[tok_lvl][own])[new], idx[perm][new])
        if off_name == "e_off":
            extra[f"t_{key}_w"] = 1.0 / lens[sbag].float()
    if ent_width:  # the most entities of one sample, per piece
        per_row = torch.zeros(b.n_opts.shape[0], dtype=torch.long, device=dev).index_add_(0, b.e_row, torch.ones_like(b.e_row))
        row_piece = torch.repeat_interleave(piece, cnt["row"], output_size=per_row.shape[0])
        cnt["entw"] = torch.zeros(M, dtype=torch.long, device=dev).scatter_reduce_(0, row_piece, per_row, "amax")
    keys = list(cnt)
    counts = dict(zip(keys, torch.stack([cnt[k] for k in keys]).tolist()))
    sizes = sizes(counts)

    where = {}  # level -> (piece of each item, its position in the padded (pieces, size) layout)

    def place(x, lvl, pad, rebase=None):
        D, nl = sizes[lvl], cnt[lvl][:, None]
        if lvl not in where:
            own = torch.repeat_interleave(piece, cnt[lvl], output_size=x.shape[0])
            where[lvl] = own, own * D + torch.arange(x.shape[0], device=dev) - start[lvl][own]
        own, dest = where[lvl]
        if rebase is not None:
            x = x - start[rebase][own]
        j = torch.arange(D, device=dev).expand(M, D)
        if pad == "zero":
            grid = torch.zeros(M, D, dtype=x.dtype, device=dev)
        elif pad == "token":
            grid = j % PAD_TOKENS
        elif pad.startswith("spread:"):  # the padded bags split the padded tokens evenly
            tok = cnt[pad[7:]][:, None]
            grid = tok + torch.div((j - nl) * (sizes[pad[7:]] - tok), D - nl, rounding_mode="floor")
        elif pad.startswith("cycle:"):  # the k-th padded item points at padded item k of that level, cycling
            to = cnt[pad[6:]][:, None]
            grid = to + torch.remainder(j - nl, sizes[pad[6:]] - to)
        else:  # a constant
            grid = torch.full((M, D), int(pad), dtype=torch.long, device=dev)
        return grid.reshape(-1).index_copy(0, dest, x).view(M, D)

    padded = PAD_FIELDS | (PAD_EVIDENCE if b.v_off is not None else {})
    out = {name: place(getattr(b, name).long(), lvl, pad, ref) for name, (lvl, ref, pad) in padded.items()}
    for key, (tok_lvl, bag_lvl, vocab, sbag, seg_at, uid) in lists.items():
        out[f"t_{key}_bag"] = place(sbag, tok_lvl, f"cycle:{bag_lvl}")  # padded tokens to padded bags: zero gradient
        out[f"t_{key}_off"] = place(seg_at, f"u_{key}", f"spread:{tok_lvl}")
        out[f"t_{key}_uid"] = place(uid, f"u_{key}", str(vocab))  # padded segments: the scratch row
    for name, w in extra.items():
        out[name] = place(w, "e", "zero")
    if ent_width:
        out["e_pos"] = place(b.e_pos.long(), "ent", "-1")
    return out, sizes, counts


def padded_batch(f: dict, ent_width: int | None = None) -> Batch:
    """A Batch of one padded piece ({field: 1-D tensor} of `pad_split`); the
    fields the structured forward does not read are None. ent_width: the
    padded shape's entities per sample (entity attention, with f["e_pos"])."""
    bag_t = {k: (f[f"t_{k}_bag"], f[f"t_{k}_off"], f[f"t_{k}_uid"], f.get(f"t_{k}_w")) for k in ("st", "e", "ot") if f"t_{k}_bag" in f}
    return Batch(
        None, None, f["e_idx"], f["e_off"], None, None, f["o_row"], f["o_pos"], None, **{k: f[k] for k in PAD_FIELDS if k not in ("e_idx", "e_off", "o_row", "o_pos")},
        bag_t=bag_t or None, e_pos=f.get("e_pos"), ent_width=ent_width, **{k: f.get(k) for k in PAD_EVIDENCE},
    )  # fmt: skip


class SequenceLayout(NamedTuple):
    """Sequence mode with static shapes: the GRU runs on a (sequences, steps)
    batch; row i goes to position pos_in[i] of it flattened (padded rows to
    a junk position, sequences * steps, after it) and reads its output from
    pos_out[i]."""

    pos_in: torch.Tensor
    pos_out: torch.Tensor
    sequences: int
    steps: int


def pad_sequences(chunks: list[list[int]], rows: int, gru: list[tuple[int, int]]):
    """GRU layouts of padded pieces: chunks[k] are the trajectory lengths of
    piece k, laid end to end in its first rows of `rows`, run as a gru[k] =
    (sequences, steps) batch. Returns (pos_in, pos_out), (pieces, rows) CPU
    tensors (see SequenceLayout)."""
    L = torch.tensor([n for c in chunks for n in c], dtype=torch.long)
    k = torch.tensor([len(c) for c in chunks], dtype=torch.long)
    T, S = torch.tensor(gru, dtype=torch.long).T
    M, total = len(chunks), int(L.sum())
    seq_piece = torch.repeat_interleave(torch.arange(M), k)
    seq = torch.arange(L.shape[0]) - _excl(k)[seq_piece]  # trajectory number within its piece
    pos = torch.repeat_interleave(seq * S[seq_piece] - _excl(L), L) + torch.arange(total)  # trajectory * steps + step
    row_piece = torch.repeat_interleave(seq_piece, L)
    dest = row_piece * rows + torch.arange(total) - _excl(torch.tensor([sum(c) for c in chunks], dtype=torch.long))[row_piece]
    junk = (T * S)[:, None]
    pos_in = junk.expand(M, rows).reshape(-1).index_copy(0, dest, pos).view(M, rows)
    pos_out = (torch.arange(rows) % junk).reshape(-1).index_copy(0, dest, pos).view(M, rows)  # padded rows: any output, in turn (they get no gradient)
    return pos_in, pos_out


TRUNKS = ("mlp", "transformer", "entity")


def _lengths(off: torch.Tensor, total: int) -> torch.Tensor:
    return torch.diff(off, append=torch.tensor([total], device=off.device, dtype=off.dtype)).long()


def _keep(idx: torch.Tensor, off: torch.Tensor, keep: torch.Tensor):
    """(idx, off) with the tokens where `keep` is False removed."""
    rows = torch.repeat_interleave(torch.arange(off.shape[0], device=idx.device), _lengths(off, idx.shape[0]))
    n = torch.zeros(off.shape[0], dtype=torch.long, device=idx.device).index_add_(0, rows[keep], torch.ones_like(rows[keep]))
    return idx[keep], _excl(n)


class _TransposedBag(torch.autograd.Function):
    """embedding_bag whose weight gradient is one more embedding bag: over
    the bag gradients, with the tokens sorted by index (`pad_split`
    precomputes that once per epoch) in chunks of at most TRANSPOSE_CHUNK
    tokens of one index, each chunk's sum added to its index's row. (One bag
    per index would be walked serially: common features occur ~10^4 times
    in a minibatch.) PyTorch's backward sorts the tokens and reduces segments
    in every step, ~3x the forward's time. Chunks with index = vocabulary
    size are padding and land in a scratch row."""

    @staticmethod
    def forward(ctx, weight, idx, off, mean, t_bag, t_off, t_uid, t_w):
        ctx.save_for_backward(t_bag, t_off, t_uid, t_w)
        ctx.rows = weight.shape[0]
        return nn.functional.embedding_bag(idx, weight, off, mode="mean" if mean else "sum")

    @staticmethod
    def backward(ctx, g):
        t_bag, t_off, t_uid, t_w = ctx.saved_tensors
        seg = nn.functional.embedding_bag(t_bag, g.contiguous(), t_off, mode="sum", per_sample_weights=t_w)
        dw = g.new_zeros(ctx.rows + 1, g.shape[1]).index_add_(0, t_uid, seg)
        return dw[:-1], None, None, None, None, None, None, None


def _bag(b: Batch, key: str, weight: torch.Tensor, idx: torch.Tensor, off: torch.Tensor, mean: bool = False) -> torch.Tensor:
    """Bags of `idx` (sum, or mean) of embedding rows `weight`; with the
    transposed weight gradient when `b` carries it."""
    t = b.bag_t.get(key) if b.bag_t else None
    if t is None:
        return nn.functional.embedding_bag(idx, weight, off, mode="mean" if mean else "sum")
    return _TransposedBag.apply(weight, idx, off, mean, *t)


ENTITY_ATTN_HEADS = 4
ENTITY_ATTN_FFN = 2  # feed-forward width, times hidden


class _AttentionBlock(nn.Module):
    """A pre-norm transformer encoder layer (no dropout) whose residual
    branches end in zero-initialised linears, so the block starts as the
    identity: bit-exactly, since x + 0 == x. A model with new blocks then
    computes what the model without them did (fine-tuning from a checkpoint
    without attention, `load_partial`), and a fresh model starts as the
    deep-sets encoder it extends."""

    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden {hidden} is not divisible by {heads} heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, ffn), nn.ReLU(), nn.Linear(ffn, hidden))
        for last in (self.out, self.ff[2]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, slot: torch.Tensor, keep: torch.Tensor, rows: int, width: int) -> torch.Tensor:
        """x (E, H): flat entity vectors; slot (E,) their places in the
        (rows, width) layout (rows * width: none); keep (rows, 1, 1, width)
        bool: the keys each place attends to. Only the attention itself runs
        on the layout: the norms, projections and feed-forward run on the
        entities (~2.5x fewer rows than places at width = the most entities)."""
        E, H = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.new_zeros(rows * width + 1, 3 * H).index_copy(0, slot, qkv)[:-1].view(rows, width, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep).transpose(1, 2).reshape(rows * width, H)
        x = x + self.out(torch.cat([a, a.new_zeros(1, H)]).index_select(0, slot))
        return x + self.ff(self.norm2(x))


class EntityAttention(nn.Module):
    """`layers` self-attention blocks over the entity vectors of each sample
    (permanents and stack items see each other; samples never mix). For the
    attention the flat (E, H) entity vectors go to a (samples, width) layout
    with a key padding mask, and back; the layout's shape comes from the
    batch's sizes (`Batch.ent_width`), never from its values, so a padded
    batch keeps its static shapes (CUDA graphs). Padded entities (position
    -1) attend to nothing and are attended to by nothing."""

    def __init__(self, hidden: int, layers: int, heads: int = ENTITY_ATTN_HEADS, ffn: int = ENTITY_ATTN_FFN):
        super().__init__()
        self.layers = nn.ModuleList(_AttentionBlock(hidden, heads, ffn * hidden) for _ in range(layers))

    def forward(self, ents: torch.Tensor, e_row: torch.Tensor, e_pos: torch.Tensor, rows: int, width: int) -> torch.Tensor:
        if ents.shape[0] == 0:
            return ents
        W = max(int(width), 1)
        slot = torch.where(e_pos >= 0, e_row * W + e_pos, rows * W)  # padded entities: one junk place past the layout
        keep = torch.zeros(rows * W + 1, dtype=torch.bool, device=ents.device).index_fill(0, slot, True)[:-1].view(rows, W)
        keep = keep | (torch.arange(W, device=ents.device) == 0)  # a sample without entities attends to one zero place (all keys masked would be NaN); nothing reads it
        keep = keep[:, None, None, :]
        for blk in self.layers:
            ents = blk(ents, slot, keep, rows, W)
        return ents


class EntityEncoder(nn.Module):
    """State = global features, then entity segments each opened by the
    separator `dim` (rl/features.py). Global features are summed; each
    entity's features are summed and passed through an MLP; with
    `attn_layers`, self-attention over each sample's entities
    (`EntityAttention`); entity vectors are summed into the state (a
    deep-sets encoder, so counts survive) and kept so options can point at
    them."""

    def __init__(self, dim: int, hidden: int, attn_layers: int = 0):
        super().__init__()
        self.sep = dim
        self.emb = nn.Embedding(dim, hidden)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.ent = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.attn = EntityAttention(hidden, attn_layers) if attn_layers else None

    def forward(self, b: Batch):
        """Returns (state sum (B, H), entity vectors (E, H), first entity row
        of each sample (B,), None when `b` carries the precomputed structure)."""
        if b.bag_off is not None:  # one bag per sample's globals and per entity, no masks
            bags = _bag(b, "st", self.emb.weight, b.st_idx, b.bag_off)
            ents = self.ent(bags.index_select(0, b.e_bag))
            if self.attn is not None:
                B = b.g_bag.shape[0]
                e_pos, width = b.e_pos, b.ent_width
                if e_pos is None or width is None:  # a structure without them: derive (waits for the device)
                    n_ent = torch.zeros(B, dtype=torch.long, device=ents.device).index_add_(0, b.e_row, torch.ones_like(b.e_row))
                    e_pos = torch.arange(ents.shape[0], device=ents.device) - _excl(n_ent)[b.e_row] if e_pos is None else e_pos
                    width = int(n_ent.max()) if B else 0
                ents = self.attn(ents, b.e_row, e_pos, B, width)
            ents = ents.to(bags.dtype)
            return bags.index_select(0, b.g_bag).index_add(0, b.e_row, ents), ents, None
        idx, off = b.s_idx, b.s_off
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
        if self.attn is not None:
            ents = self.attn(ents, owner, torch.arange(E, device=dev) - base[owner], B, int(n_ent.max()) if B else 0)
        ents = ents.to(g.dtype)
        return g.index_add(0, owner, ents), ents, base


VALUE_NETS = ("shared", "separate")
VALUE_BOUNDS = ("none", "tanh")


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

    def __init__(self, hidden: int, memory: str, trunk: str, state_dim: int, option_dim: int, entity_attn: int = 0):
        super().__init__()
        self.hidden = hidden
        self.memory = memory
        self.sep = state_dim
        self.entities = None  # (entity vectors, first row per sample) of the last forward, trunk="entity"
        self.seq_bf16 = False  # sequence mode (the PPO update) runs the GRU in BF16: set by `ppo.ppo_update` from PPOConfig.gru_precision
        if trunk == "transformer":
            self.state = TokenEncoder(state_dim, hidden)
        elif trunk == "entity":
            self.state = EntityEncoder(state_dim, hidden, entity_attn)
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
        # Recurrent accumulation stays FP32 even when dense/attention work
        # is autocast. In particular cuDNN's BF16 GRU support varies by shape.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = self.gru.weight_ih_l0.dtype
            return self._recurrent(x.to(dtype), None if hidden is None else hidden.to(dtype), lengths)

    def _gru_seq(self, xp: torch.Tensor) -> torch.Tensor:
        """The GRU over a padded batch (sequences, steps, features) from zero state. With `seq_bf16`
        cuDNN runs it in BF16 (weights cast per call, so the gradients reach the FP32 parameters;
        ~2x faster at h256, where the FP32 kernels are launch-latency bound); the output is FP32."""
        if not self.seq_bf16 or not xp.is_cuda:
            return self.gru(xp)[0]
        w = [p.to(torch.bfloat16) for p in self.gru._flat_weights]
        out = torch._VF.gru(xp.to(torch.bfloat16), torch.zeros(1, xp.shape[0], self.hidden, device=xp.device, dtype=torch.bfloat16), w, True, 1, 0.0, self.training, False, True)[0]
        return out.to(xp.dtype)

    def _recurrent(self, x, hidden, lengths):
        if lengths is None:  # step mode
            h0 = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=x.dtype) if hidden is None else hidden
            out, hn = self.gru(x[:, None], h0[None])
            return out[:, 0], hn[0]
        # Sequence mode. Padding the trajectories at the end is exact for a
        # GRU that starts from zero: outputs at real steps never see the
        # padding, and the padding's outputs get no gradient. Up to hidden
        # 128 cuDNN runs a padded batch with persistent kernels (H100: 0.7 ms
        # forward and backward for 2k decisions, against 5 ms packed, which
        # launches kernels per time step); above that it has none, and padded
        # is ~15% slower than packed (twice the rows; measured at 256 and
        # 512). On the CPU padded is faster at every size.
        if isinstance(lengths, SequenceLayout):  # static shapes (captured step): always padded
            pos_in, pos_out, T, S = lengths
            xp = x.new_zeros(T * S + 1, x.shape[1]).index_copy(0, pos_in, x)[:-1].view(T, S, -1)
            return self._gru_seq(xp).reshape(-1, self.hidden).index_select(0, pos_out), None
        if x.is_cuda and self.hidden > 128:
            rows, batch_sizes = _sequence_layout(lengths, x.device, packed=True)
            y = self.gru(PackedSequence(x.index_select(0, rows), batch_sizes))[0].data
            return y.new_empty(y.shape).index_copy(0, rows, y), None
        pos, steps = _sequence_layout(lengths, x.device, packed=False)
        xp = x.new_zeros(len(lengths) * steps, x.shape[1]).index_copy(0, pos, x).view(len(lengths), steps, -1)
        return self._gru_seq(xp).reshape(-1, self.hidden).index_select(0, pos), None

    def forward(self, b: Batch, hidden, lengths):
        if self.trunk_kind == "entity":
            g, ents, base = self.state(b)
            s = self.trunk(g)
            self.entities = (ents, base)
        else:  # these trunks treat entity features as part of the flat bag
            idx, off = (b.st_idx, b.st_off) if b.st_idx is not None else _keep(b.s_idx, b.s_off, b.s_idx != self.sep)
            s = self.state(idx, off) if self.trunk_kind == "transformer" else self.trunk(_bag(b, "st", self.state_emb.weight, idx, off))
        e = _bag(b, "e", self.event_emb.weight, b.e_idx, b.e_off, mean=True)
        z, hn = self._memory(torch.cat([s, e], dim=-1), hidden, lengths)
        return s + z, hn


def _sequence_layout(lengths: list[int], device: torch.device, packed: bool):
    """Sequence mode: where the rows of trajectories laid end to end go in
    the GRU's input, built on the CPU and copied without waiting for the
    device, so the GRU needs one gather or scatter each way.
    packed=False: (position in a (trajectories, longest) batch of each row, longest).
    packed=True: (row of each position of a PackedSequence, time-major and
    longest first; its batch sizes, on the CPU as cuDNN takes them)."""
    L = torch.tensor(lengths)
    if packed:
        by_len = torch.sort(L, descending=True, stable=True).indices
        sizes = (L[by_len][None, :] > torch.arange(max(lengths))[:, None]).sum(1)  # trajectories still running at each step
        t = torch.repeat_interleave(torch.arange(sizes.shape[0]), sizes)
        idx, layout = _excl(L)[by_len][torch.arange(t.shape[0]) - _excl(sizes)[t]] + t, sizes
    else:
        layout = max(lengths)
        idx = torch.arange(int(L.sum())) + torch.repeat_interleave(torch.arange(len(lengths)) * layout - _excl(L), L)
    if device.type == "cuda":
        idx = idx.pin_memory().to(device, non_blocking=True)
    return idx, layout


class BeliefOut(NamedTuple):
    """Belief head predictions per decision. arch (B, archetypes) logits;
    nonbasic (B, nonbasic cards, MAX_NONBASIC + 1) and basic (B, basic
    lands, MAX_BASIC + 1, or None without any) count logits, cards in the
    order of `BeliefNet.nonbasic_idx` / `basic_idx` (vocabulary indices)."""

    arch: torch.Tensor
    nonbasic: torch.Tensor
    basic: torch.Tensor | None


DEFAULT_BELIEF_COEF = {"archetype": 0.1, "counts": 0.1}


class BeliefNet(nn.Module):
    """The belief encoder and heads. Reads only the evidence triples: per
    triple card (exact vocabulary index), slot and copy-count embeddings,
    summed, ReLU, Linear, ReLU; summed per decision; Linear, ReLU, Linear,
    ReLU; then the archetype head and the per-card count heads."""

    def __init__(self, spec: BeliefSpec, hidden: int):
        super().__init__()
        V, A = len(spec.vocab), len(spec.archetypes)
        self.hidden = hidden
        self.card = nn.Embedding(V, hidden)
        self.slot = nn.Embedding(SLOTS, hidden)
        self.count = nn.Embedding(MAX_BASIC + 1, hidden)
        for e in (self.card, self.slot, self.count):
            nn.init.normal_(e.weight, std=0.05)
        self.tok = nn.Linear(hidden, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.arch = nn.Linear(hidden, A)
        basic = spec.basic
        nb, bs = [i for i in range(V) if not basic[i]], [i for i in range(V) if basic[i]]
        # Persistent buffers: `rollout.load_net` builds on the meta device and assigns the state dict.
        self.register_buffer("nonbasic_idx", torch.tensor(nb, dtype=torch.long))
        self.register_buffer("basic_idx", torch.tensor(bs, dtype=torch.long))
        caps = torch.tensor([MAX_BASIC if b else MAX_NONBASIC for b in basic], dtype=torch.float32)
        self.register_buffer("caps", caps)
        self.nonbasic = nn.Linear(hidden, len(nb) * (MAX_NONBASIC + 1))
        self.basic = nn.Linear(hidden, len(bs) * (MAX_BASIC + 1)) if bs else None
        self.n_vocab, self.n_arch = V, A

    def forward(self, b: Batch, rows: int) -> BeliefOut:
        if b.v_off is None or b.v_card.shape[0] == 0:
            pooled = self.tok.weight.new_zeros(rows, self.hidden)
        else:
            card, slot, cnt = b.v_card.long(), b.v_slot.long(), b.v_cnt.long().clamp(max=MAX_BASIC)
            x = torch.relu(self.tok(torch.relu(self.card(card) + self.slot(slot) + self.count(cnt))))
            n = _lengths(b.v_off, card.shape[0])
            row = torch.repeat_interleave(torch.arange(rows, device=x.device), n, output_size=card.shape[0])
            pooled = x.new_zeros(rows, self.hidden).index_add(0, row, x)
        h = self.trunk(pooled)
        nb = self.nonbasic(h).view(rows, -1, MAX_NONBASIC + 1)
        bs = self.basic(h).view(rows, -1, MAX_BASIC + 1) if self.basic is not None else None
        return BeliefOut(self.arch(h), nb, bs)

    def summary(self, out: BeliefOut) -> torch.Tensor:
        """(archetype probabilities, expected copies / range per vocabulary
        card): (B, archetypes + vocabulary), what the policy reads."""
        probs = torch.softmax(out.arch.float(), -1)
        exp = torch.zeros(probs.shape[0], self.n_vocab, device=probs.device, dtype=probs.dtype)
        exp = exp.index_copy(1, self.nonbasic_idx, (torch.softmax(out.nonbasic.float(), -1) * torch.arange(MAX_NONBASIC + 1, device=probs.device)).sum(-1))
        if out.basic is not None:
            exp = exp.index_copy(1, self.basic_idx, (torch.softmax(out.basic.float(), -1) * torch.arange(MAX_BASIC + 1, device=probs.device)).sum(-1))
        return torch.cat([probs, exp / self.caps], 1)


class PolicyNet(nn.Module):
    """hidden: policy width. trunk: "mlp" (EmbeddingBag + MLP),
    "transformer" (2 layers over the active state features) or "entity"
    (deep sets over permanents and stack items; options point at entities). value_net:
    "shared" (a linear value head on the policy core) or "separate" (its own
    embeddings, trunk and memory of width value_hidden, no shared gradients;
    Andrychowicz et al. 2020 found separate, wider value networks better).
    The recurrent state of a separate value net is appended to the policy's,
    so `state_size` = hidden (+ value_hidden). value_bound "tanh" squashes
    the value into (-1, 1), the range of the terminal rewards (no weights:
    any checkpoint loads either way); it is in `config` only when set, so
    configs of unbounded nets are unchanged. entity_attn (trunk "entity"
    only): self-attention layers over each sample's entity vectors, in every
    core, before they are pooled and before options point at them (4 heads,
    feed-forward 2 * width; 0: none, and the config and weights of the model
    without them). features: the feature-set version the network reads
    (`encode.FEATURE_VERSIONS`; no weights depend on it, the hashed inputs
    do): 1, the default and what a config without the key means, or 2;
    in `config` only when not 1, so older configs are unchanged.

    belief (feature set 8, required there and only there): a `BeliefSpec`
    (or its `to_config()`), which adds the belief head (`BeliefNet`,
    `belief_net.*`) and its zero-initialised projection into the policy
    core (`belief_proj.*`). `net.belief` is the spec (None without). In
    `config` as `belief` (the spec's config) and `belief_coef` (the
    auxiliary loss coefficients, {"archetype", "counts"}, default 0.1 each,
    read by `ppo_update`). A set-7 checkpoint loads into a belief network
    with `load_partial` (the new weights keep their init; the projection is
    zero, so the network computes what the checkpoint did)."""

    def __init__(
        self,
        hidden: int = 128,
        memory: str = "gru",
        state_dim: int = STATE_DIM,
        option_dim: int = OPTION_DIM,
        trunk: str = "mlp",
        value_net: str = "shared",
        value_hidden: int = 0,
        value_bound: str = "none",
        entity_attn: int = 0,
        features: int = 1,
        belief: BeliefSpec | dict | None = None,
        belief_coef: dict | None = None,
    ):
        super().__init__()
        check_features(features)
        if isinstance(belief, dict):
            belief = BeliefSpec.from_config(belief)
        if (belief is not None) != (features >= BELIEF_FEATURES):
            raise ValueError(f"feature set {features}: a belief head (BeliefSpec) is {'required' if features >= BELIEF_FEATURES else 'only for feature sets >= ' + str(BELIEF_FEATURES)}")
        if memory not in MEMORY_KINDS or trunk not in TRUNKS or value_net not in VALUE_NETS or value_bound not in VALUE_BOUNDS:
            raise ValueError(f"memory in {MEMORY_KINDS}, trunk in {TRUNKS}, value_net in {VALUE_NETS}, value_bound in {VALUE_BOUNDS}")
        if entity_attn < 0 or (entity_attn and trunk != "entity"):
            raise ValueError(f"entity_attn ({entity_attn}) needs trunk 'entity' and must be >= 0")
        value_hidden = value_hidden or hidden
        self.config = {"hidden": hidden, "memory": memory, "state_dim": state_dim, "option_dim": option_dim, "trunk": trunk, "value_net": value_net, "value_hidden": value_hidden}
        if value_bound != "none":
            self.config["value_bound"] = value_bound
        if entity_attn:  # only then: checkpoints without attention stay readable by code that predates it
            self.config["entity_attn"] = entity_attn
        if features != 1:
            self.config["features"] = features
        self.belief = belief
        if belief is not None:
            self.config["belief"] = belief.to_config()
            self.config["belief_coef"] = dict(DEFAULT_BELIEF_COEF, **(belief_coef or {}))
        self.features = features
        self.value_bound = value_bound
        self.hidden = hidden
        self.memory = memory
        self.value_net = value_net
        self.policy_core = _Core(hidden, memory, trunk, state_dim, option_dim, entity_attn)
        self.option_emb = nn.EmbeddingBag(option_dim, hidden, mode="sum")
        nn.init.normal_(self.option_emb.weight, std=0.05)
        self.option_mlp = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.option_dim = option_dim
        self.pointer = nn.Linear(hidden, hidden, bias=False) if trunk == "entity" else None
        self.scorer = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if value_net == "separate":
            self.value_core = _Core(value_hidden, memory, trunk, state_dim, option_dim, entity_attn)
            self.value_head = nn.Sequential(nn.Linear(value_hidden, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))
            last = self.value_head[-1]
        else:
            self.value_head = nn.Linear(hidden, 1)
            last = self.value_head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.value_hidden = value_hidden if value_net == "separate" else 0
        self.state_size = hidden + self.value_hidden
        if belief is not None:
            self.belief_net = BeliefNet(belief, hidden)
            self.belief_proj = nn.Linear(len(belief.archetypes) + len(belief.vocab), hidden)
            nn.init.zeros_(self.belief_proj.weight)
            nn.init.zeros_(self.belief_proj.bias)
        else:
            self.belief_net = self.belief_proj = None

    def initial_state(self, n: int = 1) -> torch.Tensor:
        return torch.zeros(n, self.state_size)

    def forward(self, b: Batch, hidden: torch.Tensor | None = None, lengths: list[int] | SequenceLayout | None = None, max_options: int | None = None, aux: bool = False):
        """Returns (logits (B, max_opts), values (B,), new hidden (B, state_size) or None).
        `max_options` (= n_opts.max()) can be passed to avoid a device sync.
        Sequence mode takes `lengths` or, for padded pieces, a SequenceLayout.
        aux: also the belief head's predictions, a fourth item (`BeliefOut`,
        None without a belief head)."""
        hp = hv = None
        if hidden is not None:
            hp, hv = hidden[:, : self.hidden].contiguous(), hidden[:, self.hidden :].contiguous()
        c, hn = self.policy_core(b, hp, lengths)
        bel = None
        if self.belief_net is not None:
            bel = self.belief_net(b, c.shape[0])
            c = c + self.belief_proj(self.belief_net.summary(bel).detach()).to(c.dtype)  # PPO gradients stop here
        if self.value_net == "separate":
            cv, hvn = self.value_core(b, hv, lengths)
            values = self.value_head(cv).squeeze(-1)
            if hn is not None:
                hn = torch.cat([hn, hvn], dim=-1)
        else:
            values = self.value_head(c).squeeze(-1)
        if self.value_bound == "tanh":
            values = torch.tanh(values)
        a = self.option_mlp(self._options(b))
        cr = c.index_select(0, b.o_row)
        scores = self.scorer(torch.cat([cr, a, cr * a], dim=-1)).squeeze(-1)
        width = int(b.n_opts.max()) if max_options is None else max_options
        logits = torch.full((c.shape[0], width), float("-inf"), device=c.device, dtype=scores.dtype)
        logits[b.o_row, b.o_pos] = scores
        return (logits, values, hn, bel) if aux else (logits, values, hn)

    def _options(self, b: Batch) -> torch.Tensor:
        """Summed option token embeddings, plus the projected vectors of the
        entities an option points at (tokens >= option_dim)."""
        if b.ot_idx is not None:  # pointers already split off (`structure`), no masks
            a = _bag(b, "ot", self.option_emb.weight, b.ot_idx, b.ot_off)
            if self.pointer is not None:
                a = a.index_add(0, b.p_opt, self.pointer(self.policy_core.entities[0].index_select(0, b.p_ent)).to(a.dtype))
            return a
        ptr = b.o_idx >= self.option_dim
        idx, off = _keep(b.o_idx, b.o_off, ~ptr)
        a = self.option_emb(idx, off)
        if self.pointer is not None and self.policy_core.entities is not None:
            ents, base = self.policy_core.entities
            opt = torch.repeat_interleave(torch.arange(b.o_off.shape[0], device=a.device), _lengths(b.o_off, b.o_idx.shape[0]))[ptr]
            k = b.o_idx[ptr].long() - self.option_dim
            a = a.index_add(0, opt, self.pointer(ents[base[b.o_row[opt]] + k]).to(a.dtype))
        return a

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts checkpoints from before the core refactor (state_emb, trunk,
        event_emb, gru/mix at the top level)."""
        old = ("state_emb.", "trunk.", "event_emb.", "gru.", "mix.")
        if any(k.startswith(old) for k in state_dict):
            state_dict = {("policy_core." + k if k.startswith(old) else k): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


NEW_LAYER_KEYS = (".state.attn.", "belief_net.", "belief_proj.")  # weights a checkpoint may lack: `load_partial` leaves them at their init (identity, or a zero belief projection)


def load_partial(net: PolicyNet, state_dict: dict) -> list[str]:
    """Load a checkpoint's weights into `net` when `net` only adds layers
    that start as the identity (entity attention) or add zero (the belief
    head behind its zero projection: upgrading a set-7 checkpoint to set 8):
    every weight of the checkpoint must fit, and the weights it lacks must all be such layers.
    Returns the names of those weights, which keep their initialisation, so
    `net` computes what the checkpoint's model did."""
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    bad = [k for k in missing if not any(s in k for s in NEW_LAYER_KEYS)]
    if bad or unexpected:
        raise ValueError(f"checkpoint does not fit the model: missing {bad}, unexpected {list(unexpected)}")
    return list(missing)


def masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp.masked_fill(torch.isinf(logits), 0.0)).sum(-1)
