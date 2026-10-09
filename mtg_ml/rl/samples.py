"""Recorded network inputs, packed.

A rollout records one (state indices, [option indices, ...], event indices)
sample per learner decision. As Python lists of lists, a few hundred
thousand of them cost the trainer seconds to unpickle per iteration, in one
process: more than the workers need to play the games. `PackedSamples`
stores them as flat int32 arrays (lengths + values), which pickle as raw
bytes, and `rl.model.collate_packed` builds a training batch from any subset
with a handful of vectorized ops. It still behaves like a list of tuples
(`len`, indexing, iteration). No torch dependency: rollout workers use it.
"""

from __future__ import annotations

from array import array

FIELDS = ("s_len", "e_len", "n_opts", "o_len", "s_idx", "e_idx", "o_idx")


class PackedSamples:
    __slots__ = FIELDS + ("_starts", "_cache")

    def __init__(self):
        for f in FIELDS:
            setattr(self, f, array("i"))
        self._starts = None
        self._cache = None  # torch views for collate_packed

    # -- building ------------------------------------------------------------

    def append(self, sample) -> None:
        """sample: (state, opts, events) or the flat form (state, option
        lengths, option tokens, events) of `features.featurize_flat`."""
        if len(sample) == 4:
            state, o_len, o_flat, events = sample
            self.n_opts.append(len(o_len))
            self.o_len.extend(o_len)
            self.o_idx.extend(o_flat)
        else:
            state, opts, events = sample
            self.n_opts.append(len(opts))
            for toks in opts:
                self.o_len.append(len(toks))
                self.o_idx.extend(toks)
        self.s_len.append(len(state))
        self.s_idx.extend(state)
        self.e_len.append(len(events))
        self.e_idx.extend(events)
        self._starts = self._cache = None

    def extend(self, other) -> None:
        if isinstance(other, PackedSamples):
            for f in FIELDS:
                getattr(self, f).extend(getattr(other, f))
            self._starts = self._cache = None
        else:
            for s in other:
                self.append(s)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def take_ranges(self, ranges):
        """Copy complete trajectory ranges without expanding token lists."""
        from itertools import accumulate

        starts = {f: list(accumulate(getattr(self, f), initial=0)) for f in ("s_len", "e_len", "n_opts", "o_len")}
        out = PackedSamples()
        for lo, hi in ranges:
            for f in ("s_len", "e_len", "n_opts"):
                getattr(out, f).extend(getattr(self, f)[lo:hi])
            for f, length in (("s_idx", "s_len"), ("e_idx", "e_len"), ("o_len", "n_opts")):
                getattr(out, f).extend(getattr(self, f)[starts[length][lo] : starts[length][hi]])
            a, b = starts["n_opts"][lo], starts["n_opts"][hi]
            out.o_idx.extend(self.o_idx[starts["o_len"][a] : starts["o_len"][b]])
        return out

    # -- list-like access (tests, debugging) ---------------------------------

    def __len__(self) -> int:
        return len(self.s_len)

    def _offsets(self):
        if self._starts is None:
            s = e = o = t = 0
            ss, es, os_, ts = [], [], [], [0]
            for i in range(len(self.s_len)):
                ss.append(s)
                es.append(e)
                os_.append(o)
                s += self.s_len[i]
                e += self.e_len[i]
                o += self.n_opts[i]
            for n in self.o_len:
                t += n
                ts.append(t)
            self._starts = (ss, es, os_, ts)
        return self._starts

    def __getitem__(self, i: int):
        if i < 0:
            i += len(self)
        ss, es, os_, ts = self._offsets()
        state = self.s_idx[ss[i] : ss[i] + self.s_len[i]].tolist()
        events = self.e_idx[es[i] : es[i] + self.e_len[i]].tolist()
        opts = [self.o_idx[ts[k] : ts[k + 1]].tolist() for k in range(os_[i], os_[i] + self.n_opts[i])]
        return state, opts, events

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # -- pickling: the arrays as bytes ---------------------------------------

    def __getstate__(self):
        return tuple(getattr(self, f).tobytes() for f in FIELDS)

    def __setstate__(self, state) -> None:
        for f, b in zip(FIELDS, state):
            a = array("i")
            a.frombytes(b)
            setattr(self, f, a)
        self._starts = self._cache = None
