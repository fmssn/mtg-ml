"""Central inference server: one process runs every policy on the GPU for all
rollout workers.

Without it every worker runs its own CPU copy of the network (`rollout.py`,
`inference="local"`), and with the Rust engine that inference is what limits
throughput. With `inference="server"`:

* workers only play: engine, `featurize()`, event hashes, recording. They do
  not import torch.
* a request is a set of decisions from one worker. The worker packs the
  ragged feature lists into its own shared-memory block (int32, no pickling
  of features) and puts a small header on the request queue.
* the server drains the queue, concatenates all pending requests per policy
  into one batch, runs it, samples an option per decision, and writes
  (action, log-prob, value) into the worker's shared output block, then
  answers on that worker's response queue.
* recurrent state lives on the server: a GPU table of hidden states indexed
  by (worker, slot), where a worker's slot is one (game, seat) of its job.
  A request row flags `fresh` at the start of a sequence (zero state).
* policies are identified by (checkpoint path, version) exactly like
  `rollout.load_policy`; the server loads them on first use and drops older
  versions of the same path.

Each worker keeps two requests in flight (its games are split in two
groups), so the CPU work of one group overlaps the GPU round trip of the
other.
"""

from __future__ import annotations

import os
import queue
import time
from array import array
from dataclasses import dataclass
from multiprocessing import shared_memory

SLOTS_PER_WORKER = 8192  # (game, seat) slots per worker: jobs of up to 4096 games
_I32 = 4


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class _Block:
    """A growable shared-memory block owned by the worker."""

    def __init__(self, nbytes: int):
        self.shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1 << 16))

    def ensure(self, nbytes: int) -> None:
        if nbytes > self.shm.size:
            self.close()
            self.shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 2 * self.shm.size))

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self) -> None:
        self.shm.close()
        self.shm.unlink()


class InferenceClient:
    """A worker's connection to the server. Requests are submitted per group
    (at most one in flight per group) and collected by ticket."""

    def __init__(self, wid: int, req_q, resp_q, groups: int = 8):
        self.wid = wid
        self.req_q = req_q
        self.resp_q = resp_q
        self.inp = [_Block(1 << 20) for _ in range(groups)]
        self.out = [_Block(1 << 16) for _ in range(groups)]
        self.ticket = 0
        self.done: set[int] = set()

    def submit(self, group: int, rows: list) -> tuple:
        """rows: (policy key, slot, fresh, state, opts, events), grouped by
        policy key (rows of one policy contiguous). Returns a handle."""
        n = len(rows)
        slot, fresh, s_len, e_len, n_opts = array("i"), array("i"), array("i"), array("i"), array("i")
        o_len, s_idx, e_idx, o_idx = array("i"), array("i"), array("i"), array("i")
        policies: list[list] = []
        for r, (key, sl, fr, state, opts, events) in enumerate(rows):
            if not policies or policies[-1][0] != key:
                policies.append([key, r, r])
            policies[-1][2] = r + 1
            slot.append(sl)
            fresh.append(fr)
            s_len.append(len(state))
            s_idx.extend(state)
            e_len.append(len(events))
            e_idx.extend(events)
            n_opts.append(len(opts))
            for toks in opts:
                o_len.append(len(toks))
                o_idx.extend(toks)
        parts = (slot, fresh, s_len, e_len, n_opts, o_len, s_idx, e_idx, o_idx)
        total = sum(len(p) for p in parts)
        inp, out = self.inp[group], self.out[group]
        inp.ensure(total * _I32)
        out.ensure(n * 3 * _I32)
        mv = inp.shm.buf.cast("i")
        pos = 0
        for p in parts:
            mv[pos : pos + len(p)] = p
            pos += len(p)
        mv.release()
        self.ticket += 1
        counts = tuple(len(p) for p in parts)
        self.req_q.put((self.wid, self.ticket, inp.name, out.name, counts, [(k, a, b) for k, a, b in policies]))
        return (self.ticket, group, n)

    def collect(self, handle: tuple) -> tuple[list[int], list[float], list[float]]:
        ticket, group, n = handle
        while ticket not in self.done:
            t = self.resp_q.get()
            if isinstance(t, str):
                raise RuntimeError(f"inference server failed: {t}")
            self.done.add(t)
        self.done.discard(ticket)
        buf = self.out[group].shm.buf
        acts = buf[: n * _I32].cast("i").tolist()
        logp = buf[n * _I32 : 2 * n * _I32].cast("f").tolist()
        vals = buf[2 * n * _I32 : 3 * n * _I32].cast("f").tolist()
        return acts, logp, vals

    def close(self) -> None:
        for b in self.inp + self.out:
            b.close()


_CLIENT: InferenceClient | None = None


def init_worker(counter, req_q, resp_qs, cpus=None) -> None:
    """Pool initializer for server mode: claim a worker id, pin to a CPU
    (round robin over `cpus`, Linux) and connect."""
    global _CLIENT
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpus[wid % len(cpus)]})
    _CLIENT = InferenceClient(wid, req_q, resp_qs[wid], groups=8)
    import atexit

    atexit.register(_CLIENT.close)


def client() -> InferenceClient:
    if _CLIENT is None:
        raise RuntimeError("this process is not connected to an inference server (inference='server' needs InferenceServer.pool())")
    return _CLIENT


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    device: str = "cuda"
    max_rows: int = 16384  # stop draining the queue past this many decisions
    max_wait_ms: float = 1.0  # wait this long for more requests when the batch is small
    min_rows: int = 512
    threads: int = 2
    cpus: tuple[int, ...] | None = None  # pin the server process (Linux); give it cores of its own


def _attach(cache: dict, name: str):
    shm = cache.get(name)
    if shm is None:
        # The worker owns (and unlinks) the block. Spawned processes share the
        # parent's resource tracker, which de-duplicates the registration.
        shm = shared_memory.SharedMemory(name=name)
        cache[name] = shm
        if len(cache) > 4096:  # blocks replaced by bigger ones
            for k in list(cache)[:1024]:
                cache.pop(k).close()
    return shm


class _Server:
    def __init__(self, cfg: ServerConfig, n_workers: int):
        import torch

        self.torch = torch
        torch.set_num_threads(cfg.threads)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.models: dict = {}
        self.hidden = None  # (n_workers * SLOTS_PER_WORKER, H) on device
        self.n_workers = n_workers
        self.shm: dict = {}
        self.stats = {"batches": 0, "rows": 0, "requests": 0, "busy_s": 0.0}

    def model(self, key):
        torch = self.torch
        from .model import PolicyNet

        if key not in self.models:
            path, version = key
            if version:
                for k in [k for k in self.models if k[0] == path]:
                    del self.models[k]
            ck = torch.load(path, map_location="cpu", weights_only=False)
            net = PolicyNet(**ck["config"])
            net.load_state_dict(ck["model"])
            net.eval().to(self.device)
            if net.memory == "gru" and self.hidden is None:
                self.hidden = torch.zeros(self.n_workers * SLOTS_PER_WORKER, net.hidden, device=self.device)
            if net.memory == "gru" and self.hidden.shape[1] != net.hidden:
                raise ValueError("all recurrent policies served together must share the hidden size")
            self.models[key] = net
        return self.models[key]

    def process(self, msgs: list) -> None:
        """One batch: every pending request, one forward pass per policy.

        Everything index-shaped (row selection per policy, offsets, option
        rows) is computed on the CPU, where it costs microseconds; each
        policy's batch then goes to the device in one copy, and results come
        back in one copy. No step on the device forces a host sync."""
        torch = self.torch
        t0 = time.perf_counter()
        per_field = [[] for _ in range(9)]  # slot fresh s_len e_len n_opts o_len s_idx e_idx o_idx
        row_wid, row_pol, n_rows = [], [], []
        keys: dict = {}
        for wid, ticket, in_name, out_name, counts, policies in msgs:
            flat = torch.frombuffer(_attach(self.shm, in_name).buf, dtype=torch.int32, count=sum(counts))
            for f, part in enumerate(torch.split(flat, counts)):
                per_field[f].append(part)
            n_rows.append(counts[0])
            row_wid.append((wid, counts[0]))
            for key, a, b in policies:
                row_pol.append((keys.setdefault(tuple(key), len(keys)), b - a))
        cat = lambda fl: (torch.cat(fl) if len(fl) > 1 else fl[0]).long()  # noqa: E731
        slot, fresh, s_len, e_len, n_opts, o_len, s_idx, e_idx, o_idx = (cat(fl) for fl in per_field)
        B = slot.shape[0]
        wid = torch.repeat_interleave(torch.tensor([w for w, _ in row_wid]), torch.tensor([n for _, n in row_wid]))
        gslot = slot + wid * SLOTS_PER_WORKER
        acts = torch.empty(B, dtype=torch.long)
        logp = torch.empty(B)
        vals = torch.empty(B)
        if len(keys) == 1:
            groups = [(next(iter(keys)), None)]
        else:
            pol = torch.repeat_interleave(torch.tensor([k for k, _ in row_pol]), torch.tensor([n for _, n in row_pol]))
            groups = [(key, torch.nonzero(pol == k).squeeze(1)) for key, k in keys.items()]
        with torch.no_grad():
            outs = []
            for key, rows in groups:
                net = self.model(key)
                outs.append((rows, self._run(net, _select(torch, rows, gslot, fresh, s_len, e_len, n_opts, o_len, s_idx, e_idx, o_idx))))
            for rows, (a, lp, v) in outs:  # one device-to-host copy per policy
                res = torch.stack([a.to(torch.float32), lp, v]).cpu()
                if rows is None:
                    acts, logp, vals = res[0].long(), res[1], res[2]
                else:
                    acts[rows], logp[rows], vals[rows] = res[0].long(), res[1], res[2]
        r = 0
        for (wid_, ticket, in_name, out_name, counts, policies), n in zip(msgs, n_rows):
            buf = _attach(self.shm, out_name).buf
            torch.frombuffer(buf, dtype=torch.int32, count=n).copy_(acts[r : r + n].to(torch.int32))
            o = torch.frombuffer(buf, dtype=torch.float32, count=2 * n, offset=n * _I32)
            o[:n].copy_(logp[r : r + n])
            o[n:].copy_(vals[r : r + n])
            r += n
            self.resp_qs[wid_].put(ticket)
        self.stats["batches"] += 1
        self.stats["requests"] += len(msgs)
        self.stats["rows"] += B
        self.stats["busy_s"] += time.perf_counter() - t0

    def _run(self, net, cpu: list):
        """Forward + sampling on the device for one policy's prepared batch."""
        torch = self.torch
        from .model import Batch

        sizes = [int(t.shape[0]) for t in cpu]
        dev = torch.cat(cpu).to(self.device, non_blocking=False)
        gs, fr, s_v, s_off, e_v, e_off, o_v, o_off, o_row, o_pos, n_opts = torch.split(dev, sizes)
        batch = Batch(s_v, s_off, e_v, e_off, o_v, o_off, o_row, o_pos, n_opts)
        hidden = None
        if net.memory == "gru":
            hidden = self.hidden[gs]
            hidden[fr.bool()] = 0.0
        logits, values, hn = net(batch, hidden, max_options=int(cpu[-1].max()))
        if hn is not None:
            self.hidden[gs] = hn
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        acts = dist.sample()
        return acts, dist.log_prob(acts), values


def _select(torch, rows, gslot, fresh, s_len, e_len, n_opts, o_len, s_idx, e_idx, o_idx) -> list:
    """CPU: the model inputs of decisions `rows` (None = all), as one list of
    int64 tensors: gslot, fresh, state values/offsets, event values/offsets,
    option values/offsets, option row, option position, options per row."""

    def excl(x):
        out = torch.zeros_like(x)
        torch.cumsum(x[:-1], 0, out=out[1:])
        return out

    def segments(flat, start, length):
        off = excl(length)
        idx = torch.repeat_interleave(start - off, length) + torch.arange(int(length.sum()))
        return flat[idx], off

    if rows is None:
        s_v, s_off = s_idx, excl(s_len)
        e_v, e_off = e_idx, excl(e_len)
        o_v, o_off = o_idx, excl(o_len)
        no, gs, fr = n_opts, gslot, fresh
    else:
        no = n_opts[rows]
        s_v, s_off = segments(s_idx, excl(s_len)[rows], s_len[rows])
        e_v, e_off = segments(e_idx, excl(e_len)[rows], e_len[rows])
        opt_ids = torch.repeat_interleave(excl(n_opts)[rows] - excl(no), no) + torch.arange(int(no.sum()))
        o_v, o_off = segments(o_idx, excl(o_len)[opt_ids], o_len[opt_ids])
        gs, fr = gslot[rows], fresh[rows]
    o_row = torch.repeat_interleave(torch.arange(no.shape[0]), no)
    o_pos = torch.arange(o_row.shape[0]) - torch.repeat_interleave(excl(no), no)
    return [gs, fr, s_v, s_off, e_v, e_off, o_v, o_off, o_row, o_pos, no]


def serve(cfg: ServerConfig, n_workers: int, req_q, resp_qs, stats_q) -> None:
    """Server process main loop. A `None` request stops it; ("stats",) asks
    for the counters on `stats_q`."""
    if cfg.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cfg.cpus))
    srv = _Server(cfg, n_workers)
    srv.resp_qs = resp_qs
    prof_path = os.environ.get("MTG_SERVER_PROFILE")
    if prof_path:  # debugging aid: cProfile of the whole server loop
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
        try:
            _serve_loop(cfg, srv, req_q, resp_qs, stats_q)
        finally:
            prof.disable()
            prof.dump_stats(prof_path)
        return
    _serve_loop(cfg, srv, req_q, resp_qs, stats_q)


def _serve_loop(cfg: ServerConfig, srv: "_Server", req_q, resp_qs, stats_q) -> None:
    """Take every request that is already queued (waiting up to max_wait_ms
    for more while the batch is small), run them as one batch, repeat."""
    wait = cfg.max_wait_ms / 1000
    ready = req_q._reader.poll  # noqa: SLF001 - SimpleQueue has no get(timeout)
    while True:
        m = req_q.get()
        if m is None:
            break
        if m == ("stats",):
            stats_q.put(dict(srv.stats))
            continue
        msgs = [m]
        rows = m[4][0]
        deadline = time.perf_counter() + wait
        stop = False
        while rows < cfg.max_rows:
            left = 0.0 if rows >= cfg.min_rows else max(0.0, deadline - time.perf_counter())
            if not ready(left):
                break
            nxt = req_q.get()
            if nxt is None:
                stop = True
                break
            if nxt == ("stats",):
                stats_q.put(dict(srv.stats))
                continue
            msgs.append(nxt)
            rows += nxt[4][0]
        try:
            srv.process(msgs)
        except Exception as e:  # noqa: BLE001 - surface the error in the workers
            import traceback

            msg = "".join(traceback.format_exception(e))
            for wid, *_ in msgs:
                resp_qs[wid].put(msg)
            raise
        if stop:
            break


class InferenceServer:
    """Owns the server process and the queues; `pool()` makes a worker pool
    whose processes are connected to it."""

    def __init__(self, n_workers: int, cfg: ServerConfig | None = None, worker_cpus: tuple[int, ...] | None = None):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.cfg = cfg or ServerConfig()
        self.n_workers = n_workers
        self.worker_cpus = worker_cpus
        # SimpleQueue writes in the calling thread. A Queue hands writes to a
        # feeder thread, which can wait for the GIL for a whole switch
        # interval while its process is busy: milliseconds per round trip.
        self.req_q = self.ctx.SimpleQueue()
        self.resp_qs = [self.ctx.SimpleQueue() for _ in range(n_workers)]
        self.stats_q = self.ctx.SimpleQueue()
        self.counter = self.ctx.Value("i", 0)
        self.proc = self.ctx.Process(target=serve, args=(self.cfg, n_workers, self.req_q, self.resp_qs, self.stats_q), daemon=True, name="inference-server")
        self.proc.start()

    def pool(self):
        return self.ctx.Pool(self.n_workers, initializer=init_worker, initargs=(self.counter, self.req_q, self.resp_qs, self.worker_cpus))

    def stats(self) -> dict:
        self.req_q.put(("stats",))
        return self.stats_q.get()

    def close(self) -> None:
        if self.proc.is_alive():
            self.req_q.put(None)
            self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()


def torch_loaded() -> bool:
    """Whether this process has imported torch (workers in server mode should not)."""
    import sys

    return "torch" in sys.modules


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


__all__ = ["InferenceServer", "ServerConfig", "client", "default_device", "init_worker"]

