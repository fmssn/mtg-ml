"""Central inference server: one process per GPU runs every policy for its
share of the rollout workers.

Without it every worker runs its own CPU copy of the network (`rollout.py`,
`inference="local"`). With `inference="server"`:

* workers only play: engine, `featurize()`, event hashes, recording. They do
  not import torch (numpy only, to count entity separators).
* transport is shared memory, no pickling and no pipes on the hot path. A
  server owns two blocks: a control block (one request header per worker
  group, one reply ticket per group, global words) and a data block (per
  worker group: the request area, then the reply area). A worker writes its
  decisions into its request area, the header after them and the header's
  ticket last, then spins on its reply ticket. The server polls the tickets
  (one numpy compare over all groups), builds a small header for the batch
  on the host and runs the rest on the GPU, which reads the requests straight
  from the data block (registered with CUDA, mapped into the device's address
  space) and writes (action, log-prob, value) per decision straight into the
  reply areas.
* every policy the server holds (pool checkpoints and the learner; one
  config) is stacked (`rl/stacked.py`): one batched forward serves all rows
  of a batch, rows sorted by policy, so one forward per batch, not one per
  policy. The step-mode forward never syncs with the host (every size comes
  from the headers) and is captured in a CUDA graph per padded shape
  bucket: per batch the server copies one header and replays one graph.
  Trunks the stack does not cover (transformer) and policies of another
  config run per policy, eagerly (`_legacy`).
* recurrent state lives on the server: a table of hidden states indexed by
  (worker, slot), where a worker's slot is one (game, seat) of its live
  games, reused when a game ends. A request row flags `fresh` at the start
  of a sequence (zero state).
* policies are identified by (checkpoint path, version) exactly like
  `rollout.load_policy`; a worker registers a key once (one round trip on the
  request queue) and then refers to it by the slot id the server returns. A
  new learner version (version > 0) replaces the previous ones, whatever
  their path, in place in the stack.
* `InferenceServer(..., devices=[...])` starts one server per device and
  shards the workers over them in contiguous blocks; each server process is
  pinned to its GPU's NUMA node unless given CPUs.

Each worker keeps `Job.groups` requests in flight (its live games are split
into groups), so the CPU work of one group overlaps the GPU round trip of
the other. `InferenceServer.pool()` also hands every worker the pool's
shared spec counters, so `rollout.run_specs` can stream games to them.
"""

from __future__ import annotations

import os
import time
from array import array
from dataclasses import dataclass, replace
from multiprocessing import shared_memory

SLOTS_PER_WORKER = 8192  # (game, seat) slots per worker: jobs of up to 4096 games
MAX_ROWS = 4096  # decisions per request: one per live game of a group (rollout.MAX_LIVE)
REC = 9  # ints per decision record: slot, fresh, s_off, s_len, e_off, e_len, opt_off, n_opts, o_off
HDR = 128  # ints per request header
H_TICKET, H_ROWS, H_OPTS, H_S, H_E, H_O, H_ENT, H_RUNS = range(8)
RUNS0 = 16  # header: (policy id, rows) per run of equal policy from here
MAX_RUNS = (HDR - RUNS0) // 2
RESP_STRIDE = 16  # ints between reply tickets (a cache line each)
GLOBAL = 16  # ints of global words at the start of the control block: [0] the server failed
OUT_INTS = 3 * MAX_ROWS
LEGACY = 1 << 20  # ids of policies outside the stack start here
_I32 = 4


@dataclass(frozen=True)
class Layout:
    """Shared-memory layout of one server (sizes in int32 words)."""

    n_workers: int
    groups: int
    request_ints: int

    @property
    def slots(self) -> int:  # request slots: one per (worker, group)
        return self.n_workers * self.groups

    @property
    def hdr0(self) -> int:
        return GLOBAL

    @property
    def resp0(self) -> int:
        return GLOBAL + self.slots * HDR

    @property
    def ctrl_ints(self) -> int:
        return self.resp0 + self.slots * RESP_STRIDE

    @property
    def stride(self) -> int:
        return self.request_ints + OUT_INTS

    def in_base(self, k: int) -> int:
        return k * self.stride

    def out_base(self, k: int) -> int:
        return k * self.stride + self.request_ints

    @property
    def dummy_out(self) -> int:  # where padded rows write their outputs
        return self.slots * self.stride

    @property
    def data_ints(self) -> int:
        return self.dummy_out + 64


def _create_blocks(layout: Layout) -> tuple:
    ctrl = shared_memory.SharedMemory(create=True, size=layout.ctrl_ints * _I32)
    data = shared_memory.SharedMemory(create=True, size=layout.data_ints * _I32)
    return ctrl, data


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class InferenceClient:
    """A worker's connection to its server. Requests are submitted per group
    (at most one in flight per group) and collected by handle."""

    def __init__(self, wid: int, req_q, resp_q, names: tuple[str, str], layout: Layout):
        from .features import STATE_DIM

        self.wid = wid  # id on its server
        self.req_q = req_q
        self.resp_q = resp_q
        self.layout = layout
        self.sep = STATE_DIM
        self.ctrl = shared_memory.SharedMemory(name=names[0])
        self.data = shared_memory.SharedMemory(name=names[1])
        self.c = self.ctrl.buf.cast("i")
        self.d = self.data.buf.cast("i")
        self.f = self.data.buf.cast("f")
        self.ids: dict = {}  # policy key -> id on the server
        # Continue from the slot's last reply: a ticket must differ from the
        # one the server saw last (a new pool can reuse a slot).
        self.tickets = [self.c[layout.resp0 + (wid * layout.groups + g) * RESP_STRIDE] for g in range(layout.groups)]
        try:
            import numpy as np

            self.np = np
        except ImportError:  # pragma: no cover
            self.np = None

    def _id(self, key) -> int:
        pid = self.ids.get(key)
        if pid is None:
            self.req_q.put(("register", self.wid, key))
            r = self.resp_q.get()
            if isinstance(r, str):
                raise RuntimeError(f"inference server failed: {r}")
            pid = self.ids[key] = r
        return pid

    def submit(self, group: int, rows: list) -> tuple:
        """rows: (policy key, slot, fresh, state, option lengths, option
        tokens, events), grouped by policy key (rows of one policy
        contiguous). Returns a handle."""
        L = self.layout
        if not 0 <= group < L.groups:
            raise ValueError(f"group {group}: the server has {L.groups} request slots per worker (ServerConfig.groups)")
        n = len(rows)
        if n > MAX_ROWS:
            raise ValueError(f"{n} decisions in one request, at most {MAX_ROWS}")
        rec, olen, s, e, o = array("i"), array("i"), array("i"), array("i"), array("i")
        runs = array("i")
        last = None
        for key, sl, fr, st, ol, of, ev in rows:
            if key != last:
                if len(runs) == 2 * MAX_RUNS:
                    raise ValueError(f"more than {MAX_RUNS} policies in one request")
                runs.append(self._id(key))
                runs.append(0)
                last = key
            runs[-1] += 1
            rec.extend((sl, fr, len(s), len(st), len(e), len(ev), len(olen), len(ol), len(o)))
            s.extend(st)
            e.extend(ev)
            olen.extend(ol)
            o.extend(of)
        k = self.wid * L.groups + group
        pos = L.in_base(k)
        if REC * n + len(olen) + len(s) + len(e) + len(o) > L.request_ints:
            raise ValueError(f"request of {REC * n + len(olen) + len(s) + len(e) + len(o)} ints does not fit (ServerConfig.request_ints = {L.request_ints})")
        d = self.d
        for part in (rec, olen, s, e, o):
            d[pos : pos + len(part)] = part
            pos += len(part)
        if self.np is not None and len(s):
            n_ent = int(self.np.count_nonzero(self.np.frombuffer(s, dtype=self.np.int32) == self.sep))
        else:
            n_ent = s.count(self.sep)
        h = L.hdr0 + k * HDR
        c = self.c
        c[h + 1 : h + RUNS0] = array("i", (n, len(olen), len(s), len(e), len(o), n_ent, len(runs) // 2) + (0,) * (RUNS0 - 8))
        c[h + RUNS0 : h + RUNS0 + len(runs)] = runs
        ticket = self.tickets[group] = self.tickets[group] % 0x3FFFFFFF + 1
        c[h] = ticket  # publish last: the server reads the header once it sees the ticket
        return (ticket, group, n)

    def collect(self, handle: tuple) -> tuple[list[int], list[float], list[float]]:
        ticket, group, n = handle
        L = self.layout
        i = L.resp0 + (self.wid * L.groups + group) * RESP_STRIDE
        if self.c[i] != ticket:
            self._wait(i, ticket)
        b = L.out_base(self.wid * L.groups + group)
        out = self.f[b : b + 3 * n].tolist()  # rows of (action, log-prob, value)
        return [int(a) for a in out[0::3]], out[1::3], out[2::3]

    def _wait(self, i: int, ticket: int) -> None:
        """Spin on the reply ticket (this core has nothing else to do);
        after 50 ms (the server loads a policy or captures a graph) sleep
        between checks."""
        c = self.c
        spins, t0 = 0, None
        while c[i] != ticket:
            spins += 1
            if spins & 0x3FFF == 0:
                if c[0]:
                    raise RuntimeError(f"inference server failed: {self._error()}")
                if t0 is None:
                    t0 = time.monotonic()
                elif time.monotonic() - t0 > 0.05:
                    time.sleep(0.0002)

    def _error(self) -> str:
        if self.resp_q._reader.poll(5):  # noqa: SLF001 - SimpleQueue has no get(timeout)
            return str(self.resp_q.get())
        return "(no message)"

    def close(self) -> None:
        for v in ("c", "d", "f"):
            getattr(self, v).release()
        self.ctrl.close()
        self.data.close()


_CLIENT: InferenceClient | None = None


def init_worker(counter, shards, cpus=None, claim=None) -> None:
    """Pool initializer for server mode: claim a worker id, pin to a CPU
    (round robin over `cpus`, Linux), connect to the server whose block of
    workers it falls in, and take the pool's spec counters
    (`rollout.connect`). shards: (first worker, workers, layout, block
    names, request queue, response queues) per server."""
    global _CLIENT
    from .rollout import connect

    connect(claim)
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpus[wid % len(cpus)]})
    total = sum(sh[1] for sh in shards)
    wid %= total
    for first, n, layout, names, req_q, resp_qs in shards:
        if first <= wid < first + n:
            _CLIENT = InferenceClient(wid - first, req_q, resp_qs[wid - first], names, layout)
            break
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
    max_rows: int = 16384  # largest batch (decisions); more waits for the next one
    max_wait_ms: float = 0.0  # wait this long for more requests while the batch is under min_rows
    min_rows: int = 0
    threads: int = 2
    cpus: tuple[int, ...] | None = None  # pin the server process (Linux); default: its GPU's NUMA node, minus the workers' CPUs
    dry_run: bool = False  # benchmark aid: uniform random options, no network (measures the pipeline alone)
    groups: int = 4  # request slots per worker (Job.groups must not exceed it)
    request_ints: int = 1 << 18  # int32 words per request slot (a decision takes ~260)
    policy_slots: int = 32  # policies the stack holds before it grows (growing drops the CUDA graphs)
    graphs: bool = True  # CUDA graphs over padded shape buckets (CUDA only)
    max_graphs: int = 48


def _bucket(n: int, lo: int) -> int:
    """Smallest of lo * {1, 1.5} * 2^k that holds n."""
    b = lo
    while b < n:
        b = b * 3 // 2 if (b & (b - 1)) == 0 else b * 4 // 3
    return b


_KEEP: list = []
_MINS = (64, 128, 8192, 1024, 1024, 1024)  # rows, options, state / event / option tokens, entities


class _CAI:
    """A device pointer as a __cuda_array_interface__ (to wrap mapped host memory)."""

    def __init__(self, ptr: int, n: int, typestr: str):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": typestr, "data": (ptr, False), "version": 3, "strides": None}


class _Server:
    def __init__(self, cfg: ServerConfig, n_workers: int, layout: Layout | None = None, names: tuple | None = None, resp_qs=None):
        import numpy as np
        import torch

        self.np, self.torch = np, torch
        torch.set_num_threads(cfg.threads)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.layout = L = layout or Layout(n_workers, cfg.groups, cfg.request_ints)
        self.n_workers = n_workers
        self.resp_qs = resp_qs
        self._owned = None
        if names is None:  # standalone (tests, tools/profile_server.py): own blocks
            import weakref

            self._owned = _create_blocks(L)
            names = (self._owned[0].name, self._owned[1].name)
            weakref.finalize(self, _unlink, self._owned)
        self.names = names
        self.ctrl = shared_memory.SharedMemory(name=names[0])
        self.data = shared_memory.SharedMemory(name=names[1])
        _KEEP.append((self.ctrl, self.data))  # mapped for the process's life: numpy and CUDA views point into them
        self.c = np.frombuffer(self.ctrl.buf, dtype=np.int32)
        self.hdr = self.c[L.hdr0 : L.resp0].reshape(L.slots, HDR)
        self.req = self.hdr[:, H_TICKET]
        self.resp = self.c[L.resp0 : L.ctrl_ints : RESP_STRIDE]
        self.seen = np.zeros(L.slots, dtype=np.int32)
        self.dn = np.frombuffer(self.data.buf, dtype=np.int32)
        self.df = self.dn.view(np.float32)
        self.in_base = np.arange(L.slots, dtype=np.int64) * L.stride
        self.out_base = self.in_base + L.request_ints
        self.wbase = (np.arange(L.slots, dtype=np.int64) // L.groups) * SLOTS_PER_WORKER
        self._mapped = False
        if self.device.type == "cuda" and not cfg.dry_run:
            torch.cuda.set_device(self.device)
            ptr = self.dn.ctypes.data
            err = torch.cuda.cudart().cudaHostRegister(ptr, self.data.size, 2)  # cudaHostRegisterMapped
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister of the request block failed: {err}")
            self._mapped = True
            self.region = torch.as_tensor(_CAI(ptr, self.dn.shape[0], "<i4"), device=self.device)
        else:
            self.region = torch.frombuffer(self.data.buf, dtype=torch.int32)
        self.region_f = self.region.view(torch.float32)
        Q = L.slots
        self.Q = Q
        self.stage = torch.zeros(8 + 7 * Q + 3 * (L.slots * 64 + 1), dtype=torch.int32, pin_memory=self.device.type == "cuda")
        self.stack = None
        self.ids: dict = {}  # policy key -> id (stack slot, or LEGACY + n)
        self.cpu_nets: dict = {}  # id -> the network on the CPU (memory-mapped)
        self.dev_nets: dict = {}  # id -> on the device (legacy path)
        self.free: list[int] = []
        self.next_legacy = LEGACY
        self.hidden = None  # (n_workers * SLOTS_PER_WORKER + 1, state size); the last row is the padded rows' dummy
        self.state_size = None
        self.graphs: dict = {}
        self.pool = None
        self.stats = {"batches": 0, "rows": 0, "requests": 0, "busy_s": 0.0, "prep_s": 0.0, "infer_s": 0.0, "reply_s": 0.0, "padded_rows": 0, "graphs": 0, "eager": 0, "legacy": 0, "capture_s": 0.0}

    # -- policies --------------------------------------------------------

    @property
    def models(self) -> dict:
        """Loaded policies by key (tests)."""
        return {k: self.cpu_nets[v] for k, v in self.ids.items()}

    def model(self, key):
        """Load (register) a policy; returns its network on the CPU."""
        return self.cpu_nets[self.register_key(tuple(key))]

    def register_key(self, key: tuple) -> int:
        if key in self.ids:
            return self.ids[key]
        from .features import OPTION_DIM, STATE_DIM
        from .rollout import load_net
        from .stacked import PolicyStack

        path, version = key
        if version:  # a new learner version replaces the others, whatever their path
            for k in [k for k in self.ids if k[1]]:
                pid = self.ids.pop(k)
                self.cpu_nets.pop(pid)
                self.dev_nets.pop(pid, None)
                if pid < LEGACY:
                    self.free.append(pid)
        net = load_net(path)
        if net.memory == "gru":
            if self.hidden is None:
                self.state_size = net.state_size
                self.hidden = self.torch.zeros(self.n_workers * SLOTS_PER_WORKER + 1, net.state_size, device=self.device)
            elif net.state_size != self.state_size:
                raise ValueError("all recurrent policies served together must share the hidden size")
        stackable = PolicyStack.supports(net.config) and net.config["state_dim"] == STATE_DIM and net.config["option_dim"] == OPTION_DIM and not self.cfg.dry_run
        if stackable and self.stack is None:
            self.stack = PolicyStack(net.config, self.cfg.policy_slots, self.device)
            self.free = list(range(self.cfg.policy_slots - 1, -1, -1))
        if stackable and net.config == self.stack.config:
            if not self.free:
                self._grow()
            pid = self.free.pop()
            self.stack.load(pid, net)
            if self.device.type == "cuda":
                self.torch.cuda.current_stream().synchronize()  # the source is memory-mapped, copied asynchronously
        else:
            pid = self.next_legacy
            self.next_legacy += 1
        self.cpu_nets[pid] = net
        self.ids[key] = pid
        return pid

    def _grow(self) -> None:
        from .stacked import PolicyStack

        old = self.stack
        new = PolicyStack(old.config, 2 * old.capacity, self.device)
        for name, t in old.tables.items():
            new.tables[name][: old.capacity * old.dims[name]].copy_(t[: old.capacity * old.dims[name]])
        for name, t in old.dense.items():
            new.dense[name][: old.capacity].copy_(t)
        self.free = list(range(new.capacity - 1, old.capacity - 1, -1))
        self.stack = new
        self.graphs.clear()  # they point at the old tensors

    # -- batches ---------------------------------------------------------

    def pending(self):
        """Request slots with a new ticket."""
        return self.np.flatnonzero(self.req != self.seen)

    def process(self, pend) -> None:
        """One batch: the requests in slots `pend`."""
        np = self.np
        t0 = time.perf_counter()
        H = self.hdr[pend]  # copies: the header of each request
        n = len(pend)
        nruns = H[:, H_RUNS]
        runs = H[:, RUNS0 : RUNS0 + 2 * MAX_RUNS].reshape(n, MAX_RUNS, 2)
        mask = np.arange(MAX_RUNS)[None, :] < nruns[:, None]
        cnt = np.where(mask, runs[:, :, 1], 0)
        start = np.cumsum(cnt, 1) - cnt
        req = np.broadcast_to(np.arange(n)[:, None], mask.shape)[mask]
        pid, ln, st = runs[:, :, 0][mask], cnt[mask], start[mask]
        order = np.argsort(pid, kind="stable")  # rows sorted by policy
        req, pid, ln, st = req[order], pid[order], ln[order], st[order]
        R = int(ln.sum())
        row_req = np.repeat(req, ln)
        row_loc = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(R)
        row_pol = np.repeat(pid, ln)
        tot = H[:, H_ROWS : H_RUNS].sum(0)  # rows, options, state / event / option tokens, entities
        t1 = time.perf_counter()
        if self.cfg.dry_run:
            self._dry(pend, row_req, row_loc)
        elif self.stack is not None and pid.size and int(pid.max()) < LEGACY:
            self._stacked(pend, H, tot, row_req, row_loc, row_pol)
        else:
            self._legacy(pend, row_req, row_loc, row_pol)
        t2 = time.perf_counter()
        self.resp[pend] = H[:, H_TICKET]
        self.seen[pend] = H[:, H_TICKET]
        t3 = time.perf_counter()
        s = self.stats
        s["batches"] += 1
        s["requests"] += n
        s["rows"] += R
        s["busy_s"] += t3 - t0
        s["prep_s"] += t1 - t0
        s["infer_s"] += t2 - t1
        s["reply_s"] += t3 - t2

    def _dry(self, pend, row_req, row_loc) -> None:
        np = self.np
        rb = self.in_base[pend][row_req] + REC * row_loc
        no = self.dn[rb + 7]
        ob = self.out_base[pend][row_req] + 3 * row_loc
        self.df[ob] = np.floor(np.random.random(len(no)) * no)
        self.df[ob + 1] = -np.log(no)
        self.df[ob + 2] = 0.0

    def _stacked(self, pend, H, tot, row_req, row_loc, row_pol) -> None:
        torch = self.torch
        R, N, S, Ev, O, E = (int(x) for x in tot)
        need = (R + 1, N + 1, S, Ev, O, E + 1)
        use_graphs = self.device.type == "cuda" and self.cfg.graphs
        dims = self._choose(need) if use_graphs else need
        R_p = dims[0]
        Q, n = self.Q, len(pend)
        hl = 8 + 7 * Q + 3 * R_p
        if hl > self.stage.shape[0]:
            self.stage = torch.zeros(2 * hl, dtype=torch.int32, pin_memory=self.stage.is_pinned())
        st = self.stage.numpy()
        st[:8] = (R, N, S, Ev, O, E, n, 0)
        q = st[8 : 8 + 7 * Q].reshape(7, Q)
        q[0, :n] = self.in_base[pend]
        q[1, :n] = self.out_base[pend]
        q[2, :n] = self.wbase[pend]
        q[3:, :n] = H[:, H_ROWS : H_E + 1].T  # rows, options, state tokens, event tokens
        r = st[8 + 7 * Q : hl].reshape(3, R_p)
        r[0, :R], r[0, R:] = row_req, 0
        r[1, :R], r[1, R:] = row_loc, 0
        r[2, :R], r[2, R:] = row_pol, row_pol[-1]  # padded rows keep the order sorted
        self.stats["padded_rows"] += R_p
        if use_graphs:
            g = self.graphs.get(dims) or self._capture(dims)
            if g is not None:
                graph, hdr = g
                hdr.copy_(self.stage[:hl], non_blocking=True)
                graph.replay()
                torch.cuda.current_stream().synchronize()
                return
        self.stats["eager"] += 1
        hdr = self.stage[:hl].to(self.device, non_blocking=True)
        with torch.no_grad():
            self._forward(hdr, dims)
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    def _choose(self, need: tuple) -> tuple:
        """The padded shape to run: a captured one that covers `need` with at
        most 2x padding per dimension, else need's own buckets."""
        own = tuple(_bucket(x, lo) for x, lo in zip(need, _MINS))
        if own in self.graphs:
            return own
        best, cost = None, None
        for k in self.graphs:
            if all(a >= b for a, b in zip(k, need)):
                c = max(a / b for a, b in zip(k, own))
                if c <= 2 and (cost is None or c < cost):
                    best, cost = k, c
        if best is not None or len(self.graphs) >= self.cfg.max_graphs:
            return best or own
        return own

    def _capture(self, dims: tuple):
        """Capture the forward for padded shape `dims` (None: run eagerly).
        Warm-up and capture run on an all-padding header, so they leave the
        hidden states and replies alone."""
        torch = self.torch
        if len(self.graphs) >= self.cfg.max_graphs:
            return None
        t0 = time.perf_counter()
        hdr = torch.zeros(8 + 7 * self.Q + 3 * dims[0], dtype=torch.int32, device=self.device)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._forward(hdr, dims)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._forward(hdr, dims)
        torch.cuda.current_stream().synchronize()
        self.graphs[dims] = (graph, hdr)
        self.stats["graphs"] = len(self.graphs)
        self.stats["capture_s"] += time.perf_counter() - t0
        return self.graphs[dims]

    def _forward(self, hdr, dims: tuple) -> None:
        """Gather the batch from the request areas (host memory mapped into
        the device), run the stacked step forward and write each row's
        (action, log-prob, value) into its reply area. Fixed shapes, no
        host syncs: everything a size depends on is in `hdr`."""
        torch = self.torch
        from .stacked import StepInput, excl_cumsum, segments, step_forward

        R_p, N_p, S_p, Ev_p, O_p, E_p = dims
        Q = self.Q
        region = self.region
        dev = region.device
        h = hdr.long()
        R, N, S, Ev, O = h[0], h[1], h[2], h[3], h[4]
        in_base, out_base, wbase, q_rows, q_opts, q_s, q_ev = h[8 : 8 + 7 * Q].view(7, Q)
        req, loc, pol = h[8 + 7 * Q : 8 + 7 * Q + 3 * R_p].view(3, R_p)
        ar = lambda k: torch.arange(k, device=dev)  # noqa: E731
        rvalid = ar(R_p) < R
        rec = region[(in_base[req] + REC * loc)[:, None] + ar(REC)].long()
        rec = torch.where(rvalid[:, None], rec, 0)
        slot, fresh, s_off, s_len, e_off, e_len, opt_off, n_opt, o_off = rec.unbind(1)
        a_opt = in_base + REC * q_rows  # areas of each request: option lengths, state, event and option tokens
        a_s = a_opt + q_opts
        a_e = a_s + q_s
        a_o = a_e + q_ev
        s_row, s_k = segments(s_len, S_p)
        s_valid = ar(S_p) < S
        s_tok = region[torch.where(s_valid, a_s[req[s_row]] + s_off[s_row] + s_k, 0)].long()
        ev_row, ev_k = segments(e_len, Ev_p)
        ev_valid = ar(Ev_p) < Ev
        ev_tok = region[torch.where(ev_valid, a_e[req[ev_row]] + e_off[ev_row] + ev_k, 0)].long()
        o_row, o_k = segments(n_opt, N_p)
        o_valid = ar(N_p) < N
        o_len = torch.where(o_valid, region[torch.where(o_valid, a_opt[req[o_row]] + opt_off[o_row] + o_k, 0)].long(), 0)
        o_start = excl_cumsum(o_len)
        first = excl_cumsum(n_opt).clamp_(max=N_p - 1)
        ot_opt, ot_k = segments(o_len, O_p)
        ot_valid = ar(O_p) < O
        r = o_row[ot_opt]
        ot_tok = region[torch.where(ot_valid, a_o[req[r]] + o_off[r] + o_start[ot_opt] - o_start[first[r]] + ot_k, 0)].long()
        x = StepInput(
            pol=pol, gslot=torch.where(rvalid, slot + wbase[req], self.n_workers * SLOTS_PER_WORKER), fresh=(fresh != 0) | ~rvalid,
            s_len=s_len, e_len=e_len, n_opt=n_opt, s_tok=s_tok, s_row=s_row, s_valid=s_valid, ev_tok=ev_tok, ev_valid=ev_valid,
            o_len=o_len, o_row=o_row, ot_tok=ot_tok, ot_opt=ot_opt, ot_valid=ot_valid, n_ent=E_p,
        )  # fmt: skip
        act, logp, val = step_forward(self.stack, x, self.hidden)
        out = torch.stack([act.to(torch.float32), logp, val], 1)
        pos = torch.where(rvalid, out_base[req] + 3 * loc, self.layout.dummy_out)
        self.region_f[(pos[:, None] + ar(3)).view(-1)] = out.view(-1)

    def _legacy(self, pend, row_req, row_loc, row_pol) -> None:
        """Policies outside the stack (transformer trunk, another config):
        one eager forward per policy from the request areas, on the host."""
        torch, np = self.torch, self.np
        from .rollout import _batch

        self.stats["legacy"] += 1
        dn, df = self.dn, self.df
        L = self.layout
        rows_by_pol: dict = {}
        for r in range(len(row_req)):
            rows_by_pol.setdefault(int(row_pol[r]), []).append(r)
        with torch.no_grad():
            for pid, rows in rows_by_pol.items():
                net = self.dev_nets.get(pid)
                if net is None:
                    net = self.dev_nets[pid] = self.cpu_nets[pid].to(self.device)
                xs, gslots, fresh, outs = [], [], [], []
                for r in rows:
                    k = int(pend[row_req[r]])
                    base = L.in_base(k)
                    hd = self.hdr[k]
                    a_opt = base + REC * int(hd[H_ROWS])
                    a_s = a_opt + int(hd[H_OPTS])
                    a_e = a_s + int(hd[H_S])
                    a_o = a_e + int(hd[H_E])
                    sl, fr, s_off, s_len, e_off, e_len, opt_off, n_opt, o_off = (int(v) for v in dn[base + REC * row_loc[r] : base + REC * (row_loc[r] + 1)])
                    ol = dn[a_opt + opt_off : a_opt + opt_off + n_opt]
                    parts = (dn[a_s + s_off : a_s + s_off + s_len], ol, dn[a_o + o_off : a_o + o_off + int(ol.sum())], dn[a_e + e_off : a_e + e_off + e_len])
                    xs.append(tuple(array("i", p.tobytes()) for p in parts))
                    gslots.append(sl + int(self.wbase[k]))
                    fresh.append(fr)
                    outs.append(L.out_base(k) + 3 * int(row_loc[r]))
                batch, width = _batch(torch, xs)
                batch = batch.to(self.device)
                hidden = None
                if net.memory == "gru":
                    gs = torch.tensor(gslots, device=self.device)
                    hidden = torch.where(torch.tensor(fresh, device=self.device).bool()[:, None], 0.0, self.hidden[gs])
                logits, values, hn = net(batch, hidden, max_options=width)
                if hn is not None:
                    self.hidden[gs] = hn
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                a = dist.sample()
                res = torch.stack([a.to(torch.float32), dist.log_prob(a), values], 1).cpu().numpy()
                pos = np.array(outs)
                for j in range(3):
                    df[pos + j] = res[:, j]
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    def close(self) -> None:
        if self._mapped:
            self.torch.cuda.synchronize()
            self.graphs.clear()
            del self.region, self.region_f
            self.torch.cuda.cudart().cudaHostUnregister(self.dn.ctypes.data)
            self._mapped = False


def _unlink(blocks) -> None:
    for b in blocks:
        try:
            b.close()
            b.unlink()
        except (FileNotFoundError, BufferError):
            pass


def gpu_local_cpus(device) -> tuple[int, ...] | None:
    """The CPUs of the NUMA node a CUDA device is attached to (Linux), from
    /sys/bus/pci/devices/<bus id>/local_cpulist."""
    try:
        import torch

        p = torch.cuda.get_device_properties(torch.device(device))
        bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bus}/local_cpulist") as f:
            return parse_cpus(f.read().strip())
    except (AttributeError, OSError, RuntimeError, ValueError):
        return None


def parse_cpus(spec: str | None) -> tuple[int, ...] | None:
    """'0-3,8' -> (0, 1, 2, 3, 8)."""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return tuple(out)


def serve(cfg: ServerConfig, layout: Layout, names: tuple, req_q, resp_qs, stats_q, worker_cpus=None) -> None:
    """Server process main loop. A `None` request stops it; ("stats",) asks
    for the counters on `stats_q`; ("register", worker, key) loads a policy
    and answers its id on the worker's response queue."""
    if cfg.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cfg.cpus))
    elif cfg.device.startswith("cuda") and hasattr(os, "sched_setaffinity"):
        local = gpu_local_cpus(cfg.device)
        if local:
            free = set(local) - set(worker_cpus or ())
            os.sched_setaffinity(0, free or set(local))
    srv = _Server(cfg, layout.n_workers, layout, names, resp_qs)
    prof_path = os.environ.get("MTG_SERVER_PROFILE")
    if prof_path:  # debugging aid: cProfile of the whole server loop
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
        try:
            _serve_loop(cfg, srv, req_q, stats_q)
        finally:
            prof.disable()
            prof.dump_stats(prof_path)
        return
    _serve_loop(cfg, srv, req_q, stats_q)


def _serve_loop(cfg: ServerConfig, srv: _Server, req_q, stats_q) -> None:
    """Poll the request tickets; run everything pending as one batch (after
    waiting up to max_wait_ms for min_rows); between polls, answer queue
    messages. Idle for 10 ms, it sleeps 0.2 ms between polls."""
    import traceback

    reader = req_q._reader  # noqa: SLF001 - SimpleQueue has no get(timeout)
    wait = cfg.max_wait_ms / 1000
    last = time.perf_counter()
    first_seen = None
    try:
        while True:
            if reader.poll():
                m = req_q.get()
                if m is None:
                    return
                if m[0] == "stats":
                    stats_q.put(dict(srv.stats, policies=len(srv.ids), device=cfg.device))
                elif m[0] == "register":
                    _, wid, key = m
                    try:
                        srv.resp_qs[wid].put(srv.register_key(tuple(key)))
                    except Exception as e:  # noqa: BLE001 - the worker raises it
                        srv.resp_qs[wid].put("".join(traceback.format_exception(e)))
                continue
            pend = srv.pending()
            now = time.perf_counter()
            if pend.size:
                if cfg.min_rows:
                    first_seen = first_seen or now
                    rows = int(srv.hdr[pend, H_ROWS].sum())
                    if rows < cfg.min_rows and now - first_seen < wait:
                        continue
                if int(srv.hdr[pend, H_ROWS].sum()) > cfg.max_rows:  # oldest first is unknown; take a prefix
                    cum = srv.np.cumsum(srv.hdr[pend, H_ROWS])
                    pend = pend[: max(1, int(srv.np.searchsorted(cum, cfg.max_rows, side="right")))]
                srv.process(pend)
                first_seen = None
                last = time.perf_counter()
            elif now - last > 0.01:
                reader.poll(0.0002)
    except Exception as e:
        msg = "".join(traceback.format_exception(e))
        srv.c[0] = 1
        for q in srv.resp_qs:
            q.put(msg)
        raise


class InferenceServer:
    """Owns the server processes (one per device), their shared blocks and
    queues; `pool()` makes a worker pool whose processes are connected to
    them, workers sharded over the servers in contiguous blocks.

    devices: e.g. ["cuda:0", "cuda:1"] (default: [cfg.device]).
    server_cpus: CPUs per server (default: cfg.cpus with one device, else
    each GPU's NUMA node minus `worker_cpus`)."""

    def __init__(self, n_workers: int, cfg: ServerConfig | None = None, worker_cpus: tuple[int, ...] | None = None, devices=None, server_cpus=None):
        import multiprocessing as mp

        from .rollout import MAX_MATCHUPS

        self.ctx = mp.get_context("spawn")
        self.cfg = cfg or ServerConfig()
        self.devices = list(devices) if devices else [self.cfg.device]
        k = len(self.devices)
        if not 1 <= k <= n_workers:
            raise ValueError("need between 1 and n_workers devices")
        self.n_workers = n_workers
        self.worker_cpus = worker_cpus
        self.counter = self.ctx.Value("i", 0)
        self.claim = self.ctx.Array("i", MAX_MATCHUPS)  # spec counters of rollout.run_specs
        # SimpleQueue writes in the calling thread. A Queue hands writes to a
        # feeder thread, which can wait for the GIL for a whole switch
        # interval while its process is busy: milliseconds per round trip.
        self.stats_q = self.ctx.SimpleQueue()
        self.shards, self.procs, self.blocks = [], [], []
        bounds = [s * n_workers // k for s in range(k + 1)]
        for s, dev in enumerate(self.devices):
            n = bounds[s + 1] - bounds[s]
            layout = Layout(n, self.cfg.groups, self.cfg.request_ints)
            blocks = _create_blocks(layout)
            names = (blocks[0].name, blocks[1].name)
            req_q = self.ctx.SimpleQueue()
            resp_qs = [self.ctx.SimpleQueue() for _ in range(n)]
            cpus = server_cpus[s] if server_cpus else (self.cfg.cpus if k == 1 else None)
            scfg = replace(self.cfg, device=dev, cpus=tuple(cpus) if cpus else None)
            proc = self.ctx.Process(target=serve, args=(scfg, layout, names, req_q, resp_qs, self.stats_q, worker_cpus), daemon=True, name=f"inference-server-{s}")
            proc.start()
            self.blocks.append(blocks)
            self.shards.append((bounds[s], n, layout, names, req_q, resp_qs))
            self.procs.append(proc)

    @property
    def proc(self):  # the first server process (single-device callers)
        return self.procs[0]

    def pool(self):
        pool = self.ctx.Pool(self.n_workers, initializer=init_worker, initargs=(self.counter, self.shards, self.worker_cpus, self.claim))
        pool.claim = self.claim
        return pool

    def stats(self) -> dict:
        """Counters summed over the servers ("servers": each one's)."""
        for sh in self.shards:
            sh[4].put(("stats",))
        each = [self.stats_q.get() for _ in self.shards]
        out = {k: sum(e[k] for e in each) for k, v in each[0].items() if isinstance(v, (int, float))}
        out["servers"] = each
        return out

    def close(self) -> None:
        for sh, p in zip(self.shards, self.procs):
            if p.is_alive():
                sh[4].put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        for blocks in self.blocks:
            for b in blocks:
                b.close()
                try:
                    b.unlink()
                except FileNotFoundError:
                    pass
        self.blocks = []


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


__all__ = ["InferenceServer", "Layout", "ServerConfig", "client", "default_device", "gpu_local_cpus", "init_worker", "parse_cpus"]
