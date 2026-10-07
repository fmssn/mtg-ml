"""Worker pools driven from a process of their own, and the trainer's CPU layout.

The trainer's main thread should do nothing but the PPO update and the
publish of its weights: a thread of the same process that merges rollout
results (`rollout.play`: unpickle, `unpark`, list extends) holds the GIL a
third as often as the update's kernel-launch loop and slowed the update by
~30%. So the pool that plays the games is owned by a collector process
(`PoolProcess`), which runs any `fn(pool, *args)` it is sent, one call at a
time in submission order, and sends the value back. A `Result` (the merged
rollout) travels as a `SharedResult`: the samples, actions, log-probs,
advantages and returns in one shared-memory block, parked once by the
collector; the trainer maps it without copying the samples
(`SharedResult.attach`) and frees it after the update (`release`).

`PoolThread` is the same interface over a pool in this process, calls run
in one background thread (the trainer before this module; tests use it to
patch the pool). The evaluation runs on a second `PoolProcess` with its own
small pool and its own cores (`Trainer` evaluates asynchronously).

`cpu_layout` splits the CPUs this process may use between the trainer, the
inference server, the evaluation and the rollout workers, trainer first on
the GPU's NUMA node.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .rollout import Result, create_pool
from .samples import FIELDS, PackedSamples

_FLOATS = (("actions", "q"), ("logps", "f"), ("advantages", "f"), ("returns", "f"), ("kinds", "b"))  # the typecodes ppo_update converts to


class SharedResult:
    """A merged `Result` parked in one shared-memory block for the trip from
    the collector to the trainer. The pipe then carries ~100 KB (games,
    lengths, block layout) instead of hundreds of MB of samples, and the
    trainer neither unpickles nor copies them: `attach` gives a Result
    whose sample fields are views of the block (`torch.frombuffer` and the
    host-to-device copy read them in place) and whose float fields are
    arrays (`ppo_update` copies them with one memcpy each). log-probs,
    advantages and returns are stored as float32, which is what
    `ppo_update` makes of them anyway."""

    def __init__(self, res: Result):
        from multiprocessing import shared_memory

        parts = [memoryview(getattr(res.samples, f)).cast("B") for f in FIELDS]
        parts += [memoryview(array(t, getattr(res, name))).cast("B") for name, t in _FLOATS]
        self.sizes = [len(b) for b in parts]
        shm = shared_memory.SharedMemory(create=True, size=max(sum(self.sizes), 1))
        pos = 0
        for b in parts:
            shm.buf[pos : pos + len(b)] = b
            pos += len(b)
        self.name = shm.name
        shm.close()
        self.lengths, self.games, self.timing = res.lengths, res.games, res.timing

    def attach(self) -> Result:
        """The Result, its samples viewing the block. The block's name is
        unlinked at once (the mapping lives on until `release`), so a crash
        cannot leak it."""
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=self.name)
        shm.unlink()
        out = Result(lengths=self.lengths, games=self.games, timing=self.timing)
        out.samples = PackedSamples()
        pos, views = 0, []
        names = list(FIELDS) + [n for n, _ in _FLOATS]
        types = ["i"] * len(FIELDS) + [t for _, t in _FLOATS]
        for name, t, n in zip(names, types, self.sizes):
            v = shm.buf[pos : pos + n]
            pos += n
            if name in FIELDS:
                v = v.cast(t)
                views.append(v)
                setattr(out.samples, name, v)
            else:
                a = array(t)
                a.frombytes(v)
                v.release()
                setattr(out, name, a)
        out._shared = (shm, views)
        return out


_UNMAPPED: list = []  # blocks whose views were still exported at release: retried at the next release


def release(res: Result) -> None:
    """Unmap the block behind a Result from `SharedResult.attach` (no-op for
    any other Result). Its samples are emptied; tensors still viewing them
    (the CPU tensors `packed_tensors` caches on `res.samples`) must be gone,
    or the unmap waits for the next call."""
    shared = getattr(res, "_shared", None)
    if shared is None:
        return
    del res._shared
    for f in FIELDS:
        setattr(res.samples, f, array("i"))
    res.samples._cache = None
    _UNMAPPED.append(shared)
    for item in list(_UNMAPPED):
        shm, views = item
        try:
            for v in views:
                v.release()
            shm.close()
        except BufferError:  # still viewed by a live tensor
            continue
        _UNMAPPED.remove(item)


# -- pool in a process ---------------------------------------------------------


def _serve(conn, workers: int, inference: str, server_cfg, worker_cpus, cpus, nice: int) -> None:
    """Main of a `PoolProcess`: build the pool, then run calls until told to stop."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwind, so the pool's workers are terminated too
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
    if nice:
        os.nice(nice)  # inherited by the pool's workers
    pool = server = None
    clean = False
    try:
        try:
            if inference == "server":
                from .inference import InferenceServer

                server = InferenceServer(workers, server_cfg, worker_cpus)
            pool = create_pool(workers, inference, server, worker_cpus)
        except Exception as e:  # noqa: BLE001 - reported to the owner
            conn.send(("err", None, _portable(e), time.monotonic()))
            return
        while True:
            try:
                msg = conn.recv()
            except EOFError:  # the owner is gone
                break
            if msg is None:
                clean = True
                break
            tag, fn, args = msg
            try:
                value = fn(pool, *args)
                if isinstance(value, Result):
                    value = SharedResult(value)
                reply = ("ok", tag, value, time.monotonic())
            except Exception as e:  # noqa: BLE001 - re-raised by the owner
                reply = ("err", tag, _portable(e), time.monotonic())
            conn.send(reply)
    finally:
        if pool is not None:
            if clean:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        if server is not None:
            server.close()


def _portable(e: BaseException) -> BaseException:
    """The exception if it pickles, else a RuntimeError with its traceback."""
    import pickle

    try:
        pickle.dumps(e)
        return e
    except Exception:  # noqa: BLE001
        return RuntimeError("".join(traceback.format_exception(e)))


class Pending:
    """A call in flight. `wait()` returns its value (a SharedResult is
    attached) and sets `t_ready`, the `time.monotonic()` at which the value
    was ready (in whichever process computed it)."""

    def __init__(self, owner, tag, t_submit: float):
        self.owner, self.tag, self.t_submit = owner, tag, t_submit
        self.t_ready = 0.0
        self._value = None
        self._done = False

    def done(self) -> bool:
        return self._done or self.owner._ready(self.tag)

    def wait(self):
        if not self._done:
            status, value, self.t_ready = self.owner._take(self.tag)
            if status == "err":
                self._done = True
                raise value
            self._value = value.attach() if isinstance(value, SharedResult) else value
            self._done = True
        return self._value


class PoolProcess:
    """A worker pool (`rollout.create_pool`, or a server pool) owned by a
    process of its own: `call(fn, *args)` runs `fn(pool, *args)` there.
    Calls run one at a time in submission order. Not daemonic (a daemon may
    not have children): `close` or `terminate` it."""

    def __init__(self, workers: int, inference: str = "local", server_cfg=None, worker_cpus=None, cpus=None, nice: int = 0, name: str = "collector"):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, workers, inference, server_cfg, worker_cpus, cpus, nice), name=name)
        self.proc.start()
        child.close()
        self.name = name
        self._tag = 0
        self._replies: dict = {}

    def call(self, fn, *args) -> Pending:
        self._tag += 1
        self.conn.send((self._tag, fn, args))
        return Pending(self, self._tag, time.monotonic())

    def _pump(self) -> None:
        try:
            status, tag, value, t = self.conn.recv()
        except EOFError:
            raise RuntimeError(f"{self.name} process died (exit code {self.proc.exitcode})") from None
        if tag is None:  # the pool could not be built
            raise value
        self._replies[tag] = (status, value, t)

    def _ready(self, tag) -> bool:
        while tag not in self._replies and self.conn.poll():
            self._pump()
        return tag in self._replies

    def _take(self, tag):
        while tag not in self._replies:
            self._pump()
        return self._replies.pop(tag)

    def close(self, timeout: float = 60) -> None:
        if self.proc.is_alive():
            try:
                self.conn.send(None)
            except OSError:
                pass
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.terminate()
        self.conn.close()

    def terminate(self) -> None:
        """Stop now, abandoning a call in flight; the process terminates its pool."""
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(30)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()


class PoolThread:
    """`PoolProcess`'s interface over a pool in this process: calls run one
    at a time in a background thread (daemon, so a call abandoned after an
    error cannot block the exit once the pool is terminated)."""

    def __init__(self, pool):
        self.pool = pool
        self._exec = ThreadPoolExecutor(1, thread_name_prefix="rollout")
        self._tag = 0
        self._futures: dict = {}

    def call(self, fn, *args) -> Pending:
        def run():
            try:
                return ("ok", fn(self.pool, *args), time.monotonic())
            except BaseException as e:  # noqa: BLE001 - re-raised by wait()
                return ("err", e, time.monotonic())

        self._tag += 1
        self._futures[self._tag] = self._exec.submit(run)
        return Pending(self, self._tag, time.monotonic())

    def _ready(self, tag) -> bool:
        return self._futures[tag].done()

    def _take(self, tag):
        return self._futures.pop(tag).result()

    def close(self) -> None:
        self._exec.shutdown(wait=True)
        self.pool.close()
        self.pool.join()

    def terminate(self) -> None:
        self.pool.terminate()
        self._exec.shutdown(wait=False, cancel_futures=True)


# -- CPU layout ------------------------------------------------------------------


def parse_cpus(s: str) -> tuple[int, ...]:
    """'32-47,50' -> (32, ..., 47, 50); '' -> ()."""
    out: list[int] = []
    for part in filter(None, (p.strip() for p in s.split(","))):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


def fmt_cpus(cpus) -> str:
    """(32, 33, 34, 40) -> '32-34,40'."""
    cpus, runs = sorted(cpus or ()), []
    for c in cpus:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs) or "-"


def available_cpus() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def gpu_numa_cpus(device: str) -> tuple[int | None, tuple[int, ...]]:
    """(NUMA node, its CPUs) of a CUDA device, from sysfs; (None, ()) if unknown."""
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None, ()
        p = torch.cuda.get_device_properties(dev.index or 0)
        bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
            node = int(f.read())
        if node < 0:
            return None, ()
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return node, parse_cpus(f.read().strip())
    except (OSError, AttributeError, ValueError, RuntimeError):
        return None, ()


@dataclass
class Layout:
    trainer: tuple[int, ...]  # the trainer process (the update's torch threads)
    server: tuple[int, ...]  # the inference server, with --inference server
    evaluator: tuple[int, ...]  # the evaluation pool's workers (shared with the rollout workers if no core is spare)
    workers: tuple[int, ...]  # the rollout workers, one per core (empty: unpinned)
    node: int | None = None  # the GPU's NUMA node
    shared_eval: bool = False  # the evaluation runs niced on the workers' cores
    collector: tuple[int, ...] = ()  # the collector process: the workers' cores off the GPU's node if there are any

    def describe(self) -> str:
        node = f" (GPU NUMA node {self.node})" if self.node is not None else ""
        ev = fmt_cpus(self.evaluator) + (" shared with the workers, niced" if self.shared_eval else "")
        srv = f" | server {fmt_cpus(self.server)}" if self.server else ""
        return f"cpu layout{node}: trainer {fmt_cpus(self.trainer)}{srv} | workers {fmt_cpus(self.workers)} | collector {fmt_cpus(self.collector)} | evaluation {ev}"


def cpu_layout(workers: int, device: str = "cpu", server: bool = False, evaluation: bool = True, trainer_cpus: str = "", worker_cpus: str = "", eval_cpus: str = "", avail=None) -> Layout:
    """Split the CPUs this process may use (or `avail`). Unless given, the
    trainer takes one core, on the GPU's NUMA node when it is known, then
    the server one, the rollout workers one each from the other end (other
    nodes first), the evaluation up to 8 of what is left, and the trainer
    the rest. With no core left for the evaluation it shares the workers'
    cores at a lower priority. Too few cores for one per worker: workers
    unpinned (empty tuple)."""
    avail = tuple(avail if avail is not None else available_cpus())
    node, local = gpu_numa_cpus(device)
    order = [c for c in avail if c in local] + [c for c in avail if c not in local]
    given = {k: parse_cpus(v) for k, v in (("trainer", trainer_cpus), ("workers", worker_cpus), ("eval", eval_cpus)) if v}
    free = [c for c in order if not any(c in v for v in given.values())]
    take = lambda n, end=False: [free.pop(-1 if end else 0) for _ in range(min(n, len(free)))]  # noqa: E731
    trainer = given.get("trainer") or tuple(take(1))
    srv = tuple(take(1)) if server and len(free) > workers else ()
    if "workers" in given:
        wk = given["workers"]
    else:
        wk = tuple(sorted(take(workers, end=True))) if len(free) >= workers else ()
    ev = given.get("eval") or (tuple(take(min(8, len(free)))) if evaluation else ())
    if free and "trainer" not in given:
        trainer = tuple(sorted(set(trainer) | set(free)))
    shared = evaluation and not ev
    if shared:
        ev = wk or avail
    return Layout(tuple(trainer), srv, tuple(ev), tuple(wk), node, shared, tuple(c for c in wk if c not in local) or tuple(wk))
