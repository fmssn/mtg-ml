"""Synchronous whole-trajectory PPO across persistent learner processes.

Rank zero alone collects, shuffles, publishes and checkpoints. Each global
optimizer minibatch is divided into complete trajectories. Gradients are
averaged before clipping/Adam; unequal and empty shards are weighted by the
global real-row count. CUDA graphs capture local forward/backward only:
communication and optimizer updates stay outside capture.
"""
from __future__ import annotations

import os
import tempfile
import time
from array import array
from datetime import timedelta
from itertools import accumulate

import torch
import torch.distributed as dist

from .collect import SharedResult, release
from .model import collate_packed, packed_tensors, split, structure
from .ppo import N_STATS, STATS, TRANSPOSED_BAGS, _StepGraphs, _by_kind, _clear_cublas_workspaces, _compiled_losses, _drop_entities, _layout, _losses, _padded_epoch, _padded_losses, load_optimizer_state, make_optimizer, set_lr, trajectory_minibatches
from .rollout import KINDS


def trajectory_shards(trajectories, world):
    """Balance complete (source-row start, length) trajectories, deterministically."""
    shards, work = [[] for _ in range(world)], [0] * world
    for start, length in sorted(trajectories, key=lambda t: (-t[1], t[0])):
        rank = min(range(world), key=lambda r: (work[r], r))
        shards[rank].append((start, length))
        work[rank] += length
    return [sorted(shard, key=lambda t: trajectories.index(t)) for shard in shards]


class _BackwardGraphs(_StepGraphs):
    """All graph shapes write the same persistent flattened gradient buffer."""

    def __init__(self, net, opt, cfg):
        super().__init__(net, opt, cfg)
        params = list(net.parameters())
        self.grad = torch.zeros(sum(p.numel() for p in params), device=params[0].device)
        off = 0
        for p in params:
            p.grad = self.grad[off : off + p.numel()].view_as(p)
            off += p.numel()
        self.fingerprint = self.fingerprint_of(net, opt, cfg)

    @staticmethod
    def fingerprint_of(net, opt, cfg):
        return (tuple((p.data_ptr(), p.grad.data_ptr() if p.grad is not None else None) for p in net.parameters()),
                cfg.clip, cfg.vf_coef, cfg.ent_coef, cfg.capture, cfg.precision)

    def prepare(self, shapes):
        key = tuple(sorted(shapes.items()))
        if key not in self.graphs:
            if len(self.graphs) >= self.MAX_SHAPES:
                old = next(iter(self.graphs))
                del self.graphs[old], self.pools[old]
            self.graphs[key], self.pools[key] = {}, torch.cuda.graph_pool_handle()
        return key

    def backward(self, net, cfg, shapes, gru, ints, flts):
        key = self.prepare(shapes)
        if gru not in self.graphs[key]:
            self.graphs[key][gru] = self._capture_backward(net, cfg, shapes, gru, ints, flts, self.pools[key])
        graph, si, sf = self.graphs[key][gru]
        si.copy_(ints)
        sf.copy_(flts)
        graph.replay()

    def _capture_backward(self, net, cfg, shapes, gru, ints, flts, pool):
        losses = _compiled_losses() if cfg.capture >= 2 else _padded_losses
        si, sf = ints.clone(), flts.clone()
        t = time.monotonic()
        main, side = torch.cuda.current_stream(), self.stream
        side.wait_stream(main)
        _drop_entities(net)
        with torch.cuda.stream(side):
            for _ in range(2):
                self.grad.zero_()
                loss, _ = losses(net, cfg, shapes, gru, si, sf)
                loss.backward()
                del loss
        main.wait_stream(side)
        _drop_entities(net)
        graph = torch.cuda.CUDAGraph()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            _clear_cublas_workspaces()
            graph.capture_begin(pool=pool)
            try:
                self.grad.zero_()
                loss, stats = losses(net, cfg, shapes, gru, si, sf)
                loss.backward()
                self.acc.add_(stats)
                del loss, stats
            finally:
                graph.capture_end()
                _clear_cublas_workspaces()
        main.wait_stream(side)
        _drop_entities(net)
        self.captures += 1
        self.capture_s += time.monotonic() - t
        return graph, si, sf


def distributed_update(net, opt, data, cfg, rank, world, gen=None, graphs=None, control=None):
    """One global PPO update; every rank returns the same reduced statistics."""
    dev = next(net.parameters()).device
    n = len(data.actions)
    if not n:
        return {**{k: 0.0 for k in STATS}, "updates": 0, "early_stop": False, "explained_var": float("nan")}
    floats = lambda xs: torch.frombuffer(array("f", xs), dtype=torch.float32)  # noqa: E731
    old, adv, ret = floats(data.logps), floats(data.advantages), floats(data.returns)
    variance = ret.var().item() if n > 1 else 0.0
    explained = float("nan") if not variance else 1 - adv.var().item() / variance
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if n > 1 else adv - adv.mean()
    rec = torch.stack([old, adv, ret]).to(dev)
    samples = packed_tensors(data.samples, dev)
    actions = torch.tensor(data.actions, dtype=torch.long, device=dev)
    kinds = torch.tensor(list(data.kinds) if len(data.kinds) == n else [0] * n, dtype=torch.long, device=dev)
    widths = torch.frombuffer(data.samples.n_opts, dtype=torch.int32)
    net.train()
    totals = torch.zeros(N_STATS, dtype=torch.float64, device=dev)
    steps, stopped, epoch_kl = 0, False, []
    caps0 = graphs.captures if graphs else 0
    sd, od = net.config["state_dim"], net.config["option_dim"]
    transpose = {"st": (sd, "bag_off" if net.config["trunk"] == "entity" else "st_off"), "e": (od, "e_off"), "ot": (od, "ot_off")} if TRANSPOSED_BAGS else None
    for _ in range(cfg.epochs):
        schedule = None
        if rank == 0:
            order, global_chunks = trajectory_minibatches(data.lengths, cfg.minibatch, gen)
            pos, schedule = 0, []
            for chunk in global_chunks:
                trajectories = []
                for length in chunk:
                    trajectories.append((int(order[pos]), length))
                    pos += length
                schedule.append(trajectories)
        box = [schedule]
        if control is not None:
            dist.monitored_barrier(group=control, timeout=timedelta(seconds=600), wait_all_ranks=True)
        dist.broadcast_object_list(box, src=0)
        schedule = box[0]
        local = [trajectory_shards(batch, world)[rank] for batch in schedule]
        counts = [sum(length for _, length in shard) for shard in local]
        # Empty ranks run a valid, zero-weight dummy row and every collective.
        local = [shard or [(0, 1)] for shard in local]
        chunks = [[length for _, length in shard] for shard in local]
        idx = torch.cat([torch.arange(start, start + length) for shard in local for start, length in shard])
        bounds = list(accumulate((sum(c) for c in chunks), initial=0))
        rows = idx.to(dev)
        big = structure(collate_packed(samples, rows), net.config["state_dim"], net.config["option_dim"])
        acts, records, kind = actions[rows], rec[:, rows], kinds[rows]
        width = widths[idx]
        acc = graphs.acc if graphs else torch.zeros_like(totals)
        acc.zero_()
        if graphs:
            shapes, gru, ints, flts = _padded_epoch(big, bounds, chunks, acts, records, width, graphs.choose, transpose,
                                                  choose_gru=graphs.choose_gru, kinds=kind, ent_width=bool(net.config.get("entity_attn")))
            layout = _layout(shapes)
            _, n_start, _ = layout["n"]
            _, w_start, w_len = layout["weight"]
            for m, batch in enumerate(schedule):
                flts[m, n_start] = sum(length for _, length in batch) / world
                if not counts[m]:
                    flts[m, w_start : w_start + w_len].zero_()
                    for name in ("old_logp", "adv", "ret"):
                        _, start, length = layout[name]
                        flts[m, start : start + length].zero_()
            del big
        pieces = None if graphs else split(big, bounds)
        params = list(net.parameters())
        if graphs:
            grad = graphs.grad
        else:
            grad = torch.zeros(sum(p.numel() for p in params), device=dev)
        for m, batch in enumerate(schedule):
            used = None
            if graphs:
                graphs.prepare(shapes)
                if any(p not in opt.state for p in params):
                    # Match ordinary PPO: Adam's first step uses the uncompiled padded loss.
                    grad.zero_()
                    loss, stats = _padded_losses(net, cfg, shapes, gru[m], ints[m], flts[m])
                    loss.backward()
                    acc.add_(stats)
                    del loss
                else:
                    graphs.backward(net, cfg, shapes, gru[m], ints[m], flts[m])
            else:
                grad.zero_()
                opt.zero_grad(set_to_none=True)
                lo, hi = bounds[m : m + 2]
                weight = torch.ones(hi - lo, device=dev) if counts[m] else torch.zeros(hi - lo, device=dev)
                denominator = torch.tensor(sum(length for _, length in batch) / world, device=dev)
                local_rec = records[:, lo:hi] if counts[m] else torch.zeros_like(records[:, lo:hi])
                loss, stats = _losses(net, cfg, pieces[m], chunks[m], int(width[lo:hi].max()), acts[lo:hi], *local_rec,
                                     w=weight, n=denominator, kind=kind[lo:hi])
                loss.backward()
                used = torch.tensor([p.grad is not None for p in params], device=dev, dtype=torch.int32)
                off = 0
                for p in params:
                    if p.grad is not None:
                        grad[off : off + p.numel()].copy_(p.grad.flatten())
                    off += p.numel()
                acc.add_(stats)
                del loss
            # CUDA collectives enqueue asynchronously. A faster rank must not
            # enqueue reductions while a peer compiles/captures a cold shape:
            # that host work can exceed NCCL's failure-detection timeout.
            # Gloo coordinates host readiness without outstanding GPU reads.
            if control is not None:
                dist.monitored_barrier(group=control, timeout=timedelta(seconds=600), wait_all_ranks=True)
            if used is not None:
                dist.all_reduce(used, op=dist.ReduceOp.MAX)
                off = 0
                for p, active in zip(params, used.tolist()):
                    p.grad = grad[off : off + p.numel()].view_as(p) if active else None
                    off += p.numel()
            dist.all_reduce(grad)
            grad.div_(world)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
        epoch = acc.clone()
        epoch[: len(STATS)].div_(world)  # losses were scaled for gradient averaging
        dist.all_reduce(epoch)
        steps += len(schedule)
        totals.add_(epoch)
        values = epoch.tolist()
        nt_n, _, nt_kl = _by_kind(values).sum(0).tolist()
        kl = values[STATS.index("approx_kl")] / len(schedule)
        epoch_kl.append((kl, nt_kl / nt_n if nt_n else float("nan")))
        if cfg.target_kl is not None and kl > cfg.target_kl:
            stopped = True
            break
    values = totals.tolist()
    out = {k: v / steps for k, v in zip(STATS, values)}
    out.update(updates=steps, early_stop=stopped, explained_var=explained, learner_world_size=world,
               approx_kl_first=epoch_kl[0][0], nt_approx_kl_first=epoch_kl[0][1],
               approx_kl_last=epoch_kl[-1][0], nt_approx_kl_last=epoch_kl[-1][1])
    nt_n, nt_ent, nt_kl = _by_kind(values).sum(0).tolist()
    out.update(nt_frac=nt_n / (n * len(epoch_kl)), nt_entropy=nt_ent / nt_n if nt_n else float("nan"), nt_approx_kl=nt_kl / nt_n if nt_n else float("nan"))
    for name, (count, ent, kl) in zip(KINDS, _by_kind(values).tolist()):
        if count:
            out.update({f"kind/{name}/share": count / nt_n, f"kind/{name}/entropy": ent / count, f"kind/{name}/approx_kl": kl / count})
    if graphs:
        out.update(captures=graphs.captures - caps0, graph_shapes=len(graphs.graphs))
    if dev.type == "cuda":
        peaks = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world)]
        dist.all_gather(peaks, torch.tensor([torch.cuda.max_memory_allocated(dev)], dtype=torch.long, device=dev))
        out["learner_peak_bytes_by_rank"] = [int(p.item()) for p in peaks]
    return out


def _init_group(rank, devices, init):
    dev = torch.device(devices[rank])
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", init_method=init, rank=rank, world_size=len(devices), timeout=timedelta(seconds=90))
    return dist.new_group(backend="gloo", timeout=timedelta(seconds=600)) if dev.type == "cuda" else None


def _worker(rank, devices, init, config, state, optimizer, cfg, cpus, conn):
    from .model import PolicyNet

    data = None
    try:
        torch.set_num_threads(1)
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
        net = PolicyNet(**config).to(devices[rank])
        net.load_state_dict(state)
        opt = make_optimizer(net.parameters(), cfg.lr, devices[rank])
        load_optimizer_state(opt, optimizer)
        control = _init_group(rank, devices, init)
        graphs = None
        while True:
            msg = conn.recv()
            if msg is None:
                break
            shared, cfg, lr = msg
            data = shared.attach(unlink=False)
            conn.send("ready")
            set_lr(opt, lr)
            if cfg.capture and next(net.parameters()).is_cuda:
                if graphs is None or graphs.fingerprint != _BackwardGraphs.fingerprint_of(net, opt, cfg):
                    graphs = _BackwardGraphs(net, opt, cfg)
            else:
                graphs = None
            distributed_update(net, opt, data, cfg, rank, len(devices), graphs=graphs, control=control)
            release(data)
            data = None
            conn.send(None)
    except BaseException:
        import traceback

        conn.send(traceback.format_exc())
        raise
    finally:
        if data is not None:
            release(data)
        if dist.is_initialized():
            dist.destroy_process_group()
        conn.close()


class DistributedLearner:
    """Rank zero remains the trainer; other ranks hold synchronized replicas."""

    def __init__(self, net, opt, cfg, devices, cpus=()):
        import multiprocessing as mp
        from .train import _cpu

        if len(devices) < 2:
            raise ValueError("distributed learning requires at least two devices")
        if next(net.parameters()).device != torch.device(devices[0]):
            raise ValueError("first learner device must match --device")
        kinds = {torch.device(d).type for d in devices}
        if len(kinds) != 1 or not kinds <= {"cpu", "cuda"}:
            raise ValueError("learner devices must all be CPU or all CUDA")
        if kinds == {"cuda"} and len(set(devices)) != len(devices):
            raise ValueError("CUDA learner devices must be distinct")
        if cpus and len(cpus) != len(devices):
            raise ValueError("one CPU list is required per learner rank")
        if dist.is_initialized():
            raise RuntimeError("a distributed process group already exists")
        fd, self.path = tempfile.mkstemp(prefix="mtg_ppo_group_")
        os.close(fd)
        init = "file://" + self.path
        ctx = mp.get_context("spawn")
        self.devices, self.procs, self.conns = devices, [], []
        self.graphs = None
        state, optimizer = _cpu(net.state_dict()), _cpu(opt.state_dict())
        try:
            for rank in range(1, len(devices)):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(rank, devices, init, net.config, state, optimizer, cfg, cpus[rank] if cpus else (), child), name=f"ppo-learner-{rank}")
                proc.start()
                child.close()
                self.procs.append(proc)
                self.conns.append(parent)
            self.control = _init_group(0, devices, init)
        except BaseException:
            self.close()
            raise

    def update(self, net, opt, data, cfg, lr, gen=None):
        from multiprocessing import shared_memory

        shared = SharedResult(data)
        try:
            for conn in self.conns:
                conn.send((shared, cfg, lr))
            ready = self._replies(90)
            if ready != ["ready"] * len(self.procs):
                raise RuntimeError(f"learner failed before update: {ready}")
            if cfg.capture and next(net.parameters()).is_cuda:
                if self.graphs is None or self.graphs.fingerprint != _BackwardGraphs.fingerprint_of(net, opt, cfg):
                    self.graphs = _BackwardGraphs(net, opt, cfg)
            else:
                self.graphs = None
            stats = distributed_update(net, opt, data, cfg, 0, len(self.devices), gen=gen, graphs=self.graphs, control=self.control)
            errors = self._replies(90)
            if any(errors):
                raise RuntimeError(f"learner failed: {errors}")
            return stats
        finally:
            block = shared_memory.SharedMemory(name=shared.name)
            block.unlink()
            block.close()

    def _replies(self, timeout):
        deadline, out = time.monotonic() + timeout, []
        for proc, conn in zip(self.procs, self.conns):
            while not conn.poll(0.1):
                if not proc.is_alive():
                    raise RuntimeError(f"learner rank died (PID {proc.pid}, exit {proc.exitcode})")
                if time.monotonic() > deadline:
                    raise TimeoutError("learner acknowledgement timed out")
            try:
                out.append(conn.recv())
            except EOFError:
                raise RuntimeError(f"learner rank died (PID {proc.pid}, exit {proc.exitcode})") from None
        return out

    def close(self):
        for proc, conn in zip(self.procs, self.conns):
            if proc.is_alive():
                try:
                    conn.send(None)
                except (OSError, EOFError):
                    pass
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            conn.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(self.path):
            os.unlink(self.path)
