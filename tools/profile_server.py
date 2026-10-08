"""Profile the inference server's batch processing in isolation (no worker
processes): real featurized decisions, written into the server's shared
blocks exactly as workers write them, rows split over `--policies` stacked
checkpoints (the learner gets half the rows, the pool the rest).

    CUDA_VISIBLE_DEVICES=<uuid> python tools/profile_server.py --device cuda --requests 31 --rows 32 --policies 26
    ... --no-graphs        # the same forward run eagerly (kernel launches from Python)
    ... --profile          # cProfile of the host side
"""

from __future__ import annotations

import argparse
import atexit
import cProfile
import os
import pstats
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _ListQ(list):
    def put(self, x):
        self.append(x)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--requests", type=int, default=31, help="requests per batch (one per worker group)")
    ap.add_argument("--rows", type=int, default=32, help="decisions per request")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--policies", type=int, default=26, help="distinct checkpoints (training: learner + pool)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--entity-attn", type=int, default=0)
    ap.add_argument("--features", type=int, default=0)
    ap.add_argument("--matchup", default="jund_blue")
    ap.add_argument("--min-entities", type=int, default=0, help="wide-board stress screen; 0 keeps all decisions")
    ap.add_argument("--init", help="inherit checkpoint architecture/weights, optionally adding attention")
    ap.add_argument("--legacy-attention", action="store_true", help="same-code eager attention baseline")
    ap.add_argument("--trunk", default="entity")
    ap.add_argument("--value-net", default="shared")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-compile", action="store_true", help="no torch.compile of the forward (ServerConfig.compile)")
    ap.add_argument("--kernels", action="store_true", help="torch.profiler table of the GPU kernels (eager forward)")
    args = ap.parse_args(argv)
    if min(args.requests, args.rows, args.policies, args.iters, args.warmup) < 1:
        ap.error("requests, rows, policies, iters and warmup must be positive")

    import torch

    from mtg_ml.backend import game_class
    from mtg_ml.match import game_args, parse_matchups
    from mtg_ml.rl.features import encode_event_hashes, featurize_flat
    from mtg_ml.rl.inference import InferenceClient, ServerConfig, _Server

    from mtg_ml.rl.model import PolicyNet, load_partial

    tmp = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tmp)
    keys = []
    for k in range(args.policies):
        torch.manual_seed(k)
        kw = torch.load(args.init, map_location="cpu", weights_only=False)["config"] if args.init else dict(hidden=args.hidden, trunk=args.trunk, value_net=args.value_net)
        kw = {**kw, "entity_attn": args.entity_attn or kw.get("entity_attn", 0)}
        if args.features:
            kw["features"] = args.features
        net = PolicyNet(**kw)
        if args.init:
            load_partial(net, torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        path = os.path.join(tmp, f"p{k}.pt")
        torch.save({"config": net.config, "model": net.state_dict()}, path)
        keys.append((path, 1 if k == 0 else 0))
    G = game_class("native")
    samples, r, s = [], random.Random(0), 0
    matchups = parse_matchups(args.matchup)
    while len(samples) < args.requests * args.rows:
        if s >= 10000:
            raise RuntimeError("could not collect enough wide-board decisions in 10000 games")
        matchup = r.choices([m for m, _ in matchups], [w for _, w in matchups])[0]
        g = G(**game_args(1 + s % 2, matchup), seed=s)
        while not g.over and len(samples) < args.requests * args.rows:
            st, ol, of = featurize_flat(g, g.decision.player, features=net.features)
            if list(st).count(net.config["state_dim"]) >= args.min_entities:
                samples.append((st, ol, of, encode_event_hashes([r.randrange(1 << 15) for _ in range(r.randrange(8))])))
            g.step(r.randrange(len(ol)))
        s += 1
    cfg = ServerConfig(device=args.device, graphs=not args.no_graphs, groups=1, compile=not args.no_compile, stacked_attention=not args.legacy_attention)
    srv = _Server(cfg, args.requests)
    atexit.register(srv.close)
    srv.resp_qs = [_ListQ() for _ in range(args.requests)]
    t = time.perf_counter()
    ids = {k: srv.register_key(k) for k in keys}
    print(f"loaded {len(keys)} policies in {time.perf_counter() - t:.2f}s ({srv.stack.nbytes() / 1e9:.2f} GB stacked)" if srv.stack else "")
    clients = []
    for w in range(args.requests):
        c = InferenceClient(w, None, None, srv.names, srv.layout)
        c.ids = dict(ids)
        clients.append(c)
        atexit.register(c.close)
    rng = random.Random(1)

    def submit_all():
        for w, c in enumerate(clients):
            rows = []
            for k in range(args.rows):  # half learner, half one pool opponent per request (as the claimer arranges)
                pol = keys[0] if k < args.rows // 2 or len(keys) == 1 else keys[1 + (w + k) % (len(keys) - 1)]
                rows.append((pol, k, int(rng.random() < 0.05), *samples[w * args.rows + k]))
            rows.sort(key=lambda x: x[0])
            c.submit(0, rows)

    def batch():
        submit_all()
        pend = srv.pending()
        assert len(pend) == args.requests
        srv.process(pend)
        srv.drain()

    t = time.perf_counter()
    for _ in range(args.warmup):
        batch()
    print(f"warm-up (captures): {time.perf_counter() - t:.2f}s, {srv.stats['graphs']} graphs")
    for k in ("busy_s", "prep_s", "launch_s", "infer_s", "rows", "requests", "padded_rows"):
        srv.stats[k] = 0.0
    srv.stats["batches"] = 0
    pr = cProfile.Profile() if args.profile else None
    sub = 0.0
    if pr:
        pr.enable()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        ts = time.perf_counter()
        submit_all()
        sub += time.perf_counter() - ts
        srv.process(srv.pending())
        srv.drain()
    dt = (time.perf_counter() - t0 - sub) / args.iters
    if pr:
        pr.disable()
    n = args.requests * args.rows
    st = srv.stats
    b = st["batches"]
    print(
        f"{args.device}{' eager' if args.no_graphs else ''}: {n} decisions in {args.requests} requests, {args.policies} policies: {dt * 1000:.2f} ms/batch = {n / dt:,.0f} decisions/s "
        f"(prep {st['prep_s'] / b * 1000:.3f}, launch {st['launch_s'] / b * 1000:.3f}, launch to answer {st['infer_s'] / b * 1000:.3f} ms; padded rows {st['padded_rows'] / max(b, 1):.0f})"
        + (f"; peak device memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if args.device.startswith("cuda") else "")
    )
    if args.device.startswith("cuda") and not args.no_graphs and srv.lanes[0].graphs:  # GPU time of one replay
        graph, _ = next(iter(srv.lanes[0].graphs.values()))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(50):
            graph.replay()
        e1.record()
        torch.cuda.synchronize()
        print(f"GPU time per graph replay: {e0.elapsed_time(e1) / 50 * 1000:.0f} us")
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(20)
    if args.kernels:  # GPU kernels of the forward, run eagerly
        from torch.profiler import ProfilerActivity, profile

        srv.cfg.graphs = False
        srv.pad_eager = not args.no_graphs  # the shapes the graphs run
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                batch()
            torch.cuda.synchronize()
        rows = sorted((e for e in prof.key_averages() if e.device_time_total > 0 and not e.key.startswith(("aten::", "Torch-Compiled", "## Call", "mtg::"))), key=lambda e: -e.self_device_time_total)
        b = 5
        print(f"GPU kernels per batch (eager{' compiled' if not args.no_compile else ''} forward at the graphs' shapes): {sum(e.self_device_time_total for e in rows) / b:.0f} us")
        for e in rows[:20]:
            print(f"{e.self_device_time_total / b:8.1f} us {e.count / b:4.0f}x  {e.key[:150]}")


if __name__ == "__main__":
    main()
