"""Profile the inference server's batch processing in isolation (no workers):
real featurized decisions packed exactly as workers pack them.

    CUDA_VISIBLE_DEVICES=4 python tools/profile_server.py --device cuda --requests 16 --rows 32
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import random
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
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--rows", type=int, default=32, help="decisions per request")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--policies", type=int, default=1, help="distinct checkpoints, rows split evenly (training: learner + pool)")
    args = ap.parse_args(argv)

    import torch

    from mtg_ml.backend import game_class
    from mtg_ml.match import match_decks
    from mtg_ml.rl.features import encode_event_hashes, featurize
    from mtg_ml.rl.inference import InferenceClient, ServerConfig, _Server
    from mtg_ml.rl.model import PolicyNet

    tmp = tempfile.mkdtemp()
    paths = []
    for k in range(args.policies):
        net = PolicyNet()
        paths.append(os.path.join(tmp, f"p{k}.pt"))
        torch.save({"config": net.config, "model": net.state_dict()}, paths[-1])
    # real decisions
    G = game_class("native" if "--python" not in sys.argv else "python")
    samples, r = [], random.Random(0)
    s = 0
    while len(samples) < args.requests * args.rows:
        g = G(match_decks(1), seed=s)
        while not g.over and len(samples) < args.requests * args.rows:
            st, opts = featurize(g, g.decision.player)
            samples.append((st, [len(o) for o in opts], [t for o in opts for t in o], encode_event_hashes([1, 2, 3])))  # flat form, as featurize_flat
            g.step(r.randrange(len(g.legal_options())))
        s += 1
    q = _ListQ()
    clients = [InferenceClient(w, q, _ListQ(), groups=1) for w in range(args.requests)]
    srv = _Server(ServerConfig(device=args.device), args.requests)
    srv.resp_qs = [c.resp_q for c in clients]

    def make_msgs():
        q.clear()
        for w, c in enumerate(clients):
            rows = [((paths[k * args.policies // args.rows], 1), k, int(k == 0), *samples[w * args.rows + k]) for k in range(args.rows)]
            c.submit(0, rows)
        return list(q)

    msgs = make_msgs()
    for _ in range(5):
        srv.process(msgs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    pr = cProfile.Profile() if args.profile else None
    t = time.perf_counter()
    if pr:
        pr.enable()
    for _ in range(args.iters):
        srv.process(msgs)
    if pr:
        pr.disable()
    dt = (time.perf_counter() - t) / args.iters
    n = args.requests * args.rows
    print(f"{args.device}: {n} decisions in {args.requests} requests: {dt * 1000:.2f} ms/batch = {n / dt:,.0f} decisions/s")
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(18)
    for c in clients:
        c.close()


if __name__ == "__main__":
    main()
