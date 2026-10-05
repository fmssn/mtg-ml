"""End-to-end rollout throughput: engine + featurize + policy inference, the
way `rl.train` collects games, without the PPO update.

    python tools/bench_rollout.py --engine native --inference server --workers 32 --games 4096
    taskset -c 0-31 python tools/bench_rollout.py ...      # pin the whole run (workers inherit it)

A fresh network (default size) plays training-like games: half self-play,
half against a second checkpoint (two policies, as with the opponent pool).
Prints decisions/s per round after one warm-up round and, for the server,
its batch statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--inference", default="server", choices=("local", "server"))
    ap.add_argument("--device", default=None, help="server device (default: cuda if available)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=1024, help="games per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--memory", default="gru")
    ap.add_argument("--max-rows", type=int, default=16384)
    ap.add_argument("--server-cpus", default=None, help="CPUs reserved for the server, e.g. 32-33")
    ap.add_argument("--worker-cpus", default=None, help="CPUs the workers are pinned to (one each, round robin)")
    ap.add_argument("--groups", type=int, default=2, help="requests in flight per worker")
    args = ap.parse_args(argv)

    import multiprocessing as mp

    import torch

    from mtg_ml.backend import ENV_VAR, engine_name
    from mtg_ml.rl.inference import InferenceServer, ServerConfig, default_device
    from mtg_ml.rl.model import PolicyNet
    from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, run_job, split_games, worker_init

    os.environ[ENV_VAR] = engine_name(args.engine)

    def cpus(spec):
        if not spec:
            return None
        out = []
        for part in spec.split(","):
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
        return tuple(out)

    tmp = tempfile.mkdtemp(prefix="bench_rollout_")
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        net = PolicyNet(hidden=args.hidden, memory=args.memory)
        paths.append(os.path.join(tmp, f"p{k}.pt"))
        torch.save({"config": net.config, "model": net.state_dict()}, paths[-1])
    server = None
    if args.inference == "server":
        server = InferenceServer(
            args.workers, ServerConfig(device=args.device or default_device(), max_rows=args.max_rows, cpus=cpus(args.server_cpus)), worker_cpus=cpus(args.worker_cpus)
        )
        procs = server.pool()
    else:
        procs = mp.get_context("spawn").Pool(args.workers, initializer=worker_init)
    try:
        rates = []
        for r in range(args.rounds + 1):
            specs = []
            for g in range(args.games):
                seats = (LEARNER, LEARNER) if g % 2 == 0 else ((LEARNER, paths[1]) if g % 4 == 1 else (paths[1], LEARNER))
                specs.append(GameSpec(seed=r * 1_000_000 + g, seats=seats, match_game=1 + g % 2))
            jobs = [Job(c, paths[0], 1, record=True, inference=args.inference, groups=args.groups) for c in split_games(specs, args.workers)]
            t = time.perf_counter()
            res = procs.map(run_job, jobs)
            dt = time.perf_counter() - t
            n = sum(sum(gm[4] for gm in x.games) for x in res)
            if r == 0:
                print(f"warm-up: {n / dt:,.0f} decisions/s", flush=True)
                if server is not None:
                    server.stats()
                continue
            rates.append(n / dt)
            print(f"round {r}: {n} decisions in {dt:.1f}s = {n / dt:,.0f} decisions/s", flush=True)
        line = f"{args.engine} {args.inference} workers={args.workers} games/round={args.games}: {sum(rates) / len(rates):,.0f} decisions/s"
        if server is not None:
            st = server.stats()
            line += f" | server: {st['rows'] / max(st['batches'], 1):.0f} decisions/batch, {st['requests'] / max(st['batches'], 1):.1f} requests/batch, busy {st['busy_s']:.1f}s, device {server.cfg.device}"
        print(line)
    finally:
        procs.close()
        procs.join()
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
