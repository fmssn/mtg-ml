"""End-to-end rollout throughput: engine + featurize + policy inference, the
way `rl.train` collects games, without the PPO update.

    python tools/bench_rollout.py --engine native --inference server --workers 32 --games 4096
    taskset -c 0-31 python tools/bench_rollout.py ...      # pin the whole run (workers inherit it)
    python tools/bench_rollout.py --run runs/x ...         # a trained run's learner and opponent pool

Fresh networks (or a run's checkpoints) play training-shaped games, drawn as
`Trainer._train_specs` draws them: half self-play, half against the opponent
pool (the newest snapshot half the time), half of all games post-sideboard.
The learner gets a new version every round, so workers reload it as in
training. Games go through the shared game queue (`rollout.run_specs`), or
with `--split` through the fixed split (`split_games` + `run_job`). Prints
decisions/s per round after one warm-up round, the spread of job wall times
(loading the learner included), the share of time jobs wait for inference
(local inference: run it) and, for the server, its batch statistics.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from dataclasses import replace

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
    ap.add_argument("--trunk", default="mlp", choices=("mlp", "transformer", "entity"))
    ap.add_argument("--value-net", default="shared", choices=("shared", "separate"))
    ap.add_argument("--pool", type=int, default=1, help="fresh frozen opponents (each one more policy, like the opponent pool)")
    ap.add_argument("--run", default=None, help="a training run's checkpoints instead of fresh nets: RUN/latest.pt and RUN/pool/*.pt")
    ap.add_argument("--split", action="store_true", help="fixed split of the games over the jobs (before the shared queue)")
    ap.add_argument("--inflight", type=int, default=64, help="live games per worker with the shared queue")
    ap.add_argument("--max-rows", type=int, default=16384)
    ap.add_argument("--server-cpus", default=None, help="CPUs reserved for the server, e.g. 32-33; one spec per server separated by ';' with --devices")
    ap.add_argument("--devices", default=None, help="one inference server per device, e.g. cuda:0,cuda:1 (workers sharded over them)")
    ap.add_argument("--no-graphs", action="store_true", help="server runs the forward eagerly (no CUDA graphs)")
    ap.add_argument("--compile", action="store_true", help="server compiles its forward (torch.compile)")
    ap.add_argument("--worker-cpus", default=None, help="CPUs the workers are pinned to (one each, round robin)")
    ap.add_argument("--groups", type=int, default=2, help="requests in flight per worker")
    ap.add_argument("--dry-run", action="store_true", help="server returns random options without running the network (pipeline overhead only)")
    args = ap.parse_args(argv)

    import torch

    from mtg_ml.backend import ENV_VAR, engine_name
    from mtg_ml.rl.inference import InferenceServer, ServerConfig, default_device
    from mtg_ml.rl.model import PolicyNet
    from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, Result, create_pool, run_job, run_specs, split_games

    os.environ[ENV_VAR] = engine_name(args.engine)

    from mtg_ml.rl.inference import parse_cpus as cpus

    if args.run:
        learner = os.path.join(args.run, "latest.pt")
        pool = sorted(os.path.join(args.run, "pool", f) for f in os.listdir(os.path.join(args.run, "pool")))
    else:
        tmp = tempfile.mkdtemp(prefix="bench_rollout_")
        paths = []
        for k in range(1 + args.pool):
            torch.manual_seed(k)
            net = PolicyNet(hidden=args.hidden, memory=args.memory, trunk=args.trunk, value_net=args.value_net)
            paths.append(os.path.join(tmp, f"p{k}.pt"))
            torch.save({"config": net.config, "model": net.state_dict()}, paths[-1])
        learner, pool = paths[0], paths[1:]
    server = None
    if args.inference == "server":
        devices = args.devices.split(",") if args.devices else None
        per_server = [cpus(x) for x in args.server_cpus.split(";")] if args.server_cpus and ";" in args.server_cpus else None
        cfg = ServerConfig(
            device=args.device or default_device(), max_rows=args.max_rows, cpus=None if per_server else cpus(args.server_cpus),
            dry_run=args.dry_run, graphs=not args.no_graphs, groups=max(4, args.groups), compile=args.compile,
        )  # fmt: skip
        server = InferenceServer(args.workers, cfg, worker_cpus=cpus(args.worker_cpus), devices=devices, server_cpus=per_server)
    procs = create_pool(args.workers, args.inference, server, cpus(args.worker_cpus))
    try:
        rates = []
        for r in range(args.rounds + 1):
            specs, rng = [], random.Random(r)
            for g in range(args.games):
                opp = pool[-1] if rng.random() < 0.5 else rng.choice(pool)
                seats = (LEARNER, LEARNER) if g % 2 == 0 else ((LEARNER, opp) if rng.random() < 0.5 else (opp, LEARNER))
                specs.append(GameSpec(seed=r * 1_000_000 + g, seats=seats, match_game=1 + (rng.random() < 0.5)))
            job = Job([], learner, r + 1, record=True, inference=args.inference, groups=args.groups)  # a new learner version every round
            t = time.perf_counter()
            if args.split:
                res = Result()
                for x in procs.map(run_job, [replace(job, games=c) for c in split_games(specs, args.workers)]):
                    for f in ("games", "actions", "timing"):
                        getattr(res, f).extend(getattr(x, f))
            else:
                res = run_specs(procs, specs, job, args.workers, args.inflight)
            dt = time.perf_counter() - t
            n = sum(gm[4] for gm in res.games)
            if r == 0:
                print(f"warm-up: {n / dt:,.0f} decisions/s", flush=True)
                if server is not None:
                    st0 = server.stats()  # counters from here on: the measured rounds only
                continue
            rates.append(n / dt)
            walls = sorted(t[0] for t in res.timing)
            wait = sum(t[1] for t in res.timing) / max(sum(t[0] for t in res.timing), 1e-9)
            print(
                f"round {r}: {n} decisions ({len(res.actions)} recorded) in {dt:.2f}s = {n / dt:,.0f} decisions/s; "
                f"job wall min/median/max {walls[0]:.2f}/{walls[len(walls) // 2]:.2f}/{walls[-1]:.2f}s, waiting for inference {wait:.0%}",
                flush=True,
            )
        mode = "split" if args.split else f"queue inflight={args.inflight}"
        line = f"h{args.hidden} {args.trunk} value={args.value_net} {args.engine} {args.inference} {mode} workers={args.workers} games/round={args.games} policies={1 + len(pool)}: {sum(rates) / len(rates):,.0f} decisions/s"
        if server is not None:
            st = server.stats()
            st = {k: v - st0[k] if k in ("batches", "rows", "requests", "padded_rows", "eager", "legacy") or k.endswith("_s") else v for k, v in st.items()}
            b = max(st["batches"], 1)
            line += (
                f" | server x{len(server.devices)}: {st['rows'] / b:.0f} decisions/batch (padded {st['padded_rows'] / b:.0f}), {st['requests'] / b:.1f} requests/batch, "
                f"{st['busy_s'] / b * 1000:.2f} ms/batch busy (prep {st['prep_s'] / b * 1000:.2f}, infer {st['infer_s'] / b * 1000:.2f}, reply {st['reply_s'] / b * 1000:.2f}), "
                f"busy {st['busy_s']:.1f}s, {st['graphs']} graphs ({st['capture_s']:.1f}s capturing), {st['eager']} eager, {st['legacy']} legacy batches, "
                f"devices {','.join(server.devices)}{' DRY RUN' if args.dry_run else ''}"
            )
        print(line)
    finally:
        procs.close()
        procs.join()
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
