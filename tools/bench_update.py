"""PPO update speed: ms per optimizer step and projected seconds per iteration.

    python tools/bench_update.py --games 512 --workers 8 --save-data /tmp/upd.pkl   # play games, keep them
    python tools/bench_update.py --data /tmp/upd.pkl --device cuda --trunk entity   # time the update on them

The decisions come from N native-engine games played by a fresh network
through `rollout.run_job` (local inference, half self-play, half against a
second checkpoint, games 1 and 2, as in tools/bench_rollout.py), or from a
pickled `Result` (`--data`); `--save-data` writes one, so before/after
measurements run on identical decisions. The update runs `--warmup-epochs`
epochs, then `--epochs` timed epochs with a fixed minibatch order and no KL
early stop. The projection scales the timed update (all epochs) to
`--iter-decisions` decisions, the size of one training iteration of the h128
entity run.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pickle
import pstats
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def play(args, config: dict):
    import multiprocessing as mp
    from dataclasses import fields

    import torch

    from mtg_ml.rl.model import PolicyNet, load_partial
    from mtg_ml.match import parse_matchups
    from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, Result, run_job, split_games, worker_init

    tmp = tempfile.mkdtemp(prefix="bench_update_")
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        paths.append(os.path.join(tmp, f"p{k}.pt"))
        net = PolicyNet(**config)
        if args.init:
            load_partial(net, torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        torch.save({"config": config, "model": net.state_dict()}, paths[-1])
    specs = []
    import random

    rng, matchups = random.Random(0), parse_matchups(args.matchup)
    for g in range(args.games):
        seats = (LEARNER, LEARNER) if g % 2 == 0 else ((LEARNER, paths[1]) if g % 4 == 1 else (paths[1], LEARNER))
        matchup = rng.choices([m for m, _ in matchups], [w for _, w in matchups])[0]
        specs.append(GameSpec(seed=g, seats=seats, match_game=1 + (rng.random() < args.postboard_frac), matchup=matchup))
    jobs = [Job(c, paths[0], 1, engine=args.engine) for c in split_games(specs, args.workers)]
    t = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, initializer=worker_init) as procs:
        results = procs.map(run_job, jobs)
    data = Result()
    for r in results:
        for f in fields(Result):
            getattr(data, f.name).extend(getattr(r, f.name))
    print(f"played {args.games} games, {len(data.actions)} decisions in {time.perf_counter() - t:.1f}s")
    return data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="pickled Result to train on (instead of playing games)")
    ap.add_argument("--save-data", default=None, help="write the played Result here")
    ap.add_argument("--collect-only", action="store_true", help="save representative rollout data without updating")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--entity-attn", type=int, default=0)
    ap.add_argument("--features", type=int, default=0)
    ap.add_argument("--init", help="inherit a checkpoint's architecture and weights; attention may be added as identity")
    ap.add_argument("--matchup", default="jund_blue")
    ap.add_argument("--postboard-frac", type=float, default=0.5)
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--learner-devices", help="comma-separated devices, first matches --device")
    ap.add_argument("--learner-cpus", help="semicolon-separated CPU lists, one per learner")
    ap.add_argument("--trunk", default="entity", choices=("mlp", "transformer", "entity"))
    ap.add_argument("--value-net", default="shared", choices=("shared", "separate"))
    ap.add_argument("--memory", default="gru", choices=("gru", "none"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4, help="timed epochs")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--warmup-updates", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--iter-decisions", type=int, default=378_000)
    ap.add_argument("--capture", type=int, default=2, help="PPOConfig.capture (2: also compile the forward and losses)")
    ap.add_argument("--gru-precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--mode", default=None, choices=("eager", "padded", "graph"), help="ppo_update mode (default: graph on CUDA)")
    ap.add_argument("--profile", action="store_true", help="cProfile the timed epochs")
    ap.add_argument("--torch-profile", action="store_true", help="torch.profiler table (CUDA kernels) of the timed epochs")
    ap.add_argument("--torch-profile-json", default=None, help="with --torch-profile: also write every kernel row (name, self CUDA us, calls) here, for tools/profile_breakdown.py")
    args = ap.parse_args(argv)
    if min(args.minibatch, args.epochs, args.warmup_epochs, args.warmup_updates, args.repeats) < 1:
        ap.error("minibatch, epochs, warmup epochs/updates and repeats must be positive")

    import torch

    from mtg_ml.rl.model import PolicyNet, load_partial
    from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update

    kw = torch.load(args.init, map_location="cpu", weights_only=False)["config"] if args.init else dict(hidden=args.hidden, memory=args.memory, trunk=args.trunk, value_net=args.value_net)
    kw = {**kw, "entity_attn": args.entity_attn or kw.get("entity_attn", 0)}
    if args.features:
        kw["features"] = args.features
    config = PolicyNet(**kw).config
    if args.data:
        with open(args.data, "rb") as f:
            data = pickle.load(f)
    else:
        data = play(args, config)
    if args.save_data:
        with open(args.save_data, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.collect_only:
        if not args.save_data:
            raise ValueError("--collect-only requires --save-data")
        return
    n, lens = len(data.actions), data.lengths
    print(f"{n} decisions in {len(lens)} trajectories (mean {n / len(lens):.0f}, max {max(lens)}), {len(data.samples.s_idx) / n:.0f} state tokens per decision")

    dev = torch.device(args.device)
    torch.manual_seed(0)
    net = PolicyNet(**config).to(dev)
    if args.init:
        load_partial(net, torch.load(args.init, map_location="cpu", weights_only=False)["model"])
    cfg = PPOConfig(minibatch=args.minibatch, epochs=args.warmup_epochs, target_kl=None, capture=args.capture, precision=args.precision, gru_precision=args.gru_precision)
    opt = make_optimizer(net.parameters(), cfg.lr, dev)
    learner = None
    if args.learner_devices:
        import atexit

        from mtg_ml.rl.collect import parse_cpus
        from mtg_ml.rl.distributed import DistributedLearner

        if args.mode == "padded":
            raise ValueError("distributed benchmarking supports eager or graph mode")
        if args.mode == "eager":
            cfg.capture = 0
        cpus = tuple(parse_cpus(s) for s in args.learner_cpus.split(";")) if args.learner_cpus else ()
        learner = DistributedLearner(net, opt, cfg, tuple(args.learner_devices.split(",")), cpus)
        atexit.register(learner.close)
    update = lambda: learner.update(net, opt, data, cfg, cfg.lr, gen=gen) if learner else ppo_update(net, opt, data, cfg, device=dev, gen=gen, mode=args.mode)  # noqa: E731
    gen = torch.Generator().manual_seed(0)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    t = time.perf_counter()
    for _ in range(args.warmup_updates):
        update()
    sync()
    warm = time.perf_counter() - t
    cfg.epochs = args.epochs
    pr = cProfile.Profile() if args.profile else None
    if pr:
        pr.enable()
    from mtg_ml.rl import ppo

    g0 = learner.graphs if learner else ppo._GRAPHS.get(opt)
    caps0, cap_s0 = (g0.captures, g0.capture_s) if g0 else (0, 0.0)
    tp = None
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        tp = profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else []))
        tp.__enter__()
    t = time.perf_counter()
    steps = 0
    for _ in range(args.repeats):
        stats = update()
        steps += stats["updates"]
    sync()
    dt = time.perf_counter() - t
    if pr:
        pr.disable()
    if tp:
        tp.__exit__(None, None, None)
    mem = f", peak device memory {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" if dev.type == "cuda" else ""
    print(
        f"config={config} precision={args.precision} {dev}: {steps} steps of ~{n * args.epochs * args.repeats / steps:.0f} decisions "
        f"in {dt:.2f}s = {dt / steps * 1000:.2f} ms/step; projected {dt * args.iter_decisions / n / args.repeats:.1f}s per iteration of "
        f"{args.iter_decisions} decisions x {args.epochs} epochs{mem}"
    )
    g = learner.graphs if learner else ppo._GRAPHS.get(opt)
    print(
        f"mode={args.mode or 'default'} capture={args.capture} minibatch={args.minibatch}: {dt / args.epochs / args.repeats:.3f} s/epoch = {dt / args.epochs / args.repeats * 250_000 / n:.3f} s per epoch of 250k decisions; "
        f"warm-up {warm:.2f}s"
        + (
            f"; {g.captures - caps0} graphs captured in the timed epochs ({g.capture_s - cap_s0:.2f}s; without them {(dt - g.capture_s + cap_s0) / args.epochs / args.repeats:.3f} s/epoch), "
            f"{g.captures} in all; shapes {[dict(k) | {'gru': sorted(v)} for k, v in g.graphs.items()]}"
            if g
            else ""
        )
    )
    print({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()})
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)
    if tp and args.torch_profile_json:
        import json

        rows = [{"name": e.key, "cuda_us": e.self_device_time_total, "cpu_us": e.self_cpu_time_total, "calls": e.count} for e in tp.key_averages()]
        with open(args.torch_profile_json, "w") as f:
            json.dump({"steps": steps, "wall_s": dt, "rows": rows}, f)
    if tp:
        print(tp.key_averages().table(sort_by="cuda_time_total" if dev.type == "cuda" else "cpu_time_total", row_limit=40, max_name_column_width=60))
    if learner:
        learner.close()


if __name__ == "__main__":
    main()
