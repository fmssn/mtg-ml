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

    from mtg_ml.rl.model import PolicyNet
    from mtg_ml.rl.rollout import LEARNER, GameSpec, Job, Result, run_job, split_games, worker_init

    tmp = tempfile.mkdtemp(prefix="bench_update_")
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        paths.append(os.path.join(tmp, f"p{k}.pt"))
        torch.save({"config": config, "model": PolicyNet(**config).state_dict()}, paths[-1])
    specs = []
    for g in range(args.games):
        seats = (LEARNER, LEARNER) if g % 2 == 0 else ((LEARNER, paths[1]) if g % 4 == 1 else (paths[1], LEARNER))
        specs.append(GameSpec(seed=g, seats=seats, match_game=1 + g % 2))
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
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--trunk", default="entity", choices=("mlp", "transformer", "entity"))
    ap.add_argument("--value-net", default="shared", choices=("shared", "separate"))
    ap.add_argument("--memory", default="gru", choices=("gru", "none"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4, help="timed epochs")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--iter-decisions", type=int, default=378_000)
    ap.add_argument("--profile", action="store_true", help="cProfile the timed epochs")
    args = ap.parse_args(argv)

    import torch

    from mtg_ml.rl.model import PolicyNet
    from mtg_ml.rl.ppo import PPOConfig, make_optimizer, ppo_update

    config = PolicyNet(hidden=args.hidden, memory=args.memory, trunk=args.trunk, value_net=args.value_net).config
    if args.data:
        with open(args.data, "rb") as f:
            data = pickle.load(f)
    else:
        data = play(args, config)
    if args.save_data:
        with open(args.save_data, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    n, lens = len(data.actions), data.lengths
    print(f"{n} decisions in {len(lens)} trajectories (mean {n / len(lens):.0f}, max {max(lens)}), {len(data.samples.s_idx) / n:.0f} state tokens per decision")

    dev = torch.device(args.device)
    torch.manual_seed(0)
    net = PolicyNet(**config).to(dev)
    cfg = PPOConfig(minibatch=args.minibatch, epochs=args.warmup_epochs, target_kl=None)
    opt = make_optimizer(net.parameters(), cfg.lr, dev)
    gen = torch.Generator().manual_seed(0)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    ppo_update(net, opt, data, cfg, device=dev, gen=gen)
    sync()
    cfg.epochs = args.epochs
    pr = cProfile.Profile() if args.profile else None
    if pr:
        pr.enable()
    t = time.perf_counter()
    stats = ppo_update(net, opt, data, cfg, device=dev, gen=gen)
    sync()
    dt = time.perf_counter() - t
    if pr:
        pr.disable()
    steps = stats["updates"]
    mem = f", peak device memory {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" if dev.type == "cuda" else ""
    print(
        f"h{args.hidden} {args.trunk} value={args.value_net} memory={args.memory} {dev}: {steps} steps of ~{n * args.epochs / steps:.0f} decisions "
        f"in {dt:.2f}s = {dt / steps * 1000:.2f} ms/step; projected {dt * args.iter_decisions / n:.1f}s per iteration of "
        f"{args.iter_decisions} decisions x {args.epochs} epochs{mem}"
    )
    print({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()})
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
