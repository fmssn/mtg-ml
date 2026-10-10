"""Campaign r9-base (2026-10-10): two long-horizon base-model runs from scratch on one host.

Both start from the r7 lr075 recipe (the base flags of tools/r8_campaign.py `flags()`, before any arm override:
feature set 7, entity trunk + GRU, inference server, 24 workers, 2048 games/iter, bf16, capture 2, matchup = r7 matrix mix)
and change only the following, so the base recipe cannot drift from r8:

  hidden / lr      256 at 7.5e-5 -> 7.5e-6;  512 at 5.0e-5 -> 5.0e-6 (lr scaled by 1/sqrt(width) from the lr075 optimum)
  seed             10 for both
  anti-cycling     --self-play-frac 0.35 (was 0.5), --pool-recent-frac 0.25 (was 0.5), --pool-sampling pfsp
                   (the other 0.75 of pool games go to snapshots weighted by (1 - win rate)^2); --snapshot-every 122 unchanged
  length           --total-games 40000000, --lr-anneal-games 40000000 (linear)

`prepare` writes campaign.json (schema of the r7/r8 campaigns, so the tracker and `run/status/stop` of
overnight_campaign.py read it); `run` is the r7 supervisor: one detached trainer per arm, bounded logs, no restart.

  python tools/base_campaign.py prepare --root ~/mtg-ml-base/campaigns/20261010-r9-base --code ~/mtg-ml/code \\
      --ladder-src ~/mtg-ml-base/ladder-src --layout h256=0A:18:0,h512=87:90:32 [--smoke]
  R8_PREFIX=mtg-base CODE=~/mtg-ml/code R8_TOOL=<this file> tools/r8_host.sh start ROOT <run id>
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_campaign as oc  # noqa: E402
import r8_campaign as rc  # noqa: E402

DATE = "20261010"
# arm key -> (run ID = run directory = campaign entry name, handle, hidden, ppo-lr)
ARMS = {
    "h256": (f"{DATE}-r9-fs7h256-all-scratch-base-s10", "r9-base-h256", 256, 7.5e-5),
    "h512": (f"{DATE}-r9-fs7h512-all-scratch-base-s10", "r9-base-h512", 512, 5.0e-5),
}
TOTAL_GAMES = 40_000_000
R9_ARMS = ("h256", "h512")

# r10-base-h256-var (2026-10-11): the r9-base-h256 recipe on a whole 8-GPU box (h100-private4), from scratch with a new seed, on varied
# decklists (`--variants train`, the 92 registered 75s of the train split) with the fast recipe (ppo-epochs 2, BF16 GRU in the update).
# One run, 4 learner ranks (cuda:0-3, NUMA node 0) + 4 inference servers (cuda:4-7, node 1); everything that scales with the GPU count is x4:
# 8192 games per iteration and a GLOBAL PPO minibatch of 8192 (the distributed learner splits each global minibatch into whole trajectories
# per rank) with lr x2 (1.5e-4 -> 1.5e-5), the mb8192 arm of the throughput A/B (ledger 20261010-throughput-ab).
VAR_RUN = ("20261011-r10-fs7h256-all-scratch-var-s11", "r10-base-h256-var")
VAR_BUSES = ("0A", "18", "2F", "38", "87", "90", "BE", "C7")  # GPU order = PCI bus id order, cuda:0-3 learners, cuda:4-7 servers
VAR_CPUS = dict(  # SMT siblings of the driver CPUs (N + 64) stay idle; node 0 = 0-31,64-95, node 1 = 32-63,96-127
    learner=("0-1", "2", "3", "4"), server=("32", "33", "34", "35"), eval="36-39", workers="5-31,40-63,69-95,100-127")


def var_flags(f: dict) -> dict:
    """The r10-base-h256-var overrides on top of the r9 h256 flags (from `arm_flags`)."""
    f.update({
        "seed": 11, "variants": "train",
        "device": "cuda:0", "learner-devices": ",".join(f"cuda:{i}" for i in range(4)),
        "server-devices": ",".join(f"cuda:{i}" for i in range(4, 8)),
        "learner-cpus": ";".join(VAR_CPUS["learner"]), "trainer-cpus": VAR_CPUS["learner"][0], "server-cpus": ";".join(VAR_CPUS["server"]),
        "eval-cpus": VAR_CPUS["eval"], "worker-cpus": VAR_CPUS["workers"], "workers": 106, "eval-workers": 4,
        "games-per-iter": 8192, "ppo-minibatch": 8192, "ppo-epochs": 2, "ppo-gru-precision": "bf16",
        "ppo-lr": 1.5e-4, "ppo-lr-final": 1.5e-5, "checkpoint-every": 25, "snapshot-every": 30,  # 30 x 8192 = 246k games, as 122 x 2048 = 250k
    })
    del f["server-device"]
    return f


def arm_flags(key: str, run: Path, root: Path, rungs: dict, offset: int, smoke_iters: int | None) -> dict:
    _, _, hidden, lr = ARMS[key if key in ARMS else "h256"]
    f = rc.flags("base", run, root, rungs, offset, "", smoke_iters)  # no arm override applies to "base"
    f.update({
        "hidden": hidden, "ppo-lr": lr, "ppo-lr-final": lr / 10, "seed": 10,
        "self-play-frac": 0.35, "pool-recent-frac": 0.25, "pool-sampling": "pfsp",
        "lr-anneal-games": TOTAL_GAMES, "total-games": TOTAL_GAMES,
    })
    if key == "var256":
        var_flags(f)
    if smoke_iters is not None:
        f.update({"iterations": smoke_iters, "total-games": 0})
    return f


def prepare(args):
    root, code = Path(args.root).resolve(), Path(args.code).resolve()
    if (root / "campaign.json").exists():
        raise FileExistsError("campaign exists; use a new directory")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
        raise ValueError("deploy a clean committed source tree")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True)
    gpu = {r.split(",")[0].strip().split(":")[-2]: r.split(",")[1].strip() for r in rows.splitlines()}
    root.mkdir(parents=True, exist_ok=True)
    rungs = rc.ladder(root, Path(args.ladder_src))
    layout = dict(kv.split("=") for kv in args.layout.split(",") if kv)  # h256=0A:18:0,h512=87:90:32
    entries = []
    for key in args.arms.split(","):
        var = key == "var256"
        name, handle = VAR_RUN if var else ARMS[key][:2]
        b0, b1, off = ("0A", "C7", "0") if var else layout[key].split(":")
        offset = int(off)
        run = root / name
        run.mkdir()
        f = arm_flags(key, run, root, rungs, offset, args.smoke_iters if args.smoke else None)
        command = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"] + [p for k, v in f.items() for p in ("--" + k, str(v))]
        buses = list(VAR_BUSES) if var else [b0, b1]
        env = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(gpu[b] for b in buses), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MTG_ENGINE="native",
                   TORCHINDUCTOR_COMPILE_THREADS="1", MTG_EVAL_LOCK=str(run / "evaluation.lock"), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        entries.append(dict(
            name=name, arm=name, handle=handle, mode="training", run=str(run), code=str(code), code_ref=source, command=command,
            environment=env, gpus=env["CUDA_VISIBLE_DEVICES"].split(","),
            cpus=[offset, offset + 1, offset + 2, *range(offset + 4, offset + 32)] if not var else list(range(128)),
            eval_cpus=list(range(offset + 4, offset + 8)) if not var else [36, 37, 38, 39], seed=f["seed"], features=7,
            target_games=int(f["total-games"]) or 51200, warmup=5, initial_tensor_sha256="n/a",
            parent=None, created_at=time.time(), smoke=args.smoke, buses=buses))
    oc.atomic(root / "campaign.json", entries)
    print(root / "campaign.json")


def archive(entry):
    dest = Path.home() / "mtg-ml-base" / "archive" / entry["name"]
    if (dest / "COMPLETE").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    run = Path(entry["run"])
    for name in ("latest.pt", "metrics.jsonl", "process.json", "train.log"):
        target = dest / ("final.pt" if name == "latest.pt" else name)
        if not target.exists():
            import shutil
            shutil.copy2(run / name, target)
    import shutil
    for folder in ("pool", "policy"):
        shutil.copytree(run / folder, dest / folder, dirs_exist_ok=True)
    oc.atomic(dest / "launch.json", entry)
    (dest / "SHA256").write_text(hashlib.file_digest((dest / "final.pt").open("rb"), "sha256").hexdigest() + "  final.pt\n")
    (dest / "COMPLETE").write_text(datetime.now().isoformat() + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        ap = argparse.ArgumentParser()
        ap.add_argument("action")
        ap.add_argument("--root", required=True)
        ap.add_argument("--code", required=True)
        ap.add_argument("--arms", default=",".join(R9_ARMS), help="h256,h512 (r9-base) or var256 (r10-base-h256-var: all 8 GPUs, fixed layout)")
        ap.add_argument("--layout", default="", help="h256=learnerbus:serverbus:cpuoffset,h512=...; not used by var256")
        ap.add_argument("--ladder-src", required=True, help="directory with the L1 rungs and ladder.json")
        ap.add_argument("--smoke", action="store_true")
        ap.add_argument("--smoke-iters", type=int, default=3)
        prepare(ap.parse_args())
        return
    oc.ARMS = tuple(v[0] for v in ARMS.values()) + (VAR_RUN[0],)
    oc.archive = archive
    oc.tensor_hash = lambda path=None: "n/a"  # the r7 initial-tensor identity check does not apply here
    oc.main()


if __name__ == "__main__":
    main()
