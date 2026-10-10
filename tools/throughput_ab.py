"""Learning-equivalence arms for training-throughput changes (docs/experiments/throughput.md).

Every arm starts from the same `--init` (the r7 lr075 policy), plays the same matrix mix with the same
seed and game budget, and differs from the control only in the flags named by its entry in ARMS. The
rest is the r8 production recipe (`tools/r8_campaign.flags`, arm C, with the matrix mix and a
`--lr-anneal-games` equal to the budget).

    python tools/throughput_ab.py write --root ~/mtg-ml/opt/ab --code ~/mtg-ml/code --parent ~/mtg-ml/parents/r7-lr075-policy.pt \
        --arms base,mb4096 --layout base=0:1:0,mb4096=4:5:32
    bash ~/mtg-ml/opt/ab/<arm>/launch.sh        # detached tmux session ab-<arm>

--layout is ARM=learner GPU index:server GPU index:first CPU (24 workers + trainer/server/eval on 32 CPUs from there).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_campaign as oc  # noqa: E402
import r8_campaign as r8  # noqa: E402

BASE_LR = 7.5e-5
ARMS = {  # arm -> overrides of the production flags
    "base": {},
    "mb4096": {"ppo-minibatch": 4096, "ppo-lr": BASE_LR * 2**0.5},
    "mb8192": {"ppo-minibatch": 8192, "ppo-lr": BASE_LR * 2},
    "mb4096-lr1": {"ppo-minibatch": 4096},
    "ep2": {"ppo-epochs": 2},
    "ep2-mb4096": {"ppo-epochs": 2, "ppo-minibatch": 4096},
    "gru-bf16": {"ppo-gru-precision": "bf16"},
    "mb4096-gru": {"ppo-minibatch": 4096, "ppo-lr": BASE_LR * 2**0.5, "ppo-gru-precision": "bf16"},
}


def arm_flags(arm: str, run: Path, parent: str, offset: int, games: int, seed: int) -> dict:
    f = r8.flags("C", run, run.parent, {}, offset, parent, None)
    f.update({"matchup": oc.matrix_mix(), "lr-anneal-games": games, "total-games": games, "seed": seed})
    for k in ("ladder", "ladder-ratings", "ladder-games"):  # the L1 rungs live on h100-private
        f.pop(k, None)
    f.update(ARMS[arm])
    f["ppo-lr-final"] = f["ppo-lr"] / 10
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--games", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    root, code = Path(a.root).expanduser().resolve(), Path(a.code).expanduser().resolve()
    layout = {k: v.split(":") for k, v in (kv.split("=") for kv in a.layout.split(","))}
    for arm in a.arms.split(","):
        run = root / arm
        run.mkdir(parents=True, exist_ok=True)
        g0, g1, off = layout[arm]
        f = arm_flags(arm, run, str(Path(a.parent).expanduser()), int(off), a.games, a.seed)
        f["device"], f["server-device"] = "cuda:0", "cuda:1"
        cmd = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"] + [p for k, v in f.items() for p in ("--" + k, str(v))]
        env = f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={g0},{g1} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MTG_EVAL_LOCK={run}/evaluation.lock MTG_ENGINE=native"
        power = f"nvidia-smi -i {g0},{g1} --query-gpu=timestamp,index,power.draw,utilization.gpu --format=csv,noheader -l 5 > {run}/power.csv &"
        (run / "launch.sh").write_text(f"#!/bin/bash\ncd {code}\n{power}\nenv {env} {shlex.join(cmd)} >> {run}/train.log 2>&1\nkill %1\n")
        (run / "launch.sh").chmod(0o755)
        (run / "flags.txt").write_text(shlex.join(cmd[3:]) + "\n")
        print(f"tmux new -d -s ab-{arm} bash {run}/launch.sh")


if __name__ == "__main__":
    main()
