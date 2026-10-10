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


def arm_flags(key: str, run: Path, root: Path, rungs: dict, offset: int, smoke_iters: int | None) -> dict:
    _, _, hidden, lr = ARMS[key]
    f = rc.flags("base", run, root, rungs, offset, "", smoke_iters)  # no arm override applies to "base"
    f.update({
        "hidden": hidden, "ppo-lr": lr, "ppo-lr-final": lr / 10, "seed": 10,
        "self-play-frac": 0.35, "pool-recent-frac": 0.25, "pool-sampling": "pfsp",
        "lr-anneal-games": TOTAL_GAMES, "total-games": TOTAL_GAMES,
    })
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
    layout = dict(kv.split("=") for kv in args.layout.split(","))  # h256=0A:18:0,h512=87:90:32
    entries = []
    for key, (name, handle, _, _) in ARMS.items():
        b0, b1, off = layout[key].split(":")
        offset = int(off)
        run = root / name
        run.mkdir()
        f = arm_flags(key, run, root, rungs, offset, args.smoke_iters if args.smoke else None)
        command = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"] + [p for k, v in f.items() for p in ("--" + k, str(v))]
        env = dict(CUDA_VISIBLE_DEVICES=f"{gpu[b0]},{gpu[b1]}", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MTG_ENGINE="native",
                   TORCHINDUCTOR_COMPILE_THREADS="1", MTG_EVAL_LOCK=str(run / "evaluation.lock"), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        entries.append(dict(
            name=name, arm=name, handle=handle, mode="training", run=str(run), code=str(code), code_ref=source, command=command,
            environment=env, gpus=env["CUDA_VISIBLE_DEVICES"].split(","),
            cpus=[offset, offset + 1, offset + 2, *range(offset + 4, offset + 32)],
            eval_cpus=list(range(offset + 4, offset + 8)), seed=f["seed"], features=7,
            target_games=int(f["total-games"]) or 51200, warmup=5, initial_tensor_sha256="n/a",
            parent=None, created_at=time.time(), smoke=args.smoke, buses=[b0, b1]))
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
        ap.add_argument("--layout", required=True, help="h256=learnerbus:serverbus:cpuoffset,h512=...")
        ap.add_argument("--ladder-src", required=True, help="directory with the L1 rungs and ladder.json")
        ap.add_argument("--smoke", action="store_true")
        ap.add_argument("--smoke-iters", type=int, default=3)
        prepare(ap.parse_args())
        return
    oc.ARMS = tuple(v[0] for v in ARMS.values())
    oc.archive = archive
    oc.tensor_hash = lambda path=None: "n/a"  # the r7 initial-tensor identity check does not apply here
    oc.main()


if __name__ == "__main__":
    main()
