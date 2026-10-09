"""Campaign r8 (2026-10-09): three single-host arms on top of tools/overnight_campaign.py.

`prepare` writes a campaign.json in the same schema as the r7 campaign (so the
dashboard and `overnight_campaign.py run/status/stop` read it) for the arms
that live on this host. Everything else (`run`, `status`, `stop`) is the r7
supervisor: one detached trainer per arm, bounded logs, no automatic restart.

Arms (all feature set 7/8, h256, seed 8, r7 lr075 settings unless stated):
  A  r8-lr075-continue  resume the r7 lr075 run dir; lr anneal continues
  B  r8-fs8-belief      fresh, feature set 8 + belief head (needs match rollouts)
  C  r8-jund-blue-ft    --init from the r7 lr075 final policy, jund_blue only
  D  r8-jund-blue-ft-s2 replication of C with seed 9
  E  r8-jund-mirror-ft  like C on --matchup jund_mirror

Run from the deployed checkout's venv:
  python tools/r8_campaign.py prepare --root ROOT --code CODE --arms A --ladder-src DIR ...
  python tools/r8_campaign.py run --root ROOT --arm r8-lr075-continue --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_campaign as oc  # noqa: E402

PARENT_RUN = "20261008-r7-fs7-h256-lr075"
SKIP_COPY = ("train.log", "train.log.1", "train.log.2", "train.log.3", "process.json", "supervisor.lock", "evaluation.lock")
# arm -> (host layout slot: GPU buses (learner, server), CPU offset)
ARMS = {"A": "r8-lr075-continue", "B": "r8-fs8-belief", "C": "r8-jund-blue-ft", "D": "r8-jund-blue-ft-s2", "E": "r8-jund-mirror-ft"}
RUNG_RATINGS = {488: 0.0, 1464: 33.3, 2440: 50.8, 3904: 78.8}


def ladder(root: Path, src: Path) -> dict[str, float]:
    """Copy the fixed L1 rungs next to the campaign and write ladder.json against the copies."""
    original = json.loads((src / "ladder.json").read_text())
    dst = root / "ladder"
    dst.mkdir(exist_ok=True)
    mapped = {}
    for it, rating in RUNG_RATINGS.items():
        name = f"iter_{it:05d}.pt"
        if [v for k, v in original["ratings"].items() if Path(k).name == name] != [rating]:
            raise ValueError(f"ladder rating changed for {name}")
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)
        mapped[str(dst / name)] = rating
    oc.atomic(root / "ladder.json", {**original, "ratings": mapped})
    return mapped


def flags(arm: str, run: Path, root: Path, rungs: dict, offset: int, parent: str, smoke_iters: int | None) -> dict:
    smoke = smoke_iters is not None
    f = {
        "run": str(run), "hidden": 256, "trunk": "entity", "entity-attn": 1, "value-net": "shared",
        "value-bound": "none", "memory": "gru", "features": 7, "seed": 8, "engine": "native",
        "device": "cuda:0", "inference": "server", "server-device": "cuda:1",
        "server-cpus": str(offset + 2), "trainer-cpus": f"{offset}-{offset + 1}",
        "worker-cpus": f"{offset + 8}-{offset + 31}", "eval-cpus": f"{offset + 4}-{offset + 7}",
        "workers": 24, "eval-workers": 4, "games-per-iter": 2048, "server-policy-slots": 64,
        "server-resident-limit": 64, "server-stacked-attention": 1, "ppo-precision": "bf16",
        "ppo-capture": 2, "ppo-minibatch": 2048, "ppo-epochs": 4, "ppo-target-kl": 0.03,
        "ppo-lr": 7.5e-5, "ppo-lr-final": 7.5e-6, "lr-anneal-games": 20000000, "lr-schedule": "linear",
        "pipeline": 1, "self-play-frac": 0.5, "pool-recent-frac": 0.5, "pool-sampling": "uniform",
        "bot-frac": 0, "postboard-frac": 0.2, "matchup": oc.matrix_mix(), "checkpoint-every": 25,
        "snapshot-every": 122, "iterations": 100000000, "total-games": 20000000, "eval-every": 0,
        "eval-every-games": 250000, "eval-extra-matchups": 0, "eval-final": 1, "eval-games": 0,
        "eval-blocks": "", "bench-games": 1000, "bench-greedy-games": 1000, "bench-bo3-matches": 0,
        "eval-bo3-matches": 0, "ladder-games": 200, "ladder": ",".join(rungs),
        "ladder-ratings": str(root / "ladder.json"),
    }
    if arm == "B":  # the only differences from r7 lr075: set 8 + belief head, which requires whole-match rollouts with varied lists
        f.update({"features": 8, "belief": 1, "match-rollouts": 1, "variants": "train"})
    if arm in ("C", "D", "E"):
        for k in ("features", "entity-attn", "hidden", "trunk", "value-net", "memory"):
            del f[k]  # architecture comes from --init
        f.update({"init": parent, "matchup": "jund_blue", "lr-anneal-games": 6000000, "total-games": 6000000})
        if arm == "D":
            f["seed"] = 9
        if arm == "E":
            f["matchup"] = "jund_mirror"
            f["ppo-capture"] = 0  # mirror games are long: the CUDA-graph captures filled the 80 GB learner GPU (OOM at iteration 41 with capture 2)
    if smoke:
        f.update({"iterations": smoke_iters, "total-games": 0, "eval-every-games": 0, "bench-games": 8,
                  "bench-greedy-games": 8, "ladder-games": 8, "checkpoint-every": 2})
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
    rungs = ladder(root, Path(args.ladder_src))
    layout = dict(kv.split("=") for kv in args.layout.split(","))  # e.g. A=0A:18:0,B=87:90:32
    entries = []
    for arm in args.arms.split(","):
        name = ARMS[arm]
        run = root / name
        b0, b1, off = layout[arm].split(":")
        offset = int(off)
        smoke_iters = None
        if arm == "A":
            if run.exists():
                raise FileExistsError(run)
            shutil.copytree(args.parent_run, run, ignore=shutil.ignore_patterns(*SKIP_COPY))
            if args.smoke:
                import torch
                smoke_iters = int(torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["iteration"]) + 4
        else:
            run.mkdir()
            smoke_iters = 6 if args.smoke else None
        parent = args.parent_policy if arm in ("C", "D", "E") else None
        f = flags(arm, run, root, rungs, offset, parent, smoke_iters)
        command = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"] + [p for k, v in f.items() for p in ("--" + k, str(v))]
        env = dict(CUDA_VISIBLE_DEVICES=f"{gpu[b0]},{gpu[b1]}", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TORCHINDUCTOR_COMPILE_THREADS="1", MTG_EVAL_LOCK=str(run / "evaluation.lock"), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        entries.append(dict(
            name=name, arm=name, mode="training", run=str(run), code=str(code), code_ref=source, command=command,
            environment=env, gpus=env["CUDA_VISIBLE_DEVICES"].split(","),
            cpus=[offset, offset + 1, offset + 2, *range(offset + 4, offset + 32)],
            eval_cpus=list(range(offset + 4, offset + 8)), seed=f["seed"], features=f.get("features", 7),
            target_games=int(f["total-games"]) or 51200, warmup=5, initial_tensor_sha256="n/a",
            parent=(PARENT_RUN if arm in ("A", "C", "D", "E") else None), created_at=time.time(), smoke=args.smoke,
            buses=[b0, b1]))
    oc.atomic(root / "campaign.json", entries)
    print(root / "campaign.json")


def archive(entry):
    dest = Path.home() / "mtg-ml-r8" / "archive" / entry["name"]
    if (dest / "COMPLETE").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    run = Path(entry["run"])
    for name in ("latest.pt", "metrics.jsonl", "process.json", "train.log"):
        target = dest / ("final.pt" if name == "latest.pt" else name)
        if not target.exists():
            shutil.copy2(run / name, target)
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
        ap.add_argument("--arms", required=True, help="comma list of A,B,C")
        ap.add_argument("--layout", required=True, help="ARM=learnerbus:serverbus:cpuoffset, comma separated")
        ap.add_argument("--ladder-src", required=True)
        ap.add_argument("--parent-run", help="A: the r7 lr075 run dir to copy and resume")
        ap.add_argument("--parent-policy", help="C: policy.pt of the r7 lr075 final")
        ap.add_argument("--smoke", action="store_true")
        prepare(ap.parse_args())
        return
    oc.ARMS = tuple(ARMS.values())
    oc.archive = archive
    oc.tensor_hash = lambda path=None: "n/a"  # the r7 initial-tensor identity check does not apply here
    oc.main()


if __name__ == "__main__":
    main()
