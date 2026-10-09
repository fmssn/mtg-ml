"""Prepare/replay isolated width-256 attention experiments on released resources.

Preparation writes exact commands without launching training. Running refuses
occupied GPUs/CPUs and never stops another job. Outcome JSONL includes failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtg_ml.rl.collect import parse_cpus

MATCHUPS = "jund_blue:2,affinity_elves,affinity_tron,blue_affinity,blue_elves,blue_madness,blue_tron,elves_tron,jund_affinity,jund_elves,jund_madness,jund_tron,madness_affinity,madness_elves,madness_tron"
PROJECT_BUSES = {"0A", "18", "87", "90", "C7"}
ARMS = {
    "legacy": dict(legacy=True, same_gpu=True, slots=0),
    "stacked": dict(same_gpu=True, slots=0),
    "separate": dict(slots=0),
    "resident64": dict(slots=64),
    "resident128": dict(slots=128),
    "bf16": dict(precision="bf16"),
    "learners2": dict(precision="bf16", learners=2),
    "learners4": dict(precision="bf16", learners=4),
    "batch8192-lr150": dict(precision="bf16", minibatch=8192, lr="1.5e-4"),
    "batch8192-lr300": dict(precision="bf16", minibatch=8192, lr="3e-4"),
    "epochs2": dict(precision="bf16", epochs=2),
    "lag2": dict(precision="bf16", pipeline=2),
}


def prepare(args):
    root, frozen, code = Path(args.root).resolve(), Path(args.frozen).resolve(), Path(args.code).resolve()
    arms = args.arms.split(",")
    cpus, gpus = parse_cpus(args.cpus), args.gpus.split(",")
    if args.hours not in (0, 2, 6):
        raise ValueError("use 0 hours for screens, 2 for finalists or 6 for extended finalists")
    if len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError(f"choose distinct arms from {', '.join(ARMS)}")
    if len(set(gpus)) != len(gpus) or any(not gpu.startswith("GPU-") for gpu in gpus):
        raise ValueError("assign distinct GPU UUIDs")
    if len(cpus) < 8:
        raise ValueError("reserve at least 8 CPUs: four learners, inference, workers, evaluation")
    for arm in arms:
        settings = ARMS[arm]
        learners = settings.get("learners", args.base_learners if arm in ("bf16", "batch8192-lr150", "batch8192-lr300", "epochs2", "lag2") else 1)
        required = learners + int(not settings.get("same_gpu", False))
        if len(gpus) < required:
            raise ValueError(f"{arm} needs {required} GPU UUIDs")
        for seed in (0, 1, 2) if args.hours else (0,):
            if (root / f"{arm}-seed{seed}").exists():
                raise FileExistsError(root / f"{arm}-seed{seed}")
    if (root / "campaign.json").exists():
        raise FileExistsError(root / "campaign.json")
    manifest = json.loads((frozen / "manifest.json").read_text())
    if not manifest.get("opponents"):
        raise ValueError("freeze the historical opponent pool with --opponent-pool first")
    continuation = manifest.get("continuation")
    if not continuation:
        raise ValueError("freeze parent schedule metadata with --training-checkpoint first")
    training = continuation["train_config"]
    ppo = training["ppo"]
    anneal = training["lr_anneal_games"]
    position = continuation["lr_position"]
    fraction = min(1, position / anneal) if anneal else 0
    if training["lr_schedule"] == "cosine":
        fraction = 0.5 * (1 - math.cos(math.pi * fraction))
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for arm in arms:
        settings = ARMS[arm]
        lr = ppo["lr"]
        if "lr" in settings:
            if fraction >= 1:
                raise ValueError("larger-batch LR tests require an unfinished parent schedule")
            # Set the requested starting LR at the parent's schedule position.
            lr = (float(settings["lr"]) - fraction * ppo["lr_final"]) / (1 - fraction)
        learners = settings.get("learners", args.base_learners if arm in ("bf16", "batch8192-lr150", "batch8192-lr300", "epochs2", "lag2") else 1)
        same = settings.get("same_gpu", False)
        gpu_count = learners if same else learners + 1
        if len(gpus) < gpu_count:
            raise ValueError(f"{arm} needs {gpu_count} GPU UUIDs")
        for seed in (0, 1, 2) if args.hours else (0,):
            name = f"{arm}-seed{seed}"
            run = root / name
            if run.exists():
                raise FileExistsError(run)
            run.mkdir(parents=True)
            devs = ",".join(f"cuda:{k}" for k in range(learners))
            server = "cuda:0" if same else f"cuda:{learners}"
            flags = {
                "run": str(run), "init": str(frozen / "attention.pt"), "hidden": 256, "trunk": "entity", "value-net": "shared",
                "opponent-pool": str(frozen / "pool"),
                "entity-attn": 1, "features": 6, "engine": "native", "device": "cuda:0", "learner-devices": devs,
                "learner-cpus": ";".join(str(c) for c in cpus[:learners]), "inference": "server", "server-device": server,
                "server-cpus": str(cpus[4]), "trainer-cpus": str(cpus[0]), "worker-cpus": ",".join(map(str, cpus[5:-1])),
                "eval-cpus": str(cpus[-1]), "workers": len(cpus[5:-1]), "eval-workers": 1,
                "server-policy-slots": 64, "server-resident-limit": settings.get("slots", 64),
                "server-stacked-attention": int(not settings.get("legacy")), "games-per-iter": 2048,
                "matchup": MATCHUPS, "postboard-frac": 0.2, "seed": seed, "pipeline": settings.get("pipeline", 1),
                "ppo-precision": settings.get("precision", "fp32"), "ppo-minibatch": settings.get("minibatch", 2048),
                "ppo-epochs": settings.get("epochs", 4), "ppo-lr": lr, "ppo-lr-final": ppo["lr_final"],
                "lr-anneal-games": anneal, "lr-anneal-offset-games": position, "lr-schedule": training["lr_schedule"],
                "shaping": training["shaping"], "shaping-anneal-iters": training["shaping_anneal_iters"],
                "shaping-offset-iters": continuation["shaping_position"], "snapshot-every": 122, "checkpoint-every": 25,
                "iterations": 100000000 if args.hours else 25, "duration-seconds": args.hours * 3600,
                "eval-every": 0, "eval-every-games": 250000 if args.hours else 0,
                "eval-final": int(bool(args.hours)),
                "bench-games": 1000, "bench-greedy-games": 1000, "bench-bo3-matches": 0, "eval-bo3-matches": 0,
                "ladder-games": 200, "ladder": ",".join(str(Path(args.ladder) / f"iter_{k:05d}.pt") for k in (488, 1464, 2440, 3904)),
                "ladder-ratings": str(Path(args.ladder) / "ladder.json"),
            }
            for key in ("gamma", "lam", "gamma_turn", "lam_turn", "value_clamp", "self_play_frac", "bot_frac", "bot_seat", "pool_recent_frac", "pool_sampling", "pfsp_power", "pfsp_ema", "max_turns", "auto_mana", "auto_pass"):
                flags[key.replace("_", "-")] = training[key]
            for key in ("clip", "vf_coef", "ent_coef", "max_grad_norm", "target_kl", "capture"):
                flags["ppo-" + key.replace("_", "-")] = ppo[key] if ppo[key] is not None else "none"
            command = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"]
            command += [word for k, v in flags.items() for word in (f"--{k}", str(v))]
            entries.append(dict(name=name, arm=arm, seed=seed, run=str(run), command=command, settings=settings,
                                code=str(code), code_ref=args.code_ref, parent=manifest, gpus=gpus[:gpu_count], cpus=list(cpus),
                                hours=args.hours, warmup=5, timed=20))
    path = root / "campaign.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(path)


def blockers(entry):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used", "--format=csv,noheader,nounits"], text=True)
    inventory = {r[0].strip(): (r[1].strip().split(":")[-2].upper(), int(r[2])) for r in csv.reader(raw.splitlines())}
    out = []
    for uuid in entry["gpus"]:
        if uuid not in inventory:
            out.append(f"GPU {uuid} absent")
        else:
            bus, mem = inventory[uuid]
            if bus not in PROJECT_BUSES:
                out.append(f"GPU {uuid} bus {bus} is excluded")
            if mem > 1024:
                out.append(f"GPU {uuid} occupied ({mem} MiB)")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    for uuid, pid in csv.reader(apps.splitlines()):
        if uuid.strip() in entry["gpus"]:
            out.append(f"GPU {uuid.strip()} owned by PID {pid.strip()}")
    processes, roots = {}, set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
            # stat's command field can contain spaces and parentheses.
            parent = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            processes[int(proc.name)] = parent
            if b"-m mtg_ml.rl.train" in cmd:
                roots.add(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    descendants = set(roots)
    while extra := {p for p, parent in processes.items() if parent in descendants} - descendants:
        descendants |= extra
    for pid in descendants:
        try:
            overlap = set(os.sched_getaffinity(pid)) & set(entry["cpus"])
            if overlap:
                out.append(f"training process {pid} owns CPUs {sorted(overlap)}")
        except ProcessLookupError:
            continue
    return out


def summarize(entry, charged_s):
    rows = [json.loads(s) for s in (Path(entry["run"]) / "metrics.jsonl").read_text().splitlines()]
    timed = rows[entry["warmup"]:]
    if not entry["hours"] and len(timed) != entry["timed"]:
        raise RuntimeError(f"expected 20 timed iterations, got {len(timed)}")
    wall = sum(r["wall_s"] for r in timed)
    report = dict(name=entry["name"], verdict="inconclusive", timed_iterations=len(timed),
                  decisions_per_s=sum(r["decisions"] for r in timed) / wall,
                  games_per_hour=sum(r["games"] for r in timed) * 3600 / wall,
                  charged_gpu_hours=charged_s * len(entry["gpus"]) / 3600,
                  timed_gpu_hours=wall * len(entry["gpus"]) / 3600,
                  learner_peak_bytes=max(r["learner_peak_bytes"] for r in rows),
                  learner_peak_bytes_by_rank=rows[-1].get("learner_peak_bytes_by_rank", [max(r["learner_peak_bytes"] for r in rows)]),
                  median_ppo_s=statistics.median(r["ppo_s"] for r in timed),
                  median_rollout_s=statistics.median(r["rollout_s"] for r in timed),
                  median_publication_s=statistics.median(r["publish_s"] for r in timed),
                  mean_residency_selection_s=statistics.mean(r.get("residency", {}).get("selection_s", 0) for r in timed),
                  mean_residency_merge_s=statistics.mean(r.get("residency", {}).get("merge_s", 0) for r in timed),
                  mean_residency_waves=statistics.mean(r.get("residency", {}).get("waves", 1) for r in timed))
    initial, final = rows[entry["warmup"] - 1].get("inference_stats", {}), rows[-1].get("inference_stats", {})
    report["inference_peak_bytes"] = [s["peak_bytes"] for s in final.get("servers", [])]
    if final:
        report["padding_ratio"] = (final["padded_rows"] - initial["padded_rows"]) / max(final["rows"] - initial["rows"], 1)
    # Each evaluation retains both game count and elapsed time for matched comparisons.
    report["evaluations"] = [{k: v for k, v in r.items() if k.startswith(("eval/", "bench/", "ladder/")) or k in ("iteration", "games_total", "elapsed_s")}
                             for r in rows if "ladder/elo" in r]
    return report


def run(args):
    manifest = Path(args.manifest).resolve()
    entries = json.loads(manifest.read_text())
    outcomes = manifest.parent / "outcomes.jsonl"
    done = {json.loads(line)["name"] for line in outcomes.read_text().splitlines()} if outcomes.exists() else set()
    for entry in entries:
        if entry["name"] in done or (args.arms and entry["arm"] not in args.arms.split(",")):
            continue
        if not entry["hours"] and entry["arm"] in ("learners2", "learners4"):
            reports = {r["name"]: r for r in (json.loads(s) for s in outcomes.read_text().splitlines())} if outcomes.exists() else {}
            previous = reports.get("bf16-seed0" if entry["arm"] == "learners2" else "learners2-seed0")
            baseline = reports.get("bf16-seed0")
            reason = None
            if previous is None or "median_ppo_s" not in previous:
                reason = "run the preceding learner screen successfully first"
            elif previous["median_ppo_s"] <= previous["median_rollout_s"]:
                reason = "learning is no longer the limiting stage"
            elif entry["arm"] == "learners4" and (baseline is None or previous["games_per_hour"] <= 1.05 * baseline["games_per_hour"]):
                reason = "two learner GPUs did not improve total throughput by at least 5%"
            if reason:
                report = dict(name=entry["name"], verdict="skipped", reason=reason, code_ref=entry["code_ref"])
                if not args.check_only:
                    with open(outcomes, "a") as f:
                        f.write(json.dumps(report) + "\n")
                print(json.dumps(report), flush=True)
                continue
        deadline = time.monotonic() + args.wait_seconds
        while reasons := blockers(entry):
            print(json.dumps(dict(name=entry["name"], blocked=reasons)), flush=True)
            if args.check_only or time.monotonic() >= deadline:
                return
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        if args.check_only:
            print(json.dumps(dict(name=entry["name"], ready=True)))
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(entry["gpus"]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "TORCHINDUCTOR_COMPILE_THREADS": "1", "MTG_ENGINE": "native"}
        # Bound each log by iterations/time. A failed arm is recorded once, never respawned.
        t = time.monotonic()
        with open(Path(entry["run"]) / "train.log", "w") as log:
            proc = subprocess.Popen(["taskset", "-c", ",".join(map(str, entry["cpus"])), *entry["command"]], cwd=entry["code"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=entry["hours"] * 3600 + 7200 if entry["hours"] else 7200)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    import signal

                    os.killpg(proc.pid, signal.SIGKILL)  # only this arm's process group
                    proc.wait()
                rc = -1
        charged = time.monotonic() - t
        if rc:
            stop_group(proc.pid)
        try:
            report = summarize(entry, charged) if rc == 0 else dict(name=entry["name"], verdict="failed", exit_code=rc, charged_gpu_hours=charged * len(entry["gpus"]) / 3600)
        except Exception as e:
            report = dict(name=entry["name"], verdict="failed", error=str(e))
        report.update(code_ref=entry["code_ref"], parent_sha256=entry["parent"]["parent_sha256"], command=entry["command"], gpus=entry["gpus"], cpus=entry["cpus"])
        with open(outcomes, "a") as f:
            f.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        if rc:
            return  # fix a failure before spending more GPUs


def stop_group(pid):
    """Escalate only within the arm's newly created process group."""
    import signal

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    for name in ("root", "frozen", "code", "code-ref", "gpus", "cpus", "ladder"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--hours", type=float, default=0, help="0: 5 warm-up + 20 timed iterations; 2 or 6: three paired learning seeds")
    p.add_argument("--base-learners", type=int, choices=(1, 2, 4), default=1, help="hardware chosen by the screens, for learning finalists")
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--arms")
    r.add_argument("--wait-seconds", type=float, default=0)
    r.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    (prepare if args.operation == "prepare" else run)(args)


if __name__ == "__main__":
    main()
