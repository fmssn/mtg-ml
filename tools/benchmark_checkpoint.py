#!/usr/bin/env python3
"""Checkpoints against the benchmark specialists (and the legacy bots for reference).

.venv/bin/python tools/benchmark_checkpoint.py A.pt B.pt --blocks 100 --workers 32 \
    --contract diagnostic --out .context/ckpt-bench

Each cell plays four-slot blocks (both seats, both starts) on a shared seed
stream, so every checkpoint meets the same deals. The output is a development
report, not a released BenchmarkResult. Checkpoints with feature sets below 7
see hidden information and need --contract diagnostic.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import time

from mtg_ml.benchmark.adapters import CheckpointAdapter, LegacyAdapter, ScriptedAdapter
from mtg_ml.benchmark.artifacts import DIAGNOSTIC, FAIR
from mtg_ml.benchmark.blue import BOT_ID as BLUE_ID, PARAMETERS as BLUE_PARAMETERS, BenchmarkBlue
from mtg_ml.benchmark.bots.jund import BenchmarkJundBot
from mtg_ml.benchmark.bots.parameters import PARAMETERS_SHA256 as JUND_PARAMETERS_SHA256
from mtg_ml.benchmark.jsonio import digest, file_digest, write_json
from mtg_ml.benchmark.runner import run_episodes
from mtg_ml.benchmark.schedule import episodes
from mtg_ml.benchmark.views import thaw
from mtg_ml.bots import make_bot
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.cards import SPEC_PATH
from mtg_ml.rl.evaluate import wilson

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(max_turns=100, max_decisions=10000, auto_single=False, auto_mana=False, auto_pass=False)
STREAM = "benchmark-v1/dev"
JUND, BLUE = "jund_wildfire", "mono_blue_terror"
OPPONENTS = {"benchmark-jund@1": JUND, BLUE_ID: BLUE, "legacy-jund": JUND, "legacy-blue": BLUE}
# learner deck, opponent; the first is the standing ledger matchup.
CELLS = {"jund_vs_sblue": (JUND, BLUE_ID), "blue_vs_sjund": (BLUE, "benchmark-jund@1"),
         "jund_vs_sjund": (JUND, "benchmark-jund@1"), "blue_vs_sblue": (BLUE, BLUE_ID),
         "jund_vs_lblue": (JUND, "legacy-blue"), "blue_vs_ljund": (BLUE, "legacy-jund")}


@dataclass(frozen=True)
class Opponent:
    name: str

    def __call__(self, seat, mode):
        if self.name == BLUE_ID:
            return ScriptedAdapter(BenchmarkBlue(), seat)
        if self.name == "benchmark-jund@1":
            return ScriptedAdapter(BenchmarkJundBot(), seat)
        deck = OPPONENTS[self.name]
        return LegacyAdapter(lambda p: make_bot(p, deck), seat)


_ADAPTERS = {}


@dataclass(frozen=True)
class Learner:
    path: str
    contract: str

    def __call__(self, seat, mode):
        # One load per worker process and role; reset() rechecks the bytes and
        # clears recurrent state between games.
        key = self.path, self.contract, seat, mode
        if key not in _ADAPTERS:
            _ADAPTERS[key] = CheckpointAdapter(self.path, seat, mode, self.contract)
        return _ADAPTERS[key]


def schedule(blocks, modes, cells):
    decks = {JUND: JUND_WILDFIRE, BLUE: MONO_BLUE_TERROR}
    d = dict(stream=STREAM, modes=list(modes), blocks_per_cell=blocks,
             decks={n: dict(cards=dict(cards), sha256=digest(cards)) for n, cards in decks.items()},
             bots=[dict(id=k, deck=v) for k, v in OPPONENTS.items()],
             cells=[dict(id=c, learner_deck=CELLS[c][0], opponent=CELLS[c][1]) for c in cells])
    return d, episodes(d)


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        for cell in sorted({r["cell"] for r in rows if r["mode"] == mode}):
            rs = [r for r in rows if r["mode"] == mode and r["cell"] == cell]
            done = [r for r in rs if r["status"] == "completed"]
            wins = sum(.5 if r["winner"] is None else float(r["winner"] == r["learner_seat"]) for r in done)
            n = len(done)
            out.setdefault(mode, {})[cell] = dict(score=wins / n if n else None, ci95=list(wilson(wins, n)) if n else None,
                                                  games=n, errors=len(rs) - n,
                                                  reasons=dict(Counter(str(r["reason"]) for r in done)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--cells", default=",".join(CELLS), help="comma list of " + ",".join(CELLS))
    ap.add_argument("--modes", default="sampled,greedy")
    ap.add_argument("--blocks", type=int, default=100, help="four-game blocks per cell and mode")
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--contract", choices=("fair", "diagnostic"), default="fair")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.contract = {"fair": FAIR, "diagnostic": DIAGNOSTIC}[args.contract]
    cells, modes = args.cells.split(","), args.modes.split(",")
    if not set(cells) <= set(CELLS) or args.blocks < 1 or args.workers < 1:
        ap.error("unknown cell, or non-positive blocks/workers")
    args.out.mkdir(parents=True, exist_ok=False)
    data, specs = schedule(args.blocks, modes, cells)
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "mtg_ml", "native", "tools"], cwd=ROOT))
    freeze = {"code_revision": code, "dirty": dirty, "card_spec_sha256": file_digest(SPEC_PATH), "stream": STREAM,
              "settings": SETTINGS, "decks": data["decks"], "cells": data["cells"], "engine": args.engine,
              "specialists": {BLUE_ID: digest(thaw(BLUE_PARAMETERS)), "benchmark-jund@1": JUND_PARAMETERS_SHA256},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "workers": args.workers}}
    write_json(args.out / "freeze.json", freeze)
    report = {"format": "CheckpointSpecialistDevelopment", "version": 1, "freeze": freeze, "checkpoints": []}
    for path in args.checkpoints:
        probe = CheckpointAdapter(path, 0, "greedy", args.contract)
        learner = Learner(str(path.resolve()), args.contract)
        started, rows = time.monotonic(), []
        for cell in cells:
            batch = [s for s in specs if s.cell == cell]
            result = run_episodes(batch, SETTINGS, learner, Opponent(CELLS[cell][1]), engine=args.engine, workers=args.workers)
            rows.extend(result)
            s = summarize(result)
            print(f"{path.name} {cell}: " + "  ".join(f"{m} {v[cell]['score']:.3f}" for m, v in s.items())
                  + f"  errors={sum(r['status'] != 'completed' for r in result)}", flush=True)
        write_json(args.out / "rows" / f"{path.stem}.json", rows)
        report["checkpoints"].append({"path": str(path), "sha256": probe.sha256, "features": probe.features,
                                      "recorded_information_contract": probe.recorded_information_contract,
                                      "contract": args.contract, "elapsed_seconds": time.monotonic() - started,
                                      "results": summarize(rows)})
        write_json(args.out / "report.json", report)
    print(f"\n{'checkpoint':<28}{'mode':<9}" + "".join(f"{c:>16}" for c in cells))
    for ck in report["checkpoints"]:
        for mode, res in ck["results"].items():
            print(f"{Path(ck['path']).stem[:27]:<28}{mode:<9}" + "".join(
                f"{res[c]['score']:>9.3f} ±{(res[c]['ci95'][1] - res[c]['ci95'][0]) / 2:.3f}" for c in cells))
    print(f"report: {args.out / 'report.json'}")
    return 1 if any(v["errors"] for ck in report["checkpoints"] for m in ck["results"].values() for v in m.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
