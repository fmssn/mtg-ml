#!/usr/bin/env python3
"""Development-only Jund smoke panel and frozen paired legacy comparison.

Example: .venv/bin/python tools/benchmark_jund.py --smoke --out .context/jund/smoke.json
The output is JundDevelopmentComparison, not a released BenchmarkResult.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from mtg_ml.benchmark.adapters import ScriptedAdapter, LegacyAdapter
from mtg_ml.benchmark.bots.jund import BenchmarkJundBot, PARAMETERS_SHA256
from mtg_ml.benchmark.jsonio import digest, file_digest, write_json
from mtg_ml.benchmark.runner import run_episodes
from mtg_ml.benchmark.schedule import episodes, bootstrap_seed
from mtg_ml.benchmark.validation import native_build
from mtg_ml.bots import make_bot
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.cards import SPEC_PATH

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "ac7e95ad583122e8031f051d0ffcac515540d9b2"
SETTINGS = dict(max_turns=100, max_decisions=10000, auto_single=False, auto_mana=False, auto_pass=False)
STREAM = "benchmark-v1/dev"


@dataclass(frozen=True)
class BotFactory:
    deck: str
    specialist: bool = False

    def __call__(self, seat, mode):
        if self.specialist:
            return ScriptedAdapter(BenchmarkJundBot(), seat)
        from functools import partial
        return LegacyAdapter(partial(make_bot, deck=self.deck), seat)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def source_freeze(baseline):
    revision = git("rev-parse", "HEAD").decode().strip()
    # A baseline commit is accepted only when the actual legacy implementations
    # loaded here match that revision. Both candidates use this engine build.
    baseline = git("rev-parse", "--verify", baseline + "^{commit}").decode().strip()
    if git("diff", "--name-only", baseline, "--", "mtg_ml/bots", "mtg_ml/engine/decks.py").strip():
        raise ValueError("legacy bot/deck sources differ from the requested baseline")
    paths = ("mtg_ml", "native", "tools/benchmark_jund.py")
    if git("diff", "--name-only", "HEAD", "--", *paths).strip() or git("ls-files", "--others", "--exclude-standard", "--", *paths).strip():
        raise ValueError("commit runtime sources before freezing a comparison")
    legacy = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / "mtg_ml/bots").glob("*.py"))}
    return dict(candidate_revision=revision, engine_revision=revision, legacy_revision=baseline,
                legacy_sources_sha256=digest(legacy), legacy_files=legacy, parameters_sha256=PARAMETERS_SHA256,
                card_spec_sha256=file_digest(SPEC_PATH), tool_sha256=file_digest(__file__))


def schedule(blocks, mode="greedy"):
    decks = {"jund_wildfire": JUND_WILDFIRE, "mono_blue_terror": MONO_BLUE_TERROR}
    d = dict(stream=STREAM, modes=[mode], blocks_per_cell=blocks,
             decks={n: dict(cards=dict(cards), sha256=digest(cards)) for n, cards in decks.items()},
             bots=[dict(id="legacy-jund", deck="jund_wildfire"), dict(id="legacy-blue", deck="mono_blue_terror")],
             cells=[dict(id="jund_vs_jund", learner_deck="jund_wildfire", opponent="legacy-jund"),
                    dict(id="jund_vs_blue", learner_deck="jund_wildfire", opponent="legacy-blue")])
    return d, episodes(d)


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    at = (len(xs) - 1) * q
    lo = math.floor(at)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (at - lo)


def game_score(row):
    return .5 if row["winner"] is None else float(row["winner"] == row["learner_seat"])


def paired_summary(rows, blocks, mode="greedy", replicates=10000):
    """Each bootstrap unit preserves a seed's four seat/start slots."""
    expected = {(pilot, cell, b, slot) for pilot in ("specialist", "legacy") for cell in ("jund_vs_jund", "jund_vs_blue")
                for b in range(blocks) for slot in range(4)}
    keyed = {(r["pilot"], r["cell"], r["block"], r["slot"]): r for r in rows}
    _, specs = schedule(blocks, mode)
    scheduled = {(s.cell, s.block, s.slot): s for s in specs}
    fields = ("mode", "simulator_seed", "actor_seed", "learner_seat", "starting_player")
    valid = all((r["cell"], r["block"], r["slot"]) in scheduled and all(
        r.get(k) == getattr(scheduled[r["cell"], r["block"], r["slot"]], k) for k in fields) for r in rows)
    if not valid or len(keyed) != len(rows) or set(keyed) != expected or any(r["status"] != "completed" for r in rows):
        return {"status": "incomplete", "claim": None, "cells": {}, "mean_difference": None, "ci95": None}
    rng = random.Random(bootstrap_seed(STREAM, "jund-specialist-development-v1", mode))
    cells = {}
    for cell in ("jund_vs_jund", "jund_vs_blue"):
        differences = []
        for block in range(blocks):
            differences.append(sum(game_score(keyed["specialist", cell, block, s]) - game_score(keyed["legacy", cell, block, s]) for s in range(4)) / 4)
        cells[cell] = dict(difference=sum(differences) / blocks, block_differences=differences)
    draws = {cell: [] for cell in cells}
    for _ in range(replicates):
        indices = rng.choices(range(blocks), k=blocks)
        # PR62 requires the same resampled block indices across fixed cells.
        for cell, summary in cells.items():
            draws[cell].append(sum(summary["block_differences"][i] for i in indices) / blocks)
    for cell, summary in cells.items():
        summary["ci95"] = [quantile(draws[cell], .025), quantile(draws[cell], .975)]
    overall = [(a + b) / 2 for a, b in zip(*draws.values())]
    interval = [quantile(overall, .025), quantile(overall, .975)]
    loss = any(c["ci95"][1] < -.03 for c in cells.values())
    claim = "stronger-on-this-panel" if interval[0] > 0 and not loss else "regression-on-this-panel" if loss or interval[1] < 0 else "inconclusive"
    return dict(status="complete", claim=claim, cells=cells, mean_difference=sum(c["difference"] for c in cells.values()) / 2,
                ci95=interval, bootstrap_replicates=replicates, bootstrap_unit="four-slot seed block; shared resampled indices across cells")


def run(args):
    if args.blocks <= 0 or args.workers <= 0:
        raise ValueError("blocks and workers must be positive")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    freeze = source_freeze(args.baseline_revision)
    build = native_build() if args.engine == "native" else None
    manifest, specs = schedule(args.blocks, args.mode)
    rows, summaries = [], {}
    started = time.time()
    for pilot in ("specialist",) if args.smoke else ("specialist", "legacy"):
        for cell, deck in (("jund_vs_jund", "jund_wildfire"), ("jund_vs_blue", "mono_blue_terror")):
            selected = [s for s in specs if s.cell == cell]
            games = run_episodes(selected, SETTINGS, BotFactory("jund_wildfire", pilot == "specialist"), BotFactory(deck),
                                 args.engine, args.workers)
            latency = [x for r in games for x in r["latency_seconds"]]
            completed = [r for r in games if r["status"] == "completed"]
            summaries[pilot + ":" + cell] = dict(games=len(games), completed=len(completed), errors=len(games)-len(completed),
                wins=sum(r["winner"] == r["learner_seat"] for r in completed), draws=sum(r["winner"] is None for r in completed),
                losses=sum(r["winner"] == 1-r["learner_seat"] for r in completed),
                score=sum(map(game_score, completed)) / len(completed) if len(completed) == len(games) else None,
                latency_median_seconds=quantile(latency, .5), latency_p95_seconds=quantile(latency, .95))
            for r in games:
                r["pilot"] = pilot
                r["score"] = game_score(r) if r["status"] == "completed" else None
                r["latency_median_seconds"] = quantile(r.pop("latency_seconds"), .5)
                # Full final board snapshots are redundant with deterministic
                # reruns. Keep identity, outcome, failure and attempted action.
                r.pop("visible_outcome", None)
                rows.append(r)
            print(pilot, cell, summaries[pilot + ":" + cell], flush=True)
    comparison = None if args.smoke else paired_summary(rows, args.blocks, args.mode)
    result = dict(format="JundDevelopmentComparison", version=1, kind="smoke" if args.smoke else "paired",
                  status="complete" if all(r["status"] == "completed" for r in rows) else "incomplete",
                  freeze=freeze, schedule=manifest, settings=SETTINGS, engine=args.engine, native_build=build,
                  runtime=dict(machine=platform.machine(), platform=platform.platform(), python=platform.python_version(), workers=args.workers),
                  elapsed_seconds=time.time()-started, planned_games=len(specs) * (1 if args.smoke else 2),
                  summary=summaries, comparison=comparison, rows=rows)
    write_json(args.out, result)
    print("Saved", args.out, result["status"], None if comparison is None else comparison["claim"], flush=True)
    return 0 if result["status"] == "complete" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--blocks", type=int, help="default: 40 for smoke, 400 for paired")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--mode", choices=("sampled", "greedy"), default="greedy")
    ap.add_argument("--baseline-revision", default=BASELINE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.blocks is None:
        args.blocks = 40 if args.smoke else 400
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
