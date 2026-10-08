"""Blue specialist development diagnostics; not the checkpoint release CLI.

.venv/bin/python tools/benchmark_blue.py --phase compare --engine native \
    --workers 4 --out .context/blue-comparison
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from mtg_ml.benchmark import LegacyAdapter, episodes, run_episodes
from mtg_ml.benchmark.blue import blue_factory, PARAMETERS, BOT_ID, RULES_REVISION
from mtg_ml.benchmark.artifacts import Artifact, FAIR, DIAGNOSTIC
from mtg_ml.benchmark.jsonio import digest, file_digest, write_json
from mtg_ml.benchmark.schedule import bootstrap_seed
from mtg_ml.benchmark.views import thaw
from mtg_ml.benchmark.validation import native_build
from mtg_ml.bots import make_bot
from mtg_ml.engine import JUND_WILDFIRE, MONO_BLUE_TERROR
from mtg_ml.engine.cards import SPEC_PATH

BASELINE_REVISION = "66345da4c5bff4a1a31ab0047ff8b1abf3869c49"
LEGACY_FILES = ("mtg_ml/bots/base.py", "mtg_ml/bots/blue.py", "mtg_ml/bots/jund.py")
SETTINGS = dict(max_turns=100, max_decisions=10000, auto_single=False, auto_mana=False, auto_pass=False)
ROOT = Path(__file__).resolve().parents[1]
DECK_HASHES = {"jund_wildfire": "998d97f37438776c175fe75ca50926eaa2490df3b869e627ea88b2391bc4b6bd",
               "mono_blue_terror": "7d7681cf282f5cf7d79ca2cc8629e3562d43af70078d13b23835afcc02fd17c8"}


@dataclass(frozen=True)
class LegacyFactory:
    deck: str

    def __call__(self, seat, mode):
        return LegacyAdapter(lambda p: make_bot(p, self.deck), seat)


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def freeze_sources(baseline):
    hashes = {}
    for path in LEGACY_FILES:
        expected = subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=ROOT)
        actual = (ROOT / path).read_bytes()
        if actual != expected:
            raise ValueError(f"legacy source differs from baseline: {path}")
        hashes[path] = sha256(actual).hexdigest()
    return hashes


def development_manifest(blocks):
    decks = {name: {"cards": dict(cards), "sha256": digest(cards)} for name, cards in
             (("jund_wildfire", JUND_WILDFIRE), ("mono_blue_terror", MONO_BLUE_TERROR))}
    if any(deck["sha256"] != DECK_HASHES[name] for name, deck in decks.items()):
        raise ValueError("deck differs from PR62 freeze")
    # Deliberately a private schedule configuration, not a release manifest:
    # fixed diagnostic opponents are mixed with a fair-input candidate.
    data = {"id": "blue-specialist-development", "stream": "benchmark-v1/dev", "modes": ["greedy"],
            "blocks_per_cell": blocks, "decks": decks,
            "bots": [{"id": "legacy-jund", "deck": "jund_wildfire"}, {"id": "legacy-blue", "deck": "mono_blue_terror"}],
            "cells": [{"id": "blue_vs_jund", "learner_deck": "mono_blue_terror", "opponent": "legacy-jund"},
                      {"id": "blue_vs_blue", "learner_deck": "mono_blue_terror", "opponent": "legacy-blue"}]}
    return Artifact(ROOT / ".context" / "blue-development-schedule.json", data, digest(data))


def identity(row):
    return row["mode"], row["cell"], row["block"], row["slot"]


def score(row):
    if row["status"] != "completed":
        raise ValueError("technical errors cannot enter paired scores")
    return .5 if row["winner"] is None else float(row["winner"] == row["learner_seat"])


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def paired_summary(candidate, baseline, planned, replicates=10000):
    """Strict joining, four-slot blocks, and shared resampling across cells."""
    expected = {s.identity: s for s in planned}
    if len(expected) != len(planned) or not expected:
        raise ValueError("empty/duplicate planned rows")
    groups = []
    for rows in (candidate, baseline):
        indexed = {identity(r): r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(expected):
            raise ValueError("incomplete, unexpected or duplicate rows")
        for key, r in indexed.items():
            spec = asdict(expected[key])
            for field in ("simulator_seed", "actor_seed", "learner_seat", "starting_player", "deck_ids", "opponent"):
                if r[field] != spec[field] and not (isinstance(r[field], list) and tuple(r[field]) == spec[field]):
                    raise ValueError(f"row mapping mismatch: {key} {field}")
            score(r)
        groups.append(indexed)
    paired = defaultdict(lambda: defaultdict(list))
    for key in sorted(expected):
        _, cell, block, _ = key
        paired[cell][block].append(score(groups[0][key]) - score(groups[1][key]))
    blocks = {cell: sorted(rows) for cell, rows in paired.items()}
    if any(len(values) != 4 for rows in paired.values() for values in rows.values()) or len({tuple(v) for v in blocks.values()}) != 1:
        raise ValueError("incomplete four-slot or unmatched cell blocks")
    differences = {cell: [sum(paired[cell][b]) / 4 for b in blocks[cell]] for cell in paired}
    n = len(next(iter(blocks.values())))
    rng = random.Random(bootstrap_seed("benchmark-v1/dev", "blue-specialist-development", "greedy"))
    sampled = {cell: [] for cell in differences}
    means = []
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        values = []
        for cell, diffs in differences.items():
            value = sum(diffs[i] for i in indices) / n
            sampled[cell].append(value)
            values.append(value)
        means.append(sum(values) / len(values))
    cells = {}
    for cell, diffs in differences.items():
        rs = [[r for r in indexed.values() if r["cell"] == cell] for indexed in groups]
        cells[cell] = {"blocks": n, "games_per_arm": len(rs[0]), "candidate_score": sum(map(score, rs[0])) / len(rs[0]),
                       "baseline_score": sum(map(score, rs[1])) / len(rs[1]), "difference": sum(diffs) / n,
                       "ci95": [percentile(sampled[cell], .025), percentile(sampled[cell], .975)] if n > 1 else None}
    mean = sum(sum(d) / n for d in differences.values()) / len(differences)
    ci = [percentile(means, .025), percentile(means, .975)] if n > 1 else None
    strong = ci is not None and ci[0] > 0 and all(c["ci95"][1] >= -.03 for c in cells.values())
    return {"per_cell": cells, "equal_weight_mean_difference": mean, "ci95": ci,
            "bootstrap": {"replicates": replicates, "unit": "four-slot-block", "shared_indices_across_cells": True,
                          "confidence": .95, "method": "percentile"}, "strength_gate_passed": strong}


def strip_timing(rows):
    return [{k: v for k, v in r.items() if k not in {"elapsed_seconds", "latency_seconds"}} for r in rows]


def replay_errors(specs, rows, settings, learner, opponent, engine, out, arm):
    by_id = {s.identity: s for s in specs}
    for r in rows:
        if r["status"] == "error":
            replay = run_episodes([by_id[identity(r)]], settings, learner, opponent, engine=engine, workers=1, record=True)[0]
            path = out / "failures" / f"{arm}-{r['cell']}-{r['block']}-{r['slot']}.json"
            write_json(path, replay)
            r["replay_reference"] = {"path": str(path.relative_to(out)), "sha256": file_digest(path)}


def run_panel(specs, engine, workers, out, arm):
    learner = blue_factory if arm == "candidate" else LegacyFactory("mono_blue_terror")
    rows = []
    # Chunked API calls provide durable progress/results during long campaigns.
    for cell in sorted({s.cell for s in specs}):
        selected = [s for s in specs if s.cell == cell]
        opponent = LegacyFactory("jund_wildfire" if cell == "blue_vs_jund" else "mono_blue_terror")
        for offset in range(0, len(selected), 160):
            batch = selected[offset:offset + 160]
            result = run_episodes(batch, SETTINGS, learner, opponent, engine=engine, workers=workers)
            replay_errors(batch, result, SETTINGS, learner, opponent, engine, out, arm)
            write_json(out / "rows" / f"{arm}-{cell}-{offset:04}.json", result)
            rows.extend(result)
            print(f"{arm} {cell}: {offset + len(result)}/{len(selected)} games; errors={sum(r['status']=='error' for r in result)}", flush=True)
            if any(r["status"] != "completed" for r in result):
                return rows
    return sorted(rows, key=identity)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("smoke", "compare", "verify"), default="smoke")
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--blocks", type=int, help="Default 40 smoke, 400 comparison, 1 verification")
    ap.add_argument("--baseline-revision", default=BASELINE_REVISION)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    blocks = args.blocks if args.blocks is not None else {"smoke": 40, "compare": 400, "verify": 1}[args.phase]
    if blocks < 1 or args.workers < 1:
        ap.error("blocks and workers must be positive")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", "mtg_ml", "tools/benchmark_blue.py"], cwd=ROOT)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "mtg_ml", "tools/benchmark_blue.py"], cwd=ROOT)
    if dirty or untracked:
        ap.error("commit source before freezing a development campaign")
    args.out.mkdir(parents=True, exist_ok=False)
    legacy_hashes = freeze_sources(args.baseline_revision)
    code = revision()
    native_revision = native_build() if args.engine == "native" or args.phase == "verify" else None
    schedule = development_manifest(blocks)
    specs = episodes(schedule)
    source_files = {p: file_digest(ROOT / p) for p in subprocess.check_output(
        ["git", "ls-files", "mtg_ml", "tools/benchmark_blue.py"], cwd=ROOT, text=True).splitlines()}
    hardware = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    provenance = {"code_revision": code, "native_build": native_revision, "legacy_revision": args.baseline_revision,
                  "legacy_sources": legacy_hashes, "card_spec_sha256": file_digest(SPEC_PATH), "decks": schedule.data["decks"],
                  "candidate": {"id": BOT_ID, "rules_revision": RULES_REVISION, "parameters_sha256": digest(thaw(PARAMETERS)),
                                "information_contract": FAIR}, "baseline_contract": DIAGNOSTIC,
                  "opponent_contract": DIAGNOSTIC, "source_files": source_files,
                  "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": hardware, "workers": args.workers}}
    write_json(args.out / "freeze.json", provenance)
    write_json(args.out / "planned.json", [asdict(s) for s in specs])
    started = time.monotonic()
    candidate = run_panel(specs, args.engine, args.workers, args.out, "candidate")
    complete = len(candidate) == len(specs) and all(r["status"] == "completed" for r in candidate)
    comparison = None
    verification = None
    if args.phase == "compare" and complete:
        baseline = run_panel(specs, args.engine, args.workers, args.out, "baseline")
        complete = len(baseline) == len(specs) and all(r["status"] == "completed" for r in baseline)
        if complete:
            comparison = paired_summary(candidate, baseline, specs)
    elif args.phase == "verify" and complete:
        native = run_panel(specs, "native", 1, args.out / "native-single", "candidate")
        python = run_panel(specs, "python", 1, args.out / "python-single", "candidate")
        # Labels/timing can differ, but structured inputs/selected actions and
        # complete public outcomes must be exactly reproducible.
        verification = {"worker_repeatable": strip_timing(candidate) == strip_timing(native) if args.engine == "native" else strip_timing(candidate) == strip_timing(python),
                        "engine_parity": strip_timing(native) == strip_timing(python)}
        complete = all(verification.values())
    latency = [t for r in candidate for t in r["latency_seconds"]]
    if any(file_digest(ROOT / p) != h for p, h in source_files.items()):
        complete = False
        comparison = None
        verification = {"source_unchanged": False}
    report = {"format": "BlueSpecialistDevelopment", "version": 1, "phase": args.phase,
              "status": "complete" if complete else "incomplete", "engine": args.engine, "freeze": provenance,
              "settings": SETTINGS, "stream": schedule.data["stream"], "mode": "greedy", "planned_games_per_arm": len(specs),
              "candidate_completed": sum(r["status"] == "completed" for r in candidate), "comparison": comparison,
              "verification": verification, "elapsed_seconds": time.monotonic() - started,
              "candidate_latency": {"median": percentile(latency,.5), "p95": percentile(latency,.95)} if latency else None,
              "row_files": [{"path": str(p.relative_to(args.out)), "sha256": file_digest(p)} for p in sorted(args.out.glob("rows/*.json"))]}
    write_json(args.out / "report.json", report)
    print(f"{report['status']}: {args.out / 'report.json'}", flush=True)
    if not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
