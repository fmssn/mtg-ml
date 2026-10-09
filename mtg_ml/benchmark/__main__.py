"""Artifact/witness validation and development tactical puzzle attempts.

`run`/`compare` (paired benchmark statistics) are not released yet; `puzzles`
writes a descriptive development report, not a BenchmarkResult.
"""

import argparse
import platform
import subprocess
import sys

from .jsonio import write_json
from .validation import validate


def puzzles(args, parser):
    from .artifacts import DIAGNOSTIC, FAIR, load_manifest
    from .tactics import ERROR, SPECIALISTS, run_puzzles, specialist_registry, summarize
    from .validation import ROOT
    manifest = load_manifest(args.manifest)
    d = manifest.data
    if not {b["id"] for b in d["bots"]} <= set(SPECIALISTS):
        parser.exit(1, "benchmark: puzzles needs a manifest pinning the two specialists\n")
    # Freeze, native build identity, witnesses, mistakes and response coverage.
    report = validate(args.manifest, (args.engine,))
    if report["status"] != "complete":
        parser.exit(1, "benchmark: " + "; ".join(report["errors"]) + "\n")
    registry = specialist_registry(d["freeze"]["code_revision"])
    cells = args.cells.split(",") if args.cells else None
    contract = {"fair": FAIR, "diagnostic": DIAGNOSTIC}[args.contract]
    if args.checkpoint:
        from .adapters import CheckpointAdapter
        from .jsonio import file_digest
        learner = CheckpointLearner(args.checkpoint, contract)
        CheckpointAdapter(args.checkpoint, 0, "greedy", contract)  # admission check before any attempt
        candidate = {"checkpoint": args.checkpoint, "sha256": file_digest(args.checkpoint), "information_contract": contract}
    else:
        from .tactics import ResponseFactory
        from dataclasses import asdict
        learner = ResponseFactory(registry, args.agent)
        metadata = registry.metadata(args.agent)
        candidate = {"agent": args.agent, **asdict(metadata)}
        # A scripted pilot only plays its own deck's puzzles.
        own = [c["id"] for c in d["cells"] if c["learner_deck"] == metadata.deck]
        if cells is not None and not set(cells) <= set(own):
            parser.exit(1, f"benchmark: {args.agent} plays only cells {own}\n")
        cells = cells or own
    modes = args.modes.split(",") if args.modes else None
    rows = run_puzzles(manifest, learner, registry, engine=args.engine, cells=cells, modes=modes, workers=args.workers)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = {"format": "BenchmarkPuzzleReport", "version": 1,
           "scope": "development puzzle attempts; descriptive only, no intervals, not a released BenchmarkResult",
           "manifest": {"path": str(manifest.path), "sha256": manifest.sha256}, "candidate": candidate,
           "engine": args.engine, "native_build": report.get("native_build"), "code_revision": revision,
           "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "workers": args.workers},
           "status": "complete" if all(r["status"] != ERROR for r in rows) else "incomplete",
           "summary": summarize(rows), "rows": rows}
    try:
        write_json(args.out, out)
    except (OSError, ValueError) as e:
        parser.exit(1, f"benchmark: {e}\n")
    if out["status"] != "complete":
        parser.exit(1, "benchmark: technical errors in puzzle attempts; see report\n")


class CheckpointLearner:
    """Picklable factory; one adapter per worker process, seat and mode."""
    _cache = {}

    def __init__(self, path, contract):
        self.path, self.contract = path, contract

    def __call__(self, seat, mode):
        from .adapters import CheckpointAdapter
        key = self.path, self.contract, seat, mode
        if key not in self._cache:
            self._cache[key] = CheckpointAdapter(self.path, seat, mode, self.contract)
        return self._cache[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate")
    command.add_argument("--manifest", required=True)
    command.add_argument("--engines", default="python,native")
    command.add_argument("--out", required=True)
    command = commands.add_parser("puzzles", help="attempt every planned puzzle case with a checkpoint or registered agent")
    command.add_argument("--manifest", required=True)
    learner = command.add_mutually_exclusive_group(required=True)
    learner.add_argument("--checkpoint")
    learner.add_argument("--agent", help="registered agent id, e.g. benchmark-jund@1 (a scripted baseline)")
    command.add_argument("--contract", choices=["fair", "diagnostic"], default="fair")
    command.add_argument("--engine", choices=["python", "native"], default="python")
    command.add_argument("--modes", help="comma-separated subset of the manifest modes")
    command.add_argument("--cells", help="comma-separated cells; selects learner-deck puzzles")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "puzzles":
        try:
            return puzzles(args, parser)
        except ValueError as e:
            parser.exit(1, f"benchmark: {e}\n")
    report = validate(args.manifest, tuple(args.engines.split(",")))
    try:
        write_json(args.out, report)
    except (OSError, ValueError) as e:
        parser.exit(1, f"benchmark: {e}\n")
    if report["status"] != "complete":
        parser.exit(1, "benchmark: " + "; ".join(report["errors"]) + "\n")


if __name__ == "__main__":
    main()
