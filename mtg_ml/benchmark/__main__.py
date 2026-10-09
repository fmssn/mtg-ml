"""Frozen benchmark: validate artifacts, run candidates, compare, and release controls.

  validate    freeze, native build, reviewed witnesses and mistakes, on each engine
  run         games + puzzles for a checkpoint or registered agent -> BenchmarkResult
  compare     paired differences of two complete, compatible results -> BenchmarkComparison
  turn-limit  rerun the same seeded games under longer turn caps -> BenchmarkTurnLimitSensitivity
  calibrate   controls: witnesses pass, mistakes fail, engines agree, errors cannot inflate
  archive     build / verify a durable bundle with SHA256SUMS
  puzzles     descriptive development puzzle attempts (not a BenchmarkResult)

Outputs are written atomically and never overwritten. `run` exits nonzero unless the result is
complete; `compare` refuses incompatible or incomplete results. See docs/benchmark-release.md.
"""

import argparse
import platform
import subprocess
import sys

from .jsonio import write_json
from .release import CheckpointLearner  # noqa: F401  (kept importable from here)
from .validation import validate


def _learner(args, parser, manifest):
    from .artifacts import DIAGNOSTIC, FAIR
    from .release import agent_learner, checkpoint_learner
    from .tactics import specialist_registry
    contract = {"fair": FAIR, "diagnostic": DIAGNOSTIC}[args.contract]
    if args.checkpoint:
        return checkpoint_learner(args.checkpoint, contract)
    return agent_learner(args.agent, manifest, specialist_registry(manifest.data["freeze"]["code_revision"]))


def _split(value):
    return value.split(",") if value else None


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


def run(args, parser):
    from .artifacts import load_manifest
    from .release import run_benchmark
    manifest = load_manifest(args.manifest)
    result = run_benchmark(args.manifest, _learner(args, parser, manifest), engine=args.engine, workers=args.workers,
                           cells=_split(args.cells), modes=_split(args.modes), out=args.out, resume=args.resume, command=sys.argv)
    print(f"{args.out}: {result['status']}")
    if result["status"] != "complete":
        parser.exit(1, "benchmark: result is not complete (technical errors); nothing was scored\n")


def compare(args, parser):
    from .release import compare_results
    compare_results(args.baseline, args.candidate, args.out)
    print(args.out)


def turn_limit(args, parser):
    from .artifacts import load_manifest
    from .release import turn_limit_sensitivity
    manifest = load_manifest(args.manifest)
    caps = tuple(int(c) for c in args.caps.split(","))
    report = turn_limit_sensitivity(args.manifest, _learner(args, parser, manifest), caps, engine=args.engine, workers=args.workers,
                                    cells=_split(args.cells), modes=_split(args.modes), out_dir=args.out_dir, command=sys.argv)
    print(f"{args.out_dir}/turn-limit.json: {report['status']}")
    if report["status"] == "incomplete":
        parser.exit(1, "benchmark: technical errors; sensitivity not evaluated\n")


def calibrate(args, parser):
    from .release import calibrate as run_calibration
    report = run_calibration(args.manifest, args.engines.split(","), args.out, args.checkpoint, workers=args.workers, blocks=args.blocks, command=sys.argv)
    for check in report["checks"]:
        print(f"{'pass' if check['passed'] else 'FAIL'}  {check['id']}" + "".join(f"\n      {p}" for p in check["problems"]))
    if report["status"] != "passed":
        parser.exit(1, "benchmark: calibration failed\n")


def archive(args, parser):
    from . import archive as bundle
    if args.action == "build":
        path = bundle.build(args.out, manifest=args.manifest, results=args.result, comparisons=args.comparison, calibration=args.calibration,
                            sensitivity=args.sensitivity, validations=args.validation)
        print(path)
    else:
        print(bundle.verify(args.directory))


def add_learner(command):
    learner = command.add_mutually_exclusive_group(required=True)
    learner.add_argument("--checkpoint")
    learner.add_argument("--agent", help="benchmark-jund@1, benchmark-blue@1 (fair), legacy-jund or legacy-blue (diagnostic); plays only its own deck's cells")
    command.add_argument("--contract", choices=["fair", "diagnostic"], default="fair", help="information contract of the candidate (checkpoints below feature set 7 need diagnostic)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    command = commands.add_parser("run", help="games and puzzles for one candidate -> BenchmarkResult")
    command.add_argument("--manifest", required=True)
    add_learner(command)
    command.add_argument("--engine", choices=["python", "native"], default="python")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--cells", help="comma-separated cells (a partial panel; never a full-suite headline)")
    command.add_argument("--modes", help="comma-separated subset of the manifest modes")
    command.add_argument("--resume", action="store_true", help="reuse clean chunks of this exact run in <out>.parts; error rows are never retried")
    command.add_argument("--out", required=True)
    command = commands.add_parser("compare", help="paired treatment-minus-baseline comparison")
    command.add_argument("--baseline", required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--out", required=True)
    command = commands.add_parser("turn-limit", help="does the turn cap change the scores?")
    command.add_argument("--manifest", required=True)
    add_learner(command)
    command.add_argument("--engine", choices=["python", "native"], default="python")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--cells")
    command.add_argument("--modes")
    command.add_argument("--caps", default="100,200,400")
    command.add_argument("--out-dir", required=True)
    command = commands.add_parser("calibrate", help="release controls on the manifest's corpus and specialists")
    command.add_argument("--manifest", required=True)
    command.add_argument("--engines", default="python,native")
    command.add_argument("--checkpoint", help="use this checkpoint (sampled) for the self-compare control instead of a specialist")
    command.add_argument("--workers", type=int, default=2, help="worker count compared against 1 in the self-compare control")
    command.add_argument("--blocks", type=int, default=4, help="blocks per cell in the game controls")
    command.add_argument("--out", required=True)
    command = commands.add_parser("archive", help="build or verify a durable release bundle")
    actions = command.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build")
    build.add_argument("--out", required=True)
    build.add_argument("--manifest", required=True)
    build.add_argument("--result", action="append", default=[])
    build.add_argument("--comparison", action="append", default=[])
    build.add_argument("--calibration")
    build.add_argument("--sensitivity")
    build.add_argument("--validation", action="append", default=[])
    check = actions.add_parser("verify")
    check.add_argument("directory")
    args = parser.parse_args(argv)
    handlers = {"puzzles": puzzles, "run": run, "compare": compare, "turn-limit": turn_limit, "calibrate": calibrate, "archive": archive}
    try:
        if args.command in handlers:
            return handlers[args.command](args, parser)
        report = validate(args.manifest, tuple(args.engines.split(",")))
        write_json(args.out, report)
    except (ValueError, OSError) as e:
        parser.exit(1, f"benchmark: {e}\n")
    if report["status"] != "complete":
        parser.exit(1, "benchmark: " + "; ".join(report["errors"]) + "\n")


if __name__ == "__main__":
    main()
