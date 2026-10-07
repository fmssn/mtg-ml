# mtg-ml checkpoint archive (h100-private)

Append-only. Nothing here is ever deleted or overwritten; run directories elsewhere may be cleaned up.
The experiment ledger in the repo (`docs/experiments/`) points at these ids.

    <id>/final.pt        resume checkpoint at the end of the run (weights + Adam state + config)
    <id>/policy.pt       the last weights-only policy file (what evaluate/replay/ladder load)
    <id>/snapshots/      pool snapshots the run itself took (iter_NNNNN.pt, weights only)
    <id>/metrics.jsonl   every training row and evaluation (bench/*, ladder/*)
    <id>/run.log         trainer stdout (warnings stripped)
    <id>/launch.txt      code ref, exact flags, launch script
    <id>/evals.txt       extra evaluations done after the run (2000-game benchmark, head to head)
    <id>/SHA256          of final.pt
    ladder/L1/           the reference ladder: rung files and ladder.json (ratings, round robin)

Archive a run: `~/mtg-ml-checkpoints/archive_run.sh <id> <run dir> <code ref> <launch script> [note]`.
