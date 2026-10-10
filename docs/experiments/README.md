# Experiment ledger

Every training run that is meant to tell us something gets an entry here, kept even when the idea failed. Four places together:

| what | where |
|---|---|
| what was done, why, and what came out (human) | [`ledger.md`](ledger.md), newest first |
| the same, machine-readable, one JSON object per run | [`ledger.jsonl`](ledger.jsonl) |
| the checkpoints, metrics and launch commands | `~/mtg-ml-checkpoints/<id>/` on h100-private (append-only, see its README) |
| which run is which: run IDs, handles, legacy names, current roles | [`naming.md`](naming.md), [`models.md`](models.md) / [`models.json`](models.json) |
| the reference ladder the Elo numbers are measured on | [`ladder.md`](ladder.md); files in `~/mtg-ml-checkpoints/ladder/L1/` |

## Recording a run

0. **Name it by the scheme in [`naming.md`](naming.md)** (`<yyyymmdd>-<campaign>-<arch>-<scope>-<init>[-<variant>]-s<seed>`, plus a short handle) and add the row to `models.md` and `models.json` when you launch, status `running`. Finished runs update their row (status, games, bench, L1, archive path). A change of role (`best/general`, `play/<deck>`, ...) is made in the same two files, with a line in `role_history` and the reason in the ledger entry or PR. Old names are never renamed; they go in the legacy column.
1. Before launching, pick the run ID (archive id `YYYYMMDD-<name>`; older runs used `YYYYMMDD-<short-name>`) and write down the parent checkpoint (an archive id), the code ref (branch @ commit, deployed as-is to the box) and the exact flags. Change one thing per arm and keep a control arm with the same parent, code and seed.
2. Train with the standard evaluation on: `--eval-every-games 250000 --bench-games 1000 --bench-greedy-games 1000 --ladder <L1 rungs> --ladder-ratings <L1 ladder.json> --ladder-games 200` (see `ladder.md`), so every run reports the same three numbers.
3. When it ends, archive it: `~/mtg-ml-checkpoints/archive_run.sh <id> <run dir> "<branch @ commit>" <launch script> [note]`. Then evaluate the final checkpoint:
   - on 2,000 benchmark games, sampled and greedy: `python -m mtg_ml.rl.evaluate <id>/policy.pt bot --jund --games 2000 [--greedy]`;
   - head to head against its control, when there is one: `python -m mtg_ml.rl.evaluate <arm> <control> --games 2000`.

   Put these in `<id>/evals.txt`.
4. Add the entry to `ledger.md` and a line to `ledger.jsonl`, with a verdict:
   - **adopt**: becomes the default or the next parent;
   - **reject**;
   - **inconclusive**, with what would decide it.

## Reading the numbers

- **Benchmark**: the learner plays Jund Wildfire against the scripted Mono Blue Terror (Delver) bot.
  - Game-1 decks, the same paired seeds every time (each seed once per starting player).
  - *Sampled* plays as in training; *greedy* takes the most likely option.
  - 1,000 games give about ±3 points (Wilson 95%), 2,000 about ±2.
  - Before 2026-10-07 the trainer benchmarked 100 games (±9 points): treat older single numbers as noise.
- **Ladder Elo**: strength against fixed past checkpoints on a fixed scale, see `ladder.md`. About ±12 per evaluation at 200 games per rung.
- **Head to head**: two checkpoints on paired seeds with seats swapped, both decks. The cleanest test between two arms.
- **Turn limit**: the default is no limit since 2026-10-11 (`max_turns=None`; games still end by decking). Every ledger entry before that was played with the old default of 100 turns (both players' turns counted, turn 101 a draw). An explicit `max_turns` still caps. Draws were near zero at 100, so older numbers should carry over, but do not compare to the last decimal.
- **Engine version**: engine changes move all numbers, so compare runs only on the same engine version (the ledger records the code ref). The attacker fix of 2026-10-07 (PR #15) moved the overnight checkpoint from 65.9% to 64.6% sampled.
