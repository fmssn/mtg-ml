# Reference ladder and Elo

## Ladder L1 (created 2026-10-07)

L1 is four frozen snapshots of the overnight self-play run `20261006-overnight-selfplay`. That run used the entity trunk at h128 with a shared value head, trained on self-play plus past snapshots, on the native engine.

Files are in `~/mtg-ml-checkpoints/ladder/L1/` on h100-private, with the ratings in `ladder.json` next to them.

| rung | snapshot | games trained | Elo |
|---|---|---|---|
| 0 (anchor) | `iter_00488.pt` | ~1.0M | **0** (fixed) |
| 1 | `iter_01464.pt` | ~3.0M | 33.3 |
| 2 | `iter_02440.pt` | ~5.0M | 50.8 |
| 3 | `iter_03904.pt` | ~8.0M | 78.8 |

The ratings come from a round robin played on 2026-10-07:
- 400 paired games per pair, sampled play;
- fixed engine, `claude/next-integration` f953f1e.

The table gives the row player's score against the column player.

| | 1M | 3M | 5M | 8M |
|---|---|---|---|---|
| 1M | | 0.470 | 0.428 | 0.370 |
| 3M | | | 0.470 | 0.458 |
| 5M | | | | 0.455 |

To rate it again:

```bash
python -m mtg_ml.rl.evaluate ladder iter_00488.pt iter_01464.pt iter_02440.pt iter_03904.pt --games 400 --out ladder.json --engine native
```

## How the Elo is computed

The code is in `mtg_ml/rl/evaluate.py` (`rate_ladder`, `fit_elo`), added in PR fmssn/mtg-ml#16.

1. **Rating the rungs.** The round robin plays every pair of rungs on paired seeds:
   - each seed twice, with the seats (decks) swapped;
   - game-1 decks;
   - a draw counts as half a win.

   The ratings are the Bradley-Terry maximum-likelihood fit on the Elo scale, P(A beats B) = 1 / (1 + 10^((R_B − R_A)/400)), solved with Hunter's MM iteration. Each pair that played gets one virtual drawn game, so a sweep stays finite. Rung 0 is pinned at 0.

   Once written, `ladder.json` is never refit. Every later measurement uses these numbers, so the scale stays fixed.
2. **Rating a checkpoint.** The trainer (`--ladder`, `--ladder-ratings`, `--ladder-games 200`) or `evaluate` plays the checkpoint against every rung: 200 paired games each, sampled unless `--ladder-greedy 1`.
   - Its Elo is the maximum-likelihood rating against the fixed rung ratings. The score equation is solved by bisection, with one virtual draw per rung.
   - The standard error comes from the Fisher information.
   - Logged as `ladder/elo`, `ladder/elo_se`, and per rung `ladder/<rung>`.
3. **Range.** The ladder spans 0 to 79, and the best checkpoint so far is ~135, already above the top rung. Ratings well above the top rung extrapolate and get noisier.

   When the best arm passes ~150, start a new ladder **L2**:
   - L2 is L1's rungs plus the new one;
   - rate the new rung against the old ones, with their ratings held fixed;
   - keep reporting L1 Elo alongside L2 for a while.

   Never edit L1.

Elo measures play against this lineage of self-play checkpoints, both decks. It is not the benchmark: a run can gain Elo and lose benchmark points (or the reverse) when it learns something specific to its own opponents. Look at both.
