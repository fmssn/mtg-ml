# Evaluation and training metrics

What the trainer measures, and how to read it. Code: `mtg_ml/rl/evaluate.py` (games, Elo fits), `mtg_ml/rl/train.py` (`rollout_stats`, when evaluations run), `mtg_ml/rl/ppo.py` (update statistics).

The [scripted bot and tactical puzzle benchmark plan](benchmark-plan.md) specifies
a proposed Jund/Blue benchmark with two stronger scripted specialists, reviewed
outcome puzzles and frozen evaluation contracts. Its interfaces and commands are
not implemented yet; the metrics below describe the existing evaluator.

## Periodic evaluation

Every `--eval-every` iterations, and/or whenever the training games cross a multiple of `--eval-every-games` (e.g. `250000`), the policy of that iteration is evaluated in a process of its own. Results go into the iteration's row of `metrics.jsonl` when they arrive.

| block | flag(s) | keys | notes |
|---|---|---|---|
| sanity blocks | `--eval-blocks random,bot,pool0`, `--eval-games 40` | `eval/<opp>/<deck>`, `_ci` | random, bot/blue and pool0 saturate early, so they are kept small. `--eval-blocks ""` turns them off. |
| bo3 vs bots | `--eval-bo3-matches 100` | `eval/bot_bo3/<deck>` | |
| benchmark, sampled | `--bench-games 1000` | `bench/jund_vs_bot`, `_ci`, `_n` | learner Jund vs the blue Delver bot, each seed once per starting player |
| benchmark, greedy | `--bench-greedy-games 1000` | `bench/jund_vs_bot_greedy`, `_ci`, `_n` | same seeds, argmax play |
| benchmark bo3 | `--bench-bo3-matches 200` | `bench/jund_vs_bot_bo3` | |
| reference ladder | `--ladder a.pt,b.pt,...`, `--ladder-games 200`, `--ladder-ratings`, `--ladder-greedy` | `ladder/<rung>`, `_ci`, `ladder/elo`, `ladder/elo_se` | replaces `pool0` when given |

With 1000 games the benchmark's 95% interval is about ±3 points, which is enough to see 5-point effects. At 100 games it is ±9 points.

### Greedy play

`Job.greedy` (`--greedy` on `python -m mtg_ml.rl.evaluate`) makes every network seat, the learner and any frozen checkpoint, take its most likely option instead of sampling. Ties go to the first option. With the local evaluator this is `logits.argmax`. The inference server gets a flag bit in each decision record and turns the Gumbel noise of those rows into a constant, so the draw becomes the argmax. Recorded (training) games refuse greedy.

The engine, the bots and the random agent are all seeded, so **a greedy evaluation on paired seeds is deterministic**: the same checkpoint gets exactly the same score every time. Differences between checkpoints then come only from the seeds and the weights, with no sampling noise. The Wilson interval still says how well N seeds pin the score down. The sampled and greedy benchmarks measure different things: the policy as it was trained, and its mode. Sampled stays the default.

### Reference ladder and Elo

A score against a single opponent saturates. The ladder is a fixed set of reference checkpoints, for example the 1M, 3M, 5M and 8M snapshots of a long run. To rate them once:

```bash
python -m mtg_ml.rl.evaluate ladder r1.pt r3.pt r5.pt r8.pt --games 200 --out ladder.json --engine native
```

This plays a round robin on paired seeds and fits Elo ratings with Bradley-Terry maximum likelihood (Hunter's MM, `fit_ratings`). The first checkpoint is rated 0. The output JSON is `{"ratings": {path: elo}, "games", "greedy", "results"}`.

The trainer's `--ladder r1.pt,...` plays `--ladder-games` paired games against each rung and logs the score against each one. It also fits the learner's Elo on the ladder's scale (`fit_elo`, a 1-D maximum-likelihood fit found by bisection) along with its standard error. The rung ratings come from `--ladder-ratings`. Without that flag they come from `<run>/ladder.json`, which the first evaluation creates by playing the round robin if the file is missing. Both fits add one virtual drawn game per pairing, so a clean sweep still gives a finite rating. Each rung only contributes useful information while the learner is within about 400 Elo of it, so add a stronger rung once `ladder/elo` passes the top one.

## Per-iteration training metrics

From the rollout (`train.rollout_stats`):

- `win_vs_pool`: the learner's win rate against pool checkpoints. Self-play and bot games are excluded.
- `win_vs_pool_by_opp`: `{snapshot: [win rate, learner games]}` for each pool opponent in the iteration.
- `win_vs_bot`: the win rate in the `--bot-frac` games.
- `decisions_per_game`: the decisions made by both seats per game, forced moves included.
- `pay_mana_share`: the share of recorded learner decisions that are mana payments.

From the PPO update (`ppo.ppo_update`):

- `entropy`, `approx_kl`, `clip_frac`, ...: means over every minibatch step.
- `approx_kl_first` and `approx_kl_last`: the KL of the first and of the last epoch that ran. A gap between them means the later epochs keep pushing.
- `nt_frac`, `nt_entropy`, `nt_approx_kl`, `nt_approx_kl_first`, `nt_approx_kl_last`: the same statistics over **non-trivial** decisions only. A decision counts as non-trivial when the behaviour policy gave the option it took a probability below 0.99. This equals p_max < 0.99, except for the rare draws of a <1% option from a near-certain decision, which also count as non-trivial. The rollout records only the taken option's log-prob, so this is the test available without extra cost. Forced moves never count.
- `kind/<kind>/share`, `kind/<kind>/entropy`, `kind/<kind>/approx_kl`: the non-trivial decisions broken down by decision kind (`rollout.KINDS`: priority, pay_mana, target, declare_attacker, ...). The rollout records each decision's kind (`Result.kinds`). The per-kind sums are computed inside the training step with one `index_add` per minibatch, so they work in the CUDA-graph-captured step and add no host syncs.
