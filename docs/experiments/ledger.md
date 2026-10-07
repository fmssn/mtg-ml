# Ledger

Newest first. How to add an entry, and what the numbers mean: [README](README.md). Elo is on ladder L1 ([ladder.md](ladder.md)). Archive ids refer to `~/mtg-ml-checkpoints/<id>/` on h100-private. Benchmark = learner Jund vs blue bot, game 1, sampled / greedy.

## Open (handoff 2026-10-07)

- `r3-attn` was still training on h100-private (`~/mtg-ml-next2/runs/r3-attn`, cores 48-63, GPUs 5+4, ~20 s per iteration, ends at 1M games). When it ends: `archive_run.sh 20261007-r3-attn ...`, `final_evals.sh` with `H2H="20261007-r3-attn 20261007-r3-control"`, `ladder_final.py 20261007-r3-attn` (from `~/mtg-ml-next2`), then fill its row.
- Suggested round 4: parent `20261007-r3-postboard`; settle postboard with a 5,000-game head to head; start ladder L2 (rungs: L1 + `r1-lranneal` + `r3-control`); one larger change (entity attention if it holds, a bigger model from scratch with the anneal and auto mana, or a new search design).
- Scripts: `tools/experiments/`.

## Summary

| id | change (vs parent) | games | bench sampled | bench greedy | L1 Elo | verdict |
|---|---|---|---|---|---|---|
| 20261007-red-madness-1m | new deck: Red Madness learner vs frozen r1-control Jund, warm-started from it | 1M | (vs Jund bot) 89.9% | 92.6% | | new-deck baseline⁴ |
| 20261007-r3-attn | R3 + 1 entity self-attention layer (identity init) | 9.4M + 1M | running | | | |
| 20261007-r3-postboard | R3 + 20% sideboarded games (vs 50%) | 9.4M + 1M | **74.9%** | 74.9% | **154 ± 13** | inconclusive (+) |
| 20261007-r3-botjund | R3 + 25% games learner Jund vs blue bot | 9.4M + 1M | 74.2% | 75.0% | 143 ± 13 | reject |
| 20261007-r3-control | lr-anneal final + 1M games at lr 3e-5, new features on | 9.4M + 1M | 73.0% | 74.7% | 146 ± 13 | control, **best parent** |
| 20261007-r2-automana | r2-features + auto mana and auto pass | 8.4M + 1M | 64.7%³ | 71.6%³ | 83 ± 12 | reject |
| 20261007-r2-pfsp | r2-features + PFSP pool sampling | 8.4M + 1M | 63.5% | 72.0% | 86 ± 12 | reject |
| 20261007-r2-botjund | r2-features + 25% games learner Jund vs blue bot | 8.4M + 1M | 67.5% | 75.0% | 59 ± 12 | reject |
| 20261007-r2-features | new state features and option previews (feature set 2) | 8.4M + 1M | 67.0% | 72.2% | 89 ± 12 | inconclusive (0) |
| 20261007-r1-lranneal | lr 3e-4 → 3e-5 linear over 1M games | 8.4M + 1M | **72.1%** | 73.1% | 147 ± 13 | **adopt** |
| 20261007-r1-control | none (fixed engine, value clamp) | 8.4M + 1M | 65.9% | 71.9% | 103 ± 13 | control |
| 20261007-r1-gammaturn | discount per game turn (γ 0.97, λ 0.95) | 8.4M + 1M | 55.8%¹ | 65.7%¹ | 35 ± 12 | reject |
| 20261007-r1-exploit-blue | exploiter, Blue only vs frozen main | 0.3M | 56.5% vs main² | | | no gap |
| 20261007-r1-exploit-jund | exploiter, Jund only vs frozen main | 0.3M | 46.1% vs main² | | | no gap |
| 20261006-search64-* | own-turn Gumbel AlphaZero search + distillation (PR #10, closed) | 500k + ~0.3M per attempt | 38–45% | | | **reject** |
| 20261006-overnight-selfplay | open-ended self-play + pool | 8.4M | 64.6% | 70.9% | 103 ± 13 | parent |
| 20261006-overnight-bot10 | + 10% games vs scripted bots | 8.4M | 65.6% | | 75 ± 12 | no effect |

Sampled/greedy of finished runs: 2,000 games on the fixed engine, final checkpoint. ¹ last in-training evaluation (1,000 games, 9.25M). ² sampled training games of the last iterations against the frozen main policy; starting levels 45% (Jund) and 55% (Blue). ³ evaluated without auto mana (it trained with it), so the model paid mana itself. L1 Elo: final checkpoint (policy file), 200 paired games per rung, on the code the run was trained with (round 1: feature set 1; rounds 2-3: feature set 2, under which the L1 rungs, trained on set 1, see untrained feature rows: about 1 benchmark point weaker, so round 2-3 Elos may read a few points high against round 1). Head-to-head results are in the round sections. ⁴ benchmark here = learner Red vs the scripted Jund bot (1,000 games); not comparable to the Jund-vs-blue rows.

---

## 20261007-red-madness-1m · a third deck against a frozen Jund

- **Question**: can a new deck (Red Madness, `docs/red-madness.md`) be learned in 1M games against a fixed opponent, and what do its greedy games show about the engine and the training setup.
- **Parent / opponent**: `20261007-r1-control` policy (Jund + Blue network, never trained against Red); the learner starts from its weights, the opponent stays frozen. (`r1-lranneal`, the stronger Jund, was not yet adopted when this launched.)
- **Code**: `claude/mono-red-burn-pauper-627e45` @ 09020f2 (PR fmssn/mtg-ml#23) with this branch's original flags `--matchup jund_madness --learner-seat 1 --opponent J --init-from J`, which the merge renamed to `--matchup jund_madness --exploit J --exploit-deck red`. Native engine.
- **Flags**: `--self-play-frac 0 --bot-frac 0 --engine native --device cuda --inference server --server-device cuda:1 --workers 22 --games-per-iter 2048 --total-games 1000000 --eval-every-games 100000 --bench-games 1000 --bench-greedy-games 1000 --eval-games 200`, defaults otherwise (postboard 0.5, shaping 0.2 annealed over 100 iterations). 489 iterations, 1.4 s each, ~15 minutes on cores 32-63 and two H100s.

| games | win vs frozen Jund (training, sampled, g1+g2) | vs frozen Jund (eval, g1) | vs Jund bot sampled / greedy |
|---|---|---|---|
| 0 | 0.8% | | |
| 0.1M | 64.6% | 68.5% | 75.7% / 82.6% |
| 0.5M | 77.2% | 83.5% | 84.7% / 89.8% |
| 1.0M | 78.7% | 88.5% | 89.9% / 92.6% |

- **Controls**: the frozen Jund scores only 41% (greedy, 1,000 games) against the scripted Red bot, and the Jund bot 21% against it, so the opponent is weak in this matchup; the learner (88.5% vs the same Jund) is well beyond the scripted bot (59%).
- **Review** (20 greedy games + statistics over 200 more): no engine mistakes found. Most of the win rate is the frozen Jund not knowing Red's threats; Red's own leaks are madness discards without {R} open (a third of Fiery Temper discards are lost), plot never used (0 / 366), greedy 1-land keeps, burn almost never aimed at creatures, sideboard cards near-unused. Details: `docs/red-madness-review.md`.
- **Verdict**: baseline for the new deck; the setup (one frozen opponent that never saw Red) inflates the number. Next: train both sides (self-play over the jund_madness matchup), see the review.
- **Archive**: `20261007-red-madness-1m` (evals.txt has the numbers above).

---

## 20261007-r3 · fine-tunes from the lr-anneal checkpoint

- **Question**: does continuing the best checkpoint at low lr keep improving, and do postboard share, bot games or entity attention help on top.
- **Parent**: `20261007-r1-lranneal` final (9.40M games), with its pool. Resumed at constant `--ppo-lr 3e-5` (the anneal's end value), 1M more games.
- **Code**: `claude/next-integration-2` @ bdad997 (next-integration + PRs #17 auto mana, #18 entity attention, #19 features) plus the RNG-restore fix (PR #22). Feature set 2 is on for every arm (the parent was trained on set 1: the new feature rows start untrained).
- **Flags**: as round 1 plus `--total-games 10400000 --ppo-lr 3e-5`; postboard adds `--postboard-frac 0.2`; botjund `--bot-frac 0.25 --bot-seat jund`. attn: fresh run `--init <lranneal final.pt> --entity-attn 1 --ppo-lr 3e-5 --total-games 1000000`, no copied pool (own snapshots only), fresh Adam state; ~20 s per iteration because the inference server has no stacked path for attention models.

| arm | sampled | greedy | L1 Elo | head to head (2,000 paired games) |
|---|---|---|---|---|
| control | 73.0% (71.0–74.9) | 74.7% (72.7–76.6) | 146 ± 13 | vs parent lranneal: 51.7% (49.6–53.9) |
| postboard 0.2 | 74.9% (72.9–76.7) | 74.9% (73.0–76.8) | 154 ± 13 | vs control: 52.0% (49.8–54.2) |
| botjund | 74.2% (72.2–76.0) | 75.0% (73.0–76.8) | 143 ± 13 | vs control: 50.0% (47.9–52.2) |
| attn | running | | 171 ± 13 at +0.25M | |

**Findings**
- *Continued low lr*: small further gains over the parent (sampled +0.9, greedy +1.6, head to head 51.7%, just inside noise). The run is close to its plateau at this lr.
- *Postboard 0.2*: best on every number (+1.9 sampled, Elo +9, head to head 52.0%) but each within noise. Inconclusive, leaning positive; cheap to keep. Decide with a longer run or 5,000-game head to head.
- *Bot games as Jund*: no gain over the control on anything. Reject (with round 2: same result twice).

## 20261007-r2 · four fine-tunes from the overnight checkpoint, new code

- **Parent and flags** as round 1 (overnight final, 1M more games at lr 3e-4); code `claude/next-integration-2` @ bdad997. **Feature set 2 was on in all four arms** (PR #19 was unconditional at the time; versioned since PR #24), so the three non-feature arms are compared against `r2-features`, and `r2-features` against `r1-control`.
- Extra flags: botjund `--bot-frac 0.25 --bot-seat jund`; pfsp `--pool-sampling pfsp`; automana `--auto-mana 1 --auto-pass 1`.

| arm | sampled | greedy | L1 Elo | head to head (2,000 paired games) |
|---|---|---|---|---|
| features | 67.0% (64.9–69.0) | 72.2% (70.2–74.1) | 89 ± 12 | vs r1-control: 49.0% (46.9–51.2) |
| botjund | 67.5% (65.5–69.6) | 75.0% (73.0–76.8) | 59 ± 12 | vs features: 48.6% (46.5–50.8) |
| pfsp | 63.5% (61.4–65.6) | 72.0% (70.0–73.9) | 86 ± 12 | vs features: 50.4% (48.2–52.6) |
| automana | 64.7%³ | 71.6%³ | 83 ± 12 | vs features: 50.2% (48.1–52.4) |

**Findings**
- *Features (items 10-12)*: no measurable effect in 1M games, as the probes predicted (lethal already encoded; the KCS preview may need more than 1M games to be picked up from untrained rows). Inconclusive; kept on (feature set 2) since it costs ~2 µs per decision.
- *Bot games as Jund*: +2.8 greedy on the benchmark it trains on, but lower Elo and 48.6% head to head: it specialises on the bot. Reject.
- *PFSP*: nothing. With a pool of near-copies, win rates are all ~0.5 and the weights stay flat. Reject.
- *Auto mana/pass*: trajectories 10% shorter (216 vs 241 decisions per game), no strength change. Reject as a strength lever; still useful for speed and as a cleaner action space for a fresh run.

---

## 20261007-r1 · four fine-tunes from the overnight checkpoint

- **Question**: which of the cheap training changes from `docs/next-actions.md` (items 7, 8, 14) moves strength, from a converged checkpoint.
- **Parent**: `20261006-overnight-selfplay` final (`latest.pt`, iteration 4100, 8.40M games), copied with its 34 pool snapshots into each run dir and resumed.
- **Code**: `claude/next-integration` @ f953f1e = main c68e474 + PRs #12 (infra), #14 (training knobs), #15 (attacker fix), #16 (measurement). Native engine built from the same tree.
- **Box**: h100-private, 16 cores and two H100s per arm (trainer GPU + inference-server GPU), four arms at once, ~4 s per iteration of 2,048 games.
- **Common flags** (`launch_next.sh` in each archive's `launch.txt`): `--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 --workers 13 --games-per-iter 2048 --checkpoint-every 25 --snapshot-every 122 --seed 1 --server-policy-slots 256 --eval-every 0 --eval-every-games 250000 --bench-games 1000 --bench-greedy-games 1000 --bench-bo3-matches 0 --eval-games 40 --eval-bo3-matches 0 --ladder <L1> --ladder-ratings <L1>/ladder.json --ladder-games 200 --total-games 9400000`. Value clamp to ±1 is on by default in this code (a bug fix, item 2).

| arm | extra flags | sampled | greedy | Elo at +0.1 / 0.35 / 0.6 / 0.85M | final entropy |
|---|---|---|---|---|---|
| control | – | 65.9% (63.8–68.0) | 71.9% (69.8–73.8) | 77 / 76 / 81 / 99 | 0.29 |
| lranneal | `--lr-anneal-games 1000000 --ppo-lr-final 3e-5` | 72.1% (70.1–74.0) | 73.1% (71.1–74.9) | 79 / 89 / 104 / 135 | 0.19 |
| gammaturn | `--gamma-turn 0.97 --lam-turn 0.95` | 55.8% (52.7–58.9)¹ | 65.7%¹ | 68 / 38 / 16 / 32 | 0.40 |
| exploit-jund | fresh run, `--exploit <overnight policy v04121> --exploit-deck jund --iterations 150`, no evaluation | | | | |
| exploit-blue | same with `--exploit-deck blue` | | | | |

Head to head, lranneal vs control final, 2,000 paired games: **55.6% (53.4–57.8)**.

**Findings**
- *lr anneal*: the clearest gain of the day. Sampled +6.2 points (outside both intervals), greedy +1.2 (inside), head to head 55.6%, Elo +36 at the last common point. KL fell from 0.013 to 0.0015 per iteration as the lr dropped. Part of the sampled gain is sharper sampling (entropy 0.29 → 0.19, so sampled play approaches greedy), but the head-to-head and Elo say it also plays better. **Adopt**: the next parent is its final checkpoint, and later fine-tunes should anneal too.
- *Per-turn discount*: entropy rose (0.30 → 0.40) and every number fell steadily; explained variance dropped to 0.59. The offline probe (below) had already shown it doubles the advantage variance without changing which decisions it favours. **Reject** in this form; a per-turn γ with per-decision λ was not tried.
- *Exploiters*: 150 iterations against the frozen main policy did not find a gap: Jund 45% → 46%, Blue 55% → 56.5% (≥60% was the bar). Shared blind spots exist (the KCS probe) but plain PPO against a fixed opponent does not find them in 300k games. A league (item 14) is low priority until something finds exploitable lines.

## 20261007 · offline probes on the overnight checkpoint (no training)

Scripts and outputs: `~/mtg-ml-probes/probes/` on h100-private. On-policy data from the final overnight checkpoint, 7.3M decisions (critic) and 729k decisions (the rest).

- **Critic capacity (item 9)**: held-out explained variance against the Monte Carlo outcome: existing shared head 0.38, fresh linear head on the frozen core 0.40, 2-layer MLP 0.40, fresh separate value net h128 0.38, h256 0.38; fine-tuning the head 0.41. A bigger critic does not help; outcome noise and the per-decision discount bound it. **Dropped** the separate-critic arm. 2.2% of values were outside [−1, 1] (range −1.35 to 1.44), so the clamp matters.
- **Discount (item 8)**: per-decision and per-turn GAE rank "play land / cast creature" over "pass" the same way; per-turn has twice the spread. 8.3 decisions per player-turn on average (p90 17), 47% passes, 12.8% mana payments.
- **KCS (item 12)**: mean P(activate Krark-Clan Shaman) 0.19 in own-main positions where it would kill an opposing X/1, 0.06 where it kills nothing: only 2–3× more likely when it matters, bimodal (near 0 or near 1).
- **Lethal (item 10)**: the core linearly encodes evasive lethal (held-out AUC 0.91 at opponent life ≤ 6, against 0.83 for a random-network control) and the policy wins 95% of turns where evasive lethal is on board. Readiness features are a minor fix.

## 20261006-search64 · own-turn search distillation (PR fmssn/mtg-ml#10, closed unmerged)

- **Question**: can a small own-turn search (Gumbel AlphaZero, budget 64) find multi-step combos the policy never samples (Krark-Clan Shaman + Toxin Analysis sweep, sampled at p ≈ 1e-7) and teach them to the policy by distillation.
- **Parent**: `model_v3-entity-500k-bot-s6` (entity h128, 500k games). Benchmark at the parent's end: 52% (100-game benchmark, old engine).
- **Code**: branch `claude/jund-toxic-analysis-misplay-752403` @ 07abffd; design and measurements in `docs/search.md` on that branch. Runs on h100-private: `~/mtg-ml-search/runs/{search64-from500k, search64-vcritic}`.
- **Attempts**: 1–4 collapsed or were dominated by a few sharp targets (KL up to 0.16). Attempt 5: margin gate 0.15, minibatch-mean distillation loss. Attempt 6: search value as the critic target too. Each ran ~300k games.
- **Result**: the search finds the line reliably (budget 64, ~840 evaluations per search), but distillation did not teach it. The benchmark stayed at 38–45% in both attempts, 7–14 points below the parent's 52%. The probed probabilities oscillated (Cast Shaman at the seed-6 decision 0.10 → 0.32 → 0.13), the sweep activation reached only 2.6%, entropy rose from 0.33 to 0.42, and critic values drifted down. Play-time search alone: 61.3% vs 62.5% without it (n = 80).
- **Verdict: reject.** Not merged. Known bugs if it is ever revived: determinization is lost below the root (`fork()` replays the action history and drops the hidden-card edits), the evaluation cap is not enforced, and the margin gate can compare against an unevaluated reference action. Ideas not tried: search as an auditor over many games, averaging leaves over determinizations, a combo-opportunity counter.

## 20261006-overnight · A/B, open-ended self-play vs + bot games

- **Question**: do 10% games against the scripted bots help self-play, and how far does the model get with ~100× more games than the 500k run.
- **Code**: main 2fccf29 + PR #12 branch @ a3a70b7 (server request size, graph recapture fixes). Old engine (attacker trap present).
- **Flags**: `--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 --workers 13 --games-per-iter 2048 --checkpoint-every 50 --eval-every 244 --snapshot-every 122 --bench-games 100 --bench-bo3-matches 0 --seed 0 --server-request-ints 4194304 --server-policy-slots 256`; B adds `--bot-frac 0.1`. Restart loop resuming `latest.pt`. Cores 32-47 / 48-63, GPUs 4+7 / 0+1.
- **Result**: both reached ~8.4M games overnight and plateaued from ~5M. The 100-game benchmark read ~70% for both; the 1,000/2,000-game re-evaluation says 65.9% / 65.3% (old engine) and 64.6% / 65.6% (fixed engine). Entropy 0.98 → 0.35 by 0.6M games, 0.30 at the end; explained variance flat at ~0.85 from 0.1M. Bot games had no measurable effect: half of them put the learner on Blue against a Jund bot it already beat, so only ~3% of training was the benchmark matchup (fixed by `--bot-seat jund`, round 2).
- Game reviews (6 sampled, 5 greedy) and four audits behind `docs/next-actions.md`.
