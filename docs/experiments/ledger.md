# Ledger

Newest first. How to add an entry, and what the numbers mean: [README](README.md). Elo is on ladder L1 ([ladder.md](ladder.md)). Archive ids refer to `~/mtg-ml-checkpoints/<id>/` on h100-private. Benchmark = learner Jund vs blue bot, game 1, sampled / greedy.

## Summary

| id | change (vs parent) | games | bench sampled | bench greedy | L1 Elo | verdict |
|---|---|---|---|---|---|---|
| 20261007-r2-* | round 2: features, bot games as Jund, PFSP, auto mana | 1M each | running | | | |
| 20261007-r1-lranneal | lr 3e-4 → 3e-5 linear over 1M games | 8.4M + 1M | **72.1%** | 73.1% | 135 ± 13 (at +0.85M) | **adopt** |
| 20261007-r1-control | none (fixed engine, value clamp) | 8.4M + 1M | 65.9% | 71.9% | 99 ± 13 (at +0.85M) | control |
| 20261007-r1-gammaturn | discount per game turn (γ 0.97, λ 0.95) | 8.4M + 1M | 55.8%¹ | 65.7%¹ | 32 ± 12¹ | reject |
| 20261007-r1-exploit-blue | exploiter, Blue only vs frozen main | 0.3M | 56.5% vs main² | | | no gap |
| 20261007-r1-exploit-jund | exploiter, Jund only vs frozen main | 0.3M | 46.1% vs main² | | | no gap |
| 20261006-overnight-selfplay | open-ended self-play + pool | 8.4M | 64.6% | 70.9% | ~79 | parent |
| 20261006-overnight-bot10 | + 10% games vs scripted bots | 8.4M | 65.6% | | | no effect |

Sampled/greedy of finished runs: 2,000 games on the fixed engine, final checkpoint. ¹ last in-training evaluation (1,000 games, 9.25M). ² sampled training games of the last iterations against the frozen main policy; starting levels 45% (Jund) and 55% (Blue).

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

## 20261006-overnight · A/B, open-ended self-play vs + bot games

- **Question**: do 10% games against the scripted bots help self-play, and how far does the model get with ~100× more games than the 500k run.
- **Code**: main 2fccf29 + PR #12 branch @ a3a70b7 (server request size, graph recapture fixes). Old engine (attacker trap present).
- **Flags**: `--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 --workers 13 --games-per-iter 2048 --checkpoint-every 50 --eval-every 244 --snapshot-every 122 --bench-games 100 --bench-bo3-matches 0 --seed 0 --server-request-ints 4194304 --server-policy-slots 256`; B adds `--bot-frac 0.1`. Restart loop resuming `latest.pt`. Cores 32-47 / 48-63, GPUs 4+7 / 0+1.
- **Result**: both reached ~8.4M games overnight and plateaued from ~5M. The 100-game benchmark read ~70% for both; the 1,000/2,000-game re-evaluation says 65.9% / 65.3% (old engine) and 64.6% / 65.6% (fixed engine). Entropy 0.98 → 0.35 by 0.6M games, 0.30 at the end; explained variance flat at ~0.85 from 0.1M. Bot games had no measurable effect: half of them put the learner on Blue against a Jund bot it already beat, so only ~3% of training was the benchmark matchup (fixed by `--bot-seat jund`, round 2).
- Game reviews (6 sampled, 5 greedy) and four audits behind `docs/next-actions.md`.
