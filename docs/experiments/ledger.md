# Ledger

Newest first. How to add an entry, and what the numbers mean: [README](README.md). Elo is on ladder L1 ([ladder.md](ladder.md)). Archive ids refer to `~/mtg-ml-checkpoints/<id>/` on h100-private. Benchmark = learner Jund vs blue bot, game 1, sampled / greedy.

## 20261009-r8 · three-arm follow-up to r7: continue lr075, feature set 8 + belief head, Jund-vs-Blue fine-tune

- **Status:** `r8-lr075-continue` and `r8-jund-blue-ft` running since 18:35 Berlin October 9 (16:35 UTC). **`r8-fs8-belief` failed at 18:43 Berlin after 108,409 games (iteration 58, 7 min)** with `NativeRulesError: decision 'choose_card' (Monstrous Emergence: choose a creature you control or reveal a creature card) has no options`, raised in a rollout worker (the native engine offered the additional-cost choice with no legal option; the smoke of 6 updates did not hit it). Only this arm samples varied deck lists (`--variants train`); the r7 matrix runs 8.35M games without it. Fixed by PR #93 (Monstrous Emergence cannot be paid for by sacrificing its only creature). **Relaunched from scratch at 19:38 Berlin** (17:38 UTC) on `0d3c0d40879de23ef65bfd711151149f1ab6b40b` (default branch with #93) in a new checkout `~/mtg-ml-r8/code-b`, campaign `~/mtg-ml-r8/campaigns/r8b-20261009`, same flags, seed 8, GPUs 87/90, CPUs 32-63. The crashed run dir stays in `r8-20261009/r8-fs8-belief` for reference. Stop: `ssh h100-private2 'CODE=~/mtg-ml-r8/code-b bash ~/mtg-ml-r8/launch/r8_host.sh stop ~/mtg-ml-r8/campaigns/r8b-20261009'`. Arms A and C run on 889798b; #93 touches only the Monstrous Emergence rule, so the code differs for B only there. Launch record: [r8-launch.json](r8-launch.json). Results will be added to this entry; the arm ids below are the archive ids.
- **Code:** `889798bc54c7f940200c01d2e7ccee01a30a8159` (origin/HEAD `claude/lucid-euler-0tjz9h` at launch), deployed unchanged to `~/mtg-ml-r8/code` on each host (own venv, native built there). Launch tooling `tools/r8_campaign.py` and `tools/r8_host.sh` (added by the PR carrying this entry) runs from `~/mtg-ml-r8/launch/`, outside the deployed checkout. Campaign `~/mtg-ml-r8/campaigns/r8-20261009` on both hosts. Engine, features and flags are those of r7 (`r7-overnight-launch.json`), so r7's lr075 trajectory is the control at matched games.
- **Common settings (r7 lr075):** h256 / entity-attn 1 / GRU / shared value, seed 8, native engine, 2048 games per iteration, bf16 compiled PPO (capture 2, minibatch 2048, 4 epochs, target KL 0.03), pipeline 1, inference server with 64 resident slots, 50% self-play + 50% recent/uniform pool, 20% postboard games, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, checkpoint every 25 updates, snapshot every 122, `--eval-every-games 250000` with 1000 sampled + 1000 greedy Jund vs Blue games and L1 ladder (4 rungs, 200 games each, unchanged ratings 0/33.3/50.8/78.8; rung copies in the campaign dir). Per arm: learner GPU + inference GPU, trainer CPUs 2, server 1, workers 24, eval 4, same pinning scheme as r7.
- **Smoke before launch:** 4 to 6 updates per arm on separate smoke dirs: finite PPO stats, benchmark and L1 evaluation ran. Arm A resumed from a copy at iteration 4075 and continued the lr schedule. Resume of arm B and C was not exercised (A's resume path is the same code).

| arm id | host, GPUs (learner / server, bus) | CPUs | parent | change vs r7 lr075 | stops at |
|---|---|---|---|---|---|
| `r8-lr075-continue` | h100-private2, 0A / 18 | 0-31 | `20261008-r7-fs7-h256-lr075` `final.pt` (sha256 dc83fd7d8160c564047b435ba0e01ec42eb7496d9c29291b90b07b68bbc3c3cd, 8.35M games; the copy resumed from checkpoint 4075 = 8.35M, 4 updates were not saved by r7), pool and optimizer copied into a new run dir | none: same flags, `--resume`; `lr_anneal_origin` is in the checkpoint, so the 7.5e-5 → 7.5e-6 linear anneal over 20M games continues where it stopped (lr 4.68e-5) | 20M total games |
| `r8-fs8-belief` | h100-private2, 87 / 90 | 32-63 | none (fresh, seed 8) | `--features 8 --belief 1`, which requires `--match-rollouts 1 --variants train` (whole best-of-three matches with sampled registered 75s; the postboard fraction no longer applies because games 2/3 are played in the match). Not changeable: the belief head cannot train on isolated games | 20M total games |
| `r8-jund-blue-ft` | h100-private, 0A / 18 | 0-31 | r7 lr075 `policy.pt` (sha256 672aaa0c2764f853198c44a6fa7847469e01d9767952ea3fb8e9d84e068f636e), `--init` (fresh optimizer, fresh pool) | `--matchup jund_blue` only (feature set 7 swaps seats with p 0.5, so the network plays both Jund and Blue; 80% game 1, 20% postboard), lr 7.5e-5 → 7.5e-6 over 6M games | 6M total games |

- **Questions:** (A) is lr075 still improving past 8.35M games; (B) does feature set 8 with the belief head change strength or L1 at matched games, against lr075's recorded trajectory at 0.25M, 1.25M, 3.25M, 6.25M and 8.25M games (L1 and benchmark; mind that B plays matches with varied lists, so it also sees different training data); (C) does Jund-vs-Blue-only training from a full-matrix parent reach the r4-control level (64% sampled vs the Blue specialist, where lr075 has 46%/49%) without losing too much elsewhere. Goal for C: a new best on the standing benchmark and on `benchmark-blue@1`.
- **Expected speed:** r7 lr075 recorded about 0.76M games/hour (smoke 1.04M). Arm A needs about 12-15 h for the remaining 11.6M games, C about 6-8 h, B more than 20 h (to be stopped by hand).
- **How to stop** (all trainers terminate on SIGTERM and keep `latest.pt`): `ssh h100-private2 'bash ~/mtg-ml-r8/launch/r8_host.sh stop ~/mtg-ml-r8/campaigns/r8-20261009'` and the same on `h100-private` for C. Resume only after review: `bash ~/mtg-ml-r8/launch/r8_host.sh start CAMPAIGN ARM --resume` (new tmux session `mtg-r8-ARM`). No automatic restart; training logs rotate at 2 MB with three backups.
- **Verdict:** pending (B: failed, no result; needs the engine fix first).

## 20261009-specialist-benchmark · checkpoints vs the Jund and Blue specialists (evaluation only)

- **Question**: how do the current r7 arms and the old best model score against the new specialists `benchmark-jund@1` (#66) and `benchmark-blue@1` (#65)?
- **Checkpoints**: r7 lr075 and lr150 at the shared frozen `evaluation-checkpoints/iter_03660` (~7.2M games, fs7, h256, fair contract) from h100-private2, and again at the final policies after the run was stopped (lr075 `v04079`, 8.35M games; lr150 `v04412`, 9.04M games; archived as `20261008-r7-fs7-h256-{lr075,lr150}/policy.pt`). 20261007-r4-control `policy.pt` (fs3) from h100-private. fs3 inputs disclose the opponent archetype (contract `legacy_archetype_disclosed`), so its numbers are **diagnostic**; r4-control trained on Jund vs Blue only, where that label carries little information. SHA-256 values are in `ledger.jsonl`.
- **Code / protocol**: `tools/benchmark_checkpoint.py` @ 86efb15, native engine. 100 four-game blocks per cell and mode (both seats × both starts on one deal, stream `benchmark-v1/dev`), so n = 400 per cell and mode with Wilson 95% CI about ±0.05. Legacy-bot cells are included as a reference. They are **not** the ledger benchmark, which always seats Jund first. Run on h100-private (60 CPU workers, ~2 min per checkpoint for 4,800 games), 0 errors. Rows: `h100-private:~/mtg-ml-bench-specialists/runs/`.

Score of the checkpoint, sampled / greedy:

| checkpoint | Jund vs Blue spec. | Blue vs Jund spec. | Jund vs Jund spec. | Blue vs Blue spec. | Jund vs legacy Blue | Blue vs legacy Jund |
|---|---|---|---|---|---|---|
| r4-control (fs3, diagnostic) | **64.0 / 65.5** | **85.3 / 87.0** | 32.8 / 36.2 | 48.7 / 48.7 | 74.3 / 74.3 | 93.0 / 93.0 |
| r7 lr075 iter 3660 | 40.7 / 49.7 | 70.8 / 74.3 | 55.5 / 66.2 | 59.0 / 66.2 | 53.2 / 67.5 | 83.3 / 90.5 |
| r7 lr150 iter 3660 | 34.5 / 44.5 | 60.5 / 65.0 | 49.0 / 57.0 | 50.5 / 61.0 | 45.5 / 56.2 | 74.5 / 81.0 |
| r7 lr075 final (8.35M) | 46.3 / 49.0 | 68.8 / 75.7 | 57.8 / 68.5 | 57.8 / 67.0 | 55.5 / 70.8 | 84.3 / 88.5 |
| r7 lr150 final (9.04M) | 38.0 / 47.7 | 62.5 / 70.8 | 51.5 / 61.5 | 52.5 / 63.5 | 50.2 / 54.2 | 79.0 / 83.3 |

**Findings**
- Both specialists are clearly stronger than the legacy bots. Each checkpoint scores 8–18 points less against the specialist than against the legacy bot of the same deck.
- The standing matchup (learner Jund vs Blue) is hard. Neither r7 arm reaches 50%; r4-control gets 64%: it spent all 17.6M training games on this matchup, against roughly 0.4M for r7 (jund_blue is 2 of 36 matchup weights).
- r4-control loses the Jund mirror to the Jund specialist (33–36%). It was trained on Jund vs Blue only. The r7 arms, trained on six decks including mirrors, win both mirrors.
- lr075 beats lr150 in every cell, by 5–11 points at iter 3660 and by 1–17 points at the final policies. This matches the head-to-head (lr075 54.8% at matched games, see `20261008-r7-fs7-h256`).
- From iter 3660 to the final policies (1.1–1.8M more games) r7 moved by about +1 point (lr075) and +3 points (lr150) on average, inside the ±5 point intervals. Neither final arm reaches 50% as Jund against the Blue specialist.
- r7's greedy play is 4–14 points above its sampled play (entropy ~0.35). r4-control's gap is at most 3.4.
- Sampled and greedy scores for r4-control coincide in some cells. Only 313 of 2,400 games were identical, so this is chance, not a mode bug.

## 20261008-r7-fs7-h256 · fresh full-matrix LR comparison

- **Status:** stopped by request at 09:32 Berlin October 9 after 10.7 h, before the
  20M-game target: lr075 at 8,353,792 games (iteration 4079), lr150 at 9,035,776
  (iteration 4412). Both trainers exited cleanly on SIGTERM (`exit_code` 143).
- **Parent:** none. Both arms initialize with seed 8 and parameter hash
  `c9b2d973e06e1c1050566c274ecd608dc8ddaafd41b6a44fd774de907baa4847`.
- **Code:** `4a32795fd4ad25e328fda69c210c08be0d40982d`, descendant of pinned
  PR #53 `b80b6306b18354f87bb60ae742c70ad0369ba5b7` (includes merged #63).
  No other PR is a launch dependency. Exact commands, UUIDs, environment and
  smoke measurements: [launch record](r7-overnight-launch.json).
- **Arms:** `20261008-r7-fs7-h256-lr150` uses 1.5e-4 → 1.5e-5;
  `20261008-r7-fs7-h256-lr075` uses 7.5e-5 → 7.5e-6. Every learning setting
  otherwise matches. Feature 7; 21 pairings weighted into a uniform 36-cell
  matrix with mirrors; 80/20 main/postboard using fixed tactical assumptions.
- **Smoke:** both completed five warm-up plus twenty measured updates (51,200
  games), finite PPO statistics, final sampled/greedy/L1 evaluation, identical
  initial hashes, then resumed iteration 25 to 26 with the correct LR. Measured
  1.248M / 1.040M games/hour on this host. These early-run rates do not establish
  strength or predict performance after a large historical pool develops.
- **Results** (final policies; benchmark on 2,000 games, L1 from the last periodic
  evaluation):

  | arm | games | bench sampled | bench greedy | L1 Elo |
  |---|---|---|---|---|
  | lr075 | 8.35M | **56.0%** (53.9–58.2) | **65.2%** (63.1–67.3) | **5 ± 12** at 8.25M |
  | lr150 | 9.04M | 46.9% (44.7–49.1) | 57.4% (55.2–59.5) | -24 ± 13 at 9.00M |

  Head to head at matched games (shared snapshot `iter_04026`, 8.25M games, all 21
  pairings, 400 games each, both seats): lr075 scores **54.8%** (paired seed-block
  bootstrap 95% CI 53.9–55.7) and wins every pairing (lr150 41.7–48.2%).
  L1 trajectory, lr075 / lr150: -134 / -150 at 0.25M, -60 / -77 at 1.25M,
  -45 / -70 at 3.25M, -13 / -82 at 6.25M, 5 / -40 at 8.25M. Against the new
  specialists see `20261009-specialist-benchmark` (#72): neither arm reaches 50% as
  Jund against `benchmark-blue@1` (lr075 46.3 / 49.0%).
- **Comparison:** lr075 is level with r6-h128 (fs6, h128, 15 pairings) at equal
  games (r6-h128 at 8.0M: 54.8% / 63.0%, L1 -3), now on the fair feature set 7 and
  with mirrors. Far below r4-control (L1 163), which trained on Jund vs Blue only.
- **Verdict:** lr075 (7.5e-5 → 7.5e-6) is the better learning rate for h256 on the
  full matrix: ahead on benchmark (+9 / +8 pts), L1 (+29) and head to head, at fewer
  games. Training was healthy (entropy ~0.35, KL 0.017 vs 0.024, explained variance
  0.85, ~0.76–0.79M games/hour). One seed; lr075 was still climbing when stopped.
  Next parent candidate for fs7 full-matrix work: `20261008-r7-fs7-h256-lr075`.
- **Operations:** [protocol and commands](r7-overnight.md). Dashboard runs on the
  server, loopback only behind Tailscale Serve8443; ComfyUI443 retained. Archived
  on h100-private as `20261008-r7-fs7-h256-lr075` and `-lr150` (final.pt,
  policy.pt, 33 / 36 snapshots, metrics, logs, launch record, evals.txt). Run dirs
  stay on h100-private2.
- **Incident (23:33 Berlin):** lr075 died at iteration 358 with a CUDA OOM in
  PPO epoch packing (43 GiB reserved but free: fragmentation). Both arms now run
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; source unchanged.
  lr075 resumed from 350, lr150 was stopped after checkpoint 425 and resumed.
  Details in [r7-overnight.md](r7-overnight.md#incidents).

## 20261008-h256-attention-scaling · implementation and completed throughput screens

- **Parent:** r6-h256 `pool/iter_02440.pt`, 4,997,120 games, source code
  `30d9ef640`, SHA-256 `0dd482bcd68e846cc7221a894b2fef711303cc183b1f4230666a5098930fbea7`.
  Frozen under `/home/taiga-support/mtg-ml-256-opt/frozen/iter_02440-v2` with
  one identity attention layer and the historical opponents. Fresh optimizer;
  continuation preserves the parent's schedule position. Candidate engine/code
  includes subsequent fixes, so compare all arms on that same candidate ref.
- **Code:** `fmssn/256-width-training-optimization`, draft PR #53; implementation
  timing screens at `05103c4`, final CUDA checks and proposed campaign at
  `aed2ee8`. Exact campaign commands are generated by
  `tools/training_campaign.py`; protocol and flags are in
  [training-256-attention.md](../training-256-attention.md).
- **Machine:** `h100-private`, H100 80GB. Existing r6 jobs preserved. Preliminary
  screens use idle project GPUs, one low-priority CPU thread on core 63; host
  contention prevents interpreting these as dedicated-resource training rates.
- **Validation:** fast CPU suite 1,228 passed; whole-trajectory unequal/empty
  Gloo shards match single-process updates. CUDA mixed attention graphs and
  recurrent resets pass. A BF16 compiled parity check exposed a difference in
  first-step initialization: distributed learning compiled immediately while
  ordinary PPO initializes Adam with an uncompiled step. That path was corrected
  before continuation experiments. All ten final CUDA checks passed, including
  two-GPU NCCL eager/graph updates with unequal and empty shards and Adam-state
  parity and compiled BF16. BF16 moment comparisons use bounded 1% norm/max
  error for shard-specific GEMM rounding (observed relative norm ≤0.59%);
  parameter/loss checks and strict FP32 moment comparisons remain unchanged.
  The final focused CPU checks passed 22 tests. Failed checks remain in the
  deployment logs. Dedicated four-rank CUDA checks subsequently passed all
  three eager FP32, graph FP32 and compiled BF16 cases in 87.77 seconds.
- **Preliminary timing:** one shared nice-19 CPU core, five warm-up / twenty
  timed batches. Eager attention: 447.53 ms for 608 decisions across 26 policies;
  stacked: 6.20–64.69 ms under variable host scheduling, with consistent
  2.436–2.438 ms CUDA-event graph replays and 4.80 GiB peak allocation. Saved
  six-deck rollout data (128 games, 21,475 decisions) gave 858 optimizer steps
  in 44.76 s FP32 / 42.27 s BF16, peak 5.0 / 4.0 GiB. Timed graph-capture cost
  and CPU contention make that small learner difference inconclusive. Actual
  traces confirmed FP32 memory-efficient SDPA, BF16 cuDNN flash SDPA, and
  substantial recurrent-kernel work. Exact flags and limits are in the protocol.
- **Additional failure:** the combined BF16 compiled two-GPU check hit a
  ninety-second NCCL watchdog while a rank was still compiling/capturing.
  Host readiness is now coordinated through a separate bounded Gloo group
  before GPU reductions; the final ten-check rerun passed. The initial FP32
  moment tolerance also rejected expected BF16 shard rounding, quantified above
  before applying explicit BF16 bounds. The large full-update trace was stopped
  after timing because host aggregation was excessive; plain timing runs and
  one-graph kernel traces were collected instead. All raw logs are retained.
- **Dedicated computational campaign:** started 2026-10-08 15:47 UTC after the
  user released the project GPUs. Exact deployed ref `e8662a5`; manifest at
  `/home/taiga-support/mtg-ml-256-opt/campaigns/screens-20261008-dedicated/campaign.json`.
  Baseline uses bus 18 and CPU 0–31 (26 rollout workers); four-rank FP32/BF16
  correctness runs concurrently on buses 0A/87/90/C7 and CPU 32–35. Persistent
  `start_computational_checks.sh` continues the remaining twelve-arm screens
  after validation and baseline succeed. Five warm-up and twenty timed
  iterations per arm; two/four learner expansion retains the bottleneck gates.
  Logs and outcomes stay in that campaign directory. ComfyUI and bus BE are
  excluded. These timing screens disable evaluation; strength comparisons and
  three paired two-hour finalists have not started.
  Four-rank parity passed all three cases (eager FP32, graph FP32, compiled
  BF16), including unequal and empty shards, in 87.77 seconds. The campaign
  finished at 16:43:24 UTC: ten completed screens, no failures, 512,000 games,
  1.397 charged GPU-hours across training screens (excludes separate validation).
  Two/four-learner throughput screens were skipped by the planned bottleneck
  gate because BF16 collection exceeded learner time. All five project GPUs
  were idle afterward.
- **Measured throughput:** legacy 138,123 games/hour / 6,212 real learner
  decisions/s; stacked 635,422 / 28,344 (**4.60×**); separate 946,019 / 41,981
  (**6.85×**); resident64 770,448 / 34,372; resident128 777,230 / 34,237;
  BF16 1,048,986 / 46,782 (**7.59×**); batch8192-lr150 1,554,951 / 71,328
  (**11.26×**); batch8192-lr300 1,162,261 / 56,992; epochs2 1,500,214 / 65,706;
  lag2 1,397,025 / 61,379. Ratios use whole-iteration timed throughput against
  legacy; publication and learning overlap collection. Full GPU-hours, padding,
  memory and stage timings are in the protocol and outcome records.
- **Limits:** one seed, 20 timed iterations, no strength evaluation. KL stops
  and new graph captures produced unequal learner work (median steps 617.5
  legacy, 386 stacked, 159.5 separate, 609 resident64, 162.5 BF16, 41 larger
  batch). Batch8192-lr150 hit KL stopping in all twenty timed iterations.
  Consequently the hardware-only rate comparisons also need confirmation with
  longer paired runs. The 21 historical opponents plus learner fit in one
  residency wave, so eviction beyond 64/128 policies remains unmeasured.
- **Archives:** all ten completed screens are archived on `h100-private` as
  `20261008-h256-screen-ARM-seed0`, with final/resume policies, metrics, exact
  campaign manifest and outcomes. `evals.txt` records that these are timing-only
  screens. The standard archive helper emitted an unmatched snapshot-glob warning
  because 25 iterations produced no new 122-iteration pool snapshot; final and
  policy files were saved successfully, and hashes were verified.
- **Training strength:** not measured; sampled/greedy benchmark and L1 Elo are
  unavailable pending the paired continuation experiments.
- **Verdict: throughput target met in short screens; strength inconclusive.**
  Stacked/separate inference exceeds the 3–5× target without changing PPO
  settings. Select defaults only after matched-game/elapsed-time policy-strength
  comparisons across three paired seeds. Larger batch, reduced epochs and lag
  remain experimental; the standard 2,000-game final evaluations remain pending.

## Open (handoff 2026-10-08)

- Round 6 (three fresh six-deck arms on feature set 6) stopped 2026-10-08 17:41 CEST on request, all archived. Numbers are the last in-training evaluations (1,000 games, ladder L1 200/rung); the 2,000-game final evals were not run.
- Next: continue `r6-h128` from its 12M checkpoint with a longer lr schedule (it was still climbing at the 3e-5 floor), and find out why the attention arm is ~6x slower per iteration before giving it more games (it was the best per game). Width: h256 needs more games to say anything.
- Ladder: r6-h128 reached 76 at 12M, at the top L1 rung (79): start L2 before the next round.

## Open (handoff 2026-10-07)

- Rounds 4 and 5 stopped 2026-10-07 17:25 UTC on request (Mix complete, the others early); all archived. The numbers below are the last in-training evaluations (1,000 games, ladder L1 200/rung); the standard 2,000-game final evals and head to heads were not run yet: `final_evals.sh` with `H2H="20261007-r4-mix 20261007-r4-control"`, `ladder_final.py`, from `~/mtg-ml-r4`.
- Next: a mix weighted toward the main matchup (`--matchup jund_blue:2,jund_madness,blue_madness`) from `r4-control`; the width question is open (h256 needs > 6M games to say anything).
- Still open: settle postboard with a 5,000-game head to head; ladder L2 (rungs: L1 + `r1-lranneal` + `r3-control`).
- Scripts: `tools/experiments/`.

## Summary

| id | change (vs parent) | games | bench sampled | bench greedy | L1 Elo | verdict |
|---|---|---|---|---|---|---|
| 20261007-fs4-ft | r4-control final + feature set 4 (combat relations, incoming damage, X and colour previews), same flags | 17.61M + 5.0M | **78.8%** | **80.3%** | **213 ± 14**⁷ | **adopt**: set 4 on by default, new best parent |
| 20261007-fs3-ctl | r4-control final, identical resume on feature set 3 (control) | 17.61M + 5.0M | 77.4% | 78.9% | 190 ± 13⁷ | control |
| 20261008-r7-fs7-h256-lr075 | fresh h256 attention + GRU, feature set 7, full 36-cell matrix with mirrors, lr 7.5e-5 → 7.5e-6 | 8.35M (stopped) | 56.0% | 65.2% | 5 ± 12 at 8.25M | best fs7 full-matrix model; beats lr150 54.8% head to head; still climbing |
| 20261008-r7-fs7-h256-lr150 | the same at lr 1.5e-4 → 1.5e-5 | 9.04M (stopped) | 46.9% | 57.4% | -24 ± 13 at 9.0M | reject: behind lr075 everywhere |
| 20261008-r6-h128 | fresh h128 entity net, feature set 6, six decks (15 pairings, jund_blue ×2), real 15-card sideboards | 12.9M (stopped) | **66.8%**⁶ at 12M | 67.2%⁶ at 12M | **76 ± 12**⁶ at 12M | works: best six-deck model; still climbing |
| 20261008-r6-h128-attn | the same + 1 entity self-attention layer | 2.4M (stopped) | 50.8%⁶ at 2M | 65.9%⁶ at 2M | -26 ± 13⁶ at 2M | promising per game (+9 pts, +52 Elo vs h128 at 2M), ~6x slower: fix speed first |
| 20261008-r6-h256 | the same at h256, lr 1.5e-4 → 1.5e-5 | 7.3M (stopped) | 50.0%⁶ at 6M | 59.3%⁶ at 6M | -4 ± 12⁶ at 6M | inconclusive: behind h128 at equal games (54.5% / 8 at 6M) |
| 20261007-red-madness-1m | new deck: Red Madness learner vs frozen r1-control Jund, warm-started from it | 1M | (vs Jund bot) 89.9% | 92.6% | | new-deck baseline⁴ |
| 20261007-r5-mix-h256 | fresh h256 entity net on the three-deck mix, lr 1.5e-4 → 1.5e-5 | 2.6M (stopped) | 46.3%⁵ | 60.4%⁵ | -32 ± 13⁵ | inconclusive: level with h128 at equal games, slightly behind |
| 20261007-r5-mix-h256-lr3e-4 | the same at the h128 lr (3e-4 → 3e-5) | 2.6M | 38.1% at 2.5M | | -69 at 2.5M | abort: steps too large |
| 20261007-r5-mix-h128 | fresh h128 entity net on the three-deck mix (width control) | 5.8M (stopped) | 56.4%⁵ | 65.5%⁵ | 40 ± 12⁵ | control (49.5% / -14 at 2.5M) |
| 20261007-r4-mix | r3-postboard + Red Madness: one network on jund_blue, jund_madness, blue_madness (feature set 3) | 10.4M + 10M | 73.4%⁵ | 71.7%⁵ | 155 ± 13⁵ | works, costs the main matchup ~5 pts vs control |
| 20261007-r4-control | r3-postboard, jund_blue only, same schedule and feature set 3 | 10.4M + 7.2M (stopped) | **78.2%**⁵ | 77.6%⁵ | **163 ± 13**⁵ (173 at 17.0M) | **best parent** |
| 20261007-r3-attn | R3 + 1 entity self-attention layer (identity init) | 9.4M + 1M | 73.3% | 75.0% | 152 ± 13 | reject (no gain, ~10× slower) |
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

Sampled/greedy of finished runs: 2,000 games on the fixed engine, final checkpoint. ¹ last in-training evaluation (1,000 games, 9.25M). ² sampled training games of the last iterations against the frozen main policy; starting levels 45% (Jund) and 55% (Blue). ³ evaluated without auto mana (it trained with it), so the model paid mana itself. L1 Elo: final checkpoint (policy file), 200 paired games per rung, on the code the run was trained with (round 1: feature set 1; rounds 2-3: feature set 2, under which the L1 rungs, trained on set 1, see untrained feature rows: about 1 benchmark point weaker, so round 2-3 Elos may read a few points high against round 1). Head-to-head results are in the round sections. ⁴ benchmark here = learner Red vs the scripted Jund bot (1,000 games); not comparable to the Jund-vs-blue rows. ⁵ last in-training evaluation (1,000 games); the 2,000-game final evals were not run (runs stopped 2026-10-07). ⁶ last in-training evaluation (1,000 games); the 2,000-game final evals were not run (runs stopped 2026-10-08). ⁷ measured on the set-4 code (759358e), where r4-control's final policy reads 199.5 ± 13.6 (its in-training 163-173 was on the PR #28 code): compare the fs4 rows with each other, not with older rows.

---

## 20261007-fs4 · feature set 4 fine-tune vs a feature set 3 control

- **Question**: do the rules-level observation gaps the r4-control game review found (feature set 4, PR fmssn/mtg-ml#33: combat relations between attackers and blockers, incoming damage, X and colour previews) make the model stronger once it trains on them.
- **Parent**: `20261007-r4-control` final (`final.pt`, 17.61M games, feature set 3) with its pool and optimizer state, resumed in both arms; pool snapshots without a feature stamp were stamped `features=3` in the run copies. The new set-4 rows start untrained.
- **Code**: `claude/feature-set-4` @ 759358e (PR #33), box copy `~/mtg-ml-fs4`. No rules change (golden digests untouched); sets 1-3 unchanged (digest-tested).
- **Flags** (`~/mtg-ml-fs4/launch_fs4.sh`, both arms, only `--features` differs): `--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:0 --workers 28 --games-per-iter 2048 --iterations 10000000 --total-games 22612800 --checkpoint-every 25 --snapshot-every 122 --seed 4 --server-policy-slots 256 --features {4|3} --postboard-frac 0.2 --ppo-lr 1e-4 --ppo-lr-final 3e-5 --lr-anneal-games 8000000 --eval-every 0 --eval-every-games 250000 --bench-games 1000 --bench-greedy-games 1000 --bench-bo3-matches 0 --eval-games 40 --eval-bo3-matches 0 --ladder <L1> --ladder-ratings <L1>/ladder.json --ladder-games 200`. fs4-ft on cores 0-31, fs3-ctl on 32-63, `OMP_NUM_THREADS=8`. Started 2026-10-07 18:49 UTC, both finished 22:49 UTC at 22.61M games (iteration 11042, 5.0M games each).
- **Result** (final policy, all on the set-4 code; benchmark 2,000 games, ladder L1 200 paired games per rung, sampled):

| | sampled | greedy | L1 Elo | in-training L1 Elo, mean of last 10 (20.25-22.5M) | audit blind spots (greedy self-play, 200 games) |
|---|---|---|---|---|---|
| fs4-ft | **78.8%** (77.0–80.5) | **80.3%** (78.6–82.0) | **212.5 ± 13.8** | 198.5 | **0.0 / 1k** (0 of 56,675; declare_blocker 0 of 947) |
| fs3-ctl | 77.4% (75.6–79.2) | 78.9% (77.1–80.6) | 190.1 ± 13.4 | 179.9 | 2.16 / 1k (125; declare_blocker 86 of 991) |
| r4-control (parent) | 76.5% (74.6–78.4) | 75.6% (73.7–77.5) | 199.5 ± 13.6 | | 2.77 / 1k (152; declare_blocker 107 of 933) |

Head to head, 2,000 paired games: fs4-ft vs fs3-ctl **52.9% (50.8–55.1)**; fs4-ft vs r4-control 51.5% (49.4–53.7).

- **Findings**
  - *Set 4 vs control* (same parent, flags, seed and games): the head to head is the one interval clear of 50% (52.9%). Benchmark +1.4 sampled and +1.4 greedy (each inside noise), final Elo +22 (inside noise alone), but the in-training Elo sits ~19 above the control across the whole last 2.25M games, so the direction is consistent everywhere.
  - *Audit*: under set 4 no two options of a decision look identical where the outcome differs (blind spots 2.16 → 0 per 1k). The control's blind spots are mostly blocks (which attacker a blocker takes: Eldrazi Spawn, Tolarian Terror, Cryptic Serpent); set 4's `e:blocking:*` / `pv:attacker_*` tokens remove them. 11 identical-option groups remain (priority, none blind).
  - *Control vs parent*: 5M more games on set 3 bought little (+0.9 sampled, +3.3 greedy, Elo 190 vs 200), so most of fs4-ft's edge over r4-control comes from the features, not from the extra games.
  - The 50 greedy review games of the fs4-ft final policy (seeds 0-49, jund_blue self-play; Jund won 23) are in `~/mtg-ml-fs4/review-games/` on the box.
- **Verdict**: **adopt**. Feature set 4 becomes the default and `20261007-fs4-ft` the next parent. The gain is small (head to head ~53%), consistent with features that mainly fix block selection. A 5,000-game head to head would narrow it further if needed.
- **Archive**: `20261007-fs4-ft`, `20261007-fs3-ctl` (evals.txt has every number above; audit json in `~/mtg-ml-fs4/audit/`).

## 20261008-r6 · six decks from scratch on feature set 6 (h128, h128 + attention, h256)

- **Question**: can one network learn all six decks (Jund Wildfire, Mono Blue Terror, Red Madness, Grixis Affinity, Elves, Tron) from scratch with the set-6 representation, and do width or entity attention help.
- **Code**: 30d9ef6 (PR #45 sideboard overhaul + Affinity/Elves/Tron, plus #48 damage-assignment cap and chunked inference requests), feature set 6, native engine. Launch scripts in `~/mtg-ml-r6/` on h100-private (start_arm.sh, stop_r6.sh, status_r6.sh).
- **Flags (shared)**: `--trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:0 --seed 6 --server-policy-slots 256 --features 6 --postboard-frac 0.2 --matchup jund_blue:2,affinity_elves,affinity_tron,blue_affinity,blue_elves,blue_madness,blue_tron,elves_tron,jund_affinity,jund_elves,jund_madness,jund_tron,madness_affinity,madness_elves,madness_tron --lr-anneal-games 10000000 --workers 19 --games-per-iter 2048 --checkpoint-every 25 --snapshot-every 122 --eval-every 0 --eval-every-games 1000000 --bench-games 1000 --bench-greedy-games 1000 --ladder-games 200 --bench-bo3-matches 0 --eval-bo3-matches 0`. Arms: `r6-h128` `--hidden 128 --ppo-lr 3e-4 --ppo-lr-final 3e-5`; `r6-h128-attn` the same + `--entity-attn 1`; `r6-h256` `--hidden 256 --ppo-lr 1.5e-4 --ppo-lr-final 1.5e-5`. GPUs 87 / 90 / C7 by bus id. Each arm restarted once (08:28 CEST) after the damage-assignment crash fixed in #48.
- **Ladder bug (fixed during the run)**: the r6 copy of `ladder.json` held a 2-game smoke re-rating (0 / -59 / -59 / -119) instead of L1's fixed ratings, so every logged `ladder/elo` up to 11M was ~100 too low. Restored from `~/mtg-ml-checkpoints/ladder/L1/` at 12:50 CEST (rung files byte-identical). The Elo below is recomputed from the per-rung scores against L1; the 12M value was logged with the right ratings and matches. Rule: a run's `--ladder-ratings` is always a copy of L1, never a fresh rating.

| games | h128 sampled / greedy | h128 Elo | attn sampled / greedy | attn Elo | h256 sampled / greedy | h256 Elo |
|---|---|---|---|---|---|---|
| 1M | 36.7 / 53.7% | -56 | 40.6 / 59.8% | -65 | 37.2 / 52.4% | -75 |
| 2M | 42.0 / 56.4% | -78 | **50.8 / 65.9%** | **-26** | 37.2 / 56.3% | -82 |
| 3M | 41.8 / 57.3% | -46 | | | 38.7 / 54.3% | -63 |
| 4M | 52.6 / 64.0% | -19 | | | 46.8 / 57.5% | -45 |
| 5M | 53.1 / 63.5% | 1 | | | 47.5 / 63.2% | -28 |
| 6M | 54.5 / 63.5% | 8 | | | 50.0 / 59.3% | -4 |
| 8M | 54.8 / 63.0% | -3 | | | | |
| 10M | 63.8 / 68.1% | 63 | | | | |
| 12M | **66.8 / 67.2%** | **76** | | | | |

± 12-13 Elo, benchmark ± ~3 points. Rows left out are within noise of their neighbours.

- **Per deck at 12M** (h128, mean over its pairings, sampled, vs the scripted bots): Jund 61%, Affinity 65%, Blue 76%, Tron 78%, Red 83%, Elves 84%. The newer decks' bots are probably weaker opponents, so these measure progress, not deck strength.
- **Findings**:
  - One network learns all six decks. At 12M games, of which only ~1.5M were Jund vs Blue, h128 plays the main matchup at L1 76: level with the old two-deck self-play run after 8M games of that matchup alone, and still 90-140 Elo below the best specialists (r4-control 163, the set-4 fine-tune ~212).
  - h128 never plateaued: 8M looked flat (noise), then +65 Elo by 10M and +13 more by 12M, with the lr already at its 3e-5 floor from 10M.
  - Attention was clearly better per game at 2M (+9 benchmark points and +52 Elo against h128 at 2M; round 3's warm-started attention arm showed no gain), but ran at 20 → 40 s/iter against h128's 4.5 → 6.3. Its arm also had fewer collector cores (21-31 against 2-20), so part of the gap may be the CPU layout rather than the model.
  - h256 learned slower per game than h128 throughout, as in round 5, at twice the cost per iteration. Not yet a verdict on capacity.
  - The h128 arm's s/iter rose ~40% over the day (the attention arm's doubled): longer games as play improves, and evaluation sharing the worker cores.
- **Verdicts**: `r6-h128` **best six-deck parent**; `r6-h128-attn` promising, fix its speed and rerun; `r6-h256` inconclusive.
- **Archive**: `~/mtg-ml-checkpoints/20261008-r6-h128`, `20261008-r6-h128-attn`, `20261008-r6-h256`.

## 20261007-r5 · width test on the three-deck mix

- **Question**: r4-mix lost ~5 points on Jund vs Blue against its control. Is that model size? Same mix from scratch at h128 and h256; if h256 is clearly better on Jund vs Blue (and on the new pairings), scale width before adding decks.
- **Parent**: none (fresh). A trained h128 cannot be widened, so both arms start from zero.
- **Code**: PR fmssn/mtg-ml#28 (matchup mix, feature set 3, stack fix), box copy `~/mtg-ml-r4`.
- **Flags** (both arms): `--hidden {128|256} --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 --workers 20 --games-per-iter 2048 --total-games 10000000 --seed 5 --features 3 --postboard-frac 0.2 --matchup jund_blue,jund_madness,blue_madness --ppo-lr 3e-4 --ppo-lr-final 3e-5 --lr-anneal-games 10000000 --eval-every-games 500000 --bench-games 1000 --bench-greedy-games 1000 --ladder <L1>`; cores 0-63 shared with r4 (speeds not comparable to other rounds).
- **First h256 attempt (aborted)**: at the h128 learning rate h256 fell behind and stalled from 1.5M games (Jund vs Blue bot 38.1% vs h128's 49.5% at 2.5M, Elo -69 vs -14, 3-13 points behind on every new pairing). Its updates were ~70% larger (approx KL 0.026 vs 0.015, clip fraction 0.15 vs 0.11). Archived as `20261007-r5-mix-h256-lr3e-4`; restarted at half the lr (`--ppo-lr 1.5e-4 --ppo-lr-final 1.5e-5`). Lesson: scale the lr down with width.
- **Result** (stopped early, last in-training evaluations, 1,000 games): at equal games h256 is level with h128 or slightly behind (2.5M: Jund vs Blue bot 46.3% vs 49.5%, Elo -32 vs -14; new pairings 1-4 points behind). h128 kept climbing to 56.4% / Elo 40 at 5.5M (Jund vs Red bot 59.5%, Red vs Jund bot 93.1%, Blue vs Red bot 70.8%, Red vs Blue bot 85.4%).
- **Verdict**: inconclusive. Width did not pay off in the first 2.5M games; wider nets often pull ahead later, so the question needs h256 past ~6M games (or a fixed-compute comparison).

## 20261007-r4 · one network for three decks

- **Question**: can one network learn Jund, Blue and Red (all three pairings) without losing Jund-vs-Blue strength, and how strong is its Red.
- **Parent**: `20261007-r3-postboard` latest (10.4M games) and its pool, resumed. Entity attention dropped (r3-attn: no gain).
- **Code**: PR fmssn/mtg-ml#28 (`claude/multi-matchup` @ main 1cf8fff + matchup mix, `blue_madness`, feature set 3 `opp:deck:`), box copy `~/mtg-ml-r4`.
- **Flags** (both arms): `--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 --workers 26 --games-per-iter 2048 --total-games 20400000 --checkpoint-every 25 --snapshot-every 122 --seed 4 --server-policy-slots 256 --features 3 --postboard-frac 0.2 --ppo-lr 1e-4 --ppo-lr-final 3e-5 --lr-anneal-games 8000000 --eval-every-games 500000 --bench-games 1000 --bench-greedy-games 1000 --ladder <L1>`; mix adds `--matchup jund_blue,jund_madness,blue_madness`.
- **Result** (last in-training evaluations, 1,000 games):

| | Jund vs Blue bot | greedy | L1 Elo | Jund vs Red bot | Red vs Jund bot | Blue vs Red bot | Red vs Blue bot |
|---|---|---|---|---|---|---|---|
| mix at 10.5M | 73.8% | 74.0% | 135 | 41.4% | 33.9% | 62.0% | 34.8% |
| mix at 11.0M (low) | 66.1% | 70.2% | 92 | 50.3% | 78.0% | 66.5% | 69.6% |
| mix at 20.0M | 73.4% | 71.7% | 155 | 64.6% | 92.7% | 79.0% | 87.1% |
| control at 17.5M | 78.2% | 77.6% | 163 (173 at 17.0M) | | | | |

- **Findings**
  - *Red is learned fast*: 34% → 93% vs the Jund bot within 10M games (the scripted Jund bot is weak against aggro, so Elo is the better yardstick for Red).
  - *The mix costs the main matchup*: Jund vs Blue fell 7.7 points in the first 500k games (the new `opp:deck:` input starts untrained, so Jund's lessons against Red bled into its play against Blue), then recovered to the parent's level (73.4%, Elo 155) but stays ~5 points and ~10-20 Elo behind the control. At equal Jund-vs-Blue practice the mix is still behind, so it is interference, not only fewer games. Value loss 0.028 vs 0.020.
  - *Control improves further*: continuing at lr 1e-4 → 3e-5 with feature set 3 gave the best Jund-vs-Blue numbers yet (78.2%, Elo 163-173).
- **Verdict**: the mix works and is the basis for multi-deck play; the control is the new best parent for Jund vs Blue. Next: weight the mix toward the main matchup and settle the width question (r5).

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
| attn | 73.3% (71.3–75.1) | 75.0% (73.0–76.8) | 152 ± 13 (171 at +0.25M) | vs control: 49.2% (47.1–51.4) |

**Findings**
- *Continued low lr*: small further gains over the parent (sampled +0.9, greedy +1.6, head to head 51.7%, just inside noise). The run is close to its plateau at this lr.
- *Postboard 0.2*: best on every number (+1.9 sampled, Elo +9, head to head 52.0%) but each within noise. Inconclusive, leaning positive; cheap to keep. Decide with a longer run or 5,000-game head to head.
- *Bot games as Jund*: no gain over the control on anything. Reject (with round 2: same result twice).
- *Entity attention (1 layer)*: level with the control (head to head 49.2%, Elo 152 vs 146, benchmark within 0.3 points); the early 171 Elo did not hold. At ~20 s per iteration (no stacked server path) vs ~1.5 s, reject for now.

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
