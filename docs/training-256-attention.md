# Width-256 attention: implementation and experiment protocol

The optimized path is implemented. Dedicated H100 computational screens measured
4.60× baseline throughput with stacked attention and 6.85× with a dedicated
inference GPU, preserving PPO settings. Policy strength per hour still needs the
paired training comparison. Existing r6 training processes were preserved.

## Implemented behavior

- Entity attention can use `PolicyStack`: policy-specific layer normalization,
  grouped QKV/output/FFN projections, and independent SDPA problems per decision.
  Entity widths enter CUDA graph buckets. Empty decisions retain a valid dummy
  attention key, and padding never enters entity sums or option pointers.
- Compatible rows stay stacked when a batch also contains incompatible policies.
  Recurrent policies of different widths have separate hidden-state tables.
  Sparse resident slot IDs skip absent policies in the grouped matrix kernel.
- `--server-devices cuda:1` separates inference from `--device cuda:0`.
  `--learner-devices cuda:0,cuda:1` enables persistent synchronous replicas;
  a dedicated server then uses `--server-device cuda:2`. CPU lists are explicit
  (`--learner-cpus '0;1'`, `--server-cpus 4`) and reserved before workers.
- `--server-resident-limit 64` includes the learner. Already-selected opponents
  are scheduled in waves; every game and the complete rollout finish before the
  PPO update. Eviction drains GPU reads, clears finished-game hidden states,
  invalidates client IDs, and reuses stable weight buffers. The default remains
  unbounded for existing runs; new large-model campaigns use 64.
- `--ppo-precision bf16` autocasts dense and attention computations. Parameters,
  Adam state, recurrent computation, categorical probabilities, and loss
  reductions remain FP32. Inference remains FP32.
- Distributed PPO assigns whole trajectories from each original global
  minibatch to ranks. Global advantage normalization and optimizer-step order
  are preserved. Loss denominators use global real rows / world size; gradient
  averaging precedes clipping, and KL statistics/stopping are global. Empty
  ranks contribute zero-weight valid rows. Explicit NCCL/Gloo reductions avoid
  DDP hooks inside local CUDA capture. Forward/backward graphs share stable
  gradient storage; communication, clipping and Adam run outside capture.
  The first optimizer step matches ordinary PPO's uncompiled initialization.
  A separate Gloo group coordinates host readiness before CUDA reductions so
  a cold compile/capture on one rank does not leave another rank's NCCL reads
  outstanding. Host preparation has a bounded ten-minute timeout; GPU
  communication retains its ninety-second timeout.
- `--pipeline 2` keeps two future rollouts queued. Checkpoints retain their exact
  games, behavior-policy versions and post-draw RNG, with durable policy-file
  pins until a replacement checkpoint succeeds. `--duration-seconds` stops at
  an update boundary. Lag-1 duration stops preserve the prefetched draw's RNG.
- `--opponent-pool DIR` imports immutable opponents outside the new run's
  snapshot namespace. Resume restores that pool instead of deleting historical
  snapshots or overwriting files with colliding iteration numbers.

Attention uses PyTorch's automatic SDPA backend selection; FP32 and BF16 can
select different kernels. Kernel attribution must come from the actual input
shapes, not an assumption about FlashAttention. See the
[SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [CUDA graph requirements](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

## Frozen parent

On `h100-private`, the completed r6-h256 snapshot `pool/iter_02440.pt` was frozen
at 4,997,120 games. Its SHA-256 is
`0dd482bcd68e846cc7221a894b2fef711303cc183b1f4230666a5098930fbea7`.
The experiment parent and historical opponents receive one identity-initialized
attention layer: width 256, four heads, FFN 512, GRU, shared value head, features 6.
The source weights were trained under code `30d9ef640`; compare new arms only on
the same deployed candidate engine/code, which includes subsequent engine fixes.

Freeze a completed snapshot, its opponent history through that snapshot, and
its continuation schedule before preparing experiments:

```bash
.venv/bin/python tools/freeze_attention.py \
  --source RUN/pool/iter_02440.pt --source-ref 30d9ef640 \
  --opponent-pool RUN/pool --training-checkpoint RUN/latest.pt \
  --out FROZEN
```

`latest.pt` supplies schedule metadata only; `metrics.jsonl` supplies the frozen
snapshot's exact game count. The output includes weight hashes and pool hashes.
Fresh optimizers start at the parent's annealing/shaping positions. Larger-batch
arms explicitly set their requested starting LR at that position, retaining the
same remaining horizon and final LR. Frozen output is never overwritten.

## Resource-aware campaign

`tools/training_campaign.py prepare` writes reviewable commands; it starts no
jobs. GPU arguments are UUIDs in allocation order. At least eight CPU cores are
required; more workers/evaluation cores are preferable. The first four CPU slots
are learner ranks, slot five inference, the remainder workers plus evaluation.

```bash
.venv/bin/python tools/training_campaign.py prepare \
  --root SCREENS --frozen FROZEN --code CODE --code-ref COMMIT \
  --gpus UUID_A,UUID_B,UUID_C,UUID_D,UUID_E --cpus RESERVED_CPUS \
  --ladder /home/taiga-support/mtg-ml-checkpoints/ladder/L1
.venv/bin/python tools/training_campaign.py run \
  --manifest SCREENS/campaign.json --check-only
```

The runner refuses excluded buses (including ComfyUI and BE), occupied GPU
compute contexts, and CPU overlap with live trainers **and their descendants**.
It never stops another job. For detached execution, put `run --wait-seconds N`
in a server-owned launch script. Waiting is bounded; failures are recorded once,
the arm's own process group is cleaned up, and no respawn loop is used.

Screen arms, each with five warm-up and twenty timed iterations:

| Arm | Change from preceding relevant control |
|---|---|
| legacy | Eager attention, learner and inference share one GPU |
| stacked | Stacked attention, same GPU and PPO settings |
| separate | One learner and one dedicated inference GPU |
| resident64 / resident128 | Bound stable GPU slots; measure selection/wave cost |
| bf16 | Selective BF16 learning, FP32 inference |
| learners2 / learners4 | Two/four learner GPUs plus dedicated inference |
| batch8192-lr150 / lr300 | Global minibatch 8192, starting LR 1.5e-4 / 3e-4 |
| epochs2 | Two PPO epochs instead of four |
| lag2 | Policy lag two instead of one |

Expansion to two learner GPUs requires learning to limit throughput. Expansion
to four also requires the two-learner arm to improve total throughput by at least
5%. A skipped arm is recorded. Screens preserve 2048 games/rollout and the
original whole-trajectory minibatches unless that arm explicitly changes them.

After screening, prepare selected finalists in a **new** directory with
`--hours 2 --arms ARM,CONTROL --base-learners N`. This creates three paired seeds.
Use `--hours 6` for inconclusive finalists. Do not treat a kernel result as a
strength result. Evaluation uses fixed L1 ratings, sampled/greedy benchmarks and
per-deck results, every 250k games and at the final policy. Compare at common
game-count checkpoints and common elapsed times, and select strength per hour.
Finish with the ledger's 2,000-game sampled/greedy and paired head-to-head checks,
then archive using `~/mtg-ml-checkpoints/archive_run.sh`.

`outcomes.jsonl` records exact flags, code/parent hashes, hardware, real recorded
decisions/s, games/hour, padding, learner/inference memory, publication time,
wave overhead and charged/timed GPU-hours. Metrics retain queue delay separately
from actual collection time, and distributed learners report memory by rank.
Keep both successful and failed screens in the experiment ledger.

A proposed allocation was prepared on the server at
`/home/taiga-support/mtg-ml-256-opt/campaigns/screens-aed2ee8-proposed/campaign.json`.
It uses buses 18, 0A, 87, 90, C7 and CPU cores 48–63 as a **template**, not a
reservation. The read-only resource check correctly rejected overlap with the
live r6 trainer and its descendants. Reprepare in a new directory with the
actual dedicated CPU allocation and final deployed ref before launching. Prefer
cores local to the learner/inference GPUs' NUMA nodes. The dedicated campaign
was subsequently started after the user released the project GPUs; see the
experiment ledger for its live directory and allocation.

## Completed dedicated computational screens

The campaign on `h100-private` ran from 15:47:23 to 16:43:24 UTC on 2026-10-08,
using immutable code `e8662a5`, the frozen parent above, seed 0, 26 workers on
CPU cores 0–31, and H100 buses 18/0A. Each completed arm trained 25 iterations
of 2,048 games: five warm-up and twenty timed iterations. Rates below divide
real recorded games/learner decisions by whole-iteration wall time; stages
overlap. Evaluation was disabled for timing. All ten launched screens completed
without failure, totaling 512,000 games. Two/four-learner timing arms were skipped
by the planned bottleneck gate: after BF16, collection exceeded learner time.
Four-rank correctness separately passed all three eager FP32, graph FP32 and
compiled BF16 cases with unequal/empty shards (87.77 s).

| Arm | Games/hour | Learner decisions/s | Baseline ratio | Charged GPU-hours |
|---|---:|---:|---:|---:|
| legacy | 138,123 | 6,212 | 1.00× | 0.383 |
| stacked | 635,422 | 28,344 | 4.60× | 0.085 |
| separate | 946,019 | 41,981 | 6.85× | 0.137 |
| resident64 | 770,448 | 34,372 | 5.58× | 0.144 |
| resident128 | 777,230 | 34,237 | 5.63× | 0.136 |
| bf16 | 1,048,986 | 46,782 | 7.59× | 0.123 |
| batch8192-lr150 | 1,554,951 | 71,328 | 11.26× | 0.091 |
| batch8192-lr300 | 1,162,261 | 56,992 | 8.41× | 0.103 |
| epochs2 | 1,500,214 | 65,706 | 10.86× | 0.097 |
| lag2 | 1,397,025 | 61,379 | 10.11× | 0.098 |

These are single-seed short-screen rates, not estimates of stronger policies per
hour. KL stopping and new graph captures remain active: median optimizer steps
per iteration vary from 617.5 (legacy) to 386 (stacked), 159.5 (separate), 609
(resident64), 162.5 (BF16), and 41 (batch8192-lr150). The latter stopped on KL in
all twenty timed iterations. Different sampled actions produce different
trajectories even with paired seeds. These differences prevent attributing each
arm's entire speedup to faster kernels or interpreting the small residency/BF16
differences as established hardware effects. Longer paired strength comparisons
and identical-data learner timings are needed before changing defaults.

Only 22 policies were resident, including the learner; every residency arm used
one wave. This tests the bounded path but does not establish eviction-wave cost
for pools larger than 64/128. Stacked padding allocated approximately 2.75–2.83
rows per real row. Learner peaks were 20.35–24.58 GiB; inference peaks and all
stage/publication timings are retained in `outcomes.jsonl`. The live dashboard
at <http://127.0.0.1:55012> shows completed results and keeps polling every 15 s.

Artifacts are under
`/home/taiga-support/mtg-ml-256-opt/campaigns/screens-20261008-dedicated/`.
All ten final checkpoints, policies, metrics, commands and outcome records are
archived as `~/mtg-ml-checkpoints/20261008-h256-screen-ARM-seed0/`.
Sampled/greedy benchmarks, L1 Elo and final paired evaluations remain pending
for the strength stage; these timing archives explicitly record that limitation.
The five project GPUs were idle after completion; ComfyUI remained untouched.

## Preliminary stage measurements

These checks use `05103c4` on `h100-private`, buses 18/0A, one nice-19 thread on
shared CPU core 63, PyTorch 2.14.1 / CUDA 13.0,
`OMP_NUM_THREADS=MKL_NUM_THREADS=1`, and one compilation thread. They do **not**
establish an end-to-end training speedup.

Inference replay uses the frozen feature-6 parent, 26 policy copies, 19 requests
of 32 real decisions, the six-deck matchup mixture, five warm-up and twenty
timed batches. Eager attention took 447.53 ms/batch. Stacked attention took
6.20 ms/batch in one run and 64.69 ms/batch in a repeat as host scheduling varied;
CUDA-event graph replay remained 2.436–2.438 ms. The corrected padding counter
reports 1,488 padded rows for 608 real decisions. Peak process allocation was
4.80 GiB (4.51 GB for the 32-slot weight stack). FP32 attention selected
PyTorch's memory-efficient CUTLASS kernel. These are random legal decisions and
copies of the parent, rather than a completed mature-pool training rollout.

Learner timing uses 21,475 real decisions / 192 trajectories collected from 128
six-deck games with the frozen parent; identical saved data feeds both arms.
Each uses minibatch 2048, four PPO epochs, five warm-up updates of four epochs,
and twenty timed updates, with KL stopping disabled for the timing tool.

| Precision | Time for 858 optimizer steps | Mean step | Peak process allocation |
|---|---:|---:|---:|
| FP32 | 44.76 s | 52.17 ms | 5.0 GiB |
| BF16 dense/attention | 42.27 s | 49.26 ms | 4.0 GiB |

Both timed regions captured 27 additional recurrent shapes; capture time was
7.07 s / 2.38 s, respectively. CPU contention and overlapping isolated checks
prevent treating the small wall-time difference as a reliable BF16 gain. Keep
FP32 as the default pending dedicated-resource measurement.

Profiling one actual learner graph (`gru=(20,512)`, entity-width 48, padded rows
2816) confirmed FP32 CUTLASS memory-efficient SDPA and BF16 cuDNN flash SDPA.
Both still launch 512 recurrent matrix products in each direction, plus the
recurrent elementwise kernels. The width-256 GRU remains a substantial learner
cost; the attention-only synthetic gain did not translate into a comparable
complete-update gain here. Raw logs and the replay profiler are retained in the
isolated deployment. A large full-update trace was stopped after its timed work
completed because aggregation consumed excessive host time; the timing table
above comes from separate runs without a profiler.

## Validation and current limits

Local validation passed 1,228 fast CPU tests, plus the focused slow tests for
distributed unequal/empty shards and multi-server residency. The focused
training/inference/attention suite passed 138 tests. CUDA checks cover compiled
and uncompiled mixed-policy attention, trained nonzero weights, pointers,
recurrent reset, zero-entity decisions, wide buckets, sparse policy IDs, and
small→large→small backward graphs with stable gradient addresses and allocator
cache flushing. Ten final CUDA checks passed on buses 18/0A at `aed2ee8`,
including two-GPU NCCL eager/graph parity and compiled BF16 against single-GPU
updates with unequal/empty shards. FP32 Adam moments use strict comparison;
BF16 moment errors are bounded to 1% in both norm and largest entry because
sharded dense GEMMs round gradients separately. Observed maximum relative norm
error was 0.59%; loss and parameter checks keep their original tolerances.
The final coordination/campaign CPU checks passed 22 tests. Four-rank CUDA
correctness subsequently passed three additional checks on the dedicated
allocation. Multi-learner throughput was gated off as described above.
Campaign validation also covers occupied compute contexts,
excluded GPUs, invalid allocations, and refusing to overwrite manifests.

The isolated deployment is under `/home/taiga-support/mtg-ml-256-opt/`, with an
environment and native build for each deployed code ref. Existing r6 jobs and
ComfyUI were preserved. The separate video-evidence vLLM job released bus 0A;
isolated checks use buses 18/0A, one low-priority thread on CPU 63. Host-side
timings under that contention are not an end-to-end training estimate. Full
computational campaign execution subsequently completed on dedicated resources.

The real width-256-attention baseline and optimized throughput are now measured
in short screens, exceeding the 3–5× throughput target. Learning-rate/batch/epoch/
lag selection and three-seed strength results remain open. CPU bookkeeping has
not been moved into Rust because its post-optimization bottleneck has not been
established.
