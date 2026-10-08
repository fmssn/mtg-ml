# Width-256 attention: implementation and experiment protocol

The optimized path is implemented; the 3–5× end-to-end target still needs the
full training comparison. Existing r6 training processes were preserved.

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

## Validation and current limits

Local validation passed 1,228 fast CPU tests, plus the focused slow tests for
distributed unequal/empty shards and multi-server residency. The focused
training/inference/attention suite passed 138 tests. CUDA checks cover compiled
and uncompiled mixed-policy attention, trained nonzero weights, pointers,
recurrent reset, zero-entity decisions, wide buckets, sparse policy IDs, and
small→large→small backward graphs with stable gradient addresses and allocator
cache flushing. Nine CUDA checks passed on buses 18/0A, including two-GPU NCCL
eager/graph parity against single-GPU updates with unequal/empty shards and Adam
state comparison. Four-rank CUDA correctness and scaling still require the
dedicated resources. Campaign validation also covers occupied compute contexts,
excluded GPUs, invalid allocations, and refusing to overwrite manifests.

The isolated deployment is under `/home/taiga-support/mtg-ml-256-opt/`, with an
environment and native build for each deployed code ref. Existing r6 jobs and
ComfyUI were preserved. The separate video-evidence vLLM job released bus 0A;
isolated checks use buses 18/0A, one low-priority thread on CPU 63. Host-side
timings under that contention are not an end-to-end training estimate. Full
campaign execution waits for dedicated GPU/CPU resources.

The real width-256-attention unoptimized training baseline, 3–5× end-to-end
target, learning-rate/batch/epoch/lag selection, and three-seed strength results
remain unmeasured. CPU bookkeeping has not been moved into Rust because its
post-optimization bottleneck has not been established.
