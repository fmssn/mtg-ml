# Training throughput: update profile and A/Bs (h100-private4, from 2026-10-10)

Machine for every number: h100-private4, 8x H100 80GB (SXM, 700 W), 128 logical CPUs, torch 2.14.1+cu130. Code: this branch (`claude/update-throughput`) on top of `claude/lucid-euler-0tjz9h`; every new knob defaults to the current behaviour.

## 1. Where one PPO update goes

Setup: `tools/bench_update.py` on a fixed batch of 2048 games (323k decisions, 3072 trajectories, mean 105 steps, max 580, 540 state tokens per decision), played by the r7 lr075 parent (`--init`) on the matrix mix (postboard 0.2); feature set 7, h256, bf16, capture 2, minibatch 2048, 4 epochs, no KL stop, one H100, 8 cores. `--torch-profile-json` + `tools/profile_breakdown.py` group the kernels by name.

Wall: 2.35 s per epoch of 153 steps = 15.5 ms per step (9.4 s for 4 epochs). Summed kernel time is 14.9 ms per step, i.e. the GPU is back to back busy with 1,850 kernel launches per step, replayed as one CUDA graph. cProfile of the host: 8.4 of 9.4 s sit in `CUDAGraph.replay` (the host waits for the queue), the rest (pad_split 0.2, structure 0.1, `tolist` 0.1, `.to` 0.1, empty_cache 0.1 s) is per-epoch preparation. A step draws about 210 W of 700 W at 1980 MHz.

| part | ms/step | share | launches/step |
|---|---:|---:|---:|
| GRU recurrence (cuDNN per-step gemm + cell, forward and backward, FP32/TF32) | 5.9 | 40% | 1,289 |
| embedding bags (gather forward + transposed gradient, bandwidth bound) | 2.9 | 20% | 22 |
| other (index/copy/arange/sub of the per-epoch layout, mostly outside the graph) | 1.7 | 12% | 40 |
| entity attention (cuDNN flash fwd/bwd) | 1.2 | 8% | 4 |
| compiled losses and pointwise (Inductor) | 1.1 | 8% | 113 |
| dense gemms (MLPs, scorer, projections) | 0.8 | 5% | 56 |
| memset/memcpy | 0.7 | 5% | 321 |
| fused Adam + grad norm | 0.5 | 3% | 3 |

- **The GRU is the update's latency floor.** cuDNN has persistent kernels only up to h128; at h256 each time step is a ~8 us gemm plus a ~2 us cell kernel forward and about 8 us backward, ~20 us per step in all, and a minibatch runs `longest trajectory in it` steps (about 320 on average at minibatch 2048, 49k steps per epoch). It does not depend on the number of sequences (+0.06 us per sequence), so a larger minibatch costs almost no extra GRU time.
- Embedding bags move 1.1M tokens x 256 values per minibatch: ~0.36 ms each, bandwidth bound, linear in decisions.
- Host-device syncs and Python are ~5-8% of the update (per-epoch layout and the one `tolist` for the KL stop).
- Standalone cuDNN GRU, T=320 steps, 20 sequences, fwd+bwd in a graph: FP32 7.6 ms, BF16 3.7 ms, FP16 3.7 ms.

## 2. Cheap wins (update only, same fixed batch)

Seconds per epoch of 323k decisions, graph captures excluded, 3 repeats after warm-up, eight benches at once on separate GPUs and cores. Power: mean board power while the GPU is busy.

| variant | s/epoch | speedup | W |
|---|---:|---:|---:|
| baseline (minibatch 2048, capture 2) | 2.35 | 1.00 | 272 |
| minibatch 4096 | 1.30 | 1.81 | 294 |
| minibatch 8192 | 1.20 | 1.96 | 326 |
| minibatch 16384 | 0.97 | 2.42 | 358 |
| capture 1 (no Inductor losses) | 2.93 | 0.80 | 339 |
| capture 0 (eager) | 3.08 | 0.76 | 229 |
| epochs 2 | 2.44 per epoch, half the epochs | 2.0 per update | 270 |
| GRU in BF16 (`--ppo-gru-precision bf16`) | 1.53 | 1.53 | |

Fused Adam is already used (`make_optimizer`), clipping is folded into it (`FOLD_CLIP`), and the step is one CUDA graph; nothing is left to gain there. Beyond minibatch 4096 the update turns bandwidth bound (embedding gathers, attention) and gains flatten.

Numerics of the BF16 GRU (`tools/check_update_numerics.py`, 20 minibatches, same batches, params fixed): gradient cosine 0.9787 (min 0.77) against the FP32-GRU update, the same level as the existing FP32 -> BF16 autocast gap (cosine 0.9783). It is not identical math, so it needs the learning A/B below.

Learning-equivalence arms (below, `tools/throughput_ab.py`): see the ledger.

## 3. Learning equivalence and packing

Learning results (2M games per arm from the r7 lr075 parent, L1 and head to head) are in the ledger entry `20261010-throughput-ab`. Summary: ep2 +45% games/h with L1 150 vs 94 and h2h +0.07 (one seed, replicate); minibatch 4096 (+13%), 8192 (+25%), BF16 GRU (+18%) and mb4096+GRU (+31%) equal base within noise. PR #116 (featurize, +55% rollout) is merged, so the update is the bound again.

### Packing on h100-private3 (next to the two production base runs, checkout 280ebc5, untouched)

Throughput only (no evaluation), r7 lr075 init, matrix mix, production recipe, one run per learner GPU + server GPU pair. Production baseline over 13 min: h256 858k games/h (wall 8.5 s), h512 654k (11.6 s), box CPU 40%. The box has 64 CPUs without SMT siblings, so a first "third run on siblings" attempt failed at start.

| configuration | extra run games/h | h256 prod | h512 prod | box CPU | notes |
|---|---|---|---|---|---|
| 3 runs: third on GPUs 2F+38, workers on cores 8-31 | 743k (wall 9.9 s) | 0.99 | 1.01 | 60% | no production loss over 28 min; third run at 87% of the baseline h256 rate |
| 4 runs: fourth on GPUs BE+C7, workers over 8-31,40-63 | 601k (wall 12.3 s), short window | 0.71 in the first minutes (startup/compile; wall 8.5 -> 11-12 s) | 0.98 | 64% | ended after about 7 min, before a steady state; the h256 dip exceeds the 25% limit |

Verdict: schedule up to three runs per box (two production runs plus one with workers pinned to the idle cores); do not add a fourth on a 64-CPU box. Server co-location on the learner GPU was not measured.

### Recommendation

1. Adopt `--ppo-epochs 2` after a second-seed replication (about +45% games/h and stronger at equal games in this test).
2. If epochs stay at 4: `--ppo-minibatch 4096 --ppo-lr 1.06e-4` (x sqrt 2) and `--ppo-gru-precision bf16`, +13% and +18% (+31% together), no measurable learning cost.
3. Not tested: ep2 combined with the speed knobs.
