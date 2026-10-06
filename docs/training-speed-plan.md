# Training speed plan for the h128 entity encoder (2026-10-06)

The h128 entity encoder (`--trunk entity --hidden 128 --value-net shared`) is the best policy so far
(52% game-1 vs the blue bot at 490k games, run `bench500k-h128-entity` on h100-private). Longer runs
(2-5M games) need a faster iteration. This note records what one iteration costs today, why, and the
plan to cut it. Results are appended at the end as the work lands.

## Measured today (h100-private, 31 workers pinned to cores 0-31, local CPU inference, update on one H100)

From `runs/bench500k-h128-entity/metrics.jsonl` (240 iterations, 2048 games each):

| iterations | decisions / iter | rollout_s | update_s | optimizer steps / iter | ms / step |
|---|---:|---:|---:|---:|---:|
| 0-20 | 260k | 3.1 | 8.9 | 495 | 18.0 |
| 60-120 | 348k | 5.2 | 11.8 | 656 | 17.9 |
| 120-240 | 378k | 6.0 | 12.6 | 711 | 17.7 |

- Non-eval iteration: **16.9 s = 6.0 s rollout (31%) + 12.6 s update (69%)**, run sequentially. 500k games took 72 min.
- The PPO update takes **17.8 ms per minibatch step** for an h128 network on an H100. The arithmetic is tiny; the time is CPU-side work and host-device synchronisation.
- The rollout delivers ~63k recorded decisions/s on 31 cores, about 2.7k decisions/s per worker including the unrecorded pool-opponent seats, against a 60k decisions/s per core engine + featurize path (`docs/native-engine.md`). The workers spend almost all their time outside the engine.
- Step-mode CPU inference (one call per worker per step, 1 thread, M3): entity and MLP trunks cost about the same, 0.35-0.55 ms per call for 1-34 decisions, i.e. 12-16 µs per decision at batch 34 but 350 µs per decision at batch 1. So the per-call overhead, not the trunk, is what hurts once a job's batch shrinks.

## Key issues, ranked by expected gain

1. **Rollout and update run one after the other.** The GPU idles during the rollout and 31 cores idle during the update. Overlapping them (collect iteration k+1 with the weights of iteration k while updating on iteration k's data; the standard one-step policy lag, the clipped ratio handles the stale behaviour policy) makes the wall time per iteration max(rollout, update) instead of the sum. Expected: 16.9 s -> ~12.6 s now, and the whole gain of item 2 afterwards.
2. **The update is synchronisation-bound.** Per minibatch, `ppo_update` collates on the CPU (`collate_packed`), copies 13 tensors to the device one by one, and the forward pass forces several device syncs: `int(length.sum())` in `_segments`, `int(n_ent.sum())` and boolean-mask indexing (`idx[glob]`, `idx[inent]`, `_keep`) in `EntityEncoder` and `_options`, `int(b.n_opts.max())`, a Python loop over trajectories in the GRU unpad, and six `.item()` calls for the statistics. Adam runs unfused over ~150 parameter tensors (about 750 small kernels per step). Target: **<= 5 ms per step (<= 4 s per iteration)** with identical math: dataset moved to the device once per update, minibatch gathers built without syncs, entity segment structure precomputed in the collate, statistics accumulated on the device with one sync per epoch, fused Adam.
3. **Rollout tail and per-step overhead.** Each worker plays a fixed set of ~66 games; as games end its inference batch shrinks to a handful of decisions at ~0.35 ms per call per policy, and the round waits for the slowest worker. Every worker also reloads the 204 MB `latest.pt` (model + Adam state) at the start of each iteration although it only needs the 68 MB of weights, and `_LocalEvaluator.submit` does per-row Python work (hidden-state dict, per-row `int()`/`float()`). Target: a shared game queue so each worker keeps a constant number of games in flight and all workers finish within about one game of each other; vectorised evaluator bookkeeping; weights-only policy files for the workers. Expected: 6 s -> ~3 s at 31 workers; the central GPU server (`--inference server`) is to be re-measured for the entity trunk at 31 and 60 workers.
4. **Evaluation blocks training** (~2,000 extra games every 5-10 iterations): about 5% of wall time; acceptable for now, left as is.

Not pursued now: the transformer trunk, multi-GPU data parallelism (the update is launch-bound, not compute-bound; a second GPU would not help), Rust writing samples straight into shared buffers (the engine side is a few percent of a worker's time).

## Plan: three independent workstreams

| workstream | files owned | deliverable | success criterion |
|---|---|---|---|
| WS1 update | `rl/ppo.py`, `rl/model.py`, `tools/bench_update.py`, `tests/test_ppo_fast.py` | sync-free, device-resident PPO update; benchmark tool | <= 5 ms/step on the H100 for the h128 entity config on the same recorded data; new collate equals the reference collate; `ppo_update` statistics unchanged on CPU for a fixed seed |
| WS2 pipeline | `rl/train.py`, `tests/test_train_pipeline.py` | rollouts overlap the update (`--pipeline`, default on; `--pipeline 0` is today's order); versioned weights-only policy files; non-blocking `latest.pt` | iteration wall = max(rollout, update) + < 0.5 s; `--pipeline 0` reproduces today's metrics; resume works |
| WS3 rollout | `rl/rollout.py`, `rl/inference.py`, `tools/bench_rollout.py`, `tests/test_rollout_stream.py` | shared game queue (constant games in flight per worker), vectorised local evaluator, lighter policy loading, entity trunk in the rollout benchmark, local vs server numbers | 2048-game entity rollout at 31 workers <= 3 s on the box; exact-replay tests still pass |

After the merge: `train.py` adopts the WS3 rollout helper and WS1's fused optimizer; a 20-iteration integration run of the entity config on the box measures s/iteration against 16.9 s; `docs/inference-server.md` and the handoff note get the numbers.

## Verification

- `python -m pytest tests/test_rl.py tests/test_inference.py` plus the new test files, on both engines.
- Exactness: rollouts record exactly what the network computes on replay (`tests/test_inference.py`), for every trunk and both inference modes.
- Benchmarks on h100-private with the entity run's configuration (`--hidden 128 --trunk entity --value-net shared --games-per-iter 2048 --workers 31 --engine native --device cuda`), pinned CPUs, GPUs chosen by UUID; GPU 6 (bus BE:00.0) is never used.

## Results (2026-10-06, same box, GPU 4, cores 0-31, 31 workers, 2048 games per iteration, same seed)

Old code is the `bench500k-h128-entity` code (`~/mtg-ml-v3`), new code this branch. Ten iterations each, means over iterations 3-10:

| | decisions / iter | rollout_s | update_s | wall_s / iter | µs / decision | ms / optimizer step | 10 iterations |
|---|---:|---:|---:|---:|---:|---:|---:|
| old (sequential) | 226k | 3.30 | 7.80 | 11.10 | 49.2 | 18.1 | 120 s |
| new, `--pipeline 0` | 231k | 1.80 | 1.89 | 3.71 | 16.0 | 4.3 | 45 s |
| **new, `--pipeline 1`** (default) | 248k | 1.93 | 2.50 | **2.52** | **10.2** | 5.3 | **35 s** |

**4.4x less wall time per iteration, 4.8x per decision.** Early iterations have short games and few pool opponents; late in a run (378k decisions, 25 pool checkpoints) the measured pieces are 2.7-2.9 s rollout and ~2.8 s update, so the 16.9 s iteration of the old run should take about 3.5-4 s: 500k games in roughly 17 minutes instead of 72.

Per workstream (each measured in isolation on the box):

- **WS1, PPO update** (`rl/ppo.py`, `rl/model.py`, `tools/bench_update.py`): 15.2 -> 3.9 ms per optimizer step for h128 entity (h128 mlp 14.0 -> 3.4, h512 separate 32 -> 23). The decisions go to the device once per update as int32; each epoch is laid out in minibatch order and the entity structure (`model.structure`) derived once, minibatches are slices (`model.split`); statistics are summed on the device and read once per epoch; the GRU runs on end-padded trajectories (exact for zero initial state; cuDNN persistent kernels up to hidden 128, packed above); `make_optimizer` gives fused Adam on CUDA. Statistics match the old implementation to run-to-run CUDA noise; 25 new tests compare the new paths against the old ones. Peak device memory 0.4 -> 6.5 GiB (the epoch lives on the device).
- **WS2, pipelined trainer** (`rl/train.py`): `--pipeline 1` plays iteration k+1 with the weights of k during the update of k (one-step policy lag, logged as `policy_lag`); weights-only policy files `<run>/policy/vNNNNN.pt` (68 MB instead of the 204 MB `latest.pt`, newest 3 kept) are what workers and the server load; `latest.pt` is written in a background thread; while pipelining the update uses only the cores the workers leave free (8 OMP threads next to 31 busy workers slowed it by 60%). New metrics `wall_s`, `wait_s`, `policy_lag`. `--pipeline 0` reproduces the old loop bit for bit.
- **WS3, rollouts** (`rl/rollout.py`, `rl/inference.py`, `tools/bench_rollout.py`): a worker's time was 81-84% inference, 38% of it after half its games had ended. Now every worker keeps 64 games live and claims the next from pool-wide counters (`create_pool` + `run_specs`/`play`), sticking to one opponent network at a time (about 2 forward passes per step instead of up to 9); hidden states live in a slot table; samples return through shared memory; checkpoints are memory-mapped (220 -> 6 ms per load). 2048-game entity rollout at 31 workers with 25 pool opponents: 6.1-7.6 s -> 2.7-2.9 s (74k -> 182k decisions/s; 240k at 60 workers). The GPU inference server is **slower** for this trunk (15k decisions/s with 26 policies, 68k with 2): its per-batch cost is kernel launches and syncs in the step-mode forward, so use `--inference local` for h128 entity.

Recommended command (box, one run on cores 0-31):

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=<uuid> taskset -c 0-31 python -m mtg_ml.rl.train --run runs/X \
  --hidden 128 --trunk entity --value-net shared --inference local --engine native --device cuda \
  --workers 31 --games-per-iter 2048 --total-games 2000000 --iterations 100000 --eval-every 5
```

Open: with pipelining the update is 30% slower than alone (2.50 vs 1.89 s: one torch thread and NUMA contention; 30 workers with 2 trainer threads may be the better split, untested); h512 separate stays at 23 ms per step (two packed GRUs above hidden 128); evaluation still blocks the loop (~5%); worker processes are seeded from torch's random seed, so multi-process runs are not reproducible (they never were).
