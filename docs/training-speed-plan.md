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

## Results

(filled in as the workstreams land)
