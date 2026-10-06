# Rollout throughput: packed samples and the central inference server

With the Rust engine (see [native-engine.md](native-engine.md)) the rules engine stopped being the limit, so this pass looked at the rest of the rollout pipeline. It found and fixed a bottleneck in the trainer itself, and added a central GPU inference server.

```bash
python -m mtg_ml.rl.train --engine native ...                        # local CPU inference in each worker
python -m mtg_ml.rl.train --engine native --inference server ...     # one GPU process serves every policy
```

## Results (h100-private, 64 cores; measured 2026-10-06 with the box otherwise idle)

`tools/bench_rollout.py` runs training-shaped rollouts (half self-play, half against a second checkpoint, games 1 and 2, trajectories recorded) without the PPO update. Each worker is pinned to its own core.

| decisions/s | before this pass | now |
|---|---:|---:|
| 30 workers, local inference (GRU, hidden 128) | 155k | **300k** |
| 60 workers, local inference, hidden 128 | n/a | **520k** |
| 60 workers, local inference, hidden 256 / 512 | n/a | 433k / 237k |
| 60 workers, server pipeline without the network (`--dry-run`) | 146k (30 workers) | **640k** |

**With the GPU** (after a reboot fixed CUDA on 2026-10-06; one H100, 60 workers, server on 2 cores):

| decisions/s | local inference | server on GPU |
|---|---:|---:|
| hidden 128 (current model) | 520k | 337k |
| hidden 512 | 237k | **332k** |

The server's throughput hardly depends on network size; for the current tiny network local CPU inference is still faster. The server is now limited by its own inference step: 23 of 27 busy seconds, about 8.7 ms per batch of ~1,800 decisions from two policies, almost the same at hidden 128 and 512. That cost is launch and preparation overhead, not GPU compute, which makes it the next thing to cut (CUDA graphs, fusing the two policies' batches). In isolation the server handles ~1,000 decisions from two policies in 3.4 ms, and 8,192 decisions of one policy in 5.6 ms (`tools/profile_server.py`).

One GPU on h100-private (PCI 0000:be:00.0, index 6 when all 8 are present) fell off the bus on 2026-10-06 and took CUDA down for every process until the reboot. Select GPUs by UUID or bus id and avoid that card.

## Finding 1: unpickling samples capped every rollout

Workers finished their jobs in at most 1.6 s, but a round took 5.5 s. The trainer spent the difference unpickling the recorded samples (one list of lists of ints per decision), in a single process. That capped rollouts at ~150-200k decisions/s, whatever the engine or inference speed.

`rl/samples.py` adds `PackedSamples`: flat int32 arrays (lengths + values) that pickle as raw bytes, 31× faster than the lists. `rl.model.collate_packed(samples, idx)` builds a PPO minibatch from any subset with vectorized ops and gives tensors identical to `collate`. `Result.samples` is now a `PackedSamples`; it still supports `len`, indexing and iteration, returning the old tuples. Workers record `features.featurize_flat` (options already flattened), so packing a decision takes three `array.extend` calls instead of one per option.

## Finding 2: CPU inference is cheap for today's network

With samples packed, local inference scales nearly linearly with workers (520k decisions/s on 60 cores), because the network is tiny: about 17M parameters, almost all of them hashed embeddings. The server's pipeline, measured without any network, tops out at about 640k on the same cores. So for the current model the server is worth little. It pays off as the network grows: local throughput halves from hidden 128 to 512, while the server's cost per batch hardly depends on network size. That makes the server the way to run the planned bigger trunk (a transformer over board objects) and search.

## How the server works

`mtg_ml/rl/inference.py`:

- **Workers** (`InferenceClient`) only play: engine, `featurize_flat`, event hashes, recording. They do not import torch, so they could also run on PyPy. Each request is one worker's batch of decisions, written into that worker's own shared-memory block. Only a small header crosses the request queue, so no features are pickled.
- **The server process** (`serve`) takes every queued request, waiting up to `max_wait_ms` for more while the batch is small. It gathers each field across requests with one vectorized index, and selects rows per policy on the CPU (no device syncs). Each policy's batch goes to the GPU in one copy; the server samples there and returns (action, log-prob, value) per decision through the worker's output block, with one copy per request.
- **Recurrent state** stays on the GPU: a hidden-state table indexed by (worker, slot), where a slot is one (game, seat) of the worker's job. The first decision of a sequence carries `fresh = 1` (zero state).
- **Policies** are (checkpoint path, version), as in `rollout.load_policy`. The server loads them on first use and drops older versions of the learner.
- Queues are `SimpleQueue`s, which write in the calling thread. A `Queue` hands writes to a feeder thread that can wait a whole GIL switch interval (5 ms) while its process is busy.
- Pinning: `ServerConfig.cpus` gives the server its own cores, and `InferenceServer(..., worker_cpus=...)` pins workers one per core. `Job.groups` sets the requests in flight per worker; 1 measured best on 60 workers, because the server's cost grows with the number of requests.

**Correctness:** `tests/test_inference.py` checks that rollouts through the server record exactly the log-probs and values the network computes when it replays the recorded trajectories from scratch (sequence mode, as in PPO). The check runs for GRU and MLP memory, with two policies, scripted and random seats, and several workers batched together. It also checks that server-mode workers never import torch.

## Measuring

```bash
# end to end, one pinned process per core; --dry-run skips the network
python tools/bench_rollout.py --engine native --inference server --workers 60 --games 8160 --server-cpus 0-1 --worker-cpus 2-61 --groups 1
taskset -c 2-61 python tools/bench_rollout.py --engine native --inference local --workers 60 --games 8160
# server batch processing in isolation (real featurized decisions)
CUDA_VISIBLE_DEVICES=4 python tools/profile_server.py --device cuda --requests 30 --rows 33 --policies 2
# MTG_SERVER_PROFILE=/tmp/srv.prof dumps a cProfile of the server loop
```

`bench_rollout.py` also prints per-job wall times and the share of time workers wait for inference. The server prints its busy time split into parse, infer and reply.

## Next steps

1. Cut the server's fixed cost per batch: CUDA graphs or `torch.compile` for the forward pass, and pinned staging buffers.
2. Raise the worker ceiling: the Rust side could write requests and samples straight into the shared buffers, saving the remaining Python packing.
