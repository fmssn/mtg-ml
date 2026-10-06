# Training speed plan 2: use the whole box (2026-10-06)

Follow-up to [training-speed-plan.md](training-speed-plan.md). That pass cut an iteration from 11.1 s to
2.5 s by removing overhead inside each stage. This one changes the division of labour, because the
stages are still placed wrongly for the hardware: 2x Xeon 8462Y+ (64 cores, 4 NUMA nodes of 16),
8x H100 of which 4-6 are usable. Target: the cores play games at the speed of the engine, one GPU
serves every policy, one GPU updates, and nothing waits on anything else.

## Where the throughput goes today (one run, 31 workers on cores 0-31, GPU 4)

| | decisions/s |
|---|---:|
| Rust engine alone, 64 cores (`docs/native-engine.md`) | 12,400k |
| rollout CPU path, engine + featurize + glue, no network, 64 cores | 3,950k |
| rollout with local CPU inference, 31 workers (60 workers) | 182k (240k) |
| trained end to end at 2.5 s per 2048-game iteration | ~100k |

The rollout runs at about 5% of what the cores can play; a worker spends 81-84% of its time in CPU
torch inference of a 17M-parameter network at batch ~32. The PPO update takes 4-5 ms per step for
arithmetic that is a few percent of an H100; it only looks busy. Profiles of the current code taken
for this plan are in the results section at the end.

## What is left on the table, ranked

1. **Inference runs on the CPU.** The GPU server was measured slower for the entity trunk (15k
   decisions/s with 26 policies) and set aside. That number is the server's implementation, not the
   GPU: one forward per policy, each ~3 ms of launches with a `.cpu()` sync per policy per batch, 26
   syncs per batch. With every policy served by one batched forward, structure built without syncs
   and the forward captured in CUDA graphs, one H100 serves 63 workers.
2. **Python glue in the worker** becomes the floor once inference leaves: 60k decisions/s per core on
   the rollout CPU path against 190k for the engine. `prepare`/`apply` loop in Python per decision,
   `featurize_flat` lists re-packed into `array("i")`, `torch.distributions.Categorical` per call, GAE in
   Python. A Rust lockstep loop makes a step one call each way.
3. **NUMA is backwards.** `nvidia-smi topo -m`: GPUs 0-3 are local to cores 0-15 (node 0), GPUs 4-7 to
   cores 32-47 (node 2). Today's run pins the trainer and workers to 0-31 and uses GPU 4, so every
   transfer crosses the sockets.
4. **The trainer's rollout thread shares the GIL with the update.** `_Rollout` merges results
   (`imap_unordered`, `unpark`, list extends) in a thread of the trainer process while `ppo_update`
   launches kernels from Python. That is the likelier cause of the 30% slower update under pipelining.
5. **The update is launch-bound at 4-5 ms per step** and hidden behind the rollout only as long as the
   rollout is slow. Bigger minibatches and a captured step bring it under 1 ms.
6. **Evaluation blocks the loop** and plays about a quarter of all games on the training cores.
7. **31 -> 60 workers scales 1.3x.** Something serial saturates; nobody profiled it.
8. **The loop is iteration-shaped.** At 1M decisions/s a 2048-game iteration arrives every 0.25 s; a
   learner draining a continuous queue (lag 1-3, clipped IS) removes the tail wait and the coupling.

Not a target: data-parallel multi-GPU for h128 (launch-bound, no compute to split). The extra GPUs
are for concurrent runs and for a trunk that has compute (transformer, search).

## Workstreams

Independent, each in its own worktree and its own scratch copy on the box, each with a reserved
CPU range and GPU so the measurements do not disturb each other.

| | files owned | box resources | success criterion |
|---|---|---|---|
| **WS4 server** | `rl/inference.py`, `rl/model.py` (step-mode forward), `tools/profile_server.py`, `tools/bench_rollout.py`, `tests/test_inference.py` | cores 0-31, GPU 0 (node 0, local) | h128 entity, 25 pool opponents, 31 workers: server >= 2x local (>= 360k decisions/s); exactness tests pass |
| **WS5 update** | `rl/ppo.py`, `rl/model.py` (sequence mode), `tools/bench_update.py`, `tests/test_ppo_fast.py` | cores 60-63, GPU 5 | an epoch over 250k decisions <= 0.3 s (from ~0.6 s); statistics unchanged for minibatch 2048; minibatch 8192 and 16384 measured |
| **WS6 trainer** | `rl/train.py`, `rl/evaluate.py`, `tests/test_train_pipeline.py` | cores 32-59, GPU 4 (node 2, local) | `update_s` pipelined == `update_s` alone (+-5%); evaluation off the training loop's wall time; NUMA-aware defaults |

After these merge: WS7, the Rust lockstep loop (`native/src/py.rs`, `rl/rollout.py`), sized from the
worker profile after WS4 lands, and the 63-worker end-to-end measurement with the server.

### WS4 server

- One batched forward for all policies: pool checkpoints stacked (`torch.func.stack_module_state` +
  `vmap`, or batched weights and `bmm`), with the learner as one more member; rows carry a policy
  index. Hidden-state tables per policy become one table indexed by (policy, worker, slot).
- Entity structure (`model.structure`) without host syncs in step mode: lengths and counts come from
  the worker's request header (it already builds the int32 arrays), so the server knows every size
  before touching the device.
- CUDA graphs over a few padded shape buckets (rows, tokens, options), the hidden-state gather and
  scatter inside the graph. Measure with `tools/profile_server.py` first: per-batch cost today ~3 ms
  per policy, target <= 1 ms per batch regardless of policies.
- One server process per GPU with workers sharded (`InferenceServer` takes a device list), pinned to
  the GPU's NUMA node.

### WS5 update

- `torch.compile(mode="reduce-overhead")` or a hand-captured CUDA graph of forward + backward +
  fused Adam over bucketed shapes (the epoch layout is known up front, so buckets can be chosen per
  epoch). Measure per step with `tools/bench_update.py`.
- Minibatch 8192 and 16384: time per epoch, and a learning check (same games, bench score after the
  same number of decisions at lr scaled by sqrt of the batch ratio, 3 seeds each; any regression is
  reported, not hidden).

### WS6 trainer

- Collection in its own process: the rollout result is merged there and handed over through shared
  memory, so the update's Python loop owns the GIL.
- Evaluation in its own process on spare cores, reading `policy/vNNNNN.pt`, results merged into
  `metrics.jsonl` by iteration; the training loop never waits for it.
- Defaults: trainer process and server pinned to the GPU's NUMA node (`nvidia-smi topo`, or
  `/sys/bus/pci/devices/*/numa_node`), workers on the rest; `--worker-cpus`/`--trainer-cpus` to override.
- Optional, if time allows: a continuous-queue mode (`--pipeline 2+`), rollouts never stop, the
  learner trains on whatever arrived (lag logged per sample).

## Verification

- `python -m pytest tests/test_rl.py tests/test_inference.py tests/test_ppo_fast.py tests/test_rollout_stream.py tests/test_train_pipeline.py` on both engines.
- Exactness: rollouts record exactly what the network computes on replay, every trunk, both
  inference modes.
- Every number in this doc comes from the box with the stated CPU range and GPU UUID; GPU 6 (bus
  BE:00.0) is never used; the box is shared, so `uptime` and `ps` first, never `pkill -f`.

## Results

### Profiles of the current code (2026-10-06, box idle, py-spy 100 Hz with `--subprocesses`)

60-worker local rollout, h128 entity, 25 pool opponents (`tools/bench_rollout.py`), samples inside a
worker's job: CPU forward 57% (entity encoder with its mask ops ~11%, GRU 7%, options + scorer 12%),
`featurize_flat` + scripted steps 20%, recording loop 8%, `Categorical` sample + log-prob 4%, batch
building 3.5%. The bench's own line: "waiting for inference 72%".

Pipelined training (31 workers, cores 0-31, GPU 4), samples of the trainer process: `ppo_update`
25%, the rollout thread (`play` / `run_specs` / `unpark`) 14%, `_save` 6%, `_cpu` of the state dicts
3% (`_checkpoint` copies the 136 MB Adam state to the host and pickles 204 MB every iteration),
`packed_tensors` 4%. The merge thread holds the GIL about a third as often as the update thread
while both run.

### WS4 results

(pending)

### WS5 results

Setup: `tools/bench_update.py --device cuda --trunk entity --hidden 128 --value-net shared` on one
recorded set (`--save-data`: 2048 native games of a fresh network, 321,682 decisions in 3,072
trajectories, mean 105 and longest 316 decisions, 219 state tokens per decision), cores 60-63. The
first measurements ran on GPU 5; from 18:52 another session's training shared GPU 5, so every
number below was retaken on **GPU 7** (`GPU-bd861242`, idle at the time). "Per 250k" scales an epoch
over the 321,682 decisions to 250k.

**The step was not only launch-bound.** A capture of the unchanged step replayed in 2.6 ms against
3.4 ms eager: the H100 itself needed ~2.4 ms per minibatch of 2048 (`torch.profiler`): embedding-bag
backward 0.6 ms (PyTorch sorts the ~460k state tokens and reduces segments in every step), cuDNN GRU
0.6 ms, embedding-bag forward 0.2 ms, fused Adam 0.18 ms, the rest small kernels. Capturing alone could
not reach the target; the GPU work had to shrink too.

What was built (`rl/ppo.py`, sequence-mode `rl/model.py`):

- **Padded minibatches** (`model.pad_split`): every minibatch of an epoch is padded to one set of
  sizes per level (rows, options, entities, bags, state/event/option tokens, pointers), sizes rounded
  to 8 steps per octave so they repeat. Padded rows have weight 0 in the loss (means over real rows),
  one option each (finite logits), and padding points only at padding, spread round-robin: no
  `index_add` piles onto one row and no embedding bag holds all padded tokens (both serialise: a single
  junk bag made the step 22.8 ms). The GRU runs each minibatch as a (trajectories, steps) batch chosen
  per minibatch (4 sizes per octave; cuDNN time is ~steps x (1.9 + 0.06 x trajectories) µs).
- **Embedding weight gradients from a per-epoch transpose** (`model._TransposedBag`): tokens sorted by
  index once per epoch, cut into chunks of <= 32 tokens of one index; the backward is one more embedding
  bag over the bag gradients plus an `index_add` per chunk. (One bag per index walked the 10^4
  occurrences of common features serially: 7 ms per step.) 0.83 -> 0.53 ms per step.
- **The whole step in a CUDA graph** (`_StepGraphs`): forward, losses, backward, gradient norm, fused
  Adam (the clip factor passed as Adam's `grad_scale`, saving clip's pass over 17M gradients), the
  statistics sum. One graph per padded shape and GRU batch, replayed after one `copy_` each of the
  minibatch's int64 and float32 buffers; still one host sync per epoch. The first step of a fresh
  optimizer runs eagerly (Adam allocates its state). The graphs are dropped when the weights', Adam
  state's or hyperparameters' addresses/values change (`fingerprint`). New shapes grow by 5% to the
  elementwise max of the captured ones and a minibatch reuses a captured GRU batch costing <= 15% more,
  so captures stop after the first updates; captures use `capture_begin/end` on one side stream
  (`torch.cuda.graph` empties the allocator cache each time: 160 -> 48 ms per capture).
- **Inductor inside the graphs** (`PPOConfig.capture=2`, the default): the forward and losses are
  `torch.compile`d (default mode) and captured with the rest; fusion removes ~0.3 ms of small kernels
  per step. ~8 s of compiling once per process.
- `mode="padded"` runs the padded math eagerly, so the CPU tests cover it; CPU and the transformer
  trunk stay eager; `--ppo-capture 0` gives the old eager loop, `1` graphs without Inductor.

Step by step at minibatch 2048 (GPU 5 until 18:52, the before/after rows retaken on GPU 7):

| change | ms / step | s / epoch (322k) | s / 250k |
|---|---:|---:|---:|
| **before** (eager, `d90af0e`, GPU 7) | 4.00 | 0.609 | **0.474** |
| padded + graph, padded tokens in one junk bag | 22.8 | 3.47 | 2.70 |
| padded tokens spread over padded bags and 64 indices | 3.54 | 0.54 | 0.42 |
| padded pointers round-robin (no atomics on one row) | 3.44 | 0.52 | 0.41 |
| transposed embedding gradient, one bag per index | 6.95 | 1.06 | 0.82 |
| ... in chunks of 32 tokens | 3.24 | 0.49 | 0.38 |
| GRU batch per minibatch, clip folded into Adam, steady state | 2.41 | 0.42 | 0.33 |
| **after, `capture=1`**, steady state (GPU 7) | 2.4 | 0.384 | 0.299 |
| **after, `capture=2`** (default), steady state (GPU 7) | **2.0** | **0.328** | **0.255** |

"Steady state" excludes captures (they stop once shapes settle; a bench of 8 epochs on one data set
still captured 8 graphs, 0.36 s: 0.373 s/epoch with them). An epoch is ~34 ms of preparation
(structure, padding, transposes, one sort of the epoch's tokens) plus the replays. Remaining GPU time
per replay at `capture=1` (2.35-2.44 ms, 240 kernels, no gaps): GRU ~0.75 ms (cuDNN persistent kernels,
latency-bound over ~250 steps), embedding bags 0.42 ms, fused Adam 0.18 ms (17M parameters at ~2.7 TB/s,
the dense embedding rows included), gradient norm 0.03 ms, gathers/scatters 0.17 ms, small elementwise
~0.3 ms (most of which Inductor fuses). Peak device memory 6.7 -> 11.3 GiB at minibatch 2048.

`torch.compile` alone (no hand capture; forward + losses compiled, backward through AOTAutograd, clip
and Adam eager), on 32 padded minibatches of one shape: eager padded 3.48 ms/step, `mode="default"`
2.85, `mode="reduce-overhead"` 2.75 (9.4 s to compile and record), against 2.35-2.43 for the hand-
captured whole step and ~2.0 with both. Recompiles per shape would also cost seconds each.

Statistics (minibatch 2048, 1 warm-up + 4 epochs, same data and minibatch order): the unchanged code
twice gives pg_loss -0.00442/-0.00442, v_loss 0.01871/0.01876, entropy 0.98769/0.98780, approx_kl
0.00721/0.00710, clip_frac 0.0641/0.0608 (run-to-run CUDA noise); `capture=0`: -0.00441, 0.01869,
0.98775, 0.00715, 0.0610; `capture=1`: -0.00442, 0.01875, 0.98776, 0.00714, 0.0619; `capture=2`:
-0.00441, 0.01874, 0.98776, 0.00714, 0.0625. All within the noise. `explained_var` is computed before
the epochs, unchanged.

**Minibatch size** (s per epoch of 322k decisions, steady state, GPU 7):

| minibatch | before (eager) | `capture=1` | `capture=2` | s / 250k, `capture=2` |
|---:|---:|---:|---:|---:|
| 2048 | 0.602 (3.96 ms/step) | 0.384 | 0.328 | 0.255 |
| 8192 | 0.303 (7.8 ms/step) | 0.295 | 0.265 | 0.206 |
| 16384 | 0.278 (13.9 ms/step) | 0.294 | 0.269 | 0.209 |

At 8192 and above the step is GPU-bound (GRU and embedding bags scale with the batch) and the capture
gains little; the floor for this network is ~0.2 s per 250k decisions per epoch.

**Learning check** (full trainer, h128 entity, 40 iterations x 2048 games = 82k games, `--eval-every 10`,
seeds 0 and 1, 15 workers on cores 48-63, GPU 7, `capture=2`; `bench` = learner Jund vs the blue bot,
1000 games (95% CI about +-2.5 pp), `bo3` = 200 best-of-three matches (+-5-6 pp)):

| minibatch | lr | optimizer steps / iter | bench @30 | bench @40 (seed 0, 1) | bench bo3 @40 | eval bot jund / blue @40 | update_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 3e-4 | 460 | 0.098 | **0.161** (0.181, 0.142) | **0.125** | 0.20 / 0.65 | 1.50 |
| 8192 | 3e-4 | 122 | 0.070 | 0.138 (0.144, 0.132) | 0.070 | 0.06 / 0.56 | 1.30 |
| 8192 | 6e-4 | 122 | 0.096 | 0.156 (0.165, 0.147) | 0.117 | 0.13 / 0.58 | 1.29 |
| 16384 | 3e-4 | 62 | 0.047 | 0.099 (0.108, 0.090) | 0.053 | 0.08 / 0.56 | 1.32 |
| 16384 | 1.2e-3 | 62 | 0.082 | 0.143 (0.140, 0.146) | 0.105 | 0.09 / 0.62 | 1.34 |

Plainly: **at the same learning rate, larger minibatches learn slower** (16384 at 3e-4 reaches 61% of
the 2048 bench score after the same games; 8192 at 3e-4 86%, and 56% of its bo3 score). With the learning
rate raised, 8192 at 6e-4 is within seed noise of 2048 on every metric (bench 0.156 vs 0.161, bo3
0.117 vs 0.125); 16384 at 1.2e-3 is below 2048 on every metric (bench 0.143, bo3 0.105, @30 0.082 vs
0.098), each difference within two seeds' noise but all in the same direction. Caveat: 82k games is
the start of a run (a 490k-game run reached 0.52); later effects of the larger batch are not measured.
`update_s` (which also includes `_weights`/`_publish`/`_checkpoint`) shrinks only 1.50 -> 1.3 s: in
the trainer the bigger batch saves less than in the bench, and at 15 workers the rollout (2.8 s) bounds
the iteration anyway.

In the trainer at minibatch 2048 (iterations 4-20, pipelined, 15 workers, same GPU): `update_s` **2.18
s before -> 1.43 s after** (original code `d90af0e` vs this branch, ~235k decisions and ~450 steps per
iteration). Peak device memory over the 10 runs: 26.5 GB.

**Recommended `PPOConfig`: unchanged except `capture=2`** (`minibatch=2048`, `lr=3e-4`, `epochs=4`).
It reaches the target (0.256 s per 250k-decision epoch, from 0.474) without changing the math. 8192 at
6e-4 is the candidate if the update ever bounds the iteration again (0.206 s per 250k, learning within
noise here), but it needs a longer run before it becomes the default; 16384 is not recommended, and
8192/16384 at 3e-4 are clearly worse. Use `--ppo-capture 1` if Inductor's ~8 s compile per process or
its dependency is unwanted (0.30 s per 250k), `--ppo-capture 0` for the old eager loop.

**Two bugs the benchmark did not show, found by the first learning-check runs in the trainer:**

- *Device memory.* At minibatch 8192 the trainer ran out of memory after 26 updates (80 GB): every
  capture's warm-up ran on a new side stream and the allocator caches blocks per stream. Now one side
  stream serves every warm-up and capture, each padded shape has its own graph pool (evicting a shape
  frees it), and an update that captured empties the cache once at its end. Over 20 updates at 8192,
  reserved memory oscillates between 6 and 19 GiB without growing.
- *Illegal memory access after 5-33 updates* (at 2048 and 8192, with and without Inductor). The step
  graphs had baked in cuBLAS's per-stream workspace from the warm-up; `torch.cuda.empty_cache` (and
  the allocator's out-of-memory recovery) releases it, and later replays wrote to freed memory. The
  bench never emptied the cache between replays. Bisected in the trainer (40 iterations each):
  without the end-of-update flush, clean; with the workspaces dropped around the warm-up, a crash at
  the first update. Fix: drop the workspaces right before each capture, so the capture allocates its
  own inside the graph's pool, and right after (what Inductor's CUDA graph trees do). A test replays
  after dropping the workspaces, emptying the cache and refilling freed memory with junk; it fails
  with the same illegal access without the fix. After it: 40 clean updates at 2048 and at 8192, and
  the learning check below (10 runs x 40 updates). Every padded epoch of a 33-update run was also
  bounds-checked field by field (a scratch driver around `_padded_epoch`): no index out of range.

Tests: `tests/test_ppo_fast.py` (+10: padded pieces hold the split pieces and padding never points at
real items; the padded GRU layout; the padded update equals the eager one on the CPU for 4 configs
incl. early stop; CPU falls back to eager and refuses `mode="graph"`; on CUDA the captured update
(`capture` 1 and 2) matches the eager one in statistics and weights over two updates, reuses its
graphs, and recaptures after the optimizer state is replaced; replays survive an emptied cache).
Local (CPU, torch 2.11): 32 passed, 5 skipped; `tests/test_rl.py` passes. Box (CUDA, torch 2.14.1):
55 passed. `test_precomputed_structure_gives_the_same_outputs[entity-separate-gru-cuda]`
(not changed here) failed once in ~6 runs at atol 1e-5 on gradients (atomic-add order); it is flaky on
CUDA, not broken.

Open:
- The GRU is now the largest piece (~0.75 ms per step at 2048, cuDNN's per-step latency); bigger
  minibatches amortise it only up to ~8192.
- Updates that capture pay ~50 ms per graph (~16-20 graphs per padded shape) and a cache flush; in a
  long run that is the first few updates and whenever the data outgrows the shapes.
- `update_s` in `metrics.jsonl` includes `_weights`/`_publish`/`_checkpoint` and, pipelined, shares
  the GIL with the rollout thread (WS6); the bench numbers above are the update alone.
- Hidden > 128 (two packed GRUs eagerly) now runs padded in the graph; not measured.
- The cuBLAS fix uses `torch._C._cuda_clearCublasWorkspaces` (private, as in Inductor); if a torch
  upgrade drops it, `_clear_cublas_workspaces` silently does nothing and the emptied-cache test fails.
- Minibatch 8192 at lr 6e-4 deserves a longer run (500k games) before any default change.

### WS6 results

(pending)
