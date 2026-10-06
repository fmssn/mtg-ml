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
3. **NUMA is backwards, but it does not cost anything yet.** `nvidia-smi topo -m`: GPUs 0-3 are local
   to cores 0-15 (node 0), GPUs 4-7 to cores 32-47 (node 2). Today's run pins the trainer and workers
   to 0-31 and uses GPU 4. Measured clean (12 iterations each, no profiler): cores 0-31 + GPU 4 gives
   2.42 s per iteration, cores 32-63 + GPU 4 gives 2.44 s. At today's transfer volumes the sockets do
   not matter; the layout becomes hygiene for the server path (WS4), not a win on its own.
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

h128 entity, shared value, native engine, `tools/bench_rollout.py --games 2048 --rounds 4`. The final numbers
were taken on cores 32-63 while they were otherwise idle (a foreign training run held cores 0-31 the whole time,
so there is no 31-worker number on 0-31): 27 workers on cores 33-59, server on core 32, GPU 4
(GPU-3e32a273, node 2). "Old server" is commit d90af0e run from a scratch copy on the same cores.

| decisions/s, 27 workers | 25 pool opponents (26 policies) | 2 pool opponents (3 policies) |
|---|---:|---:|
| local CPU inference (8 OMP threads, cores 32-59) | 158.5k | 178k |
| old server | 13.5k | 50.6k |
| **new server** | **374k (2.36x local)** | **422k (2.38x local)** |

15 workers on node 2 alone (cores 33-47, server on 32): server 254k, local 90k (2.8x). The 31-worker baseline
taken before the work on cores 0-31, GPU 0 (box idle): local 176k, old server 14.2k (pool 25) and 49k (pool 2).

Per batch, in isolation (`tools/profile_server.py`, 26 policies, real decisions): the old server spent ~36 ms on
a batch of ~510 decisions from 26 policies (one forward and one `.cpu()` per policy) and ~8 ms with 3. Now the
host spends ~0.1 ms per batch (numpy header, one copy, one graph replay) and the GPU 0.54 ms for ~200
decisions, 0.72 ms for ~1,000 (padded to 1,488 rows), independent of the number of policies: up to ~1.07M
decisions/s per server. In the 27-worker runs the server answers a batch of ~290 decisions 0.63 ms after
launching it; its host work is ~15% of the wall time, and a batch is on the GPU ~85% of it (~1,400 batches per
1.06 s round), so the GPU's latency per batch is what the workers wait on.

What changed (`rl/inference.py`, new `rl/stacked.py`):

1. **Transport without pipes or pickling.** Each server owns a control block (request headers, reply tickets)
   and a data block (request and reply area per worker group), shared memory registered with CUDA and mapped
   into the GPU's address space. A worker writes its decisions and header, publishes a ticket and spins on its
   reply ticket; the server polls all tickets with one numpy compare. The graph reads the requests straight from
   host memory and writes (action, log-prob, value) straight into the reply areas. The dry run (no network) at
   31 workers was 225k decisions/s with the old transport, the server's Python parse and reply alone ~1.8 ms
   per batch.
2. **One forward for all policies** (`PolicyStack`): embedding tables stacked into one table per kind with a
   block per policy and one zero row (separators, pointers, padding look it up, which is what removing them
   does), dense weights stacked (slots, in, out). Rows are sorted by policy on the host, so entities and options
   are too, and every dense layer is one grouped matmul (Triton kernel over policy-sorted rows, fp32 IEEE; tiles
   16x32x32 for small batches: 3-10x faster than the first version). The learner is replaced in place.
3. **Sync-free step structure.** Every size comes from the request headers (the worker counts its entity
   separators with numpy); bags, entity rows and pointers are derived with fixed-shape ops (searchsorted,
   cumsum differences, embedding bags over the entity vectors with per-sample weights for pointers). No
   `.item()`, no boolean indexing, no atomics on the hot path.
4. **CUDA graphs + torch.compile.** One graph per size level (rows 32 * 1.5^k; other sizes in fixed ratios to
   rows), ~11 per run; padding goes to extra 64-token bags. `torch.compile` (default on CUDA) fuses the ~300 small
   kernels: 1.25 -> 0.72 ms GPU per 1,000-decision batch. Sampling is Gumbel-max with log-prob = score -
   logsumexp of the row's options.
5. **One server per GPU**: `InferenceServer(..., devices=["cuda:0", "cuda:1"])` shards workers in contiguous
   blocks; a server without `cpus` pins itself to its GPU's NUMA node minus the workers' CPUs (from
   `/sys/bus/pci/devices/<bus>/local_cpulist`). Checked with GPU 0 + GPU 4 and 14 workers (246k decisions/s,
   the same per worker as one server: at these worker counts one H100 is enough).

What did not pay: `torch._grouped_mm` (fp32 falls back to a host loop with a sync, 0.5 ms); `include_last_offset`
to drop padding from embedding bags (CUDA still sums the tail into the last bag, 2 ms per call); compiled
`index_add` (becomes a sort-based `index_put`, 1 ms per call, so scatter-free formulations instead); two CUDA
streams with concurrent batches (361k vs 374k at 27 workers, equal at 15; kept as `ServerConfig.streams`,
default 1); `--groups 3 --inflight 96` (+4%, defaults unchanged).

Exactness: `tests/test_inference.py` passes on the CPU and on the H100: server rollouts replay exactly for every
trunk (mlp and entity through the stack, transformer through the per-policy path), graphs with and without
`torch.compile` and the eager forward; the stacked forward matches per-policy `PolicyNet` forwards (log-prob,
value, hidden state within 1e-5 for 5 variants, CPU and CUDA); the step structure puts the same tokens in each
bag and the same pointers as `model.structure`; Gumbel-max draws follow the softmax. The verification suite
(`test_rl`, `test_inference`, `test_rollout_stream`, `test_ppo_fast`, `test_train_pipeline`): 125 passed on the
box.

Where the time goes now (27 workers, pool 25): workers spend ~75% of a job on the CPU (engine, featurize,
Python glue: ~41 us per decision) and ~25% waiting for the server's ~0.6 ms round trip; a 2048-game round takes
~1.06 s of which the median job runs 0.82 s, so ~20% of the round is outside the jobs (start, tail, merging
samples in `run_specs`). Both are WS7 territory (Rust lockstep loop, continuous collection).

Open: the 31- and 63-worker numbers on an idle box; a trunk with real compute (transformer) in the stack; the
GPU floor of ~0.5 ms per batch (kernel count; a hand-fused gather + structure kernel would cut it); the server
holds 2.2 GB per 26 h128 policies (32 slots preallocated, grows by doubling, which recaptures the graphs).

### WS5 results

(pending)

### WS6 results

Box, cores 32-59 (`taskset`), GPU 4, 27 workers, h128 entity, local inference, 2048 games per
iteration, 12 iterations, means over iterations 3-12. Old code (`d90af0e`) and new code were run back
to back several times (the box is shared; other runs used cores 0-31 and 60-63 meanwhile).

| | update_s | µs update / decision | wall_s | wait_s |
|---|---:|---:|---:|---:|
| old, `--pipeline 1` (3 runs) | 2.52-2.58 | 9.98-10.33 | 2.54-2.60 | 0.00 |
| **new, `--pipeline 1`** (4 runs) | **2.21-2.26** | **9.03-9.15** | **2.22-2.27** | 0.00-0.01 |
| old, `--pipeline 0` (update alone) | 1.94 | 7.91 | 4.07 | |
| new, `--pipeline 0` | 1.93 | 8.05 | 3.97 | |
| new, `--pipeline 0`, trainer pinned to 1 core, 1 thread | 2.04 | 8.28 | 4.17 | |
| new, `--pipeline 1`, 12 workers on node 3 only (48-59) | 1.93 | 7.75 | 3.35 | 1.43 |
| new, `--pipeline 1`, 14 workers on node 2 only (33-46) | 2.18 | 8.71 | 3.11 | 0.93 |

- **Collection in its own process** (`rl/collect.py`, `--collector process`, the default): the pool
  belongs to a collector process that streams the games, merges the results and parks the merged batch
  in one shared-memory block; the trainer maps it (no copy of the samples; the float fields are one
  memcpy each) and unmaps it after the update. A test checks that the mapped batch trains bit for bit
  like the original. py-spy of the trainer process while pipelining, after: `ppo_update` 68% of its
  samples, `_publish` (the 68 MB policy file) 2%, the checkpoint thread 4% (start and end only); the
  rollout thread, `unpark`, the per-iteration Adam-state copy and `latest.pt` pickling are gone (before:
  `ppo_update` 25%, rollout thread 14%, `_save` 6%, `_cpu` 3%). The collector is pinned to the workers'
  cores off the trainer's node (~1%, kept).
- **The update is 12-13% faster pipelined and the iteration with it (2.58 -> 2.24 s), but still
  ~12% slower than alone.** The rest is not the GIL: it is the trainer's NUMA node. With the workers
  only on another node the pipelined update runs exactly as fast as alone (1.93 s, 12 workers on
  48-59), with 14 workers on the trainer's node it is 2.18 s; core clocks stay at 3.4-3.6 GHz either
  way (sampled from `scaling_cur_freq`), so it is the node's shared L3 slice / memory controller
  (sub-NUMA clustering, 4 nodes of 16). Two trainer cores and threads (26 workers) did not help (2.28 s).
  The +-5% criterion is met when the trainer's node carries no workers; on a 28-core budget that
  costs more rollout than it saves update, so the measured config keeps 15 workers on node 2. The
  default layout fills the trainer's node with workers last, so on the whole box (63 cores, 31-48
  workers) the trainer's node stays free.
- **Evaluation off the loop**: `--eval-every 5`, default eval sizes (~2,000 games per evaluation).

  | | wall_s, no eval | wall_s, eval every 5 | cost per iteration |
  |---|---:|---:|---:|
  | old (inline, 27 workers, ~2.5 s blocked per evaluation) | 2.58 | 3.08 | +0.50 s (+19%) |
  | new, 27 workers, evaluation shares their cores niced (4 eval workers; 2 runs) | 2.22-2.26 | 2.23-2.32 | +0.00-0.06 s (<= 3%) |
  | new, 24 workers + 3 evaluation cores | 2.27 | 2.37 | +0.10 s (+4%) |

  The evaluation process reads the policy file of the iteration it evaluates (pinned against pruning),
  takes 5-16 s and lands 2-7 iterations later in that iteration's row (`eval_s`, `eval_lag`). The
  remaining cost is again the trainer's node: the 3 dedicated evaluation cores are on it. Benchmark
  values match the inline evaluation (0.001-0.007 at iterations 5 and 10 in both).
- **Checkpoint cost**: `latest.pt` (204 MB with Adam state) every `--checkpoint-every` iterations
  (default 10) and at the end; the weights-only policy file stays per iteration. Resume tested in
  `tests/test_train_pipeline.py` (killed between checkpoints, both pipeline modes: same games as an
  uninterrupted run, lost rows moved to `metrics-dropped.jsonl`, lost policy files and snapshots removed)
  and on the box: SIGKILL of the trainer at iteration 8 with `--checkpoint-every 5` left no process
  behind (collector, evaluator and their 31 workers exit on the closed pipe within 20 s), the restart
  resumed at 5, dropped rows 6-8 and finished 12 iterations with evaluations merged.
- **Layout** printed at start (box, `taskset -c 32-59`, 27 workers):
  `cpu layout (GPU NUMA node 2): trainer 32 | workers 33-59 | collector 48-59 | evaluation 33-59 shared with the workers, niced`.
  With 24 workers: `trainer 32 | workers 36-59 | collector 48-59 | evaluation 33-35`.
  `--trainer-cpus`, `--worker-cpus`, `--eval-cpus` override; the GPU's node comes from
  `torch.cuda.get_device_properties` (PCI ids) and sysfs.

Open: the continuous-queue mode (`--pipeline 2+`) is not built: the loop waits 0.00-0.01 s per
iteration now, so at this worker count it has nothing to remove; it matters once the server makes the
rollout much faster than the update. An evaluation in flight when the trainer dies is lost (its row stays
without `eval/*`). The evaluation process uses local CPU inference even with `--inference server`.
`metrics.jsonl` is rewritten (through `os.replace`) at every merged evaluation, so `tail -f` readers must
reopen it. torch's global rng is not in `latest.pt`, so a resumed run plays the same games but samples
different actions than an uninterrupted one (as before).
