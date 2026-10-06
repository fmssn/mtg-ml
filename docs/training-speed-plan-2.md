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

(pending)

### WS6 results

(pending)
