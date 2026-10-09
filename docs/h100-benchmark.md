# Training on `h100-private`: benchmark and plan (2026-10-05)

> Historical (2026-10-05): the box setup, GPU assignments and plan below are the state of that day; later work (the Rust engine, the inference server, the r7 campaign) superseded them. See the [documentation index](README.md) for current guides.

## Machine

64 cores of Intel Xeon Platinum 8462Y+, 2 TB RAM and 8× H100 80GB. The box is shared: ComfyUI runs on GPUs 2 and 3, and another GPU has 78 GB in use. Pin runs to free GPUs with `CUDA_VISIBLE_DEVICES`.

Setup: the repo is in `~/mtg-ml` (rsynced, no git), with a uv venv in `.venv` (Python 3.12, torch 2.14+cu130). The system Python there is 3.10, too old for this repo.

## Results (3 PPO iterations, evaluation off)

| Setup | Wall time | Decisions | Decisions/s, end to end | Per iteration (games + update) |
|---|---|---|---|---|
| Mac M3, 7 workers, defaults | 34.6 s | 106k | ~3.1k | 3.4–5.3 s + ~6 s |
| H100 box, defaults | killed after 10 min | — | — | — |
| H100 box, `OMP_NUM_THREADS=1` | 102.5 s | 108k | ~1.1k | 2 s + ~43 s |
| H100 box, `OMP_NUM_THREADS=8` | 54.9 s | ~105k | ~1.9k | — |
| H100 box, 8 threads, `--games-per-iter 2048` | 257 s | ~860k | ~3.4k | — |
| **H100 box, 8 threads, 2048 games, `--device cuda`** | **76.7 s** | **865k** | **~11k** | **4–6 s + 12–14 s** |

Findings:

- **Without a thread cap the run hangs.** torch spreads the PPO update in the main process over all 64 threads (5300% CPU) on tiny operations, while the game workers sit idle. `OMP_NUM_THREADS` has to be set.
- **Game play scales well.** 63 workers reach 45–65k decisions/s at 2048 games per iteration (32 games per worker).
- **The update is the bottleneck on CPU.** On the GPU with large batches the box is about 4× faster than the M3. The remaining 12–14 s update is probably mostly Python batch building (`collate`, padding) in the main process. Not profiled yet.
- **2048 games per iteration changes training dynamics** (8× more data per update), so the PPO hyperparameters need retuning.
- **The model is tiny** (about 17M parameters, almost all in hashed embeddings), so the GPU is almost idle. The Python engine (about 60k decisions/s on 64 cores) is the hard limit. Update: with the Rust engine (`--engine native`, [native-engine.md](native-engine.md)) the engine side of a rollout worker is about 13× cheaper (3.95M decisions/s for featurize + step on 64 pinned cores), so CPU inference in the workers is now the limit.

Recommended command:

```bash
OMP_NUM_THREADS=8 python -m mtg_ml.rl.train --run runs/ppo1 --iterations 200 --games-per-iter 2048 --device cuda
```

## Next steps, by expected gain

1. ~~**Faster engine:** PyPy in the worker processes, then a Rust port checked against the Python engine by replaying seeds.~~ Done: PyPy gives 2-3× but cannot host torch workers; the Rust port gives 12.6-21.6× per core with identical games ([native-engine.md](native-engine.md)).
2. **Search plus distillation:** tree search (MCTS) guided by the network, with the policy trained on what the search picks. Can be built on the current engine, but only becomes practical at scale after step 1.
3. **Central GPU inference server and a bigger network**, a transformer over the cards and objects on the board.
4. **Overlap game play with the update, and build batches in the workers:** about 1.5–2× faster wall time, small change.
5. **Hyperparameter sweep on the free GPUs** (games per iteration, PPO epochs, learning rate), runnable now.
