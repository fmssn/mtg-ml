# Research ideas for improving the model (2026-10-10)

Candidate improvements drawn from the RL literature, collected after the
[generalized Pauper review](generalized-pauper-review.md). They exclude what the
review already covers (oracle guiding / PerfectDou / Suphx, PSRO, R-NaD /
DeepNash, Expert Iteration, Gumbel search). None of them has been run. Each needs
a ledger entry with a matched control before it is adopted
([recording requirements](experiments/README.md)).

Citations were written from memory and have not been checked against the papers
yet; verify before quoting results from them.

Ordered by fit with the current state of the project; the cheapest useful test is
named for each.

## 1. Distil the per-deck pilots into the generalist

- **Papers:** Rusu et al. 2016, *Policy Distillation*; Teh et al. 2017, *Distral*;
  Schmitt et al. 2018, *Kickstarting Deep Reinforcement Learning*.
- **Idea:** during PPO on the six-deck mix, add a KL term from the generalist's
  policy to the pilot of the deck it is playing, and anneal it to zero
  (Kickstarting). No separate supervised phase.
- **Why here:** the r8/r9 pilots beat the generalist on their own decks, and the
  [interference diagnostics](experiments/interference.md) point at data share or
  under-training per deck rather than gradient conflict.
- **Why it differs from more training:** see below.
- **Test:** fine-tune the generalist with the KL term against a matched
  continuation without it; compare with the trust test and the pilot head-to-head
  matrix (`tools/eval_matrix.py`). If the gap to the pilots does not close, the
  deficit is more likely capacity, which distillation cannot fix.

### Why distillation is not the same as training longer

1. **Much denser signal.** PPO learns from the game result: one win/loss per
   game, spread over 70–150 decisions, plus a noisy critic. Distillation gives a
   target at every decision: the pilot's full probability distribution over the
   options, including how much worse each alternative is. Training is
   supervised, so it is far less noisy than credit assignment from a game
   result.
2. **It reaches moves the generalist does not sample.** RL only improves moves
   the policy actually tries. If the pilot learned a line to which the
   generalist gives near-zero probability, more PPO rarely discovers it. The KL
   term pulls probability onto it directly, wherever the pilot plays it.
3. **It reuses compute already spent.** Each pilot spent millions of games on
   one deck; the generalist gets about a sixth of its data per deck. Training
   the generalist longer has to rediscover that with its smaller share.
   Distillation copies the result.
4. **A fixed target.** Self-play targets move as the opponent pool changes; a
   frozen pilot does not, so this part of the update is stable.

Distillation alone cannot make the student better than its teachers; annealing
the KL term lets PPO take over again once the student has caught up. Why keep
one model at all: transfer to new lists and cards, and one network to serve.

## 2. Self-imitation of the model's own good moves

- **Paper:** Oh et al. 2018, *Self-Imitation Learning*.
- **Idea:** keep past (state, action) pairs whose outcome beat the critic's
  estimate and add an imitation loss on those only.
- **Why here:** a cheap route to rare winning lines (the Krark-Clan Shaman +
  Toxin Analysis sweep) compared with search distillation
  ([rejected](experiments/ledger.md)). It only helps if the line is found
  sometimes, so it pairs with idea 3.

## 3. Restart training from interesting positions in the model's own games

- **Papers:** Ecoffet et al. 2021, *First return, then explore* (Go-Explore,
  Nature); Jiang et al. 2021, *Prioritized Level Replay*.
- **Idea:** start a small share of training games from saved positions chosen by
  large critic error or by game-review findings. `Game.copy()` already restores
  exact states; no human labels are needed.
- **Caveat (from the review):** replay the game's earlier decisions through the
  GRU so the model's memory matches the position.

## 4. Magnet regularisation for PPO in hidden-information games

- **Papers:** Sokota et al. 2023, *A Unified Approach to Reinforcement Learning,
  Quantal Response Equilibria, and Two-Player Zero-Sum Games* (Magnetic Mirror
  Descent, ICLR); Rudolph et al. 2025, *Reevaluating Policy Gradient Methods for
  Imperfect-Information Games*.
- **Finding:** PPO-style methods with a KL penalty towards a reference ("magnet")
  policy and well-tuned entropy match specialised game-theoretic algorithms on
  standard benchmarks.
- **Why here:** a few lines in `ppo.py` instead of an algorithm swap; targets
  cycling and exploitability, the same problem as the r9 anti-cycling mix.

## 5. Check for loss of plasticity in long fine-tune chains

- **Papers:** Nikishin et al. 2022, *The Primacy Bias in Deep Reinforcement
  Learning*; Lyle et al. 2023, *Understanding Plasticity in Neural Networks*;
  Dohare et al. 2024, *Loss of plasticity in deep continual learning* (Nature).
- **Why here:** most of the improvement happens in the first ~1M games
  ([plateau analysis](plateau-analysis.md)), and checkpoints are continued run
  after run (r7 → r8 → r9).
- **Test:** measure dormant units on current checkpoints (no training). If many
  are dead, try shrink-and-perturb or resetting the value head before the next
  fine-tune.

## 6. Phasic Policy Gradient for the critic

- **Paper:** Cobbe et al. 2021, *Phasic Policy Gradient*.
- **Idea:** train the value function in separate phases with more epochs and
  distil it back into the shared network without disturbing the policy.
- **Why here:** held-out explained variance stays around 0.4 and a bigger or
  separate critic did not help ([offline probes](experiments/ledger.md)); this
  is a different test from that one.

## 7. Population Based Training for learning rate and entropy

- **Paper:** Jaderberg et al. 2017, *Population Based Training of Neural
  Networks*.
- **Why here:** results depend a lot on learning rate (r7 comparison), and there
  are 8 GPUs per machine. PBT tunes during one run instead of launching one run
  per setting. It costs the most of the seven, so it comes after the cheaper
  ideas.

## Suggested start

Idea 1 builds directly on the pilots just trained; idea 5 needs no training.
