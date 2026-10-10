# Multi-deck interference: gradient conflict and scale mismatch (2026-10-10)

Evaluation only, no training. Tool: [`tools/diag_interference.py`](../../tools/diag_interference.py); raw results and the tool's full tables for every run: [`interference/`](interference/). Ledger entry: `20261010-interference-diag`.

Hypothesis from the research handoff: the per-deck specialists (`r8-jund-pilot`, `r8-blue-pilot`) beat the one generalist (`r7-lr075`) mainly because of **interference** (negative transfer) between decks that play very differently; the competing explanations are capacity (h256 is too small for six decks) and data share (a specialist sees 6x more samples of its own deck). Two cheap diagnostics at the checkpoints' own weights:

1. **Gradient conflict**: the PPO loss gradient of each deck, cosine similarities between decks per parameter group, against a noise ceiling.
2. **Scale mismatch**: advantages, returns, game lengths, batch shares and gradient norms per deck, and whether the global per-batch advantage normalisation lets one deck dominate.

## Verdict

**Interference is not supported where it can be measured, and cannot be measured where it would most plausibly live. Scale mismatch is mild. The overall answer is inconclusive, leaning against a first-order gradient conflict.**

- *Where the noise ceiling allows a reading* (the GRU, the value head, partly the option scorer) the six decks' gradients **point the same way**: noise-corrected cosines are positive for every usable deck pair (GRU, pure policy gradient: mean +0.77, 15 of 15 pairs positive, none negative; value head: mean +0.81, 15 of 15). Decks that play very differently do not pull the shared recurrent state in opposite directions at this checkpoint.
- *Where interference would most plausibly live* (the shared encoder and trunk, and the hashed embedding tables) the within-deck noise ceiling is 0.04 to 0.05 at 1,200 games per half: the gradient of 84k to 182k decisions is still pure noise there, so those cross-deck cosines say nothing. Reading them needs about 25x more games per deck (see "What would decide it").
- A positive alignment of the *expected* gradients does not rule out interference through capacity (shared weights that cannot represent six policies at once); it only says the first-order update directions do not oppose each other. Capacity and data share are not tested by this diagnostic.
- *Scale*: advantage sd differs 1.6x between decks (0.17 to 0.30), decisions per game 2.2x (70 to 152), batch share 11% to 25%; gradient-norm and G2 shares stay within 0.68x to 1.33x of the batch share. The global advantage normalisation does not let a deck dominate the update; it mildly over-weights Blue and Affinity and under-weights Red.
- *The one striking finding is about the value head*, not the cross-deck geometry: the generalist has a strong, persistent value-head gradient on **every** deck (within-deck cosine 0.69 to 0.97), while the Jund specialist has none on Jund (within-deck cosine -0.14, gradient norm 0.018 against the generalist's 0.110) and keeps it on the deck it never trained (Blue, 0.100). That fits "the generalist's value function is not converged on any deck and 3M games of one deck fix that deck": data share or under-training, not conflict, since the six decks' value gradients agree in direction.

## Method

For each of the six decks, `--chunks 4` independent chunks of 600 games (42k to 91k recorded decisions per chunk, 0.17M to 0.37M per deck). In a chunk the learner plays the deck in its normal pairings (every matchup of the r7 mix that seats it: five opponents at weight 2, the mirror at weight 1), the other seat plays the **same checkpoint** (self-play; only the learner's seat is recorded, so every recorded decision is the deck's), sampled play, 20% of the games postboard (`--postboard-frac 0.2`), seat swap as in the trainer, no variants, no shaping (`gamma .995`, `lam .95`, value clamp 1, `max_turns` 100: the trainer's defaults).

The gradient is **the repo's own**: `ppo.ppo_update` (eager, one epoch, clipping off, FP32) with a stand-in optimizer that adds up each minibatch's gradient weighted by its rows instead of applying it, so it is the full-batch mean gradient of `pg + 0.5 * v_loss - 0.01 * entropy` at the checkpoint (first step of an update, ratio 1, clip inactive). `ppo._losses` is wrapped only to read the row count and, for the global normalisation, to apply an affine change to the advantages. Tests (`tests/test_diag_interference.py`) check the sum against one full-batch step and the affine transform.

*Advantage normalisation.* The trainer normalises once per batch over every deck. `--norm global` reproduces that: each deck's batch-normalised advantages are mapped to `(a - M) / S`, with `M`, `S` the mean and sd of a batch that mixes the six decks by their share of a training batch's decisions. All numbers below use it; the per-deck normalisation only rescales the policy gradient per deck (directions barely change) and was dropped from the long runs to halve the gradient cost (the tool has it: `--norm global,per_deck`).

*Noise ceiling.* The four chunks of a deck are put into two halves in all three balanced ways. Per parameter group:

| quantity | meaning |
|---|---|
| **within(A)** | cosine of the two halves of deck A: the ceiling. For independent noise it equals `|s|^2 / (|s|^2 + |n|^2)` of a half-batch gradient (`s` signal, `n` noise) |
| **cross(A, B)** | mean cosine of a half of A with a half of B |
| **corrected** | `cross / sqrt(within(A) * within(B))`: the cosine of the noise-free gradients under independent noise. Only shown as a reading where both ceilings are at least 0.10 (below that it divides by noise and leaves [-1, 1]) |

A cross-deck cosine of 0.1 against ceilings of 0.1 is perfect alignment; against ceilings of 0.9 it is nearly orthogonal.

Parameter groups (35.2M parameters): `emb_tables` (the three hashed embedding tables, 33.6M), `trunk` (entity encoder, attention, trunk MLP), `gru`, `option_scorer` (option MLP, pointer, scorer), `value_head` (a 257-parameter linear layer on the GRU output; the value loss also reaches the trunk, GRU and embeddings through it), and `embedding+trunk` (the first two together).

**Measured on** h100-private (2x Intel Xeon Platinum 8462Y+, 64 cores, shared with running r8 trainers), CPU only: `nice -n 19 taskset -c 60-63`, `OMP_NUM_THREADS=1`, 4 rollout workers, 4 torch threads for the gradients, native engine, CPU torch 2.14.1, Python 3.11. Wall time on those four cores: 32 min for the generalist (rollouts 12 min, gradients 20 min), 20 min for its policy-gradient-only repeat (rollouts cached), 14 min and 15 min for the two specialists. Code: this PR on `84ccb08`, run from a git-bundle checkout next to `~/mtg-ml-eval/code` (`~/mtg-ml-eval/diag-code`, engine identical to the code checkout's native build); outputs in `~/mtg-ml-eval/diag/`.

```
cd ~/mtg-ml-eval/diag-code   # per run, as launched by ~/mtg-ml-eval/diag/run_all.sh and run_pg.sh
OMP_NUM_THREADS=1 nice -n 19 taskset -c 60-63 ~/mtg-ml-eval/code/.venv/bin/python tools/diag_interference.py \
  --checkpoint ~/mtg-ml-eval/ckpt/r7-lr075-policy.pt --decks jund_wildfire,mono_blue_terror,red_madness,grixis_affinity,elves,tron \
  --chunks 4 --games-per-chunk 600 --workers 4 --threads 4 --norm global --cache ~/mtg-ml-eval/diag/cache-r7-lr075 --out r7-lr075.json
# policy gradient only: add  --vf-coef 0 --ent-coef 0   (same rollouts, from the cache)
# specialists: --checkpoint r8-jund-pilot-final.pt / r8-blue-pilot-final.pt  --decks jund_wildfire,mono_blue_terror
```

Caveats: the diagnostic plays **100% self-play** and records one seat, where training mixed in 50% pool opponents; FP32 here, BF16 in the trainer; only the first step of an update (ratio 1) is measured, not the multi-epoch drift; deck shares in a training batch are estimated from these rollouts' decisions per game under the mix weights (self-play and pool games assumed to record alike); the sd in the tables is the spread over the three splits, not a confidence interval (they share chunks).

## Scale mismatch (diagnostic 2), `r7-lr075`

2,400 games per deck. "Returns" are the value targets (the value loss regresses on them: the same numbers). "Value" is the clamped network value GAE used. "dec/game" counts the learner's recorded decisions; "turns" is the engine's turn counter (each player's turn counts). Win rate: the learner on that deck against the same checkpoint on the opposing deck of the mix.

| deck | dec/game | turns/game | batch share (decisions) | batch share (games) | adv mean ± sd | return (value target) mean ± sd | value mean ± sd | v_loss | explained var | win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| jund | 152 | 19.8 ± 7.2 | 0.252 | 0.167 | +0.009 ± 0.185 | -0.107 ± 0.551 | -0.116 ± 0.525 | 0.0171 | 0.89 | 0.429 |
| blue | 70 | 16.5 ± 5.8 | 0.113 | 0.167 | +0.008 ± 0.302 | -0.111 ± 0.657 | -0.119 ± 0.588 | 0.0459 | 0.79 | 0.503 |
| red | 91 | 13.8 ± 4.6 | 0.145 | 0.167 | +0.005 ± 0.254 | -0.027 ± 0.598 | -0.032 ± 0.541 | 0.0324 | 0.82 | 0.586 |
| affinity | 117 | 17.4 ± 5.7 | 0.190 | 0.167 | +0.013 ± 0.217 | -0.159 ± 0.578 | -0.173 ± 0.537 | 0.0237 | 0.86 | 0.458 |
| elves | 99 | 14.8 ± 4.6 | 0.156 | 0.167 | +0.007 ± 0.233 | -0.072 ± 0.636 | -0.079 ± 0.597 | 0.0271 | 0.87 | 0.518 |
| tron | 87 | 16.7 ± 6.4 | 0.143 | 0.167 | +0.002 ± 0.243 | +0.159 ± 0.638 | +0.157 ± 0.590 | 0.0295 | 0.85 | 0.546 |

The training batch's advantages (decks mixed by decision share) have mean +0.008 and sd 0.233, so under the global normalisation Blue's advantages are scaled by 0.302 / 0.233 = 1.30x and Jund's by 0.80x relative to a per-deck normalisation. The mean return differs by deck (Tron +0.16, Affinity -0.16): the matchup-weighted strength of each deck against the checkpoint itself.

Gradient of the full loss (mean over a deck's decisions, global normalisation), and who drives the batch gradient `G = sum_d share_d * g_d`:

| deck | |g| whole | |g| emb_tables | gru | trunk | option_scorer | value_head | batch share | norm share | G2 share | G2 share / batch share | cos(g_deck, G) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jund | 0.144 | 0.027 | 0.029 | 0.076 | 0.035 | 0.110 | 0.252 | 0.200 | 0.242 | 0.96 | 0.79 |
| blue | 0.248 | 0.063 | 0.056 | 0.182 | 0.078 | 0.123 | 0.113 | 0.154 | 0.135 | 1.19 | 0.57 |
| red | 0.165 | 0.041 | 0.055 | 0.123 | 0.055 | 0.068 | 0.145 | 0.132 | 0.099 | 0.68 | 0.49 |
| affinity | 0.200 | 0.038 | 0.037 | 0.102 | 0.046 | 0.157 | 0.190 | 0.209 | 0.252 | 1.33 | 0.79 |
| elves | 0.183 | 0.050 | 0.043 | 0.124 | 0.053 | 0.105 | 0.156 | 0.157 | 0.158 | 1.01 | 0.66 |
| tron | 0.189 | 0.051 | 0.043 | 0.143 | 0.063 | 0.083 | 0.143 | 0.148 | 0.114 | 0.80 | 0.50 |

Norm share = `share_d * |g_d| / sum`; G2 share = `share_d * <g_d, G> / |G|^2` (sums to 1). Blue has 1.7x Jund's gradient norm but 0.45x its decisions, so the two roughly cancel. **No deck dominates**: every deck's part of the update is within 0.68x to 1.33x of its batch share. Most of each deck's gradient norm is noise (see the ceilings), which is why it tracks `1 / sqrt(decisions)` and the advantage sd rather than anything about the deck.

## Gradient conflict (diagnostic 1)

Summary over the 15 deck pairs, half level (1,200 games per half), `r7-lr075`. "Pairs usable": both within-deck ceilings at least 0.10.

Full loss (policy + 0.5 value + entropy):

| group | ceiling mean (min..max) | cross mean | corrected mean | pairs usable | of them negative |
|---|---|---|---|---|---|
| emb_tables | 0.050 (0.003..0.174) | +0.022 | n/a | 0 of 15 | 0 |
| gru | 0.246 (0.073..0.501) | +0.127 | +0.60 | 6 of 15 | 0 |
| trunk | 0.037 (-0.031..0.210) | +0.007 | n/a | 0 of 15 | 0 |
| option_scorer | 0.106 (0.044..0.251) | +0.060 | +0.59 | 1 of 15 | 0 |
| value_head | 0.874 (0.694..0.967) | +0.706 | +0.81 | 15 of 15 | 0 |
| embedding+trunk | 0.039 (-0.025..0.207) | +0.009 | n/a | 0 of 15 | 0 |

Pure policy gradient (`--vf-coef 0 --ent-coef 0`, same rollouts), which removes the entropy and value terms as a common-mode confound:

| group | ceiling mean (min..max) | cross mean | corrected mean | pairs usable | of them negative |
|---|---|---|---|---|---|
| emb_tables | 0.064 (0.005..0.189) | +0.029 | n/a | 0 of 15 | 0 |
| gru | 0.388 (0.136..0.612) | +0.275 | +0.77 | 15 of 15 | 0 |
| trunk | 0.046 (-0.024..0.212) | +0.011 | n/a | 0 of 15 | 0 |
| option_scorer | 0.134 (0.060..0.280) | +0.079 | +0.62 | 3 of 15 | 0 |
| embedding+trunk | 0.048 (-0.021..0.209) | +0.013 | n/a | 0 of 15 | 0 |

The two matrices that carry a reading. Diagonal (bold) = noise ceiling, above = cross-deck cosine, below = noise-corrected cosine.

GRU, pure policy gradient:

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.581** | 0.267 | 0.412 | 0.460 | 0.413 | 0.187 |
| blue | 0.90 | **0.153** | 0.227 | 0.216 | 0.216 | 0.113 |
| red | 0.69 | 0.74 | **0.612** | 0.371 | 0.396 | 0.164 |
| affinity | 0.96 | 0.87 | 0.75 | **0.399** | 0.353 | 0.162 |
| elves | 0.81 | 0.83 | 0.76 | 0.84 | **0.445** | 0.174 |
| tron | 0.66 | 0.79 | 0.57 | 0.69 | 0.71 | **0.136** |

Value head, full loss:

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.960** | 0.781 | 0.621 | 0.914 | 0.873 | 0.656 |
| blue | 0.86 | **0.864** | 0.644 | 0.789 | 0.749 | 0.642 |
| red | 0.76 | 0.83 | **0.694** | 0.687 | 0.607 | 0.522 |
| affinity | 0.95 | 0.86 | 0.84 | **0.967** | 0.815 | 0.590 |
| elves | 0.91 | 0.83 | 0.75 | 0.85 | **0.949** | 0.697 |
| tron | 0.74 | 0.77 | 0.70 | 0.67 | 0.80 | **0.808** |

Reading:

- **GRU**: ceilings 0.14 to 0.61 (Red, Jund, Elves, Affinity clearly above noise, Blue and Tron barely). The cross cosines sit at 0.1 to 0.46, below the ceilings but of the size the ceilings predict: corrected 0.57 to 0.96. Every pair is positive. The recurrent core is being pushed the same way by all six decks.
- **Value head**: the cross cosines (0.52 to 0.91) are close to the ceilings (0.69 to 0.97), corrected 0.67 to 0.95. The decks' value errors agree in direction; the lower cross than within values are deck-specific offsets (the mean return differs by deck).
- **Option scorer**: ceilings 0.04 to 0.28; in the pure policy gradient only Jund, Red and Elves clear 0.10 (3 pairs). Corrected 0.40 to 0.95, all positive, but weak evidence. The full per-group matrices are in the `interference/*.md` files.
- **Trunk, embedding tables, their union**: ceilings -0.03 to 0.21 (Red is the exception at 0.2). The expected policy gradient there is small next to the noise of 1,200 games, so the cross cosines (0.00 to 0.04) are noise against noise. Not readable. (For the hashed tables the decks also touch mostly disjoint rows, so near-orthogonality is expected by construction and says little about interference.)
- Chunk level (600 games per piece, in the JSON): the GRU ceilings fall from a mean of 0.39 to 0.25 (0.07..0.43, pure policy gradient), as `s / (1 + s)` with `s` proportional to the sample size predicts; so the ceilings are limited by the sample size, not a fixed property of the net.

## Contrast: the per-deck fine-tunes

Same protocol, two decks only (Jund and Blue, 2,400 games each, full loss, global normalisation over the two decks), same parent as `r7-lr075`. Jund is the deck `r8-jund-pilot` was fine-tuned on (the six Jund pairings, 3M games); Blue the deck `r8-blue-pilot` was fine-tuned on.

| checkpoint | deck | value_head norm | value_head ceiling | GRU ceiling | option_scorer ceiling | trunk ceiling | v_loss | explained var | cross(jund, blue) GRU / value |
|---|---|---|---|---|---|---|---|---|---|
| `r7-lr075` | jund | 0.110 | 0.96 | 0.29 | 0.13 | -0.03 | 0.0171 | 0.89 | +0.101 / +0.781 |
| `r7-lr075` | blue | 0.123 | 0.86 | 0.07 | 0.07 | 0.01 | 0.0459 | 0.79 | |
| `r8-jund-pilot` | **jund (trained)** | **0.018** | **-0.14** | 0.27 | 0.02 | -0.04 | 0.0143 | 0.90 | +0.201 / +0.142 |
| `r8-jund-pilot` | blue | 0.100 | 0.72 | 0.20 | 0.04 | 0.13 | 0.0376 | 0.80 | |
| `r8-blue-pilot` | jund | 0.163 | 0.98 | 0.73 | 0.03 | 0.05 | 0.0189 | 0.87 | +0.246 / +0.626 |
| `r8-blue-pilot` | **blue (trained)** | 0.063 | 0.72 | 0.18 | 0.03 | 0.02 | 0.0334 | 0.81 | |

(The GRU cross for `r7-lr075` is the Jund-Blue entry of the six-deck matrix, full loss.) Fine-tuning on Jund drives the value-head gradient on Jund to nothing (norm 0.110 to 0.018, ceiling 0.96 to -0.14) and leaves it on Blue (the value head still disagrees consistently with the Blue returns); the Blue fine-tune halves Blue's value-head norm but does not remove it (0.123 to 0.063) and leaves Jund's large. The policy-side ceilings (GRU, scorer, trunk) do not shrink with specialisation: a policy that is trained 3M games on Jund still has a consistent GRU gradient on Jund (0.27), so the persistent GRU signal is not specific to the generalist.

## What the diagnostics say about the hypothesis

- **Interference (negative transfer)**: no evidence of conflicting first-order gradients in the groups the data can read (GRU, value head), and no reading in the trunk and embeddings. A generalist stuck at a joint stationary point because the decks pull against each other would show a ceiling well above noise and **negative** cross cosines; the usable pairs show positive corrected cosines of +0.6 to +0.95 and none negative. What this cannot exclude: conflict that only shows in the shared trunk and embeddings, and capacity limits (no cosine would show them).
- **Data share / under-training of the generalist**: favoured by the value-head contrast. The generalist's value head still has a strong, deck-independent, consistently signed gradient on every deck after 20M games, and 3M games on one deck remove it for that deck. That is what 6x more samples per deck would do; it is also what a capacity limit would look like in the value head. The 100% self-play of this diagnostic against training's 50% pool games may contribute, but the Jund pilot, trained on the same mix, has no such gradient on Jund in the same diagnostic.
- **Capacity**: not separable from data share here. A capacity test needs a wider generalist at equal games or a per-deck game budget sweep (below).
- **Scale mismatch**: mild. Advantage sd differs 1.6x, decisions per game 2.2x, but no deck's share of the update leaves 0.68x to 1.33x of its batch share, and the extremes are mutually offsetting (Blue: large advantages, few decisions). Per-deck normalisation or per-deck loss weights are unlikely to be the lever. The one visible imbalance is the value loss: Blue's `v_loss` is 2.7x Jund's (0.046 against 0.017), explained variance 0.79 against 0.89.

## What would decide it

1. **A bigger noise ceiling for the trunk and embeddings**: the trunk ceiling is 0.04 at 1,200 games per half, i.e. signal-to-noise `s = 0.04`; a ceiling of 0.5 needs `s = 1`, about 25x the games: roughly 30k games per half, 60k per deck, 360k games for six decks (about 40M decisions). On these four CPU cores that is 4 h of rollouts and about 12 h of gradients (1 ms per decision with 4 threads); a GPU should cut the gradients to minutes. Proposal: rerun after 12:00 on a GPU box with the tool extended by `--device` (the tool is CPU only today) and 4 chunks of 15k games per deck; the chunk files then also give the scaling curve of the ceiling.
2. **Per-deck game budget**: fine-tune the generalist on one deck with the games of the generalist's own share of it (3.3M games of Jund = 1/6 of 20M, in the same mix) against 3M games of Jund only; if the share-matched run reaches the specialist's strength, data share explains the gap, otherwise interference or capacity does.
3. **Capacity**: the same generalist at h384 or h512 at equal games (the entity net scales: see `docs/training-256-attention.md`).

## Reproducing

`python tools/diag_interference.py --report docs/experiments/interference/r7-lr075.json` prints every table of a finished run (the JSON holds the chunk-level cosines too). `pytest tests/test_diag_interference.py` runs the tool end to end on a tiny network.
