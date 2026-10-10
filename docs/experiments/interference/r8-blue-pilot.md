| deck | games | dec/game | turns/game | batch share (dec.) | batch share (traj.) | adv mean ± sd | return mean ± sd | value mean ± sd | v_loss | expl. var | win rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jund | 2400 | 175 | 19.5 ± 7.6 | 0.500 | 0.500 | -0.019 ± 0.191 | -0.039 ± 0.527 | -0.019 ± 0.510 | 0.0189 | 0.87 | 0.386 |
| blue | 2400 | 95 | 16.8 ± 5.8 | 0.500 | 0.500 | -0.008 ± 0.258 | -0.109 ± 0.592 | -0.101 ± 0.538 | 0.0334 | 0.81 | 0.493 |

Training-batch advantage statistics (decks mixed by decision share): mean -0.0137, sd 0.2268.

Norm of each deck's mean loss gradient, `global` normalisation:

| deck | whole | emb_tables | gru | trunk | option_scorer | value_head |
|---|---|---|---|---|---|---|
| jund | 0.2162 | 0.03644 | 0.07609 | 0.109 | 0.03515 | 0.1629 |
| blue | 0.2354 | 0.06185 | 0.07226 | 0.1971 | 0.05954 | 0.06327 |

Who drives the batch gradient, `global` (G = sum of share x deck gradient): norm share = share x |g| / sum, G2 share = share x <g, G> / |G|^2.

| deck | batch share | norm share | G2 share | cos(g_deck, G) |
|---|---|---|---|---|
| jund | 0.500 | 0.479 | 0.464 | 0.744 |
| blue | 0.500 | 0.521 | 0.536 | 0.790 |


| norm | group | ceiling mean (min..max) | cross mean | corrected mean | pairs with both ceilings >= 0.10 | of them negative |
|---|---|---|---|---|---|---|
| global | emb_tables | 0.037 (0.006..0.069) | +0.030 | n/a | 0 of 1 | 0 |
| global | gru | 0.456 (0.181..0.731) | +0.246 | +0.68 | 1 of 1 | 0 |
| global | trunk | 0.037 (0.020..0.054) | -0.003 | n/a | 0 of 1 | 0 |
| global | option_scorer | 0.031 (0.028..0.033) | +0.014 | n/a | 0 of 1 | 0 |
| global | value_head | 0.850 (0.723..0.978) | +0.626 | +0.74 | 1 of 1 | 0 |
| global | embedding+trunk | 0.037 (0.018..0.056) | +0.000 | n/a | 0 of 1 | 0 |

### Gradient cosines, `global` advantage normalisation, half level

**emb_tables**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.069** ±0.01 | 0.030 |
| blue | 1.50 | **0.006** ±0.02 |

**gru**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.731** ±0.02 | 0.246 |
| blue | 0.68 | **0.181** ±0.01 |

**trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.054** ±0.03 | -0.003 |
| blue | -0.09 | **0.020** ±0.02 |

**option_scorer**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.028** ±0.02 | 0.014 |
| blue | 0.45 | **0.033** ±0.01 |

**value_head**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.978** ±0.01 | 0.626 |
| blue | 0.74 | **0.723** ±0.01 |

**embedding+trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.056** ±0.03 | 0.000 |
| blue | 0.01 | **0.018** ±0.02 |

