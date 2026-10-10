| deck | games | dec/game | turns/game | batch share (dec.) | batch share (traj.) | adv mean ± sd | return mean ± sd | value mean ± sd | v_loss | expl. var | win rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jund | 2400 | 152 | 19.8 ± 7.2 | 0.252 | 0.167 | +0.009 ± 0.185 | -0.107 ± 0.551 | -0.116 ± 0.525 | 0.0171 | 0.89 | 0.429 |
| blue | 2400 | 70 | 16.5 ± 5.8 | 0.113 | 0.167 | +0.008 ± 0.302 | -0.111 ± 0.657 | -0.119 ± 0.588 | 0.0459 | 0.79 | 0.503 |
| red | 2400 | 91 | 13.8 ± 4.6 | 0.145 | 0.167 | +0.005 ± 0.254 | -0.027 ± 0.598 | -0.032 ± 0.541 | 0.0324 | 0.82 | 0.586 |
| affinity | 2400 | 117 | 17.4 ± 5.7 | 0.190 | 0.167 | +0.013 ± 0.217 | -0.159 ± 0.578 | -0.173 ± 0.537 | 0.0237 | 0.86 | 0.458 |
| elves | 2400 | 99 | 14.8 ± 4.6 | 0.156 | 0.167 | +0.007 ± 0.233 | -0.072 ± 0.636 | -0.079 ± 0.597 | 0.0271 | 0.87 | 0.518 |
| tron | 2400 | 87 | 16.7 ± 6.4 | 0.143 | 0.167 | +0.002 ± 0.243 | +0.159 ± 0.638 | +0.157 ± 0.590 | 0.0295 | 0.85 | 0.546 |

Training-batch advantage statistics (decks mixed by decision share): mean +0.0077, sd 0.2327.

Norm of each deck's mean loss gradient, `global` normalisation:

| deck | whole | emb_tables | gru | trunk | option_scorer | value_head |
|---|---|---|---|---|---|---|
| jund | 0.09849 | 0.0283 | 0.03937 | 0.07742 | 0.03687 | 0 |
| blue | 0.2159 | 0.06288 | 0.05975 | 0.1813 | 0.079 | 0 |
| red | 0.1536 | 0.04124 | 0.05968 | 0.1227 | 0.05729 | 0 |
| affinity | 0.1285 | 0.03836 | 0.04194 | 0.1053 | 0.04703 | 0 |
| elves | 0.1526 | 0.04969 | 0.04968 | 0.124 | 0.05467 | 0 |
| tron | 0.1703 | 0.05085 | 0.0437 | 0.143 | 0.06369 | 0 |

Who drives the batch gradient, `global` (G = sum of share x deck gradient): norm share = share x |g| / sum, G2 share = share x <g, G> / |G|^2.

| deck | batch share | norm share | G2 share | cos(g_deck, G) |
|---|---|---|---|---|
| jund | 0.252 | 0.172 | 0.198 | 0.556 |
| blue | 0.113 | 0.170 | 0.140 | 0.401 |
| red | 0.145 | 0.155 | 0.158 | 0.494 |
| affinity | 0.190 | 0.170 | 0.182 | 0.520 |
| elves | 0.156 | 0.165 | 0.172 | 0.506 |
| tron | 0.143 | 0.169 | 0.150 | 0.430 |


| norm | group | ceiling mean (min..max) | cross mean | corrected mean | pairs with both ceilings >= 0.10 | of them negative |
|---|---|---|---|---|---|---|
| global | emb_tables | 0.064 (0.005..0.189) | +0.029 | n/a | 0 of 15 | 0 |
| global | gru | 0.388 (0.136..0.612) | +0.275 | +0.77 | 15 of 15 | 0 |
| global | trunk | 0.046 (-0.024..0.212) | +0.011 | n/a | 0 of 15 | 0 |
| global | option_scorer | 0.134 (0.060..0.280) | +0.079 | +0.62 | 3 of 15 | 0 |
| global | value_head | 0.000 (0.000..0.000) | +0.000 | n/a | 0 of 15 | 0 |
| global | embedding+trunk | 0.048 (-0.021..0.209) | +0.013 | n/a | 0 of 15 | 0 |

### Gradient cosines, `global` advantage normalisation, half level

**emb_tables**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.058** ±0.02 | 0.014 | 0.061 | 0.061 | 0.049 | 0.022 |
| blue | 0.42 | **0.019** ±0.02 | 0.019 | -0.011 | 0.025 | 0.012 |
| red | 0.58 | 0.31 | **0.189** ±0.05 | 0.051 | 0.024 | 0.016 |
| affinity | 1.07 | -0.33 | 0.50 | **0.056** ±0.06 | 0.036 | 0.025 |
| elves | 0.87 | 0.79 | 0.23 | 0.66 | **0.055** ±0.03 | 0.026 |
| tron | 1.24 | 1.22 | 0.52 | 1.43 | 1.54 | **0.005** ±0.00 |

**gru**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.581** ±0.03 | 0.267 | 0.412 | 0.460 | 0.413 | 0.187 |
| blue | 0.90 | **0.153** ±0.04 | 0.227 | 0.216 | 0.216 | 0.113 |
| red | 0.69 | 0.74 | **0.612** ±0.02 | 0.371 | 0.396 | 0.164 |
| affinity | 0.96 | 0.87 | 0.75 | **0.399** ±0.02 | 0.353 | 0.162 |
| elves | 0.81 | 0.83 | 0.76 | 0.84 | **0.445** ±0.01 | 0.174 |
| tron | 0.66 | 0.79 | 0.57 | 0.69 | 0.71 | **0.136** ±0.02 |

**trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **-0.012** ±0.02 | -0.018 | 0.057 | 0.044 | 0.033 | -0.001 |
| blue | nan | **0.007** ±0.01 | -0.054 | -0.033 | 0.023 | 0.006 |
| red | nan | -1.42 | **0.212** ±0.03 | 0.069 | 0.024 | -0.008 |
| affinity | nan | -1.80 | 0.68 | **0.049** ±0.04 | 0.001 | 0.015 |
| elves | nan | 1.34 | 0.25 | 0.03 | **0.044** ±0.00 | 0.012 |
| tron | nan | nan | nan | nan | nan | **-0.024** ±0.02 |

**option_scorer**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.175** ±0.02 | 0.067 | 0.133 | 0.122 | 0.094 | 0.070 |
| blue | 0.56 | **0.082** ±0.01 | 0.071 | 0.063 | 0.056 | 0.044 |
| red | 0.60 | 0.47 | **0.280** ±0.01 | 0.109 | 0.099 | 0.054 |
| affinity | 0.94 | 0.70 | 0.66 | **0.097** ±0.01 | 0.087 | 0.062 |
| elves | 0.68 | 0.59 | 0.57 | 0.85 | **0.109** ±0.01 | 0.053 |
| tron | 0.68 | 0.63 | 0.42 | 0.81 | 0.65 | **0.060** ±0.01 |

**value_head**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **0.000** ±0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| blue | nan | **0.000** ±0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| red | nan | nan | **0.000** ±0.00 | 0.000 | 0.000 | 0.000 |
| affinity | nan | nan | nan | **0.000** ±0.00 | 0.000 | 0.000 |
| elves | nan | nan | nan | nan | **0.000** ±0.00 | 0.000 |
| tron | nan | nan | nan | nan | nan | **0.000** ±0.00 |

**embedding+trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue | red | affinity | elves | tron |
|---|---|---|---|---|---|---|
| jund | **-0.004** ±0.02 | -0.014 | 0.057 | 0.046 | 0.035 | 0.001 |
| blue | nan | **0.008** ±0.01 | -0.047 | -0.030 | 0.024 | 0.007 |
| red | nan | -1.13 | **0.209** ±0.03 | 0.067 | 0.024 | -0.005 |
| affinity | nan | -1.52 | 0.66 | **0.049** ±0.04 | 0.006 | 0.016 |
| elves | nan | 1.23 | 0.24 | 0.12 | **0.045** ±0.01 | 0.014 |
| tron | nan | nan | nan | nan | nan | **-0.021** ±0.02 |

