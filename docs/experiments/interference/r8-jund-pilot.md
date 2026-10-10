| deck | games | dec/game | turns/game | batch share (dec.) | batch share (traj.) | adv mean ± sd | return mean ± sd | value mean ± sd | v_loss | expl. var | win rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jund | 2400 | 171 | 19.4 ± 7.3 | 0.500 | 0.500 | -0.002 ± 0.169 | -0.056 ± 0.532 | -0.054 ± 0.507 | 0.0143 | 0.90 | 0.444 |
| blue | 2400 | 84 | 16.0 ± 5.4 | 0.500 | 0.500 | -0.007 ± 0.273 | -0.079 ± 0.607 | -0.072 ± 0.540 | 0.0376 | 0.80 | 0.515 |

Training-batch advantage statistics (decks mixed by decision share): mean -0.0045, sd 0.2272.

Norm of each deck's mean loss gradient, `global` normalisation:

| deck | whole | emb_tables | gru | trunk | option_scorer | value_head |
|---|---|---|---|---|---|---|
| jund | 0.1046 | 0.02871 | 0.03673 | 0.08715 | 0.0292 | 0.01786 |
| blue | 0.2846 | 0.07294 | 0.07877 | 0.2332 | 0.07078 | 0.1004 |

Who drives the batch gradient, `global` (G = sum of share x deck gradient): norm share = share x |g| / sum, G2 share = share x <g, G> / |G|^2.

| deck | batch share | norm share | G2 share | cos(g_deck, G) |
|---|---|---|---|---|
| jund | 0.500 | 0.269 | 0.124 | 0.361 |
| blue | 0.500 | 0.731 | 0.876 | 0.940 |


| norm | group | ceiling mean (min..max) | cross mean | corrected mean | pairs with both ceilings >= 0.10 | of them negative |
|---|---|---|---|---|---|---|
| global | emb_tables | -0.009 (-0.068..0.050) | -0.006 | n/a | 0 of 1 | 0 |
| global | gru | 0.233 (0.200..0.265) | +0.201 | +0.87 | 1 of 1 | 0 |
| global | trunk | 0.045 (-0.042..0.133) | -0.025 | n/a | 0 of 1 | 0 |
| global | option_scorer | 0.028 (0.016..0.041) | +0.015 | n/a | 0 of 1 | 0 |
| global | value_head | 0.288 (-0.141..0.717) | +0.142 | n/a | 0 of 1 | 0 |
| global | embedding+trunk | 0.040 (-0.045..0.125) | -0.023 | n/a | 0 of 1 | 0 |

### Gradient cosines, `global` advantage normalisation, half level

**emb_tables**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **-0.068** ±0.03 | -0.006 |
| blue | nan | **0.050** ±0.03 |

**gru**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.265** ±0.03 | 0.201 |
| blue | 0.87 | **0.200** ±0.04 |

**trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **-0.042** ±0.03 | -0.025 |
| blue | nan | **0.133** ±0.04 |

**option_scorer**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **0.016** ±0.01 | 0.015 |
| blue | 0.56 | **0.041** ±0.00 |

**value_head**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **-0.141** ±0.18 | 0.142 |
| blue | nan | **0.717** ±0.17 |

**embedding+trunk**: diagonal = within-deck cosine (noise ceiling); above it cross-deck cosine; below it noise-corrected cosine (cross / sqrt of the two ceilings).

| | jund | blue |
|---|---|---|
| jund | **-0.045** ±0.03 | -0.023 |
| blue | nan | **0.125** ±0.04 |

