# Source notes and evidence boundaries

## Snapshot and method

The papers describe **10 October 2026** using repository revision
`18a3efa3e891b2b4ecddb51db22b615189a71ef5` (the starting revision of the Halifax
workspace). The papers' later authoring commit does not move that evidence
cutoff. `evidence-manifest.json` records SHA-256 hashes of the local evidence
files used. Public literature metadata was checked against primary sources
during authoring.

Architecture claims were checked against source code and current contracts,
not just the older README narrative. Numerical results are transcribed from the
dated experiment ledger; training and evaluation were **not rerun** for these
papers. The authors did not download remote checkpoints or the raw per-seed
evaluation outputs. The uncertainty intervals are the experiment's reported
intervals, not independently re-estimated intervals. GitHub evidence links are
pinned to the snapshot; readers need repository access if it is private.

## Claim-to-source map

All paths below refer to the snapshot above.

| Claim group | Evidence |
|---|---|
| Six decks, matchup registration, BO3 and default sideboarding | `mtg_ml/engine/decks.py`, `mtg_ml/match.py`, `mtg_ml/engine/sideboard.py`, `docs/sideboarding.md` |
| Cleansing Wildfire / indestructible Bridge interaction | `mtg_ml/engine/cards.toml`, `data/oracle_cards.json`, `tests/test_cards_jund.py` |
| Python reference, Rust port, common card specification | `CLAUDE.md`, `docs/native-engine.md`, `native/src/cards.rs`, `mtg_ml/backend.py` |
| Engine comparison and knowledge restrictions | `mtg_ml/difftest.py`, `mtg_ml/engine/view.py`, `tests/test_view_env.py` |
| Feature versions, card shapes, complete entity coverage, previews | `docs/features.md`, `mtg_ml/encode.py`, `mtg_ml/rl/features.py` |
| Entity attention, recurrent core, option scorer, value head, belief branch | `mtg_ml/rl/model.py`, `mtg_ml/rl/ppo.py`, `docs/belief-head.md` |
| Reward / GAE, trajectory updates, generic defaults | `mtg_ml/rl/rollout.py`, `mtg_ml/rl/ppo.py`, `mtg_ml/rl/train.py` |
| Featured r7 architecture and inherited r8 settings | `docs/experiments/r7-overnight.md`, `docs/experiments/r7-overnight-launch.json`, `tools/r8_campaign.py`, ledger entries `20261009-r8-results` and `20261008-r7-fs7-h256` |
| Shared-memory inference, stacked policies, CUDA graphs | `mtg_ml/rl/inference.py`, `mtg_ml/rl/stacked.py`, `mtg_ml/rl/ppo.py` |
| Completed performance and trust checks | `docs/experiments/ledger.md`: `20261010-r8-pilot-matrix`, `20261009-r8-results` |
| Current versus ongoing models | `docs/experiments/models.md`; ledger `20261010-r9-base`; `docs/experiments/interference.md` |
| Hosted architecture and pinned weights | `docs/hosting.md`, `mtg_ml/play_config.toml`, `deploy/compose.yaml` |
| Missed forestcycling and withdrawn complaint | `docs/playtest-reviews/2026-10-10.md` |

The papers do not repeat historical speed claims because those measurements
used different networks, features, and execution paths. All performance figures
retained here are playing-strength scores, with machine provenance below.

## The Jund matrix and its chart

Source: ledger section `20261010-r8-pilot-matrix`. Figure 2 in the general paper
uses the **Jund delta** column, not the column measuring the pilot on the other
deck. Candidate and baseline both play Jund; the opponent is always `r7-lr075`
playing the corresponding opposing deck.

- Candidate: `r8-jund-pilot`, benchmarked `policy/v01464.pt`; SHA-256
  `71c758dcf4513deb3bc7021652410d7c573c366c97d8c01a4da1b5f036078379`.
- Reference: `r7-lr075`, SHA-256
  `672aaa0c2764f853198c44a6fa7847469e01d9767952ea3fb8e9d84e068f636e`.
- Both use feature set 7. The pilot has three million additional training games
  across six equally weighted Jund pairings, training both seats.
- **Checkpoint caveat:** archive `policy.pt` is the later v01465. The measured
  v01464 is retained as `policy-v01464.pt`. The paper does not identify those
  files as interchangeable.
- Sampled: 800 games per cell, 200 seeds; each seed crosses both starting
  players and physical seats. Greedy: 400 games, 100 of the same seeds.
- Preboard/game-1 decklists; a draw contributes half a point.
- Chart bars are paired normal 95% intervals over four-game seed-block
  differences. They are **not** a collection of independent-game error bars.
  The ledger's individual cell summaries use Wilson intervals.
- Reported average improvement: sampled 18.0 percentage points; greedy 13.8.
  Blue sampled delta: 22.0 +/- 4.3, giving the derived endpoints 17.7 and 26.3.
- The JSON preserves published one-decimal deltas. For Tron, subtracting the
  separately rounded displayed scores (53.9 minus 35.6) gives 18.3, while the
  reported paired delta is 18.2. Retain the recorded delta rather than
  reconstructing it from rounded cells.
- Hardware: h100-private, two Intel Xeon Platinum 8462Y+ CPUs; 8 workers pinned
  to CPUs 56-63; `OMP_NUM_THREADS=1`; native engine; no GPU. Evaluation code
  revision: `6ca2fb59e1cfab95a5aa290a76e49a948d18f6b7`.

The reported intervals concern evaluation deals conditional on selected
checkpoints. They do not measure training-seed variability or prove a generally
better learning algorithm. No pooled interval is invented for the six-pairing
average.

## The separate scripted-specialist result

Source: `20261009-r8-results`, table **Final specialist benchmark**, column
`jund_vs_sblue`. This is a different opponent and protocol from the matrix.

| Model as Jund | Sampled score | Greedy score |
|---|---:|---:|
| `r8-jund-blue` v02930 | 75.2% | 79.0% |
| `r7-lr075` v04079 | 46.3% | 49.0% |

The candidate trained for six million additional games. Its policy SHA-256 is
`b2ca176382139cf5f6733937605e43b836af1973b18624e8ae19d68a850c9e71`.
Each cell and mode uses 100 four-game blocks (400 games), fair information
contract; the ledger describes Wilson 95% uncertainty of roughly +/-5
percentage points. The candidate's final evaluation ran on h100-private;
the inherited parent row uses the same specialist-benchmark protocol. The
matrix hardware statement is not used as an invented hardware specification
for every historical specialist result.

The 75.2% is **not** the legacy-Blue in-training benchmark of 83.7%, nor the
82.0% legacy-Blue final specialist-panel cell. The 46.3% parent baseline is
from the same scripted-specialist column. Keeping these separate prevents
an attractive but invalid comparison.

The trust tables report overall FAIL for `r7-lr075`, `r8-jund-blue`, and
`r8-jund-pilot`. The Jund pilot passes a per-deck bias subcheck, but not the
complete test. No trust test is recorded for `r8-jund-mirror`; the paper says
"two tested r8 fine-tunes" to avoid implying otherwise. Pool-based
vulnerability tests do not calculate formal exploitability, and closeness to
a human matchup matrix alone would not establish human-level strength.

## Gameplay example and open work

The Tron playtest review reconstructs all 425 recorded choices. The cited
turn-9, main-phase-2 decision offered pass (0.92, selected) and forestcycling
for one mana (0.08), with no land drop used and five untapped mana sources.
The review marks this a clear policy error; its suggestion that rare-action
learning explains it is untested. The paper omits private player identities
and operational access details. The withdrawn complaint concerns a different
decision, not the landcycling error.

The r9 base runs and feature-set-8 belief work are described as ongoing at the
cutoff. The interference diagnostic is inconclusive; the paper does not claim
that gradient conflict explains specialization gains. Native/Python agreement
is implementation-consistency evidence rather than an independent rules oracle.

## Literature verification

Primary sources, checked 10 October 2026:

1. Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), [PPO](https://arxiv.org/abs/1707.06347).
2. Schulman, Moritz, Levine, Jordan, and Abbeel (2015 preprint; ICLR 2016),
   [GAE](https://arxiv.org/abs/1506.02438). The venue is also listed in the
   [official ICLR 2016 archive](https://www.iclr.cc/archive/www/2016.html).
3. Huang and Ontañón (2022; 2020 preprint),
   [invalid-action masking](https://arxiv.org/abs/2006.14171). The arXiv record
   identifies FLAIRS volume 35 and its proceedings DOI.
4. Silver et al. (2017), [AlphaZero preprint](https://arxiv.org/abs/1712.01815).
   The title/year identify the 2017 preprint, not the differently titled 2018
   Science article.
5. Brown, Bakhtin, Lerer, and Gong (2020), [ReBeL](https://arxiv.org/abs/2007.13544).
   [Official NeurIPS 33 record](https://proceedings.nips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html).
6. Churchill, Biderman, and Herrick (2019),
   [Magic: The Gathering is Turing Complete](https://arxiv.org/abs/1904.09828).
   The paper cites the verified preprint year and does not transfer its
   undecidability result to the bounded simulator.

These references explain method lineage and conceptual limits. They are not
evidence for mtg-ml's measured scores. The papers use original summaries and
no verbatim passages from those publications.
