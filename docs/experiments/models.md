# Model registry

One row per run, every archived run plus the r8 runs, with the handle (see [naming.md](naming.md)), the immutable run ID, **all legacy names** so nothing gets lost, and where the weights and the evidence are. Machine-readable twin: [`models.json`](models.json) (keep both in sync; `tests/test_model_registry.py` checks that they agree). Dated 2026-10-10; games and evals of running runs are the last values read from their `metrics.jsonl` that day.

Numbers: bench = learner Jund vs the legacy Blue bot, game 1, sampled / greedy (1,000 games in training, 2,000 after the run, see the [README](README.md)); L1 = Elo on ladder L1 ([ladder.md](ladder.md), SE about 12), at the end of the run unless the column gives the game count (`@17.5M` is the last evaluation, not the end). Runs on different feature sets and matchups share these two numbers only loosely. Seeds before r7 were not recorded (shown as s0, the flag default). The archive column is relative to `h100-private:~/mtg-ml-checkpoints/`.

## Current roles

| role | handle | why |
|---|---|---|
| `best/general` | `r7-lr075` | Only full-matrix model; r8-continue plateaued; r8-belief would replace it if it overtakes at matched games |
| `best/jund_blue` | `r8-jund-blue` | Final specialist benchmark jund_vs_sblue 75.2/79.0 (r4-control 64.0/65.5), L1 238.6 (fs4-ft 212.5); a specialist, its trust test fails |
| `best/blue_vs_jund` | `r4-control` | Blue as learner vs the Jund specialist 85.3/87.0 (specialist benchmark); r8-jund-blue 83.8/84.0 (level within +-5) |
| `best/jund_jund` | `r8-jund-mirror` | Mirror vs the Jund specialist 83.0/88.5 (r7-lr075 57.8/68.5); forgets Blue |
| `best/jund` | `r8-jund-pilot` | Deck vs field; beats r7-lr075 as Jund in all six pairings on paired seeds (sampled +10.1 to +28.0, mean +18.0; greedy mean +13.8), ledger 20261010-r8-pilot-matrix |
| `best/blue` | `r7-lr075` | No deck-specific Blue model yet |
| `best/madness` | `r7-lr075` | rm-1m is a one-matchup baseline, not a full model |
| `best/affinity` | `r7-lr075` | Only full-matrix model |
| `best/elves` | `r7-lr075` | Only full-matrix model |
| `best/tron` | `r7-lr075` | Only full-matrix model; the trust test flags Tron as overrated |
| `play/jund` | `r8-jund-pilot` | Play site offer r8-jund-pilot, greedy (default Jund opponent): beats r7-lr075 as Jund in all six pairings, 67% vs the Blue specialist, 80% in the mirror. r8-jund (the Jund vs Blue specialist) and r7-jund (legacy) stay offered |
| `play/blue` | `r8-blue-pilot` | Play site offer r8-blue-pilot, greedy (the only Blue opponent besides legacy Delver; replaces r7-blue): trained on all six Blue pairings, 74 to 77% against the Blue specialist bot in the mirror, 82% against the Jund specialist |
| `play/madness` | `r7-lr075` | Play site offer r7-madness, greedy |
| `play/affinity` | `r7-lr075` | Play site offer r7-affinity, greedy |
| `play/elves` | `r7-lr075` | Play site offer r7-elves, greedy |
| `play/tron` | `r8-tron-pilot` | Play site offer r8-tron-pilot, greedy (replaces r7-tron): trained on all six Tron pairings; trust test (six pilots) Tron bias +0.13; Tron audit (PR #108), game 1, 40 games each, win rate vs Jund 0.53, Blue 0.65, Madness 0.57, Affinity 0.75, Elves 0.35 |

| in flight | compared against | question |
|---|---|---|
| `r8-belief` | `r7-lr075` | does feature set 8 plus the belief head beat the lr075 trajectory at matched games |
| `r9-jund-pilot` | `r8-jund-pilot` | does a second round against frozen round-1 pilots (all six decks, same recipe) beat each round-1 pilot head to head and on the trust test |

Roles are pointers: change them here and in `models.json` (`roles`, plus a line in `role_history`) when a better model is established, and say why in the ledger. Open: fs4-ft was never run against the Jund and Blue specialists (r8-jund-blue beats it on L1, 238.6 against 212.5, and r4-control on the specialist benchmark). The play site pins `r7-lr075/policy`, `r8-jund-pilot/policy` (the `r8-jund-pilot` opponent), `r8-jund-blue/policy` (the `r8-jund` opponent), `r8-blue-pilot/policy` (the `r8-blue-pilot` opponent), `r8-tron-pilot/policy` (the `r8-tron-pilot` opponent) and the legacy `r4-control/policy` in `mtg_ml/play_config.toml`; those keys are not renamed.

## Registry

Status: `running`, `stopped` (ended by hand), `finished` (ended, not yet archived), `planned` (prepared, not launched), `archived`, `superseded` (archived and replaced by a better parent, still a valid reference), `rejected`, `crashed`.

### r8 (running and finished 2026-10-09 to 10)

| handle | run ID | legacy names | parent | scope | games | bench s/g | L1 | status | archive (`h100-private:~/mtg-ml-checkpoints/`) | ledger |
|---|---|---|---|---|---|---|---|---|---|---|
| `r8-continue` | `20261009-r8-fs7h256-all-ft-r7-lr075-continue-s8` | `r8-lr075-continue`, `r8-20261009/r8-lr075-continue`, `20261009-r8-lr075-continue`, arm A, `A` | `r7-lr075` | `all` | 10.05M | 56.7 / 69.2 | 37.3 @10.0M | stopped | `20261009-r8-lr075-continue` | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-belief` | `20261009-r8-fs8h256-all-scratch-belief-s8` | `r8-fs8-belief`, `r8b-20261009/r8-fs8-belief`, `20261009-r8-fs8-belief`, `r8b`, `r8b-20261009`, arm B, `B` |  | `all` | 3.64M | 49.0 / 62.0 | -51.8 @3.5M | running | not archived yet | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-belief-x1` | `20261009-r8-fs8h256-all-scratch-belief-s8.x1` | r8-fs8-belief (first attempt), `r8-20261009/r8-fs8-belief` |  | `all` | 0.11M |  |  | crashed | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-blue` | `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s8` | `r8-jund-blue-ft`, `r8-20261009/r8-jund-blue-ft`, `20261009-r8-jund-blue-ft`, arm C, `C` | `r7-lr075` | `mu-jund-blue` | 6.00M | 83.7 / 85.7 | 238.6 | archived | `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s8` | [entry](ledger.md#20261009-r8-results--final-results-and-archives-of-r8-jund-blue-r8-jund-pilot-r8-jund-mirror) |
| `r8-jund-blue-s2` | `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s9` | `r8-jund-blue-ft-s2`, `r8d-20261009/r8-jund-blue-ft-s2`, `20261009-r8-jund-blue-ft-s2`, arm C-s2, `C-s2`, `r8d`, `r8d-20261009` | `r7-lr075` | `mu-jund-blue` | 1.02M | 68.8 / 79.9 | 112.2 @1.0M | archived | `20261009-r8-jund-blue-ft-s2` | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-mirror` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8` | `r8-jund-mirror-ft`, `r8e3-20261009/r8-jund-mirror-ft`, `20261009-r8-jund-mirror-ft`, arm D, `D`, `r8e3`, `r8e3-20261009` | `r7-lr075` | `mu-jund-jund` | 2.00M |  |  | archived | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8` | [entry](ledger.md#20261009-r8-results--final-results-and-archives-of-r8-jund-blue-r8-jund-pilot-r8-jund-mirror) |
| `r8-jund-mirror-x1` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8.x1` | `r8e-20261009`, `r8e`, `r8e-20261009/r8-jund-mirror-ft` | `r7-lr075` | `mu-jund-jund` | 0.08M |  |  | crashed | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-mirror-x2` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8.x2` | `r8e2-20261009`, `r8e2`, `r8e2-20261009/r8-jund-mirror-ft` | `r7-lr075` | `mu-jund-jund` | 0.06M |  |  | stopped | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-pilot` | `20261009-r8-fs7h256-dk-jund-ft-r7-lr075-s8` | `r8-jund-pilot`, `r8f-20261009/r8-jund-pilot`, `20261009-r8-jund-pilot`, `pilot`, Jund pilot, `r8f`, `r8f-20261009` | `r7-lr075` | `dk-jund` | 3.00M | 74.3 / 77.7 | 148.4 | archived | `20261009-r8-fs7h256-dk-jund-ft-r7-lr075-s8` | [entry](ledger.md#20261009-r8-results--final-results-and-archives-of-r8-jund-blue-r8-jund-pilot-r8-jund-mirror) |
| `r8-blue-pilot` | `20261010-r8-fs7h256-dk-blue-ft-r7-lr075-s8` | `r8-blue-pilot`, `r8g-20261010/r8-blue-pilot`, `20261010-r8-blue-pilot`, blue pilot, `r8g-20261010` | `r7-lr075` | `dk-blue` | 3.00M |  |  | finished | not archived yet | pending |
| `r8-madness-pilot` | `20261010-r8-fs7h256-dk-madness-ft-r7-lr075-s8` | `r8-madness-pilot`, `r8h1-20261010/r8-madness-pilot`, `20261010-r8-madness-pilot`, madness pilot, `r8h1-20261010` | `r7-lr075` | `dk-madness` | 3.00M |  |  | finished | not archived yet | pending |
| `r8-affinity-pilot` | `20261010-r8-fs7h256-dk-affinity-ft-r7-lr075-s8` | `r8-affinity-pilot`, `r8h1-20261010/r8-affinity-pilot`, `20261010-r8-affinity-pilot`, affinity pilot, `r8h1-20261010` | `r7-lr075` | `dk-affinity` | 3.00M |  |  | finished | not archived yet | pending |
| `r8-elves-pilot` | `20261010-r8-fs7h256-dk-elves-ft-r7-lr075-s8` | `r8-elves-pilot`, `r8h2-20261010/r8-elves-pilot`, `20261010-r8-elves-pilot`, elves pilot, `r8h2-20261010` | `r7-lr075` | `dk-elves` | 3.00M |  |  | finished | not archived yet | pending |
| `r8-tron-pilot` | `20261010-r8-fs7h256-dk-tron-ft-r7-lr075-s8` | `r8-tron-pilot`, `r8h2-20261010/r8-tron-pilot`, `20261010-r8-tron-pilot`, tron pilot, `r8h2-20261010` | `r7-lr075` | `dk-tron` | 3.00M |  |  | finished | not archived yet | pending |
| `r9-jund-pilot` | `20261010-r9-fs7h256-dk-jund-ft-r8-jund-pilot-s8` | `r9-jund-pilot`, round 2 jund | `r8-jund-pilot` | `dk-jund` |  |  |  | planned |  | planned |
| `r9-blue-pilot` | `20261010-r9-fs7h256-dk-blue-ft-r8-blue-pilot-s8` | `r9-blue-pilot`, round 2 blue | `r8-blue-pilot` | `dk-blue` |  |  |  | planned |  | planned |
| `r9-madness-pilot` | `20261010-r9-fs7h256-dk-madness-ft-r8-madness-pilot-s8` | `r9-madness-pilot`, round 2 madness | `r8-madness-pilot` | `dk-madness` |  |  |  | planned |  | planned |
| `r9-affinity-pilot` | `20261010-r9-fs7h256-dk-affinity-ft-r8-affinity-pilot-s8` | `r9-affinity-pilot`, round 2 affinity | `r8-affinity-pilot` | `dk-affinity` |  |  |  | planned |  | planned |
| `r9-elves-pilot` | `20261010-r9-fs7h256-dk-elves-ft-r8-elves-pilot-s8` | `r9-elves-pilot`, round 2 elves | `r8-elves-pilot` | `dk-elves` |  |  |  | planned |  | planned |
| `r9-tron-pilot` | `20261010-r9-fs7h256-dk-tron-ft-r8-tron-pilot-s8` | `r9-tron-pilot`, round 2 tron | `r8-tron-pilot` | `dk-tron` |  |  |  | planned |  | planned |

### r7 and earlier, newest first

| handle | run ID | legacy names | parent | scope | games | bench s/g | L1 | status | archive (`h100-private:~/mtg-ml-checkpoints/`) | ledger |
|---|---|---|---|---|---|---|---|---|---|---|
| `r7-lr150` | `20261008-r7-fs7h256-all-scratch-lr150-s8` | `20261008-r7-fs7-h256-lr150`, r7 lr150, `lr150` |  | `all` | 9.04M | 46.9 / 57.4 |  | archived | `20261008-r7-fs7-h256-lr150` | [entry](ledger.md#20261008-r7-fs7-h256--fresh-full-matrix-lr-comparison) |
| `r7-lr075` | `20261008-r7-fs7h256-all-scratch-lr075-s8` | `20261008-r7-fs7-h256-lr075`, r7 lr075, `lr075`, `r7-lr075/policy` |  | `all` | 8.35M | 56.0 / 65.2 | 5 @8.25M | archived | `20261008-r7-fs7-h256-lr075` | [entry](ledger.md#20261008-r7-fs7-h256--fresh-full-matrix-lr-comparison) |
| `r6-h128` | `20261008-r6-fs6h128-all-scratch-s0` | `20261008-r6-h128`, `r6-h128` |  | `all` | 12.90M | 66.8 / 67.2 | 76 | archived | `20261008-r6-h128` | [entry](ledger.md#20261008-r6--six-decks-from-scratch-on-feature-set-6-h128-h128--attention-h256) |
| `r6-h128-attn` | `20261008-r6-fs6h128-all-scratch-attn-s0` | `20261008-r6-h128-attn`, `r6-h128-attn` |  | `all` | 2.41M | 50.8 / 65.9 | -26 | archived | `20261008-r6-h128-attn` | [entry](ledger.md#20261008-r6--six-decks-from-scratch-on-feature-set-6-h128-h128--attention-h256) |
| `r6-h256` | `20261008-r6-fs6h256-all-scratch-s0` | `20261008-r6-h256`, `r6-h256` |  | `all` | 7.33M | 50.0 / 59.3 | -4 | archived | `20261008-r6-h256` | [entry](ledger.md#20261008-r6--six-decks-from-scratch-on-feature-set-6-h128-h128--attention-h256) |
| `rm-1m` | `20261007-rm-fs1h128-mu-madness-jund-ft-r1-control-1m-s0` | `20261007-red-madness-1m`, `red-madness-1m`, Red Madness 1M, PR #23 | `r1-control` | `mu-madness-jund` | 1.00M |  |  | archived | `20261007-red-madness-1m` | [entry](ledger.md#20261007-red-madness-1m--a-third-deck-against-a-frozen-jund) |
| `r5-mix-h128` | `20261007-r5-fs3h128-mix3-scratch-s0` | `20261007-r5-mix-h128`, `r5-mix-h128` |  | `mix3` | 5.84M | 56.4 /  | 40 | archived | `20261007-r5-mix-h128` | [entry](ledger.md#20261007-r5--width-test-on-the-three-deck-mix) |
| `r5-mix-h256` | `20261007-r5-fs3h256-mix3-scratch-s0` | `20261007-r5-mix-h256`, `r5-mix-h256` |  | `mix3` | 2.59M | 46.3 /  | -32 | archived | `20261007-r5-mix-h256` | [entry](ledger.md#20261007-r5--width-test-on-the-three-deck-mix) |
| `r5-mix-h256-lr3e4` | `20261007-r5-fs3h256-mix3-scratch-lr3e4-s0` | `20261007-r5-mix-h256-lr3e-4`, `r5-mix-h256-lr3e-4` |  | `mix3` | 2.60M | 38.1 /  | -69 | archived | `20261007-r5-mix-h256-lr3e-4` | [entry](ledger.md#20261007-r5--width-test-on-the-three-deck-mix) |
| `r4-control` | `20261007-r4-fs3h128-mu-jund-blue-ft-r3-postboard-ctl-s0` | `20261007-r4-control`, `r4-control`, `r4-ctl` | `r3-postboard` | `mu-jund-blue` | 7.22M | 78.2 / 77.6 | 163 @17.5M | superseded | `20261007-r4-control` | [entry](ledger.md#20261007-r4--one-network-for-three-decks) |
| `r4-mix` | `20261007-r4-fs3h128-mix3-ft-r3-postboard-s0` | `20261007-r4-mix`, `r4-mix` | `r3-postboard` | `mix3` | 10.00M | 73.4 / 71.7 | 155 | archived | `20261007-r4-mix` | [entry](ledger.md#20261007-r4--one-network-for-three-decks) |
| `r3-control` | `20261007-r3-fs2h128-mu-jund-blue-ft-r1-lranneal-ctl-s0` | `20261007-r3-control`, `r3-control` | `r1-lranneal` | `mu-jund-blue` | 1.00M | 73.0 / 74.7 | 145.5 | archived | `20261007-r3-control` | [entry](ledger.md#20261007-r3--fine-tunes-from-the-lr-anneal-checkpoint) |
| `r3-postboard` | `20261007-r3-fs2h128-mu-jund-blue-ft-r1-lranneal-postboard-s0` | `20261007-r3-postboard`, `r3-postboard` | `r1-lranneal` | `mu-jund-blue` | 1.00M | 74.9 / 74.9 | 154.2 | superseded | `20261007-r3-postboard` | [entry](ledger.md#20261007-r3--fine-tunes-from-the-lr-anneal-checkpoint) |
| `r3-botjund` | `20261007-r3-fs2h128-mu-jund-blue-ft-r1-lranneal-botjund-s0` | `20261007-r3-botjund`, `r3-botjund` | `r1-lranneal` | `mu-jund-blue` | 1.00M | 74.2 / 75.0 | 142.7 | archived | `20261007-r3-botjund` | [entry](ledger.md#20261007-r3--fine-tunes-from-the-lr-anneal-checkpoint) |
| `r3-attn` | `20261007-r3-fs2h128-mu-jund-blue-ft-r1-lranneal-attn-s0` | `20261007-r3-attn`, `r3-attn` | `r1-lranneal` | `mu-jund-blue` | 1.00M | 73.3 / 75.0 | 152.2 | archived | `20261007-r3-attn` | [entry](ledger.md#20261007-r3--fine-tunes-from-the-lr-anneal-checkpoint) |
| `r2-features` | `20261007-r2-fs2h128-mu-jund-blue-ft-ov-selfplay-s0` | `20261007-r2-features`, `r2-features` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 67.0 / 72.2 | 88.9 | archived | `20261007-r2-features` | [entry](ledger.md#20261007-r2--four-fine-tunes-from-the-overnight-checkpoint-new-code) |
| `r2-botjund` | `20261007-r2-fs2h128-mu-jund-blue-ft-ov-selfplay-botjund-s0` | `20261007-r2-botjund`, `r2-botjund` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 67.5 / 75.0 | 59 | archived | `20261007-r2-botjund` | [entry](ledger.md#20261007-r2--four-fine-tunes-from-the-overnight-checkpoint-new-code) |
| `r2-pfsp` | `20261007-r2-fs2h128-mu-jund-blue-ft-ov-selfplay-pfsp-s0` | `20261007-r2-pfsp`, `r2-pfsp` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 63.5 / 72.0 | 86.2 | archived | `20261007-r2-pfsp` | [entry](ledger.md#20261007-r2--four-fine-tunes-from-the-overnight-checkpoint-new-code) |
| `r2-automana` | `20261007-r2-fs2h128-mu-jund-blue-ft-ov-selfplay-automana-s0` | `20261007-r2-automana`, `r2-automana` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 64.7 / 71.6 | 82.7 | archived | `20261007-r2-automana` | [entry](ledger.md#20261007-r2--four-fine-tunes-from-the-overnight-checkpoint-new-code) |
| `r1-control` | `20261007-r1-fs1h128-mu-jund-blue-ft-ov-selfplay-ctl-s0` | `20261007-r1-control`, `r1-control` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 65.9 / 71.9 | 103.2 | archived | `20261007-r1-control` | [entry](ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) |
| `r1-lranneal` | `20261007-r1-fs1h128-mu-jund-blue-ft-ov-selfplay-lranneal-s0` | `20261007-r1-lranneal`, `r1-lranneal`, `lranneal` | `ov-selfplay` | `mu-jund-blue` | 1.00M | 72.1 / 73.1 | 147.4 | superseded | `20261007-r1-lranneal` | [entry](ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) |
| `r1-gammaturn` | `20261007-r1-fs1h128-mu-jund-blue-ft-ov-selfplay-gammaturn-s0` | `20261007-r1-gammaturn`, `r1-gammaturn` | `ov-selfplay` | `mu-jund-blue` | 1.00M |  | 34.6 | archived | `20261007-r1-gammaturn` | [entry](ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) |
| `r1-exploit-jund` | `20261007-r1-fs1h128-dk-jund-ft-ov-selfplay-exploit-s0` | `20261007-r1-exploit-jund`, `r1-exploit-jund` | `ov-selfplay` | `dk-jund` | 0.31M |  |  | archived | `20261007-r1-exploit-jund` | [entry](ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) |
| `r1-exploit-blue` | `20261007-r1-fs1h128-dk-blue-ft-ov-selfplay-exploit-s0` | `20261007-r1-exploit-blue`, `r1-exploit-blue` | `ov-selfplay` | `dk-blue` | 0.31M |  |  | archived | `20261007-r1-exploit-blue` | [entry](ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) |
| `fs4-ctl` | `20261007-fs4-fs3h128-mu-jund-blue-ft-r4-control-ctl-s0` | `20261007-fs3-ctl`, `fs3-ctl` | `r4-control` | `mu-jund-blue` | 5.00M | 77.4 / 78.9 | 190.1 | archived | `20261007-fs3-ctl` | [entry](ledger.md#20261007-fs4--feature-set-4-fine-tune-vs-a-feature-set-3-control) |
| `fs4-ft` | `20261007-fs4-fs4h128-mu-jund-blue-ft-r4-control-s0` | `20261007-fs4-ft`, `fs4-ft`, `fs4`, feature set 4 | `r4-control` | `mu-jund-blue` | 5.00M | 78.8 / 80.3 | 212.5 | archived | `20261007-fs4-ft` | [entry](ledger.md#20261007-fs4--feature-set-4-fine-tune-vs-a-feature-set-3-control) |
| `search64` | `20261006-s64-fs1h128-mu-jund-blue-ft-model_v3-entity-500k-bot-s6-gumbel-s0` | `20261006-search64`, `search64`, PR #10 |  | `mu-jund-blue` | 0.30M |  |  | rejected | not archived | [entry](ledger.md#20261006-search64--own-turn-search-distillation-pr-fmssnmtg-ml10-closed-unmerged) |
| `ov-selfplay` | `20261006-ov-fs1h128-mu-jund-blue-scratch-s0` | `20261006-overnight-selfplay`, `overnight`, `overnight-selfplay` |  | `mu-jund-blue` | 8.40M | 64.6 / 70.9 | 103.2 | superseded | `20261006-overnight-selfplay` | [entry](ledger.md#20261006-overnight--ab-open-ended-self-play-vs--bot-games) |
| `ov-bot10` | `20261006-ov-fs1h128-mu-jund-blue-scratch-bot10-s0` | `20261006-overnight-bot10`, `bot10` |  | `mu-jund-blue` | 8.40M | 65.6 /  | 74.8 | archived | `20261006-overnight-bot10` | [entry](ledger.md#20261006-overnight--ab-open-ended-self-play-vs--bot-games) |

### Throughput screens (no strength result)

| handle | run ID | legacy names | parent | scope | games | bench s/g | L1 | status | archive (`h100-private:~/mtg-ml-checkpoints/`) | ledger |
|---|---|---|---|---|---|---|---|---|---|---|
| `scr-batch8192-lr150` | `20261008-scr-fs6h256-all-ft-r6-h256-batch8192-lr150-s0` | `20261008-h256-screen-batch8192-lr150-seed0`, `h256-screen-batch8192-lr150` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-batch8192-lr150-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-batch8192-lr300` | `20261008-scr-fs6h256-all-ft-r6-h256-batch8192-lr300-s0` | `20261008-h256-screen-batch8192-lr300-seed0`, `h256-screen-batch8192-lr300` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-batch8192-lr300-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-bf16` | `20261008-scr-fs6h256-all-ft-r6-h256-bf16-s0` | `20261008-h256-screen-bf16-seed0`, `h256-screen-bf16` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-bf16-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-epochs2` | `20261008-scr-fs6h256-all-ft-r6-h256-epochs2-s0` | `20261008-h256-screen-epochs2-seed0`, `h256-screen-epochs2` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-epochs2-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-lag2` | `20261008-scr-fs6h256-all-ft-r6-h256-lag2-s0` | `20261008-h256-screen-lag2-seed0`, `h256-screen-lag2` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-lag2-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-legacy` | `20261008-scr-fs6h256-all-ft-r6-h256-legacy-s0` | `20261008-h256-screen-legacy-seed0`, `h256-screen-legacy` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-legacy-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-resident128` | `20261008-scr-fs6h256-all-ft-r6-h256-resident128-s0` | `20261008-h256-screen-resident128-seed0`, `h256-screen-resident128` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-resident128-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-resident64` | `20261008-scr-fs6h256-all-ft-r6-h256-resident64-s0` | `20261008-h256-screen-resident64-seed0`, `h256-screen-resident64` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-resident64-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-separate` | `20261008-scr-fs6h256-all-ft-r6-h256-separate-s0` | `20261008-h256-screen-separate-seed0`, `h256-screen-separate` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-separate-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |
| `scr-stacked` | `20261008-scr-fs6h256-all-ft-r6-h256-stacked-s0` | `20261008-h256-screen-stacked-seed0`, `h256-screen-stacked` | `r6-h256` | `all` |  |  |  | archived | `20261008-h256-screen-stacked-seed0` | [entry](ledger.md#20261008-h256-attention-scaling--implementation-and-completed-throughput-screens) |

## Not models

Evaluation-only ledger entries (`20261009-specialist-benchmark`, `20261009-r8-evals-1`, the offline probes) and the r8 launch records have no weights of their own and are not listed. The exploiter runs (`r1-exploit-*`) are listed because they are archived.
