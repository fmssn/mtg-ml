# Model registry

One row per run, every archived run plus the r8 runs, with the handle (see [naming.md](naming.md)), the immutable run ID, **all legacy names** so nothing gets lost, and where the weights and the evidence are. Machine-readable twin: [`models.json`](models.json) (keep both in sync; `tests/test_model_registry.py` checks that they agree). Dated 2026-10-10; games and evals of running runs are the last values read from their `metrics.jsonl` that day.

Numbers: bench = learner Jund vs the legacy Blue bot, game 1, sampled / greedy (1,000 games in training, 2,000 after the run, see the [README](README.md)); L1 = Elo on ladder L1 ([ladder.md](ladder.md), SE about 12), at the end of the run unless the column gives the game count (`@17.5M` is the last evaluation, not the end). Runs on different feature sets and matchups share these two numbers only loosely. Seeds before r7 were not recorded (shown as s0, the flag default). The archive column is relative to `h100-private:~/mtg-ml-checkpoints/`.

## Current roles

| role | handle | why |
|---|---|---|
| `best/general` | `r7-lr075` | Only full-matrix model; r8-continue plateaued; r8-belief would replace it if it overtakes at matched games |
| `best/jund_blue` | `fs4-ft` | Highest L1 (212.5) among archived Jund vs Blue models. Candidates: r8-jund-blue (running, 77.9/82.1, L1 156) and r4-control. fs4-ft was never run against the specialists. |
| `best/blue_vs_jund` | `r4-control` | Blue as learner vs the Jund specialist 85.3/87.0 (specialist benchmark); r8-jund-blue 78.2/80.5 at 1.56M |
| `best/jund_jund` | `r7-lr075` | Mirror vs the Jund specialist 57.8/68.5; candidate r8-jund-mirror (running, 2M schedule) |
| `best/jund` | `r7-lr075` | Deck vs field; candidate r8-jund-pilot (running, 3M schedule) |
| `best/blue` | `r7-lr075` | No deck-specific Blue model yet |
| `best/madness` | `r7-lr075` | rm-1m is a one-matchup baseline, not a full model |
| `best/affinity` | `r7-lr075` | Only full-matrix model |
| `best/elves` | `r7-lr075` | Only full-matrix model |
| `best/tron` | `r7-lr075` | Only full-matrix model; the trust test flags Tron as overrated |
| `play/jund` | `r8-jund-blue` | Play site offer r8-jund, greedy (default Jund opponent); r7-jund stays as a fallback |
| `play/blue` | `r7-lr075` | Play site offer r7-blue, greedy |
| `play/madness` | `r7-lr075` | Play site offer r7-madness, greedy |
| `play/affinity` | `r7-lr075` | Play site offer r7-affinity, greedy |
| `play/elves` | `r7-lr075` | Play site offer r7-elves, greedy |
| `play/tron` | `r7-lr075` | Play site offer r7-tron, greedy |

| in flight | compared against | question |
|---|---|---|
| `r8-belief` | `r7-lr075` | does feature set 8 plus the belief head beat the lr075 trajectory at matched games |
| `r8-jund-blue` | `r4-control` | does Jund-vs-Blue fine-tuning reach the r4-control and fs4-ft level; replicated by r8-jund-blue-s2 |
| `r8-jund-mirror` | `r7-lr075` | does matchup fine-tuning work for the mirror (benchmark-jund@1) |
| `r8-jund-pilot` | `r8-jund-blue` | deck-vs-field fine-tuning against per-matchup fine-tuning |

Roles are pointers: change them here and in `models.json` (`roles`, plus a line in `role_history`) when a better model is established, and say why in the ledger. Open: `best/jund_blue` rests on L1 and bench only, because fs4-ft was never run against the Jund and Blue specialists; r8-jund-blue is the contender once it has finished (6M games) and been benchmarked. The play site pins `r7-lr075/policy`, `r8-jund-blue/policy` (the `r8-jund` opponent) and the legacy `r4-control/policy` in `mtg_ml/play_config.toml`; those keys are not renamed.

## Registry

Status: `running`, `stopped` (ended by hand), `archived`, `superseded` (archived and replaced by a better parent, still a valid reference), `rejected`, `crashed`.

### r8 (running and finished 2026-10-09 to 10)

| handle | run ID | legacy names | parent | scope | games | bench s/g | L1 | status | archive (`h100-private:~/mtg-ml-checkpoints/`) | ledger |
|---|---|---|---|---|---|---|---|---|---|---|
| `r8-continue` | `20261009-r8-fs7h256-all-ft-r7-lr075-continue-s8` | `r8-lr075-continue`, `r8-20261009/r8-lr075-continue`, `20261009-r8-lr075-continue`, arm A, `A` | `r7-lr075` | `all` | 10.05M | 56.7 / 69.2 | 37.3 @10.0M | stopped | `20261009-r8-lr075-continue` | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-belief` | `20261009-r8-fs8h256-all-scratch-belief-s8` | `r8-fs8-belief`, `r8b-20261009/r8-fs8-belief`, `20261009-r8-fs8-belief`, `r8b`, `r8b-20261009`, arm B, `B` |  | `all` | 3.64M | 49.0 / 62.0 | -51.8 @3.5M | running | not archived yet | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-belief-x1` | `20261009-r8-fs8h256-all-scratch-belief-s8.x1` | r8-fs8-belief (first attempt), `r8-20261009/r8-fs8-belief` |  | `all` | 0.11M |  |  | crashed | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-blue` | `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s8` | `r8-jund-blue-ft`, `r8-20261009/r8-jund-blue-ft`, `20261009-r8-jund-blue-ft`, arm C, `C` | `r7-lr075` | `mu-jund-blue` | 3.95M | 77.9 / 82.1 | 156.1 @3.75M | running | not archived yet | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-blue-s2` | `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s9` | `r8-jund-blue-ft-s2`, `r8d-20261009/r8-jund-blue-ft-s2`, `20261009-r8-jund-blue-ft-s2`, arm C-s2, `C-s2`, `r8d`, `r8d-20261009` | `r7-lr075` | `mu-jund-blue` | 1.02M | 68.8 / 79.9 | 112.2 @1.0M | archived | `20261009-r8-jund-blue-ft-s2` | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-mirror` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8` | `r8-jund-mirror-ft`, `r8e3-20261009/r8-jund-mirror-ft`, `20261009-r8-jund-mirror-ft`, arm D, `D`, `r8e3`, `r8e3-20261009` | `r7-lr075` | `mu-jund-jund` | 0.78M |  |  | running | not archived yet | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-mirror-x1` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8.x1` | `r8e-20261009`, `r8e`, `r8e-20261009/r8-jund-mirror-ft` | `r7-lr075` | `mu-jund-jund` | 0.08M |  |  | crashed | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-mirror-x2` | `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8.x2` | `r8e2-20261009`, `r8e2`, `r8e2-20261009/r8-jund-mirror-ft` | `r7-lr075` | `mu-jund-jund` | 0.06M |  |  | stopped | not archived | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |
| `r8-jund-pilot` | `20261009-r8-fs7h256-dk-jund-ft-r7-lr075-s8` | `r8-jund-pilot`, `r8f-20261009/r8-jund-pilot`, `20261009-r8-jund-pilot`, `pilot`, Jund pilot, `r8f`, `r8f-20261009` | `r7-lr075` | `dk-jund` | 1.13M | 60.4 / 71.9 | 28.5 @1.0M | running | not archived yet | [entry](ledger.md#20261009-r8--three-arm-follow-up-to-r7-continue-lr075-feature-set-8--belief-head-jund-vs-blue-fine-tune) |

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
