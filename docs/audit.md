# Indistinguishability audit (`mtg_ml.audit`)

Step 2 of the representation plan: find observation blind spots without
reading games. A **blind spot** is a decision where two options get exactly
the same network input but lead to different games. The policy scores options
one by one from their inputs, so it gives such options the same probability
whatever it learns. The motivating case: two Tolarian Terrors attack, one is
already blocked, and in feature set 3 "block Terror A" and "block Terror B"
are bit-identical, so the policy splits 0.50/0.50 and stacks the second
blocker on the blocked one half the time.

## What it measures

At every audited decision:

1. **Group options by their featurization** in the decider's feature set
   (`rl.features.featurize`, so both engines and every set): the option's
   hashed tokens, plus for each option->entity pointer the token bag of the
   entity it points at. Two options pointing at two entities with the same
   features count as identical. The state and entity bags are shared by all
   options of a decision; the model is permutation-invariant over entities and
   pointers (no positional input), so nothing else separates two options.
2. **For each group of 2+**, fork the game (`Game.fork`, replay), step each
   option and compare the successors by what the decider can observe:
   `audit.canonical_view`.
3. Same view for every option: the options really are interchangeable
   (fine). Different views: a blind spot.

### The observable digest (`canonical_view`)

From the decider's `view.observe` after the step, so the opponent's hand
(only its size and cards revealed to the decider) and library order (only
known positions) never count:

- turn, step, active player, game over / winner, lands played;
- per side: life, library size, mulligans, hand (own: names; opponent: count
  and known cards), graveyard, exile, mana pool, known library positions;
- every permanent: name, controller, types, P/T, keywords, tapped, sick,
  damage, counters, token, attacking, skip-untap, whether it is a blocked
  attacker, and its relations (what it blocks, what it is attached to);
- the stack in order: name, kind, controller, X, method, targets;
- the next decision if it is the decider's (kind and option keys), else only
  that the opponent is deciding and its kind.

Object ids are replaced by a canonical label: each object's own attributes,
refined three rounds over the `blocks`, `attached_to` and `target` relations
(Weisfeiler-Lehman). So tapping one of two interchangeable Forests gives the
same view, while "two blockers on one Terror, none on the other" and "one
blocker on each" differ. Hand, graveyard, exile and battlefield are compared
as multisets; the stack and known library positions keep their order.

Options the engine already merged (`Game.equiv_key`: identical, unreferenced
permanents) never show up as two options, so they are not counted.

What it does not catch: options whose successors look the same right after
the step but diverge later (for example through hidden cards), and
differences between options that the policy sees as different but cannot
interpret (an X value, a basic land type): those have distinct tokens, so
they are not collisions. Both need the plan's step 3 (simulated previews).

## How to run

```bash
python -m mtg_ml.audit --games 1000 --engine native                       # scripted bots, jund_blue, latest feature set
python -m mtg_ml.audit --matchup blue_madness --games 1000 --engine native
python -m mtg_ml.audit --agents model:runs/x/final.pt,model:runs/x/final.pt --games 200 --engine native
python -m mtg_ml.audit --features 4 --out set4.json                       # any feature set, bots or models
python -m mtg_ml.audit --sample-frac 0.1 --engine python                  # audit 10% of decisions
```

- `--agents`: two of `random`, `bot`, `search[:n]`, `model:<checkpoint>` (or a
  bare `.pt` path); `--greedy` makes models play argmax.
- `--features N`: audit (and featurize model agents) in set N. Without it a
  model seat is audited in its checkpoint's recorded version and a bot seat
  in the latest (`encode.FEATURES`). Any version in
  `encode.FEATURE_VERSIONS` works, so set 4 needs no change here.
- Game i uses seed `--seed + i`, seat `i % 2` starts, the game-1 decks of
  `--matchup`.
- Rebuild an example's state: `audit.rebuild(example)` (replays its
  `actions`), then `audit.audit_decision(g, features)` or
  `rl.features.featurize(g, g.decision.player, features=...)`.

## Output (`--out`, `"format": 1`)

| key | meaning |
|---|---|
| `meta` | matchup, games, seed, engine, agents, `features` per seat, sample fraction, seconds |
| `decisions`, `audited` | all decisions played, and the sampled ones that were checked |
| `collided` | audited decisions with a group of 2+ identically featurized options |
| `blind` | audited decisions with at least one blind-spot group |
| `groups`, `blind_groups` | groups found, and those that are blind spots |
| `blind_per_1k`, `collided_per_1k` | per 1000 audited decisions |
| `by_kind` | per decision kind: `audited`, `collided`, `blind`, `blind_per_1k` (of that kind) |
| `by_seat` | the same counts per deciding seat |
| `by_card` | blind-spot groups per card name the options point at |
| `examples` | up to `--max-examples` (3 per decision kind and card set): `matchup`, `seed`, `starting_player`, `decision` (index), `turn`, `player`, `deck`, `kind`, `features`, `cards`, `options` (`index`, `key`, `label`, successor `class`), `differs` (fields of the canonical view that differ), `actions` (the prefix that rebuilds the state) |

## Baseline (2026-10-07, Apple M3 MacBook, native engine)

Scripted bots, jund_blue, 1000 games (seeds 0-999), 261k decisions, all
audited:

| feature set | blind / 1k decisions | collided / 1k | main kinds (blind / 1k of that kind) |
|---|---|---|---|
| 1 | 4.23 | 4.81 | pay_mana 16.9, declare_blocker 80.1, target 12.6 |
| 2 | 1.85 | 2.40 | declare_blocker 80.1, target 3.6, priority 0.26 |
| 3 | 1.85 | 2.40 | same as set 2 (set 3 only adds `opp:deck:`, absent on jund_blue) |

Set 3 by decision kind: declare_blocker 412 blind decisions of 5144 (8%),
target 25 of 6972, priority 47 of 181091; nothing else. By card (blind
groups): Tolarian Terror 223, Cryptic Serpent 204, Eldrazi Spawn 201,
Krark-Clan Shaman 93, Writhing Chrysalis 77. Every one is a combat relation
the features do not carry:

- **declare_blocker**: two same-name attackers, one already blocked (the
  handoff's P1). `seed 10 d284`: `Eldrazi Spawn#196 blocks Tolarian Terror#206`
  vs `... Tolarian Terror#233`, #206 already blocked.
- **priority**: `Eldrazi Spawn: sacrifice: add C` on two Spawns blocking
  different attackers (`e:blocking` does not say what is blocked).
- **target**: removal on one of two same-name attackers, one blocked, or on
  one of two blockers that block different attackers.

Set 1's extra `pay_mana` blind spots are bridges targeted by Cleansing
Wildfire: tapping the targeted bridge or the other one looked the same until
set 2 added `e:targeted_by`, which the audit confirms closed them.

Other matchups (bots, set 3, 1000 games): jund_madness 0.28 / 1k
(declare_blocker 17.7 / 1k of that kind), blue_madness 0.91 / 1k
(declare_blocker 59.2 / 1k).

r4-control (`r4-control-v07714.pt`, set 3), self-play on jund_blue:

| play | games | decisions | blind / 1k | by kind (blind / 1k of that kind) |
|---|---|---|---|---|
| sampled | 400 (seeds 0-399) | 114k | 2.19 | declare_blocker 88.7, pay_mana 0.90, priority 0.68, target 2.5, sacrifice 1.2 |
| greedy (`--greedy`) | 200 (seeds 0-199) | 58k | 2.80 | declare_blocker 114.5, pay_mana 1.16, priority 0.75, target 5.6 |

The model blocks more than the bots, so it meets the same-name attacker case
more often (about one block decision in nine under greedy play). Its extra
`pay_mana` / `sacrifice` cases are Eldrazi Spawns sacrificed for mana while
blocking different attackers.

Runtime on the M3, bots included: native ~0.05 s per 1k decisions,
Python ~0.36 s per 1k; with model agents the network dominates (~1 ms per
decision). Forks are only made for collided decisions (~2 per 1k), so
`--sample-frac` matters only for the Python engine or very large runs.
