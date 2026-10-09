# Representation plan: features that scale to the Pauper meta (2026-10-07)

The game review of r4-control (PR #31,
[handoff-2026-10-07-feature-set-4.md](handoff-2026-10-07-feature-set-4.md) on that branch) found
about a dozen observation gaps, each fixed by a hand-written token: `e:blocked`, `pv:attacker_dies`,
`pv:x_is_max`, `pv:reveal_type`. That works for four decks. It does not work for the ~25 decks of
the Pauper meta and their 300+ unique cards: every deck brings new interactions, every fix is a new
feature set, and every feature set means a fine-tune because new rows start untrained. This plan
says how to stop adding features by hand after set 4.

## Two kinds of gap

| kind | examples | scales? |
|---|---|---|
| **rules-level**: the same in every deck | who blocks whom, damage still coming in, the chosen X, colours a source makes and the hand needs, what a target points at | yes: combat, the stack and mana work the same in every deck, so a fix is written once |
| **card-level**: one interaction of specific cards | Delver's reveal, Munitions-style reach, the Krark-Clan Shaman + Toxin Analysis sweep, "whenever you sacrifice" payoffs | no: grows with every deck added |

Most of the handoff's P1 to P3 is rules-level and stays worth doing. The card-level items are the
ones to stop writing by hand.

## The plan

### 1. Feature set 4, rules-level part only

Take from the handoff: P1 (combat relations, block previews), P2 (incoming damage), the `choose_x`
and colour parts of P3. Leave out the Delver reveal and P4's card-specific reach; step 3 covers them
generically. Set 4 should be the last set made of hand-written per-decision tokens.

### 2. Indistinguishability audit: find blind spots mechanically

The handoff's worst finding (two same-name attackers, policy split exactly 0.50/0.50, the second
blocker stacked on the wrong one) is detectable without an LLM: at a decision, two options whose
featurization is identical but whose engine successor states differ. Add a tool that, over N
self-play games per matchup:

- for every decision, groups options by their exact feature tokens (state is shared, so the option
  tokens and option pointers decide);
- for each group of 2+, steps a fork with each option and compares the successor states (a state
  digest, ignoring hidden information the decider cannot see);
- reports blind spots per 1k decisions, by decision kind and card, with a replay pointer for each.

This gives every deck a number before training, turns each finding into a regression test (the
options' token sets must differ), and leaves the LLM review for strategy questions. Runs on the
Rust engine, where `fork()` replays at ~190k decisions/s per core; that is enough for an audit.

### 3. Simulated option previews

Today each decision kind needs its own preview code (`option_preview` handles cast/activate and
target; blocks, X, payment and yes/no get nothing). Replace hand-written `pv:` tokens with one
mechanism: apply the option to a copy of the state, advance until the next decision of either
player (or, for combat choices, through combat damage with no further choices), and featurize the
difference:

- life change per side, poison;
- own and opposing creatures that died, by power tier;
- cards drawn / discarded / exiled per side;
- mana and colours left to the decider;
- opponent's lethal-next-turn and own lethal flags before vs after.

Covers blocks, X, colour payment, fetches, Delver and every future card with no per-card code.
Rules:

- **Hidden information:** advance only through deterministic, public steps. Stop at a draw, a
  reveal of a hidden card, a shuffle, or any opponent decision. Never featurize the opponent's
  hand or library order. A delta that would depend on hidden cards is not computed.
- **Cost:** one copy plus a short step per option per decision, so it needs O(1) state copy in Rust
  (`Game` is not `Clone` today; `fork()` replays the whole history). Deriving `Clone` on the Rust
  `Game` and its parts, plus a Python-side copy for the reference engine, is the infrastructure
  piece. It also unblocks search, if search is ever retried with a new idea (PR #10's lessons
  stand).
- **Parity:** both engines compute the delta, difftest compares it, same as every feature.

Existing set-2 previews (`pv:kills_*`, `pv:damage_lethal_to_target`, `pv:mana_left_after`,
`pv:colors_left`) already read card-spec ops, not names; they become special cases of the simulated
delta and can stay for older sets.

### 4. Cards described by what they do

Entities carry `e:name:<card>` plus types, keywords and P/T tiers. Name is the only feature that
says what an ability or trigger does, so a card the model never saw is close to blank. Derive
entity tokens from the card spec in `cards.toml`, which is already structured:

- ability shapes: `e:ab:tap`, `e:ab:sac_self`, `e:ab:mana:{C}`, `e:ab:op:{op}` per effect op;
- trigger shapes: `e:trig:{event}`, `e:trig:op:{op}`;
- cost shapes: mana value tier, colours, `flashback`, `madness`, `escape`, alternative costs;
- target kinds.

Keep `e:name` so known cards keep their learned identity; the shape tokens let a new deck's cards
start partly familiar. Also make hand cards entities (docs/features.md: "hand cards are still not
entities, so cast options point at nothing"), with the same tokens, for the decider's own hand.

### 5. Relations as pointers, read by entity attention

Blocking, targeting, attachment and "this X belongs to that spell" should be links between entities,
not name strings. Set 4's blocker pointer is the first; generalize to a small relation table
(`blocks`, `targets`, `attached_to`, `x_of`) that the encoder turns into attention biases or
pointer features. With the relation in the input, entity attention (`entity_attn >= 1`) can learn
summed blocker power and similar aggregates, so they need no hand-written token.

## Order and gates

| step | depends on | gate before the next step |
|---|---|---|
| 1. set 4, rules-level | PR #28 merged | set 1-3 digests unchanged; P1/P2 fixtures distinguishable; fine-tuned r4-control not worse on the benchmark |
| 2. audit | nothing (can run in parallel with 1) | baseline blind-spot rate per matchup on set 3 and set 4; set 4 lowers it on P1 kinds |
| 3. `Clone` in Rust, simulated previews (set 5) | step 2 to measure it | blind-spot rate down across all decision kinds; benchmark and ladder Elo not worse |
| 4. card-shape tokens, hand entities (set 5 or 6) | none | a held-out deck (cards not seen in training) plays better from step 0 than with names only |
| 5. relation table + entity attention | 1, 4 | fresh run with `entity_attn >= 1` beats the set-4 run at equal steps |
| 6. fresh six-deck run on set 7 (r7 already is one; set 7 is the first set with the hidden-list contract) | 3, 4 (set 6), all six decks and sideboards in both engines | benchmark and ladder Elo not worse than the best set-4/5 run on the matchups they share |
| 7. trust test | 6 | most pairings inside Pauper-Research's 95% interval, no per-deck bias, not exploitable by the frozen pool (see the milestone below) |
| 8. first card-choice experiment (Tron with vs without Giant's Boulder) | 7 passed | delta reported with its interval and cross-checked against Pauper-Research's with/without-card numbers |

Each step is its own PR and its own ledger entry in `docs/experiments/`. Steps 3 to 5 change the
model's inputs enough that a fresh run is likely cheaper than stacking fine-tunes; decide per step
from the ledger.

## What this does not solve

- **Engine cost per deck is small; correctness is the cost.** Agents implement a deck's cards in
  both engines from `data/oracle_cards.json`, [adding-cards.md](adding-cards.md) and XMage's
  implementations as a reference; difftest proves the engines agree, not that they follow the rules.
  Guard correctness with per-card rules tests written from the oracle text and LLM transcript
  reviews (`mtg_ml.review`). Agents should also turn `custom` ops (9 of 134 cards today) into shared
  ops, which keeps step 4's shape tokens covering new cards.
- **Training compute per deck.** 25 decks are ~325 pairings, and each new deck needs games against
  all the others. This, not card code, is what grows; it argues for one deck-conditioned model and
  for step 4's transfer.
- **Strategy.** Land-before-spell sequencing, passing with castable spells, flyers not attacking:
  the handoff lists these as training-side. Better inputs make them learnable, not learned.
- **Multi-turn combos.** The Shaman + Toxin sweep needs a plan over several decisions. Simulated
  previews show one step ahead; the exploration problem (the policy plays the sweep with
  p ≈ 1e-7, and search distillation in PR #10 did not fix it) stays.

## Next milestone: a model you can trust for card choices

### Goal

The purpose behind this work is the user's Pauper-Research project
([github.com/fmssn/Pauper-Research](https://github.com/fmssn/Pauper-Research)): use a strong model to
evaluate card choices and sideboard changes where tournament data is too thin to answer them. A card
swap moves a deck's win rate by about 1 to 3 points, which is smaller than most of the model's
current errors. So the model has to be strong, consistent across decks (a deck it plays worse gets
worse-looking cards), and validated against real results before any card conclusion is drawn from it.

### Trust test

The gate before any card conclusion (step 7 below):

1. Train one deck-conditioned model on feature set 7 (or 8) with all six decks (Jund Wildfire, Mono Blue
   Terror, Red Madness, Grixis Affinity, Elves, Tron) and their real sideboards. r7 (feature set 7,
   `docs/experiments/r7-overnight.md`) is already such a run. Sets below 7 disclose the opponent's
   archetype, so their results are diagnostic only (`--contract diagnostic` in `mtg_ml.benchmark`).
2. Play the full 6x6 matchup matrix as best-of-three with sideboarding, both seats, with enough
   matches for ±2 points per pairing (about 2,400 matches per pairing at 95%).
3. Compare with Pauper-Research's Q3 2026 combined matchup matrix
   (`reports/2026-Q3/combined/matchups.csv`, which carries Wilson intervals).
4. Play the final model against a frozen pool of its own past versions to check how exploitable it is.

| check | pass |
|---|---|
| per pairing | most pairings inside the real data's 95% interval |
| per deck | no systematic bias: no deck that over- or under-performs the real data across most of its pairings |
| exploitability | no past version in the frozen pool beats the final model by a clear margin in any matchup |

The per-deck check matters most. A deck that consistently under-performs means the model plays it
worse than people do, and every card conclusion about that deck would carry that bias. Real matchup
samples are small (±17 points at 30 matches), so the test catches bias and gross errors, not exact
values.

### Card-choice protocol

Variant decks differ from the base list by a few cards. For each variant:

1. Fine-tune from the base model for a fixed budget, the same for every variant, so the model learns
   the new card instead of playing it with untrained rows.
2. Play about 10k games per matchup against each opposing deck, both seats, best-of-three.
3. Report the win-rate delta against the base list per matchup with its confidence interval (10k
   games is about ±1 point per side, so a 1-point effect needs more games or a paired design), and
   one "against the field" number weighted by Pauper-Research meta shares.
4. Cross-check with Pauper-Research's with/without-card statistics
   (`python -m pauper_research cards <archetype>`). Agreement in sign is the minimum; a disagreement
   is a finding to look into, not a result to publish.

First experiment: Tron with vs without Giant's Boulder (step 8).

### Known risks and what addresses them

| risk | what addresses it |
|---|---|
| self-play exploitability: the model beats itself but has holes a different player finds | league training against a population (partly in place through pool snapshots); the trust test's frozen-pool check |
| hidden information: the model does not reason about the opponent's hand | opponent-hand modelling, starting with the opponent-action head from the MageZero borrow plan |
| multi-turn combos never explored (Krark-Clan Shaman + Toxin Analysis) | open: search distillation failed in PR #10, so this needs a new idea; card conclusions about combo pieces stay unreliable until then |
| skill bias per deck | the trust test detects it; fix by deck-specific training budget before drawing conclusions about that deck |

### What limits the pace

Not implementation work: agents implement decks, cards and features quickly. The limits are H100 box
time and the number of research iterations. At today's speed one run produces about 1.3M games per
hour, so a six-deck run of 50 to 100M games takes 1 to 3 days of box time. Each failed trust test
costs at least one more such run. Estimates for this milestone are therefore in box hours and
iterations (runs until the trust test passes), not calendar months.
