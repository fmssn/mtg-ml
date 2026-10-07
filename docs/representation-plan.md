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

Each step is its own PR and its own ledger entry in `docs/experiments/`. Steps 3 to 5 change the
model's inputs enough that a fresh run is likely cheaper than stacking fine-tunes; decide per step
from the ledger.

## What this does not solve

- **Engine cost per deck.** Each new deck needs its cards implemented in both engines
  ([adding-cards.md](adding-cards.md)). Once features stop being per-card, that is the bottleneck
  for adding decks. Track new ops per deck; a deck that needs no new ops is nearly free.
- **Strategy.** Land-before-spell sequencing, passing with castable spells, flyers not attacking:
  the handoff lists these as training-side. Better inputs make them learnable, not learned.
- **Multi-turn combos.** The Shaman + Toxin sweep needs a plan over several decisions. Simulated
  previews show one step ahead; the exploration problem (the policy plays the sweep with
  p ≈ 1e-7, and search distillation in PR #10 did not fix it) stays.
