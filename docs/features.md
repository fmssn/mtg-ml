# Network input features

What the policy sees at a decision point. Everything is a string hashed with
CRC32 into a fixed index space (`STATE_DIM = 2^16`, `OPTION_DIM = 2^15`), so
adding a feature changes no dimension and old checkpoints still load.

## Feature-set versions

New strings would land on embedding rows a trained model never learned
(random init, or rows it learned for colliding strings): checkpoints from
before set 2 lost ~1.3 benchmark points when evaluated with it. So the
feature set is versioned and travels with the model:

| version | contents |
|---|---|
| 1 | everything up to 2026-10-06 |
| 2 | + the readiness/lethal and known-position state features, the `e:skip_untap` / `e:targets` / `e:targeted_by` / `e:x` entity features and the `pv:` option previews below |

- `PolicyNet(features=...)` stores it in `config["features"]`, only when it
  is not 1, so a config without the key (every checkpoint before this) is
  set 1.
- Training (`--features`, default 0 = unset): a new run takes the latest
  (2); `--init` and `--exploit` keep the source checkpoint's version (no
  weights depend on it, so `--features N` may move a fine-tune to another
  set, but then the parent sees inputs it never learned); a resumed run keeps
  its checkpoint's, and `--features N` on a resume overrides it and writes it
  into every file the run saves from then on. A resumed run's pool snapshots
  that record no version are played in the run's version.
- Play and evaluation read each checkpoint's version and take an override
  for checkpoints that record the wrong one: `mtg_ml.rl.evaluate
  --features N` (the evaluated checkpoint) and `--opponent-features N`,
  `mtg_ml.rl.evaluate ladder --features N` (every rung), `mtg_ml.play
  --features N` (every `model:` agent), `ModelAgent(features=N)`,
  `Job.features` ({policy: version}). The trainer's `--ladder` rungs and
  `--exploit` opponent take no override: stamp them (below).
- The encoders take it: `encode.state_features / entity_features /
  option_preview / encode_state`, `rl.features.featurize(_flat)` (keyword
  `features`, default the latest, `encode.FEATURES`), and the native
  `featurize` / `state_features` / `entity_features`.
- Every network seat is featurized with its own policy's version: rollouts
  (learner, pool snapshots, ladder rungs, local or server inference) read it
  from each checkpoint with `rollout.policy_features` (no torch needed), and
  `ModelAgent` (replays, `play`) from its network. Scripted bots don't read
  features.
- `tests/data/features_v1_digests.json` holds digests of the set-1 output
  recorded with the code before set 2 existed;
  `test_feature_set_1_reproduces_the_features_before_set_2` checks both
  engines against it, and `make difftest` compares both versions.

### Models trained on set 2 without the key

Set 2 was on from PR #19's merge, but checkpoints only record the version
since this change, so every model trained in between learned set 2 and reads
as set 1 (see `docs/experiments/ledger.md`):

- round 2: `20261007-r2-features`, `r2-botjund`, `r2-pfsp`, `r2-automana`
- round 3: `20261007-r3-control` (the current best parent), `r3-postboard`,
  `r3-botjund`, `r3-attn` (still training at the time)
- `20261007-red-madness-1m`

Before one of them is used as a parent (`--init`, `--exploit`), a ladder
rung or an opponent, write a stamped copy and use that. The archive on
h100-private is append-only, so stamp next to it, never over it:

    python tools/stamp_features.py <archive>/final.pt <dir>/final.f2.pt --features 2

`tools/stamp_features.py` copies a full checkpoint (`latest.pt`, `final.pt`)
or a policy / pool file with `config["features"]` set, refuses an existing
destination and a source that already records another version, and never
touches the source. For a one-off evaluation, `--features 2` does the same
without a copy. A run of this list resumed in place (`r3-attn`) takes
`--features 2` once; its pool snapshots then play in set 2, as they did
when the code featurized every seat in the latest set.

Both engines produce the same strings: `mtg_ml/encode.py` and
`mtg_ml/rl/features.py` (Python reference), `native/src/features.rs` (Rust).
`make difftest` compares `featurize()` in lockstep, and
`tests/test_difftest.py::test_entity_and_preview_strings_identical` compares
the strings behind it. Numbers are thermometers (`name>=k` for every step
`k <= n`), so a summed embedding grows with the value.

## State (`encode.state_features`)

Global features of what the viewer can see: step, active player, turn, life,
library, hand, graveyard and exile sizes, cards in hand / graveyard / exile,
known library cards, floating mana, per-type board counts, total power, stack
size. Added in set 2 (representation items 10 and 11):

| feature | meaning |
|---|---|
| `{side}:ready_power>=k` | power that can attack: the active side's untapped, non-sick creatures; for the other side every creature that will untap in its next untap step (not tapped with `skip_untap`) |
| `{side}:ready_evasive_power>=k` | the part of it no untapped creature of the defender can block (`Game._can_block`: flying vs flying / reach), all of it if the defender has no untapped creatures |
| `{side}:potential_blockers>=k` | that side's untapped creatures |
| `{side}:lethal_on_board` | ready power >= the other side's life |
| `{side}:evasive_lethal_on_board` | ready evasive power >= the other side's life |
| `{side}:untapped_mana>=k` | untapped mana sources (`Game.mana_sources`: lands, Eldrazi Spawn) |
| `{side}:known_library:{pos}:{name}` | a known library card and its position from the top (`pos < 8`), or `bottom` for the last card (Deem Inferior, Brainstorm, Ponder, scry) |

`{side}` is `self` or `opponent`, relative to the viewer.

## Entities (`encode.entity_features`)

One feature list per permanent (battlefield order) and stack item (from the
top). Options point at the entities they are about. Added in set 2:

| feature | on | meaning |
|---|---|---|
| `e:skip_untap:{n}` | permanent | does not untap during its controller's next `n` untap steps (Sleep of the Dead) |
| `e:targeted_by:{rel}` / `e:targeted_by:{rel}:{name}` | permanent or spell | a stack item controlled by `rel` (`self` / `opponent`) targets it, and its name |
| `e:targets:{kind}:{rel}` | stack item | it targets a `player`, `perm` or `spell` of `rel` |
| `e:x>=k` | stack item | the chosen X |

## Option previews (`encode.option_preview`, item 12, set 2)

Engine-computed effects of an option, from the current state without
changing it, hashed into the option's tokens next to its key tokens. They read
only public objects and the decider's own cards and mana.

| token | options | meaning |
|---|---|---|
| `pv:kills_opp>=k` / `pv:kills_self>=k` | cast / activate whose ops include `damage_each_creature` (Krark-Clan Shaman) | creatures it would destroy per side (damage marked, toughness, `without` keyword, indestructible, the source's deathtouch) |
| `pv:kills_none` | same | it would destroy nothing |
| `pv:damage_lethal_to_target` | target choice for an item with `damage_target` | the damage destroys the creature (or deathtouch) or is at least the player's life |
| `pv:target_ward:{N}` | target choice | the permanent has ward {N} and is not the decider's |
| `pv:ward_payable` | same | the item's cost plus {N} is still payable from the decider's mana |
| `pv:mana_left_after:{k}` | cast / activate | untapped mana sources + floating mana - the cost's mana value (X = 0, cost reductions applied), capped at 8. An estimate: payment is a separate decision |
| `pv:colors_left:{C}` | same | colour C (WUBRG) can still be produced after paying (exact: cost + {C} is payable) |

Effects are read from the card spec ops (`cards.toml`), not card names, so a
new card that uses `damage_each_creature` or `damage_target` gets previews
without code. Not covered yet: hand cards are still not entities, so cast
options point at nothing; `destroy_target` and other removal get no lethality
preview.
