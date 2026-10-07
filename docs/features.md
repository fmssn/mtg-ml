# Network input features

What the policy sees at a decision point. Everything is a string hashed with
CRC32 into a fixed index space (`STATE_DIM = 2^16`, `OPTION_DIM = 2^15`), so
adding a feature changes no dimension and old checkpoints still load (the new
strings start as untrained embedding rows that collide with existing ones).

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
size. Added in 2026-10 (representation items 10 and 11):

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
top). Options point at the entities they are about. Added:

| feature | on | meaning |
|---|---|---|
| `e:skip_untap:{n}` | permanent | does not untap during its controller's next `n` untap steps (Sleep of the Dead) |
| `e:targeted_by:{rel}` / `e:targeted_by:{rel}:{name}` | permanent or spell | a stack item controlled by `rel` (`self` / `opponent`) targets it, and its name |
| `e:targets:{kind}:{rel}` | stack item | it targets a `player`, `perm` or `spell` of `rel` |
| `e:x>=k` | stack item | the chosen X |

## Option previews (`encode.option_preview`, item 12)

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
