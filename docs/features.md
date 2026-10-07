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
| 2 | + the readiness/lethal and known-position state features, the `e:skip_untap` / `e:targets` / `e:targeted_by` / `e:x` entity features and the `pv:` option previews below; from PR #23 the `self:deck:{deck}` state feature (a seat not on its usual deck) and the ` (plotted)` mark on exiled card names (`view._exiled`; set 1 uses the plain name, the view text keeps the mark) |
| 3 | + `opp:deck:{deck}` (the opponent not on its seat's usual deck), so one network can play every matchup of a `--matchup` mix; on jund_blue identical to set 2, so `--init <set-2 checkpoint> --features 3` changes nothing there |
| 4 | + the rules-level gaps found by the r4-control game review ([below](#feature-set-4)): combat relations (who blocks whom, blocked / unblocked attackers, block previews), incoming combat damage, `choose_x` previews and an X option -> spell pointer, mana colours (sources, hand needs, colour previews on land plays, basic searches and mana payment). Only adds strings: every set-3 string is still there. New rows start untrained, so a set-3 checkpoint needs a fine-tune with `--features 4` |
| 5 | + cards described by what they do ([below](#card-shapes-and-hand-entities-set-5)): shape tokens from the card spec on permanents and stack items, the ops a stack item will resolve, the decider's own hand cards as entities, and cast / play-land / plot options pointing at them. Step 4 of the representation plan (PR #32, `docs/representation-plan.md`) |
| 6 | + simulated option previews ([below](#simulated-option-previews-set-6)): every option of every decision is applied to a copy of the game, advanced while that needs no hidden information and no choice of the opponent, and the observable change is featurized (`pv:sim:*`: life, creatures and permanents lost / gained, zones, mana and colours left, lethal flags, why it stopped). Step 3 of the representation plan. Option tokens only, so the model is unchanged |

- `PolicyNet(features=...)` stores it in `config["features"]`, only when it
  is not 1, so a config without the key (every checkpoint before this) is
  set 1.
- Training (`--features`, default 0 = unset): a new run takes the latest
  (6); `--init` and `--exploit` keep the source checkpoint's version (no
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
  `tests/data/features_v3_digests.json` does the same for sets 2 and 3
  (recorded before set 4, all three matchups, with option previews):
  `test_feature_sets_2_and_3_are_unchanged`; `features_v4_digests.json` for
  set 4 on the same games, recorded before set 5:
  `test_feature_set_4_is_unchanged`; `features_v5_digests.json` for set 5,
  recorded before set 6: `test_feature_set_5_is_unchanged`.

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
top), and from set 5 on per card in the decider's hand. Options point at the
entities they are about. Added in set 2:

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
without code. Not covered yet: `destroy_target` and other removal get no
lethality preview. (Before set 5 hand cards were not entities, so cast
options pointed at nothing.)

To check whether a feature set lets the policy tell apart options that lead
to different games, run the indistinguishability audit ([audit.md](audit.md),
`python -m mtg_ml.audit --features N`).

## Feature set 4

Rules-level gaps from the review of 50 r4-control games (PR #31's game
review, `docs/representation-plan.md` step 1): inputs that are the same in
every deck. Card-specific gaps (Delver's reveal, Munitions-style reach) are
left to the plan's generic later steps. `tests/data/feature_fixtures.json`
holds the review decisions as constructor arguments plus action prefixes;
`tests/test_view_env.py::test_feature_set_4_*` checks them on both engines.

### Combat

At s1000 d153 two attacking Tolarian Terrors, one already blocked, were
bit-identical entities and the policy split 0.50 / 0.50 between blocking
them. Now:

| feature | on | meaning |
|---|---|---|
| `{side}:attacking_power>=k` | state | total power of the attacking creatures (`{side}`: the attacker); `{side}:ready_power` drops the attackers once they tap |
| `{side}:unblocked_power>=k` | state | power of the attackers not blocked by the blocks declared so far (`Game.blocked`: an attacker stays blocked when its blockers leave; trample excess not counted) |
| `{side}:incoming_lethal` | state | that unblocked power >= the defender's life |
| `{side}:life_after_unblocked>=k` | state | the defender's life minus it (`{side}`: the defender) |
| `e:attack_slot:{j}` | attacker | its place in the declaration order (`Game.attackers`) |
| `e:blocked` / `e:unblocked` | attacker | blocked or not |
| `e:blockers>=k` | attacker | creatures blocking it |
| `e:block_power>=k` | attacker | their summed power |
| `e:block_lethal` | attacker | that power (or a deathtouch blocker) destroys it (`_dies_to`: damage marked, indestructible) |
| `e:blocking:slot:{j}` | blocker | the `e:attack_slot` of the attacker it blocks |
| `e:blocking:name:{name}` | blocker | that attacker's name |
| `e:blocking:power>=k` / `e:blocking:toughness>=k` | blocker | that attacker's power and toughness |
| `e:blocking:kills` / `e:blocking:dies` | blocker | one on one, it destroys the attacker / the attacker destroys it |

The blocker -> attacker relation is a set of tokens on the blocker, not an
entity pointer. Entity tokens are hashed into the state embedding, and an
entity-to-entity pointer would need a new input kind in the model
(`Batch` structure, the native batcher, the inference server and new
weights), which is step 5 of the representation plan (a relation table
read by entity attention). Until then the slot pair (`e:attack_slot:{j}` on
the attacker, `e:blocking:slot:{j}` on the blocker) is the identity link
an attention layer can match, and the name and P/T tokens carry the content
for a model without attention. The option pointers of a `declare_blocker`
option already reach both the blocker and the attacker entity, so the
blocked / unblocked flags are what tells two same-name attackers apart.

`option_preview` for `declare_blocker` (the blocker is the first id in the
option label):

| token | meaning |
|---|---|
| `pv:attacker_already_blocked` | the attacker is already blocked |
| `pv:attacker_dies` | its blockers plus this one destroy it (power summed, any deathtouch) |
| `pv:blocker_dies` | the attacker's power destroys this blocker |
| `pv:unblocked_damage_left>=k` | power of the attackers still unblocked if this option is taken and no further blocks follow ("does not block": all currently unblocked) |
| `pv:lethal_left` | that power >= the decider's life |

### Choose X

`choose_x` options were hashed categories (`X=5` unrelated to `X=4`), and
the Hydra was cast with X = 0 while mana was spare.

| token | meaning |
|---|---|
| `pv:x>=k` | X as a thermometer (k = 1..10) |
| `pv:x_is_max` | the largest X offered |
| `pv:mana_left_after:{k}`, `pv:colors_left:{C}` | the cast preview's tokens with this X included in the cost (`_item_cost` of the item on top of the stack) |
| `pv:enters_power>=k` | `etb_x_counters` spells: the base power plus X |

Each X option also points at the spell or ability on top of the stack
(`option_object_ids(option, game, kind, features)`, both engines), so it
reads that entity's vector.

### Colours

Nothing said which colours a land makes or what the hand needs, so the
policy tapped the only black source for a generic and fetched Swamps with
red cards stranded in hand.

| feature | on | meaning |
|---|---|---|
| `self:sources:{C}>=k` | state | the viewer's permanents whose mana ability makes colour C (WUBRG), tapped or not |
| `self:hand_needs:{C}` | state | a card in the viewer's hand has C in its mana cost |
| `self:hand_missing:{C}` | state | needed, and no permanent of the viewer makes it |
| `e:produces:{C}` | permanent | its mana ability makes C |
| `pv:adds_color:{C}` | land play, `('search', <card>)` options | the land makes C |
| `pv:adds_missing_color` | same | one of those colours is needed by the hand and made by nothing the player controls |
| `pv:colors_left:{C}` | `pay_mana` option | C can still be produced after this unit and the rest of the cost are paid (exact: the remaining cost minus this unit plus {C} is payable without this source). The engines keep the pending payment in `Game.paying` for it; it changes no rule |

## Card shapes and hand entities (set 5)

Before set 5 the name was the only entity feature that said what an ability,
trigger or spell does, so a card the network never saw was close to blank.
Set 5 adds tokens derived from the card's spec in `cards.toml`. Both engines
compute them once per card, face and token when the pool loads
(`cards.card_shape`, `native/src/cards.rs::card_shape`;
`mtg_ml_native.card_shapes()` lists them). `e:name` stays, so known cards
keep their learned identity, and the shape tokens let a new deck's cards
start partly familiar. Two cards whose specs differ only in name (Swamp and
Vault of Whispers, or an unseen Chain Lightning and Lightning Bolt) share
every shape token.

### Shape tokens (`CardDef.shape`)

| token | from the spec |
|---|---|
| `e:mv>=k` (k = 1..7) | mana value of the mana cost (X counts 0) |
| `e:cost:x` | `{X}` in the mana cost |
| `e:color:{C}` | each colour (WUBRG) of the face |
| `e:cost:reduction:{kind}` | `cost_reduction` (affinity, spells in the graveyard, cards drawn) |
| `e:cost:additional_sac`, `e:cost:additional_sac:{filter}` | `additional_sac` |
| `e:cost:additional_discard` | `additional_discard` |
| `e:cost:{flashback,escape,madness,bestow,plot,overload}` | that alternative way to cast |
| `e:cost:flashback` + `e:cost:sac_lands` | `flashback_cost` (sacrifice lands) |
| `e:cost:alternative` + `e:cost:sac_lands` | `alternative_cost` (Fireblast) |
| `e:ward`, `e:enters_tapped`, `e:etb_x_counters`, `e:transforms` | `ward`, `enters_tapped`, `etb_x_counters`, `back` |
| `e:spell:target:{kind}` | the spell's target kinds, every mode's too |
| `e:spell:op:{op}` | the spell's ops, every mode's and the overload effect's |
| `e:spell:modal` | `modes` |
| `e:ab:zone:{battlefield,hand}` | per ability: where it is activated |
| `e:ab:mana`, `e:ab:mana:{C}` | a mana ability and each colour it makes (C for colourless) |
| `e:ab:mv:{1..3}` | the activation's mana value, capped at 3 |
| `e:ab:{tap,sac_self,sac_other,discard_self,discard_other,exile_self}`, `e:ab:sac_other:{filter}` | the activation's other costs |
| `e:ab:x`, `e:ab:sorcery_speed` | `x_target_mv` / `x_reveal`, `sorcery_speed` |
| `e:ab:target:{kind}`, `e:ab:op:{op}` | the ability's targets and ops |
| `e:trig:{event}` | per trigger: its event (`etb`, `cast`, `you_cast`, ...) |
| `e:trig:cond:{key}:{value}` | its condition (`spell:noncreature`, `sacrificed_subtype:Eldrazi`) |
| `e:trig:op:{op}` | its ops |

Every op token `{prefix}{op}` is followed by `{prefix}{op}:n>=k` (k = 1..4)
when the op has an amount `n` (damage, cards drawn, life), then by the same
under `e:op:` (`e:op:damage_target:n>=1` on Lava Dart and on Makeshift
Munitions' ability alike), then by the ops of an `optional_payment`. Every
target token `{prefix}{kind}` is followed by `e:target:{kind}` in the same way. A `custom` op is just `op:custom`: the name says the
rest. The list is a bag over all of a card's abilities and triggers, deduplicated
with the first occurrence first.

### Where they appear

| entity | set-5 tokens |
|---|---|
| permanent | its current face's shape (a transformed Delver has the Aberration's) |
| spell on the stack | its card's shape, plus `e:res:op:{op}` |
| ability or trigger on the stack | `e:res:op:{op}`: the ops it will resolve (the chosen mode's for a modal spell, the overload effect when overloaded); the engine's ward and madness triggers have none |
| hand card (new) | `e:zone:hand`, `e:name:{name}`, `e:ctrl:self`, printed `e:type:` / `e:kw:` / `e:power>=k` / `e:toughness>=k`, its shape, and `e:castable` when the viewer is at a priority decision with an option to cast it from hand |

Hand entities are the viewer's own hand only, in hand order, after the
permanents and the stack. The opponent's hand contents never appear, and its
size stays a state feature. `MAX_ENTITIES = 64` truncates in that order
(permanents, stack, hand). Cast, play-land and plot options point at their
hand card (`option_object_ids`), and hand abilities (cycling) already point
at their card, so they now reach the entity too. Cards cast from another zone
(flashback, escape, plot, madness) are not entities, so those options have no
card pointer.

The model needs no change. Entities per sample are dynamic, and a hand adds
about 7 entities of 10-20 tokens each, far below the inference server's
per-decision budget (`REQUEST_INTS_PER_GAME`).

## Simulated option previews (set 6)

Sets 2 and 4 preview options with hand-written code per decision kind
(`pv:kills_*`, `pv:colors_left:*`, block and X previews); everything else got
nothing. Set 6 adds one mechanism for every decision kind (step 3 of the
representation plan): each option is applied to a copy of the game
(`Game.copy()`, PR #36), the copy is advanced while that needs no hidden
information and no choice of the opponent, and the change the decider can
observe is featurized as `pv:sim:*` option tokens. The set 2-5 previews stay
as they are. Code: `encode.sim_previews` / `_simulate` (reference),
`native/src/sim.rs`; `encode.option_previews(game, player, features)` returns
every option's previews of a decision (the starting state is summarized
once).

### How far a simulation runs

`g = game.copy(); g.sim_viewer = decider; g.step(option)`, then:

| stop | when | tokens |
|---|---|---|
| `pv:sim:stop:hidden_info` | a library was shuffled, a card left, entered or moved in a library, or a library card the decider did not know was looked at (`_hidden_touched`: draws, mills, searches, scry, Ponder, explore; `Game.shuffles` counts shuffles). Revealing a card the decider already knows (Delver) is not hidden | this token only: nothing about the result is featurized |
| `pv:sim:stop:own_decision` | the decider's next decision | the delta below |
| `pv:sim:stop:opponent_decision` | any decision of the opponent except a forced one (below) | the delta |
| `pv:sim:stop:game_over` | the game ended | `pv:sim:won` / `pv:sim:lost` / `pv:sim:draw_game` and the delta |
| `pv:sim:stop:step_cap` | `SIM_MAX_STEPS` = 64 engine steps | the delta |
| `pv:sim:skipped` | the decision has more than `SIM_MAX_OPTIONS` = 32 options, or the game has no step-start snapshot to copy from (the mulligans, before any step began; a game edited outside `step()` since its step began) | this token only, on every option |

The opponent's decisions are taken only when they are **forced**: exactly
one option, and a kind whose options depend only on public information
(`SIM_FORCED_KINDS`: target, pay_mana, sacrifice, exile_from_graveyard,
order_triggers, declare_attacker, declare_blocker, assign_damage, choose_x).
On a played game the engine takes every single-option decision itself
(`auto_single`); on a simulation copy (`Game.sim_viewer` set) the opponent's
are asked, because whether it has a choice can depend on its hand. Priority
always stops a simulation, the decider's or the opponent's, and the engine
stops there before it lists the options: the opponent's would read its
hidden hand (an opponent holding Counterspell and one holding a land look
the same), and the decider's are not needed. So a spell the decider casts
does not resolve in its own previews (the opponent gets priority first), but
passing as the second player shows the top of the stack resolving: letting
the opponent's Lightning Bolt resolve shows the creature dying, or the life
loss.

Combat choices stop at the next priority too (after the last block the
attacker gets priority), so damage is not simulated: set 4's block previews
(`pv:attacker_dies`, `pv:blocker_dies`, `pv:unblocked_damage_left`) keep
covering it, and `pv:sim:` adds the lethal flags of the declared blocks.

### Tokens (the delta)

Computed from two summaries of what the decider can observe (`_sim_summary`):
before the option, and at the stop. `{side}` is `self` or `opponent`.

| token | meaning |
|---|---|
| `pv:sim:next:{self,opponent}:{kind}` | the decision the simulation stopped at |
| `pv:sim:new_turn`, `pv:sim:step:{step}` | the turn / the step changed (the step it is in now) |
| `pv:sim:{side}:life+>=k` / `life->=k` | life change (signed thermometer, life steps) |
| `pv:sim:{side}:creatures_lost>=k`, `power_lost>=k` | creatures that left the battlefield, their summed power (before) |
| `pv:sim:{side}:lost_power_tier:{t}` (+ `#2`..) | one per lost creature, power capped at 5, counted |
| `pv:sim:{side}:creatures_gained>=k`, `power_gained>=k` | creatures that entered |
| `pv:sim:{side}:perms_lost>=k`, `perms_gained>=k` | all permanents (by object id) |
| `pv:sim:{side}:tapped>=k`, `untapped>=k` | permanents that stayed and became tapped / untapped |
| `pv:sim:{side}:damage>=k` | damage newly marked on creatures that stayed |
| `pv:sim:{side}:{hand,graveyard,exile,library}+>=k` / `->=k` | zone size changes (counts only, for the opponent's hand and both libraries the only thing known) |
| `pv:sim:stack+>=k` / `->=k` | stack size change |
| `pv:sim:mana_left>=k` | the decider's untapped mana sources + floating mana at the stop, capped at 8 |
| `pv:sim:color:{C}` | a colour (WUBRG) the decider can still make |
| `pv:sim:{flag}`, `pv:sim:gained:{flag}`, `pv:sim:lost:{flag}` | the lethal flags at the stop, and those that changed: `self:` / `opponent:lethal_on_board`, `evasive_lethal_on_board` (set 2) and `incoming_lethal` (set 4) |

The engine has no poison counters, so there is no poison token yet; it
belongs next to life when a card adds them.

### Hidden information

The summaries read only public objects, zone sizes, the decider's own mana
and the decider-relative lethal flags; a simulation never continues through
a library touch, and never lists the opponent's options. Checked by
`tests/test_sim_previews.py` (both engines):

- `test_hidden_cards_never_change_sim_tokens`: pairs of games built from the
  same public position whose hidden cards differ (the opponent's hand and
  library, the same cards split and ordered differently, and in half of the
  cases the decider's library order) are played with the same choices while
  the decider's observation stays the same; at every decider decision every
  option's previews must be equal. Without `Game.sim_viewer` it fails on
  every matchup (an opponent's single-option priority was auto-passed, so a
  spell resolved or not depending on its hand).
- `test_an_opponent_without_a_response_is_not_revealed` (Counterspell vs a
  land in the opponent's hand), `test_a_draw_stops_the_simulation`.

### Parity and cost

Both engines produce the same strings:
`tests/test_difftest.py::test_entity_and_preview_strings_identical` compares
`option_preview` and `option_previews` for sets 3-6 over all matchups and
checks that every stop reason except `game_over` (scenario test) and the
common deltas occur; `make difftest` compares `featurize()` in set 6.
Copies drop cheaply in Rust: a suspended copy is resumed with an abort index
that makes `ask` return `Stop::Abort` up to `main`, instead of unwinding the
coroutine (`Game::release_co`, ~4× cheaper).

`python tools/bench_sim.py --engine native --games 30` (random play, all
matchups, 4048 decisions, 2.8 options per decision; Apple M3 laptop, shared
with other agents, load ~8): `featurize_flat` per decision

| | set 5 | set 6 | added |
|---|---:|---:|---:|
| Rust, mean | 14.6 µs | 32.4 µs | +17.8 µs (2.2×) |
| Rust, median | 14.3 µs | 27.5 µs | +12.9 µs (1.9×) |
| Python, mean (6 games) | 186 µs | 981 µs | +795 µs (5.3×) |

About 6 µs per option in Rust, two thirds of it the copy (state clone,
coroutine restart and replay of the ~1.7 actions since the step began, which
lists their options again) and its drop. A game featurized in set 6 takes a
snapshot at every step start (`Game.copy`), included in the engine step
time: Rust step mean 5.8 µs. No decision of these games had more than 12
options, so the 32-option cap did not trigger.

### Blind-spot audit

`mtg_ml.audit` (PR #34) groups options by identical featurization and
compares their successors. On scripted bots (50 games per matchup, native)
sets 5 and 6 both have 0 blind spots per 1k decisions on jund_blue (14633
decisions), jund_madness (11413) and blue_madness (8363); with random play
(30 games each) also 0, and set 5 left a single colliding group (truly
equivalent, same sim tokens). Set 5's hand entities and pointers already
separate the options, so the audit cannot measure set 6's gain; what set 6
adds is the consequence of an option, which the audit does not score. The
other direction holds: of 311 option groups whose successors the audit
calls equivalent (random play, 8 games per matchup), 2 get different sim
tokens, both momentary public differences the audit's longer successor
washes out (which land stays untapped until the next untap step; a damage
split before cleanup).
