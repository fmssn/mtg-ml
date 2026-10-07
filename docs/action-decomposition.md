# Action decomposition: auto mana and auto pass

Item 17 of the next actions: shorter trajectories for the RL agent by taking
decisions out of its hands that are not strategic. Two opt-in game options,
both off by default (the default game, its decisions and the golden digests
are unchanged), implemented identically in the Python reference engine
(`mtg_ml/engine/game.py`) and the Rust port (`native/src/engine.rs`,
`native/src/state.rs`):

```python
Game(decks, seed=..., auto_mana=True, auto_pass=True)   # also NativeGame
```

| where | flag |
|---|---|
| `mtg_ml.rl.train` | `--auto-mana 1`, `--auto-pass 1` (TrainConfig ints, like the other switches; passed to rollouts *and* evaluations) |
| `mtg_ml.rl.rollout.Job` | `auto_mana`, `auto_pass` |
| `mtg_ml.rl.evaluate` | `--auto-mana`, `--auto-pass` (evaluate a policy trained with them under the same rules) |
| `mtg_ml.trace.Scenario` | `auto_mana`, `auto_pass` |
| `mtg_ml.difftest fuzz` | `--auto-mana`, `--auto-pass` |
| `tools/decision_stats.py` | measures all four combinations |

## `auto_mana`: colour-preserving payer

Mana costs are paid one unit at a time through `pay_mana` decisions (one per
unit, filtered so every offer keeps the payment completable). With
`auto_mana` each such decision is resolved by a deterministic payer, unless
the choice is strategic:

1. **Floating mana first.** If a pool option is offered, take the first one
   (the pool empties at the end of the step anyway).
2. **Sacrifices stay with the player.** If any option sacrifices its source
   (an Eldrazi Spawn, the only sacrifice-for-mana source in the pool), the
   decision is asked as before, with all options. After the player's pick the
   next unit is checked again (once no Spawn is offered, the rest is
   auto-paid). There are no life-payment mana sources in the card pool.
3. **Otherwise tap the source that hurts least**, by the key (lowest first):
   * *hurt*: sum over the colours the source can produce of the hand's colour
     need, where need = coloured pips of the non-land cards in the payer's
     hand, instants counted twice (they are cast from what stays untapped);
   * 1 if the source has another `{T}` ability (Twisted Landscape keeps its
     land search), else 0;
   * the number of colours the source produces (keep duals);
   * 0 if the unit pays a coloured pip, 1 if it pays generic;
   * option order (the engines' deterministic object order).

The payer is greedy, but it only ever picks among offered options, and every
offered option leaves the rest of the cost payable, so it never gets stuck.
Choices are logged as `  p0 pay_mana: Tap Swamp#12 for B (auto)`.

The scripted bots pay with a similar heuristic, so their win rate does not
move (below).

## `auto_pass`: collapse uneventful priority passes

`auto_single` (default on) already skips every decision with one option, and
plain lands are never tapped at priority (only while paying), so a player
with nothing castable is never asked for priority. The remaining "pass or do
something useless" case is a priority decision whose only non-pass options
are sacrifice-for-mana abilities (Eldrazi Spawn). With `auto_pass` the
player passes without being asked when all of these hold:

* the stack is empty (with a spell or ability on the stack, mana floated
  now can pay for something after it resolves, e.g. a sorcery in the same
  main phase, or the Spawn may not survive the resolution);
* every non-pass option is such a mana ability;
* the Spawn has no triggers of its own and is not attacking, blocking or
  blocked (sacrificing a blocker after blocks is a real play);
* the player controls nothing with a "whenever you sacrifice" trigger
  (Gixian Infiltrator, Writhing Chrysalis grow from it).

Anything that uses the mana is already in the option list (castability
counts every source, Spawns included), and the pool empties at the end of
the step, so in that position passing is never worse. Everything else stays
a decision: instants, activated abilities (Krark-Clan Shaman, Clues,
islandcycling) and land drops are real choices even when a pass is usual.

## Measurements

`python tools/decision_stats.py --games 2000 --engine native` (scripted bot
vs bot, game 1 decks, seeds 0-1999, alternating starting player; Apple M3,
2026-10-07):

| config | decisions/game | per turn | turns/game | Jund win rate |
|---|---|---|---|---|
| default | 263.8 | 15.83 | 16.7 | 33.0% |
| auto_mana | 234.0 | 14.12 | 16.6 | 32.5% |
| auto_pass | 262.0 | 15.72 | 16.7 | 33.0% |
| both | 232.2 | 14.01 | 16.6 | 32.5% |

Per game (200 games, seeds 0-199), by kind: `pay_mana` 34.3 -> 4.8 with
`auto_mana` (the rest are payments where a Spawn could be sacrificed);
`priority` 190.1 -> 188.5 with `auto_pass`. The win rate is unchanged within
noise (about 1 point standard error at 2000 games).

So `auto_mana` removes ~11% of all decisions (~85% of the payment
decisions); `auto_pass` is small, because most Spawn-only priority stops
happen with Writhing Chrysalis (which made the Spawns and grows from their
sacrifice) on the battlefield. The bulk of the remaining ~190 priority
decisions per game are stops where the player holds a real instant-speed
option (Brainstorm, Counterspell, Cast Down, Krark-Clan Shaman, Clue...) and
passes; collapsing those would need a learned or heuristic "nothing worth
doing" judgement, which changes the game rather than its interface.

## Verification

* `tests/test_action_decomposition.py` (both engines): payer choices,
  Spawn payments left to the player, floating mana first, auto_pass positive
  and negative cases, whole games with every combination.
* `tests/test_difftest.py::test_lockstep_action_decomposition` and
  `python -m mtg_ml.difftest fuzz --games 2000 --jobs 8 --auto-mana --auto-pass`.
