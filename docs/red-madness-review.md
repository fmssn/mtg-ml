# Red Madness after 1M games: review of greedy games (2026-10-07)

Run `20261007-red-madness-1m` (ledger): Red Madness, warm-started from the
Jund network `20261007-r1-control`, trained 1M games against that same
network frozen, on the `jund_madness` matchup. This note reviews how it
plays, for two questions: are there engine mistakes, and which misplays
point at the learning setup.

## Method

- 20 greedy games, both networks taking their most likely option (seeds 0-13
  game 1, 14-19 game 2, each seat starting half of them), recorded with
  `python -m mtg_ml.replay record --matchup jund_madness --greedy` and read
  as transcripts (`tools/replay_text.py`). Each game was read decision by
  decision by one of four reviewers. Every life change was reconciled with
  the engine log, and every Guttersnipe / Kessig Flamebreather trigger, madness
  cast, Sneaky Snacker return and land sacrifice was checked against the card
  text. Engine findings were then re-checked against the replay JSON.
- 200 more greedy games (150 game 1, 50 game 2) for counts of the patterns
  the reviews found.

Red won 16 of the 20 games, and 169 of the 200 (84% of game 1s, 86% of game 2s).

## 1. Engine mistakes

**None found.** The checks covered:

- **Damage and life:** all life totals.
- **Cast triggers:** on normal, flashback, madness, plot and alternative-cost casts, and never on creature spells.
- **Madness:** the cast offered only with {R} available, otherwise straight to the graveyard; discards forced by the opponent (Duress, Refurbished Familiar) and cleanup discards.
- **Red cards:**
  - Grab the Prize's land condition.
  - Highway Robbery's choices.
  - Fireblast and Lava Dart sacrificing tapped or untapped Mountains.
  - Sneaky Snacker on exactly the third draw, and not when discarded after it.
  - Gorilla Shaman offering only affordable targets.
  - Electrickery overload.
  - Searing Blaze landfall.
- **Jund cards and rules:**
  - Krark-Clan Shaman sparing fliers.
  - Writhing Chrysalis counters with several Chrysalises.
  - Familiar affinity and its empty-hand draw.
  - Cleansing Wildfire on indestructible Bridges.
  - Lembas.
  - Nihil Spellbomb.
  - Blocks staying blocked after the blocker is sacrificed.
  - Cleanup discard.

Two things look odd in transcripts but are correct:

- Eldrazi Spawn mana floats across priority passes inside a step and empties at the end of the step, as the rules say. Lands are only tapped while paying, an engine convention.
- Indestructible Bridges are legal Gorilla Shaman targets.

## 2. What the high win rate is made of

The frozen Jund network never saw a Red card in training, and it shows:

| | |
|---|---|
| Frozen Jund vs the scripted Red bot | 41% (greedy, 1,000 games) |
| Scripted Jund bot vs the scripted Red bot | 21% |
| Red learner vs the frozen Jund (game 1) | 88.5% sampled, 84% greedy |

So Red learned far more than the scripted bot (88% against the same
opponent where the bot gets 59%), but against an opponent that cannot rank
its threats.

The frozen Jund's typical errors, all with probabilities that look like
"unknown card":

- It aims Cast Down at Voldaren Epicure or Sneaky Snacker instead of Guttersnipe at 2-3 life. This decided two games.
- It blocks Kessig Flamebreather instead of Guttersnipe.
- It holds Cast Down all game against two Flamebreathers.
- It never casts Krark-Clan Shaman against a board of 2/2s and 1/3s.
- It trades away Lembas life at burn range.
- Its value head stays positive at 7 life the turn before dying.

## 3. Red's own leaks, and the setup cause behind each

| leak | evidence | likely cause |
|---|---|---|
| Fiery Temper discarded with no {R} open | a third of all Temper discards lose the madness cast (51 of 152 in 200 games); decided a 20-game loss | discard options do not say whether madness can be paid (option previews cover casts and targets, not discards); the habit "discard Temper" was learned where it pays and generalised |
| Plot never used | Highway Robbery's plot chosen 0 times in 366 offers | exploration: the warm-started policy starts at entropy 0.19, and plot is a new key shape it never sampled; nothing forces it |
| Greedy mulligans | keeps 71% of 1-land 7-card hands (40 of 56); also mulligans good 3-land hands | one sparse decision far from the reward, saturated at p≈1; a slow frozen opponent rarely punishes a stumble |
| Burn almost never aimed at creatures | Bolt 184 times face vs 21 at creatures; 4/5 Writhing Chrysalis left alive to deal all 20 in a loss; Searing Blaze and Electrickery cast 10 of 174 and 9 of 88 offers in game 2 | the training opponent rarely deploys a big threat, so racing is almost always right against it; sideboard cards only appear in half the games and start near p=0 |
| Sloppy once winning | missed lethal twice in one game (Fireblast sacrificed the untapped Mountains, then no attack with Jund at 1); skipped land drops; a flashback Lava Dart at its own face (p=0.64) | value saturated at V≈+1: per-decision discount 0.995 gives almost no reward for winning a turn sooner, and the life shaping is annealed away |
| Holding burn on its own turn | taps out every turn, rarely holds Bolt or madness mana for the opponent's turn | warm-start habits from the Jund network plus no pressure from the opponent |

Things it did learn well:

- The Blood-into-madness Temper line (p=1.00).
- Grab the Prize with a Snacker already in the graveyard.
- Lethal bursts through Guttersnipe and Flamebreather counts.
- Sacrificing tapped rather than untapped Mountains to Fireblast (16 of 20 choices).

## 4. What to change, in order

1. **Train both sides of the matchup.** Self-play on `jund_madness` with the learner on both decks and a snapshot pool, as for Jund vs Blue: Jund learns Red's threats and Red faces an opponent that punishes racing. A frozen opponent that has never seen the deck is the main thing inflating today's numbers. Measure against `20261007-red-madness-1m` and the bots.
2. **A madness preview on discard options.** Add `pv:madness_payable` to every discard option (cost and resolution-time discards, Blood, Looting, Robbery, Grab, cleanup): an engine-computed check that the owner can pay {R} after this choice. This is cheap, fits `encode.option_preview`, and targets the biggest learnable leak.
3. **Exploration for a new deck.** Either start the new deck from the warm start with a higher `--ppo-ent-coef` for the first ~200k games, or compare against a fresh network. The warm start brought low entropy (0.19), so new option shapes like plot are never tried.
4. **Win sooner when ahead.** A small reward for the number of turns saved, or a lethal-availability feature (`lethal_on_board` exists for creatures, not for burn in hand), to stop the saturated-value sloppiness. Per-turn discounting (`r1-gammaturn`) was rejected for Jund, so prefer the feature.
5. **Mulligans.** Watch keep rates per land count as a metric. If self-play does not fix them, consider an entropy floor on the mulligan decision kind.

Use the stronger `20261007-r1-lranneal` Jund (Elo 135 vs 99) as the starting point for the next round.
