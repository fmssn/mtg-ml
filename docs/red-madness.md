# Red Madness

The third deck: Pauper Red Madness (a burn deck that discards Fiery Temper and
Sneaky Snacker for value), with its full 15-card sideboard, in both engines.
Matchups pick the deck per seat (`match.MATCHUPS`): `jund_madness` puts Jund
Wildfire in seat 0 and Red Madness in seat 1.

## Decklist (`engine/decks.py`)

| main | |
|---|---|
| 3 Guttersnipe | 4 Lava Dart |
| 4 Kessig Flamebreather | 3 Fireblast |
| 4 Voldaren Epicure | 4 Fiery Temper |
| 4 Sneaky Snacker | 4 Faithless Looting |
| 4 Lightning Bolt | 4 Highway Robbery |
| 18 Mountain | 4 Grab the Prize |

Given by the project owner on 2026-10-07. Sneaky Snacker ({U}{B}) is never
cast here: it is discarded and returns from the graveyard on the third draw
of a turn.

## Sideboard and plans (games 2 and 3)

Since 2026-10-07 the sideboard is the typical Q3 2026 15 (4 Pyroblast, 3 Relic
of Progenitus, 2 Red Elemental Blast, 2 Cast into the Fire, 2 Searing Blaze, 2
End the Festivities) and the plans live in `engine/sideboard_plans.toml`
([sideboarding.md](sideboarding.md)). The first sideboard (Gorilla Shaman,
Martyr of Ashes, Electrickery) stays in the card pool.

## Rules added

| rule | where | notes |
|---|---|---|
| madness | `Game.discard`, `MADNESS_TRIGGER` | every discard (costs, effects, cleanup, the opponent's Duress / Familiar) of a madness card exiles it; the owner's trigger casts it for {R} at any speed or puts it into the graveyard |
| discard as a cost | `additional_discard` (Grab the Prize), `discard_other` (Blood) | Grab the Prize remembers whether a land was discarded |
| land-sacrifice costs | `alternative_cost`, `flashback_cost` | Fireblast (two Mountains instead of its mana cost), Lava Dart flashback (a Mountain); tapped Mountains can be sacrificed, so tapping them for another spell first loses nothing |
| plot | `Plot X` special action, cast mode `plot` | Highway Robbery: pay {1}{R} at sorcery speed, exile it face up (shown as `X (plotted)` in views), cast it free as a sorcery on a later turn |
| overload | cast mode `overload`, `overload_effect` | Electrickery hits each creature you don't control |
| cast triggers from the battlefield | event `you_cast`, condition `{ spell = ... }` | Kessig Flamebreather (noncreature), Guttersnipe (instant or sorcery) |
| third draw | event `third_draw` | Sneaky Snacker returns tapped from the graveyard |
| landfall | `Player.landfall_turn` | Searing Blaze deals 3 instead of 1 |
| dependent target | `creature_of_target_player` | Searing Blaze's creature belongs to its target player; the player target only offers players who control a creature, so no dead ends |
| X = target mana value | `x_target_mv` | Gorilla Shaman pays {X}{X}{1} for the chosen artifact (choosing another X can only fizzle, so it is not offered); only affordable targets are offered |
| reveal X red cards | `x_reveal` | Martyr of Ashes: choose X, then reveal that many red cards one by one |
| exile as a cost | `exile_self` | Relic of Progenitus |
| color-checked effects | `if_color` | Pyroblast targets any spell or permanent and only hits blue ones |

`self:deck:<name>` is a state feature for a player whose deck is not its
seat's usual one (`match.deck_names`), so networks trained on Jund vs Blue
see exactly their old inputs when they play Jund here.

## Bot

`bots/red.py`: lands, Guttersnipe / Flamebreather first, burn at creatures
worth killing or at the face once the opponent is low or the hand can finish
it, the rest in the opponent's end step; card flow feeds Fiery Temper's
madness and Sneaky Snacker; Fireblast and the Lava Dart flashback only when
they finish the game or lands are spare. It beats the Jund bot in 79% of
game-1 games (200 games, Python engine) and the random agent in all of 100.

## Training a new deck against a fixed opponent

    python -m mtg_ml.rl.train --run runs/red --matchup jund_madness --exploit jund.pt --exploit-deck red

Exploiter mode (`rl/train.py`) with a matchup: the learner only plays Red
Madness, every training game is against the frozen Jund checkpoint, and it
starts from that checkpoint's weights (`--init` to start elsewhere).
Evaluation: `eval/opponent/red` (the frozen checkpoint, in place of pool0),
`eval/random/red`, `bench/red_vs_bot*` (learner Red vs the Jund bot);
`evaluate.py --matchup jund_madness --seat 1` for one-off evaluations.
