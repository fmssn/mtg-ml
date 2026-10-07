# Tron

The fourth deck: Pauper Tron, the Q3 2026 green artifact-ramp build. It
assembles Urza's Mine, Power-Plant and Tower (seven mana from three lands),
digs for the missing piece with Expedition Map, Ancient Stirrings and Crop
Rotation, and casts seven- and eight-drops. Matchups put it in seat 1:
`jund_tron`, `blue_tron`, `madness_tron` (`match.MATCHUPS`).

## Decklist (`engine/decks.py`)

| main | | | sideboard |
|---|---|---|---|
| 4 Urza's Mine | 4 Expedition Map | 4 Bramble Wurm | 2 Breath Weapon |
| 4 Urza's Power Plant | 4 Ancient Stirrings | 2 Generous Ent | 3 Relic of Progenitus |
| 4 Urza's Tower | 3 Crop Rotation | 4 Boulderbranch Golem | 2 Call Damage Control |
| 2 Forest | 4 Barrels of Blasting Jelly | 4 Pinnacle Kill-Ship | 2 Scour from Existence |
| 1 Bojuka Bog | 2 Bonder's Ornament | 4 Maelstrom Colossus | 1 Kaervek's Torch |
| 1 Conduit Pylons | 4 Candy Trail | | 1 Pulse of Murasa |
| 1 Haunted Fengraf | 4 Unfathomable Truths | | 2 Monstrous Emergence |
| | | | 2 Blue Elemental Blast |

**Source:** the project owner's Pauper research data for Q3 2026
(2026-07-05 to 2026-10-02, 443 Tron lists; `q3_typical_lists.txt`,
`deck_decisions/data.js`). Main: every card played in at least half of the
lists, at its modal count, sums to 64. The four Giant's Boulder (played in
51%, the least of them, and 0 copies in 49% of lists) are cut, which gives
exactly 60. Sideboard: only 2 Breath Weapon (80%) and 3 Relic of Progenitus
(74%) are in at least half of the sideboards; the rest is filled with the
next most played at their modal counts: Call Damage Control (46%), Scour
from Existence (36%), Kaervek's Torch (36%), Pulse of Murasa (27%),
Monstrous Emergence (26%), and 2 Blue Elemental Blast (23%, standing in for
the equally played Hydroblast: both are blue anti-red cards). Whispersilk
Cloak (28%) is skipped: equipment is not in the engine.

## Sideboard plans (`engine/sideboard_plans.toml`)

| deck vs opponent | in | out | why |
|---|---|---|---|
| Tron vs Jund | 2 Breath Weapon, 2 Call Damage Control | 2 Candy Trail, 2 Unfathomable Truths | Breath Weapon sweeps Spawn, Krark-Clan Shaman and the Familiars; Call Damage Control rebuys lands lost to Cleansing Wildfire |
| Jund vs Tron | 2 Duress | 1 Makeshift Munitions, 1 Lembas | Duress takes Stirrings, Map, Crop Rotation, Truths |
| Tron vs Blue | 2 Breath Weapon, 2 Scour from Existence | 2 Candy Trail, 2 Boulderbranch Golem | instant-speed answer to Delver, Scour for Terror and Serpent |
| Blue vs Tron | 2 Dispel | 2 Sleep of the Dead | Dispel counters Crop Rotation and Truths |
| Tron vs Red | 2 Blue Elemental Blast, 2 Breath Weapon, 1 Pulse of Murasa | 2 Unfathomable Truths, 2 Maelstrom Colossus, 1 Bonder's Ornament | counter burn, sweep small creatures, gain life; the slowest cards out |
| Red vs Tron | 2 Gorilla Shaman | 2 Voldaren Epicure | Shaman eats Maps, Barrels, Candy Trails, Ornaments |

Rows for Affinity and Elves, to add to the table when those decks land (the
loader requires both decks to exist):

```toml
[[plan]]
deck = "tron"
opponent = "grixis_affinity"
why = "Breath Weapon kills Refurbished Familiar, Krark-Clan Shaman and Kenku Artificer, Kaervek's Torch a Myr Enforcer; Candy Trail is too slow."
in = { "Breath Weapon" = 2, "Kaervek's Torch" = 1 }
out = { "Candy Trail" = 2, "Haunted Fengraf" = 1 }

[[plan]]
deck = "tron"
opponent = "elves"
why = "Breath Weapon wipes the mana elves, Priest of Titania and Timberwatch Elf; Monstrous Emergence kills Avenging Hunter or a Hydra; the slow card draw goes out."
in = { "Breath Weapon" = 2, "Monstrous Emergence" = 2 }
out = { "Unfathomable Truths" = 2, "Candy Trail" = 2 }
```

## Rules added (both engines)

| rule | where | notes |
|---|---|---|
| multi-unit mana | ability `mana_amount = { n, if_control }` | Urza's lands make 2 (Tower 3) while their controller controls the other two subtypes; the extra units float in the pool (spent by later payments of the same step) |
| mana filters | a mana ability with a `cost` ({1} only), `once_per_turn` | Barrels of Blasting Jelly ({1}: any colour, once each turn) and Conduit Pylons ({1}, {T}): offered while paying as "Activate X for U". Feasibility is exact: a filter turns one coloured symbol into generic (`can_pay(..., wild)`: a maximum matching may leave that many coloured symbols unmatched, {C} symbols first); a filter that taps a land also loses that land's own mana |
| prototype | card `prototype` + `prototype_face` (a `[[face]]`), cast mode `prototype` | Boulderbranch Golem as a green 3/3 for {3}{G}; the face is named "Boulderbranch Golem (prototype)" so views and option keys tell the two apart; it is a normal card again in any other zone |
| station | card `station = { n, keywords }`, ability `tap_other = "creature"`, op `station` | tap another untapped creature (summoning sickness does not matter) at sorcery speed: charge counters equal to its power; at 7+ Pinnacle Kill-Ship is a 7/7 flying artifact creature (`Game.is_stationed`) |
| cascade | op `cascade` on a `cast` trigger | exile until a nonland card with lower mana value, may cast it free (any timing, modal cards per mode, additional costs still paid), the rest on the bottom in a random order (game RNG) |
| triggers with targets | trigger `targets`, `optional_targets` | shared with the sideboard branch; `optional_targets` adds "No target" first (Kill-Ship's "up to one target creature"); ward triggers on them |
| scry N > 1 | op `scry`, `n = 2` | one ORDER decision over every top/bottom split in every order (Candy Trail: six options) |
| surveil 1 | op `surveil` | Conduit Pylons |
| graveyard abilities | ability `zone = "graveyard"` | Bramble Wurm: {2}{G}, exile it from your graveyard: gain 5 |
| land sacrifice cost | `additional_sac = "land"` | Crop Rotation (a land tapped for its {G} can be the one sacrificed) |
| choose a creature as a cost | `additional_choose_creature` | Monstrous Emergence: a creature you control or a creature card revealed from hand; its power (last known if it left) is the damage |

New ops: `dig` (Ancient Stirrings), `cascade`, `station`, `surveil`,
`damage_target_from` (`from = "x"`: Kaervek's Torch; `"chosen_power"`:
Monstrous Emergence), `return_random_from_graveyard` (Haunted Fengraf),
`return_from_graveyard` (Call Damage Control, Pulse of Murasa). New
parameters: `draw.each_controlling` (Bonder's Ornament), `gain_life.n_from =
"source_power"` (Golem), search filter `colorless`, sacrifice filter
`land`. `exile_target` (Scour from Existence) is the sideboard branch's.
Generous Ent and its Food token come from the Elves branch.

## Known simplifications

- **Ancient Stirrings** puts the rest on the bottom in their current order
  instead of an order of the player's choice (nothing in the pool reads the
  bottom of a library).
- **Call Damage Control** and **Pulse of Murasa** choose their cards as they
  resolve: the engine has no graveyard targets, so a graveyard-hate response
  cannot fizzle them.
- **Kaervek's Torch**: the {2} tax on spells that target it on the stack is
  not modelled (only Counterspell / Force Spike / Dispel / blasts could
  target it).
- **Cascade into Boulderbranch Golem** casts the full 6/5 (casting it
  prototyped for free is legal but never better).
- **Boulderbranch Golem's ETB** reads the Golem's current power, or its card's
  if it has left the battlefield (no last-known 3 for a prototyped Golem
  killed in response).
- **Charge counters** are not in `observe()` or the features yet; a
  stationed Kill-Ship shows up as a creature.
- Feature previews (`pv:mana_left_after`) count mana sources, not units, so
  they undercount assembled Tron (feature-set territory).

## Bot (`bots/tron.py`)

A missing Tron piece first (land drop, Map, Stirrings, Crop Rotation when it
completes Tron), cheap artifacts early, then the biggest affordable threat
(Colossus, Kill-Ship on their best creature, Wurm, Golem; prototyped Golem
while seven mana is far away). Card draw (Truths, Candy Trail, Ornament)
waits for the opponent's end step; summoning-sick creatures station the
Kill-Ship in main 2; Ent forestcycles while a Forest can still be found;
the board attacks all-out when the unblockable part is lethal. Sideboard
cards: Breath Weapon (two or more of theirs die), Scour, Blue Elemental
Blast, Torch, Emergence, Call Damage Control, Pulse.

## Bot-vs-bot (200 game-1 games per matchup, native engine, Apple M3 laptop, shared with other jobs, 2026-10-07)

| matchup | Tron wins | Q3 2026 Tron match win rate | turns mean / p50 / p90 / max | decisions mean / p90 / max |
|---|---|---|---|---|
| Jund Wildfire vs Tron | 72.5% | 47% vs Jund Midrange (38 matches) | 16.6 / 16 / 22 / 36 | 318 / 542 / 1065 |
| Mono Blue Terror vs Tron | 62.5% | 38% (55 matches) | 14.5 / 14 / 18 / 31 | 189 / 309 / 598 |
| Red Madness vs Tron | 49.0% | 46% (90 matches) | 11.5 / 11 / 15 / 28 | 173 / 277 / 380 |
| for reference: Jund vs Blue | | | 15.8 / 15 / 21 / 27 | 243 / 395 / 593 |
| Jund vs Red | | | 13.5 / 13 / 20 / 25 | 245 / 407 / 579 |
| Blue vs Red | | | 11.6 / 11 / 15 / 20 | 164 / 228 / 340 |

Game 2 (both sides sideboarded, 200 games each): Tron 72% vs Jund, 67.5% vs
Blue, 53.5% vs Red.

Red is close to the field data. Tron beats the Jund and Blue bots far more
often than real Tron beats those decks because those bots were written for
each other: the Jund bot spends Cast Down on Kill-Ships and Golems but has no
plan against seven-drops and never pressures early, and the Blue bot taps
out for its own threats instead of holding Counterspell for Tron's
threats or Crop Rotation (real Blue wins this matchup by countering the
payoff). "Jund Wildfire" is also not "Jund Midrange" of the field data.

**Game length.** Tron games are the longest (Jund vs Tron: p90 22 turns,
max 36; two of the 200 sideboarded games ended with Jund decked), and a turn with assembled Tron costs
more decisions (each floating {C} is paid by its own decision unless
`auto_mana` is on). Every cap still fits: `max_turns` is 100 everywhere
(engine, rollouts, evaluation), and `SearchBot.max_decisions` is 4000
against a maximum of about 1100 decisions per game.
