# Tron

The fourth deck: Pauper Tron, the Q3 2026 green artifact-ramp build. It
assembles Urza's Mine, Power-Plant and Tower (seven mana from three lands),
digs for the missing piece with Expedition Map, Ancient Stirrings and Crop
Rotation, and casts seven- and eight-drops. Matchups put it in seat 1:
`jund_tron`, `blue_tron`, `madness_tron` (`match.MATCHUPS`).

## Decklist (`engine/decks.py`)

| main | | | sideboard |
|---|---|---|---|
| 4 Urza's Mine | 4 Expedition Map | 4 Bramble Wurm | 3 Breath Weapon |
| 4 Urza's Power Plant | 4 Ancient Stirrings | 2 Generous Ent | 3 Relic of Progenitus |
| 4 Urza's Tower | 3 Crop Rotation | 3 Boulderbranch Golem | 2 Call Damage Control |
| 2 Forest | 4 Barrels of Blasting Jelly | 4 Pinnacle Kill-Ship | 2 Monstrous Emergence |
| 1 Bojuka Bog | 4 Giant's Boulder | 4 Maelstrom Colossus | 2 Scour from Existence |
| 1 Conduit Pylons | 2 Bonder's Ornament | | 1 Kaervek's Torch |
| 1 Haunted Fengraf | 3 Candy Trail | | 1 Pulse of Murasa |
| | 2 Unfathomable Truths | | 1 Whispersilk Cloak |

**Source:** the project owner's Pauper research dashboard
(`pauper-research/reports/dashboard/index.html`, generated 2026-10-04),
period 90 days (2026-07-05 to 2026-10-02 = Q3 2026), scope online + paper,
"Builds and decisions" for Tron: the typical list of all 443 Tron lists (the
dashboard's aggregate decklist, Frank Karsten's method: every (card, k-th
copy) ranked by how many lists play it, the top 60 / 15). The 60 and the 15
are taken unchanged; every card in them is implemented.

**Why the archetype's typical list and not the largest build's.** The
dashboard splits the 443 lists into builds. The largest (179 lists, 40%:
no Nyxborn Hydra, 3+ Candy Trail, 2+ Generous Ent) plays no Giant's Boulder
and instead 2 Breath Weapon and 2 Prophetic Prism main; the next ones (83, 40,
37, 37, 35 lists) all play 4 Boulder. Across all lists Boulder is in 51% and
almost always as a full playset (4 in 49%, 0 in 49%). The project owner rules
that Giant's Boulder is important, and it is what this kind of Tron does with
its mana: an early scry 2 and colour fixer that later destroys any permanent
for {7}. The aggregate list keeps it, so it is the list used. The largest
build's 60 (if wanted instead): 4 Mine, 4 Power Plant, 4 Tower, 3 Forest, 1
Bog, 1 Pylons, 1 Fengraf, 4 Map, 4 Stirrings, 3 Crop Rotation, 2 Barrels, 2
Bonder's Ornament, 2 Prophetic Prism, 4 Candy Trail, 1 Truths, 2 Breath
Weapon, 4 Wurm, 3 Ent, 3 Golem, 4 Kill-Ship, 4 Colossus (Prophetic Prism is
not implemented).

The previous version of this deck (PR head 3f57415) cut the four Boulders to
get from 64 to 60 and left Whispersilk Cloak out because the engine had no
equipment rules; both are back, and equipment is in both engines.

## Sideboard plans (`engine/sideboard_plans.toml`)

| deck vs opponent | in | out | why |
|---|---|---|---|
| Tron vs Jund | 3 Breath Weapon, 2 Call Damage Control, 1 Whispersilk Cloak | 2 Candy Trail, 2 Unfathomable Truths, 1 Bonder's Ornament, 1 Boulderbranch Golem | Breath Weapon sweeps Spawn, Krark-Clan Shaman and the Familiars; Call Damage Control rebuys lands lost to Cleansing Wildfire; the Cloak keeps a seven-drop safe from Cast Down |
| Jund vs Tron | 2 Duress | 1 Makeshift Munitions, 1 Lembas | Duress takes Stirrings, Map, Crop Rotation, Truths |
| Tron vs Blue | 2 Breath Weapon, 2 Scour from Existence, 1 Whispersilk Cloak | 2 Candy Trail, 2 Boulderbranch Golem, 1 Bramble Wurm | instant-speed answer to Delver, Scour for Terror and Serpent, the Cloak against Deem Inferior; life gain matters little |
| Blue vs Tron | 2 Dispel | 2 Sleep of the Dead | Dispel counters Crop Rotation and Truths |
| Tron vs Red | 3 Breath Weapon, 1 Kaervek's Torch, 1 Pulse of Murasa | 2 Unfathomable Truths, 2 Maelstrom Colossus, 1 Bonder's Ornament | sweep the small creatures, kill one at instant speed, gain life; the slowest cards out |
| Red vs Tron | 2 Gorilla Shaman | 2 Voldaren Epicure | Shaman eats Maps, Barrels, Boulders, Candy Trails, Ornaments |

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
| Equipment (301.5, 702.6) | `subtypes = "Equipment"`, card `equipped = { keywords, power, toughness }`, an `equip` ability (`sorcery_speed`, `targets = ["creature_you_control"]`) with op `attach` | Whispersilk Cloak. `Card.attached_to` holds the equipped creature (bestow Auras use the same field; `Game.is_bestowed` now excludes Equipment, `Game.equipment_on` lists what a creature wears). The grants are static: `keywords` / `power` / `toughness` read them while attached. Equipping again moves it; when the creature leaves or stops being a creature, the SBA that frees bestowed Auras unattaches the Equipment, which stays on the battlefield |
| shroud | keyword `shroud` | `Game.targetable`: never a target candidate and an illegal target on resolution, for either player (so an equip ability cannot target a creature that already wears the Cloak) |
| can't be blocked | keyword `unblockable` | `Game._can_block` (and the bots' `_blockers_for`, the `ready_evasive_power` feature in both engines) |
| filter that taps an artifact | `{1}, {T}` mana ability on Giant's Boulder | the existing filter rules; a tapped Boulder can neither filter nor pay its {7} ability's {T} |

New ops: `dig` (Ancient Stirrings), `cascade`, `station`, `surveil`,
`damage_target_from` (`from = "x"`: Kaervek's Torch; `"chosen_power"`:
Monstrous Emergence), `return_random_from_graveyard` (Haunted Fengraf),
`return_from_graveyard` (Call Damage Control, Pulse of Murasa). New
parameters: `draw.each_controlling` (Bonder's Ornament), `gain_life.n_from =
"source_power"` (Golem), search filter `colorless`, sacrifice filter
`land`. `exile_target` (Scour from Existence) is the sideboard branch's.
Generous Ent and its Food token come from the Elves branch. `attach`
(equip) is new with Whispersilk Cloak; Giant's Boulder needs no new op
(`scry` 2, a filter, `destroy_target` on `permanent`).

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
- **Which creature wears the Cloak** is not an entity feature yet: the Cloak
  has `e:attached`, the equipped creature `e:kw:shroud` and
  `e:kw:unblockable`, but the link between the two is not encoded (for the
  feature set work, like bestow's `attached_to`).
- Feature previews (`pv:mana_left_after`) count mana sources, not units, so
  they undercount assembled Tron (feature-set territory).

## Bot (`bots/tron.py`)

A missing Tron piece first (land drop, Map, Stirrings, Crop Rotation when it
completes Tron), cheap artifacts early, then the biggest affordable threat
(Colossus, Kill-Ship on their best creature, Wurm, Golem; prototyped Golem
while seven mana is far away). Card draw (Truths, Candy Trail, Ornament)
waits for the opponent's end step; summoning-sick creatures station the
Kill-Ship in main 2; Ent forestcycles while a Forest can still be found;
the board attacks all-out when the unblockable part is lethal (creatures
wearing the Cloak always count as unblocked). Giant's Boulder is a cheap
artifact on turn one or two (scry 2, a colour filter); its {7} ability
destroys their best creature at once when it is a big threat (creature
value 6+, e.g. Cryptic Serpent, Tolarian Terror), else a smaller one in
their end step. Whispersilk Cloak is cast once there is a creature to wear
it, equipped in main 1 to the creature that gains most (power, and +3 if it
can attack this turn) and moved only to one that is clearly better.
Sideboard cards: Breath Weapon (two or more of theirs die), Scour, Torch,
Emergence, Call Damage Control, Pulse, the Cloak.

## Bot-vs-bot (200 games per matchup and game, native engine, Apple M3 laptop, shared with other jobs, 2026-10-07)

Seeds 0-199, starting player alternating, Tron in seat 1. "Before" is the
previous list (no Boulder, no Cloak, Blue Elemental Blast in the 15) on the
same seeds and code; the engine and bot changes since only touch Boulder and
Cloak, so it replays the old games. One standard error at 200 games is about
3.5 points.

| matchup | Tron wins game 1 (before → now) | game 2, both sideboarded | Q3 2026 Tron match win rate | turns mean / p50 / p90 / max (game 1) | decisions mean / p90 / max (game 1) |
|---|---|---|---|---|---|
| Jund Wildfire vs Tron | 74.0% → 79.0% | 73.5% → 79.0% | 47% vs Jund Midrange (38 matches) | 16.4 / 16 / 22 / 37 | 323 / 560 / 1186 |
| Mono Blue Terror vs Tron | 68.0% → 71.5% | 66.5% → 66.5% | 38% (55 matches) | 14.7 / 14 / 20 / 25 | 197 / 345 / 511 |
| Red Madness vs Tron | 51.5% → 46.0% | 57.0% → 53.0% | 46% (90 matches) | 11.2 / 11 / 15 / 18 | 167 / 261 / 411 |
| for reference: Jund vs Blue | | | | 15.8 / 15 / 21 / 27 | 243 / 395 / 593 |
| Jund vs Red | | | | 13.5 / 13 / 20 / 25 | 245 / 407 / 579 |
| Blue vs Red | | | | 11.6 / 11 / 15 / 20 | 164 / 228 / 340 |

How often Tron's new cards were used (200 games each): Giant's Boulder was
cast 181 / 184 / 138 times in game 1 vs Jund / Blue / Red and its {7}
ability destroyed something 78 / 63 / 0 times (Red's creatures are below
the bot's threshold; it is a mana filter there). Whispersilk Cloak, a
one-of brought in for game 2 against Jund and Blue, was cast 26 / 21 times
and equipped 28 / 18 times.

Against Jund and Blue the Boulder is removal for their biggest creatures, and the win rate goes
up. Against Red it replaces faster cards (Golem, Candy Trail, Truths) with a
{1} artifact that does little against burn, and Tron loses about 5 points,
within two standard errors.

Red is close to the field data. Tron beats the Jund and Blue bots far more
often than real Tron beats those decks because those bots were written for
each other: the Jund bot spends Cast Down on Kill-Ships and Golems but has no
plan against seven-drops and never pressures early, and the Blue bot taps
out for its own threats instead of holding Counterspell for Tron's
threats or Crop Rotation (real Blue wins this matchup by countering the
payoff). "Jund Wildfire" is also not "Jund Midrange" of the field data.

**Game length.** Tron games are the longest (Jund vs Tron: p90 22 turns,
max 37 in game 1, 39 in game 2), and a turn with assembled Tron costs more
decisions (each floating {C} is paid by its own decision unless `auto_mana`
is on). Every cap still fits: `max_turns` is 100 everywhere (engine,
rollouts, evaluation), and `SearchBot.max_decisions` is 4000 against a
maximum of about 1200 decisions per game.
