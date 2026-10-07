# Elves

The fourth deck: Pauper Elves (mana Elves, Priest of Titania, Timberwatch
Elf pumps, Avenging Hunter's initiative), with its 15-card sideboard, in
both engines. Elves sits in seat 1 of three matchups (`match.MATCHUPS`):
`jund_elves`, `blue_elves`, `madness_elves`.

## Decklist (`engine/decks.py`)

Source: the project owner's Pauper-Research data for Q3 2026 (2026-07-05 to
2026-10-02, all events, 399 Elves lists, deck win rate 53.1%; dashboard
"Builds and decisions"). Main and sideboard are the typical lists of the
most played builds, card for card: main "Stock list" (123 of 399 lists,
31%), sideboard "Stock list" (200 of 399, 50%). Nothing is left out for
missing rules (Land Grant was implemented for it) and nothing is trimmed;
Deglamer is in the typical sideboard itself, not a fill.

| main | | sideboard | |
|---|---|---|---|
| 10 Forest | 4 Masked Vandal | 4 Monstrous Emergence | 3 Vitu-Ghazi Inspector |
| 1 Gingerbread Cabin | 4 Nyxborn Hydra | 4 Spinewoods Paladin | 1 Deglamer |
| 3 Llanowar Elves | 4 Avenging Hunter | 3 Faerie Macabre | |
| 3 Fyndhorn Elves | 4 Generous Ent | | |
| 2 Elvish Mystic | 3 Sagu Wildling | | |
| 4 Priest of Titania | 4 Winding Way | | |
| 4 Quirion Ranger | 4 Lead the Stampede | | |
| 4 Timberwatch Elf | 2 Land Grant | | |

Faerie Macabre ({1}{B}{B}) is never cast: Elves has no black mana. It is a
free graveyard-hate discard (shared card from the sideboard branch).

## Sideboard plans (`engine/sideboard_plans.toml`)

| deck vs opponent | in | out | why |
|---|---|---|---|
| Elves vs Jund | 3 Vitu-Ghazi Inspector, 1 Deglamer | 2 Elvish Mystic, 1 Fyndhorn Elves, 1 Sagu Wildling | Krark-Clan Shaman sweeps x/1 Elves; Inspector (1/3 reach) survives it, Deglamer removes a Bridge or Wellspring for good |
| Jund vs Elves | 2 Breath Weapon, 1 Terminate | 3 Cleansing Wildfire | Breath Weapon sweeps x/1 and x/2 Elves, Terminate kills any; Wildfire on a basic Forest only cycles |
| Elves vs Blue | 3 Faerie Macabre, 1 Monstrous Emergence | 4 Masked Vandal | Macabre shrinks Terror / Serpent and stops Sleep of the Dead's escape; no targets for Vandal |
| Blue vs Elves | 3 Gut Shot, 1 Envelop | 3 Force Spike, 1 Deem Inferior | Gut Shot kills a mana Elf for free, Envelop counters Lead the Stampede, Winding Way or Land Grant |
| Elves vs Red | 4 Spinewoods Paladin, 3 Vitu-Ghazi Inspector | 4 Masked Vandal, 2 Elvish Mystic, 1 Fyndhorn Elves | life and toughness against burn |
| Red vs Elves | 2 End the Festivities, 2 Cast into the Fire, 2 Searing Blaze | 4 Highway Robbery, 2 Grab the Prize | sweepers and two-for-ones for a board of x/1s |

## Rules added (both engines)

| rule | spec | notes |
|---|---|---|
| creature mana | existing `mana` + `tap` | summoning sickness already applied to tap abilities of creatures |
| one mana per Elf | `mana_amount = "elves"` | Priest of Titania: one activation makes N units; the payment spends one, the rest floats in the pool. Feasibility counts N units |
| changeling | keyword `changeling` | `has_creature_type`: Elf counts (Masked Vandal) and the Eldrazi condition of Writhing Chrysalis |
| land-return cost, once each turn | `return_land = "forest"`, `once_per_turn` | Quirion Ranger; the Forest is chosen in a `sacrifice` decision (key `return_land`). Lands are only tapped while paying, so the usual line is: pay with the Forest, return it, replay it |
| targeted triggers | trigger `targets` | shared with the sideboard branch: chosen as the trigger goes on the stack, removed without a legal target (603.3d) |
| omen | `omen = true`, `back` | Sagu Wildling's Roost Seek: cast mode `omen`, on the stack the card is the back face (a green sorcery), shuffled into the library on resolution |
| conditional ETB | `enters_tapped_unless_forests`, condition `entered_untapped` / `cast_mode` | Gingerbread Cabin, Vitu-Ghazi Inspector |
| collect evidence | `collect_evidence = 6` | cast mode `evidence`; exile graveyard cards one at a time until the total mana value is 6+ |
| additional cost: a creature's power | `additional_power` | Monstrous Emergence: choose a creature you control (its current power on resolution if it is still there) or reveal a creature card |
| initiative, Undercity | `[[dungeon]]`, op `take_initiative` | `Game.initiative`, `Player.dungeon_room`; take = venture; the holder ventures in their upkeep; combat damage to the holder takes it (a trigger of the attacker's controller). Rooms are triggers of the Undercity, whose source is an object outside the zones (oid 0) |
| scry N | `scry`, n > 1 | one ORDER decision over (top in order, bottom in order), deduplicated by names (Lost Well: scry 2) |
| hexproof until your next turn | `Card.hexproof` | Throne of the Dead Three; cleared in its controller's untap step; opponents' targets skip it |
| menace | keyword `menace` | Skeleton token. Blocking a menace creature is offered only while a second blocker is possible; a lone block left at the end is undone |
| Treasure | `[[token]]` | a self-sacrificing mana source of any colour. Payment feasibility with an additional sacrifice tries subsets of non-{C} sacrifice sources. It is not offered as a priority mana ability (floating it only matters for sacrifice triggers) |
| reveal-hand alternative cost | `alternative_cost = { reveal_hand = true }` | Land Grant: cast mode `alternative` for free, offered only with no other land card in hand; the hand becomes known to both players |

New ops: `untap_target`, `pump_target`, `counters_target`, `shuffle_target_into_library`, `dig`,
`may_exile_from_graveyard`, `damage_chosen_power`, `take_initiative`, `reveal_to_battlefield`;
`scry` takes n > 1. No `custom` effects. New target kind `artifact_or_enchantment` (Deglamer).
Views and dumps gain `initiative` and `dungeon_room`, only once someone took the initiative, so
existing views (and golden digests) are unchanged. The feature sets do not encode the initiative yet.

## Known simplifications

- **Arena** (goad target creature) is left out of the Undercity: the engine has no attack
  requirements. Forge leads only to Trap!, Lost Well only to Stash.
- A completed Undercity is not tracked: venturing from Throne of the Dead Three starts a new one.
- **Lead the Stampede** takes every creature it finds (never worse; extras can be discarded) and
  puts the rest on the bottom in their old order instead of a chosen order.
- **Menace** is enforced after the blocks are declared (a lone block is undone) rather than by
  a joint block declaration.
- **Faerie Macabre**'s graveyard choice is made on resolution (the engine has no graveyard
  targets; shared card, see the sideboard branch).
- Floating Priest of Titania mana empties at the end of the step like any pool.

## Bot (`bots/elves.py`)

Lands, mana Elves and Priest first, then Avenging Hunter, Generous Ent, Timberwatch, Ranger,
Sagu Wildling (the omen when it needs a land), Winding Way (creature, or land when short) and
Lead the Stampede to refill. Timberwatch pumps after blockers (an unblocked attacker first);
Quirion Ranger untaps a tapped Timberwatch for a second pump, or in main 1 untaps a tapped mana
creature by returning a tapped Forest that is replayed. Masked Vandal eats the opponent's best
artifact when a creature card is in the graveyard; Faerie Macabre is discarded at two or more
graveyard targets (spells for Terror, escape, flashback, Sneaky Snacker). The base bot now
handles Undercity targets (Trap! at the opponent, Forge on its own best creature), scry N, the
Throne choice and optional exiles, for every deck that steals the initiative.

## Bot games (200 games per matchup, native engine, MacBook, 2026-10-07)

Elves bot against each deck's bot, starting player alternating:

| matchup | game 1 | game 2 (postboard) | Q3 2026 (human matches) |
|---|---|---|---|
| Elves vs Jund Wildfire | 76.5% | 76.0% | 55% vs Jund Midrange (56 matches) |
| Elves vs Mono Blue Terror | 73.0% | 79.0% | 56% (48) |
| Elves vs Red Madness | 44.0% | 42.0% | 28% (71) |

The order matches Q3 (Red is by far the worst matchup), but the bot over-performs against Jund
and Blue. Likely reasons: the Jund and Blue bots were tuned for each other and do not answer a
wide board (Jund's Krark-Clan Shaman and Blue's tempo plays), nobody attacks to take the
initiative back, blockers cannot see Timberwatch pumps that come after blocks, and the
initiative alone is strong (taken in about 75% of games). Max permanents per side in these games:
31 (Elves), 49 on the whole battlefield, under the encoder's `MAX_ENTITIES = 64` (permanents and
stack items).
