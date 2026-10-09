# Grixis Affinity

One of the six decks: Pauper Grixis Affinity (artifact lands and cheap artifacts
that make Myr Enforcer, Utrom Monitor, Refurbished Familiar and Thoughtcast
free), with its full 15-card sideboard, in both engines. Affinity has a matchup
against each other deck (`match.MATCHUPS`): seat 1 in `jund_affinity`,
`blue_affinity` and `madness_affinity`, seat 0 in `affinity_elves` and
`affinity_tron`, and `affinity_mirror` (only where named).

## Decklist (`engine/decks.py`)

The typical lists of the most played builds in the Q3 2026 Pauper-Research
data (2026-07-05 to 2026-10-02, all events, "Builds and decisions"): main
"Stock list", 228 of 524 lists (44%); sideboard "Stock list", 174 of 524
(33%). Card for card: nothing left out for missing rules (Black Mage's Rod
and Envelop were implemented for them) and nothing trimmed. Cards shared
with Jund Wildfire (Refurbished Familiar, Krark-Clan Shaman, Ichor
Wellspring, Nihil Spellbomb, Toxin Analysis, Makeshift Munitions,
Drossforge Bridge, Vault of Whispers) reuse their existing specs.

| main | | sideboard | |
|---|---|---|---|
| 4 Myr Enforcer | 4 Ichor Wellspring | 4 Hydroblast | 1 Blue Elemental Blast |
| 4 Refurbished Familiar | 3 Nihil Spellbomb | 4 Pyroblast | 1 Krark-Clan Shaman |
| 3 Utrom Monitor | 2 Blood Fountain | 2 Envelop | 1 Nihil Spellbomb |
| 3 Krark-Clan Shaman | 2 Black Mage's Rod | 2 Extract a Confession | |
| 4 Thoughtcast | 1 Sewer-veillance Cam | | |
| 4 Galvanic Blast | 1 Makeshift Munitions | | |
| 4 Reckoner's Bargain | 4 Mistvault Bridge, 4 Drossforge Bridge | | |
| 2 Toxin Analysis | 4 Vault of Whispers, 3 Seat of the Synod | | |
| | 2 Silverbluff Bridge, 2 Great Furnace | | |

Until 2026-10-07 the list was the modal count of the most played cards
(2 Kenku Artificer and 1 Mountain instead of 2 Black Mage's Rod and the
third Seat; Red Elemental Blast, Unexpected Fangs and a second Blue
Elemental Blast in the 15 instead of Envelop and Nihil Spellbomb). Kenku
Artificer (43% of lists, the second build's card) and Unexpected Fangs stay
in the card pool.

## Sideboard plans (games 2 and 3)

Rows in `engine/sideboard_plans.toml` (format: [sideboarding.md](sideboarding.md)):

| deck vs opponent | in | out |
|---|---|---|
| Affinity vs Jund | 2 Hydroblast, 2 Extract a Confession, 1 Krark-Clan Shaman | 2 Black Mage's Rod, 1 Sewer-veillance Cam, 1 Makeshift Munitions, 1 Toxin Analysis |
| Jund vs Affinity | 2 Troublemaker Ouphe, 2 Breath Weapon, 2 Red Elemental Blast, 1 Terminate | 2 Lembas, 2 Gixian Infiltrator, 2 Eviscerator's Insight, 1 Nyxborn Hydra |
| Affinity vs Blue | 4 Pyroblast | 2 Toxin Analysis, 1 Makeshift Munitions, 1 Sewer-veillance Cam |
| Blue vs Affinity | 4 Annul, 1 Envelop, 3 Gut Shot | 3 Force Spike, 2 Sleep of the Dead, 1 Plunder the Trollshaws, 2 Ponder |
| Affinity vs Red | 4 Hydroblast, 1 Blue Elemental Blast, 1 Krark-Clan Shaman | 2 Black Mage's Rod, 3 Nihil Spellbomb, 1 Sewer-veillance Cam |
| Red vs Affinity | 2 End the Festivities, 2 Cast into the Fire, 2 Pyroblast | 2 Highway Robbery, 2 Grab the Prize, 2 Sneaky Snacker |

Rows for Elves and Tron (both directions) are in those decks' sections of
the file.

## Rules added

| rule | where | notes |
|---|---|---|
| affinity for artifacts | `cost_reduction = "artifacts_you_control"` | existing reduction, now on Myr Enforcer, Utrom Monitor, Thoughtcast |
| metalcraft | `damage_target.n_metalcraft` | Galvanic Blast: 4 instead of 2 with three or more artifacts |
| sacrificed permanent's mana value | `gain_life.sacrificed_mv`, spell data `sacrificed_mv` | Reckoner's Bargain; the additional-cost sacrifice records it on the spell |
| "up to one target" triggers | trigger keys `targets`, `up_to` | a "No target" option; an up-to trigger is never removed for lack of targets (603.3d) |
| leaves-the-battlefield triggers | event `leaves_battlefield` | Sewer-veillance Cam (fires from last-known information) |
| animated artifacts | `animate_target {power, toughness, keywords}`, card state `animated`, `granted` | Kenku Artificer makes a noncreature artifact a 0/0 flier with three +1/+1 counters; it stays a creature until it leaves the battlefield |
| counters with keywords | `counters_on_target {n, keywords}` | Kenku's +1/+1 counters, Unexpected Fangs' lifelink counter |
| tap or untap | `tap_or_untap_target` | Sewer-veillance Cam |
| return from graveyard | `return_from_graveyard {type, n}` | Blood Fountain |
| flash | keyword `flash` | casts a non-instant at instant speed (Sewer-veillance Cam) |
| collect evidence | card key `collect_evidence = N`, cast mode `evidence`, decision `exile_from_graveyard` | Extract a Confession: offered only when the graveyard holds mana value N; exiles cards one at a time |
| edict | `opponent_sacrifices {greatest_power_if_evidence}` | Extract a Confession |
| Equipment | card keys `equipped_power`, `equipped_toughness`; op `attach_source_to_target` (equip ability, sorcery speed); `create_token.attach_source` (job select) | Black Mage's Rod. An attached permanent without bestow is Equipment: it adds its bonus instead of the bestow counters and keeps its own types; it falls off (stays on the battlefield) when the creature leaves or stops being a creature, or when it becomes a creature itself (Kenku, CR 301.5c) |
| granted cast trigger | trigger condition `{ equipped = true, spell = ... }` | "Equipped creature has 'whenever you cast a noncreature spell ...'": the Rod's own `you_cast` trigger, live only while attached |
| sorcery counter | target kind `sorcery_spell` (from #37) | Envelop |

No `custom` effects. Simplifications, all invisible in this pool: Black
Mage's Rod deals its 1 damage as the source (the rules say the equipped
creature does; nothing here tells them apart) and does not make the creature
a Wizard;
Blood Fountain chooses its two creature cards on resolution rather than
targeting them (differs only if the graveyard changes in response), and Kenku
Artificer's Homunculus subtype is not tracked.

## Card shapes (feature set 5)

New spec fields are classified in `SHAPE_*_FIELDS` in both engines, with
these shape tokens (new ops and the `leaves_battlefield` event get their
`e:*:op:` / `e:trig:` tokens automatically):

| field | tokens |
|---|---|
| card `collect_evidence` | `e:cost:collect_evidence` |
| card `equipped_power` / `equipped_toughness` | `e:equipment_bonus` |
| op flag `create_token.attach_source` | `<prefix>create_token:attach_source` |
| trigger condition `equipped` | `e:trig:cond:equipped:true` |
| trigger `up_to` | `e:trig:up_to` |
| trigger `targets` (from #37) | `e:trig:target:<kind>`, `e:target:<kind>` |
| card `bargain` (from #37) | `e:cost:bargain` |
| phyrexian mana in `cost` (from #37) | `e:cost:phyrexian` |
| op params `n_metalcraft`, `sacrificed_mv`, `greatest_power_if_evidence` | `<prefix><op>:<param>` |

Trigger conditions with non-string values (`{ bargained = true }`) render as
`e:trig:cond:bargained:true` in both engines.

## Bot

`bots/affinity.py` (a `JundBot` subclass, which already knows the shared
cards): artifact lands first, then the cheap artifacts, then the affinity
threats and Thoughtcast once they are cheap; Black Mage's Rod is cast as a
cheap artifact and re-equipped (evasive creature first, then the strongest)
when it falls off and no threat is castable; Envelop counters an opponent's
sorcery; Kenku animates an indestructible Bridge; Galvanic Blast kills a creature worth it or goes to the face at 6
life or less; Reckoner's Bargain sacrifices spare artifacts.

## Bot games vs the Q3 data

200 bot-vs-bot games per matchup, Affinity in seat 1, starting player
alternating, seeds 1000-1199, max 60 turns; identical results on the Python
and Rust engines (the project owner's 8-core Mac: 8 to 25 s for all 1200
games on Rust, 23 to 80 s on Python, depending on load). Game 2 uses both
decks' sideboard plans. Q3 is the Grixis Affinity match win rate in the
Q3 2026 data (Jund Midrange stands in for Jund Wildfire).

| matchup | Affinity wins, game 1 | game 2 | Q3 (matches) |
|---|---|---|---|
| vs Jund Wildfire | 55.0% (59.0% before) | 64.5% (63.5%) | 37.7% vs Jund Midrange (n=102) |
| vs Mono Blue Terror | 49.0% (46.5%) | 43.5% (46.5%) | 49.5% (n=103) |
| vs Red Madness | 56.0% (57.0%) | 71.0% (62.5%) | 62.4% (n=178) |

The current numbers are with the most played builds (2026-10-07, Rust
engine, the same seeds; starting player `seed % 2`), the ones in
parentheses with the earlier modal list and sideboards.

Blue and Red land close to the real data. Jund does about 20 points worse
here than Jund Midrange does in the data. Seen in the traces: the Jund bot
stacks five Makeshift Munitions activations (sacrificing three lands) on a
1/1 that the first one kills, and Cast Down cannot answer an animated,
indestructible Bridge. Whether the gap is the bots or the archetype
difference is open. Jund's artifact hate does not help its bot in game 2
(Affinity goes from 59.0% to 63.5%): the Jund bot's rules for Troublemaker
Ouphe, Breath Weapon and Ancient Grudge come from #37 and are worth a look.
Games end by damage (4 of 200 game-1 Jund games by decking, after board
stalls); none hit the turn limit.

## Traces read

Three full Python-engine game-1 logs (seed 1000 of each matchup) were read
for rules errors; none found. Checked on the way: affinity costs against the
artifact count (Myr Enforcer for free with 8 artifacts, Thoughtcast and
Familiar for one mana), Bridges entering tapped and untapped artifact lands
not, metalcraft Galvanic Blast killing a 3/2, Reckoner's Bargain sacrifices,
Kenku Artificer animating a Bridge that attacks the same turn (it had been
under control since an earlier turn), Nihil Spellbomb's optional {B} draw,
Blood Fountain returning two Familiars, Makeshift Munitions stacks fizzling
once the target died.
