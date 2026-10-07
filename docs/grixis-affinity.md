# Grixis Affinity

The fourth deck: Pauper Grixis Affinity (artifact lands and cheap artifacts
that make Myr Enforcer, Utrom Monitor, Refurbished Familiar and Thoughtcast
free), with its full 15-card sideboard, in both engines. Matchups put Affinity
in seat 1: `jund_affinity`, `blue_affinity`, `madness_affinity`
(`match.MATCHUPS`).

## Decklist (`engine/decks.py`)

The modal count of each card over the 524 Grixis Affinity lists in the
Q3 2026 Pauper-Research data (cards in at least half of the lists, then the
most played to reach 60 / 15). Cards shared with Jund Wildfire (Refurbished
Familiar, Krark-Clan Shaman, Ichor Wellspring, Nihil Spellbomb, Toxin
Analysis, Makeshift Munitions, Drossforge Bridge, Vault of Whispers) reuse
their existing specs.

| main | | sideboard | |
|---|---|---|---|
| 4 Myr Enforcer | 4 Ichor Wellspring | 4 Hydroblast | 1 Red Elemental Blast |
| 4 Refurbished Familiar | 3 Nihil Spellbomb | 2 Blue Elemental Blast | 1 Krark-Clan Shaman |
| 3 Utrom Monitor | 2 Blood Fountain | 2 Extract a Confession | 1 Unexpected Fangs |
| 3 Krark-Clan Shaman | 1 Sewer-veillance Cam | 4 Pyroblast | |
| 2 Kenku Artificer | 1 Makeshift Munitions | | |
| 4 Thoughtcast | 4 Mistvault Bridge | | |
| 4 Galvanic Blast | 4 Drossforge Bridge | | |
| 4 Reckoner's Bargain | 4 Vault of Whispers | | |
| 2 Toxin Analysis | 2 Seat of the Synod | | |
| | 2 Silverbluff Bridge, 2 Great Furnace, 1 Mountain | | |

Left out: Black Mage's Rod (33% of lists) and Envelop (39% of sideboards).

## Sideboard plans (games 2 and 3)

Rows in `engine/sideboard_plans.toml` (format: [sideboarding.md](sideboarding.md)):

| deck vs opponent | in | out |
|---|---|---|
| Affinity vs Jund | 2 Hydroblast, 2 Extract a Confession, 1 Krark-Clan Shaman | 2 Kenku Artificer, 1 Sewer-veillance Cam, 1 Makeshift Munitions, 1 Toxin Analysis |
| Jund vs Affinity | 2 Red Elemental Blast, 2 Duress | 2 Cleansing Wildfire, 2 Lembas |
| Affinity vs Blue | 4 Pyroblast, 1 Red Elemental Blast | 2 Toxin Analysis, 2 Kenku Artificer, 1 Makeshift Munitions |
| Blue vs Affinity | 2 Steel Sabotage, 2 Dispel, 2 Blue Elemental Blast | 2 Sleep of the Dead, 3 Force Spike, 1 Plunder the Trollshaws |
| Affinity vs Red | 4 Hydroblast, 2 Blue Elemental Blast, 1 Unexpected Fangs, 1 Krark-Clan Shaman | 2 Kenku Artificer, 3 Nihil Spellbomb, 1 Sewer-veillance Cam, 1 Makeshift Munitions, 1 Blood Fountain |
| Red vs Affinity | 2 Gorilla Shaman, 2 Electrickery, 2 Pyroblast | 2 Highway Robbery, 2 Grab the Prize, 2 Sneaky Snacker |

Rows for Elves and Tron (both directions) are at the end of the file,
commented out until those decks are in `decks.DECKS`.

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

No `custom` effects. Simplifications, all invisible in this pool:
Blood Fountain chooses its two creature cards on resolution rather than
targeting them (differs only if the graveyard changes in response), and Kenku
Artificer's Homunculus subtype is not tracked.

## Bot

`bots/affinity.py` (a `JundBot` subclass, which already knows the shared
cards): artifact lands first, then the cheap artifacts, then the affinity
threats and Thoughtcast once they are cheap; Kenku animates an indestructible
Bridge; Galvanic Blast kills a creature worth it or goes to the face at 6
life or less; Reckoner's Bargain sacrifices spare artifacts.

## Bot games vs the Q3 data

200 bot-vs-bot games per matchup, Affinity in seat 1, starting player
alternating, seeds 1000-1199, max 60 turns; identical results on the Python
and Rust engines (the project owner's 8-core Mac: about 25 s for all 1200
games on Rust, 40 to 80 s on Python depending on load). Game 2 uses both
decks' sideboard plans. Q3 is the Grixis Affinity match win rate in the Q3 2026 data (Jund Midrange stands in for Jund
Wildfire).

| matchup | Affinity wins, game 1 | game 2 | Q3 (matches) |
|---|---|---|---|
| vs Jund Wildfire | 59.0% | 58.0% | 37.7% vs Jund Midrange (n=102) |
| vs Mono Blue Terror | 46.5% | 54.0% | 49.5% (n=103) |
| vs Red Madness | 57.0% | 62.5% | 62.4% (n=178) |

Blue and Red land close to the real data. Jund does about 20 points worse
here than Jund Midrange does in the data. Seen in the traces: the Jund bot
stacks five Makeshift Munitions activations (sacrificing three lands) on a
1/1 that the first one kills, and Cast Down cannot answer an animated,
indestructible Bridge. Whether the gap is the bots or the archetype
difference is open. Games end by damage (4 of 200 Jund games by decking,
after board stalls); none hit the turn limit.

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
