# Tron audit (2026-10-10)

Question: the trust tests rate Tron too strong against the Q3 2026 Pauper-Research results (lr075 bias +0.16, six pilots +0.13; worst Grixis Affinity vs Tron 0.18 against 0.36 real, Jund vs Tron 0.36 against 0.53, Red Madness vs Tron 0.33-0.43 against 0.54). Is it (a) an engine or card bug, (b) the decklist, (c) the opponents' play? This audit covers (a) and (b).

**Result: no engine or card bug found in either engine; the decklist matches the source and the typical real builds. Nothing was changed in `mtg_ml/engine/` or `native/`, so no behaviour change and no golden re-record.** The remaining suspect is (c), plus pilots that play Tron's mulligan, scry and dig decisions very well.

## Cards (spec in `cards.toml` against `data/oracle_cards.json`, Python and Rust)

Every card of the 60 and the 15 was read line by line. All OK:

| card | checked |
|---|---|
| Urza's Mine / Power Plant / Tower | 2 / 2 / 3 only with the other two kinds under the same controller; 1 otherwise; opponent's lands do not count; tapped lands still count; two sets are not a double bonus (the extra units float and empty at the end of the step) |
| Forest, Bojuka Bog, Conduit Pylons, Haunted Fengraf | Bog enters tapped (also when fetched), Pylons surveil 1 and its {1},{T} filter, Fengraf returns a random creature |
| Expedition Map, Ancient Stirrings, Crop Rotation | Map {2}, T, sac; Stirrings top 5, one colorless card (lands, devoid, artifacts are colorless; green cards are not), the rest on the bottom; Crop Rotation sacrifices a land as a cost, fetched land is untapped (Bog tapped) and is not a land drop |
| Barrels of Blasting Jelly, Giant's Boulder, Bonder's Ornament | filter once per turn (Barrels), scry 2 and {7} destroy (Boulder), Ornament draws only for players who control one |
| Candy Trail, Unfathomable Truths | scry 2, {2} gain 3 and draw one; Truths draws exactly 3 and makes one Spawn, colorless |
| Bramble Wurm, Generous Ent, Boulderbranch Golem | Wurm ETB 5 and the graveyard ability once (exiles itself); Ent forestcycling finds only a Forest; Golem full 6/5 or prototype 3/3, life equal to its power |
| Pinnacle Kill-Ship | ETB 10 damage to up to one creature; station is sorcery speed, taps another untapped creature, counters equal to its power, creature (7/7 flying) only at 7+ |
| Maelstrom Colossus | one cascade (confirmed against the printed oracle text, Commander Legends); only nonland cards with lower mana value; the rest go to the bottom in random order; works when Colossus is countered |
| Sideboard (Breath Weapon, Relic, Call Damage Control, Monstrous Emergence, Scour, Kaervek's Torch, Pulse, Whispersilk Cloak) | match oracle; known simplifications are in `docs/tron.md` |

Nothing in the list loops or repeats value: Ornament draws are {4}+T each, Candy Trail and Map sacrifice themselves, Fengraf sacrifices itself, no flicker or untap effects exist in the pool. Crop Rotation can tap a land for mana, sacrifice it and fetch an untapped one that taps again, which is legal Magic.

`tests/test_cards_tron_audit.py` (both engines) adds the gap tests: exact 7 with all three kinds, no double set bonus, opponent lands, tapped lands, floating mana gone after the step, fetch is not a land drop, Bog tapped, Pylons surveil, forestcycling, station limits, cascade skips equal and higher mana value, Candy Trail.

## Decklist

`decks.TRON` is the Q3 2026 aggregate list of 443 lists unchanged (`docs/tron.md`). A web check of August 2026 Pauper Challenge lists shows the same core (4 Colossus, 4 Kill-Ship, 4 Giant's Boulder, 4 Wurm, 4 Map; others 3-4 Candy Trail, Prophetic Prism builds). No difference worth flagging; 17 lands, 12 Urza lands, 4 Map + 4 Stirrings + 3 Crop Rotation are the real tutor package.

## Game patterns (Tron pilot `r8-tron-pilot-final.pt` against each opponent's r8 pilot, game 1 main decks, native engine, 40 games each, sampled)

Tron win rate (n=40, so +-0.15): Jund 0.53, Blue 0.65, Madness 0.57, Affinity 0.75, Elves 0.35. Tron has Urza's Mine + Power Plant + Tower together on its own turn 3 in 53% of the 200 games (turn 4: 24%, turn 5 or later: 11%, never: 12%); first Kill-Ship or Colossus cast on own turn 4 (median); wins on own turn 7-10 (median) by attacking with Kill-Ship, Wurm and Golems into boards with no answer, often after Giant's Boulder removes the one blocker. Colossus is cast in 0.84 of the games and cascades into Wurm, Kill-Ship or Golem 25+16+18 times. No impossible turn-2 Tron (an earlier miscount was a Crop Rotation that sacrificed one Urza land; re-checked with leave events). Lifegain is Wurm, Golem, Candy Trail and Food only.

Reading: Tron's speed is what the deck's tutors buy a perfect-information-of-its-own-library pilot (scry 2 from Boulder and Candy Trail, surveil, Stirrings, mulligans), and the opponents rarely have an answer to a turn-4 seven-drop. That points at (c), the opponents' pace and interaction, not at an engine bug. Worth checking next: how often the opponent pilots race (damage dealt by turn 4) and whether Affinity and Madness hold their artifact and burn answers for Kill-Ship.
