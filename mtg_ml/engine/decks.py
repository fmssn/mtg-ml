"""The fixed Pauper decklists and their sideboards.

Sideboard plans (what comes in and out against each opponent) are data:
`sideboard_plans.toml`, loaded and validated by `sideboard.py`.

Maindecks: MTGGoldfish archetype pages for Pauper "Jund Wildfire" and
"Blue Terror", fetched 2026-10-05; Red Madness: the 60 given by the project
owner on 2026-10-07 (a 2026 Pauper Challenge style list).

Sideboards (since 2026-10-07): the typical list of the most played
sideboard build in the Q3 2026 data of the Pauper-Research project (fmssn,
2026-07-05 to 2026-10-02, all events; dashboard "Builds and decisions",
`prototypes/deck_decisions/data.js`; "Jund Midrange" there is Jund
Wildfire). Jund: the "Stock list" build, 124 of 428 lists (29%); Mono Blue
Terror: "Stock list", 261 of 666 (39%); Red Madness: one build, all 970
lists. Each is that build's typical 15 card for card: nothing is left out
for missing rules (Envelop was implemented for it) and nothing is trimmed.
"""

JUND_WILDFIRE = {
    "Refurbished Familiar": 4,
    "Writhing Chrysalis": 4,
    "Krark-Clan Shaman": 3,
    "Nyxborn Hydra": 1,
    "Gixian Infiltrator": 2,
    "Cleansing Wildfire": 4,
    "Cast Down": 4,
    "Fanatical Offering": 4,
    "Eviscerator's Insight": 2,
    "Toxin Analysis": 2,
    "Makeshift Munitions": 1,
    "Ichor Wellspring": 4,
    "Lembas": 2,
    "Nihil Spellbomb": 3,
    "Drossforge Bridge": 4,
    "Slagwoods Bridge": 4,
    "Twisted Landscape": 4,
    "Vault of Whispers": 2,
    "Swamp": 3,
    "Mountain": 1,
    "Forest": 2,
}

MONO_BLUE_TERROR = {
    "Island": 16,
    "Delver of Secrets": 4,
    "Tolarian Terror": 4,
    "Cryptic Serpent": 4,
    "Brainstorm": 4,
    "Ponder": 3,
    "Thought Scour": 4,
    "Mental Note": 4,
    "Counterspell": 4,
    "Lorien Revealed": 4,
    "Force Spike": 3,
    "Deem Inferior": 3,
    "Sleep of the Dead": 2,
    "Plunder the Trollshaws": 1,
}

RED_MADNESS = {
    "Guttersnipe": 3,
    "Kessig Flamebreather": 4,
    "Voldaren Epicure": 4,
    "Sneaky Snacker": 4,
    "Lightning Bolt": 4,
    "Lava Dart": 4,
    "Fireblast": 3,
    "Fiery Temper": 4,
    "Faithless Looting": 4,
    "Highway Robbery": 4,
    "Grab the Prize": 4,
    "Mountain": 18,
}

# Sideboards: exactly 15 cards each, from the Q3 2026 typical lists (see the
# module docstring). The maindecks above are left as they were so the
# benchmark stays comparable across PRs.
JUND_WILDFIRE_SIDEBOARD = {
    "Weather the Storm": 3,
    "Duress": 3,
    "Troublemaker Ouphe": 2,
    "Breath Weapon": 2,
    "Red Elemental Blast": 2,
    "Faerie Macabre": 1,
    "Nihil Spellbomb": 1,
    "Terminate": 1,
}
MONO_BLUE_TERROR_SIDEBOARD = {
    "Hydroblast": 4,
    "Annul": 4,
    "Blue Elemental Blast": 3,
    "Gut Shot": 3,
    "Envelop": 1,
}
RED_MADNESS_SIDEBOARD = {
    "Pyroblast": 4,
    "Relic of Progenitus": 3,
    "Red Elemental Blast": 2,
    "Cast into the Fire": 2,
    "Searing Blaze": 2,
    "End the Festivities": 2,
}

DECKS = {"jund_wildfire": JUND_WILDFIRE, "mono_blue_terror": MONO_BLUE_TERROR, "red_madness": RED_MADNESS}
SIDEBOARDS = {"jund_wildfire": JUND_WILDFIRE_SIDEBOARD, "mono_blue_terror": MONO_BLUE_TERROR_SIDEBOARD, "red_madness": RED_MADNESS_SIDEBOARD}


def expand(decklist: dict[str, int]) -> list[str]:
    return [name for name, n in decklist.items() for _ in range(n)]


for _d, _list in DECKS.items():
    assert sum(_list.values()) == 60 and sum(SIDEBOARDS[_d].values()) == 15, _d
