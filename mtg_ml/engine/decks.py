"""The fixed Pauper decklists and their sideboards.

Sideboard plans (what comes in and out against each opponent) are data:
`sideboard_plans.toml`, loaded and validated by `sideboard.py`.

Sources: MTGGoldfish archetype pages for Pauper "Jund Wildfire" and
"Blue Terror", fetched 2026-10-05; Red Madness: the 60 + 15 given by the
project owner on 2026-10-07 (a 2026 Pauper Challenge style list); Tron: the
typical list of 443 Q3 2026 lists (docs/tron.md).
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

# Tron (Q3 2026, the green artifact-ramp build): the typical list of all
# 443 Tron lists from 2026-07-05 to 2026-10-02 (Pauper-Research dashboard,
# "Builds and decisions": the aggregate decklist), unchanged (docs/tron.md).
TRON = {
    "Urza's Mine": 4,
    "Urza's Power Plant": 4,
    "Urza's Tower": 4,
    "Forest": 2,
    "Bojuka Bog": 1,
    "Conduit Pylons": 1,
    "Haunted Fengraf": 1,
    "Expedition Map": 4,
    "Ancient Stirrings": 4,
    "Crop Rotation": 3,
    "Barrels of Blasting Jelly": 4,
    "Giant's Boulder": 4,
    "Bonder's Ornament": 2,
    "Candy Trail": 3,
    "Unfathomable Truths": 2,
    "Bramble Wurm": 4,
    "Generous Ent": 2,
    "Boulderbranch Golem": 3,
    "Pinnacle Kill-Ship": 4,
    "Maelstrom Colossus": 4,
}

# Sideboards: Jund and Blue are lean and matchup-relevant only (6 cards each),
# built from public sideboard guides for their matchup rather than a specific
# 15-card list. Red Madness has its full 15.
JUND_WILDFIRE_SIDEBOARD = {
    "Red Elemental Blast": 2,
    "Go for the Throat": 2,
    "Duress": 2,
    "Weather the Storm": 3,
}
MONO_BLUE_TERROR_SIDEBOARD = {
    "Blue Elemental Blast": 2,
    "Dispel": 2,
    "Steel Sabotage": 2,
}
# Tron: the typical 15 of the same 443 lists, unchanged (docs/tron.md).
TRON_SIDEBOARD = {
    "Breath Weapon": 3,
    "Relic of Progenitus": 3,
    "Call Damage Control": 2,
    "Monstrous Emergence": 2,
    "Scour from Existence": 2,
    "Kaervek's Torch": 1,
    "Pulse of Murasa": 1,
    "Whispersilk Cloak": 1,
}
RED_MADNESS_SIDEBOARD = {
    "Gorilla Shaman": 2,
    "Martyr of Ashes": 2,
    "Pyroblast": 4,
    "Searing Blaze": 2,
    "Electrickery": 2,
    "Relic of Progenitus": 3,
}

DECKS = {"jund_wildfire": JUND_WILDFIRE, "mono_blue_terror": MONO_BLUE_TERROR, "red_madness": RED_MADNESS, "tron": TRON}
SIDEBOARDS = {
    "jund_wildfire": JUND_WILDFIRE_SIDEBOARD,
    "mono_blue_terror": MONO_BLUE_TERROR_SIDEBOARD,
    "red_madness": RED_MADNESS_SIDEBOARD,
    "tron": TRON_SIDEBOARD,
}


def expand(decklist: dict[str, int]) -> list[str]:
    return [name for name, n in decklist.items() for _ in range(n)]


for _d, _list in DECKS.items():
    assert sum(_list.values()) == 60 and sum(SIDEBOARDS[_d].values()) <= 15, _d
