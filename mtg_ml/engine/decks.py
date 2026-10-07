"""The fixed Pauper decklists, their sideboards and sideboard plans.

Sources: MTGGoldfish archetype pages for Pauper "Jund Wildfire" and
"Blue Terror", fetched 2026-10-05; Red Madness: the 60 + 15 given by the
project owner on 2026-10-07 (a 2026 Pauper Challenge style list).
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

# Sideboards: Jund and Blue are lean and matchup-relevant only (6 cards each),
# built from public sideboard guides for their matchup rather than a specific
# 15-card list. Red Madness has its full 15.
JUND_WILDFIRE_SIDEBOARD = {
    "Red Elemental Blast": 2,
    "Go for the Throat": 2,
    "Duress": 2,
}
MONO_BLUE_TERROR_SIDEBOARD = {
    "Blue Elemental Blast": 2,
    "Dispel": 2,
    "Steel Sabotage": 2,
}
RED_MADNESS_SIDEBOARD = {
    "Gorilla Shaman": 2,
    "Martyr of Ashes": 2,
    "Pyroblast": 4,
    "Searing Blaze": 2,
    "Electrickery": 2,
    "Relic of Progenitus": 3,
}

# Games 2 and 3 of a match: what a deck boards in and out, by (deck, opponent).
SIDEBOARD_PLANS = {
    ("jund_wildfire", "mono_blue_terror"): {
        "in": {"Red Elemental Blast": 2, "Go for the Throat": 2, "Duress": 2},
        "out": {"Lembas": 2, "Toxin Analysis": 2, "Makeshift Munitions": 1, "Nyxborn Hydra": 1},
    },
    ("mono_blue_terror", "jund_wildfire"): {
        "in": {"Blue Elemental Blast": 2, "Dispel": 2, "Steel Sabotage": 2},
        "out": {"Force Spike": 3, "Sleep of the Dead": 2, "Deem Inferior": 1},
    },
    # Against Red Madness Jund wants cheap removal for Guttersnipe and
    # Flamebreather and Duress for burn; Red Elemental Blast has no target.
    # Lembas, Toxin Analysis (lifelink) and Munitions (pings x/1s) stay in.
    ("jund_wildfire", "red_madness"): {
        "in": {"Go for the Throat": 2, "Duress": 2},
        "out": {"Cleansing Wildfire": 2, "Nyxborn Hydra": 1, "Eviscerator's Insight": 1},
    },
    # Red against Jund: Electrickery (overloaded) sweeps Spawn, Krark-Clan
    # Shaman, Gixian Infiltrator and Refurbished Familiar; Searing Blaze kills
    # Writhing Chrysalis; Gorilla Shaman eats Clues, Maps, Spellbombs and
    # Wellsprings. Out: the x/1s that die to Krark-Clan Shaman and the
    # slowest card draw.
    ("red_madness", "jund_wildfire"): {
        "in": {"Electrickery": 2, "Searing Blaze": 2, "Gorilla Shaman": 2},
        "out": {"Voldaren Epicure": 2, "Highway Robbery": 2, "Guttersnipe": 1, "Sneaky Snacker": 1},
    },
    ("mono_blue_terror", "red_madness"): {
        "in": {"Blue Elemental Blast": 2, "Dispel": 2},
        "out": {"Sleep of the Dead": 2, "Deem Inferior": 2},
    },
    ("red_madness", "mono_blue_terror"): {
        "in": {"Pyroblast": 4, "Electrickery": 2},
        "out": {"Highway Robbery": 4, "Grab the Prize": 2},
    },
}

DECKS = {"jund_wildfire": JUND_WILDFIRE, "mono_blue_terror": MONO_BLUE_TERROR, "red_madness": RED_MADNESS}
SIDEBOARDS = {"jund_wildfire": JUND_WILDFIRE_SIDEBOARD, "mono_blue_terror": MONO_BLUE_TERROR_SIDEBOARD, "red_madness": RED_MADNESS_SIDEBOARD}


def postboard(deck: str, opponent: str) -> dict[str, int]:
    """The maindeck of `deck` after applying its sideboard plan against `opponent`."""
    main, side, plan = dict(DECKS[deck]), SIDEBOARDS[deck], SIDEBOARD_PLANS[(deck, opponent)]
    for name, n in plan["out"].items():
        if main.get(name, 0) < n:
            raise ValueError(f"{deck}: cannot board out {n} {name}")
        main[name] -= n
        if not main[name]:
            del main[name]
    for name, n in plan["in"].items():
        if side.get(name, 0) < n:
            raise ValueError(f"{deck}: sideboard has fewer than {n} {name}")
        main[name] = main.get(name, 0) + n
    return main


def expand(decklist: dict[str, int]) -> list[str]:
    return [name for name, n in decklist.items() for _ in range(n)]


for _d, _list in DECKS.items():
    assert sum(_list.values()) == 60 and sum(SIDEBOARDS[_d].values()) <= 15, _d
for _d, _o in SIDEBOARD_PLANS:
    assert sum(postboard(_d, _o).values()) == 60
