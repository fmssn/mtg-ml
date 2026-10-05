"""The two fixed Pauper decklists, their sideboards and sideboard plans.

Source: MTGGoldfish archetype pages for Pauper "Jund Wildfire" and
"Blue Terror", fetched 2026-10-05.
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

# Sideboards: lean and matchup-relevant only (6 cards each), built from public
# sideboard guides for this matchup rather than a specific 15-card list.
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

# Games 2 and 3 of a match: what each deck boards in and out against the other.
SIDEBOARD_PLANS = {
    "jund_wildfire": {
        "in": {"Red Elemental Blast": 2, "Go for the Throat": 2, "Duress": 2},
        "out": {"Lembas": 2, "Toxin Analysis": 2, "Makeshift Munitions": 1, "Nyxborn Hydra": 1},
    },
    "mono_blue_terror": {
        "in": {"Blue Elemental Blast": 2, "Dispel": 2, "Steel Sabotage": 2},
        "out": {"Force Spike": 3, "Sleep of the Dead": 2, "Deem Inferior": 1},
    },
}

DECKS = {"jund_wildfire": JUND_WILDFIRE, "mono_blue_terror": MONO_BLUE_TERROR}
SIDEBOARDS = {"jund_wildfire": JUND_WILDFIRE_SIDEBOARD, "mono_blue_terror": MONO_BLUE_TERROR_SIDEBOARD}


def postboard(deck: str) -> dict[str, int]:
    """The maindeck of `deck` after applying its sideboard plan."""
    main, side, plan = dict(DECKS[deck]), SIDEBOARDS[deck], SIDEBOARD_PLANS[deck]
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


assert sum(JUND_WILDFIRE.values()) == 60
assert sum(MONO_BLUE_TERROR.values()) == 60
for _d in DECKS:
    assert sum(postboard(_d).values()) == 60 and sum(SIDEBOARDS[_d].values()) <= 15
