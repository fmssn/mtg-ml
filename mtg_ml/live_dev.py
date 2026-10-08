"""Dev scenarios for the play client: games that start in a set position, so
the board states the UI must handle (a crowded board, a damage assignment,
floating mana) can be reached on demand. Offered only with
`serve --scripted-bot` (a development server), never to players.

A scenario is described from the human's side: `me` and `opp` take the
keys of `tests/helpers.scenario` (hand, battlefield, graveyard, library,
life); battlefield entries are names or (name, add_card kwargs). The game
starts at `step` of the human's turn.
"""

from __future__ import annotations

from .backend import game_class
from .match import deck_names, matchup_decks

BASIC = {"jund_wildfire": "Swamp", "mono_blue_terror": "Island", "red_madness": "Mountain", "grixis_affinity": "Island", "elves": "Forest", "tron": "Forest"}

SCENARIOS: dict[str, dict] = {
    "crowded": {
        "title": "Crowded board (Elves, 15 creatures)",
        "matchup": "madness_elves",
        "seat": 1,
        "me": {
            "battlefield": ["Llanowar Elves"] * 3 + ["Elvish Mystic"] * 2 + ["Fyndhorn Elves"] * 2 + ["Timberwatch Elf"] * 2
            + ["Priest of Titania", "Quirion Ranger", ("Nyxborn Hydra", {"counters": 3}), "Avenging Hunter", "Masked Vandal", "Sagu Wildling"]
            + ["Forest"] * 6 + ["Gingerbread Cabin"],
            "hand": ["Lead the Stampede", "Winding Way", "Forest", "Elvish Mystic"],
        },
        "opp": {"battlefield": ["Sneaky Snacker"] * 2 + ["Voldaren Epicure"] * 2 + ["Kessig Flamebreather", "Guttersnipe"] + ["Mountain"] * 5, "hand": ["Lightning Bolt"] * 3},
    },
    "trample": {
        "title": "Damage assignment (trampling Hydra, they must chump)",
        "matchup": "madness_elves",
        "seat": 1,
        "me": {"battlefield": [("Nyxborn Hydra", {"counters": 5}), "Forest", "Forest", "Forest"], "hand": ["Forest"]},
        "opp": {"battlefield": ["Sneaky Snacker", "Voldaren Epicure", "Mountain"], "life": 4},
    },
    "regress": {
        "title": "Regression: drop an uncastable card, attack two turns running",
        "matchup": "madness_elves",
        "seat": 1,
        "me": {"battlefield": ["Masked Vandal", "Forest"], "hand": ["Generous Ent"]},
        "opp": {"battlefield": ["Mountain", "Mountain"], "library": ["Mountain"] * 20},
    },
    "reduced": {
        "title": "Reduced cost (Tolarian Terror, three instants in the graveyard)",
        "matchup": "jund_blue",
        "seat": 1,
        "me": {"hand": ["Tolarian Terror", "Counterspell"], "graveyard": ["Brainstorm", "Ponder", "Counterspell"], "battlefield": ["Island"] * 3},
        "opp": {"battlefield": ["Swamp", "Mountain"]},
    },
    "trade": {
        "title": "Combat death (attack a 1/1 into their 2/2)",
        "matchup": "madness_elves",
        "seat": 1,
        "me": {"battlefield": ["Elvish Mystic", "Llanowar Elves", "Forest"]},
        "opp": {"battlefield": ["Guttersnipe", "Mountain"]},
    },
    "instant": {
        "title": "Instant on their turn (Bolt with mana up, they attack)",
        "matchup": "jund_madness",
        "seat": 1,
        "active": "opp",
        "me": {"battlefield": ["Mountain", "Mountain", "Kessig Flamebreather"], "hand": ["Lightning Bolt", "Lightning Bolt"]},
        "opp": {"battlefield": ["Swamp", "Forest", "Mountain", "Tolarian Terror", "Gixian Infiltrator"]},
    },
    "floating": {
        "title": "Floating mana (Tron lands pay a one-drop)",
        "matchup": "jund_tron",
        "seat": 1,
        "me": {"battlefield": ["Urza's Mine", "Urza's Power Plant", "Urza's Tower"], "hand": ["Expedition Map", "Candy Trail", "Bonder's Ornament"]},
        "opp": {"battlefield": ["Swamp", "Mountain"]},
    },
}


def options() -> dict[str, str]:
    return {k: v["title"] for k, v in SCENARIOS.items()}


def new_game(name: str, seed: int, engine: str | None):
    """(game, matchup, human seat) for scenario `name`."""
    sc = SCENARIOS[name]
    matchup, seat = sc["matchup"], sc["seat"]
    decks = matchup_decks(matchup)
    specs = {seat: sc["me"], 1 - seat: sc["opp"]}

    def setup(g) -> None:
        for idx in (0, 1):
            spec = specs[idx]
            for zone in ("hand", "graveyard", "exile"):
                for card in spec.get(zone, []):
                    g.add_card(card, idx, zone)
            for entry in spec.get("battlefield", []):
                card, kw = (entry, {}) if isinstance(entry, str) else entry
                g.add_card(card, idx, "battlefield", **kw)
            for card in spec.get("library", [BASIC[decks[idx]]] * 20):
                g.add_card(card, idx, "library")
            g.players[idx].life = spec.get("life", 20)

    first = 1 - seat if sc.get("active") == "opp" else seat
    g = game_class(engine)(([], []), seed=seed, starting_player=first, setup=setup, start_step=sc.get("step", "main1"), log=True, deck_names=deck_names(matchup))
    return g, matchup, seat
