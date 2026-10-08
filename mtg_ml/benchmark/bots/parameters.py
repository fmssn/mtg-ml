"""Immutable version-one Jund valuations, thresholds and priority bands."""

from ..views import freeze, thaw
from ..jsonio import digest

PARAMETERS = freeze({
    "combat": {"power": 1.5, "toughness": .5, "flying": 1.5, "empty_token": .3,
               "lethal": 100000., "survival": 10000., "block_life": .35,
               "attack_life": .7, "attack_gain": .4, "crackback": .15, "kill": 10.},
    "tactics": {"cheap_fodder": 2.5, "growth": 1.5, "toxin_cost": 1., "low_life": 10,
                "gain_low": .8, "gain_normal": .25, "ward_attack": 5., "engine_threat": 2.,
                "counter_risk": .5, "early_play": 4., "color_access": 2.},
    "priority": {"land": 100., "removal": 30., "toxin": 40., "sweep": 45.,
                 "save_fodder": 35., "draw": 8., "wildfire": 18., "munitions": 25.,
                 "graveyard": 14., "fetch": 16., "utility": 3., "life": 10.,
                 "bestow_attack": 10., "bestow_later": 4.},
    "sacrifice": {"needed_land": 20., "extra_land": 4., "wellspring": .2,
                  "cantrip": .6, "spellbomb_unpaid": 1.5, "token": .3,
                  "creature": 3., "other": 3., "threatened": .2},
    "keep_lands": {"7": [2, 4], "6": [2, 4]}, "land_goal": 5,
    "removal_min": 4., "spellbomb_fuel": 4, "lembas_life": 8,
    "curve": {"Writhing Chrysalis": 9., "Refurbished Familiar": 8., "Gixian Infiltrator": 7.,
              "Nyxborn Hydra": 6., "Ichor Wellspring": 5., "Lembas": 4.5,
              "Krark-Clan Shaman": 4., "Nihil Spellbomb": 4., "Makeshift Munitions": 3.},
    "cards": {"Cast Down": 5., "Writhing Chrysalis": 4.5, "Refurbished Familiar": 4.,
              "Cleansing Wildfire": 3.5, "Fanatical Offering": 3., "Eviscerator's Insight": 3.,
              "Gixian Infiltrator": 3., "Toxin Analysis": 2.5, "Nyxborn Hydra": 2.5,
              "Krark-Clan Shaman": 2., "Ichor Wellspring": 2., "Makeshift Munitions": 2.,
              "Lembas": 1.5, "Nihil Spellbomb": 1.5},
})
PARAMETERS_SHA256 = digest(thaw(PARAMETERS))
