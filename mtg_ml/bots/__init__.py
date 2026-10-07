"""Scripted heuristic bots, one per deck, as fixed baseline opponents.

    from mtg_ml.bots import make_bot
    agents = [make_bot(0), make_bot(1)]   # Jund Wildfire, Mono Blue Terror
    agents = [make_bot(0), make_bot(1, "red_madness")]
"""

from .affinity import AffinityBot
from .blue import BlueBot
from .jund import JundBot
from .red import RedBot

BOTS = {"jund_wildfire": JundBot, "mono_blue_terror": BlueBot, "red_madness": RedBot, "grixis_affinity": AffinityBot}


def make_bot(seat: int, deck: str | None = None):
    """The bot for `deck` in `seat` (default: the seat's usual deck, 0 =
    Jund Wildfire, 1 = Mono Blue Terror)."""
    if deck is None:
        deck = ("jund_wildfire", "mono_blue_terror")[seat]
    return BOTS[deck](seat)


__all__ = ["AffinityBot", "BOTS", "BlueBot", "JundBot", "RedBot", "make_bot"]
