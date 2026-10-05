"""Scripted heuristic bots, one per deck, as fixed baseline opponents.

    from mtg_ml.bots import make_bot
    agents = [make_bot(0), make_bot(1)]   # Jund Wildfire, Mono Blue Terror
"""

from .blue import BlueBot
from .jund import JundBot


def make_bot(seat: int):
    """The bot for the deck in `seat` (0 = Jund Wildfire, 1 = Mono Blue Terror)."""
    return JundBot(seat) if seat == 0 else BlueBot(seat)


__all__ = ["BlueBot", "JundBot", "make_bot"]
