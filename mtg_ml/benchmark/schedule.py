"""Deterministic row identities, independent of the legacy matchup registry."""

from dataclasses import dataclass
import hashlib

from .artifacts import MODES, require, integer
from .jsonio import canonical_bytes


def seed(parts):
    return int.from_bytes(hashlib.sha256(canonical_bytes(parts)).digest()[:8], "big") % (2**31)


def simulator_seed(stream, cell, block):
    return seed([stream, cell, block])


def actor_seed(stream, cell, block, slot, mode):
    return seed([stream, cell, block, slot, mode, "actor"])


def puzzle_seed(stream, puzzle, repetition, mode):
    return seed([stream, puzzle, repetition, mode, "actor"])


def bootstrap_seed(stream, suite, mode):
    return seed([stream, suite, mode, "bootstrap"])


def expand(cards):
    return tuple(name for name, n in sorted(cards.items()) for _ in range(n))


@dataclass(frozen=True)
class EpisodeSpec:
    mode: str
    cell: str
    block: int
    slot: int
    simulator_seed: int
    actor_seed: int
    learner_seat: int
    starting_player: int
    deck_ids: tuple[str, str]
    decks: tuple[tuple[str, ...], tuple[str, ...]]
    opponent: str

    @property
    def identity(self):
        return self.mode, self.cell, self.block, self.slot

    def game_args(self, settings):
        return dict(decks=self.decks, registered_main=self.decks, registered_sideboards=((), ()), deck_names=self.deck_ids,
                    seed=self.simulator_seed, starting_player=self.starting_player, match_game=1, log=True,
                    **{k: v for k, v in settings.items() if k != "max_decisions"})


def episodes(manifest, cells=None, modes=None):
    d = manifest.data if hasattr(manifest, "data") else manifest
    selected = [c["id"] for c in d["cells"]] if cells is None else list(cells)
    modes = d["modes"] if modes is None else list(modes)
    require(bool(selected) and len(set(selected)) == len(selected) and set(selected) <= {c["id"] for c in d["cells"]}, "cells", "invalid selection")
    require(bool(modes) and len(set(modes)) == len(modes) and set(modes) <= set(d["modes"]) & MODES, "modes", "invalid selection")
    integer(d["blocks_per_cell"], "blocks_per_cell")
    bots = {b["id"]: b for b in d["bots"]}
    out = []
    for mode in modes:
        for cell in d["cells"]:
            if cell["id"] not in selected:
                continue
            own, opp = cell["learner_deck"], bots[cell["opponent"]]["deck"]
            for block in range(d["blocks_per_cell"]):
                s = simulator_seed(d["stream"], cell["id"], block)
                for slot, (seat, start) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
                    ids = (own, opp) if seat == 0 else (opp, own)
                    out.append(EpisodeSpec(mode, cell["id"], block, slot, s, actor_seed(d["stream"], cell["id"], block, slot, mode),
                                           seat, start, ids, tuple(expand(d["decks"][name]["cards"]) for name in ids), cell["opponent"]))
    return sorted(out, key=lambda x: x.identity)
