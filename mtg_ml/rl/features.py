"""Turn a decision point into network inputs.

State: the hashed sparse features of `encode.state_features` plus the seat
(which deck we are) and the decision kind, as indices into a state
vocabulary. Options: each legal option's key is expanded into hashed tokens
(every key element with its position, and every key prefix), so options
that share structure ("cast X", "pay with Swamp") share parameters. The
policy scores options one by one; illegal actions never exist, which is
action masking over the open-ended key vocabulary.
"""

from __future__ import annotations

import zlib

from ..encode import state_features

STATE_DIM = 1 << 16
OPTION_DIM = 1 << 15


def _h(s: str, dim: int) -> int:
    return zlib.crc32(s.encode()) % dim


def option_tokens(kind: str, key: tuple) -> list[str]:
    full = (kind,) + tuple(key)
    toks = [f"{i}={e}" for i, e in enumerate(full)]
    toks += ["|".join(map(str, full[: n + 1])) for n in range(1, len(full))]
    return toks


def featurize(game, player: int, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM):
    """(state indices, [option indices, ...]) for `player` at the current decision."""
    d = game.decision
    feats = state_features(game, player)
    feats.append(f"seat:{player}")
    feats.append(f"decision:{d.kind}")
    state = sorted({_h(f, state_dim) for f in feats})
    opts = [sorted({_h(t, option_dim) for t in option_tokens(d.kind, o.key)}) for o in game.legal_options()]
    return state, opts


# Decision kinds whose chosen option is public when the opponent makes it.
# Excluded: choose_card / order / choose_mode keys can name hidden cards
# (Brainstorm put-backs, Ponder order, scry and Deem Inferior choices).
PUBLIC_KINDS = frozenset(
    {
        "priority",
        "choose_x",
        "target",
        "pay_mana",
        "sacrifice",
        "exile_from_graveyard",
        "yes_no",
        "order_triggers",
        "declare_attacker",
        "declare_blocker",
        "assign_damage",
        "mulligan",
    }
)
MAX_EVENT_TOKENS = 256


def event_tokens(kind: str, key: tuple, mine: bool) -> list[str]:
    """Tokens describing a taken action, as seen by a player: their own
    action ('self>'), or the opponent's public action ('opp>')."""
    if not mine and kind not in PUBLIC_KINDS:
        return [f"opp>{kind}"]
    tag = "self>" if mine else "opp>"
    return [tag + t for t in option_tokens(kind, key)]


def encode_events(tokens: list[str], option_dim: int = OPTION_DIM) -> list[int]:
    """Hash the tokens gathered since a player's previous decision (a bag
    with counts), plus a bucketed count of how much happened."""
    n = len(tokens)
    tokens = tokens[-MAX_EVENT_TOKENS:] + [f"events:{min(n // 8, 16)}"]
    return [_h(t, option_dim) for t in tokens]
