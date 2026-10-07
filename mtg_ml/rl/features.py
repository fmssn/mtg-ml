"""Turn a decision point into network inputs.

State: the hashed sparse features of `encode.state_features` plus the seat
(which deck we are) and the decision kind, as indices into a state
vocabulary, followed by one segment per entity (`encode.entity_features`:
permanents and stack items), each opened by the separator `state_dim`.
Options also get a pointer `option_dim + k` for every entity k they are
about (the attacker, blocker, mana source, target, ...), so the network can
score "block with this Terror" from that Terror's own vector. Options: each legal option's key is expanded into hashed tokens
(every key element with its position, and every key prefix), so options
that share structure ("cast X", "pay with Swamp") share parameters, plus
`encode.option_preview` tokens (what the option would do: creatures a
sweeper kills, ward and lethal damage on a target, mana left after a cast). The
policy scores options one by one; illegal actions never exist, which is
action masking over the open-ended key vocabulary.
"""

from __future__ import annotations

import zlib

from ..encode import entity_features, option_object_ids, option_preview, state_features

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
    if getattr(game, "NATIVE", False):
        return game.featurize(player, state_dim, option_dim)
    d = game.decision
    feats = state_features(game, player)
    feats.append(f"seat:{player}")
    feats.append(f"decision:{d.kind}")
    state = sorted({_h(f, state_dim) for f in feats})
    ents, index = entity_features(game, player)
    for e in ents:
        state.append(state_dim)
        state.extend(sorted({_h(t, state_dim) for t in e}))
    opts = []
    for i, o in enumerate(game.legal_options()):
        toks = sorted({_h(t, option_dim) for t in option_tokens(d.kind, o.key) + option_preview(game, player, i)})
        toks += sorted({option_dim + index[i] for i in option_object_ids(o) if i in index})
        opts.append(toks)
    return state, opts


def featurize_flat(game, player: int, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM):
    """`featurize` with the options flattened: (state, option lengths, all
    option tokens). What rollouts record and send to the inference server."""
    if getattr(game, "NATIVE", False):
        return game.featurize_flat(player, state_dim, option_dim)
    state, opts = featurize(game, player, state_dim, option_dim)
    return state, [len(o) for o in opts], [t for o in opts for t in o]


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
    return encode_event_hashes([_h(t, option_dim) for t in tokens], option_dim)


def event_hashes(game, index: int, option_dim: int = OPTION_DIM) -> tuple[list[int], list[int]]:
    """Hashed `event_tokens` of option `index` of the current decision, as
    seen by (the decider, the opponent). Native games hash in Rust."""
    if getattr(game, "NATIVE", False):
        return game.event_hashes(index, option_dim)
    d = game.decision
    key = d.options[index].key
    return (
        [_h(t, option_dim) for t in event_tokens(d.kind, key, mine=True)],
        [_h(t, option_dim) for t in event_tokens(d.kind, key, mine=False)],
    )


def encode_event_hashes(hashes: list[int], option_dim: int = OPTION_DIM) -> list[int]:
    """`encode_events` for tokens that are already hashed."""
    n = len(hashes)
    return hashes[-MAX_EVENT_TOKENS:] + [_h(f"events:{min(n // 8, 16)}", option_dim)]
