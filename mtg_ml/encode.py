"""Sparse hashed state features and stable action keys.

Inspired by MageZero's StateEncoder: a state is a set of string features
(hierarchical: a permanent contributes its name, its name+state and coarse
type-level features), hashed into a fixed index space. Everything here works
from `observe()`, so encoders never see hidden information.
"""

from __future__ import annotations

import zlib

from .engine.view import observe

DEFAULT_DIM = 1 << 16


def _bucket(n: int, edges=(0, 1, 2, 3, 5, 8, 13, 20)) -> int:
    b = 0
    for i, e in enumerate(edges):
        if n >= e:
            b = i
    return b


def state_features(game, viewer: int) -> list[str]:
    if getattr(game, "NATIVE", False):
        return game.state_features(viewer)
    o = observe(game, viewer)
    f = [f"step:{o['step']}", f"active:{o['active']}", f"postboard:{o['match_game'] > 1}"]
    for side in ("self", "opponent"):
        s = o[side]
        f.append(f"{side}:life:{_bucket(s['life'])}")
        f.append(f"{side}:library:{_bucket(s['library_count'])}")
        f.append(f"{side}:mulligans:{s['mulligans']}")
        for name in s["graveyard"]:
            f.append(f"{side}:gy:{name}")
        for c, n in s["pool"].items():
            f.append(f"{side}:pool:{c}:{n}")
        for _, name in s["library_known"][:3]:
            f.append(f"{side}:known_library:{name}")
    for name in o["self"]["hand"]:
        f.append(f"self:hand:{name}")
    f.append(f"opponent:hand_count:{_bucket(o['opponent']['hand_count'])}")
    for name in o["opponent"]["hand_known"]:
        f.append(f"opponent:hand_known:{name}")
    for p in o["battlefield"]:
        side = p["controller"]
        base = f"{side}:bf:{p['name']}"
        f.append(base)
        for t in p["types"]:
            f.append(f"{side}:bf_type:{t}")
        if p["tapped"]:
            f.append(base + ":tapped")
        if p["sick"]:
            f.append(base + ":sick")
        if p["attacking"]:
            f.append(base + ":attacking")
        if p["blocking"] is not None:
            f.append(base + ":blocking")
        if p["power"] is not None:
            f.append(f"{base}:pt:{p['power']}/{p['toughness']}")
        if p["counters"]:
            f.append(f"{base}:counters:{p['counters']}")
    for i, it in enumerate(reversed(o["stack"])):
        f.append(f"stack:{min(i, 3)}:{it['controller']}:{it['name']}")
    return f


def hash_feature(feature: str, dim: int = DEFAULT_DIM) -> int:
    return zlib.crc32(feature.encode()) % dim


def encode_state(game, viewer: int, dim: int = DEFAULT_DIM) -> list[int]:
    """Sorted unique feature indices (a multi-hot sparse vector)."""
    return sorted({hash_feature(x, dim) for x in state_features(game, viewer)})


def action_keys(game) -> list[tuple]:
    """Stable, id-free keys of the current options (for policy vocabularies)."""
    return [(game.decision.kind,) + o.key for o in game.legal_options()]
