"""Record games for review: a replay file plus what the reviewer and `verify` need.

`meta["review"]` adds to the replay format (mtg_ml.replay): the exact game
constructor arguments and agent specs (so `verify` can rebuild and fork the
game at any decision), the decklists, each model's training state and a plain
description of the training reward, each model's structured configuration
(`policy_configs`), and the faults seeded into the game, if any (`faults`).
"""

from __future__ import annotations

import pathlib

from ..match import DEFAULT_MATCHUP, match_decks
from ..replay import _agent_name, record

REWARD = (
    "terminal +1 win / -1 loss / 0 draw, discounted; plus potential-based life shaping "
    "0.2 * (own life - opponent life) / 20, linearly annealed to 0 over the first 100 "
    "iterations of each training run (later iterations train on the terminal reward only)"
)


def _policy_metadata(spec: str) -> tuple[str | None, dict | None]:
    """Human description and structured config from the same checkpoint read."""
    if not spec.startswith("model:"):
        return None, None
    import torch

    path = spec[len("model:"):]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    parts = [f"checkpoint {_agent_name(spec)[6:]}"]
    if "iteration" in ck:
        parts.append(f"iteration {ck['iteration']} of its run")
    if "games_total" in ck:
        parts.append(f"{ck['games_total']:,} training games in total")
    arch = ", ".join(f"{k} {cfg[k]}" for k in ("trunk", "hidden", "memory", "features", "entity_attn", "value_net") if k in cfg)
    if arch:
        parts.append(arch)
    return "; ".join(parts), cfg


def record_game(
    specs: list[str],
    seed: int,
    matchup: str = DEFAULT_MATCHUP,
    match_game: int = 1,
    starting_player: int | None = None,
    engine: str | None = None,
    greedy: bool = False,
    wrap=None,
    reward: str = REWARD,
) -> dict:
    """Play one game and return the review file dict. `wrap(seat, agent)` may
    replace an agent (fault injection)."""
    from ..match import matchup_decks
    from ..play import make_agent

    decks = matchup_decks(matchup)
    agents = [make_agent(k, i, seed, greedy=greedy, deck=decks[i]) for i, k in enumerate(specs)]
    if wrap is not None:
        agents = [wrap(i, a) for i, a in enumerate(agents)]
    names = [_agent_name(k) for k in specs]
    rep = record(agents, seed=seed, engine=engine, names=names, matchup=matchup, match_game=match_game, starting_player=starting_player)
    policies = [_policy_metadata(k) for k in specs]
    rep["meta"]["review"] = {
        "specs": list(specs),
        "greedy": greedy,
        "matchup": matchup,
        "game": {"seed": seed, "match_game": match_game, "starting_player": starting_player, "engine": engine},
        "decklists": [list(d) for d in match_decks(match_game, matchup)],
        "policies": [description for description, _ in policies],
        "policy_configs": [config for _, config in policies],
        "reward": reward if any(k.startswith("model:") for k in specs) else None,
        "faults": [],
    }
    return rep


def game_path(out: pathlib.Path, rep: dict, tag: str = "") -> pathlib.Path:
    m = rep["meta"]
    rv = m["review"]
    names = "-".join(n.replace(":", "_") for n in m["agents"])
    return out / f"{rv['matchup']}-g{m['match_game']}-{names}-s{m['seed']}{tag}.json"
