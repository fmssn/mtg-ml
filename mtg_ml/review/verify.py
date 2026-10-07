"""Check "option B was better than the chosen A" claims by counterfactual rollouts.

The game is rebuilt from its recorded constructor arguments and actions up to
the decision, with the recorded agents replaying the prefix so recurrent
policies carry the memory they had in the real game. From there, for each of
`n` determinizations (everything the deciding seat could not see is
re-dealt, `view.determinize`, and future shuffles are re-seeded), both A and
B are played out by the same agents with the same random seeds (common random
numbers: the paired difference has far less variance than two independent
win rates). The verdict compares the mean paired difference in the deciding
seat's result (win 1, draw 0.5, loss 0) with its standard error.

Rollouts play the policy against itself, so "better" means better for this
policy's continuation, which is what matters for diagnosing it. Opponent
models keep the memory of the real game, which may mention cards the
determinization moved; a small inconsistency, the same for A and B.
"""

from __future__ import annotations

import copy
import math
import random

from ..agents import take
from ..backend import game_class
from ..engine.view import determinize
from ..match import game_args


def rebuild(rep: dict):
    """A fresh game with the review file's constructor arguments."""
    gm = rep["meta"]["review"]["game"]
    args = game_args(gm["match_game"], rep["meta"]["review"]["matchup"])
    return game_class(gm["engine"])(**args, seed=gm["seed"], log=False, starting_player=gm["starting_player"])


def _agents(rep: dict):
    from ..match import matchup_decks
    from ..play import make_agent

    rv = rep["meta"]["review"]
    decks = matchup_decks(rv["matchup"])
    return [make_agent(k, i, rv["game"]["seed"], greedy=rv["greedy"], deck=decks[i]) for i, k in enumerate(rv["specs"])]


def _clone(agent, game, seed: int):
    a = copy.copy(agent)
    if hasattr(a, "hidden"):  # ModelAgent: own memory and sampler, bound to `game`
        a.hidden = None if agent.hidden is None else agent.hidden.clone()
        a.events = list(agent.events)
        a.game = game
        a.gen = agent.torch.Generator().manual_seed(seed)
    elif hasattr(a, "rng"):
        a.rng = random.Random(seed)
    return a


def _playout(g, agents, max_decisions: int) -> int | None:
    n = 0
    while not g.over and n < max_decisions:
        take(g, agents, agents[g.decision.player].act(g))
        n += 1
    return g.winner if g.over else None


def prefixes(rep: dict, targets: list[int]):
    """Yield (decision index, game, agents) at each target decision, in order,
    replaying the recorded game once."""
    chosen = [f["decision"]["chosen"] for f in rep["frames"] if f["decision"]]
    g = rebuild(rep)
    agents = _agents(rep)
    want = sorted(set(targets))
    for i, a in enumerate(chosen):
        if not want:
            return
        d = rep["frames"][i]["decision"]
        if [o.label for o in g.decision.options] != d["options"]:
            raise RuntimeError(f"rebuilt game diverged from the record at d{i}")
        agents[g.decision.player].act(g)  # memory only; the recorded choice is played
        if i == want[0]:
            want.pop(0)
            yield i, g, agents
        take(g, agents, a)


def compare(g, agents, seat: int, a: int, b: int, n: int = 32, max_decisions: int = 4000, seed: int = 0) -> dict:
    diffs, res = [], {a: [], b: []}
    for k in range(n):
        s = seed * 100003 + k
        out = {}
        for opt in (a, b):
            h = determinize(g, seat, random.Random(s))
            h.rng = random.Random(s ^ 0x5EED)
            ags = [_clone(x, h, s * 2 + i) for i, x in enumerate(agents)]
            take(h, ags, opt)
            w = _playout(h, ags, max_decisions)
            out[opt] = 0.5 if w is None else float(w == seat)
            res[opt].append(out[opt])
        diffs.append(out[b] - out[a])
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n)
    if mean > 2 * se and mean >= 0.05:
        verdict = "confirmed"
    elif mean < -2 * se and mean <= -0.05:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {
        "chosen": a,
        "alternative": b,
        "n": n,
        "win_chosen": round(sum(res[a]) / n, 3),
        "win_alternative": round(sum(res[b]) / n, 3),
        "diff": round(mean, 3),
        "se": round(se, 3),
        "verdict": verdict,
    }


def verify(rep: dict, findings: list[dict], n: int = 32, max_decisions: int = 4000) -> list[dict]:
    """Adds `rollout` to every finding that names a better option."""
    if any(f["kind"] in ("mask", "blunder") for f in rep["meta"]["review"].get("faults", [])):
        # recorded choices index filtered option lists; the rebuilt game would diverge
        for f in findings:
            f["rollout"] = {"verdict": "skipped: game has seeded option faults"}
        return findings
    todo = {}
    for f in findings:
        d, b = f.get("decision"), f.get("better_option")
        if d is None or b is None or d >= len(rep["frames"]) or rep["frames"][d]["decision"] is None:
            continue
        dec = rep["frames"][d]["decision"]
        if not 0 <= b < len(dec["options"]) or b == dec["chosen"]:
            f["rollout"] = {"verdict": "invalid option"}
            continue
        todo.setdefault(d, []).append(f)
    for d, g, agents in prefixes(rep, list(todo)):
        dec = rep["frames"][d]["decision"]
        for f in todo[d]:
            f["rollout"] = compare(g, agents, dec["player"], dec["chosen"], f["better_option"], n=n, max_decisions=max_decisions, seed=d)
    return findings
