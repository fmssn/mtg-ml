"""Bounded development games. No candidate/puzzle campaign CLI."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import time

from ..backend import game_class
from ..engine.view import observe
from ..replay import FORMAT, snapshot, visible_events
from ..rl.features import PUBLIC_KINDS
from .adapters import take
from .artifacts import integer, require
from .views import inputs, thaw


def play_episode(spec, settings, learner_factory, opponent_factory, engine="python", record=False):
    """Factories receive (physical seat, mode), and create isolated adapters."""
    integer(settings["max_decisions"], "max_decisions")
    row = {k: v for k, v in asdict(spec).items() if k != "decks"}
    row.update(status="error", winner=None, reason=None, decisions=0, turns=0, latency_seconds=[], attempted_action=None)
    g, frames, card_info = None, [], {}
    started = time.perf_counter()
    try:
        g = game_class(engine)(**spec.game_args(settings))
        agents = [None, None]
        agents[spec.learner_seat] = learner_factory(spec.learner_seat, spec.mode)
        agents[1 - spec.learner_seat] = opponent_factory(1 - spec.learner_seat, spec.mode)
        from collections import Counter
        for seat, a in enumerate(agents):
            a.reset(dict(Counter(spec.decks[seat])), spec.actor_seed)
        seen = len(g.log)
        while not g.over:
            if row["decisions"] >= settings["max_decisions"]:
                raise RuntimeError("runner_decision_cap")
            decider, d = g.decision.player, g.decision
            tick = time.perf_counter()
            index = agents[decider].act(g)
            elapsed = time.perf_counter() - tick
            if decider == spec.learner_seat:
                row["latency_seconds"].append(elapsed)
            row["attempted_action"] = {"player": decider, "kind": d.kind, "index": index if type(index) is int else repr(index)}
            if type(index) is int and 0 <= index < len(d.options):
                _, legal = inputs(g, decider, dict(Counter(spec.decks[decider])))
                row["attempted_action"]["semantic"] = thaw(asdict_action(legal[index])) if decider == spec.learner_seat or d.kind in PUBLIC_KINDS else None
            if record:
                decision = None
                if decider == spec.learner_seat:
                    decision = {"player": decider, "kind": d.kind, "prompt": d.prompt,
                                "options": [o.label for o in d.options], "chosen": index if type(index) is int else None}
                elif type(index) is int and 0 <= index < len(d.options):
                    decision = {"player": decider, "kind": d.kind, "prompt": "", "options": [d.options[index].label] if d.kind in PUBLIC_KINDS else [d.kind], "chosen": 0}
                frames.append({"state": snapshot(g, card_info, viewer=spec.learner_seat),
                               "events": visible_events(g.log[seen:], spec.learner_seat), "decision": decision})
            seen = len(g.log)
            take(g, agents, index)
            row["decisions"] = len(g.actions)
        row.update(status="completed", winner=g.winner, reason=g.end_reason)
    except Exception as e:
        row.update(status="error", reason=str(e), error_type=type(e).__name__)
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
        if g is not None:
            row["turns"], row["decisions"] = g.turn, len(g.actions)
            row["visible_outcome"] = observe(g, spec.learner_seat)
            if record:
                frames.append({"state": snapshot(g, card_info, viewer=spec.learner_seat), "events": visible_events(g.log[seen:], spec.learner_seat), "decision": None})
                row["replay"] = {"format": FORMAT, "meta": {"engine": engine, "winner": g.winner, "turns": g.turn,
                                                            "seed": spec.simulator_seed, "viewer": spec.learner_seat,
                                                            "agents": ["learner" if s == spec.learner_seat else spec.opponent for s in (0, 1)],
                                                            "end_reason": row["reason"]}, "cards": card_info, "frames": frames}
    return row


def asdict_action(action):
    return {"index": action.index, "kind": action.kind, "key": action.key, "label": action.label, "data": action.data}


def run_episodes(specs, settings, learner_factory, opponent_factory, engine="python", workers=1, record=False):
    integer(workers, "workers")
    specs = list(specs)
    require(len({s.identity for s in specs}) == len(specs), "episodes", "duplicate identity")
    def run(spec):
        return play_episode(spec, settings, learner_factory, opponent_factory, engine, record)
    if workers == 1:
        rows = list(map(run, specs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run, specs))
    return sorted(rows, key=lambda r: (r["mode"], r["cell"], r["block"], r["slot"]))
