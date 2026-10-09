"""Agent-driven tactical puzzle attempts, scored by the resulting outcome.

The evaluated agent chooses every decision (targets, payments, order, passes)
from the compiled position; the case's response policy plays the other seat.
Witness and mistake lines run through the same attempt path, so the runner and
the reviewed corpus are checked by one classifier.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import gc
import multiprocessing
import pickle
import random

from ..engine.view import observe
from ..expert.scenarios import canonical, compile_scenario, select
from .adapters import AgentRegistry, LegacyAdapter, ScriptedAdapter, take
from .artifacts import identifier, load, load_puzzle, reference, require, validate_bundle
from .jsonio import digest, read_json
from .runner import _worker_init
from .schedule import puzzle_plan

SUCCESS, FAILURE, ERROR = "success", "failure", "error"
STRATEGIC = {"objective_met", "objective_failed", "horizon_exhausted"}
TECHNICAL = {"illegal_action", "agent_exception", "response_policy_error", "engine_error"}
EXPECTABLE = {"objective_failed", "horizon_exhausted"}
SPECIALISTS = ("benchmark-jund@1", "benchmark-blue@1")


class LearnerError(RuntimeError):
    pass


class ResponseError(RuntimeError):
    pass


def specialist_registry(source_revision, registry=None):
    """Both frozen specialists, registered at the caller's verified revision."""
    from .blue import register_blue
    from .bots.jund import register_jund
    registry = AgentRegistry() if registry is None else registry
    register_jund(registry, source_revision)
    register_blue(registry, source_revision)
    return registry


def validate_lines(case, field):
    """Optional learner-only witnesses and expected-failure lines (tactical v1)."""
    for w in case.get("witnesses", []):
        require(type(w.get("learner_only", False)) is bool, field + ".witnesses.learner_only", "boolean required")
    mistakes = case.get("mistakes", [])
    require(isinstance(mistakes, list), field + ".mistakes", "list required")
    for m in mistakes:
        require(isinstance(m, dict) and isinstance(m.get("actions"), list) and bool(m["actions"]), field + ".mistakes", "action selectors required")
        identifier(m.get("note"), field + ".mistakes.note")
        require(m.get("expect") in EXPECTABLE, field + ".mistakes.expect", "objective_failed or horizon_exhausted required")
        require(m.get("learner_only", True) is True, field + ".mistakes.learner_only", "mistakes play against the response policy")


def default_choice(g):
    """Pass priority / finish declarations once a scripted line is exhausted."""
    for i, o in enumerate(g.legal_options()):
        if tuple(o.key) == ("pass",) or (g.decision.kind in {"declare_attacker", "declare_blocker"} and o.key[-1] is None):
            return i
    raise LearnerError(f"scripted line exhausted at a {g.decision.kind} decision")


class SelectorAgent:
    """Trusted validation learner: replays reviewed selectors, then passes."""

    def __init__(self, actions, seat):
        self.actions, self.seat = list(actions), seat

    def bind(self, objects):
        self.objects = objects

    def reset(self, own_deck, actor_seed=0):
        self.position = 0

    def act(self, game):
        if self.position < len(self.actions):
            selector = self.actions[self.position]
            self.position += 1
            try:
                return select(game, selector, self.objects)
            except ValueError as e:
                raise LearnerError(f"reviewed line step {self.position}: {e}") from e
        return default_choice(game)


@dataclass(frozen=True)
class SelectorFactory:
    actions: tuple

    def __call__(self, seat, mode):
        return SelectorAgent(self.actions, seat)


class RandomAgent:
    """Seeded uniformly random legal learner, for response-coverage smoke."""

    def __init__(self, seed, seat):
        self.seed, self.seat = seed, seat

    def reset(self, own_deck, actor_seed=0):
        self.rng = random.Random(self.seed)

    def act(self, game):
        return self.rng.randrange(len(game.legal_options()))


@dataclass(frozen=True)
class RandomFactory:
    seed: int

    def __call__(self, seat, mode):
        return RandomAgent(self.seed, seat)


class PassAgent:
    def reset(self, own_deck, actor_seed=0):
        pass

    def act(self, game):
        return default_choice(game)


def pass_learner(seat, mode):
    return PassAgent()


@dataclass(frozen=True)
class ResponseFactory:
    """Resolves a pinned response policy from the registry for its seat."""
    registry: AgentRegistry
    name: str

    def __call__(self, seat, mode):
        agent = self.registry.create(self.name)
        if self.registry.metadata(self.name).kind == "legacy":
            return LegacyAdapter(lambda _seat: agent, seat)
        return ScriptedAdapter(agent, seat)


def load_case(puzzle, case):
    scenario = read_json(reference(puzzle.path, case["scenario"], "cases.scenario"))
    evidence = read_json(reference(puzzle.path, case["evidence"], "cases.evidence"))
    require(scenario.get("group_id") == puzzle.data["source_group_id"], "puzzle.source_group_id", "scenario source-group mismatch")
    return scenario, evidence


def history_free(g, seat):
    """Reset mode feeds no earlier events, so the start must not rely on reveals."""
    obs = observe(g, seat)
    revealed = obs["opponent"]["hand_known"] or obs["opponent"]["library_known"] or obs["self"]["library_known"]
    require(not revealed, "case.history", "initial position depends on earlier reveals (missing history)")


def acceptable_indices(g, selectors, objects):
    out = set()
    for s in selectors:
        try:
            out.add(select(g, s, objects))
        except ValueError:
            continue  # selector for another decision kind, or not offered here
    return out


def attempt_case(puzzle, case, decks, learner_factory, response_factory, *, mode, actor_seed=0, engine="python",
                 max_decisions=None):
    """One reset-mode attempt; returns status/reason without plan identity keys."""
    from .validation import boundary, objective
    d = puzzle.data
    cap = d["max_decisions"] if max_decisions is None else max_decisions
    row = dict(status=ERROR, reason="engine_error", objective_met=False, first_action_correct=None, decisions=0)
    g, seat, trace, chosen_first = None, None, [], False
    try:
        scenario, evidence = load_case(puzzle, case)
        g, objects = compile_scenario(scenario, evidence, engine)
        seat = scenario["perspective"]
        history_free(g, seat)
        response = case["response_policy"]["id"]
        response_deck = decks[response_factory.registry.metadata(response).deck]["cards"] if isinstance(response_factory, ResponseFactory) else {}
        try:
            learner = learner_factory(seat, mode)
            if hasattr(learner, "bind"):
                learner.bind(objects)
            learner.reset(dict(decks[d["deck"]]["cards"]), actor_seed)
        except Exception as e:
            raise LearnerError(f"{type(e).__name__}: {e}") from e
        try:
            opponent = response_factory(1 - seat, mode)
            opponent.reset(dict(response_deck), actor_seed)
        except Exception as e:
            raise ResponseError(f"{type(e).__name__}: {e}") from e
        agents = [None, None]
        agents[seat], agents[1 - seat] = learner, opponent
        start = len(g.actions)
        while True:
            decider = g.decision.player
            labeled = None
            if decider == seat and not chosen_first:
                chosen_first = True
                if d.get("acceptable_first_actions"):
                    labeled = acceptable_indices(g, d["acceptable_first_actions"], objects)
            try:
                index = agents[decider].act(g)
            except Exception as e:
                raise (LearnerError if decider == seat else ResponseError)(f"{type(e).__name__}: {e}") from e
            if labeled is not None:
                row["first_action_correct"] = index in labeled
            if not (type(index) is int and 0 <= index < len(g.legal_options())):
                if decider == seat:
                    row.update(status=ERROR, reason="illegal_action", error=f"index {index!r}")
                    return row
                raise ResponseError(f"illegal response index {index!r}")
            # Semantic keys omit object ids, so traces compare across engines.
            trace.append([decider, g.decision.kind, canonical(g.legal_options()[index].key)])
            take(g, agents, index)
            row["decisions"] = len(g.actions) - start
            if boundary(g, seat, d["stop"]):
                met = objective(g, seat, d["objective"]["all"], objects)
                row.update(status=SUCCESS if met else FAILURE, reason="objective_met" if met else "objective_failed", objective_met=met)
                break
            if row["decisions"] >= cap:
                row.update(status=FAILURE, reason="horizon_exhausted")
                break
    except LearnerError as e:
        row.update(status=ERROR, reason="agent_exception", error=str(e))
    except ResponseError as e:
        row.update(status=ERROR, reason="response_policy_error", error=str(e))
    except Exception as e:
        row.update(status=ERROR, reason="engine_error", error=f"{type(e).__name__}: {e}")
    finally:
        if g is not None:
            row["visible_outcome"] = observe(g, seat) if seat is not None else None
            row["trace_sha256"] = digest({"trace": trace, "outcome": row["visible_outcome"]})
            if getattr(g, "NATIVE", False):
                g._cache.clear()
                g._proxies.clear()
    return row


def _plan_puzzles(manifest):
    bundle = load(reference(manifest.path, manifest.data["puzzles"], "puzzles"), validate_bundle)
    puzzles = {}
    for ref in bundle.data["puzzles"]:
        p = load_puzzle(reference(bundle.path, ref, "bundle.puzzles"))
        for case in p.data["cases"]:
            validate_lines(case, "cases")
        puzzles[p.data["id"]] = p
    return puzzles


def _attempt_planned(plan, puzzles, decks, learner_factory, registry, engine):
    p = puzzles[plan["puzzle"]]
    case = next(c for c in p.data["cases"] if c["id"] == plan["case"])
    row = attempt_case(p, case, decks, learner_factory, ResponseFactory(registry, case["response_policy"]["id"]),
                       mode=plan["mode"], actor_seed=plan["actor_seed"], engine=engine)
    return dict(plan) | row


def run_puzzles(manifest, learner_factory, registry, *, engine="python", cells=None, modes=None, workers=1):
    """Rows for every planned case, ready for results.build_result(puzzle_rows=...)."""
    puzzles = _plan_puzzles(manifest)
    bots = {b["id"]: b for b in manifest.data["bots"]}
    for p in puzzles.values():
        for case in p.data["cases"]:
            rp = case["response_policy"]
            require(rp["id"] in bots and bots[rp["id"]]["parameters_sha256"] == rp["parameters_sha256"], "response_policy", "freeze mismatch")
            registry.verify(bots[rp["id"]], manifest.data["information_contract"])
    plans = puzzle_plan(manifest, cells, modes)
    run = partial(_attempt_planned, puzzles=puzzles, decks=manifest.data["decks"], learner_factory=learner_factory,
                  registry=registry, engine=engine)
    if workers == 1:
        return list(map(run, plans))
    try:
        pickle.dumps((learner_factory, registry))
    except (TypeError, AttributeError, pickle.PicklingError) as e:
        raise ValueError("workers: factories and registry must be importable/picklable") from e
    gc.collect()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
        return list(pool.map(run, plans))


def verify_lines(puzzle, case, decks, registry, engines, robustness=4):
    """Learner-only witnesses succeed, mistakes fail as declared, and the response
    policy handles a pass learner and seeded random learners, on every engine."""
    validate_lines(case, "cases")
    response = ResponseFactory(registry, case["response_policy"]["id"])
    lines = [("witness", w, "objective_met") for w in case.get("witnesses", []) if w.get("learner_only")]
    lines += [("mistake", m, m["expect"]) for m in case.get("mistakes", [])]
    probes = [("pass", pass_learner)] + [(f"random-{s}", RandomFactory(s)) for s in range(robustness)]
    out = {"lines": [], "probes": []}
    for kind, line, expect in lines:
        rows = [attempt_case(puzzle, case, decks, SelectorFactory(tuple(line["actions"])), response, mode="greedy", engine=e) for e in engines]
        for r in rows:
            require(r["reason"] == expect, f"{kind}", f"{line['note']!r} classified {r['reason']} ({r.get('error', '')}), expected {expect}")
        _same(rows, kind)
        out["lines"].append({"kind": kind, "reason": expect, "decisions": rows[0]["decisions"], "trace_sha256": rows[0]["trace_sha256"]})
    for name, factory in probes:
        rows = [attempt_case(puzzle, case, decks, factory, response, mode="greedy", engine=e) for e in engines]
        for r in rows:
            require(r["status"] != ERROR, "response_policy", f"{name} learner produced {r['reason']}: {r.get('error', '')}")
        _same(rows, name)
        out["probes"].append({"learner": name, "reason": rows[0]["reason"], "decisions": rows[0]["decisions"]})
    return out


def _same(rows, what):
    keys = ("status", "reason", "decisions", "trace_sha256")
    require(all({k: r[k] for k in keys} == {k: rows[0][k] for k in keys} for r in rows), "parity", f"{what}: engines diverged")


def summarize(rows):
    """Descriptive development summary: no intervals, errors kept separate."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        mrows = [r for r in rows if r["mode"] == mode]
        errors = [r for r in mrows if r["status"] == ERROR]
        by_puzzle = {}
        for r in mrows:
            by_puzzle.setdefault((r["puzzle"], r["repetition"]), []).append(r)
        # A puzzle succeeds only if every declared case succeeds; any error leaves it unscored.
        scored = {k: all(r["status"] == SUCCESS for r in v) for k, v in by_puzzle.items() if all(r["status"] != ERROR for r in v)}
        categories, totals = Counter(), Counter()
        for key, ok in scored.items():
            category = by_puzzle[key][0]["category"]
            totals[category] += 1
            categories[category] += ok
        firsts = [r["first_action_correct"] for r in mrows if r.get("first_action_correct") is not None and r["status"] != ERROR]
        out[mode] = {"puzzles": len(by_puzzle), "scored_puzzles": len(scored), "technical_case_errors": len(errors),
                     "puzzle_success": sum(scored.values()) / len(scored) if scored else None,
                     "categories": {c: {"puzzles": totals[c], "success": categories[c] / totals[c]} for c in sorted(totals)},
                     "first_action_correct": sum(firsts) / len(firsts) if firsts else None,
                     "reasons": dict(Counter(r["reason"] for r in mrows))}
    return out
