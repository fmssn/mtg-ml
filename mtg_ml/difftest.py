"""Differential testing: the Python reference engine vs the Rust port.

Both engines play the same `trace.Scenario` in lockstep. The agents choose
from the reference game; the chosen index is applied to both. Before every
step the harness compares

* the decision: player, kind, prompt, option labels and keys (in order);
* `observe()` for both players and `state_features()` for both players;
* `featurize()` for the deciding player in the latest feature set and in set 1
  (native fast path vs Python);
* a dump of the full hidden state (libraries in order, `known_to`, object
  ids, damage, counters, the stack, pending triggers, combat, RNG position);
* for scripted-bot seats, the bot's choice computed on each engine (the bots
  read the object model, which the native engine serves through proxies);

and after the game the outcome, the log and the complete RNG state. Every
`fork_every` steps it also checks `fork()` and `determinize()`.

The first mismatch is reported with the path to the differing value. The
minimizer then rewrites the action script (earlier choices replaced by
option 0 where the mismatch survives) and the reproducer is saved as JSON:

    python -m mtg_ml.difftest fuzz --games 5000 --jobs 8      # long fuzz
    python -m mtg_ml.difftest fuzz --games 500 --with-card "1:Vapor Snag:4"   # a new card
    python -m mtg_ml.difftest fuzz --games 1000 --auto-mana --auto-pass       # action decomposition
    python -m mtg_ml.difftest repro difftest-failure.json      # replay one
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field

from .trace import Scenario, make_agents, scenarios

# ---------------------------------------------------------------------------
# Canonical state dumps
# ---------------------------------------------------------------------------


def _card(c) -> tuple:
    return (
        c.uid, c.oid, c.name, c.defn.name, c.owner, c.controller, c.zone, c.is_token, c.transformed, c.tapped, c.damage,
        c.deathtouch_damage, c.counters, c.sick, c.attached_to, c.skip_untap,
        [(sorted(t.keywords), t.power, t.toughness) for t in c.temp], sorted(c.known_to),
    )  # fmt: skip


def _data(d: dict) -> list:
    out = []
    for k, v in sorted(d.items()):
        out.append((k, _card(v) if k in ("card", "sacrificed") else v))
    return out


def dump_state(g) -> dict:
    """The reference engine's state in the format of the native `dump()`."""
    if getattr(g, "NATIVE", False):
        return g.dump()
    return {
        "turn": g.turn,
        "active": g.active,
        "step": g.step_name,
        "lands_played": g.lands_played,
        "starting_player": g.starting_player,
        "mulligans_taken": list(g.mulligans_taken),
        "next_id": g._next_id,
        "over": g.over,
        "winner": g.winner,
        "end_reason": g.end_reason,
        "match_game": g.match_game,
        "players": [
            {
                "life": p.life,
                "library": [_card(c) for c in p.library],
                "hand": [_card(c) for c in p.hand],
                "graveyard": [_card(c) for c in p.graveyard],
                "exile": [_card(c) for c in p.exile],
                "pool": list(p.pool.items()),
                "drew_from_empty": p.drew_from_empty,
                "cards_drawn_this_turn": p.cards_drawn_this_turn,
            }
            for p in g.players
        ],
        "battlefield": [_card(c) for c in g.battlefield],
        "stack": [
            {
                "sid": it.sid,
                "kind": it.kind,
                "controller": it.controller,
                "name": it.name,
                "target_specs": [s.kind for s in it.target_specs],
                "targets": list(it.targets),
                "card": None if it.card is None else _card(it.card),
                "source": None if it.source is None else _card(it.source),
                "method": it.method,
                "cast_from": it.cast_from,
                "x": it.x,
                "data": _data(it.data),
            }
            for it in g.stack
        ],
        "pending": [(t.controller, _card(t.source), t.tdef.name, _data(t.data)) for t in g.pending],
        "attackers": list(g.attackers),
        "blocked": sorted(g.blocked),
        "blocks": list(g.blocks.items()),
        "rng_index": g.rng.getstate()[1][-1],
    }


def canonical(x):
    if isinstance(x, (list, tuple)):
        return [canonical(e) for e in x]
    if isinstance(x, dict):
        return {k: canonical(v) for k, v in x.items()}
    return x


def first_diff(a, b, path: str = "") -> str | None:
    """Path and values of the first difference between two canonical values."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'python' if k in a else 'native'}"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        if len(a) != len(b):
            return f"{path}: length {len(a)} (python) != {len(b)} (native); extra: {(a[len(b):] or b[len(a):])[:3]!r}"
        return None
    if a != b or type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)):
        return f"{path}: python={a!r} native={b!r}"
    return None


def decision_view(g) -> dict:
    d = g.decision
    if d is None:
        return {"decision": None}
    return {"player": d.player, "kind": d.kind, "prompt": d.prompt, "labels": [o.label for o in d.options], "keys": [o.key for o in d.options]}


def snapshot(g, full: bool = True) -> dict:
    """Everything compared at one decision point."""
    from .encode import state_features
    from .engine.view import observe
    from .rl.features import event_hashes, featurize

    s = {"decision": decision_view(g), "observe0": observe(g, 0), "observe1": observe(g, 1)}
    if full:
        s["features0"] = state_features(g, 0)
        s["features1"] = state_features(g, 1)
        if g.decision is not None:
            s["featurize"] = featurize(g, g.decision.player)
            s["featurize_v1"] = featurize(g, g.decision.player, features=1)
            s["events"] = [event_hashes(g, i) for i in range(len(g.decision.options))]
        s["state"] = dump_state(g)
    return canonical(s)


def outcome(g) -> dict:
    return canonical({"winner": g.winner, "end_reason": g.end_reason, "turn": g.turn, "actions": list(g.actions), "log": list(g.log), "rng": g.rng.getstate()})


# ---------------------------------------------------------------------------
# Lockstep runs
# ---------------------------------------------------------------------------


@dataclass
class Divergence:
    scenario: Scenario
    step: int  # number of actions taken before the mismatch
    diff: str
    actions: list[int] = field(default_factory=list)  # the action script that reproduces it
    fork_every: int = 0  # the fork()/determinize() check interval it was found with (needed to reproduce it)

    def to_json(self) -> dict:
        return {"scenario": self.scenario.to_json(), "step": self.step, "diff": self.diff, "actions": self.actions, "fork_every": self.fork_every}

    @classmethod
    def from_json(cls, d: dict) -> Divergence:
        return cls(Scenario.from_json(d["scenario"]), d["step"], d["diff"], d["actions"], d.get("fork_every", 0))

    def __str__(self) -> str:
        return f"seed {self.scenario.seed} {self.scenario.agents} game {self.scenario.match_game}: step {self.step}: {self.diff}"


def _new_pair(sc: Scenario):
    from .trace import new_game

    return new_game(sc, "python"), new_game(sc, "native")


def _check(py, nat, full: bool) -> str | None:
    return first_diff(snapshot(py, full), snapshot(nat, full))


def run_lockstep(sc: Scenario, script: list[int] | None = None, full_every: int = 1, fork_every: int = 0) -> Divergence | None:
    """Play `sc` in both engines. With `script`, play those actions (clamped
    to the option count) instead of the agents, then stop; scripted-bot
    seats still compute their choice on both engines (compared, not played),
    so a bot-choice divergence reproduces from its script.

    A step that raises the same error in both engines is reported too (diff
    starting with "both raised"): the port agrees, but the reference engine
    has a bug."""
    from .engine.game import RulesError

    def div(n: int, diff: str) -> Divergence:
        return Divergence(sc, n, diff, list(taken), fork_every)

    taken: list[int] = []
    try:
        py, nat = _new_pair(sc)
    except RulesError as e:
        return div(0, f"construction failed: {e}")
    agents = make_agents(sc)
    n = 0
    while True:
        diff = _check(py, nat, full=full_every > 0 and n % full_every == 0)
        if diff:
            return div(n, diff)
        if fork_every and n % fork_every == fork_every - 1 and not py.over:
            diff = _check_fork(py, nat, sc.seed * 31 + n)
            if diff:
                return div(n, diff)
        if py.over:
            break
        seat = py.decision.player
        a = None
        if script is None or sc.agents[seat] == "bot":
            a = agents[seat].act(py)
            if sc.agents[seat] == "bot":  # bots read the object model: same choice on the native proxies
                b = agents[seat].act(nat)
                if a != b:
                    return div(n, f"bot choice: python={a} native={b}")
        if script is not None:
            if n >= len(script):
                return None
            a = min(script[n], len(py.legal_options()) - 1)
        taken.append(a)
        errs = []
        for g in (py, nat):
            try:
                g.step(a)
                errs.append(None)
            except Exception as e:  # noqa: BLE001 - reported as a divergence
                errs.append(f"{type(e).__name__}: {e}")
        n += 1
        if errs[0] or errs[1]:
            if errs[0] != errs[1]:
                return div(n, f"step raised: python={errs[0]} native={errs[1]}")
            return div(n, f"both raised (reference-engine bug, the port agrees): {errs[0]}")
    diff = first_diff(outcome(py), outcome(nat))
    return div(n, "outcome" + diff) if diff else None


def _check_fork(py, nat, seed: int) -> str | None:
    from .engine.view import determinize

    d = first_diff(snapshot(py.fork()), snapshot(nat.fork()))
    if d:
        return "fork" + d
    for viewer in (0, 1):
        a = determinize(py, viewer, random.Random(seed + viewer))
        b = determinize(nat, viewer, random.Random(seed + viewer))
        d = first_diff(snapshot(a), snapshot(b))
        if d:
            return f"determinize({viewer})" + d
        if not a.over:  # the re-dealt game must keep playing identically
            a.step(0)
            b.step(0)
            d = first_diff(snapshot(a), snapshot(b))
            if d:
                return f"determinize({viewer})+step" + d
    return None


def minimize(div: Divergence, budget: int = 200) -> Divergence:
    """Greedy: replace choices by option 0 (earliest first) while a mismatch
    still occurs, so the reproducer takes as few non-default choices as
    possible. The returned divergence is the first one of the final script."""
    script = list(div.actions)
    best = div
    tries = 0
    for i in range(len(script)):
        if tries >= budget:
            break
        if script[i] == 0:
            continue
        cand = script[:i] + [0] + script[i + 1 :]
        tries += 1
        d = run_lockstep(div.scenario, script=cand, fork_every=div.fork_every)
        if d is not None:
            script = d.actions + cand[len(d.actions) :]
            best = d
    trimmed = run_lockstep(best.scenario, script=best.actions, fork_every=div.fork_every)
    return trimmed or best


def _run_one(args) -> dict | None:
    sc, full_every, fork_every = args
    d = run_lockstep(sc, full_every=full_every, fork_every=fork_every)
    return None if d is None else d.to_json()


def fuzz(
    n: int,
    start: int = 0,
    jobs: int = 1,
    full_every: int = 1,
    fork_every: int = 97,
    out: str = "difftest-failure.json",
    quiet: bool = False,
    extra: tuple = (),
    auto_mana: bool = False,
    auto_pass: bool = False,
) -> int:
    scs = scenarios(n, start, extra, auto_mana, auto_pass)
    t = time.perf_counter()
    work = [(sc, full_every, fork_every) for sc in scs]
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(jobs) as pool:
            results = pool.imap(_run_one, work, chunksize=4)
            failures = _collect(results, len(work), quiet)
    else:
        failures = _collect(map(_run_one, work), len(work), quiet)
    dt = time.perf_counter() - t
    print(f"{n - len(failures)}/{n} games identical in both engines ({dt:.0f}s)")
    if failures:
        first = min(failures, key=lambda f: f["scenario"]["seed"])
        div = Divergence.from_json(first)
        print(f"{len(failures)} divergent games; first: {div}")
        small = minimize(div)
        print(f"minimized: step {small.step} with {sum(1 for a in small.actions if a)} non-default choices: {small.diff}")
        with open(out, "w") as f:
            json.dump(small.to_json(), f, indent=1)
        print(f"reproducer written to {out}  (python -m mtg_ml.difftest repro {out})")
    return len(failures)


def _collect(results, total: int, quiet: bool) -> list[dict]:
    failures = []
    for i, r in enumerate(results, 1):
        if r is not None:
            failures.append(r)
            if not quiet:
                print("DIVERGENCE:", Divergence.from_json(r), flush=True)
        if not quiet and i % 250 == 0:
            print(f"  {i}/{total} games, {len(failures)} divergent", flush=True)
    return failures


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.difftest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fuzz", help="play many scenarios in both engines")
    f.add_argument("--games", type=int, default=1000)
    f.add_argument("--start", type=int, default=0, help="first scenario seed")
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--full-every", type=int, default=1, help="compare features/featurize/full state every N steps (decisions and observe() always)")
    f.add_argument("--fork-every", type=int, default=97, help="check fork() and determinize() every N steps (0 = never)")
    f.add_argument("--out", default="difftest-failure.json")
    f.add_argument("--quiet", action="store_true")
    f.add_argument(
        "--with-card",
        action="append",
        default=[],
        metavar="SEAT:NAME:N",
        help="swap N copies of a card into seat SEAT's deck (0 = Jund, 1 = Blue), e.g. '1:Vapor Snag:4'; repeatable",
    )
    f.add_argument("--auto-mana", action="store_true", help="play with Game(auto_mana=True): colour-preserving auto payment")
    f.add_argument("--auto-pass", action="store_true", help="play with Game(auto_pass=True): collapse uneventful priority passes")
    r = sub.add_parser("repro", help="replay a saved divergence")
    r.add_argument("file")
    args = ap.parse_args(argv)
    if args.cmd == "fuzz":
        extra = []
        for spec in args.with_card:
            seat, name, n = spec.split(":")
            extra.append((int(seat), name, int(n)))
        sys.exit(1 if fuzz(args.games, args.start, args.jobs, args.full_every, args.fork_every, args.out, args.quiet, tuple(extra), args.auto_mana, args.auto_pass) else 0)
    with open(args.file) as fh:
        data = json.load(fh)
    saved = Divergence.from_json(data)
    d = run_lockstep(saved.scenario, script=saved.actions, fork_every=saved.fork_every)
    print("reproduced:" if d else "no longer diverges", d or "")
    sys.exit(1 if d else 0)


if __name__ == "__main__":
    main()
