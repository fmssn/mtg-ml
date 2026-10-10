"""Indistinguishability audit: find observation blind spots mechanically.

    python -m mtg_ml.audit --games 50 --engine native                       # scripted bots, jund_blue
    python -m mtg_ml.audit --agents model:runs/x/final.pt,bot --games 20    # a checkpoint (its feature set)
    python -m mtg_ml.audit --features 4 --sample-frac 0.2 --out audit.json  # another feature set, sampled

At a decision the policy scores every option from its featurization
(`rl.features.featurize`): the shared state and entity bags, plus the
option's hashed tokens and its pointers to entities. Two options whose
featurizations are identical (same hashed tokens, pointers to entities with
identical token bags) get the same score: the policy cannot tell them
apart. That is harmless when they lead to the same game, and a **blind
spot** when they do not.

For each audited decision this module groups the options by that
featurization and, for every group of two or more, forks the game, steps
each option and compares the successor states by `observable_digest`: what
the decider can observe afterwards (public zones, own hand, life, mana,
counters, the stack, combat assignments, known library cards, the next
decision if it is theirs), with object ids replaced by a canonical labelling
so that two interchangeable Forests count as the same state. Successors that
agree are genuinely equivalent; successors that differ make a blind spot.
Options the engine already merged (`Game.equiv_key`) never reach the audit,
and differences in hidden information (the opponent's hand, library order)
do not count. Reference: docs/audit.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field

from .agents import take
from .backend import engine_name, game_class
from .encode import FEATURE_VERSIONS, FEATURES, check_features, option_object_ids
from .engine.view import observe
from .match import DEFAULT_MATCHUP, MATCHUPS, game_args, matchup_decks
from .rl.features import OPTION_DIM, STATE_DIM, featurize

FORMAT = 1  # version of the JSON report
_ID = re.compile(r"#\d+")


# ---------------------------------------------------------------------------
# Featurization grouping
# ---------------------------------------------------------------------------


def option_signatures(game, player: int, features: int = FEATURES, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM) -> list[tuple]:
    """One hashable signature per option of the current decision: exactly
    what the policy sees of it in feature set `features`. The option's
    hashed tokens plus, for every entity pointer (`option_dim + k`), the
    token bag of entity k, so two options pointing at two entities with the
    same features get the same signature. The state is shared by all
    options of a decision and not part of the signature."""
    state, opts = featurize(game, player, state_dim, option_dim, check_features(features))
    ents: list[tuple[int, ...]] = []
    for t in state:
        if t == state_dim:
            ents.append(())
        elif ents:
            ents[-1] += (t,)
    sigs = []
    for toks in opts:
        hashed = tuple(t for t in toks if t < option_dim)
        pointed = tuple(sorted(ents[t - option_dim] for t in toks if t >= option_dim))
        sigs.append((hashed, pointed))
    return sigs


def colliding_groups(sigs: list[tuple]) -> list[list[int]]:
    """Option indices sharing a signature, groups of 2+ only, in option order."""
    by: dict[tuple, list[int]] = {}
    for i, s in enumerate(sigs):
        by.setdefault(s, []).append(i)
    return [g for g in by.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Observable digest
# ---------------------------------------------------------------------------

WL_ROUNDS = 3  # label refinement rounds; combat / attachment / target chains are short


def _strip(s: str) -> str:
    return _ID.sub("", s)


def canonical_view(game, viewer: int) -> dict:
    """What `viewer` can observe of `game`, id-free: `view.observe` (which
    already hides the opponent's hand and library order) plus the set of
    blocked attackers, with every object id replaced by a canonical label
    (Weisfeiler-Lehman refinement over the blocks / attached-to / targets
    relations). Zones whose order carries no information (hand, graveyard,
    exile, the battlefield) are sorted; the stack and known library
    positions keep their order. Two states that are the same up to renaming
    interchangeable objects get equal views."""
    o = observe(game, viewer)
    blocked = set(game.blocked)
    nodes: dict[int, tuple] = {}  # oid / sid -> own label
    edges: dict[int, list[tuple[str, int]]] = collections.defaultdict(list)
    for p in o["battlefield"]:
        oid = p["oid"]
        own = {k: v for k, v in p.items() if k not in ("oid", "blocking", "attached_to")}
        own["blocked"] = oid in blocked
        nodes[oid] = ("perm", tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in own.items())))
        if p["blocking"] is not None:
            edges[oid].append(("blocks", p["blocking"]))
        if p["attached_to"] is not None:
            edges[oid].append(("attached_to", p["attached_to"]))
    stack_ids = []
    for pos, it in enumerate(o["stack"]):
        sid = it["sid"]
        stack_ids.append(sid)
        targets = []
        for ti, label in enumerate(it["targets"]):
            targets.append(_strip(label))
            m = _ID.search(label)
            if m is not None and not label.startswith("player "):
                edges[sid].append((f"target{ti}", int(m.group(0)[1:])))
        nodes[sid] = ("stack", pos, it["name"], it["kind"], it["controller"], it["x"], str(it["method"]), tuple(targets))
    labels = dict(nodes)
    for _ in range(WL_ROUNDS):
        rev: dict[int, list] = collections.defaultdict(list)
        for src, out in edges.items():
            for rel, dst in out:
                rev[dst].append((rel, labels.get(src)))
        labels = {
            n: (
                nodes[n],
                tuple(sorted((rel, repr(labels.get(dst))) for rel, dst in edges.get(n, ()))),
                tuple(sorted((rel, repr(lab)) for rel, lab in rev.get(n, ()))),
            )
            for n in nodes
        }
        labels = {n: hashlib.blake2b(repr(lab).encode(), digest_size=12).hexdigest() for n, lab in labels.items()}

    def side(s: dict) -> dict:
        out = {}
        for k, v in s.items():
            if k in ("hand", "graveyard", "exile", "hand_known"):
                v = sorted(v)
            elif k == "pool":
                v = sorted(v.items())
            elif isinstance(v, list):
                v = [tuple(x) if isinstance(x, list) else x for x in v]
            out[k] = v
        return out

    d = game.decision
    if d is None:
        decision = None
    elif d.player == viewer:
        decision = ("self", d.kind, _strip(d.prompt), tuple(sorted(repr(o.key) for o in d.options)))
    else:
        decision = ("opponent", d.kind)
    return {
        "turn": o["turn"],
        "step": o["step"],
        "active": o["active"],
        "over": o["over"],
        "winner": o["winner"],
        "lands_played": o["lands_played"],
        "self": side(o["self"]),
        "opponent": side(o["opponent"]),
        "battlefield": sorted(labels[p["oid"]] for p in o["battlefield"]),
        "battlefield_names": sorted(p["name"] for p in o["battlefield"]),
        "stack": [labels[s] for s in stack_ids],
        "decision": decision,
    }


def observable_digest(game, viewer: int) -> str:
    """A short hash of `canonical_view(game, viewer)`."""
    return hashlib.blake2b(repr(sorted(canonical_view(game, viewer).items())).encode(), digest_size=12).hexdigest()


def successor_views(game, indices: list[int], viewer: int | None = None) -> list[dict]:
    """`canonical_view` (of the decider, by default) after stepping a fork
    of `game` with each option in `indices`. `game` itself is untouched."""
    viewer = game.decision.player if viewer is None else viewer
    out = []
    for i in indices:
        f = game.fork()
        f.step(i)
        out.append(canonical_view(f, viewer))
    return out


def _digest(view: dict) -> str:
    return hashlib.blake2b(repr(sorted(view.items())).encode(), digest_size=12).hexdigest()


def _diff_fields(views: list[dict]) -> list[str]:
    """Fields of the canonical view that differ between the successors."""
    out = []
    for k in views[0]:
        if k in ("self", "opponent"):
            out += [f"{k}.{kk}" for kk in views[0][k] if len({repr(v[k][kk]) for v in views}) > 1]
        elif len({repr(v[k]) for v in views}) > 1:
            out.append(k)
    return out


@dataclass
class GroupResult:
    options: list[int]  # option indices sharing one featurization
    classes: list[int]  # successor class of each option (0 = same as the first)
    diff: list[str]  # canonical-view fields that differ (blind spots only)

    @property
    def blind(self) -> bool:
        return max(self.classes) > 0


def audit_decision(game, features: int = FEATURES) -> list[GroupResult]:
    """Check the current decision: one result per group of 2+ options with
    identical featurization (for the decider, in feature set `features`)."""
    player = game.decision.player
    out = []
    for grp in colliding_groups(option_signatures(game, player, features)):
        views = successor_views(game, grp, player)
        digests = [_digest(v) for v in views]
        seen: dict[str, int] = {}
        classes = [seen.setdefault(dg, len(seen)) for dg in digests]
        out.append(GroupResult(grp, classes, _diff_fields(views) if len(seen) > 1 else []))
    return out


def is_blind_spot(game, a: int, b: int, features: int = FEATURES) -> bool | None:
    """For tests: None when options `a` and `b` featurize differently,
    else whether their successors differ (a blind spot)."""
    sigs = option_signatures(game, game.decision.player, features)
    if sigs[a] != sigs[b]:
        return None
    va, vb = successor_views(game, [a, b])
    return _digest(va) != _digest(vb)


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------


def make_agents(spec: str, seed: int, matchup: str, greedy: bool, features: int):
    """`spec`: two comma-separated agents from random, bot, search[:n],
    model:<checkpoint> (or a bare `*.pt` path)."""
    from .play import make_agent

    kinds = [k.strip() for k in spec.split(",")]
    if len(kinds) != 2:
        raise SystemExit(f"--agents needs two agents, got {spec!r}")
    kinds = [f"model:{k}" if k.endswith(".pt") and not k.startswith("model:") else k for k in kinds]
    decks = matchup_decks(matchup)
    return kinds, [make_agent(k, i, seed, greedy=greedy, deck=decks[i], features=features) for i, k in enumerate(kinds)]


def seat_features(agents, override: int) -> list[int]:
    """The feature set each seat is audited in: `override` if given, else a
    model agent's own version, else the latest."""
    if override:
        return [check_features(override)] * len(agents)
    return [getattr(a, "features", FEATURES) for a in agents]


def _names(game, option) -> list[str]:
    """Card names an option is about: the objects it points at
    (`encode.option_object_ids`), else the card names in its key."""
    names = []
    for i in option_object_ids(option):
        obj = game.perm(i) or game.stack_item(i)
        if obj is not None:
            names.append(obj.name)
    if not names:
        from .engine.cards import CARDS, TOKENS

        names = [e for e in option.key if isinstance(e, str) and (e in CARDS or e in TOKENS)]
    return sorted(set(names))


@dataclass
class Report:
    meta: dict
    decisions: int = 0
    audited: int = 0
    collided: int = 0  # audited decisions with a group of 2+ identical options
    blind: int = 0  # audited decisions with at least one blind-spot group
    groups: int = 0
    blind_groups: int = 0
    by_kind: dict = field(default_factory=lambda: collections.defaultdict(lambda: collections.Counter()))
    by_card: collections.Counter = field(default_factory=collections.Counter)
    by_seat: dict = field(default_factory=lambda: collections.defaultdict(lambda: collections.Counter()))
    examples: list = field(default_factory=list)
    _per_bucket: collections.Counter = field(default_factory=collections.Counter)

    def to_json(self) -> dict:
        per_k = lambda n, d: round(1000 * n / d, 3) if d else None  # noqa: E731
        kinds = {
            k: {**dict(c), "blind_per_1k": per_k(c["blind"], c["audited"])}
            for k, c in sorted(self.by_kind.items(), key=lambda kv: -kv[1]["blind"])
        }
        return {
            "format": FORMAT,
            "meta": self.meta,
            "decisions": self.decisions,
            "audited": self.audited,
            "collided": self.collided,
            "blind": self.blind,
            "groups": self.groups,
            "blind_groups": self.blind_groups,
            "blind_per_1k": per_k(self.blind, self.audited),
            "collided_per_1k": per_k(self.collided, self.audited),
            "by_kind": kinds,
            "by_seat": {str(s): dict(c) for s, c in sorted(self.by_seat.items())},
            "by_card": dict(self.by_card.most_common()),
            "examples": self.examples,
        }


def run(
    matchup: str = DEFAULT_MATCHUP,
    games: int = 10,
    seed: int = 0,
    engine: str | None = None,
    agents: str = "bot,bot",
    features: int = 0,
    sample_frac: float = 1.0,
    max_examples: int = 50,
    per_bucket: int = 3,
    greedy: bool = False,
    max_turns: int | None = None,
    auto_mana: bool = False,
    auto_pass: bool = False,
    progress=None,
) -> dict:
    """Play `games` games and audit their decisions; returns the JSON report.
    Game i: seed `seed + i`, seat `i % 2` starts, game 1 decks of `matchup`."""
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    eng = engine_name(engine)
    Game = game_class(eng)
    kinds, ags = make_agents(agents, seed, matchup, greedy, features)
    feats = seat_features(ags, features)
    rep = Report(meta={
        "matchup": matchup, "games": games, "seed": seed, "engine": eng, "agents": kinds, "features": feats,
        "sample_frac": sample_frac, "greedy": greedy, "max_turns": max_turns, "auto_mana": auto_mana, "auto_pass": auto_pass,
        "game": "Game(**match.game_args(1, matchup), seed=seed + i, starting_player=i % 2, max_turns, auto_mana, auto_pass)",
    })  # fmt: skip
    rng = random.Random(seed)
    decks = matchup_decks(matchup)
    t0 = time.perf_counter()
    for gi in range(games):
        gseed, start = seed + gi, gi % 2
        g = Game(**game_args(1, matchup), seed=gseed, starting_player=start, max_turns=max_turns, auto_mana=auto_mana, auto_pass=auto_pass)
        while not g.over:
            d = g.decision
            rep.decisions += 1
            if rng.random() < sample_frac:
                _audit_one(rep, g, feats[d.player], decks, matchup, gseed, start, max_examples, per_bucket)
            take(g, ags, ags[d.player].act(g))
        if progress:
            progress(gi + 1, rep, time.perf_counter() - t0)
    rep.meta["seconds"] = round(time.perf_counter() - t0, 2)
    return rep.to_json()


def _audit_one(rep: Report, g, features: int, decks, matchup: str, gseed: int, start: int, max_examples: int, per_bucket: int) -> None:
    d = g.decision
    kind, player = d.kind, d.player
    rep.audited += 1
    k = rep.by_kind[kind]
    k["audited"] += 1
    s = rep.by_seat[player]
    s["audited"] += 1
    results = audit_decision(g, features)
    if not results:
        return
    rep.collided += 1
    k["collided"] += 1
    s["collided"] += 1
    rep.groups += len(results)
    blind = [r for r in results if r.blind]
    if not blind:
        return
    rep.blind += 1
    k["blind"] += 1
    s["blind"] += 1
    rep.blind_groups += len(blind)
    opts = d.options
    for r in blind:
        names = sorted({n for i in r.options for n in _names(g, opts[i])})
        for n in names:
            rep.by_card[n] += 1
        bucket = (kind, tuple(names))
        if len(rep.examples) < max_examples and rep._per_bucket[bucket] < per_bucket:
            rep._per_bucket[bucket] += 1
            rep.examples.append({
                "matchup": matchup, "seed": gseed, "starting_player": start, "decision": len(g.actions), "turn": g.turn,
                "player": player, "deck": decks[player], "kind": kind, "features": features, "cards": names,
                "options": [{"index": i, "key": list(opts[i].key), "label": opts[i].label, "class": c} for i, c in zip(r.options, r.classes)],
                "differs": r.diff, "actions": list(g.actions),
            })  # fmt: skip


def rebuild(example: dict, engine: str | None = None, max_turns: int | None = None, auto_mana: bool = False, auto_pass: bool = False):
    """The game at an example's decision (replays its `actions`)."""
    g = game_class(engine)(**game_args(1, example["matchup"]), seed=example["seed"], starting_player=example["starting_player"], max_turns=max_turns, auto_mana=auto_mana, auto_pass=auto_pass)
    for a in example["actions"]:
        g.step(a)
    return g


def summary(rep: dict, top: int = 10) -> str:
    m = rep["meta"]
    lines = [
        f"audit {m['matchup']} [{m['engine']}] agents={','.join(m['agents'])} features={m['features']} games={m['games']} seed={m['seed']} sample={m['sample_frac']}",
        f"  decisions {rep['decisions']}, audited {rep['audited']}, with identical options {rep['collided']} ({rep['collided_per_1k']}/1k), "
        f"blind spots {rep['blind']} ({rep['blind_per_1k']}/1k), groups {rep['groups']}, blind groups {rep['blind_groups']}",
        "  by decision kind (audited, collided, blind, blind/1k of that kind):",
    ]
    for kind, c in rep["by_kind"].items():
        lines.append(f"    {kind:22s} {c.get('audited', 0):7d} {c.get('collided', 0):6d} {c.get('blind', 0):6d}  {c['blind_per_1k']}")
    if rep["by_card"]:
        lines.append("  blind-spot groups by card: " + ", ".join(f"{n} {v}" for n, v in list(rep["by_card"].items())[:top]))
    for ex in rep["examples"][:top]:
        opts = "; ".join(f"[{o['class']}] {o['label']}" for o in ex["options"])
        lines.append(f"  e.g. seed {ex['seed']} d{ex['decision']} t{ex['turn']} p{ex['player']} {ex['kind']}: {opts}  (differs: {', '.join(ex['differs'])})")
    if "seconds" in m:
        lines.append(f"  {m['seconds']}s, {1000 * m['seconds'] / max(rep['audited'], 1):.2f}s per 1k audited decisions")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mtg_ml.audit", description=__doc__.split("\n\n")[1] if __doc__ else None)
    ap.add_argument("--matchup", default=DEFAULT_MATCHUP, help=f"match.MATCHUPS: {', '.join(MATCHUPS)} (seat 0 deck first; mirrors are used only where named)")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default=None, help="python (reference) or native (Rust); default: $MTG_ENGINE, else python")
    ap.add_argument("--agents", default="bot,bot", help="seat0,seat1 from: random, bot, search[:n], model:<checkpoint> (or a bare .pt path)")
    ap.add_argument("--features", type=int, default=0, choices=(0, *FEATURE_VERSIONS), help="audit (and featurize model agents) in this feature set (0: each model's own, else the latest)")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="audit this fraction of decisions (forking per group costs)")
    ap.add_argument("--greedy", action="store_true", help="model agents play argmax instead of sampling")
    ap.add_argument("--max-turns", type=int, default=None, help="turn cap (both players' turns); default: no limit")
    ap.add_argument("--auto-mana", action="store_true")
    ap.add_argument("--auto-pass", action="store_true")
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def progress(n, rep, dt):
        if not args.quiet:
            print(f"  game {n}/{args.games}: {rep.audited} audited, {rep.blind} blind ({dt:.1f}s)", file=sys.stderr, flush=True)

    rep = run(args.matchup, args.games, args.seed, args.engine, args.agents, args.features, args.sample_frac, args.max_examples,
              greedy=args.greedy, max_turns=args.max_turns, auto_mana=args.auto_mana, auto_pass=args.auto_pass, progress=progress)  # fmt: skip
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    print(summary(rep))


if __name__ == "__main__":
    main()
