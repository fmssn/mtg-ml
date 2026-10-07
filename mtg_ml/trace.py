"""Reproducible game traces: scenario specs, driver agents and digests.

A `Scenario` fixes everything that determines a game: decks (game 1
maindecks or games 2/3 sideboarded), seed, starting player, mulligans, the
turn limit and the agents driving each seat. `play(scenario)` runs it and
`digest(scenario)` hashes everything observable (every decision with its
prompt, labels and keys, the chosen option, periodic `observe()` views,
the log and the outcome). Golden digests guard engine refactors; the
differential harness (`mtg_ml.difftest`) plays the same scenarios on two
engines in lockstep.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

AGENTS = ("random", "chaos", "bot")


@dataclass(frozen=True)
class Scenario:
    seed: int
    agents: tuple[str, str] = ("random", "random")
    match_game: int = 1
    starting_player: int | None = None
    mulligans: bool = True
    max_turns: int = 100
    # (seat, card name, copies): put cards that are in no decklist yet into
    # a deck (replacing its last `copies` cards) to fuzz them.
    extra: tuple[tuple[int, str, int], ...] = ()
    # Action decomposition options of the game (see Game.auto_mana/auto_pass).
    auto_mana: bool = False
    auto_pass: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Scenario":
        d = dict(d)
        d["agents"] = tuple(d["agents"])
        d["extra"] = tuple(tuple(e) for e in d.get("extra", ()))
        return Scenario(**d)

    def decks(self) -> tuple[list[str], list[str]]:
        from .match import match_decks

        decks = [list(d) for d in match_decks(self.match_game)]
        for seat, name, n in self.extra:
            decks[seat][-n:] = [name] * n
        return decks[0], decks[1]


def scenarios(n: int, start: int = 0, extra: tuple = (), auto_mana: bool = False, auto_pass: bool = False) -> list[Scenario]:
    """A deterministic mix: random/chaos/bot seats, games 1-3, both starting
    players, occasional short turn limits and mulligan-free games."""
    out = []
    pairs = [("random", "random"), ("chaos", "chaos"), ("bot", "bot"), ("bot", "random"), ("random", "bot"), ("chaos", "bot"), ("bot", "chaos")]
    for i in range(start, start + n):
        r = random.Random(i * 7919 + 17)
        out.append(
            Scenario(
                seed=i,
                agents=pairs[i % len(pairs)],
                match_game=r.choice((1, 1, 2, 3)),
                starting_player=r.choice((None, 0, 1)),
                mulligans=r.random() < 0.9,
                max_turns=r.choice((100, 100, 100, 12)),
                extra=tuple(extra),
                auto_mana=auto_mana,
                auto_pass=auto_pass,
            )
        )
    return out


class ChaosAgent:
    """Uniform over every option, mulligans included (to exercise the London
    mulligan and odd hand sizes); passes with probability `pass_bias`."""

    def __init__(self, seed: int, pass_bias: float = 0.3):
        self.rng = random.Random(seed)
        self.pass_bias = pass_bias

    def act(self, g) -> int:
        opts = g.legal_options()
        if g.decision.kind == "mulligan":
            return 1 if len(opts) > 1 and self.rng.random() < 0.3 else 0
        if opts[0].key == ("pass",) and self.rng.random() < self.pass_bias:
            return 0
        return self.rng.randrange(len(opts))


def make_agents(sc: Scenario):
    from .agents import RandomAgent
    from .bots import make_bot

    agents = []
    for seat, kind in enumerate(sc.agents):
        if kind == "random":
            agents.append(RandomAgent(sc.seed * 2 + seat))
        elif kind == "chaos":
            agents.append(ChaosAgent(sc.seed * 2 + seat))
        elif kind == "bot":
            agents.append(make_bot(seat))
        else:
            raise ValueError(f"unknown agent {kind!r}")
    return agents


def new_game(sc: Scenario, engine: str | None = None, log: bool = True):
    from .backend import game_class

    cls = game_class(engine)
    return cls(
        sc.decks(),
        seed=sc.seed,
        starting_player=sc.starting_player,
        max_turns=sc.max_turns,
        log=log,
        mulligans=sc.mulligans,
        match_game=sc.match_game,
        auto_mana=sc.auto_mana,
        auto_pass=sc.auto_pass,
    )


def decision_record(g) -> list:
    d = g.decision
    if d is None:
        return [None]
    return [d.player, d.kind, d.prompt, [o.label for o in d.options], [list(_jsonable(o.key)) for o in d.options]]


def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(e) for e in x]
    if isinstance(x, list):
        return [_jsonable(e) for e in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def play(sc: Scenario, engine: str | None = None, observe_every: int = 5):
    """Play `sc`; returns (game, record list)."""
    from .engine.view import observe

    g = new_game(sc, engine)
    agents = make_agents(sc)
    rec = []
    n = 0
    while not g.over:
        rec.append(decision_record(g))
        if n % observe_every == 0:
            rec.append([_jsonable(observe(g, 0)), _jsonable(observe(g, 1))])
        a = agents[g.decision.player].act(g)
        rec.append(a)
        g.step(a)
        n += 1
    rec.append([g.winner, g.end_reason, g.turn, len(g.actions)])
    rec.append(list(g.log))
    return g, rec


def digest(sc: Scenario, engine: str | None = None) -> str:
    _, rec = play(sc, engine)
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="mtg_ml.trace", description="golden trace digests")
    ap.add_argument("cmd", choices=("record", "check"))
    ap.add_argument("--file", default="tests/data/golden_digests.json")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--engine", default=None)
    args = ap.parse_args(argv)
    scs = scenarios(args.games)
    if args.cmd == "record":
        data = [{"scenario": sc.to_json(), "digest": digest(sc, args.engine)} for sc in scs]
        with open(args.file, "w") as f:
            json.dump(data, f, indent=0)
        print(f"recorded {len(data)} digests to {args.file}")
    else:
        with open(args.file) as f:
            data = json.load(f)
        bad = [d for d in data if digest(Scenario.from_json(d["scenario"]), args.engine) != d["digest"]]
        print(f"{len(data) - len(bad)}/{len(data)} digests match")
        for d in bad[:10]:
            print("  mismatch:", d["scenario"])
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
