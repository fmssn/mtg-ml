"""Independent trigger audit over game logs.

Re-derives, from log lines alone, which triggered abilities *should* have
fired (permanents entering / leaving, sacrifices, casts, upkeeps, opponents
targeting Tolarian Terror) and checks them against the "trigger -> stack"
lines the engine wrote. It does not reuse engine trigger code, so a trigger
the engine forgets to create shows up as "missed".

Rule checked: every expected trigger must be put on the stack before the next
resolution, priority action, step or turn change (CR 117.2: triggers go on the
stack the next time a player would receive priority). Unexpected stacked
triggers are reported as "spurious", stacked triggers that never resolve (and
were not countered or cut off by the game ending) as "unresolved".

    python tools/audit_triggers.py --games 100 --out logs/
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ETB = {"Ichor Wellspring": "draw a card", "Lembas": "scry 1, draw", "Refurbished Familiar": "opponent discards"}
DIES = {"Ichor Wellspring": "draw a card", "Lembas": "shuffle into library", "Nihil Spellbomb": "pay {B}: draw"}
ELDRAZI = {"Eldrazi Spawn", "Writhing Chrysalis"}

R_TURN = re.compile(r"^=== Turn \d+: player (\d) ===")
R_ENTER = re.compile(r"^enters: (.+)#(\d+) \(p(\d)\)$")
R_LEAVE = re.compile(r"^leaves: (.+)#(\d+) \(p(\d)\) -> (\w+)$")
R_SAC = re.compile(r"^p(\d) sacrifices (.+)#(\d+)$")
R_CAST = re.compile(r"^p(\d) casts (.+) \((\w+)\) from (\w+)$")
R_TARGET = re.compile(r"^  p(\d) target: Target (.+)#(\d+) \((self|opponent)\)")
R_TRANSFORM = re.compile(r"^Delver of Secrets#(\d+) transforms")
R_STACK = re.compile(r"^trigger -> stack: (.+?): (.+)$")
R_RESOLVE = re.compile(r"^resolve (.+?): (.+)$")


def audit(lines: list[str]) -> dict:
    bf: dict[int, tuple[str, int]] = {}
    transformed: set[int] = set()
    active = None
    pending: list[tuple[str, str, int]] = []
    on_stack = collections.Counter()
    out = {"missed": [], "spurious": [], "unresolved": [], "expected": 0, "stacked": 0}

    def expect(src, name, i):
        pending.append((src, name, i))
        out["expected"] += 1

    def window_closes(i):
        for src, name, j in pending:
            out["missed"].append((j, f"{src}: {name}", lines[j]))
        pending.clear()

    for i, line in enumerate(lines):
        if line.startswith("resolve ") or re.match(r"^  p\d priority:", line) or line.startswith("=== Turn") or line.startswith("-- "):
            window_closes(i)
        if m := R_TURN.match(line):
            active = int(m.group(1))
        elif line == "-- upkeep":
            for oid, (name, ctl) in bf.items():
                if name == "Delver of Secrets" and ctl == active and oid not in transformed:
                    expect("Delver of Secrets", "look at top card", i)
        elif m := R_ENTER.match(line):
            name, oid, ctl = m.group(1), int(m.group(2)), int(m.group(3))
            bf[oid] = (name, ctl)
            if name in ETB:
                expect(name, ETB[name], i)
        elif m := R_LEAVE.match(line):
            name, oid, to = m.group(1), int(m.group(2)), m.group(4)
            bf.pop(oid, None)
            transformed.discard(oid)
            if to == "graveyard" and name in DIES:
                expect(name, DIES[name], i)
        elif m := R_SAC.match(line):
            p, name, oid = int(m.group(1)), m.group(2), int(m.group(3))
            for o, (n, ctl) in bf.items():
                if o == oid or ctl != p:
                    continue
                if n == "Gixian Infiltrator":
                    expect(n, "+1/+1 counter", i)
                if n == "Writhing Chrysalis" and name in ELDRAZI:
                    expect(n, "+1/+1 counter", i)
        elif m := R_CAST.match(line):
            if m.group(2) == "Writhing Chrysalis":
                expect("Writhing Chrysalis", "create two Eldrazi Spawn", i)
        elif m := R_TARGET.match(line):
            if m.group(2) == "Tolarian Terror" and m.group(4) == "opponent":
                expect("Tolarian Terror", "ward", i)
        elif m := R_TRANSFORM.match(line):
            transformed.add(int(m.group(1)))
        elif m := R_STACK.match(line):
            src, name = m.group(1), m.group(2)
            out["stacked"] += 1
            on_stack[(src, name)] += 1
            hit = next((k for k, (s, n, _) in enumerate(pending) if s == src and n == name), None)
            if hit is None:
                out["spurious"].append((i, f"{src}: {name}", line))
            else:
                pending.pop(hit)
        elif m := R_RESOLVE.match(line):
            key = (m.group(1), m.group(2))
            if on_stack[key] > 0:
                on_stack[key] -= 1
    ended = any(l.startswith("GAME OVER") for l in lines)
    if pending and not ended:
        window_closes(len(lines) - 1)
    # Triggers still on the stack when the game ended are fine; ward triggers
    # whose spell was already gone just resolve and do nothing (still logged).
    if not ended:
        out["unresolved"] = [k for k, v in on_stack.items() if v]
    out["left_on_stack_at_end"] = sum(on_stack.values())
    return out


def main(argv=None) -> int:
    from mtg_ml.agents import RandomAgent, play_game

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory to write game logs to")
    args = ap.parse_args(argv)
    totals = collections.Counter()
    per_trigger = collections.Counter()
    bad = 0
    results = collections.Counter()
    for s in range(args.seed, args.seed + args.games):
        g = play_game([RandomAgent(s), RandomAgent(s + 10_000)], seed=s, log=True)
        results[(g.winner, g.end_reason)] += 1
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"game_{s:03d}.log"), "w") as f:
                f.write("\n".join(g.log) + "\n")
        r = audit(g.log)
        totals["expected"] += r["expected"]
        totals["stacked"] += r["stacked"]
        for line in g.log:
            if m := R_STACK.match(line):
                per_trigger[f"{m.group(1)}: {m.group(2)}"] += 1
        problems = r["missed"] + r["spurious"]
        if problems or r["unresolved"]:
            bad += 1
            print(f"seed {s}: missed={r['missed']} spurious={r['spurious']} unresolved={r['unresolved']}")
    print(f"{args.games} games, results {dict(results)}")
    print(f"expected triggers {totals['expected']}, stacked {totals['stacked']}, games with problems: {bad}")
    for k, v in per_trigger.most_common():
        print(f"  {v:5d}  {k}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
