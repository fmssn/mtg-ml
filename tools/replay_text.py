"""Replay files (`mtg_ml.replay record`) as plain-text transcripts for game reviews.

    python tools/replay_text.py replays/*.json --out reviews/

One line per decision: who decided, the decision kind, the chosen option and
its probability, the strongest alternatives (for model agents) and the value
estimate; the engine log lines between decisions; and the full (omniscient)
state whenever the step changes. Hands are shown for both players, so mark
which information the deciding player actually had when judging a play.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def _cards(cs) -> str:
    return ", ".join(c["name"] for c in cs) or "-"


def _perm(p: dict) -> str:
    s = p["name"]
    if "power" in p:
        s += f" {p['power']}/{p['toughness']}"
    flags = [k for k in ("tapped", "sick", "attacking", "token") if p.get(k)]
    if p.get("blocking"):
        flags.append("blocking")
    if p.get("damage"):
        flags.append(f"{p['damage']} dmg")
    if p.get("counters"):
        flags.append(f"+{p['counters']}")
    return s + (f" ({', '.join(flags)})" if flags else "")


def state_lines(st: dict, decks: list[str]) -> list[str]:
    out = [f"  T{st['turn']} {st['step']}, p{st['active']} active"]
    for i, p in enumerate(st["players"]):
        board = "; ".join(_perm(c) for c in st["battlefield"] if c["controller"] == i) or "-"
        out.append(f"  p{i} {decks[i]}: life {p['life']}, library {p['library']}, hand [{_cards(p['hand'])}]")
        out.append(f"      board: {board}")
        out.append(f"      graveyard: {_cards(p['graveyard'])} | exile: {_cards(p['exile'])}" + (f" | pool {p['pool']}" if p["pool"] else ""))
    if st["stack"]:
        out.append("  stack (top last): " + " | ".join(f"{it['name']} -> {', '.join(it['targets']) or '-'}" for it in st["stack"]))
    return out


def transcript(rep: dict) -> str:
    m = rep["meta"]
    decks = m["decks"]
    lines = [
        f"seed {m['seed']}, game {m.get('match_game', 1)}: p0 {decks[0]} ({m['agents'][0]}) vs p1 {decks[1]} ({m['agents'][1]});"
        f" p{m['starting_player']} starts; winner {m['winner']} ({m['end_reason']}) after {m['turns']} turns",
        "",
    ]
    last = None
    for i, fr in enumerate(rep["frames"]):
        for e in fr["events"]:
            if not e.startswith(("  p", "-- ")):  # decisions are printed below with their policy; steps in the state header
                lines.append(f"    . {e}")
        st = fr["state"]
        key = (st["turn"], st["step"], st["active"])
        if key != last:
            lines.append("")
            lines += state_lines(st, decks)
            last = key
        d = fr["decision"]
        if d is None:
            continue
        chosen = d["options"][d["chosen"]]
        pol = d.get("policy")
        s = f"  #{i} p{d['player']} {d['kind']}: {chosen}"
        if pol:
            alts = sorted((p, o) for j, (p, o) in enumerate(zip(pol, d["options"])) if j != d["chosen"])[::-1][:3]
            s += f"  [p={pol[d['chosen']]:.2f}" + "".join(f"; {o} {p:.2f}" for p, o in alts if p >= 0.02) + f"] V={d['value']:+.2f}"
        elif len(d["options"]) > 1:
            s += f"  ({len(d['options'])} options)"
        lines.append(s)
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="directory for <name>.txt (default: print)")
    args = ap.parse_args(argv)
    for f in args.files:
        text = transcript(json.loads(pathlib.Path(f).read_text()))
        if args.out:
            out = pathlib.Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / (pathlib.Path(f).stem + ".txt")).write_text(text)
        else:
            print(text)


if __name__ == "__main__":
    main()
