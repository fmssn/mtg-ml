"""Render a review game (a replay file plus `meta["review"]`) as reviewer text.

The transcript is built for an LLM reading it once, top to bottom:

* a header with the decks, the card texts as the engine implements them, the
  policy's training state and the reward the policy was trained on;
* one block per decision that offered a choice (forced decisions only appear
  as their log line). Each lists every option the engine offered, which is the
  complete legal action mask, with the policy's probability for it, marks the
  chosen one, and gives the deciding seat's value estimate;
* the omniscient state whenever it changed since it was last printed, with the
  information each seat could not see marked as hidden.

Decision ids (`d<frame index>`) are what the reviewer cites and what
`verify` forks the game at.
"""

from __future__ import annotations

from collections import Counter

CONFIDENT = 0.98  # a choice at least this likely under the policy is printed on one line


def _hand(cards) -> str:
    return ", ".join(c["name"] for c in cards) or "-"


def _perm(p: dict) -> str:
    s = f"{p['name']}#{p['oid']}"
    if "power" in p:
        s += f" {p['power']}/{p['toughness']}"
        if p.get("keywords"):
            s += " " + ",".join(p["keywords"])
    flags = [k for k in ("tapped", "sick", "attacking", "token") if p.get(k)]
    if p.get("blocking"):
        flags.append(f"blocking #{p['blocking']}")
    if p.get("damage"):
        flags.append(f"{p['damage']} dmg")
    if p.get("counters"):
        flags.append(f"counters {p['counters']}")
    if p.get("attached_to"):
        flags.append(f"on #{p['attached_to']}")
    return s + (f" ({', '.join(flags)})" if flags else "")


def render_state(state: dict) -> list[str]:
    """Labelled state lines (the step is in every decision line, so not here)."""
    lines = []
    for i, p in enumerate(state["players"]):
        known = f", known library top {p['library_top_known']}" if p["library_top_known"] else ""
        pool = f", mana pool {p['pool']}" if p["pool"] else ""
        lines.append(f"P{i} life {p['life']}, library {p['library']}{known}{pool}")
        lines.append(f"P{i} hand (hidden from P{1 - i}): {_hand(p['hand'])}")
        lines.append(f"P{i} graveyard: {_hand(p['graveyard'])}" + (f"; exile: {_hand(p['exile'])}" if p["exile"] else ""))
        perms = [_perm(c) for c in state["battlefield"] if c["controller"] == i]
        lines.append(f"P{i} battlefield: {'; '.join(perms) or '-'}")
    items = [f"{it['name']} (P{it['controller']}" + (f" -> {', '.join(it['targets'])}" if it["targets"] else "") + ")" for it in reversed(state["stack"])]
    lines.append(f"stack (top first): {' | '.join(items) or 'empty'}")
    return lines


def _cards(cards: dict) -> list[str]:
    out = []
    for name in sorted(cards):
        c = cards[name]
        typ = " ".join(c["types"]) + (" - " + " ".join(c["subtypes"]) if c["subtypes"] else "")
        pt = f" {c['power']}/{c['toughness']}" if c["power"] is not None else ""
        cost = f" {c['cost']}" if c["cost"] else ""
        out.append(f"  {name}{cost} [{typ}{pt}]: {c['text'] or '(no text)'}")
    return out


def header(rep: dict) -> list[str]:
    m = rep["meta"]
    rv = m.get("review", {})
    lines = ["# GAME", f"matchup {rv.get('matchup', '?')}, game {m['match_game']} of a match ({'maindecks' if m['match_game'] == 1 else 'sideboarded'}), engine {m['engine']}, seed {m['seed']}"]
    for i in range(2):
        lines.append(f"P{i}: {m['decks'][i]}, played by {m['agents'][i]}" + (" (greedy)" if rv.get("greedy") else " (sampling from its policy)" if m["agents"][i].startswith("model") else ""))
    winner = "draw" if m["winner"] is None else f"P{m['winner']} wins"
    lines.append(f"P{m['starting_player']} started. Result: {winner} ({m['end_reason']}) on turn {m['turns']}.")
    for i, pol in enumerate(rv.get("policies", [])):
        if pol:
            lines.append(f"P{i} policy: {pol}")
    if rv.get("reward"):
        lines.append(f"Training reward: {rv['reward']}")
    if rv.get("decklists"):
        lines.append("")
        lines.append("# DECKLISTS")
        for i, deck in enumerate(rv["decklists"]):
            lines.append(f"P{i}: " + ", ".join(f"{n} {name}" for name, n in sorted(Counter(deck).items())))
    lines.append("")
    lines.append("# CARD TEXT (as the engine implements it; compare against real Magic rules)")
    lines += _cards(rep["cards"])
    return lines


def _option(i: int, label: str, policy, chosen: int) -> str:
    p = f" {policy[i]:.2f}" if policy else ""
    return f"[{i}]{'*' if i == chosen else ''} {label}{p}"


def decision_lines(idx: int, frame: dict) -> list[str]:
    d = frame["decision"]
    s = frame["state"]
    pol = d.get("policy")
    ch = d["chosen"]
    where = f"d{idx} T{s['turn']} {s['step']} P{d['player']} {d['kind']}"
    v = f" value {d['value']:+.2f}" if "value" in d else ""
    if pol and pol[ch] >= CONFIDENT and ch == max(range(len(pol)), key=pol.__getitem__):
        others = [lbl for i, lbl in enumerate(d["options"]) if i != ch]
        return [f"{where}: chose [{ch}] {d['options'][ch]} ({pol[ch]:.2f});{v}; also legal: {' | '.join(others)}"]
    rank = ""
    if pol:
        rank = f", rank {sorted(pol, reverse=True).index(pol[ch]) + 1} of {len(pol)}, p {pol[ch]:.2f}"
    return [
        f"{where} \"{d['prompt']}\"{v}",
        "  options (complete legal set, policy prob): " + " | ".join(_option(i, lbl, pol, ch) for i, lbl in enumerate(d["options"])),
        f"  chose [{ch}] {d['options'][ch]}{rank}",
    ]


def transcript(rep: dict, turns: tuple[int, int] | None = None) -> str:
    """The whole game, or only decisions in turns [lo, hi] (the header always)."""
    lines = header(rep)
    lines += [
        "",
        "# PLAY",
        "dN = decision id; * = chosen option.",
        "Decisions with a single option are not listed: their log line (`p0 <kind>: <option>`) is the ONLY option the engine offered, so a missing alternative there (a block, an attack, a spell) is a masking question.",
        "`state:` lines are printed when they change; each one replaces the earlier line with the same label.",
    ]
    last: dict[str, str] = {}
    for idx, f in enumerate(rep["frames"]):
        t = f["state"]["turn"]
        inside = turns is None or turns[0] <= t <= turns[1]
        if inside:
            lines += ["  log: " + e.strip() for e in f["events"]]
        d = f["decision"]
        if not inside or d is None or len(d["options"]) < 2:
            continue
        for line in render_state(f["state"]):
            label = line.split(":")[0] if line.startswith("stack") else " ".join(line.split()[:2])
            if last.get(label) != line:
                lines.append("  state: " + line)
                last[label] = line
        lines += decision_lines(idx, f)
    return "\n".join(lines) + "\n"


def turn_chunks(rep: dict, max_chars: int) -> list[tuple[int, int]]:
    """Split the game into turn ranges whose transcripts fit in `max_chars`."""
    last = rep["frames"][-1]["state"]["turn"]
    if len(transcript(rep)) <= max_chars:
        return [(0, last)]
    chunks, lo = [], 0
    while lo <= last:
        hi = lo
        while hi < last and len(transcript(rep, (lo, hi + 1))) <= max_chars:
            hi += 1
        chunks.append((lo, hi))
        lo = hi + 1
    return chunks
