"""Seeded faults with known ground truth, to measure what the reviewer catches.

Three kinds, one per cause the reviewer has to tell apart:

* `mask:<seat>:<text>` (masking_gap): the seat's agent never sees options
  whose label contains `<text>`; the engine is untouched, the options are
  filtered in place before the agent decides, so the recorded option lists
  lack them exactly as an engine masking bug would. Python engine only.
* `blunder:<seat>:<n>` (sampling_noise): at up to `n` decisions where the
  policy is confident (top option p > 0.8, after turn 2) the seat plays the
  option its policy likes least.
* `life:<seat>:<turn>:<delta>` and `power:<seat>:<turn>:<delta>` (engine_bug):
  the recorded game is edited after the fact so that from the first decision
  of `<turn>` on, the seat's life total (or one of its creatures' power)
  differs by `<delta>` with no event explaining it.

Every fault records where it happened in `meta.review.faults`
({kind, seat, cause, decisions, detail}); `score` matches findings to them.
"""

from __future__ import annotations

import random


class _Wrapper:
    """Forwards everything (observe, last_info, ...) to the wrapped agent."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


class HideOptions(_Wrapper):
    def __init__(self, inner, text: str, fault: dict):
        super().__init__(inner)
        self.text = text
        self.fault = fault

    def act(self, g) -> int:
        opts = g.decision.options
        keep = [o for o in opts if self.text not in o.label]
        if keep and len(keep) < len(opts):
            self.fault["decisions"].append(len(g.actions))
            opts[:] = keep  # in place: Game.step indexes this same list
        return self.inner.act(g)


class Blunder(_Wrapper):
    def __init__(self, inner, n: int, fault: dict, seed: int):
        super().__init__(inner)
        self.left = n
        self.fault = fault
        self.rng = random.Random(seed)

    def act(self, g) -> int:
        a = self.inner.act(g)
        pol = (getattr(self.inner, "last_info", None) or {}).get("policy")
        if not pol or self.left <= 0 or g.turn < 3 or len(pol) < 2 or max(pol) < 0.8:
            return a
        if g.decision.kind not in ("priority", "declare_attacker", "declare_blocker", "target") or self.rng.random() > 0.15:
            return a
        worst = min(range(len(pol)), key=pol.__getitem__)
        self.left -= 1
        self.fault["decisions"].append(len(g.actions))
        self.fault.setdefault("detail", []).append(f"forced {g.decision.options[worst].label} (p {pol[worst]:.3f})")
        return worst


def parse(spec: str) -> dict:
    kind, seat, *rest = spec.split(":")
    f = {"kind": kind, "seat": int(seat), "decisions": [], "spec": spec}
    f["cause"] = {"mask": "masking_gap", "blunder": "sampling_noise", "life": "engine_bug", "power": "engine_bug"}[kind]
    f["args"] = rest
    return f


def wrapper(faults: list[dict], seed: int):
    def wrap(seat, agent):
        for f in faults:
            if f["seat"] != seat:
                continue
            if f["kind"] == "mask":
                agent = HideOptions(agent, f["args"][0], f)
            elif f["kind"] == "blunder":
                agent = Blunder(agent, int(f["args"][0]), f, seed * 31 + seat)
        return agent

    return wrap


def tamper(rep: dict, f: dict) -> None:
    """Apply a post-hoc engine_bug fault to the recorded frames."""
    turn, delta = int(f["args"][0]), int(f["args"][1])
    frames = rep["frames"]
    start = next((i for i, fr in enumerate(frames) if fr["state"]["turn"] >= turn and fr["decision"] and len(fr["decision"]["options"]) > 1), None)
    if start is None:
        return
    if f["kind"] == "life":
        for fr in frames[start:]:
            fr["state"]["players"][f["seat"]]["life"] += delta
        f["detail"] = [f"P{f['seat']} life shifted by {delta} from d{start} with no event"]
        f["decisions"].append(start)
        return
    # power: the seat's longest-lived creature on the battlefield at `start`
    mine = [p for p in frames[start]["state"]["battlefield"] if p["controller"] == f["seat"] and "power" in p]
    if not mine:
        return
    alive = {p["oid"]: sum(1 for fr in frames[start:] if any(q["oid"] == p["oid"] for q in fr["state"]["battlefield"])) for p in mine}
    oid = max(alive, key=alive.get)
    for fr in frames[start:]:
        for p in fr["state"]["battlefield"]:
            if p["oid"] == oid:
                p["power"] += delta
    f["detail"] = [f"{next(p['name'] for p in mine if p['oid'] == oid)}#{oid} power shifted by {delta} from d{start}"]
    f["decisions"].append(start)


def score(faults: list[dict], findings: list[dict], window: int = 3) -> list[dict]:
    """Per fault: caught (a finding of the right cause near one of its
    decisions; for mask faults, also one naming the hidden text)."""
    out = []
    for f in faults:
        if not f["decisions"]:
            out.append({**f, "caught": None})
            continue
        hit = None
        for x in findings:
            if x.get("cause") != f["cause"] or x.get("decision") is None:
                continue
            near = any(abs(x["decision"] - d) <= window for d in f["decisions"])
            # an engine-state fault persists, so a later citation counts too
            later = f["cause"] == "engine_bug" and x["decision"] >= f["decisions"][0]
            named = f["kind"] == "mask" and f["args"][0].lower() in (str(x.get("missing_play")) + str(x.get("what"))).lower()
            if near or later or named:
                hit = x
                break
        out.append({**f, "caught": hit is not None, "finding": hit})
    return out
