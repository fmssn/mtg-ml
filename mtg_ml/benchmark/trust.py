"""The trust test (step 7 of docs/representation-plan.md): is a checkpoint fit for card choices?

    python -m mtg_ml.benchmark.trust CKPT.pt --matches 400 --workers 32 \
        --pool old1.pt old2.pt --out runs/trust/CKPT

Three checks, all computed from raw game rows with the statistics of `stats` (paired
four-slot blocks, one shared bootstrap index list per replicate):

1. per pairing: the checkpoint plays itself in the 15 non-mirror pairings of the six decks;
   the win rate of the first deck of each pair is compared with the Pauper-Research
   interval (`classify`). Thin reference pairings are reported, never counted.
2. per deck: bias = mean over the deck's non-thin pairings of (model - reference win rate);
   flagged when its 95% interval excludes 0 by more than `--bias-margin` (`deck_biases`).
3. exploitability: every frozen-pool checkpoint plays the final model in each of the 30
   ordered matchups; the best pool member per matchup is reported; the check fails when
   any member's interval lies above 50% by more than `--exploit-margin` (`exploitability`).

`--deck-models deck=PATH,...` evaluates per-deck pilots instead: each side plays with its deck's
checkpoint (unlisted decks use the positional checkpoint), and a pool member plays X against Y's pilot.

Win rates are best-of-three match win rates (a tied match counts half), like the reference's
match points: game one with the maindecks, games 2 and 3 sideboarded with the plan table.
`--format game1` plays single game-one games for diagnostics only. Thresholds are uncalibrated
starting values. See docs/representation-plan.md.
"""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from ..engine.cards import SPEC_PATH
from ..engine.decks import DECKS
from . import reference as ref
from . import stats
from .artifacts import DIAGNOSTIC, FAIR, require
from .jsonio import digest, file_digest, write_json
from .runner import run_episodes
from .schedule import bootstrap_seed, episodes

SCHEMA = "TrustTestResult"
VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
STREAM = "trust-v1"
SETTINGS = dict(max_turns=100, max_decisions=10000, auto_single=False, auto_mana=False, auto_pass=False)
DEFAULTS = dict(min_inside=0.70, bias_margin=0.02, exploit_margin=0.02, replicates=2000)
THRESHOLD_STATUS = "uncalibrated starting values"
CHECKS = ("per_pairing", "per_deck", "exploitability")


# --- pure comparison logic -------------------------------------------------------------

def classify(rate, ref_low, ref_high):
    """'inside' when the model's point estimate lies within the reference 95% interval
    [ref_low, ref_high] (bounds inclusive), else 'above' / 'below'. The model's own
    interval is not used here; `overlap` reports separately whether the two intervals meet."""
    return "below" if rate < ref_low else "above" if rate > ref_high else "inside"


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def compare_pairings(cells, reference):
    """Rows per non-mirror pairing. `cells` maps 'a|b' to {score, ci95, games}. Thin or
    missing reference pairings are reported with counted=False."""
    rows = []
    for key, p in reference["pairings"].items():
        a, b = key.split("|")
        if a == b:
            continue
        c = cells.get(key)
        require(c is not None, "cells", f"missing pairing {key}")
        row = {"pairing": key, "deck_a": a, "deck_b": b, "n": c["n"], "model": c["score"], "model_ci95": c["ci95"]}
        if p.get("missing"):
            row.update(reference=None, status="no_reference", counted=False)
        else:
            lo, hi = p["ci_low"], p["ci_high"]
            row.update(reference={"win_rate": p["win_rate"], "ci95": [lo, hi], "n": p["n"], "thin": p["thin"]},
                       difference=c["score"] - p["win_rate"], status=classify(c["score"], lo, hi),
                       overlap=None if c["ci95"] is None else overlaps(c["ci95"], [lo, hi]), counted=not p["thin"])
        rows.append(row)
    return rows


def check_per_pairing(rows, min_inside):
    counted = [r for r in rows if r["counted"]]
    inside = sum(r["status"] == "inside" for r in counted)
    frac = inside / len(counted) if counted else None
    return {"passed": frac is not None and frac >= min_inside, "inside": inside, "counted": len(counted),
            "fraction_inside": frac, "min_inside": min_inside,
            "above": sum(r["status"] == "above" for r in counted), "below": sum(r["status"] == "below" for r in counted)}


def _z(confidence):
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def oriented(values, deck, a):
    """Block values of pairing 'a|b' (deck a's perspective) seen from `deck`."""
    return values if deck == a else [1 - v for v in values]


def deck_biases(block_values, reference, bias_margin, replicates, seed, confidence=0.95, include_thin=False):
    """Per deck: mean signed difference (model - reference) over its pairings.

    The interval resamples whole blocks (shared across pairings, via stats.game_intervals)
    and adds the reference's binomial variance (normal approximation, independent of the
    model's), so it is wider than the model-only interval. A deck is flagged 'over' when
    the interval lies above +bias_margin and 'under' when it lies below -bias_margin."""
    z, out = _z(confidence), {}
    for deck in reference["decks"]:
        used = []
        for key, p in reference["pairings"].items():
            a, b = key.split("|")
            if deck in (a, b) and a != b and not p.get("missing") and (include_thin or not p["thin"]):
                used.append((key, a, p))
        entry = {"pairings": [k for k, _, _ in used], "bias": None, "ci95": None, "flag": None, "status": "unavailable"}
        out[deck] = entry
        if not used:
            continue
        n = len(block_values[used[0][0]])
        per_block = [sum(oriented(block_values[k], deck, a)[i] - (p["win_rate"] if deck == a else 1 - p["win_rate"]) for k, a, p in used) / len(used)
                     for i in range(n)]
        iv = stats.game_intervals({"bias": per_block}, replicates, seed, confidence)
        bias = iv["mean"]["mean"]
        entry["bias"] = bias
        entry["pairings_above_reference"] = sum(
            sum(oriented(block_values[k], deck, a)) / n > (p["win_rate"] if deck == a else 1 - p["win_rate"]) for k, a, p in used)
        if iv["status"] != "ok":
            entry["reason"] = iv.get("reason")
            continue
        se_model = (iv["mean"]["ci95"][1] - iv["mean"]["ci95"][0]) / (2 * z)
        se_ref = math.sqrt(sum(p["win_rate"] * (1 - p["win_rate"]) / p["n"] for _, _, p in used)) / len(used)
        half = z * math.hypot(se_model, se_ref)
        lo, hi = bias - half, bias + half
        entry.update(ci95=[lo, hi], status="ok", flag="over" if lo > bias_margin else "under" if hi < -bias_margin else None,
                     se_model=se_model, se_reference=se_ref)
    return out


def check_per_deck(biases, bias_margin):
    flagged = sorted(d for d, b in biases.items() if b["flag"])
    unavailable = sorted(d for d, b in biases.items() if b["status"] != "ok")
    return {"passed": not flagged and not unavailable, "flagged": flagged, "unavailable": unavailable, "bias_margin": bias_margin}


def exploitability(pool_intervals, exploit_margin):
    """`pool_intervals` maps member -> {'x>y': {score, ci95, games}} from the member's
    perspective (x is the member's deck). Per matchup: the member with the highest win
    rate, its interval, and 'clear' when the interval's lower bound exceeds 0.5 + margin."""
    cells = sorted({c for member in pool_intervals.values() for c in member})
    rows = []
    for cell in cells:
        best = max(((m, v[cell]) for m, v in pool_intervals.items() if cell in v), key=lambda t: t[1]["score"])
        member, v = best
        lo = None if v["ci95"] is None else v["ci95"][0]
        rows.append({"matchup": cell, "member": member, "pool_win_rate": v["score"], "ci95": v["ci95"], "n": v["n"],
                     "clearly_exploitable": lo is not None and lo > 0.5 + exploit_margin})
    clear = [r["matchup"] for r in rows if r["clearly_exploitable"]]
    return rows, {"passed": bool(rows) and not clear, "clearly_exploitable": clear, "exploit_margin": exploit_margin,
                  "pool_size": len(pool_intervals), "max_pool_win_rate": max((r["pool_win_rate"] for r in rows), default=None)}


# --- games ------------------------------------------------------------------------------

class DeckModels:
    """Per-deck models (`--deck-models`): `for_deck(deck)` is the agent factory of that deck's
    pilot. `play_match` asks a factory that has `for_deck` for the factory of the deck a seat
    plays, so cell A vs B is pilot A on deck A against pilot B on deck B. Picklable."""

    def __init__(self, factories):
        self.factories = dict(factories)

    def for_deck(self, deck):
        return self.factories[deck]


def _factory(factory, deck):
    return factory.for_deck(deck) if hasattr(factory, "for_deck") else factory


class RandomLearner:
    """Smoke-test agent (`--agent random`): uniformly random legal actions, deterministic per
    episode. Picklable, one adapter per seat."""
    information_contract = DIAGNOSTIC

    def __call__(self, seat, mode):
        return _RandomAdapter(seat)


class _RandomAdapter:
    information_contract = DIAGNOSTIC

    def __init__(self, seat):
        self.seat, self.rng = seat, random.Random(0)

    def reset(self, own_deck, actor_seed=0):
        self.rng = random.Random(actor_seed)

    def act(self, game):
        return self.rng.randrange(len(game.legal_options()))


def manifest(blocks, modes, cells):
    """cells: [(id, learner_deck, opponent_deck)]. One bot per deck stands for the opponent
    seat; the runner never looks at its name."""
    decks = sorted({d for _, a, b in cells for d in (a, b)})
    return dict(stream=STREAM, modes=list(modes), blocks_per_cell=blocks,
                decks={d: {"cards": dict(DECKS[d])} for d in decks},
                bots=[{"id": f"model:{d}", "deck": d} for d in decks],
                cells=[{"id": c, "learner_deck": a, "opponent": f"model:{b}"} for c, a, b in cells])


def play_match(spec, learner, opponent, engine):
    """One best-of-three on the spec's deal: game 1 with the maindecks and the spec's starting
    player, games 2 and 3 with each seat's table plan (`mtg_ml.match`, as the training and
    `rl.evaluate` best-of-three), the previous game's loser on the play (a draw: the same
    player again), and each model seat carrying what it witnessed into the next game. A row
    shaped like a game row: `winner` is the match winner (None: tied match, scored half)."""
    from collections import Counter
    from ..backend import game_class
    from ..engine import SIDEBOARDS, expand
    from ..engine.sideboard import postboard
    from ..knowledge import EMPTY
    from ..match import game_seed, next_starting_player
    from .adapters import take
    row = {k: v for k, v in asdict(spec).items() if k != "decks"}
    row.update(status="error", winner=None, reason=None, games=[])
    started = time.perf_counter()
    try:
        names = spec.deck_ids
        agents = [None, None]
        agents[spec.learner_seat] = _factory(learner, names[spec.learner_seat])(spec.learner_seat, spec.mode)
        agents[1 - spec.learner_seat] = _factory(opponent, names[1 - spec.learner_seat])(1 - spec.learner_seat, spec.mode)
        knowledge, start, wins = [EMPTY, EMPTY], spec.starting_player, [0, 0]
        for n in (1, 2, 3):
            decks = spec.decks if n == 1 else tuple(tuple(expand(postboard(names[s], names[1 - s]))) for s in (0, 1))
            args = dict(spec.game_args(SETTINGS), decks=decks, match_game=n, seed=game_seed(spec.simulator_seed, n), starting_player=start,
                        registered_sideboards=tuple(tuple(expand(SIDEBOARDS[d])) for d in names))
            g = game_class(engine)(**args)
            for s, a in enumerate(agents):
                a.reset(dict(Counter(decks[s])), spec.actor_seed + n)
                if hasattr(getattr(a, "model", None), "set_knowledge"):
                    a.model.set_knowledge(knowledge[s])
            steps = 0
            while not g.over:
                steps += 1
                if steps > SETTINGS["max_decisions"]:
                    raise RuntimeError("runner_decision_cap")
                take(g, agents, agents[g.decision.player].act(g))
            row["games"].append({"start": start, "winner": g.winner, "reason": g.end_reason, "turns": g.turn})
            if g.winner is not None:
                wins[g.winner] += 1
            for s in (0, 1):
                if hasattr(getattr(agents[s], "model", None), "set_knowledge"):
                    knowledge[s] = knowledge[s].with_game(g.witnessed(s))
            if getattr(g, "NATIVE", False):
                g._cache.clear()
                g._proxies.clear()
            if max(wins) >= 2:
                break
            start = next_starting_player(start, g.winner)
        row.update(status="completed", winner=None if wins[0] == wins[1] else int(wins[1] > wins[0]), reason="match")
    except Exception as e:
        row.update(status="error", reason=str(e), error_type=type(e).__name__)
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def run_matches(specs, learner, opponent, engine, workers):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import gc
    import multiprocessing
    from .runner import _worker_init
    run = partial(play_match, learner=learner, opponent=opponent, engine=engine)
    if workers == 1:
        rows = list(map(run, specs))
    else:
        gc.collect()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
            rows = list(pool.map(run, specs))
    return sorted(rows, key=lambda r: (r["mode"], r["cell"], r["block"], r["slot"]))


def play(blocks, modes, cells, learner, opponent, engine, workers, fmt="bo3"):
    """-> (rows, {mode: {cell: [block value]}}), block values from the learner side. One row per
    best-of-three match (`fmt` bo3) or per game-one game (`fmt` game1, diagnostics only)."""
    specs = episodes(manifest(blocks, modes, cells))
    if fmt == "bo3":
        rows = run_matches(specs, learner, opponent, engine, workers)
    else:
        rows = run_episodes(specs, SETTINGS, learner, opponent, engine=engine, workers=workers)
    errors = [r for r in rows if r["status"] != "completed"]
    require(not errors, "game_rows", f"{len(errors)} games failed, first: {errors[0].get('error_type')} {errors[0]['reason']}" if errors else "")
    return rows, stats.block_scores(rows, [asdict(s) for s in specs])


def cell_summary(rows, mode, cell, interval):
    rs = [r for r in rows if r["mode"] == mode and r["cell"] == cell]
    c = interval["cells"][cell]
    return {"n": len(rs), "wins": sum(r["winner"] == r["learner_seat"] for r in rs), "draws": sum(r["winner"] is None for r in rs),
            "score": c["mean"], "ci95": c["ci95"]}


def pairings():
    return [(f"{a}|{b}", a, b) for i, a in enumerate(ref.DECK_ORDER) for b in ref.DECK_ORDER[i + 1:]]


def ordered_matchups():
    return [(f"{x}>{y}", x, y) for x in ref.DECK_ORDER for y in ref.DECK_ORDER if x != y]


def evaluate(reference, selfplay_rows, selfplay_blocks, pool, thresholds, confidence=0.95):
    """Pure aggregation: raw rows and block values -> per-mode report with the three checks.
    `pool` is {member: (rows, blocks)}; intervals are recomputed here from the blocks."""
    t, report = thresholds, {}
    for mode, cells in selfplay_blocks.items():
        iv = stats.game_intervals(cells, t["replicates"], bootstrap_seed(STREAM, "selfplay", mode), confidence)
        summary = {c: cell_summary(selfplay_rows, mode, c, iv) for c in cells}
        rows = compare_pairings(summary, reference)
        biases = deck_biases(cells, reference, t["bias_margin"], t["replicates"], bootstrap_seed(STREAM, "bias", mode), confidence)
        entry = {"pairings": rows, "decks": biases, "interval": {k: iv[k] for k in ("status", "blocks", "replicates", "confidence", "seed")},
                 "checks": {"per_pairing": check_per_pairing(rows, t["min_inside"]), "per_deck": check_per_deck(biases, t["bias_margin"])}}
        members = {}
        for name, (prow, pblocks) in pool.items():
            if mode in pblocks:
                piv = stats.game_intervals(pblocks[mode], t["replicates"], bootstrap_seed(STREAM, f"pool/{name}", mode), confidence)
                members[name] = {c: cell_summary(prow, mode, c, piv) for c in pblocks[mode]}
        if members:
            entry["exploitability"], entry["checks"]["exploitability"] = exploitability(members, t["exploit_margin"])
        else:
            entry["exploitability"], entry["checks"]["exploitability"] = [], {"passed": None, "reason": "no pool checkpoints given"}
        entry["checks"]["overall"] = overall(entry["checks"])
        report[mode] = entry
    return report


def overall(checks):
    values = [checks[c]["passed"] for c in CHECKS]
    return {"passed": None if None in values else all(values), "complete": None not in values}


# --- result and markdown ----------------------------------------------------------------

def build_result(*, candidate, pool, args_record, engine, report, selfplay_rows, pool_rows, code, runtime, primary):
    return {"format": SCHEMA, "version": VERSION, "schema_version": VERSION, "status": "complete",
            "candidate": candidate, "pool": pool, "code_revision": code["revision"], "code_dirty": code["dirty"],
            "card_spec_sha256": file_digest(SPEC_PATH), "engine": engine,
            "reference": {"path": str(ref.DEFAULT_PATH.relative_to(ROOT)), "sha256": file_digest(ref.DEFAULT_PATH)},
            "settings": SETTINGS, "stream": STREAM, "interval_method": stats.INTERVAL_METHOD, "parameters": args_record,
            "primary_mode": primary, "runtime": runtime, "modes": report,
            "game_rows": {"selfplay": selfplay_rows, "pool": pool_rows}, "rows_sha256": digest([selfplay_rows, pool_rows])}


def markdown(result):
    mode = result["primary_mode"]
    m = result["modes"][mode]
    c = m["checks"]
    word = {True: "PASS", False: "FAIL", None: "NOT RUN"}
    cand = result["candidate"]
    who = (f"checkpoint sha256 `{cand['sha256']}`, features {cand['features']}" if cand.get("kind") != "per_deck"
           else "per-deck models (sha256 per deck below)")
    out = [f"# Trust test: {cand['checkpoint']}", "",
           f"{who}, code `{result['code_revision'][:10]}`, engine {result['engine']}, "
           f"{result['parameters']['matches']} {result['parameters']['unit']}(s) per pairing, format {result['parameters']['format']}, "
           f"primary mode {mode}; thresholds are {THRESHOLD_STATUS}", "",
           f"**Overall: {word[c['overall']['passed']]}**"
           + ("" if result["parameters"]["format"] == "bo3" else " (DIAGNOSTIC: game one only, not a trust verdict)"), "",
           "| check | result | detail |", "|---|---|---|",
           f"| per pairing | {word[c['per_pairing']['passed']]} | {c['per_pairing']['inside']}/{c['per_pairing']['counted']} inside "
           f"(need {c['per_pairing']['min_inside']:.0%}); {c['per_pairing']['above']} above, {c['per_pairing']['below']} below |",
           f"| per deck | {word[c['per_deck']['passed']]} | flagged: {', '.join(c['per_deck']['flagged']) or 'none'} (margin {c['per_deck']['bias_margin']}) |"]
    e = c["exploitability"]
    out.append(f"| exploitability | {word[e['passed']]} | " + (e.get("reason") or f"clearly exploitable: {', '.join(e['clearly_exploitable']) or 'none'}; "
               f"max pool win rate {e['max_pool_win_rate']:.3f} (margin {e['exploit_margin']})") + " |")
    if cand.get("kind") == "per_deck":
        out += ["", "## Pilots", "", "| deck | checkpoint | sha256 | features | source |", "|---|---|---|---|---|"]
        out += [f"| {d} | {r['checkpoint']} | `{r['sha256']}` | {r['features']} | {r['source']} |" for d, r in cand["decks"].items()]
    out += ["", "## Pairings", "", "| pairing | model | reference | status | n ref |", "|---|---|---|---|---|"]
    for r in m["pairings"]:
        rf = r["reference"]
        model = f"{r['model']:.3f} [{r['model_ci95'][0]:.3f}, {r['model_ci95'][1]:.3f}]" if r["model_ci95"] else f"{r['model']:.3f}"
        ref_text = "none" if rf is None else f"{rf['win_rate']:.3f} [{rf['ci95'][0]:.3f}, {rf['ci95'][1]:.3f}]"
        flag = r["status"] + ("" if r["counted"] else " (not counted)")
        out.append(f"| {r['pairing']} | {model} | {ref_text} | {flag} | {rf['n'] if rf else 0} |")
    out += ["", "## Deck bias (model - reference)", "", "| deck | bias | 95% interval | flag |", "|---|---|---|---|"]
    for d, b in m["decks"].items():
        ci = "n/a" if b["ci95"] is None else f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
        out.append(f"| {d} | {'n/a' if b['bias'] is None else format(b['bias'], '+.3f')} | {ci} | {b['flag'] or ''} |")
    if m["exploitability"]:
        out += ["", "## Best pool member per matchup", "", "| matchup | member | pool win rate | clear |", "|---|---|---|---|"]
        for r in m["exploitability"]:
            ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r["ci95"] else ""
            out.append(f"| {r['matchup']} | {Path(r['member']).stem} | {r['pool_win_rate']:.3f} {ci} | {'yes' if r['clearly_exploitable'] else ''} |")
    others = [k for k in result["modes"] if k != mode]
    if others:
        out += ["", "Other modes: " + "; ".join(f"{k}: " + ", ".join(f"{n} {word[v['passed']]}" for n, v in result['modes'][k]['checks'].items() if n != 'overall') for k in others)]
    return "\n".join(out) + "\n"


# --- CLI --------------------------------------------------------------------------------

def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _learner(path, contract, agent):
    if agent == "random":
        return RandomLearner(), {"kind": "random", "checkpoint": "random", "sha256": None, "features": None, "information_contract": DIAGNOSTIC}
    from .release import checkpoint_learner
    learner = checkpoint_learner(path, contract)
    return learner.games, dict(learner.record, checkpoint=str(path))


def parse_deck_models(text):
    """'deck=PATH,deck=PATH' -> {deck: Path}. Unknown decks, repeats and empty paths are errors."""
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        deck, sep, path = item.partition("=")
        deck, path = deck.strip(), path.strip()
        require(sep and path, "deck_models", f"expected deck=PATH, got {item!r}")
        require(deck in ref.DECK_ORDER, "deck_models", f"unknown deck {deck!r}; use one of {list(ref.DECK_ORDER)}")
        require(deck not in out, "deck_models", f"deck {deck} given twice")
        out[deck] = Path(path).expanduser()
    return out


def deck_pilots(deck_paths, fallback, contract, load=None):
    """-> (DeckModels, candidate record) for every deck: the listed checkpoint, else `fallback`.
    Each distinct file is loaded and admitted once (`load(path)` -> (factory, record), default
    `_learner`). The record has the per-deck records, so trust.json and trust.md carry every
    checkpoint's sha."""
    load = load or (lambda path: _learner(path, contract, "checkpoint"))
    missing = [d for d in ref.DECK_ORDER if d not in deck_paths and fallback is None]
    require(not missing, "deck_models", f"no checkpoint for {', '.join(missing)}: list them or give the positional checkpoint")
    loaded, decks, factories = {}, {}, {}
    for d in ref.DECK_ORDER:
        path = deck_paths.get(d, fallback)
        if str(path) not in loaded:
            loaded[str(path)] = load(path)
        factories[d], record = loaded[str(path)]
        decks[d] = dict(record, source="deck-models" if d in deck_paths else "fallback")
    shas = {d: r["sha256"] for d, r in decks.items()}
    return DeckModels(factories), {"kind": "per_deck", "checkpoint": "per-deck models", "sha256": digest(shas), "features": None, "decks": decks,
                                   "fallback": None if fallback is None else str(fallback), "information_contract": contract}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, nargs="?", help="the final model (omit with --agent random)")
    ap.add_argument("--deck-models", default=None, metavar="DECK=PATH,...",
                    help="per-deck pilots, e.g. jund_wildfire=a.pt,mono_blue_terror=b.pt: each deck plays with its own checkpoint, so cell A vs B is "
                         "pilot A on deck A against pilot B on deck B, and a pool member plays X against Y's pilot. Decks not listed use the "
                         "positional checkpoint (required then). bo3 only")
    ap.add_argument("--pool", nargs="*", default=[], type=Path, help="frozen-pool checkpoints that play the final model")
    ap.add_argument("--matches", type=int, default=400, help="matches per pairing and mode, rounded up to four-match blocks (default 400)")
    ap.add_argument("--exploit-matches", type=int, default=200, help="matches per ordered matchup, mode and pool member (default 200)")
    ap.add_argument("--format", choices=("bo3", "game1"), default="bo3",
                    help="bo3: sideboarded best-of-three matches scored as match wins, draws half (default; the only format the checks are "
                         "defined for). game1: single maindeck games, diagnostics only")
    ap.add_argument("--modes", default="sampled,greedy")
    ap.add_argument("--primary-mode", default="sampled", help="mode the markdown summary and exit code use (default sampled)")
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--contract", choices=("fair", "diagnostic"), default="fair")
    ap.add_argument("--agent", choices=("checkpoint", "random"), default="checkpoint", help="random is a smoke-test agent")
    ap.add_argument("--min-inside", type=float, default=DEFAULTS["min_inside"],
                    help="per-pairing check: fraction of non-thin pairings whose model win rate lies in the reference 95%% interval (default 0.70)")
    ap.add_argument("--bias-margin", type=float, default=DEFAULTS["bias_margin"],
                    help="per-deck check: flag a deck when its bias interval excludes 0 by more than this (default 0.02)")
    ap.add_argument("--exploit-margin", type=float, default=DEFAULTS["exploit_margin"],
                    help="exploitability: fail when a pool member's interval lower bound exceeds 0.5 + this (default 0.02)")
    ap.add_argument("--replicates", type=int, default=DEFAULTS["replicates"], help="bootstrap replicates (default 2000)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    modes = args.modes.split(",")
    if (args.agent == "checkpoint") == (args.checkpoint is None and args.deck_models is None) or args.matches < 1 or args.exploit_matches < 1 or args.workers < 1:
        ap.error("give a checkpoint (or --agent random), and positive matches/workers")
    if args.primary_mode not in modes or not set(modes) <= {"sampled", "greedy"}:
        ap.error("--primary-mode must be one of --modes (sampled, greedy)")
    if args.deck_models is not None and (args.agent != "checkpoint" or args.format != "bo3"):
        ap.error("--deck-models needs --agent checkpoint and --format bo3")
    contract = {"fair": FAIR, "diagnostic": DIAGNOSTIC}[args.contract]
    reference = ref.load()
    if args.deck_models is None:
        final, candidate = _learner(args.checkpoint, contract, args.agent)
    else:
        try:
            final, candidate = deck_pilots(parse_deck_models(args.deck_models), args.checkpoint, contract)
        except ValueError as e:
            ap.error(str(e))
    pool_learners = {str(p): _learner(p, contract, "checkpoint") for p in args.pool}
    blocks, exploit_blocks = -(-args.matches // 4), -(-args.exploit_matches // 4)
    code = {"revision": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "mtg_ml", "native", "tools"))}
    started = time.monotonic()
    unit = "match" if args.format == "bo3" else "game"
    plural = unit + ("es" if unit == "match" else "s")

    rows, block_values = play(blocks, modes, pairings(), final, final, args.engine, args.workers, args.format)
    print(f"self-play: {len(rows)} {plural} in {time.monotonic() - started:.0f}s", flush=True)
    pool, pool_rows = {}, {}
    for name, (learner, _) in pool_learners.items():
        t0 = time.monotonic()
        prow, pblocks = play(exploit_blocks, modes, ordered_matchups(), learner, final, args.engine, args.workers, args.format)
        pool[name], pool_rows[name] = (prow, pblocks), prow
        print(f"pool {Path(name).name}: {len(prow)} {plural} in {time.monotonic() - t0:.0f}s", flush=True)

    thresholds = {k: getattr(args, k) for k in DEFAULTS}
    report = evaluate(reference, rows, block_values, pool, thresholds)
    runtime = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "workers": args.workers,
               "elapsed_seconds": time.monotonic() - started, "command": [sys.executable, "-m", "mtg_ml.benchmark.trust", *(argv if argv is not None else sys.argv[1:])]}
    record = {"format": args.format, "unit": unit, "matches": blocks * 4, "exploit_matches": exploit_blocks * 4, "blocks": blocks, "exploit_blocks": exploit_blocks,
              "modes": modes, "thresholds": thresholds, "threshold_status": THRESHOLD_STATUS, "contract": contract}
    result = build_result(candidate=candidate, pool=[dict(r, checkpoint=n) for n, (_, r) in pool_learners.items()], args_record=record,
                          engine=args.engine, report=report, selfplay_rows=rows, pool_rows=pool_rows, code=code, runtime=runtime,
                          primary=args.primary_mode)
    stem = "trust"
    write_json(args.out / f"{stem}.json", result)
    (args.out / f"{stem}.md").write_text(markdown(result))
    print(markdown(result))
    return 0 if report[args.primary_mode]["checks"]["overall"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
