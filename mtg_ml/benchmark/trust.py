"""The trust test (step 7 of docs/representation-plan.md): is a checkpoint fit for card choices?

    python -m mtg_ml.benchmark.trust CKPT.pt --games 400 --workers 32 \
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

Win rates are game win rates (draws count half) of game one without sideboarding; the
reference is match points of mostly best-of-three. See docs/representation-plan.md.
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
        row = {"pairing": key, "deck_a": a, "deck_b": b, "games": c["games"], "model": c["score"], "model_ci95": c["ci95"]}
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
        rows.append({"matchup": cell, "member": member, "pool_win_rate": v["score"], "ci95": v["ci95"], "games": v["games"],
                     "clearly_exploitable": lo is not None and lo > 0.5 + exploit_margin})
    clear = [r["matchup"] for r in rows if r["clearly_exploitable"]]
    return rows, {"passed": bool(rows) and not clear, "clearly_exploitable": clear, "exploit_margin": exploit_margin,
                  "pool_size": len(pool_intervals), "max_pool_win_rate": max((r["pool_win_rate"] for r in rows), default=None)}


# --- games ------------------------------------------------------------------------------

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


def play(blocks, modes, cells, learner, opponent, engine, workers):
    """-> (specs, rows, {mode: {cell: [block value]}}), block values from the learner side."""
    specs = episodes(manifest(blocks, modes, cells))
    rows = run_episodes(specs, SETTINGS, learner, opponent, engine=engine, workers=workers)
    errors = [r for r in rows if r["status"] != "completed"]
    require(not errors, "game_rows", f"{len(errors)} games failed, first: {errors[0].get('error_type')} {errors[0]['reason']}" if errors else "")
    return rows, stats.block_scores(rows, [asdict(s) for s in specs])


def cell_summary(rows, mode, cell, interval):
    rs = [r for r in rows if r["mode"] == mode and r["cell"] == cell]
    c = interval["cells"][cell]
    return {"games": len(rs), "wins": sum(r["winner"] == r["learner_seat"] for r in rs), "draws": sum(r["winner"] is None for r in rs),
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
    out = [f"# Trust test: {cand['checkpoint']}", "",
           f"checkpoint sha256 `{cand['sha256']}`, features {cand['features']}, code `{result['code_revision'][:10]}`, engine {result['engine']}, "
           f"{result['parameters']['games']} games per pairing, primary mode {mode}", "",
           f"**Overall: {word[c['overall']['passed']]}**", "",
           "| check | result | detail |", "|---|---|---|",
           f"| per pairing | {word[c['per_pairing']['passed']]} | {c['per_pairing']['inside']}/{c['per_pairing']['counted']} inside "
           f"(need {c['per_pairing']['min_inside']:.0%}); {c['per_pairing']['above']} above, {c['per_pairing']['below']} below |",
           f"| per deck | {word[c['per_deck']['passed']]} | flagged: {', '.join(c['per_deck']['flagged']) or 'none'} (margin {c['per_deck']['bias_margin']}) |"]
    e = c["exploitability"]
    out.append(f"| exploitability | {word[e['passed']]} | " + (e.get("reason") or f"clearly exploitable: {', '.join(e['clearly_exploitable']) or 'none'}; "
               f"max pool win rate {e['max_pool_win_rate']:.3f} (margin {e['exploit_margin']})") + " |")
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, nargs="?", help="the final model (omit with --agent random)")
    ap.add_argument("--pool", nargs="*", default=[], type=Path, help="frozen-pool checkpoints that play the final model")
    ap.add_argument("--games", type=int, default=400, help="games per pairing and mode, rounded up to four-game blocks (default 400)")
    ap.add_argument("--exploit-games", type=int, default=200, help="games per ordered matchup, mode and pool member (default 200)")
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
    if (args.agent == "checkpoint") == (args.checkpoint is None) or args.games < 1 or args.exploit_games < 1 or args.workers < 1:
        ap.error("give a checkpoint (or --agent random), and positive games/workers")
    if args.primary_mode not in modes or not set(modes) <= {"sampled", "greedy"}:
        ap.error("--primary-mode must be one of --modes (sampled, greedy)")
    contract = {"fair": FAIR, "diagnostic": DIAGNOSTIC}[args.contract]
    reference = ref.load()
    final, candidate = _learner(args.checkpoint, contract, args.agent)
    pool_learners = {str(p): _learner(p, contract, "checkpoint") for p in args.pool}
    blocks, exploit_blocks = -(-args.games // 4), -(-args.exploit_games // 4)
    code = {"revision": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "mtg_ml", "native", "tools"))}
    started = time.monotonic()

    rows, block_values = play(blocks, modes, pairings(), final, final, args.engine, args.workers)
    print(f"self-play: {len(rows)} games in {time.monotonic() - started:.0f}s", flush=True)
    pool, pool_rows = {}, {}
    for name, (learner, _) in pool_learners.items():
        t0 = time.monotonic()
        prow, pblocks = play(exploit_blocks, modes, ordered_matchups(), learner, final, args.engine, args.workers)
        pool[name], pool_rows[name] = (prow, pblocks), prow
        print(f"pool {Path(name).name}: {len(prow)} games in {time.monotonic() - t0:.0f}s", flush=True)

    thresholds = {k: getattr(args, k) for k in DEFAULTS}
    report = evaluate(reference, rows, block_values, pool, thresholds)
    runtime = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "workers": args.workers,
               "elapsed_seconds": time.monotonic() - started, "command": [sys.executable, "-m", "mtg_ml.benchmark.trust", *(argv if argv is not None else sys.argv[1:])]}
    record = {"games": blocks * 4, "exploit_games": exploit_blocks * 4, "blocks": blocks, "exploit_blocks": exploit_blocks,
              "modes": modes, "thresholds": thresholds, "contract": contract}
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
