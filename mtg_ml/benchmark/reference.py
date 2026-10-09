"""Real-world matchup reference data for the trust test (Pauper-Research).

The file ``data/reference/pauper_research_q3_2026_matchups.json`` is generated from a
Pauper-Research checkout by ``python -m mtg_ml.benchmark.reference build``. Pure Python:
no engine, no torch. Each unordered pairing is stored once, in DECK_ORDER, from the first
deck's perspective. "wins" are match points (a draw counts as half), as in the source.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
DECK_ORDER = ("jund_wildfire", "mono_blue_terror", "red_madness", "grixis_affinity", "elves", "tron")
# Pauper-Research archetype name -> our deck key. Jund is ambiguous (see "notes" in the file).
ARCHETYPES = {
    "jund_wildfire": "Jund Midrange",
    "mono_blue_terror": "Mono Blue Terror",
    "red_madness": "Red Madness",
    "grixis_affinity": "Grixis Affinity",
    "elves": "Elves",
    "tron": "Tron",
}
MIN_MATCHES = 30  # below this the 95% interval is wider than about +-17 points
Z95 = 1.959963984540054
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "pauper_research_q3_2026_matchups.json"
SOURCE_CSV = "reports/2026-Q3/combined/matchups.csv"
INTERVAL_METHOD = "wilson, z=1.959963984540054, successes=wins (draws as half), trials=n"


def wilson_interval(wins: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= wins <= n:
        raise ValueError("wins must be within [0, n]")
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _row(wins: float, n: int) -> dict:
    lo, hi = wilson_interval(wins, n)
    return {"wins": wins, "losses": n - wins, "n": n, "win_rate": wins / n, "ci_low": lo, "ci_high": hi,
            "thin": n < MIN_MATCHES}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()


def build(source_root, date_range=("2026-07-01", "2026-09-30")) -> dict:
    root = Path(source_root)
    rows = {}
    with open(root / SOURCE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["archetype"], r["opp_archetype"])] = (float(r["wins"]), int(r["matches"]))
    pairings = {}
    for i, a in enumerate(DECK_ORDER):
        for b in DECK_ORDER[i:]:
            ra, rb = rows.get((ARCHETYPES[a], ARCHETYPES[b])), rows.get((ARCHETYPES[b], ARCHETYPES[a]))
            key = pair_key(a, b)
            if a == b or ra is None:
                pairings[key] = {"n": 0, "thin": True, "missing": True}
                continue
            if rb is not None and (ra[1] != rb[1] or abs(ra[0] + rb[0] - ra[1]) > 1e-9):
                raise ValueError(f"asymmetric source rows for {key}: {ra} vs {rb}")
            pairings[key] = _row(*ra)
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Pauper-Research Q3 2026 combined matchup results for our six decks, for the trust test.",
        "source": {"repo": _git(root, "remote", "get-url", "origin") or "fmssn/Pauper-Research",
                   "commit": _git(root, "rev-parse", "HEAD"), "path": SOURCE_CSV,
                   "date_start": date_range[0], "date_end": date_range[1],
                   "scope": "combined (CardsRealm paper, MTGO, MTGmelee paper); non-mirror matches"},
        "interval": {"method": INTERVAL_METHOD, "confidence": 0.95},
        "min_matches": MIN_MATCHES,
        "decks": list(DECK_ORDER),
        "archetypes": ARCHETYPES,
        "orientation": "pairings keyed 'a|b' in deck order; wins/win_rate are from a's perspective",
        "notes": [
            "The source does not separate draws: wins are match points with draws as half, so draws is unknown and losses = n - wins.",
            "Our jund_wildfire list is mapped to the source archetype 'Jund Midrange'; the source has no 'Jund Wildfire' label, so the mapping is a judgment call and that deck's row carries the most mapping uncertainty.",
            "The source reports no mirror matches for these archetypes; mirrors are marked missing.",
            "Pairings with n below min_matches are flagged thin, not dropped.",
        ],
        "pairings": pairings,
    }


def validate(data: dict) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {data.get('schema_version')!r}")
    if tuple(data.get("decks", ())) != DECK_ORDER:
        raise ValueError("decks must equal DECK_ORDER")
    for k in ("source", "interval", "archetypes", "min_matches"):
        if k not in data:
            raise ValueError(f"missing {k}")
    for k in ("repo", "commit", "path", "date_start", "date_end"):
        if not data["source"].get(k):
            raise ValueError(f"source.{k} missing")
    expected = {pair_key(a, b) for i, a in enumerate(DECK_ORDER) for b in DECK_ORDER[i:]}
    if set(data["pairings"]) != expected:
        raise ValueError("pairings must cover every unordered pair and mirror exactly once")
    for key, p in data["pairings"].items():
        if p.get("missing"):
            if p.get("n") != 0 or not p.get("thin"):
                raise ValueError(f"{key}: a missing pairing needs n=0 and thin=true")
            continue
        n, wins = p["n"], p["wins"]
        if not (isinstance(n, int) and n > 0 and 0 <= wins <= n):
            raise ValueError(f"{key}: bad wins/n")
        if abs(p["losses"] - (n - wins)) > 1e-9 or abs(p["win_rate"] - wins / n) > 1e-9:
            raise ValueError(f"{key}: inconsistent losses or win_rate")
        lo, hi = wilson_interval(wins, n)
        if abs(p["ci_low"] - lo) > 1e-9 or abs(p["ci_high"] - hi) > 1e-9:
            raise ValueError(f"{key}: interval does not match the Wilson method")
        if p["thin"] != (n < data["min_matches"]):
            raise ValueError(f"{key}: thin flag inconsistent with min_matches")


def load(path=DEFAULT_PATH) -> dict:
    data = json.loads(Path(path).read_text())
    validate(data)
    return data


def matrix(data: dict) -> dict:
    """Win rate of the row deck against the column deck, both orders; None where missing."""
    out = {a: {b: None for b in DECK_ORDER} for a in DECK_ORDER}
    for key, p in data["pairings"].items():
        a, b = key.split("|")
        if p.get("missing"):
            continue
        out[a][b] = p["win_rate"]
        if a != b:
            out[b][a] = 1 - p["win_rate"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="regenerate the reference file from a Pauper-Research checkout")
    b.add_argument("--source", required=True, help="path to the Pauper-Research checkout")
    b.add_argument("--out", default=str(DEFAULT_PATH))
    sub.add_parser("check", help="validate the committed file").add_argument("--path", default=str(DEFAULT_PATH))
    args = ap.parse_args(argv)
    if args.cmd == "build":
        data = build(args.source)
        validate(data)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    else:
        load(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
