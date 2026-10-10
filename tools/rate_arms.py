"""Rate finished policies on the fixed L1 ladder (the same mapping as r8_campaign.ladder()).

    python tools/rate_arms.py --ladder-src ~/mtg-ml/ladder-src --games 800 --workers 24 --out l1.json arm=policy.pt ...

Each policy plays every rung on paired seeds (`evaluate.ladder_eval`, sampled); prints and writes
the Elo on L1's scale with its standard error and the per-rung scores.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import r8_campaign as r8  # noqa: E402
from mtg_ml.rl.evaluate import ladder_eval  # noqa: E402
from mtg_ml.rl.rollout import create_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-src", required=True)
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("policies", nargs="+", help="name=path")
    a = ap.parse_args()
    src = Path(a.ladder_src).expanduser()
    original = json.loads((src / "ladder.json").read_text())
    ratings = {}
    for it, rating in r8.RUNG_RATINGS.items():
        name = f"iter_{it:05d}.pt"
        assert [v for k, v in original["ratings"].items() if Path(k).name == name] == [rating]
        ratings[str(src / name)] = rating
    out = {}
    with create_pool(a.workers) as procs:
        for item in a.policies:
            name, path = item.split("=")
            res = ladder_eval(procs, path, ratings, a.games, a.workers)
            out[name] = res
            print(name, res["ladder/elo"], "+-", res["ladder/elo_se"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
