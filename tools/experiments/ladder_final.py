"""Ladder L1 Elo of archived checkpoints: python ladder_final.py <id>... (run from a code tree with PYTHONPATH=.)"""
import json, os, sys
from mtg_ml.rl.evaluate import ladder_eval
from mtg_ml.rl.rollout import create_pool
A = os.path.expanduser("~/mtg-ml-checkpoints")
L = json.load(open(f"{A}/ladder/L1/ladder.json"))["ratings"]
ratings = {f"{A}/ladder/L1/{os.path.basename(k)}": v for k, v in L.items()}
def main():
  with create_pool(int(os.environ.get("WORKERS", "12"))) as procs:
    for i in sys.argv[1:]:
        res = ladder_eval(procs, f"{A}/{i}/policy.pt", ratings, 200, int(os.environ.get("WORKERS", "12")))
        line = f"ladder L1 (200 paired games per rung, sampled): elo {res['ladder/elo']:.1f} ± {res['ladder/elo_se']:.1f}; " + ", ".join(f"{os.path.basename(k.split('/')[-1])} {v:.3f}" for k, v in res.items() if k.startswith("ladder/") and not k.endswith(("_ci", "elo", "elo_se")))
        print(i, line, flush=True)
        open(f"{A}/{i}/evals.txt", "a").write(line + "\n")


if __name__ == "__main__":
    main()
