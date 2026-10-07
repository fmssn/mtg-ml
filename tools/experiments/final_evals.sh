#!/bin/bash
# Standard post-run evaluation of archived runs: 2000-game benchmark sampled + greedy, appended to <id>/evals.txt.
# Usage: final_evals.sh <code dir> <id>... ; extra env: H2H="<id> <control id>" pairs separated by ';'
A=~/mtg-ml-checkpoints; code=$1; shift
cd "$code" || exit 1
E="nice -n 10 env MTG_ENGINE=native PYTHONPATH=. OMP_NUM_THREADS=1 $HOME/mtg-ml-v3/.venv/bin/python -m mtg_ml.rl.evaluate"
for id in "$@"; do
  for m in "" --greedy; do
    r=$($E $A/$id/policy.pt bot --jund --games 2000 --workers 12 $m | head -1)
    echo "benchmark 2000 ${m:-sampled} ($(basename $code) code): $r" | tee -a $A/$id/evals.txt
  done
done
IFS=';' read -ra PAIRS <<< "$H2H"; for pair in "${PAIRS[@]}"; do read a b <<< "$pair"; [ -z "$a" ] && continue
  r=$($E $A/$a/policy.pt $A/$b/policy.pt --games 2000 --workers 12 | tail -1)
  echo "head to head vs $b, 2000 paired games: $r" | tee -a $A/$a/evals.txt; done
