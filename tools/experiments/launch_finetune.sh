#!/bin/bash
# Next-actions fine-tunes (2026-10-07). Usage: launch_finetune.sh <name> <cpus> <gpus by PCI index: train,server> <mode: ft|fresh> [extra flags]
# ft: resume a copy of the overnight self-play checkpoint + pool (8.4M games) for 1M more games.
cd ~/mtg-ml-next2 || exit 1; mkdir -p runs
PY=~/mtg-ml-v3/.venv/bin/python
SRC=${SRC:-$HOME/mtg-ml-overnight/runs/overnight-selfplay}
name=$1 cpus=$2 gpus=$3 mode=$4; shift 4
if [ "$mode" = ft ] && [ ! -e runs/$name/latest.pt ]; then
  mkdir -p runs/$name/pool && cp $SRC/latest.pt runs/$name/ && cp $SRC/pool/*.pt runs/$name/pool/
fi
L=ladder/iter_00488.pt,ladder/iter_01464.pt,ladder/iter_02440.pt,ladder/iter_03904.pt
COMMON="--hidden 128 --trunk entity --value-net shared --engine native --device cuda --inference server --server-device cuda:1 \
 --workers 13 --games-per-iter 2048 --iterations 10000000 --checkpoint-every 25 --snapshot-every 122 --seed 1 --server-policy-slots 256 \
 --eval-every 0 --eval-every-games 250000 --bench-games 1000 --bench-greedy-games 1000 --bench-bo3-matches 0 --eval-games 40 --eval-bo3-matches 0 \
 --ladder $L --ladder-ratings ladder/ladder.json --ladder-games 200"
echo "=== start $(date -Is) $mode $*" >> runs/$name.log
OMP_NUM_THREADS=8 PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup taskset -c $cpus $PY -m mtg_ml.rl.train --run runs/$name $COMMON "$@" >> runs/$name.log 2>&1 < /dev/null &
echo "launched $name pid $!"
