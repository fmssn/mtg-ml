#!/usr/bin/env bash
# Periodic CPU-light specialist check for one run (r8 mirror arm). No respawn; exits when the trainer is no longer running.
#   r8_specialist_watch.sh RUN_DIR [CODE] [EVERY_GAMES=500000]
# Every EVERY_GAMES games: tools/benchmark_checkpoint.py on the newest policy file, cells jund_vs_sjund (Jund vs the Jund
# specialist) and jund_vs_lblue, 25 four-game blocks per cell and mode, 4 workers on the run's eval CPUs at nice 19.
set -u
run=$(cd "${1:?run dir}" && pwd)
code=${2:-$HOME/mtg-ml-r8/code}
every=${3:-500000}
out="$run/specialist-check"
mkdir -p "$out"
last=0
while true; do
    games=$(tail -n1 "$run/metrics.jsonl" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["games_total"])' 2>/dev/null || echo 0)
    due=$((games / every * every))
    if [ "$due" -gt "$last" ]; then
        policy=$(ls "$run"/policy/*.pt 2>/dev/null | sort | tail -n1)
        if [ -n "$policy" ]; then
            tag=$(printf '%09d' "$games")
            cp "$policy" "$out/policy-$tag.pt"
            (cd "$code" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 36-39 nice -n 19 .venv/bin/python tools/benchmark_checkpoint.py \
                "$out/policy-$tag.pt" --cells jund_vs_sjund,jund_vs_lblue --blocks 25 --workers 4 --engine native \
                --contract fair --out "$out/bench-$tag") >> "$out/watch.log" 2>&1
            [ "$(stat -c %s "$out/watch.log")" -gt 1000000 ] && tail -c 200000 "$out/watch.log" > "$out/watch.log.tmp" && mv "$out/watch.log.tmp" "$out/watch.log"
            last=$due
        fi
    fi
    grep -q '"status": "running"' "$run/process.json" 2>/dev/null || break
    sleep 120
done
