#!/usr/bin/env bash
# Periodic CPU-light specialist check for one r8 run. No respawn; exits once the trainer's process.json says it is not running.
#   [CELLS=a,b CPUS=36-39] r8_specialist_watch.sh RUN_DIR [CODE] [EVERY_GAMES=500000]
# Every EVERY_GAMES games: tools/benchmark_checkpoint.py on the newest COMPLETED policy file (not modified in the last 90 s),
# 25 four-game blocks per cell and mode, 4 workers at nice 19. Writes specialist-check/{watch.log,bench-<games>/,meta-<games>.json}.
set -u
run=$(cd "${1:?run dir}" && pwd)
code=${2:-$HOME/mtg-ml-r8/code}
every=${3:-500000}
cells=${CELLS:-jund_vs_sjund,jund_vs_lblue}
cpus=${CPUS:-36-39}
out="$run/specialist-check"
mkdir -p "$out"
log="$out/watch.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
exec 9>"$out/watch.lock"; flock -n 9 || { say "another watcher holds the lock; exiting"; exit 0; }
say "start run=$run cells=$cells every=$every"
last=0
missing=0
while true; do
    games=$(tail -n1 "$run/metrics.jsonl" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["games_total"])' 2>/dev/null || echo 0)
    due=$((games / every * every))
    if [ "$due" -gt "$last" ]; then
        # policy/ keeps ~3 files at a 15-30 s cadence; the newest may still be written, so take the second newest if >20 s old
        policy=$(find "$run/policy" -name '*.pt' 2>/dev/null | sort | tail -n2 | head -n1)
        [ "$(find "$run/policy" -name '*.pt' 2>/dev/null | wc -l)" -ge 2 ] && [ -n "$policy" ] && [ -n "$(find "$policy" -mmin +0.3)" ] || policy=""
        if [ -n "$policy" ]; then
            tag=$(printf '%09d' "$games")
            sha=$(sha256sum "$policy" | cut -c1-16)
            printf '{"games": %s, "policy": "%s", "sha256_16": "%s"}\n' "$games" "$(basename "$policy")" "$sha" > "$out/meta-$tag.json"
            cp "$policy" "$out/policy-$tag.pt"
            say "bench games=$games policy=$(basename "$policy") sha=$sha"
            (cd "$code" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$cpus" nice -n 19 .venv/bin/python tools/benchmark_checkpoint.py \
                "$out/policy-$tag.pt" --cells "$cells" --blocks 25 --workers 4 --engine native \
                --contract fair --out "$out/bench-$tag") >> "$log" 2>&1
            say "bench done rc=$?"
            last=$due
        fi
    fi
    if [ "$(stat -c %s "$log")" -gt 1000000 ]; then tail -c 200000 "$log" > "$log.tmp" && mv "$log.tmp" "$log"; fi
    # exit only when process.json exists and says not running; tolerate it being absent/unreadable for ~10 min
    if [ -f "$run/process.json" ]; then
        missing=0
        grep -q '"status": "running"' "$run/process.json" || { say "trainer not running; exiting"; break; }
    else
        missing=$((missing + 1)); [ "$missing" -gt 5 ] && { say "process.json missing; exiting"; break; }
    fi
    sleep 120
done
