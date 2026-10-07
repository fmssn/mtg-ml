#!/bin/bash
# Archive one finished training run into ~/mtg-ml-checkpoints/<id>/ (append-only; never delete from there).
# Usage: archive_run.sh <id> <run dir> <code ref> <launch script> [note]
set -e
id=$1 run=$2 ref=$3 launch=$4 note=${5:-}
A=~/mtg-ml-checkpoints/$id
[ -e $A/final.pt ] && { echo "$id already archived"; exit 0; }
mkdir -p $A/snapshots
cp $run/latest.pt $A/final.pt
p=$(ls $run/policy/*.pt 2>/dev/null | sort | tail -1); [ -n "$p" ] && cp $p $A/policy.pt
cp $run/metrics.jsonl $A/
[ -f $run.log ] && grep -v Warning $run.log > $A/run.log || true
# pool snapshots this run made (the copied-in parent pool is not duplicated)
first=$(head -1 $run/metrics.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['iteration'])")
for f in $run/pool/iter_*.pt; do n=$(basename $f .pt); n=$((10#${n#iter_})); [ $n -ge $first ] && cp $f $A/snapshots/ || true; done
{ echo "id: $id"; echo "archived: $(date -Is)"; echo "run_dir: $(realpath $run)"; echo "code: $ref"; echo "note: $note"
  echo "launch script: $launch"; echo "--- launch lines from run.log"; grep "=== start" $run.log 2>/dev/null || true
  echo "--- launch script"; cat $launch 2>/dev/null || true; } > $A/launch.txt
sha256sum $A/final.pt > $A/SHA256
echo "archived $id: $(du -sh $A | cut -f1)"
