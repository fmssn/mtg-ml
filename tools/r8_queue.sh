#!/usr/bin/env bash
# Sequential on-box queue for one slot of an r8_campaign.py campaign: starts each arm's trainer (r8_host.sh, tmux session
# mtg-r8-<arm>), waits until that session ends, logs the outcome, starts the next. Run it inside its own tmux session so it
# survives the launching session; it never restarts a failed or half-run arm (the trainers' own preflight also refuses to
# start on occupied GPUs or overlapping CPUs).
#   CODE=~/mtg-ml-r8/code-r3 r8_queue.sh CAMPAIGN_DIR ARM_NAME [ARM_NAME ...]
# Log: CAMPAIGN_DIR/queue.log. Stop the queue: touch CAMPAIGN_DIR/queue.stop (checked between arms) or kill its tmux session.
set -u
root=$(cd "${1:?absolute campaign directory}" && pwd)
shift
code=${CODE:-$HOME/mtg-ml-r8/code-r3}
host="$code/tools/r8_host.sh"
log="$root/queue.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
say "queue start: $*"
for arm in "$@"; do
    if [ -e "$root/queue.stop" ]; then say "queue.stop present; leaving $arm and the rest"; exit 0; fi
    state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$root/$arm/process.json" 2>/dev/null || true)
    if [ "$state" = complete ]; then say "$arm already complete, skipping"; continue; fi
    if [ -e "$root/$arm/latest.pt" ]; then say "$arm has a previous run (status '$state'); not restarting it automatically, skipping"; continue; fi
    if tmux has-session -t "mtg-r8-$arm" 2>/dev/null; then say "mtg-r8-$arm already running; waiting for it"; else
        CODE="$code" R8_TOOL="$code/tools/r8_campaign.py" bash "$host" start "$root" "$arm" >> "$log" 2>&1 || { say "start of $arm failed"; continue; }
        say "started $arm"
    fi
    while tmux has-session -t "mtg-r8-$arm" 2>/dev/null; do sleep 60; done
    say "$arm ended: process.json status '$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$root/$arm/process.json" 2>/dev/null)'"
done
say "queue done"
