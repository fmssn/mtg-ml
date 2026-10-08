#!/usr/bin/env bash
# Run on the training server. tmux owns each process; no restart loop.
set -euo pipefail
action=${1:?start, dashboard, status, stop, or stop-monitoring}
root=${2:?absolute campaign directory}
code=$(cd "$(dirname "$0")/.." && pwd)
python="$code/.venv/bin/python"
prefix=${3:-mtg-r7}
root=$(cd "$root" && pwd)

start_session() {
    local name=$1 command
    shift
    if tmux has-session -t "$prefix-$name" 2>/dev/null; then
        echo "$prefix-$name already exists" >&2
        return 1
    fi
    printf -v command '%q ' "$@"
    tmux new-session -d -s "$prefix-$name" -c "$code" "$command"
    tmux set-option -t "$prefix-$name" history-limit 2000
}

case "$action" in
    start)
        start_session lr150 "$python" tools/overnight_campaign.py run --root "$root" --arm lr150
        start_session lr075 "$python" tools/overnight_campaign.py run --root "$root" --arm lr075
        # Caller uses a separate prefix for smoke runs; the watcher is production only.
        if ! "$python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))[0]["smoke"])' "$root/campaign.json"; then
            start_session watch env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 3 nice -n 10 "$python" tools/overnight_campaign.py watch --root "$root"
        fi
        ;;
    dashboard)
        start_session dashboard taskset -c 3 nice -n 10 "$python" apps/training-dashboard/server.py --source local --campaign "$root" --cache "$root/dashboard-state.json" --port 8767 --interval 15
        ;;
    status)
        "$python" tools/overnight_campaign.py status --root "$root"
        tmux list-sessions
        ;;
    stop)
        "$python" tools/overnight_campaign.py stop --root "$root"
        ;;
    stop-monitoring)
        for name in dashboard watch; do
            if tmux has-session -t "$prefix-$name" 2>/dev/null; then
                tmux kill-session -t "$prefix-$name"
            fi
        done
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
