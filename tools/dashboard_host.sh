#!/usr/bin/env bash
# Host the training tracker on the training box: detached tmux session, loopback bind,
# `tailscale serve` HTTPS on one port (tailnet only, no Funnel). Idempotent.
#   bash tools/dashboard_host.sh start|stop|status [--port 8768] [--root GLOB]... [--campaign DIR]...
# Never touches other serve entries (ComfyUI on 8188 and the `/` entry are off limits).
set -euo pipefail
action=${1:?start, stop or status}
shift || true
port=8768 session=mtg-tracker cpu=${DASHBOARD_CPU:-} extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --port) port=$2; shift 2 ;;
        --session) session=$2; shift 2 ;;
        --cpu) cpu=$2; shift 2 ;;
        --root|--campaign|--ledger|--recent-hours) extra+=("$1" "$2"); shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [ "$port" = 443 ] || [ "$port" = 8188 ]; then echo "port $port is reserved" >&2; exit 2; fi
code=$(cd "$(dirname "$0")/.." && pwd)
python="$code/.venv/bin/python"; [ -x "$python" ] || python=$(command -v python3)
target="http://127.0.0.1:$port"

serve_status() { tailscale serve status 2>/dev/null || true; }
# 0 if our https port already proxies to $target, 1 if free, 2 if taken by something else
serve_state() {
    local block
    block=$(serve_status | awk -v p=":$port " 'index($0,p){f=1;next} /^$/{f=0} f')
    [ -z "$block" ] && return 1
    grep -q "proxy $target" <<<"$block" && return 0
    return 2
}
url() { serve_status | awk -v p=":$port " 'index($0,p){print $1; exit}'; }

case "$action" in
    start)
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "$session already running"
        else
            pin=(); if [ -n "$cpu" ] && command -v taskset >/dev/null; then pin=(taskset -c "$cpu"); fi
            printf -v cmd '%q ' env OMP_NUM_THREADS=1 ${pin[@]+"${pin[@]}"} nice -n 10 "$python" apps/training-dashboard/server.py --port "$port" ${extra[@]+"${extra[@]}"}
            tmux new-session -d -s "$session" -c "$code" "$cmd"
            tmux set-option -t "$session" history-limit 2000
            echo "started $session on $target"
        fi
        for _ in 1 2 3 4 5 6 7 8 9 10; do curl -fsS "$target/healthz" >/dev/null 2>&1 && break; sleep 1; done
        curl -fsS "$target/healthz" >/dev/null || { echo "server did not come up; see: tmux attach -t $session" >&2; exit 1; }
        rc=0; serve_state || rc=$?
        if [ "$rc" = 2 ]; then echo "tailscale https port $port is used by another entry; refusing to change it" >&2; exit 1; fi
        if [ "$rc" != 0 ]; then tailscale serve --bg --https="$port" "$target"; fi
        echo "tracker: $(url)"
        ;;
    stop)
        if tmux has-session -t "$session" 2>/dev/null; then tmux kill-session -t "$session"; echo "stopped $session"; fi
        rc=0; serve_state || rc=$?
        if [ "$rc" = 0 ]; then tailscale serve --https="$port" off; echo "removed serve entry :$port"; fi
        ;;
    status)
        if tmux has-session -t "$session" 2>/dev/null; then echo "$session: running"; else echo "$session: not running"; fi
        curl -fsS "$target/healthz" 2>/dev/null || echo "healthz: no answer"
        echo; serve_status
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
