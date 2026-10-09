#!/usr/bin/env bash
# Campaign r8 on one host. tmux owns each trainer supervisor; no restart loop.
#   r8_host.sh start CAMPAIGN_DIR ARM_NAME [--resume]   (new detached session mtg-r8-ARM_NAME)
#   r8_host.sh status|stop CAMPAIGN_DIR
# Run with CODE=<deployed checkout> (default ~/mtg-ml-r8/code); this script is not part of the deployed runtime.
set -euo pipefail
action=${1:?start, status or stop}
root=$(cd "${2:?absolute campaign directory}" && pwd)
code=${CODE:-$HOME/mtg-ml-r8/code}
python="$code/.venv/bin/python"
tool="${R8_TOOL:-$(cd "$(dirname "$0")" && pwd)/r8_campaign.py}"
prefix=${R8_PREFIX:-mtg-r8}

case "$action" in
    start)
        arm=${3:?arm name}
        shift 3
        if tmux has-session -t "$prefix-$arm" 2>/dev/null; then
            echo "$prefix-$arm already exists" >&2
            exit 1
        fi
        printf -v command '%q ' "$python" "$tool" run --root "$root" --arm "$arm" "$@"
        tmux new-session -d -s "$prefix-$arm" -c "$code" "$command"
        tmux set-option -t "$prefix-$arm" history-limit 2000
        ;;
    status)
        "$python" "$tool" status --root "$root"
        tmux list-sessions
        ;;
    stop)
        "$python" "$tool" stop --root "$root"
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
