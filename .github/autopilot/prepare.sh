#!/usr/bin/env bash
# Keep the existing invocation contract. Context construction is reusable offline.
set -euo pipefail
mode=$1 pr=$2 base=$3
ap=$(dirname "$0")
git fetch -q origin "$base"
python "$ap/context.py" "$mode" "$pr" "origin/$base"
: > "$RUNNER_TEMP/conflicts.txt"
if ! git merge --no-edit "origin/$base" > "$RUNNER_TEMP/merge.log" 2>&1; then
  git diff --name-only --diff-filter=U > "$RUNNER_TEMP/conflicts.txt"
  if [[ ! -s $RUNNER_TEMP/conflicts.txt ]]; then
    cat "$RUNNER_TEMP/merge.log" >&2
    exit 1
  fi
  {
    echo
    echo '## Actual merge conflicts (permission exceptions apply only to these files)'
    cat "$RUNNER_TEMP/conflicts.txt"
  } >> "$RUNNER_TEMP/context.md"
fi
python "$ap/native.py"
