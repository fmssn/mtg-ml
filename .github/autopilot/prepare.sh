#!/usr/bin/env bash
# Builds the model's context and merges the base branch in.
# Usage: prepare.sh <mode> <pr> <base>   mode: review | ci-fix | conflict | escalate
# Env: RUNNER_TEMP, GH_TOKEN; CI_RUN_ID for ci-fix; ESCALATION for escalate.
set -euo pipefail
mode=$1 pr=$2 base=$3
ctx=$RUNNER_TEMP/context.md
conflicts=$RUNNER_TEMP/conflicts.txt

git fetch -q origin "$base"
git rev-parse HEAD > "$RUNNER_TEMP/start_head"   # finish.py: did the branch move during the run?
mb=$(git merge-base HEAD "origin/$base")
{
  echo "# PR #$pr (autopilot mode: $mode)"
  echo
  gh pr view "$pr" --json title,body --jq '"## " + .title + "\n\n" + (.body // "")'
  echo
  echo "## Diff against $base"
  echo '```'
  git diff --stat=120 "$mb" HEAD
  echo '```'
  echo '```diff'
  # Generated data stays out of the prompt: it is re-sent on every model turn.
  # Written to a file first: piping straight into `head` kills git with SIGPIPE on a
  # diff over the cap, and pipefail turns that into exit 141 (PR #45).
  git diff "$mb" HEAD -- . ':(exclude)data/*.json' ':(exclude)*.lock' ':(exclude)tests/data/**' \
    ':(exclude)docs/experiments/ledger.jsonl' > "$RUNNER_TEMP/pr.diff"
  head -c 120000 "$RUNNER_TEMP/pr.diff"
  [ "$(wc -c < "$RUNNER_TEMP/pr.diff")" -le 120000 ] || printf '\n[diff cut at 120 KB; read the files for the rest]\n'
  echo '```'
} > "$ctx"

if [[ $mode == ci-fix && -n ${CI_RUN_ID:-} ]]; then
  {
    echo
    echo "## Failed CI log (tail)"
    echo '```'
    gh run view "$CI_RUN_ID" --log-failed | tail -n 300 | cut -c1-400
    echo '```'
  } >> "$ctx"
fi

if [[ $mode == escalate ]]; then
  { echo; echo "## Why this was escalated"; echo "${ESCALATION:-Escalated by hand (needs-opus label).}"; } >> "$ctx"
fi

: > "$conflicts"
if ! git merge --no-edit "origin/$base" > /dev/null 2>&1; then
  git diff --name-only --diff-filter=U > "$conflicts"
  {
    echo
    echo "## Merge conflicts with $base"
    echo "origin/$base was merged into this branch and conflicted. Resolve every conflict marker in these files, keeping both sides' intent:"
    sed 's/^/- /' "$conflicts"
  } >> "$ctx"
fi
