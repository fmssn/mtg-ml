#!/usr/bin/env bash
# Runs Claude Code headless on $RUNNER_TEMP/context.md.
# Usage: run-claude.sh <system-prompt-file> <model> <budget-usd> <timeout>
# Backend and auth come from the calling step's env (DeepSeek or the Claude subscription).
set -uo pipefail
tools=(Read Glob Grep Edit Write
  "Bash(ruff check:*)" "Bash(python -m pytest:*)"
  "Bash(git diff:*)" "Bash(git status:*)" "Bash(git log:*)" "Bash(git show:*)")
timeout "$4" claude -p "Work on the PR below, following your instructions." \
  --model "$2" --output-format json --max-budget-usd "$3" \
  --append-system-prompt "$(cat "$1")" \
  --allowedTools "${tools[@]}" \
  < "$RUNNER_TEMP/context.md" > "$RUNNER_TEMP/claude.json"
echo "claude exited $?"
jq -r '"cost (at Anthropic list prices): \(.total_cost_usd // "?"), turns: \(.num_turns // "?")", (.result // "" | split("\n") | .[-40:] | join("\n"))' "$RUNNER_TEMP/claude.json" || true
