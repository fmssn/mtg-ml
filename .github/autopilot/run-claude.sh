#!/usr/bin/env bash
# Runs Claude Code headless on $RUNNER_TEMP/context.md.
# Usage: run-claude.sh <system-prompt-file> <model> <budget-usd> <timeout>
# Backend and auth come from the calling step's env (DeepSeek or the Claude subscription).
# Streams each tool call to the Actions log as it happens; the final result object
# goes to $RUNNER_TEMP/claude.json for finish.py.
set -uo pipefail
tools=(Read Glob Grep Edit Write
  "Bash(ruff check:*)" "Bash(python -m pytest:*)"
  "Bash(git diff:*)" "Bash(git status:*)" "Bash(git log:*)" "Bash(git show:*)")
stream=$RUNNER_TEMP/claude.jsonl

timeout "$4" claude -p "Work on the PR below, following your instructions." \
  --model "$2" --output-format stream-json --verbose --max-budget-usd "$3" \
  --append-system-prompt "$(cat "$1")" \
  --allowedTools "${tools[@]}" \
  < "$RUNNER_TEMP/context.md" \
  | tee "$stream" \
  | jq -R --unbuffered -r '
      fromjson? |
      if .type == "assistant" then
        .message.content[]? |
        if .type == "tool_use" then
          "▶ \(.name) " + ((.input.command // .input.file_path // .input.pattern // "") | tostring | .[:160])
        elif .type == "text" and (.text | length) > 0 then
          "  " + (.text | gsub("\n"; " ") | .[:200])
        else empty end
      elif .type == "user" then
        .message.content[]? | select(.type == "tool_result" and .is_error == true) |
        "  ✗ " + ((.content | tostring) | gsub("\n"; " ") | .[:160])
      else empty end'
echo "claude exited ${PIPESTATUS[0]}"

jq -R -c 'fromjson? | select(.type == "result")' "$stream" | tail -n 1 > "$RUNNER_TEMP/claude.json"
jq -r '"cost (at Anthropic list prices): \(.total_cost_usd // "?"), turns: \(.num_turns // "?")", (.result // "" | split("\n") | .[-40:] | join("\n"))' "$RUNNER_TEMP/claude.json" || true
