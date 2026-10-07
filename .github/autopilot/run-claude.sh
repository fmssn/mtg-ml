#!/usr/bin/env bash
# Runs Claude Code headless on $RUNNER_TEMP/context.md.
# Usage: run-claude.sh <system-prompt-file> <model> <budget-usd> <timeout> <max-turns>
# Backend and auth come from the calling step's env (DeepSeek or the Claude subscription).
# Streams each tool call to the Actions log as it happens; the final result object
# goes to $RUNNER_TEMP/claude.json for finish.py.
set -uo pipefail
# Only these tools exist in the session (no Agent, WebFetch, WebSearch, ...).
builtin=(Read Glob Grep Edit Write Bash)
# Pre-approved commands. With --permission-mode dontAsk everything else is denied,
# including cargo/maturin builds, sleep, background jobs and ad-hoc probe scripts.
allowed=(Read Glob Grep Edit Write
  "Bash(ruff check:*)" "Bash(python -m pytest:*)"
  "Bash(git diff:*)" "Bash(git status:*)" "Bash(git log:*)" "Bash(git show:*)")
stream=$RUNNER_TEMP/claude.jsonl

# Background Bash tasks kept the session alive after the final answer (PR #28 ran into
# the 25-minute timeout that way). Cap every command at 5 minutes.
export CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
export BASH_DEFAULT_TIMEOUT_MS=180000 BASH_MAX_TIMEOUT_MS=300000

# PRs branched before the turn cap existed run their own (older) workflow file, which
# passes only four arguments: default the timeout and turn cap instead of aborting.
timeout "${4:-15m}" claude -p "Work on the PR below, following your instructions." \
  --model "$2" --output-format stream-json --verbose \
  --max-budget-usd "$3" --max-turns "${5:-30}" \
  --permission-mode dontAsk --permission-prompts none \
  --settings '{"sandbox":{"enabled":false}}' \
  --append-system-prompt "$(cat "$1")" \
  --tools "${builtin[@]}" \
  --allowedTools "${allowed[@]}" \
  < "$RUNNER_TEMP/context.md" \
  | tee "$stream" \
  | jq -R --unbuffered -r '
      fromjson? |
      if .type == "system" and .subtype == "init" then
        "session: model=\(.model) permissionMode=\(.permissionMode) tools=\(.tools | join(","))"
      elif .type == "assistant" then
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
jq -r '
  "cost (at Anthropic list prices): \(.total_cost_usd // "?"), turns: \(.num_turns // "?"), duration: \((.duration_ms // 0) / 60000 | floor) min",
  "usage: \(.usage // {} | tojson)",
  "models: \(.modelUsage // {} | tojson)",
  "denied tool calls: \(.permission_denials // [] | length)",
  (.result // "" | split("\n") | .[-40:] | join("\n"))' "$RUNNER_TEMP/claude.json" || true
