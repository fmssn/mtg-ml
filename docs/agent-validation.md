# Agent configuration validation — 2026-10-08

Validation runs on Fabsi's local Apple Silicon Mac (macOS 26.6.2, arm64), using
Conductor's Codex CLI 0.159.3 and Claude Code 2.1.294. The starting code revision
is `28105450d1d448306e43f3c3056b59ff889d6b7c`; agent files were uncommitted additions
at the first discovery check. This report concerns development delegation, not
checkpoint game-review calibration.

## Static checks and discovery

- Seven canonical roles and both sets of adapters parse with matching names,
  descriptions and existing contract/role references. Descriptions are 15–17
  words; role bodies are 138–205 words. Models are inherited; no new memory or
  preloaded skill configuration is present.
- JSON/TOML checks confirm the three-child limit, Claude depth one and five
  named generic-agent deny entries. `make lint` and `git diff --check` pass.
- Both clients identified all seven `mtg-*` names from their actual tool
  interfaces. Both answered the small `make native` question without spawning
  any subagents. Codex accepted project configuration with `exec --strict-config`.
- Claude's initialization event still lists the denied generic agents. Do not
  equate that inventory with permission to run them or claim their description
  tokens have been removed.

## Runtime scenarios and comparison

Further bounded role and usage trials are in progress. Their results, failures
and limitations will be added before this draft is handed back. No token or
speed improvement is claimed from static checks or discovery alone.

Raw prompts, JSONL events and timing metadata are under
`.context/agent-validation/` in the athens workspace. They remain gitignored;
the report preserves the relevant observations without publishing user-level
configuration, tool payloads or unrelated connector metadata.
