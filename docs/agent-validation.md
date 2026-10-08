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
- Claude's initialization event still lists the denied generic agents. A
  harmless attempt to run `arch-code-reviewer` with a literal-only response was
  rejected: `Agent type 'arch-code-reviewer' has been denied by permission rule
  'Agent(arch-code-reviewer)' from projectSettings.` Zero children launched.
  The four other names were checked statically; no generic workflow was run.

## Runtime scenarios and comparison

Both clients ran `mtg-engine` then `mtg-reviewer` on bounded read-only tasks.
The engine role kept the Python/Rust halves, card data and parity tests under
one writer and listed the rebuild/differential checks as **unrun**. The reviewer
checked configuration and returned no confirmed defect with a static-only
coverage limit. No tracked edits, engine tests or GPU jobs were performed by
these trials. Claude reported two completed children, depth one, zero nested
children and zero edits; its returned summaries were 155 and 175 words.

Claude's parent and all three tested roles resolved to `claude-opus-5-5`,
confirming model inheritance. Codex's persisted trials reported parent model
`gpt-6.1-sol` (the installed CLI default); child resolved models and usage were
not exposed in the captured events. The adapters set no model/effort overrides.
The two providers were tested with their own existing defaults, not compared
against each other as models.

The Codex runtime trace exposed an implementation issue: the initial role smoke
omitted `fork_turns`, whose default inherits the entire conversation. Natural-
language requests for fresh context were insufficient. The project instructions
now require `fork_turns: "none"` explicitly. A follow-up trace confirms one
`spawn_agent` with `agent_type: "mtg-scout"` and `fork_turns: "none"`. Its answer
was correct. The earlier ephemeral delegated trial could not independently
establish its fork setting and is retained only as diagnostic evidence.

The reviewer's uncertainty about current configuration keys was resolved with
the official references linked in [agents.md](agents.md) and the installed
Codex strict-config run. The Codex trace advertises four total slots (parent
plus three children). Claude limits were checked against its documented env
settings; neither runtime was stress-tested by deliberately spawning a fourth
child. Codex's no-recursion rule is an instruction; Claude additionally removes
the Agent tool from these children. Runtime overrides remain possible.

### Fixed investigation

Each serial/delegated pair asked for four facts from the same source files:

> Investigate how engine tests execute against Python and Rust. Use Makefile,
> tests/conftest.py and tests/test_difftest.py. Report the engine-module
> registration rule, exact focused differential command using workspace Python,
> make difftest game/job defaults, and the rebuild required after Rust changes.
> Give file/line evidence. Do not run tests or edit files.

The serial prompt required no delegation. The delegated prompt required exactly
one `mtg-scout`, no duplicated parent investigation, and a fresh brief. Code was
`5fa6a2653aec2f93a221f2a681d4b56e5bc7f6c0`; the source files were unchanged across
all runs. The final Codex run additionally used the explicit fork-setting fix.
Claude runs allowed only Read/Glob/Grep/Agent for the scenario; Codex ran in
read-only mode. No project permission defaults were relaxed.

| Trial | Wall time | Parent reported input | Cache read within input | Parent output | Child usage |
|---|---:|---:|---:|---:|---|
| Codex serial | 26.21 s | 41,860 | 24,960 | 534 | No child |
| Codex explicit fresh scout | 61.55 s | 101,273 | 86,272 | 931 | Unavailable |
| Claude serial | 22.08 s | 321,319 | 182,042 | 1,725 | No child |
| Claude scout | 30.05 s | 232,240 | 96,196 | 1,382 | 20,679 input; 879 output |

Codex input already includes cached input. For Claude the input column adds
`input_tokens`, `cache_creation_input_tokens` and `cache_read_input_tokens`:
serial = 6 + 139,271 + 182,042; delegated = 4 + 136,040 + 96,196.
Claude's scout separately reported 2 uncached + 9,617 cache-created + 11,060
cache-read input, 879 output, 12.898 seconds and five reads. These are captured
usage counters, not billing estimates. SDK-level aggregate model counters also
include host work and are not substituted for parent/child counters.

All four answers matched the checked source: `ENGINE_MODULES` **or** the
`test_cards_` prefix; `.venv/bin/python -m pytest -q tests/test_difftest.py`;
2,000 games / 8 jobs; rebuild with `make native`. No answer needed rework.
The native skip conditions and differential module's direct use of both engines
were also reported correctly. Both clients' small-task tests spawned no child.

Claude serial made five evidence read/search calls, revisiting Makefile and
conftest once each. Its scout read each evidence file once plus its role and
contract; the parent did not reread the evidence. Codex serial read the three
files in three shell calls without repeated evidence reads. Its fresh parent
read project/role instructions but no
evidence files; child read counts were unavailable in the captured telemetry.

**Result:** delegation was slower on this small task in both clients. Claude
used less parent input in this one run; Codex used more, and its child usage is
unknown. The small sample, cache state, inherited host context, sequential run
order and Codex fork fix prevent a general savings claim. This supports the
single-agent default for small tasks. Re-measure on substantial independent
work before asserting a throughput or token improvement.

Raw prompts, JSONL events and timing metadata are under
`.context/agent-validation/` in the athens workspace. They remain gitignored;
the report preserves the relevant observations without publishing user-level
configuration, tool payloads or unrelated connector metadata. Sanitized
timings/counters are committed in [agent-validation.json](agent-validation.json).
The role smoke persisted its own Codex trace to verify actual spawn arguments;
most other trials were ephemeral. No raw session trace is published.
