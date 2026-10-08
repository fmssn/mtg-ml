# Project instructions

Read and follow [CLAUDE.md](CLAUDE.md) for the shared project commands,
engine/feature invariants, experiment conventions and PR workflow. These rules
apply to Codex as well as Claude Code; keep them in that one source of truth.

For setup, use [docs/development.md](docs/development.md). When continuing work
from another session, planning experiments or choosing a task alongside other
agents, consult [docs/context.md](docs/context.md) and the current shared status
it points to. Historical handoffs are evidence, not a current task list.

Run Python commands with `.venv/bin/python` after `make setup`; Make targets
select this workspace's tools automatically. Rebuild with `make native` after
Rust changes. Do not use another worktree's installed native extension.
