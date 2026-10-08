# Delegated task contract

Follow `CLAUDE.md` for project rules. If it is already in context, do not reread
it. Read this contract and only your assigned role. The task brief narrows a
role's scope; a role is not permission to take over its entire subsystem.

## Parent's brief

Supply these fields in the delegation message:

- Objective and completion criteria.
- Workspace, branch and observed code revision; relevant uncommitted changes.
- Relevant paths and evidence; exact files the child may edit, or read-only.
- Dependencies, other writers and interfaces already agreed with them.
- Required checks and resource limits; output artifact location if needed.

Use a fresh context rather than a conversation fork. In Codex, set
`fork_turns: "none"` explicitly when that parameter exists; omission can mean
the full history. Otherwise use the host's equivalent fresh-context option and
report when unavailable. Route by descriptions; let the child read its own
role rather than loading every role into the parent. Do not paste whole logs,
PR bodies or historical handoffs when paths and a short explanation suffice.
Keep one active writer per file. A rules/cards change owns both engines; do not
assign Python and Rust halves to separate writers. Resolve overlapping scope
before edits. Children do not change branches, commit, push, manipulate PRs or
update the shared tracker. The parent owns those actions and final verification.
While a child works, do independent work or use completion events/longer waits;
avoid repeated short polls and duplicate reads of the child's source files.

## Child's execution

Work only on the assigned task. Do not spawn agents, recruit other sessions or
launch agent CLIs. Search narrowly, read relevant sections, and summarize noisy
output. Use the workspace's tools and environment. Leave unrelated work intact.
If required context is missing, report the specific gap to the parent; do not
silently invent a code ref, checkpoint configuration or resource assignment.
Read-only tasks may inspect and run authorized diagnostics, but cannot edit
tracked files. Write logs only to the assigned artifact location.

Before a shared-resource job, verify current load and explicit resource
assignments. A role alone never authorizes training, remote jobs or deployments.
Run the task's meaningful checks; do not add tests solely to mirror the change.
Do not weaken assertions or rerecord golden digests to hide a failure.

## Return and recovery

Return at most 300 words: outcome; changed paths (or none); evidence with
file/line or artifact references; checks actually run and their results;
unresolved issues and artifact paths. Distinguish passed, failed and unrun checks.
Save lengthy logs in `.context/` under the workspace, using task-specific names.
Never describe a partial result as complete.

On interruption or a bounded-run limit, return the partial result and smallest
remaining task. The parent resumes that child with a narrowed brief when useful,
instead of launching duplicate investigations. Verify findings before applying
them. Review snapshots need the reviewed revision; subsequent edits may require
a focused rereview, not a second full pipeline.
