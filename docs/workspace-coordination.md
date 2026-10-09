# Parallel workspaces and plan handoffs

Use one workspace and draft PR per independent deliverable. Astra can plan and
Opus can implement in a fresh chat in that workspace. Keep one implementation
writer active in a checkout; reviews identify the commit they examined. The
workspace owns the task across model changes. Subagent limits do not coordinate
separate Conductor sessions.

## Start or resume work

Read `CLAUDE.md`, `docs/context.md`, the active handoff if present, and the shared
tracker. Reconcile its PR observations before choosing overlapping work:

```bash
export MTG_PROGRESS_PATH=/Users/fabsi/repos/mtg-ml-orchestration/progress.json
.venv/bin/python tools/progress.py reconcile --repo fmssn/mtg-ml
.venv/bin/python tools/progress.py status
```

`tools/progress.py` is the versioned source for the standalone updater installed
at `/Users/fabsi/repos/mtg-ml-orchestration/progress.py`. That installed copy
defaults to the adjacent existing JSON, so older `track`, `result` and `log`
commands still work. All writers use the existing `progress.json.lock`. The
repository copy requires `MTG_PROGRESS_PATH` or `--file`; it never creates a
second tracker. On another machine configure the actual shared tracker path.
Back up and atomically replace the installed helper after testing updates;
never copy a stale progress.json over the live tracker.

Reconciliation only reads GitHub. It records PR state, draft status, head/base
and observation time in each task's `github` field, including PRs found from a
recorded branch. It preserves task status, notes and unfinished steps: merging
code does not finish its experiments. `status` shows task and PR state separately.
Failed refreshes retain earlier evidence but mark it stale. A PR binding changed
by another writer during a refresh is not overwritten. Owners close completed
tasks and record follow-ups separately; historical notes are not live PR state.

## Claim writes and dependencies

Use the workspace name as owner, with its full path and actual branch. Claim
literal repository-relative files or directories; directory claims cover
descendants. Claims are exclusive across running, review and blocked tasks.

```bash
.venv/bin/python tools/progress.py track example-fix --name 'Example fix' \
  --status running --owner manila --workspace "$PWD" \
  --branch "$(git branch --show-current)" \
  --write-path tools/example.py --write-path tests/test_example.py \
  --depends-on pr:64 --handoff "$PWD/.context/handoff.json"
```

Updates to a claimed task require the same `--owner`. Supplied write paths replace
the entire claim list. `--status done` releases effective claims while preserving
their history. There is no automatic takeover. Unclaimed legacy tasks still
need their notes and PRs checked; the tool cannot infer ownership from prose or
stop editors that ignore it. Do not seize another active session's edits.

Shared interfaces have one integration owner: normally the prerequisite/foundation
workspace. Leaf workers own specialist files and propose shared interface changes
to that owner, who lands them in the prerequisite before leaves update. This
also applies to test registration, golden digests and experiment ledgers. Record
dependencies as `pr:<number>` or task IDs. Claims do not launch or message agents.

## Activate the approved plan

The approved plan stays in `.context/plans/`. Its consistent handoff header is
the sidecar `.context/handoff.json`, pointing to exactly one active plan and
fingerprinting its contents. Older plan files can remain as history. Activate
after approval and before implementation, in the intended checkout:

```bash
.venv/bin/python tools/handoff.py activate \
  --plan .context/plans/example.md --task example-fix --owner manila \
  --base origin/claude/lucid-euler-0tjz9h --repo fmssn/mtg-ml \
  --depends-on 64 --write-path tools/example.py \
  --write-path tests/test_example.py \
  --check '.venv/bin/python -m pytest -q tests/test_example.py' \
  --check 'make lint'
```

Use the actual parent branch as `--base` for a stack. Omit `--depends-on` and
`--repo` without prerequisites. The record contains owner, task, workspace,
branch, exact HEAD/base commits, tracked and untracked change fingerprints,
dependency snapshots, permitted writes and acceptance checks. It stores hashes,
not diffs or logs. It grants no approval, runs no listed commands and changes
no branches. If the host saves plans only on leaving Plan Mode, the implementation
chat creates the record after reading the approved plan and checking its baseline.

The fresh chat runs `.venv/bin/python tools/handoff.py show` and
`.venv/bin/python tools/handoff.py check`, reads the named plan and confirms its
tracker claim. Changed plans, commits, working files or dependencies stop the
check. Inspect differences and reconcile the plan before refreshing; never
overwrite evidence just to pass. Routine expected updates need no new approval;
scope changes do. On later resumes this distinguishes expected implementation
progress from unrelated changes. It is a starting-state check, not an edit sandbox
or a test receipt. Preserve durable decisions and validation in the PR or committed
docs before archiving `.context`.

## Reserve shared compute

Use the same tracker for cooperative reservations. The local policy is one
`local:heavy` slot for native builds, full suites and fuzzing across this Mac's
workspaces. Focused short tests and lint need no slot. Run heavy foreground jobs:

```bash
.venv/bin/python tools/progress.py run example-fix --owner manila \
  --resource local:heavy -- make test-fast
```

The wrapper fails promptly if busy, preserves the command exit status and releases
after its process group stops, including failures and SIGINT/SIGTERM. Do not use
it for daemonized/background jobs. After a hard kill, verify the recorded host/PID
and actual jobs before the owning task releases the reservation. Reservations
never expire automatically or kill other tasks' jobs.

Remote training still needs authorization and live load checks. Reserve actual
GPU UUIDs and agreed CPU sets together, and release only after the job finishes:

```bash
.venv/bin/python tools/progress.py acquire example-fix --owner manila \
  --resource h100-private:gpu:GPU-ACTUAL-UUID --resource h100-private:cpu:0-7
# Launch the authorized remote job and wait for verified completion.
.venv/bin/python tools/progress.py release example-fix --owner manila \
  --resource h100-private:gpu:GPU-ACTUAL-UUID --resource h100-private:cpu:0-7
```

Names are exact shared identifiers, not hardware discovery: agree on CPU-set
names rather than inventing overlapping ranges. Multi-resource acquisition is
atomic. The tracker cannot discover unregistered jobs, enforce use outside the
wrapper or schedule GitHub-hosted CI. Retain the load checks in `docs/context.md`.

## Land dependent PRs

Keep leaves draft while their parent PR is open. After the parent merges, the
leaf owner retargets to the default branch, reconciles squash/merge history,
checks the diff and rebuilds native when needed. Refresh the handoff and rerun
meaningful validation and CI. User-confirmed readiness hands the PR to autopilot.
See [the stack merge policy](pr-autopilot.md#stacked-prs).

Validate these tools with `.venv/bin/python -m pytest -q tests/test_autopilot.py
tests/test_coordination.py`, `make lint` and `git diff --check`. Tests use disposable
trackers, repositories and GitHub responses; they never mutate PRs or run training.
