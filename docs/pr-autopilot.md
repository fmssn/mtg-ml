# PR autopilot

Ready PRs are reviewed, fixed and merged without the developer. A cheap model fixes mechanical defects, CI decides the merge, and Opus or a human gets the PR only when the cheap model can't settle it. Workflow: [`.github/workflows/pr-autopilot.yml`](../.github/workflows/pr-autopilot.yml); scripts and prompts: [`.github/autopilot/`](../.github/autopilot/).

## Flow

| event | what happens |
|---|---|
| PR marked ready, reopened, or the developer pushes to a ready PR | Round counter resets and any round still running on the PR is cancelled. DeepSeek reviews the diff, merges the base branch in, fixes what it finds, runs lint and the tests for what it changed, pushes once. Clean or fixed → auto-merge (squash) is enabled. |
| CI fails on a ready PR | Same job with the failed log as context (`ci-fix`). |
| Something lands on the default branch | Every ready PR is updated through GitHub's update-branch. A conflict starts the job in `conflict` mode. |
| CI still fails after two DeepSeek rounds, guard trip, or a concrete blocker | Escalates to Opus once (`needs-opus`, `autopilot:opus-used`). Adding `needs-opus` by hand does the same. |
| Review remains incomplete, an API fails, Opus can't settle it, changes tests or card data, or the PR fails again after Opus | `needs-human`, auto-merge off. The autopilot leaves the PR alone until the developer pushes. |

Drafts, forks and PRs by anyone but the repo owner are ignored. Pushes by the autopilot itself don't start a review; CI judges them.

## Guard rails

- The cheap model may not edit `tests/` (golden digests included), `mtg_ml/engine/cards.toml` or `.github/`, except to resolve a merge conflict in that exact file. If it does, nothing is pushed and the PR escalates. This blocks "fix" commits that bend an assertion to match a bug.
- Leftover conflict markers, failing `ruff check`, invalid results, missing file coverage or pending verification: nothing is pushed. Unfinished reviews and API failures disable auto-merge and receive `needs-human`; they do not automatically invoke Opus.
- DeepSeek runs in `--bare` mode with explicit trusted project invariants, `deepseek-flash`, and `max` reasoning effort. Opus retains subscription OAuth and its normal CLI startup. The available tools remain Read, Glob, Grep, Edit, Write and Bash. Focused lint, pytest, read-only Git and the trusted native build helper are pre-approved. Permission rules also automatically allow some read-only shell commands; the prompt requires bounded file tools and plain verification commands. No full-suite/fuzz, training or background jobs. Each tool command is capped at five minutes.
- DeepSeek gets 60 initial turns and a shared 15-minute budget. The initial phase reserves two minutes; a turn/time limit or malformed successful answer can resume the same session once for at most 10 turns with tools disabled. Finalization uses existing evidence and must report `incomplete` when review or verification remains. Opus retains 40 turns / 25 minutes for genuine blockers. API/auth failures never consume an automatic Opus review.
- If the branch moved during a run only because the base was merged in (update-branch), the result is merged onto it and pushed instead of redoing the review. A developer push cancels the running round and starts a fresh one.
- The model never commits or pushes: `finish.py` commits and pushes, as a GitHub App, so CI runs on autopilot pushes (pushes with `GITHUB_TOKEN` don't trigger workflows).
- Scripts and prompts are read from the default branch, not from the PR. Context records exact head/base/merge-base commits, every changed file, declared generated-data exclusions, and full filtered and unfiltered diff artifacts. The embedded patch is explicitly marked when truncated; omitted hunks remain available locally.
- Changes under `native/`, `mtg_ml/` or `tests/`, and native CI failures, get a cached native build before review. Build failures are supplied as context rather than aborting the model. After engine edits the reviewer rebuilds through `python <trusted-autopilot-path>/native.py` and runs focused tests on both engines. An unavailable build prevents a completed-review auto-merge.
- Results include `reviewed_files` and `pending_checks`; old receipts without a v2 context manifest retain compatibility. Critical GitHub mutations fail loudly. Auto-merge is paused when a review starts and enabled only with a matching reviewed head.
- Merges need `lint`, `python` and `native` green on an up-to-date branch (repo ruleset). The model never decides a merge.
- `ledger.jsonl` merges with `merge=union` ([`.gitattributes`](../.gitattributes)), so parallel ledger appends don't conflict.

## Models and keys

| role | backend | secret |
|---|---|---|
| review / fix | DeepSeek's Anthropic-compatible endpoint, explicitly `deepseek-flash` by default; repo variable `AUTOPILOT_DEEPSEEK_MODEL` can override it | `DEEPSEEK_API_KEY`, set only in that step as `ANTHROPIC_API_KEY` with `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic` |
| escalation | Claude subscription, `--model opus` | `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`) |
| pushes, labels, auto-merge | GitHub App | variable `AUTOPILOT_APP_ID`, secret `AUTOPILOT_APP_PRIVATE_KEY` |

No Anthropic API key exists anywhere in the workflow. DeepSeek is limited by turns and elapsed time; the historical positional budget argument remains accepted but no Anthropic dollar cap is applied to DeepSeek. Opus keeps its $10 CLI estimate cap. Receipts record completion, failure kind, continuation, phase durations, token usage, model usage and denied calls. Resumed CLI dollar estimates are conversation totals and are reported once, not summed. These client estimates are not provider billing records.

Every turn re-sends the system prompt, the diff and all tool output so far, so tokens grow with turns × context. Before the caps (2026-10-07) a review averaged ~60 turns (up to 91) and 15 minutes.

## One-time setup

1. Create a GitHub App (Settings → Developer settings → GitHub Apps): no webhook; repository permissions Contents, Pull requests, Issues and Workflows read/write, Actions read (Workflows because merging the base branch in can carry workflow changes, which GitHub rejects from an App without it). Install it on this repo. Store its App ID as repo **variable** `AUTOPILOT_APP_ID` and a generated private key as secret `AUTOPILOT_APP_PRIVATE_KEY`.
2. `gh secret set DEEPSEEK_API_KEY` and `gh secret set CLAUDE_CODE_OAUTH_TOKEN` (value from `claude setup-token`).
3. `bash .github/autopilot/setup-repo.sh`: enables auto-merge and update-branch, adds the ruleset (PR + required checks, admins can bypass), creates the labels. Run it **before** merging this workflow: without the ruleset, `gh pr merge --auto` merges immediately.
4. Turn off automatic Copilot code review (repo Settings → Rules / Copilot).

## Working with it

- Open drafts while working; drafts are never touched. `gh pr ready <n>` hands the PR to the autopilot.
- Autopilot comments start with **Autopilot:** and appear only when something was fixed or escalated.
- To stop it on a PR: convert it back to draft (`gh pr ready --undo <n>`).
- `workflow_dispatch` with a PR number and a mode (`review`, `ci-fix`, `conflict`, `escalate`) re-runs it by hand.

## Shadow replay validation

`tools/autopilot_replay.py` reconstructs the 13 audited initial reviews at the PR
and trusted-base SHAs recorded in the original checkout logs, plus PR #30
before/after its hidden-card and shutdown fixes. It creates isolated clones with
no remote, one virtual environment and freshly installed native extension each.
It merges each recorded base before building/reviewing, validates local guards
and lint, and never invokes the finish/push/merge controller. Keys come from the environment
or the gitignored `.env` and are not saved to artifacts.

```bash
npm exec --yes --package @anthropic-ai/claude-code@2.1.293 -- claude --version
.venv/bin/python tools/autopilot_replay.py --discover-only
.venv/bin/python tools/autopilot_replay.py --jobs 3
```

The audited snapshots are retained in
[`autopilot-replay-cases-2026-10-08.json`](autopilot-replay-cases-2026-10-08.json).
Use `--manifest docs/autopilot-replay-cases-2026-10-08.json` to reproduce those
cases after the original GitHub logs expire.

Artifacts default to `.context/autopilot-replay/`: exact cases, setup/review logs,
changed-code diffs, model/runner receipts and a resumable aggregate summary.
Completion target: at least 11/13 historical reviews within the model budget,
both known defects detected, and no corresponding finding after either fix.
Inspect findings and edits as well as completion counts before activating changes.
Replays are measured on the reported machine, not assumed to match CI hardware.
