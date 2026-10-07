# PR autopilot

Ready PRs are reviewed, fixed and merged without the developer. A cheap model fixes mechanical defects, CI decides the merge, and Opus or a human gets the PR only when the cheap model can't settle it. Workflow: [`.github/workflows/pr-autopilot.yml`](../.github/workflows/pr-autopilot.yml); scripts and prompts: [`.github/autopilot/`](../.github/autopilot/).

## Flow

| event | what happens |
|---|---|
| PR marked ready, reopened, or the developer pushes to a ready PR | Round counter resets. DeepSeek reviews the diff, merges the base branch in, fixes what it finds, runs lint and the related tests, pushes once. Clean or fixed → auto-merge (squash) is enabled. |
| CI fails on a ready PR | Same job with the failed log as context (`ci-fix`). |
| Something lands on the default branch | Every ready PR is updated through GitHub's update-branch. A conflict starts the job in `conflict` mode. |
| Third round, guard trip, or the model can't fix it | Escalates to Opus once (`needs-opus`, `autopilot:opus-used`). Adding `needs-opus` by hand does the same. |
| Opus can't settle it, changes tests or card data, or the PR fails again after Opus | `needs-human`, auto-merge off. The autopilot leaves the PR alone until the developer pushes. |

Drafts, forks and PRs by anyone but the repo owner are ignored. Pushes by the autopilot itself don't start a review; CI judges them.

## Guard rails

- The cheap model may not edit `tests/` (golden digests included), `mtg_ml/engine/cards.toml` or `.github/`, except to resolve a merge conflict in that exact file. If it does, nothing is pushed and the PR escalates. This blocks "fix" commits that bend an assertion to match a bug.
- Leftover conflict markers, failing `ruff check`, or no parsable result: nothing is pushed.
- The model can only read, edit, and run `ruff check`, `pytest` and read-only git. `finish.py` commits and pushes, as a GitHub App, so CI runs on autopilot pushes (pushes with `GITHUB_TOKEN` don't trigger workflows).
- Scripts and prompts are read from the default branch, not from the PR.
- Merges need `lint`, `python` and `native` green on an up-to-date branch (repo ruleset). The model never decides a merge.
- `ledger.jsonl` merges with `merge=union` ([`.gitattributes`](../.gitattributes)), so parallel ledger appends don't conflict.

## Models and keys

| role | backend | secret |
|---|---|---|
| review / fix | DeepSeek's Anthropic-compatible endpoint, `--model sonnet` → `deepseek-flash` | `DEEPSEEK_API_KEY`, set only in that step as `ANTHROPIC_API_KEY` with `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic` |
| escalation | Claude subscription, `--model opus` | `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`) |
| pushes, labels, auto-merge | GitHub App | variable `AUTOPILOT_APP_ID`, secret `AUTOPILOT_APP_PRIVATE_KEY` |

No Anthropic API key exists anywhere in the workflow. `--max-budget-usd` (3 for the fix step, 10 for Opus) is counted at Anthropic list prices, so it is a loose token cap, not the DeepSeek bill.

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
