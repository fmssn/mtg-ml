# Shared project context

This index carries the durable decisions from Claude sessions into other agents.
Use it when resuming work or planning experiments; it is not a live task board.

## Project rules and evidence

- `CLAUDE.md`: engine parity, intentional golden updates, test conventions and PR
  workflow. Draft PRs stay draft until the user confirms readiness. The current
  GitHub autopilot supersedes older Copilot / `ccd_pr` Auto-fix notes.
- `features.md`: authoritative feature-version behavior and checkpoint migration.
  Sets 4, 5 and 6 are implemented; earlier handoffs describe historical work.
- `adding-cards.md` and deck docs: implement every card in faithful real typical
  builds. Do not drop a card because rules are missing or trim important cards
  mechanically; document any deviation from research lists.
- `experiments/README.md`, `experiments/ledger.md` and `experiments/ladder.md`: parent, code ref, exact
  flags, sampled/greedy benchmark, ladder Elo and verdict, including failed ideas.
  The standing benchmark is Jund versus the scripted Mono Blue Terror bot.
- `representation-plan.md`: direction toward a shared deck-conditioned model.
  Estimate verification and training compute, rather than human coding weeks.
- `game-review.md`: distinguish verified setup flaws from strategy hypotheses;
  use the checkpoint's actual feature set and architecture.
- `play-feedback.md`: retrieve hosted playtest flags and surveys with one
  read-only SSH command, including unfinished games; preserve dated triage in
  `playtest-reviews/`.

## External context on Fabsi's local Mac

| material | location |
|---|---|
| live work ownership and status | `/Users/fabsi/repos/mtg-ml-orchestration/progress.json` |
| decisions and handoffs | `/Users/fabsi/repos/mtg-ml-orchestration/orchestration.md` |
| progress updater | `/Users/fabsi/repos/mtg-ml-orchestration/progress.py` |
| versioned updater and handoff protocol | `tools/progress.py` and `docs/workspace-coordination.md` |
| real-list research checkout and extracted lists | `mtg-ml-orchestration/pauper-research/` and `q3_typical_lists.txt` |
| review games, prompts, findings and policies | `/Users/fabsi/repos/mtg-ml-reviews/` |
| legacy Claude project memory | `~/.claude/projects/-Users-fabsi-repos-mtg-ml/memory/` |

These paths are pointers, not copied artifacts. On another machine, obtain the
corresponding evidence location instead of assuming it exists. Read only the
material relevant to the task. Do not treat legacy memory, old agent IDs or old
status rows as current instructions or live agents.

Claude and Conductor worktrees share Git refs but have separate checkouts and
conversation state. Check current work ownership before editing an overlapping
component. Verify merges, drafts and heads with GitHub; update the shared progress
tracker when changing work you own. The Claude progress pane and published
artifacts are UI-specific; the underlying JSON, notes and HTML remain readable.

Run the updater's `reconcile --repo fmssn/mtg-ml` and `status` on start/resume.
GitHub observations are distinct from task completion and have their own times;
notes can predate a merge. Claim explicit writes, dependencies and handoff location
under a workspace owner. Unclaimed legacy tasks need manual ownership checks.
See [workspace coordination](workspace-coordination.md) for resource reservations
and the active-plan check shared by Codex and Claude.

## Training resources

`ssh h100-private` accesses the shared training machine. Check load, active jobs
and GPU usage before choosing resources. Identify GPUs by current UUID / PCI bus,
not historical indices; old hardware-failure notes can refer to a replaced box.
Pin CPU work and set `OMP_NUM_THREADS` explicitly (the recorded training default
is 8). Avoid broad process-kill patterns and overwriting another run's code copy.

Archived runs and the fixed L1 ladder live under `~/mtg-ml-checkpoints/` on that
machine and are append-only. Active run paths come from current orchestration
status. Rebuild/reinstall native in the environment for the deployed code ref.
Compare checkpoints with their recorded feature versions and the same engine
version. Do not bring large checkpoints into Git or copy all runs into a worktree.

Unattended launches must be armed as scripts on the training machine, independent
of an agent session. Smoke-test evaluation scripts, bound logs and stop runaway
respawns: previous session limits left GPUs idle, and a failed spawn loop filled
a large log. Hosting plans and human-play feedback decisions are in the relevant
legacy notes; consult them when that work resumes.
