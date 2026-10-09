# mtg-ml — notes for agents

Rules engine + RL for Pauper Magic: six decks (Jund Wildfire, Mono Blue Terror, Red Madness, Grixis Affinity, Elves, Tron) and 21 matchups (15 cross-deck, 6 mirrors that run only when named). Jund vs Mono Blue Terror is the standing benchmark. Python reference engine, a bit-exact Rust port, scripted bots, masked-PPO self-play. README.md is the user-facing overview; `docs/` has the deep dives.

## Commands

```bash
make setup        # create .venv, install dev + rl tools, build native engine
make test-fast    # what to run while iterating
make test         # full suite, both engines if native is built
make lint         # ruff check (CI blocks on it)
make difftest     # Python vs Rust in lockstep
```

Make targets use this checkout's `.venv` when present. For direct Python commands,
use `.venv/bin/python` or activate `.venv`. `make native` rebuilds into that same
environment; never install one worktree's native engine into a shared global Python.
Conductor setup and run commands are described in `docs/development.md`.

`MTG_ENGINE=native` switches play, training, evaluation and trace commands to the Rust engine; `mtg_ml.play`, `mtg_ml.rl.train`, `mtg_ml.rl.evaluate` and `mtg_ml.trace` also take `--engine native`, and an explicit `--engine python` beats the variable. The test suite picks engines itself (see `tests/conftest.py`), and `mtg_ml.difftest` always runs both.

## Invariants — do not break

- **The two engines must play identical games.** Any change to rules or cards goes into *both* `mtg_ml/engine/` and `native/src/`, and must pass `tests/test_difftest.py` and `make difftest`. The Python engine is the reference.
- **Golden digests** (`tests/data/golden_digests.json`) catch accidental behaviour changes. Only re-record (`make golden`) when the change is intentional, and say so in the commit message.
- **Cards** are defined once in `mtg_ml/engine/cards.toml` and checked against `data/oracle_cards.json`. Follow `docs/adding-cards.md`.
- New tests that touch engine behaviour go into a module listed in `ENGINE_MODULES` (`tests/conftest.py`) so they run on both engines.

## Layout

| path | what |
|---|---|
| `mtg_ml/engine/` | reference rules engine, card spec, decks, player views |
| `native/` | Rust port (pyo3 / maturin crate `mtg_ml_native`) |
| `mtg_ml/bots/` | scripted baseline bots + search |
| `mtg_ml/rl/` | features, model, rollouts, PPO, trainer, evaluation, inference server |
| `tools/` | benchmarks and audits (not imported by the package) |
| `docs/` | reference docs, plans, research |

New user-facing apps (replay viewer, dashboards) go under `apps/<name>/` and should read versioned files produced by `mtg_ml` (traces, replays, results), not import engine internals.

## Working in this repo

- For durable preferences and external evidence, see `docs/context.md`. When
  resuming shared work, read current orchestration status and verify PR state on
  GitHub before acting; old handoffs and agent IDs can be stale.
- Follow [workspace coordination](docs/workspace-coordination.md): reconcile the
  existing tracker on start/resume, claim explicit write paths under a stable
  workspace owner, and read/check `.context/handoff.json` when present. After an
  approved plan, record its starting revision and active plan before implementation.
  Resolve changed handoff evidence before editing; never refresh it blindly.
- Shared interface changes land through one prerequisite/foundation owner.
  Reserve `local:heavy` through the shared progress updater for native builds,
  full suites and fuzzing; reserve current GPU/CPU assignments for authorized
  remote jobs. Subagent limits are per session, not a shared compute budget.
- Many sessions run in parallel worktrees. Keep PRs to one component, branch from the default branch (`git remote set-head origin -a`, then `origin/HEAD`), and avoid editing shared hotspots (README.md, `.gitignore`, `play.py`, golden digests) unless the change needs it.
- CI (`.github/workflows/ci.yml`) runs lint, the Python suite with CPU torch, and the native build + differential fuzz. It must be green before merge. Its native job runs only tests marked `native`, which `tests/conftest.py` applies to tests with "native" in their id and to `test_difftest.py`; any other test that needs the Rust engine gets `@pytest.mark.native`. PRs touching only `docs/` and top-level `*.md` skip the test jobs.
- Mark tests that take more than a few seconds `@pytest.mark.slow`; GPU-only tests `@pytest.mark.gpu` (CI runs on CPU and deselects both).
- Don't commit checkpoints, `runs/`, or large binaries.
- Benchmark numbers in docs should say which machine they were measured on.

## Delegation

Use focused `mtg-*` agents for independent, substantial subtasks or noisy
investigations; do straightforward work directly. Default to one or two
children, at most three active per coordinator (or the runtime's lower limit).
Children do not delegate. Use fresh task briefs and the shared
[handoff contract](.agents/roles/contract.md). Route by descriptions; each child
loads only its assigned role. For Codex, explicitly set `fork_turns: "none"`
when supported; omitting it can inherit the entire conversation.
Inherit the selected model and reasoning settings; avoid generic quality
pipelines and duplicate reviews. The parent integrates, verifies, commits and
handles PRs. Give each file one writer, including paired Python/Rust changes;
serialize shared ledger, golden-digest and test-registration edits.
Independent deliverables use separate existing workspaces; same-deliverable
children may share a checkout with disjoint ownership. Check current ownership
through `docs/context.md` before assigning writes. Runtime permissions still
apply. The calibrated game-review skill/Workflow keeps its own protocol and
model choices; these development roles do not replace it.
See [agent usage and validation](docs/agents.md) for routing and host fallbacks.

## Training experiments

Every training run meant to answer a question is recorded in `docs/experiments/` (how: its README): an entry in `ledger.md` and `ledger.jsonl` with parent checkpoint, code ref, exact flags, benchmark (sampled and greedy) and ladder L1 Elo (`ladder.md` says how it is computed), and a verdict, failed ideas included. Finished runs are archived on h100-private with `~/mtg-ml-checkpoints/archive_run.sh`; that directory is append-only.

## PR workflow (the PR autopilot reviews, fixes and merges)

1. **Open a draft PR early**, as soon as the first meaningful commit is pushed: `gh pr create --draft`. CI runs on drafts; the autopilot ignores them.
2. Keep pushing to the draft and fix CI failures yourself.
   Stacked PRs remain draft until their prerequisite merges. Then retarget to
   the default branch, reconcile the diff and rerun validation/CI before readiness.
3. **When the user confirms the work is ready:** `gh pr ready <n>`. From there the autopilot ([docs/pr-autopilot.md](docs/pr-autopilot.md)) reviews, pushes fixes, keeps the branch current and auto-merges once CI is green. Don't request Copilot reviews.
4. Act on a PR again only when it gets `needs-human` (read the latest **Autopilot:** comment) or the user asks. Pushing to the PR restarts the autopilot.
5. A PR whose test changes alter expected behaviour (assertions, golden digests) should say so in its description: green CI is the only merge gate.
