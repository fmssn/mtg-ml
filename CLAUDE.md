# mtg-ml — notes for agents

Rules engine + RL for Pauper Magic (Jund Wildfire vs Mono Blue Terror). Python reference engine, a bit-exact Rust port, scripted bots, masked-PPO self-play. README.md is the user-facing overview; `docs/` has the deep dives.

## Commands

```bash
make setup        # uv pip install -e '.[dev,rl]' + build native engine
make test-fast    # what to run while iterating
make test         # full suite, both engines if native is built
make lint         # ruff check (CI blocks on it)
make difftest     # Python vs Rust in lockstep
```

`MTG_ENGINE=native` switches play, training, evaluation and trace commands to the Rust engine; `mtg_ml.play`, `mtg_ml.rl.evaluate` and `mtg_ml.trace` also take `--engine native`. The test suite picks engines itself (see `tests/conftest.py`), and `mtg_ml.difftest` always runs both.

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

- Many sessions run in parallel worktrees. Keep PRs to one component, branch from the default branch (`git remote set-head origin -a`, then `origin/HEAD`), and avoid editing shared hotspots (README.md, `.gitignore`, `play.py`, golden digests) unless the change needs it.
- CI (`.github/workflows/ci.yml`) runs lint, the Python suite with CPU torch, and the native build + differential fuzz. It must be green before merge.
- Mark tests that take more than a few seconds `@pytest.mark.slow`; GPU-only tests `@pytest.mark.gpu` (CI runs on CPU and deselects both).
- Don't commit checkpoints, `runs/`, or large binaries.
- Benchmark numbers in docs should say which machine they were measured on.

## PR workflow (GitHub Copilot reviews every PR)

1. **Open a draft PR early**, as soon as the first meaningful commit is pushed: `gh pr create --draft`. This starts CI and Copilot's review while work continues.
2. Keep pushing to the draft. Address Copilot and CI feedback as it comes in.
3. **When the user confirms the work is ready:** `gh pr ready <n>`, then `gh pr edit <n> --add-reviewer @copilot` to request a fresh review of the final state.
4. **Turn on Auto-fix** for the PR in the desktop app (or tell the user to), then wait for Copilot's review and CI instead of polling. On each event: fix, verify, push, reply in each review thread saying what changed, and resolve the thread.
5. Don't merge until CI is green and Copilot's latest review has no open findings. Merging is the user's call.
