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

`MTG_ENGINE=native` (or `--engine native`) switches any command to the Rust engine.

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

- Many sessions run in parallel worktrees. Keep PRs to one component, branch from `main`, and avoid editing shared hotspots (README.md, `.gitignore`, `play.py`, golden digests) unless the change needs it.
- CI (`.github/workflows/ci.yml`) runs lint, the Python suite with CPU torch, and the native build + differential fuzz. It must be green before merge.
- Mark tests that take more than a few seconds `@pytest.mark.slow`; GPU-only tests `@pytest.mark.gpu`.
- Don't commit checkpoints, `runs/`, or large binaries.
- Benchmark numbers in docs should say which machine they were measured on.
