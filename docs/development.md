# Development in Conductor, Codex and Claude Code

Run `make setup` in each new checkout. It creates `.venv` with Python 3.11 (the CI
version), installs the editable project, dev/RL dependencies, Ruff and maturin,
then builds this checkout's Rust extension. Set `MTG_PYTHON=3.12` to choose another
supported interpreter before the first setup. An existing `.venv` is reused.
Prerequisites are uv and Rust; the scripts add `~/.cargo/bin` to PATH. Cloud setup
uses CPU torch. No global Python packages are changed.

```bash
make setup
make test-fast
make lint
make native                 # after editing Rust
.venv/bin/python -m mtg_ml.play bench --games 20 --engine native
```

Make selects workspace tools without shell activation. For direct commands,
activate with `source .venv/bin/activate` or use `.venv/bin/python`. Rust's build
cache remains shared across worktrees, while each native extension is installed
into its own environment. CI can still use its already-activated environment
when there is no workspace `.venv`.

## Conductor run commands

`.conductor/settings.toml` defines setup, replay, live play and tests. Local
Conductor reads shared settings from the remote default branch; no
machine-wide settings change is needed.

| command | behavior |
|---|---|
| `make replay` | viewer on `127.0.0.1:$CONDUCTOR_PORT` |
| `make live` | play against checkpoints on `127.0.0.1:($CONDUCTOR_PORT + 1)` |
| `make test-fast` | existing test selection, using workspace Python |

Outside Conductor the ports are 8765 and 8766. Run commands use the workspace's
replays and environment, so different Conductor workspaces can run concurrently.
Set `MTG_MODELS_DIR` to the directory containing local checkpoints before live
play; it defaults to `runs/`. Checkpoints stay outside Git. For example:

```bash
MTG_MODELS_DIR=/path/to/policy-directory make live
```

Conductor's Files to copy copies `.env*` by default. The current review key is in
`.env`; keep it gitignored. Do not copy `.venv`, Rust binaries or training runs
between workspaces. Claude's `.claude/launch.json` remains usable in Claude;
Conductor uses the Make run commands above.

## Running tests on h100-private

Full suites, native builds and the differential fuzz saturate this Mac when
several workspaces run them at once. The `-remote` targets run them on the
training box instead:

```bash
make test-remote                              # full suite, minus gpu tests
make test-fast-remote ARGS="tests/test_rl.py -k ppo"
make difftest-remote
```

`scripts/remote-test.sh` pushes HEAD over ssh into `~/mtg-ml-tests/<workspace>/`
(some tests read git history), overlays the working tree with uncommitted
changes, and keeps a per-workspace venv (CPU torch, like CI) and native build
there, reinstalling only when `pyproject.toml` or `native/` changes. Tests run
with xdist under `nice` on CPUs 48-63 (`REMOTE_CPUS`, `REMOTE_JOBS`,
`REMOTE_HOST` override), so training keeps priority. Reserve those cores
through the tracker, as `h100-private:cpu:tests` (see
[workspace coordination](workspace-coordination.md#reserve-shared-compute)).
Focused tests on a file or two stay faster locally.

Measured on h100-private (Xeon 8462Y+, 16 cores, 2026-10-09): `test-remote`
11:46 (2134 tests; the tail is unmarked `test_copy.py` and `test_difftest.py`
cases of 60-200 s each), `test-fast-remote` 4:48, `difftest-remote` 3:50.
Sync and checks add about 6 s once the venv and native build exist.

## Shared instructions and review

`CLAUDE.md` holds shared project rules. `AGENTS.md` directs Codex to those rules
and the context index. The Codex `game-review` skill lives in `.agents/skills/`;
Claude retains its existing skill and Workflow runner. Both use the same Python
recording, prompt, reply-import and verification tools; see `game-review.md`.
The GitHub autopilot continues to run independently of the local coding agent.

Project `mtg-*` subagents share short role instructions across Codex and Claude.
See [agents.md](agents.md) for routing, scoped briefs, limits, validation and
fallbacks. Small tasks stay single-agent; independent deliverables retain their
own workspace and PR.

For planning in one chat and implementing in another, use the saved plan plus
`.context/handoff.json`. The [coordination guide](workspace-coordination.md)
documents activation, baseline checks, shared write claims and resource slots.
The installed shared tracker helper is maintained in `tools/progress.py`.
