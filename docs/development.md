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
Conductor reads shared settings from the remote default branch, so the new
configuration takes effect in the app after merge. For this unmerged workspace,
run the Make commands in its terminal; no machine-wide settings change is needed.

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

## Shared instructions and review

`CLAUDE.md` holds shared project rules. `AGENTS.md` directs Codex to those rules
and the context index. The Codex `game-review` skill lives in `.agents/skills/`;
Claude retains its existing skill and Workflow runner. Both use the same Python
recording, prompt, reply-import and verification tools; see `game-review.md`.
The GitHub autopilot continues to run independently of the local coding agent.
