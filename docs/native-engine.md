# Native engine (`mtg_ml_native`)

The pure-Python engine (`mtg_ml/engine/`) is the reference. `native/` is a Rust port of it that plays **exactly the same games**: same seeds, same decisions, same options in the same order (labels and keys), same observations and the same RNG stream. It is an optional backend:

Reviewed [expert scenarios](expert-data.md) use the same setup interface on both
engines. Native setup exposes `turn`, `lands_played` and
`spells_cast_this_turn`; setters invalidate cached copy snapshots.

```bash
MTG_ENGINE=native python -m mtg_ml.play bench          # or --engine native on any play/train/evaluate command
python -m mtg_ml.rl.train --engine native ...           # rollout workers use the Rust engine
MTG_ENGINE=native python -m mtg_ml.rl.train ...         # the same; an explicit --engine python wins
```

| single core, decisions/s | Python | Rust | speed-up |
|---|---:|---:|---:|
| engine only (h100-private, Xeon 8462Y+) | 8,764 | 189,666 | 21.6× |
| rollout CPU path: featurize + event tokens + step (h100) | 4,738 | 59,667 | 12.6× |
| 64 pinned processes, rollout CPU path (h100) | 297,608 | **3,951,052** | 13.3× |

Full numbers, the PyPy experiment and the method are under [Benchmarks](#benchmarks).

## Contents

- [Building](#building)
- [Choosing the engine](#choosing-the-engine)
- [Architecture](#architecture)
- [Copying a game](#copying-a-game)
- [Determinism: how the two engines stay identical](#determinism-how-the-two-engines-stay-identical)
- [Differential testing](#differential-testing)
- [Benchmarks](#benchmarks)
- [PyPy](#pypy)
- [What limits training throughput now](#what-limits-training-throughput-now)
- [Limitations](#limitations)

## Building

Needs a Rust toolchain (`rustup`, stable) and `maturin`. The wheel is abi3 (CPython ≥ 3.11).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
pip install "maturin>=1.5,<2"
cd native
maturin develop --release                                  # into the active virtualenv
# or: maturin build --release -o ../target-wheels && pip install ../target-wheels/mtg_ml_native-*.whl
cargo test --no-default-features --release                 # Rust unit tests (RNG, mana, card spec)
```

Without the module everything still works on the Python engine, and the native test variants are skipped.

On h100-private (repo copy in `~/mtg-ml`, uv venv in `~/mtg-ml/.venv`):

```bash
cd native
taskset -c 48-63 ~/mtg-ml/.venv/bin/python -m maturin build --release -i ~/mtg-ml/.venv/bin/python -o ../target-wheels
~/.local/bin/uv pip install --python ~/mtg-ml/.venv/bin/python --reinstall ../target-wheels/mtg_ml_native-*.whl
```

## Choosing the engine

`mtg_ml/backend.py` is the switch. An explicit `engine=` beats `$MTG_ENGINE`, which beats the default (`python`).

| where | how |
|---|---|
| any code | `from mtg_ml.backend import game_class; Game = game_class()` (or `game_class("native")`) |
| `agents.play_game`, `match.play_match`, `env.MTGEnv` | `engine=` argument |
| `python -m mtg_ml.play ...` | `--engine python\|native` |
| `python -m mtg_ml.rl.train` | `--engine python\|native` (`TrainConfig.engine`, default empty: `$MTG_ENGINE`, else python); the resolved engine is exported as `$MTG_ENGINE` to the spawned workers |
| `python -m mtg_ml.rl.evaluate` | `--engine native` |
| `rollout.Job` | `engine=` field (`None` = `$MTG_ENGINE`) |

`NativeGame` (`mtg_ml/engine/native.py`) has the `Game` API: `decision` (`player`, `kind`, `prompt`, `options` with `label`, `key`, `value`), `legal_options()`, `step(i)`, `over` / `winner` / `end_reason`, `copy()` / `fork()`, `players`, `battlefield`, `stack`, `perm()`, `power()` and the other rules queries, `log`, `actions`, `rng`. `view.observe`, `view.determinize`, `encode.state_features` and `rl.features.featurize` dispatch to the native implementations. The scripted bots and `SearchBot` run unchanged on it.

## Architecture

```
native/src/
  rng.rs       CPython random.Random (MT19937) bit for bit
  mana.rs      ManaCost, Remaining, can_pay (Kuhn matching)            <- mana.py
  cards.rs     parses cards.toml into CardDefs and effect ops           <- cards.toml (shared)
  state.rs     game state + every rule that asks nothing                <- non-generator half of game.py
  engine.rs    every rule that asks (turns, priority, casting, payment,
               combat) and the effect ops / custom effects              <- generator half of game.py, cards.py
  game.rs      driver: owns the state and the engine coroutine
  features.rs  featurize() and event-token hashes                      <- encode.py, rl/features.py
  sim.rs       simulated option previews (`pv:sim:*`, feature set 6)    <- encode.py sim_previews
  lib.rs       crate root: module list; py.rs only with the `python` feature
  py.rs        PyO3 bindings (module mtg_ml_native)
mtg_ml/engine/native.py   NativeGame: the Game API on top of py.rs
```

**The generator, as a coroutine.** The Python engine is one generator: `ask()` yields a `Decision`, `step(i)` sends the choice back. The port runs the same code on a stackful coroutine (`corosensei`): `Eng::ask` stores the decision and suspends; `Game::step` resumes it with the index. So `engine.rs` reads line by line like `game.py` (same functions, same order of side effects), which is what makes exact parity tractable. A coroutine stack is 256 KiB of reserved (lazily committed) memory per live game; finished games hand theirs to the next game on the thread (a pool of 64). Copies: see [Copying a game](#copying-a-game).

**State and aliasing.** The state lives behind a raw pointer shared by the driver (which reads it between steps, and writes it for scenario setup and determinization) and the coroutine. Engine code never holds a `&mut State` across `ask()`: `Eng::s()` hands out a borrow tied to `&mut self`, and `ask()` takes `&mut self`, so the borrow checker enforces it. The split mirrors the code: `state.rs` (plain methods on `State`) never asks, `engine.rs` (`Eng`) does.

**Object model.** Cards live in an arena (`State.cards`), zones hold indices. Python's per-object identity maps to:

| Python | Rust |
|---|---|
| a `Card` object (followed through zone changes; `oid` changes per CR 400.7, `uid` does not) | arena index `CIdx` |
| `card.snapshot()` (last known information) | `Src::Snap(Box<Card>)`; a live reference is `Src::Live(CIdx)` |
| `StackItem` identity (`item in self.stack`) | its unique `sid` |
| option values (`("cast", card, mode)`, `("source", card, "B")`, refs, splits, ...) | `enum Val` |
| option keys (tuples of str / int / None / tuple) | `Vec<KI>`; every string is `&'static` (card, ability and mode names come from the static card database) |
| effect functions | `enum Op` lists from `cards.toml`, plus `enum Custom` |

**Python wrapper.** The hot path (`decision.kind/player`, option keys, `step`, `observe`, `featurize`, event hashes) is one native call each. Everything else (`players`, `battlefield`, `stack`, `Card` attributes, option `value`s) is served by proxies: one `NativeCard` per physical card for the whole game (stable identity, like Python objects), whose attributes are re-read after every step. `card.face` and `card.defn` are the Python `CardDef`s, so bots read static card data exactly as before. Scenario tests may mutate what they used to mutate on Python objects (`players[i].life`, `cards_drawn_this_turn`, `card.tapped`, `card.attached_to`, `card.temp.append(...)`, `g.active`, `g.pending.clear()`, `g.destroy()`, `g.sacrifice()`, `g._untap_step()`).

**One card spec.** Both engines load `mtg_ml/engine/cards.toml`. The wrapper passes the file's text to `mtg_ml_native.load_cards()` at import, so a card that only uses existing ops needs no Rust rebuild; a test asserts both engines run the same spec. See [adding-cards.md](adding-cards.md).

## Copying a game

`g.copy()` (both engines; `fork()` now means `copy()`, `fork(replay=True)` is the old replay) returns an independent game in exactly the same state (object ids, RNG state, log, actions) that continues identically. It is the infrastructure for simulated option previews (representation plan, step 3; `docs/representation-plan.md` on its own branch).

A literal clone is impossible: both engines are coroutines (a Python generator; a `corosensei` stack in Rust), and a suspended coroutine cannot be copied (the Rust stack holds heap pointers and pointers into itself). So copies restart the engine from a **snapshot taken at the start of the current step** and replay only the actions since. At a step start the coroutine holds nothing the data does not carry except which step it is and whether the draw is skipped, so `_main(resume=(step, skip_draw))` / `Eng::main(.., resume)` can continue the turn from there. The data copy is `State: Clone` in Rust (constructor args and the snapshot behind `Rc`) and a hand-written structured copy in Python (`game._copy_state`: each `Card` copied once and remapped, card definitions shared; ~20× faster than `copy.deepcopy`).

- Snapshots cost something, so a game takes them only once it has been copied. Its first `copy()` replays the whole history once (and adopts the replay's snapshot); later ones are cheap. A copy shares its parent's snapshot until its next step begins and takes none of its own unless it is copied itself, so a preview or playout that is never copied pays nothing.
- Edits outside `step()` (determinization, `add_card` after the start, `set_card_def`, setting the RNG state: every native `PyGame::mutate`, and `view.determinize` in Python) drop the snapshot. Until the next step begins, a copy of an edited game falls back to a replay, which, like the old `fork()`, does not carry the edits; from the next step on, copies carry them. Direct attribute writes on the Python engine are not tracked.
- Checked by `tests/test_copy.py` (both engines: copy vs replay fork, state + features + log + RNG at every step of random continuations, independence, copies of copies, scenario setups, determinized games) and by the difftest fork check, which now also compares `copy()` with the replay in each engine.

Benchmark (`python tools/bench_copy.py --engine {python,native} --games 40`): scripted-bot games, median time per copy at a decision index, Apple M3 laptop (16 GB) with seven other agent sessions running (load ~60), so absolute numbers are noisy:

| decision | games | Python replay | Python `copy()` | speed-up | Rust replay | Rust `copy()` | speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 40 | 2.36 ms | 0.17 ms | 14× | 0.139 ms | 0.019 ms | 7× |
| 200 | 27 | 8.78 ms | 0.24 ms | 37× | 0.458 ms | 0.022 ms | 21× |
| 400 | 7 | 20.3 ms | 0.25 ms | 83× | 1.032 ms | 0.018 ms | 58× |

Copy cost no longer grows with the game: it is the state copy plus the actions since the step began (median 1, max 33 in this run). A Rust copy is ~12 µs to build and was ~5 µs to drop (unwinding the suspended coroutine); since feature set 6 a suspended game is dropped by resuming its `ask` with an abort index, which returns `Stop::Abort` up to `main` through ordinary returns (`Game::release_co`), about 4× cheaper. The price is a snapshot at every step start in a game that has been copied: per decision under random play, Rust 5.9 → 6.5 µs, Python 92 → 261 µs (random play crosses several steps per decision; the Python state copy is ~80 µs).

## Determinism: how the two engines stay identical

- **RNG.** `rng.rs` is CPython's `random.Random`: MT19937, `init_by_array` seeded with |seed| split into little-endian 32-bit words, `getrandbits(k) = genrand >> (32 - k)`, `_randbelow(n)` by rejection on `getrandbits(n.bit_length())`, and `shuffle` as `for i in reversed(range(1, n)): j = _randbelow(i + 1)`. `rng.getstate()` returns CPython's tuple and is compared after every differential game; `SearchBot` re-seeds the engine RNG through `rng.setstate`.
- **One id counter** for card `uid`s, `oid`s, ability and trigger `sid`s, consumed in the same order (ids show up in labels and views).
- **Ordering.** Python dicts are insertion ordered: the mana pool and combat `blocks` are ordered vectors with the same insert / delete semantics. Python sets are only used for membership. Dedupe passes (`_dedupe_by_name`, `_dedupe_by_equiv`, trigger keys) keep first occurrences in the same order. `itertools.permutations` order is reproduced for Ponder; `_compositions` for damage assignment.
- **Live vs last-known references.** Triggers and abilities hold either the live card or a snapshot, exactly where Python does (`Src`), so e.g. Gixian Infiltrator's counter, Krark-Clan Shaman's deathtouch damage and Lembas's shuffle see the same object.
- **Text.** Labels, prompts and log lines are byte-identical: `str.capitalize`, `ManaCost.__str__`, Python `repr` of lists of names (`'...'` vs `"..."` quoting) and of dicts.
- **Features.** `featurize` hashes the same strings with the same CRC32 as `zlib.crc32`.
- **Errors.** A `RulesError` in either engine surfaces as `RulesError`; a decision with no options, stepping a finished game and out-of-range indices raise the same exceptions.

## Differential testing

`mtg_ml/difftest.py` plays a `trace.Scenario` (seed, agents per seat, match game 1-3, starting player, mulligans on/off, turn limit) in both engines in lockstep. Agents choose on the reference game; the same index goes to both. Compared **before every step**:

- the decision: player, kind, prompt, all labels and keys in order;
- `observe()` for both players, `state_features()` for both players, `featurize()` for the decider and the event hashes of every option;
- the full hidden state: libraries in order, every card's ids, zone, owner, controller, damage, counters, temporary effects and `known_to`, the stack with targets / sources / data, pending triggers, combat, the RNG position;
- for scripted-bot seats, the bot's choice computed on each engine (the bots read the object model through the proxies).

After the game: winner, end reason, turns, the action list, the full log and the full RNG state. Every 97 steps (`--fork-every`), `fork()` (plus one step on the copies), `copy()` against the replay fork within each engine, and `determinize()` for both viewers are compared too, plus one step on the re-dealt games. Agents: `random` (`RandomAgent`), `chaos` (uniform over everything, mulligans 30% of the time), `bot` (scripted bots).

```bash
pytest tests/test_difftest.py                                     # fast subset (36 scenarios, ~30 s)
python -m mtg_ml.difftest fuzz --games 5000 --jobs 16             # long fuzz
python -m mtg_ml.difftest fuzz --games 500 --with-card "1:Vapor Snag:4"   # fuzz a card that is in no deck yet
python -m mtg_ml.difftest repro difftest-failure.json             # replay a saved divergence
```

On a mismatch the fuzzer prints the path to the first differing value (e.g. `.state.players[1].library[7][17]: python=[1] native=[]`), then **minimizes**: it replaces earlier choices with option 0 wherever the mismatch survives, so the reproducer has as few non-default choices as possible, and saves scenario + action script as JSON.

Further guards:

- Every rules, card, bot, view and fuzz test runs on both engines (`tests/conftest.py` parametrizes those modules; `@pytest.mark.python_only` marks the tests that need reference internals; two in `tests/test_correctness_foundations.py` use it).
- `tests/test_golden.py` pins 210 recorded games of the reference engine (`python -m mtg_ml.trace check`), so refactors of the Python engine (like moving the cards to `cards.toml`) are checked for exact behavioural equivalence before the port is compared against it.
- `tests/test_card_spec.py` checks `cards.toml` against the oracle snapshot.

Status at the time of writing: the test suite passes on macOS (arm64, 412 tests including the native variants) and on h100-private (x86_64 Linux); differential fuzz: 9,600 games, 0 divergences (7,700 locally, 1,500 on h100-private, 400 with Vapor Snag swapped in).

## Benchmarks

`tools/bench_engine.py` plays whole games for a fixed time and counts decisions:

- `engine`: random choices by index only (the engine's own cost);
- `agents`: `RandomAgent` through the public API (what `python -m mtg_ml.play bench` does);
- `rollout`: the CPU work of an RL rollout worker except the network: `featurize()` for the decider, event hashes for both seats, `encode_event_hashes`, a random choice, `step`;
- `bots`: scripted bot vs scripted bot.

```bash
python tools/bench_engine.py --engine native --mode rollout --seconds 15 --cpus 48
python tools/bench_engine.py --engine native --mode rollout --seconds 15 --procs 64 --cpus 0-63
```

**h100-private** (2× Xeon Platinum 8462Y+, 64 cores, no SMT), CPython 3.12, each process pinned with `--cpus`; measured with the box otherwise idle apart from a ComfyUI server:

| single core (CPU 48), decisions/s | Python | Rust | speed-up |
|---|---:|---:|---:|
| engine | 8,764 | 189,666 | 21.6× |
| agents (`play bench`) | 8,745 | 127,550 | 14.6× |
| rollout CPU path | 4,738 | 59,667 | 12.6× |
| scripted bots | 11,361 | 23,426 | 2.1× |

| pinned processes | Python rollout | Rust rollout | Rust engine only |
|---|---:|---:|---:|
| 16 | | 1,002,879 | |
| 32 | | 1,929,133 | |
| 64 | 297,608 | 3,951,052 | 12,431,748 |

Scaling is linear (61.7k per process at 64 processes vs 62.7k at 16): games share nothing.

**Apple M-series laptop**, single core: engine 10,511 → 169,471 (16.1×), agents 10,543 → 137,437 (13.0×), rollout CPU path 5,741 → 77,871 (13.6×), bots 12,573 → 28,066 (2.2×). End to end, one rollout worker with 64 games in lockstep and the GRU policy on CPU (`rollout.run_job`): 4,989 → 22,773 decisions/s (4.6×).

The scripted bots gain least because they are Python heuristics reading the object model through proxies; the engine part of a bot game is now small.

## PyPy

The engine has no dependencies, so PyPy runs it unchanged (`cards.toml` needs `tomllib`, so PyPy ≥ 3.11).

| decisions/s, single core | CPython 3.12 | PyPy 3.11 | speed-up |
|---|---:|---:|---:|
| h100-private, engine (10 s JIT warm-up) | 8,764 | 22,069 | 2.5× |
| h100-private, agents | 8,745 | 24,306 | 2.8× |
| h100-private, rollout CPU path | 4,738 | 9,938 | 2.1× |
| h100-private, scripted bots | 11,361 | 22,328 | 2.0× |
| h100-private, `python -m mtg_ml.play bench --games 2000` (warm-up included) | 8,971 | 20,063 | 2.2× |
| laptop, `play bench --games 4000`, PyPy 3.10 before the cards.toml change | 11,689 | 30,016 | 2.6× |

PyPy is a 2-3× quick win for the engine alone, but it is **not used for training**: with local inference (the default) rollout workers import torch (`mtg_ml/rl/rollout.py` runs the policy in-process), and torch does not run on PyPy. The central inference server (`--inference server`, [inference-server.md](inference-server.md)) now exists and its workers do not import torch, so PyPy could host them, but nothing in the repo runs them on PyPy. The Rust engine is 5-9× faster than PyPy on the same work, so PyPy is now mainly interesting for Python-heavy code that has no native path (the scripted bots: 2.0× on PyPy vs 2.1× from the Rust engine).

## What limits training throughput now

Follow-up work on the rest of the rollout pipeline is in [inference-server.md](inference-server.md). The trainer was unpickling recorded samples in one process, which capped rollouts at ~150-200k decisions/s; packed samples doubled throughput (300k decisions/s with 30 workers, 520k with 60). A central GPU inference server (`--inference server`) is built and tested, and pays off once the network grows.

## Limitations

- The proxies are read-only except for the mutations listed above; changing other card attributes from Python does not reach the engine.
- `NativeGame` is not picklable (like `Game`, games are rebuilt from constructor arguments + actions: `fork(replay=True)`).
- Adding a card that needs a new op or a new rule means implementing it in both engines (see [adding-cards.md](adding-cards.md)); the differential suite is what keeps them equal.
