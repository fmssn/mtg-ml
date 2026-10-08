# mtg-ml

A rules engine and self-play environment for **Magic: The Gathering (Pauper)**, built so a reinforcement-learning agent can be trained on one fixed matchup:

**Jund Wildfire** (player 0) vs. **Mono Blue Terror** (player 1). Games use the London mulligan. Matches are best of three, with real 15-card sideboards and a sideboarding decision before games 2 and 3. Every rule the cards need is implemented in full.

Background research and the roadmap: [`docs/research-mtg-pauper-rl.md`](docs/research-mtg-pauper-rl.md).

## Quickstart

```bash
pip install -e '.[dev]'          # no runtime dependencies, Python >= 3.11
python -m pytest                 # per-card rules, combat, fuzzing, golden traces (+ native variants, see below)
python -m mtg_ml.play watch --seed 3     # random vs random, full game log
python -m mtg_ml.play human --seat 1     # play Mono Blue Terror in the terminal
python -m mtg_ml.play bench --games 200  # throughput
python -m mtg_ml.play match --matches 200  # best-of-three bot matches with sideboards
python -m mtg_ml.replay record --agents bot,bot --games 10   # write replays/*.json
python -m mtg_ml.replay record --agents model:runs/x/latest.pt,bot --games 5   # a trained model vs the blue bot
python -m mtg_ml.replay serve                                # watch them at http://127.0.0.1:8765
python -m mtg_ml.replay serve --models runs/                 # ... and play against any checkpoint under runs/ ("Play vs model")

# optional: the Rust engine, 13-22x faster, identical games (docs/native-engine.md)
(cd native && maturin develop --release)
python -m mtg_ml.play bench --games 2000 --engine native   # or MTG_ENGINE=native for everything
```

```python
from mtg_ml.engine import Game, expand, JUND_WILDFIRE, MONO_BLUE_TERROR

g = Game((expand(JUND_WILDFIRE), expand(MONO_BLUE_TERROR)), seed=1)
while not g.over:
    d = g.decision                     # who decides, what kind, prompt
    for i, opt in enumerate(d.options):
        print(i, opt.label, opt.key)   # every legal choice, nothing else
    g.step(0)
print(g.winner, g.end_reason)
```

## Design

**Every rules choice is an explicit decision.** The engine is one Python generator that pauses at each choice. Decision kinds (`mtg_ml/engine/objects.py`):

| kind | examples |
|---|---|
| `priority` | pass, play land, cast (normal / bestow / flashback / escape), activate abilities, cycling, sacrifice an Eldrazi Spawn for mana |
| `choose_x`, `target` | X for Nyxborn Hydra; targets, re-checked on resolution (fizzling) |
| `pay_mana` | which source pays which part of the cost, one mana at a time, including floating mana |
| `sacrifice`, `exile_from_graveyard` | additional costs (Fanatical Offering, Krark-Clan Shaman, escape) |
| `yes_no`, `choose_card`, `choose_mode`, `order` | Delver reveal, Force Spike / ward / Spellbomb payments, discards, Brainstorm put-backs, searches, Ponder order, scry, explore, Deem Inferior |
| `order_triggers` | ordering your own simultaneous triggers (APNAP) |
| `declare_attacker`, `declare_blocker`, `assign_damage` | combat |

**No dead ends.** An option is offered only if it can be completed legally. Cost feasibility is a bipartite mana matching that also checks sacrifice costs. Example: an Eldrazi Spawn is not offered as mana when it is the only thing left to sacrifice.

**No duplicate options.** Indistinguishable objects (same name, controller and state, not targeted or in combat) are offered once. Attack declarations are enumerated as canonical subsets, so every legal attack has exactly one decision path. Plain lands are tapped only while paying a cost and never "floated" at priority, because floating land mana has no effect that tapping during payment lacks. Mana abilities with side effects (sacrificing Spawn) are offered at priority. With `auto_single=True` (the default), forced single-option decisions are taken automatically.

**Ideas taken from MageZero:**
- decision types map to separate policy heads;
- `Option.key` is a stable, id-free action descriptor for per-deck action vocabularies;
- `mtg_ml/encode.py` produces hashed sparse state features from what the player can see;
- `Game.fork()` (exact copy by deterministic replay) and `view.determinize()` (re-sample hidden cards consistently with one player's knowledge) are what MCTS / AlphaZero-style search needs.

**Hidden information.** Every card tracks who knows it (`known_to`). `view.observe(game, player)` exposes:
- public zones;
- your own hand;
- the opponent's hand size plus any of their cards you have seen;
- library cards you learned through Brainstorm, Ponder, Delver, scry, Deem Inferior or explore.

Shuffling forgets library knowledge.

## Native engine

`native/` is a Rust port of the engine (PyO3 / maturin, module `mtg_ml_native`) that plays exactly the same games as the Python engine: same seeds, same decision sequence, same options with the same labels and keys, same views and the same RNG stream. Python stays the reference; the native engine is an optional backend chosen with `--engine native`, `engine=` or `MTG_ENGINE=native` (`mtg_ml/backend.py`). Training uses it with `python -m mtg_ml.rl.train --engine native`.

- **Speed** (one core of h100-private): 190k decisions/s engine-only (21.6x), 60k decisions/s for a rollout worker's CPU work of featurize + event tokens + step (12.6x); 3.95M decisions/s on 64 pinned cores.
- **Correctness** is checked differentially: `mtg_ml/difftest.py` plays scenarios (random, chaos and bot agents, mulligans, games 1-3) in both engines in lockstep and compares every decision, view, feature vector and the full hidden state at every step; all rules, card, bot and fuzz tests also run on both engines.
- **Cards** are one declarative spec, `mtg_ml/engine/cards.toml`, loaded by both engines. A card that only uses existing effect ops needs no code. How to add cards and decks: [`docs/adding-cards.md`](docs/adding-cards.md).
- **PyPy** runs the Python engine 2-3x faster, but rollout workers import torch, so it only becomes usable once inference moves to a central server.

Architecture, determinism, the differential suite and all benchmarks: [`docs/native-engine.md`](docs/native-engine.md). Rollout throughput, packed samples and the central GPU inference server (`--inference server`): [`docs/inference-server.md`](docs/inference-server.md).

## Mulligans, sideboards and matches

- **Mulligan** (`Game(mulligans=True)`, the default): London mulligan (CR 103.5) with no free mulligan. Each decision is a `mulligan` decision ("Keep" / "Mulligan"), starting with the starting player. Once a player keeps, they put one card per mulligan on the bottom of their library: one `choose_card` decision per card, private to that player. Mulligan counts are public (`observe()["self"/"opponent"]["mulligans"]`). `RandomAgent` always keeps. The bots keep 7 with 2–5 lands (Jund) or 2–4 lands (Blue; one land is fine with two cantrips), and always keep 5.
- **Sideboards** (`mtg_ml/engine/decks.py`): each deck's typical 15 from the Q3 2026 tournament data. What comes in and out against each opponent is a table, `mtg_ml/engine/sideboard_plans.toml` (one row per deck pair, with the reason); [docs/sideboarding.md](docs/sideboarding.md) has the format and the policy interface.
  - Jund vs Blue brings in 3 Duress, 2 Red Elemental Blast, 1 Faerie Macabre, 1 Nihil Spellbomb and 1 Terminate for 2 Lembas, 3 Cleansing Wildfire, 1 Makeshift Munitions, 1 Nyxborn Hydra and 1 Toxin Analysis.
  - Blue vs Jund brings in 4 Annul and 2 Hydroblast for 3 Force Spike, 2 Sleep of the Dead and 1 Deem Inferior.
  - Modal spells ("choose one") are cast as separate options, for example `Cast Red Elemental Blast (counter)` with key `(..., "normal", "counter")`.
- **Matches** (`mtg_ml/match.py`):
  - game 1 uses the maindecks and a random starting player;
  - before games 2 and 3 each seat's `SideboardPolicy` picks its plan (default `PlanMatrixPolicy`: the table's row), and the loser of the previous game plays first (after a draw, the previous starting player starts again);
  - a match ends at two game wins or after three games.
  - `Game.match_game` and the `postboard:` state feature tell agents which decks are in play.
- **Training:** `--postboard-frac` (default 0.5) sets the share of training games played with sideboarded decks. Each evaluation includes `--eval-bo3-matches` best-of-three matches against the bots. `python -m mtg_ml.rl.evaluate CKPT bot --bo3` runs that evaluation on its own.

Bot vs bot, 200 matches each (2026-10-07, 8-core dev Mac, native engine; re-run after the sideboard fidelity fix): Jund wins 30.0% of game 1 and 34.8% of games 2/3 against Blue (22.5% of games 2/3 without sideboarding); Blue beats Red in 63.8% of games 2/3 (35.0% of game 1). The Q3 2026 match data puts Jund at 46% against Blue, so the bots still underplay Jund; the card rules themselves have been audited.

## Scripted bots

`mtg_ml/bots/` holds one heuristic bot per deck, used as a fixed baseline opponent. Each bot scores every legal option and takes the best one.

- **Jund** (`jund.py`):
  - plays Bridges early, while entering tapped costs nothing;
  - casts its artifacts and creatures on curve;
  - casts Cleansing Wildfire on its own indestructible Bridges (ramp plus a card);
  - fires Cast Down at the biggest threat, preferably once Blue is down to fewer than UU untapped;
  - feeds its sacrifice-to-draw spells with cheap fodder;
  - uses Nihil Spellbomb when Blue's graveyard fuels Terror, Serpent or escape.
- **Blue** (`blue.py`):
  - casts Delver early and cantrips to dig, milling itself to make Terror and Serpent cheaper;
  - counters removal aimed at its threats, creatures and card draw, but never a spell that Terror's ward will counter anyway;
  - plays Force Spike when Jund is tapped out;
  - spends spare mana on card draw at Jund's end step.
- **Shared** (`base.py`):
  - combat: attacks unless a blocker kills the attacker for free; blocks for free kills, good trades and chumps against lethal;
  - mana payment, sacrifice fodder and the remaining card choices.
- **Hidden information:** the bots are deterministic, and a test checks that re-sampling hidden cards (`determinize`) never changes a decision.
- **Tuning:** each judgment call is a named constant at the top of its file.

| matchup (paired seeds) | Jund score |
|---|---|
| Jund bot vs random | 0.98 |
| random vs Blue bot | 0.00 |
| Jund bot vs Blue bot | 0.33 |

**Search bot** (`mtg_ml/bots/search.py`): determinized flat Monte Carlo on top of a scripted bot. At each key decision it takes the base bot's top 4 options and plays N games to the end from the current position for each one. Every playout re-deals the cards this player can't see (`determinize`, which now depends only on the player's knowledge), and both sides play on as the scripted bots. All options are scored on the same re-dealt worlds. Each playout costs about 20 ms, so this is meant for evaluation and for larger machines (`--agents search:64,bot`), not for training-scale self-play.

**Jund strength work so far.** Diagnostics showed that Jund often tapped out on its own turn or kept no black source for instant-speed removal, and held unused sacrifice spells while flooded. The Jund bot now pays with colour awareness (`colour_needs`) and keeps {1}{B} open for removal it holds (`HOLD_REMOVAL_MANA`). It also treats spare lands as sacrifice fodder once it has 6 or more (`SPARE_LAND_COUNT`). None of this changed the bot-vs-bot rate beyond noise (31–33%). An 8-playout search Jund scored 27% (CI 18–39%, 70 games), and 8 playouts are too few to separate options. Whether Jund can reach the real-world 55–61% therefore needs much more search, or a learned policy.

```bash
python -m mtg_ml.play watch --agents bot,bot --seed 2
python -m mtg_ml.play human --seat 1 --opponent bot
python -m mtg_ml.rl.evaluate runs/ppo1/latest.pt bot
```

## Training (masked PPO self-play)

```bash
pip install -e '.[dev,rl]'
python -m mtg_ml.rl.train --run runs/ppo1 --iterations 200   # rerun with a higher --iterations to resume
python -m mtg_ml.rl.train --help                              # all hyperparameters
```

- **Inputs** (`mtg_ml/rl/features.py`): the hashed state features from `encode.py`, plus seat and decision kind. Each legal option's key is expanded into hashed tokens: its elements and its prefixes.
- **Network** (`mtg_ml/rl/model.py`): an EmbeddingBag + MLP state trunk and an option encoder. A scorer turns each (state, option) pair into one logit. Logits are padded per decision and masked to `-inf`, so the softmax covers exactly the legal options. A value head sits on the trunk.
- **Memory** (`--memory gru`, the default): a GRU runs over each player's own decisions in a game. At every decision it reads the current state plus an event bag of what happened since that player's last decision: their own previous choice, and the opponent's public actions (casts, attacks, blocks, targets, payments). Opponent choices that could name hidden cards (`choose_card`, `order`, `choose_mode`) are passed only as their kind. This is also how a target or payment decision knows which spell it belongs to. PPO replays whole trajectories, so hidden states are never stale. `--memory none` swaps the GRU for an MLP over the same inputs, for ablations.
- **Rollouts** (`mtg_ml/rl/rollout.py`): worker processes run many games in lockstep and batch inference per policy. Rewards are terminal ±1 (0 for a draw), plus optional potential-based life-difference shaping that is annealed to 0. Advantages use GAE.
- **Loop** (`mtg_ml/rl/train.py`): one deck-conditioned network plays both seats. Half the games are pure self-play; the other half are against a frozen checkpoint from `pool/` (50% the newest, otherwise uniform). Every 10 iterations the loop takes a snapshot and evaluates against the random agent and the first checkpoint, playing both seats on paired seeds with Wilson CIs. Metrics are written to `metrics.jsonl`.

Head-to-head matches between checkpoints, or against the random agent, use paired seeds:

```bash
python -m mtg_ml.rl.evaluate runs/a/latest.pt runs/b/latest.pt --games 600
```

On 7 workers a run does about 10k decisions/s (`--memory gru` takes about 60% longer per iteration than `none`, mostly in the update). Twenty iterations (about 4 minutes) take Jund from 18% to over 90% against the random agent. Blue already beats random about 98% of the time. Against the scripted bots the same agent scores only about 0.03 as Jund and 0.40 as Blue, so the bots are the benchmark to beat. The bot-vs-bot baselines are 0.33 and 0.67. `--bot-frac` mixes bot games into training.

## Rules coverage

Implemented:
- full turn structure, with the starting player skipping their first draw;
- priority with "all pass in succession" resolution, the stack (LIFO), instant vs. sorcery timing and one land per turn;
- mana pools that empty between steps;
- state-based actions: life, drawing from an empty library, lethal and deathtouch damage, 0 toughness, unattached bestow Auras;
- triggered abilities (ETB, dies, cast, sacrifice, upkeep, ward) put on the stack in APNAP order;
- new object on zone change (CR 400.7) and last-known information;
- combat: summoning sickness, flying/reach, multiple blockers with the current free damage division, trample's lethal-first rule, deathtouch, lifelink;
- cleanup discard to seven, damage and "until end of turn" effects wearing off;
- alternative and additional costs: flashback (exiled afterwards), escape, bestow (including the illegal-target fallback to a creature), cycling and islandcycling, X costs, affinity, the other cost reductions, and sacrifice costs;
- ward, transform, indestructible, tokens (Eldrazi Spawn, Clue, Map) and explore.

Card list and oracle text: [`mtg_ml/engine/decks.py`](mtg_ml/engine/decks.py), [`mtg_ml/engine/cards.toml`](mtg_ml/engine/cards.toml), [`data/oracle_cards.json`](data/oracle_cards.json). The lists are from MTGGoldfish (fetched 2026-10-05).

**Not implemented**, because no card in the pool needs it: first strike, vigilance, planeswalkers, control change, copies, replacement effects, layers beyond counters, auras and temporary effects, and the legend rule. The engine is deliberately limited to this card pool. A turn limit (default 100) ends a game as a draw, so self-play always terminates.

## Testing

- `tests/test_cards_*.py` and `tests/test_rules.py` set up exact positions and assert both the resulting state and the exact set of offered options (helpers in `tests/helpers.py`).
- `tests/test_fuzz.py` plays random games and checks three things:
  - invariants hold: card conservation, zone consistency, no negative mana;
  - from sampled positions, every offered option can be taken without a rules error;
  - random play reaches every card, alternative cost and token ability.
- `tools/audit_triggers.py` plays random games and re-derives from the log alone which triggers should have fired (ETB, dies, cast, sacrifice, upkeep, ward), then checks them against what the engine put on the stack. On 100 games: 1,121 triggers expected, 1,121 stacked, none missed or spurious. Run `python tools/audit_triggers.py --games 100 --out logs/` to also keep the logs.
- `tests/test_view_env.py` checks that observations and features never depend on hidden cards and that determinization keeps a player's view intact.
- `tests/test_golden.py` replays recorded games and compares digests of every decision, view, log line and outcome (`python -m mtg_ml.trace check`), so engine refactors are checked for exact equivalence.
- `tests/test_card_spec.py` checks `cards.toml` against the oracle snapshot.
- The rules, card, bot, view and fuzz tests run on both engines (`tests/conftest.py`); `tests/test_difftest.py` and `python -m mtg_ml.difftest fuzz --games 5000 --jobs 8` compare the engines in lockstep.

Throughput on one CPU core of h100-private: about 8,800 decisions/s (Python engine), 190,000 decisions/s (native engine). `python tools/bench_engine.py --help` measures engine-only, agent, rollout and bot throughput, optionally on many pinned cores.

## Layout

```
mtg_ml/engine/   mana.py objects.py game.py cards.toml cards.py decks.py view.py native.py (NativeGame)
mtg_ml/backend.py  engine switch (MTG_ENGINE)
mtg_ml/trace.py    reproducible scenarios, golden digests
mtg_ml/difftest.py differential testing of the two engines
mtg_ml/encode.py hashed state features, action keys
mtg_ml/env.py    two-player step/reset wrapper
mtg_ml/agents.py random and human agents, game runner
mtg_ml/match.py  best-of-three matches with sideboarding
mtg_ml/play.py   CLI
mtg_ml/bots/     base.py jund.py blue.py (scripted baseline bots)
mtg_ml/rl/       features.py model.py rollout.py ppo.py train.py evaluate.py (masked PPO self-play)
native/src/      rng.rs mana.rs cards.rs state.rs engine.rs game.rs features.rs py.rs (Rust port)
tools/           audit_triggers.py bench_engine.py
docs/            research-mtg-pauper-rl.md native-engine.md adding-cards.md
```
