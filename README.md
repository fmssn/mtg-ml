# mtg-ml

A rules engine and self-play environment for **Magic: The Gathering (Pauper)**, built so a reinforcement-learning agent can be trained on one fixed matchup:

**Jund Wildfire** (player 0) vs. **Mono Blue Terror** (player 1). Maindecks only, no sideboarding, no mulligans. Every other rule the 35 cards need is implemented in full.

Background research and the roadmap: [`docs/research-mtg-pauper-rl.md`](docs/research-mtg-pauper-rl.md).

## Quickstart

```bash
pip install -e '.[dev]'          # no runtime dependencies, Python >= 3.11
python -m pytest                 # 75 tests: per-card rules, combat, fuzzing
python -m mtg_ml.play watch --seed 3     # random vs random, full game log
python -m mtg_ml.play human --seat 1     # play Mono Blue Terror in the terminal
python -m mtg_ml.play bench --games 200  # throughput
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

Card list and oracle text: [`mtg_ml/engine/decks.py`](mtg_ml/engine/decks.py), [`data/oracle_cards.json`](data/oracle_cards.json). The lists are from MTGGoldfish (fetched 2026-10-05).

**Not implemented**, because no card in the pool needs it: first strike, vigilance, planeswalkers, control change, copies, replacement effects, layers beyond counters, auras and temporary effects, and the legend rule. The engine is deliberately limited to this card pool. A turn limit (default 100) ends a game as a draw, so self-play always terminates.

## Testing

- `tests/test_cards_*.py` and `tests/test_rules.py` set up exact positions and assert both the resulting state and the exact set of offered options (helpers in `tests/helpers.py`).
- `tests/test_fuzz.py` plays random games and checks three things:
  - invariants hold: card conservation, zone consistency, no negative mana;
  - from sampled positions, every offered option can be taken without a rules error;
  - random play reaches every card, alternative cost and token ability.
- `tools/audit_triggers.py` plays random games and re-derives from the log alone which triggers should have fired (ETB, dies, cast, sacrifice, upkeep, ward), then checks them against what the engine put on the stack. On 100 games: 1,121 triggers expected, 1,121 stacked, none missed or spurious. Run `python tools/audit_triggers.py --games 100 --out logs/` to also keep the logs.
- `tests/test_view_env.py` checks that observations and features never depend on hidden cards and that determinization keeps a player's view intact.

Throughput: about 6,000 decisions/s, or 25 random games/s, on one CPU core. That is enough for engine validation and small experiments. For large-scale training the plan is a faster port (e.g. Rust) checked against this engine by replaying the same seeds and action sequences in both.

## Layout

```
mtg_ml/engine/   mana.py objects.py game.py cards.py decks.py view.py
mtg_ml/encode.py hashed state features, action keys
mtg_ml/env.py    two-player step/reset wrapper
mtg_ml/agents.py random and human agents, game runner
mtg_ml/play.py   CLI
```
