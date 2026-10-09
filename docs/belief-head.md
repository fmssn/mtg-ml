# Belief head and best-of-three knowledge (feature set 8)

Feature set 7 forgot everything between games: training sampled isolated
post-sideboard games, batched evaluation carried only match results, and live
play built a fresh model agent per game. Feature set 8 (opt-in) keeps what a
player **witnessed** of the opponent's cards for the whole match and adds a
**belief head** that predicts the opponent's archetype and the card counts of
its registered 75 from that evidence alone.

"No witnessed evidence is discarded" is the transfer guarantee. It is not
certainty about unseen cards or about the opponent's current sideboard
configuration.

## Evidence

`Game.witnessed(viewer)` (both engines, `make difftest` compares it at every
step): per opponent card name (front face, tokens excluded), the most distinct
copies visible to `viewer` at one moment of the game. Visible means the
engine's `known_to`: public zones, opponent hand cards revealed to the viewer,
library cards whose position the viewer knows, and transient reveals. The
engine records the counts before a visible card can leave visibility (a move
into the library the viewer does not follow, a shuffle of known cards), so
cards later shuffled away or put on the bottom still count. Repeat sightings
and bounce/recast never inflate a count, and no hidden identity links copies
across a shuffle. The record is monotonic within a game and copied with
snapshots; option previews run on copies and leave it unchanged.
`tests/test_witness.py` covers the counting cases on both engines.

`knowledge.MatchKnowledge` is one player's record of the finished games of a
match (`with_game(game.witnessed(seat))`), seat-relative and JSON-safe.
Copies combine across games by **maximum**, never by sum. Game 3 carries
games 1 and 2; a new match, a rematch or an unrelated game starts from
`EMPTY`.

## Model

`rl/belief.py` (`BeliefSpec`, saved in the checkpoint config as `belief`):
the exact card vocabulary (every registered non-token card, sorted; no
hashing) and archetype order. Each decision's evidence is a list of
(card, slot, copies) triples: slot 0 is the current game, slot g the finished
game g, so earlier games are never read as cards in hand or library now.
Copies are capped at 4 (75 for basic lands).

`PolicyNet(features=8, belief=BeliefSpec)` adds `BeliefNet`, which reads only
those triples (card, slot and count embeddings, pooled, MLP) and predicts
archetype logits and per-card count distributions (0..4, basics 0..75). The
policy core gets a zero-initialised projection of the **detached**
(archetype probabilities, expected counts), so PPO gradients never reach the
belief branch. `forward(..., aux=True)` returns the predictions as a fourth
item (`BeliefOut`); the default return is unchanged. `ModelAgent.last_info`
includes the archetype probabilities.

The belief head never receives the opponent's registered list, archetype,
matchup, seed or sideboard choice. The sampled archetype and registered 75 are
supervised **targets** only (`Result.belief_targets`, one per recorded
trajectory, from `rollout.registered_lists`), and travel in training data,
never into inference. Loss: `0.1 * CE(archetype) + 0.1 * mean over cards
CE(count)` (`belief_coef` in the config, `--ppo-belief-coef a,c` overrides).

Hashed set-8 features are identical to set 7; only the evidence channel and
the head are new.

## Matches

`GameSpec(bo3=True)` is a whole best-of-three: the worker plays games 1, 2
and 3 (stopping at two wins, draws count as games) with fixed registrations,
policies, variant lists and seat mapping, the loser of a game starting the
next (after a draw the same player). Each game gets fresh trajectories and a
fresh recurrent state; each player's `MatchKnowledge` moves to the next game
inside the worker before its first mulligan decision, independent of worker
placement or inference slot. Rewards, GAE and trajectories stay per game. Game
seeds are `match.game_seed(match seed, n)`; `Result.matches` reports the
matches next to the per-game rows.

Used by:
- **training**: `--match-rollouts 1` plays about `games_per_iter / 2.5`
  matches per iteration; budgets and schedules count the games actually
  played and `matches` is logged separately. Isolated games stay the default
  (legacy) mode.
- **batched evaluation**: `evaluate.head_to_head_bo3` hands each match to the
  rollout as one spec instead of scheduling rounds.
- **`match.play_match`**: calls `agent.set_knowledge(k)` before each game.
- **live play**: the match dict carries the model's knowledge between games;
  a concession keeps what was seen, a rematch starts empty.

## Deck variants

`--variants train` samples each canonical position's registered 75 from the
frozen manifest (`docs/deck-variants.md`) once per match; sideboarding only
repartitions it. Dev and test splits are held out for evaluating the belief
head.

## Training

```bash
# new run
python -m mtg_ml.rl.train --run runs/b1 --belief 1 --match-rollouts 1 --variants train --matchup <mix>
# upgrade a set-7 checkpoint: same weights, new belief head, zero projection
python -m mtg_ml.rl.train --run runs/b2 --init <set7>.pt --belief 1 --match-rollouts 1 --variants train
```

A belief head requires varied-list match rollouts (without `--exploit`); the
multi-device learner (`rl/distributed.py`) does not carry belief targets yet.
Logged: `belief/*` losses, archetype accuracy, count accuracy and absolute
error per card.

`tools/belief_smoke.py` is the learning smoke: it plays matches on train and
held-out variants with the scripted bots, trains only the belief branch, and
reports held-out archetype log loss (against the training prior), accuracy,
calibration (ECE) and per-card count error, by game of the match, plus how
distinguishing evidence moves the prediction. Claims of stronger play need a
specialist gameplay evaluation (`docs/benchmark-plan.md`).
