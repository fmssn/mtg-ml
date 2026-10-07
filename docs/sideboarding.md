# Sideboarding

Matches are best of three (`mtg_ml/match.py`). Game 1 is played with the maindecks; before games 2 and 3 each seat makes one **sideboarding decision**: which cards of its 15-card sideboard come in, and which maindeck cards go out.

## Data

| what | where |
|---|---|
| maindecks and 15-card sideboards | `mtg_ml/engine/decks.py`: `DECKS`, `SIDEBOARDS` |
| the plan table: one row per ordered (deck, opponent) | `mtg_ml/engine/sideboard_plans.toml` |
| loader, validation, `SideboardPlan` | `mtg_ml/engine/sideboard.py` |
| the decision step, policies | `mtg_ml/match.py`: `SideboardPolicy`, `PlanMatrixPolicy`, `play_match(policies=...)` |

A row of the plan table:

```toml
[[plan]]
deck = "jund_wildfire"            # key of decks.DECKS
opponent = "mono_blue_terror"     # key of decks.DECKS
why = "one line: what the swap is for"
in = { "Duress" = 3, "Pyroblast" = 2 }               # from SIDEBOARDS[deck]
out = { "Lembas" = 3, "Nyxborn Hydra" = 1, "Toxin Analysis" = 1 }   # from DECKS[deck]
```

Counts are always relative to the registered maindeck. At import every row is validated: both decks exist, `out` names only maindeck cards and `in` only sideboard cards in the counts available, as many cards in as out, the result is 60 cards with at most 4 copies of any non-basic. `tests/test_sideboard.py` requires a row for every ordered pair of decks, so **a new deck adds one row against each existing deck and one row for each existing deck against it** (and its 60 + 15 in `decks.py`).

Plans are applied in Python: both engines just receive the resulting decklists, so nothing about sideboarding lives in the Rust port. Every sideboard card must be implemented in both engines like any other card (docs/adding-cards.md), and the deck's scripted bot must know how to cast it (bots never cast unknown cards).

## The decision step

```python
class SideboardPolicy(Protocol):
    def choose(self, deck: str, opponent: str, game_no: int, history: Sequence[SeatGame]) -> SideboardPlan: ...
```

`history` is the seat's own view of the earlier games of the match (`MatchResult.history(seat)`), one `SeatGame` per game:

| field | |
|---|---|
| `game_no`, `on_the_play` | game number and whether this seat started |
| `result`, `reason` | `"win"` / `"loss"` / `"draw"`, how the game ended |
| `plan` | the plan this seat played that game with |
| `opponent_seen` | the opponent's cards this seat could name at the end of the game: graveyard, exile, nontoken permanents, hand cards revealed to it (sorted names; cards that went back to a hidden zone are missed) |

`play_match` validates the returned plan (same rules as the table) before building the decks. Policies shipped:

- `PlanMatrixPolicy` (default): the table's row, whatever the history. Because it ignores the history, the batched paths that build games from a `GameSpec` (training's `--postboard-frac`, the benchmark's best-of-three in `rl/evaluate.py`) use `match.game_args(game_no, matchup)`, which reads the same table, and play the same decks as `play_match`.
- `NoSideboardPolicy`: always the maindeck (a baseline for measuring what the plans are worth).

## A learned policy (not built yet)

The intended drop-in replacement for `PlanMatrixPolicy`:

- **Action space:** a set of swap pairs (sideboard card in, maindeck card out) under the 60/15 constraint, i.e. a plan with `sum(in) == sum(out)`. Practically: a sequence of up to K (about 8) pair choices from the masked list of legal (in, out) pairs plus a "done" action, each choice updating the counts so later masks stay legal. Every sequence ends in a plan that `validate_plan` accepts.
- **Observation:** deck, opponent, game number, on the play or draw, and the `history` above (results, the plan played, `opponent_seen` as a bag of card names, hashed like the game features).
- **Reward:** the result of the game played with the chosen plan (game 2, game 3), from the seat's side; a match-level variant credits both decisions with the match result.
- **Training:** the decision happens once per game, so it is cheap to collect alongside ordinary self-play; the in-game policy should see sideboarded decks of every kind (keep a share of `PlanMatrixPolicy` and random legal plans) so it does not overfit to one plan per matchup.
- **Engine paths:** history-dependent plans need `play_match`-style sequential games; the batched `GameSpec` paths would carry the chosen decklists in the spec instead of deriving them from `match_game`.
