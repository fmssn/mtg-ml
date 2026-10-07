# Training options (next actions 2, 7, 8, 13, 14, 15)

Flags of `python -m mtg_ml.rl.train` added for the follow-ups in `docs/next-actions.md`. Apart from the value clamp (a bug fix, on by default), the defaults keep the earlier behaviour. None of them has been measured in a full run yet.

## Value targets

- `--value-clamp 1.0` (default on; `0` turns it off): `rollout._finish` clamps the recorded values to ±(value_clamp + shaping) before the GAE deltas and uses the clamped values for the returns. The raw values stay in the trajectory. The unbounded linear head reached +1.41 / −1.12, which biased the advantages near the end of a game. The `+ shaping` margin is there because life shaping (annealed to 0) moves the true returns up to about that far past ±1.
- `--value-bound tanh`: squashes the value head into (−1, 1) (model and stacked inference forward). It adds no weights. The flag applies to new runs and to `--init`. A resumed run keeps the architecture stored in its checkpoint. `PolicyNet.config` has a `value_bound` key only when it is set, so old checkpoints and configs are unchanged.

## Learning-rate anneal

- `--lr-anneal-games N` (`0` = constant lr), `--ppo-lr-final 3e-5`, `--lr-schedule linear|cosine`. The lr moves from `--ppo-lr` to `--ppo-lr-final` over N training games and then stays at `--ppo-lr-final`. Counting starts from the run's `games_total` when annealing was first switched on. That is the start of the run, or the resume point for an older run resumed with the flag. The start point (`lr_anneal_origin`) is stored in `latest.pt`, so resuming after a crash continues the schedule. Each metrics row has `lr`.
- How the lr is set: with annealing on, the optimizer's lr is a one-element tensor on the training device (`make_optimizer(tensor_lr=True)`), and `ppo.set_lr` updates it in place before each update. The captured CUDA graphs of the update read the lr from device memory. The graph fingerprint therefore uses the tensor's address, not its value, so a new lr each iteration does not trigger a recapture. A float lr is still part of the fingerprint, and changing it recaptures, as before.
- Suggested values from next-actions item 7: `--lr-anneal-games 1000000 --ppo-lr-final 3e-5`, and possibly `--ppo-epochs 2 --ppo-minibatch 4096`.

## Discount per game turn

- `--gamma-turn G`, `--lam-turn L` (`0` = off: per-decision `--gamma` / `--lam`). Each recorded decision stores the game's turn counter (`Trajectory.turns`; every player's turn counts as one). Between decisions t and t+1, the discount is G ** (turn[t+1] − turn[t]), and lambda is L ** (the same number). Decisions within one turn are therefore not discounted against each other. The shaping potential term uses the same per-step discount. The two flags work independently. For example, `--gamma-turn 0.97` with the default `--lam 0.95` discounts gamma per turn and lambda per decision.
- Suggested value: `--gamma-turn 0.97 --lam-turn 0.95` (about 6 decisions per turn).

## Game mix

- `--postboard-frac 0.2` (item 13): the flag already existed and its default is still 0.5.
- `--matchup a[:w],b[:w],...`: a matchup mix. Each training game draws its matchup by weight (default 1; `match.MATCHUPS`: `jund_blue`, `jund_madness`, `blue_madness`), so one network learns all three decks. The first matchup gets the full evaluation (keys unchanged); every other one a benchmark of each seat against the other deck's bot (`bench/<matchup>/<deck>_vs_bot*`). Logged per matchup: `matchup_share/<m>`, `seat0_wins_selfplay/<m>`. Needs feature set 3 (`opp:deck:`), else seat 0 cannot tell Blue from Red: pass `--features 3` with `--init` from an older checkpoint. Not with exploiter mode.
- `--bot-seat both|jund` (item 15): `jund` makes every bot game the benchmark matchup, with the learner as Jund (seat 0) against the blue bot. Bot games keep coming out of the pool share (`--bot-frac`), and only the learner's seat is recorded. Suggested: `--bot-frac 0.25 --bot-seat jund`.
- `--pool-sampling uniform|pfsp`, `--pfsp-power 2`, `--pfsp-ema 0.05` (item 14): the pool games that do not go to the newest snapshot (`1 − pool_recent_frac`) pick snapshot i with weight (1 − p_i) ** power. Here p_i is a running estimate of the learner's win rate against snapshot i. It starts at 0.5, each game moves it `pfsp_ema` toward the result (1 win, 0.5 draw, 0 loss), and it is stored in `latest.pt`. If every weight is 0, the pick falls back to uniform.

## Exploiters

- `--exploit MAIN.pt --exploit-deck jund|blue`: every training game puts the learner on that deck against the frozen policy file `MAIN.pt`, with no self-play, bot or pool opponents. A fresh exploiter run starts from `--init` if given, otherwise from `MAIN.pt` itself. Each metrics row has `win_vs_main` (the training games of that iteration, sampled play). Pool snapshots are still written, and the evaluation still runs.
- Cheapest test (item 14): from the final checkpoint, train one Jund and one Blue exploiter for 150 iterations each. A `win_vs_main` of 0.60 or more confirms gaps that can be exploited:

```bash
python -m mtg_ml.rl.train --run runs/exploit-jund --exploit runs/main/policy/vNNNNN.pt --exploit-deck jund --iterations 150
python -m mtg_ml.rl.train --run runs/exploit-blue --exploit runs/main/policy/vNNNNN.pt --exploit-deck blue --iterations 150
```

- `--init PATH` (any mode): a fresh run (no `latest.pt` yet) starts from the weights and architecture of a policy file or checkpoint. Flags such as `--hidden` are ignored. `--value-bound` is applied on top, and so is `--entity-attn N`: attention layers the source lacks start as the identity (`model.load_partial`), so the first rollouts play like the source. The optimizer state is not loaded. A resumed run takes its architecture from `latest.pt` and refuses an `--entity-attn` that differs from it.
