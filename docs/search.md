# Own-turn search

`mtg_ml/rl/search.py`. Trainer flags `--search-budget N` (0 = off), `--search-frac`,
`--search-depth`, `--search-root`, `--search-floor`; evaluation `--search-budget`;
`ModelAgent(path, seat, search=SearchConfig(...))` for recorded games.

## Why

PPO improves only the actions it samples. Replay `model_v3-entity-500k-bot-s6`,
frame 168 (turn 13, Jund to move, blue tapped out with a 6/5 Cryptic Serpent):
Jund holds Krark-Clan Shaman and Vault of Whispers with five untapped sources, an
Eldrazi Spawn and an Ichor Wellspring on the battlefield, and Toxin Analysis is
castable. The line is cast the Shaman, give it deathtouch with Toxin Analysis,
sacrifice the Wellspring: every creature without flying dies, including the
Serpent, and the Wellspring draws a card. The model passed (67%), and when it
did cast Toxin Analysis it targeted the Spawn.

Probing the checkpoint along the forced line shows what was learned and what
was not: with the Shaman on the battlefield the policy casts Toxin Analysis on
it at 100%, but then activates the sweep with probability 1.3e-7, *lower* than
with deathtouch on a Spawn (logit -8.8 vs -1.3). The network reads the
`e:kw:deathtouch` entity feature (it moves the logit by 7), it has just never
been rewarded for the sweep: the sweep without deathtouch kills its own board,
so the policy learned "never sweep" before the deathtouch variant was ever
sampled, and at 1e-7 it never will be. More games do not fix this. The critic is
nearly as blind: it rates the board after the sweep 0.07 better than passing.
The scripted Jund bot has the same hole (`sweep_value` ignores deathtouch).

## What the search does

At the learner's own-turn decisions in a main phase with at least two non-pass
options (`eligible`), a small Gumbel AlphaZero search (Danihelka et al. 2022)
runs on a determinized fork of the game:

- **Tree.** Nodes are the searcher's decisions of the branching kinds (priority
  with an empty stack, targets, sacrifices, X, modes, attackers) within the
  root's phase. Mana abilities are not branched on. Everything else is played by
  the same network with argmax: mana payment, holding priority over its own
  spell, and every opponent decision (opponent recurrent state starts at zero;
  the hidden cards come from one `view.determinize`). Passing out of the phase
  hands the rest of the turn to the policy: searching main 2 from main 1 would
  duplicate the whole tree one phase later.
- **Leaves.** A path ends at the searcher's first decision after the turn passed
  to the opponent (`horizon="turn"`, usually priority in their upkeep) or, with
  `"opponent"`, at its first decision of its next turn. Leaves are always the
  searcher's decisions because the value head is trained on nothing else;
  evaluating it at an opponent's decision gave +0.3 for a board worth -0.1.
- **Root.** Sequential Halving over the Gumbel top-m root actions (m = all of
  them when there are at most `max_root`), so every root action is tried.
- **Below the root.** Unexpanded children first (activated abilities, then by
  prior), then the Gumbel AlphaZero rule `argmax pi'(a) - N(a)/(1+sum N)` with
  completed Q. Leaf children are never revisited (their value is exact).
- **Backup: max** over children. Every node is the searcher's own decision in a
  deterministic tree, so a node is worth its best line. With a mean backup the
  seed-6 combo is found at budget 64 but valued -0.04 against the policy line's
  -0.06: the garbage lines in the subtree dilute it.
- **Output.** The chosen root action and the improved policy
  `softmax(prior + sigma(completed Q))` over all legal options.

## What the rollout and the update do with it

`Job.search` (set from `--search-budget`): in `_LocalEvaluator.collect` the
forward pass that the rollout makes anyway hands the root logits and the new
recurrent state to the search, which picks the action. The trajectory records
the behaviour policy's log-prob of that action (so replaying the trajectory
still reproduces the recorded log-probs) and the improved policy as a target
(`Result.target_len`, `Result.targets`). `ppo_update` trains searched
decisions with cross-entropy to that target (`--ppo-distill-coef`, default
1.0) instead of the policy-gradient term, whose action they did not sample from
the policy; they still train the value head. Metrics: `searched` (decisions
per iteration), `distill_loss`.

Server inference (`--inference server`) is not supported with search yet: the
search needs synchronous evaluations in the worker.

## Measured on the seed-6 position (h128 entity checkpoint, Python engine, M3)

| budget | backup | chosen | value of chosen line | search policy on Shaman | time | evals |
|---|---|---|---|---|---|---|
| policy | - | Toxin Analysis (18%) / pass (67%) | -0.064 | 0.096 | - | 1 |
| 32 | max | Cast Shaman, but line stops at the Clue | +0.06 | 0.985 | 0.6 s | 314 |
| 64 | max | Shaman, Toxin on Shaman, Clue, Vault, **sweep**, sac Wellspring, cast Wellspring | **+0.44** | 1.000 | 1.2 s | 840 |
| 64 | mean | same line | -0.04 | 0.950 | 1.3 s | 860 |
| 128 | max | same line | +0.44 | 1.000 | 1.5 s | 1074 |

Once the sweep has resolved the critic does see the wiped board (+0.44 at the
opponent's upkeep vs +0.10 for the Clue line), which is why the search works
with the current critic at all: the blind spot is in the policy's last step and
in the critic's view of the intermediate states, not in the leaf.

## What the first training attempt taught

Resumed from the 500k checkpoint with budget 64, 10% of eligible decisions,
`c_scale` 1.0 and `distill_coef` 1.0, the policy collapsed within three
updates: win rate against its own pool 0.56 -> 0.31, entropy 0.31 -> 0.64,
games three turns shorter. Two causes, both fixed:

- `approx_kl` included the searched rows, whose behaviour probability can be
  1e-7, so it read 8.6e7 and `target_kl` stopped every epoch after one
  minibatch. The KL and clip statistics now cover policy-gradient rows only.
- With `c_scale` 1.0 a 0.05 difference in leaf value, the critic's noise level,
  is 3+ logits, so the distillation targets were one-hot on whichever leaf the
  critic overrated, and 6,000 such targets per iteration dragged the shared
  trunk. The target now has its own scale (`target_scale` 0.25,
  `--search-scale`): noise stays under a logit, the Shaman sweep's +0.5 is
  still 6+. The tree keeps the full scale, because with 0.25 inside the tree
  the visits follow the prior and never resolve the sweep. `distill_coef` is
  0.3 by default now. Sacrifice costs are no longer branching nodes: the
  policy picks the Wellspring at 0.998 anyway, and the sweep's child is then
  the resolved board instead of a sacrifice decision the critic cannot judge.
  With that, budget 32 finds the line too.

Search at play time alone does not move the aggregate: 80 benchmark games on
the Mac, 61.3% with search (budget 64, before these fixes) vs 62.5% without,
intervals overlapping. The combo position is rare and the critic scoring the
leaves is the one that never saw a resolved sweep; the training run tests
whether distilling the search's choices changes that.

## Second attempt: intervene only by a margin

With the soft target the policy no longer collapsed, but the benchmark still
fell from about 0.50 to 0.29 within five iterations (0.27 at ten) while the
win rate against the pool read a healthy 0.6: the learner searches in pool
games and the pool does not, so that number hides the plain policy's decline.
Searching every eligible Jund decision of the seed-6 game (56 of them) showed
why: the search disagreed with the policy at about half of them, mostly by
value margins inside +-0.1, which is the critic's noise plus the optimism of a
max backup, and the Gumbel-sampled root pick was sometimes worth less than the
policy's own line. The real finds stood out: the Shaman lines at +0.17, +0.40
and +1.19.

So the search now intervenes only when its action differs from the policy's
argmax and beats that action's own searched value by `margin` (0.15,
`--search-margin`); otherwise the rollout plays the policy's sampled action
and records no target. On the seed-6 game that keeps 11 of the 56 decisions,
all three Shaman lines among them. The in-tree opponent also gets its real
recurrent state and pending events when the learner plays that seat too
(self-play games): a GRU started from zero in the middle of a game is a
different, worse player, and part of the value swings came from it. Metrics:
`searches` (searches run) next to `searched` (interventions distilled).

Attempt 4 (the gate, 17% of searches intervening) still jumped to KL 0.16 and
entropy 0.54 on its first update: the distillation loss was averaged over the
searched rows only, so five one-hot targets in a 2,048-row minibatch pulled as
hard as 5,800 soft ones had. It is now a mean over the whole minibatch
(`distill_coef` back to 1.0), so the pressure scales with the share of
decisions the search changed.

The first version of the update kept the targets as a dense tensor of all
rollout decisions by the widest decision's option count. Attack declarations
can have thousands of options, so it reached 25 GB on the GPU and varied per
iteration, which looked like a leak until it crashed the run at iteration 316.
Targets are flat now and expanded per minibatch to that minibatch's width.

## Attempt 5 verdict, and the search value as critic target

Attempt 5 (margin gate, soft target, minibatch-normalized distillation) ran 75
iterations without damage but without gain: benchmark 0.39 to 0.43 against the
baseline's final 0.52, entropy 0.31 to 0.41. The plain policy moved at the
probed positions (seed 6, frame 168: pass 0.68 to 0.34, Shaman 0.10 to 0.23;
frame 321: the sweep from 1e-7 to 0.006) but the mass spread over several
options rather than concentrating on the line: soft targets from noisy leaf
values broaden the policy more than they teach the combo.

The bottleneck is the critic's view of intermediate states (Shaman on board,
Toxin in hand is worth nothing to it until the sweep has resolved), which
forces the deep horizon and makes every margin noisy. Attempt 6 therefore
trains the value head at searched decisions towards a mix of the game return
and the search's value of the action taken, its line to the horizon
(`--ppo-search-value-coef`, default 0.5; `Result.search_values`). That value
exists for every searched decision, intervention or not, so about 6,000
decisions per iteration get it. Metric `search_valued`.

## Cost and open items

- Every node and every auto-played decision is one forward pass: about 13 evals
  per expansion here, 1.4 ms each on the Mac with the Python engine. On the H100
  box with the native engine expect ~0.5 s per searched decision per core, and
  15-30 eligible decisions per Jund game, so a searched game costs 20-30x a plain
  one. `--search-frac` bounds that: 0.1-0.2 is a 3-5x rollout slowdown.
- Cheaper evaluations: batch the leaves of one simulation, auto-play the
  opponent's passes without the network when they have no untapped mana, move
  the auto-play into the native engine, inference-server support.
- The critic's intermediate blindness means the search's root value for the
  combo depends on reaching the post-resolution leaf; `max_depth` 8 is the
  minimum for a five-step line with a sacrifice decision. Scenario-seeded
  training positions would speed up the critic.
- Hidden information: one determinization per search; the opponent's responses
  come from the same network at a zero recurrent state.
- Not yet covered: the opponent's turn (instant-speed decisions, blocks). The
  same machinery applies once leaf evaluation is cheap enough.

Reproduce the table: `python -m mtg_ml.replay record --agents model:runs/x/model.pt,bot --seed 6`
on the replay-viewer branch, then search frame 168 with `ModelAgent(..., search=SearchConfig(budget=64))`.
