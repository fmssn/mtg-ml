# Why PPO training plateaus at ~29% vs the blue bot (2026-10-06)

Analysis of the `bench500k-h128` run (hidden 128 MLP, shared value head, 245 iterations, 501,760 games) plus the two running comparison runs (`bench500k-tf`, transformer, 110k games; `bench500k-h512mlp`, 240k games) on `h100-private`. All numbers below come from the runs' `metrics.jsonl` and from small CPU-only evaluations (400 games each, scratch dir `~/plateau-analysis` on the box, scripts `diag.py` and `mull.py`). Nothing in the repo or the run directories was changed.

## Summary

The benchmark is genuinely flat, not noisy-rising: from iteration 150 (307k games) to 245 the game-1 win rate vs the blue bot averages 0.282 with a standard deviation of 0.016 across 20 evaluations, which equals the binomial noise floor of 1000-game evaluations (0.014); the fitted slope is 0.0007 per 10k games against 0.0093 before iteration 150. The policy has stalled in every sense, not only on the benchmark: the final checkpoint beats its own iteration-150 and iteration-200 snapshots 52% of the time (CI includes 50%) and its iteration-100 snapshot 56.5%. The most concrete, confirmed cause is the state representation. `encode.state_features` is hashed into a set, so the network cannot see counts: how many lands it has, how many copies of a card are in hand or on the battlefield, how many cards are in a graveyard, or what turn it is. A direct symptom is the mulligan policy: it keeps 7-card hands based on the number of distinct land names rather than the number of lands (2 lands with the same name: keep 29%; 2 lands with different names: keep 70%), mulligans twice as often as the Jund bot and loses 90% of games that start on 5 cards. Secondary causes are a learning setup that has converged (constant LR, PPO updates of ~600 minibatches per iteration that no longer change behaviour, half of the recorded decisions being pass-priority or mana-payment choices), and the fact that the learner never trains against the opponent it is benchmarked on. Shaping annealing is not the cause. The 512-wide MLP and the transformer are currently behind the 128 run at equal games, most likely because they early-stop every PPO update at the default learning rate, so capacity has not been fairly tested yet.

## Evidence base

### The plateau in numbers (h128)

| iterations | games | bench mean | bench sd (20 evals) | slope per 10k games | bo3 mean | eval/bot/jund (50 games) | eval/bot/blue |
|---|---|---|---|---|---|---|---|
| 5-100 | 10k-205k | 0.124 | 0.063 | +0.0105 | 0.088 | 0.092 | 0.540 |
| 100-150 | 205k-307k | 0.236 | 0.033 | +0.0093 | 0.209 | 0.185 | 0.680 |
| 150-245 | 307k-502k | 0.282 | 0.016 | +0.0007 | 0.293 | 0.261 | 0.700 |

Binomial sd of a 1000-game evaluation at p=0.28 is 0.014. Everything between 0.25 and 0.32 after iteration 150 is noise around 0.28-0.29.

Optimization metrics, means per 50 iterations:

| iterations | jund_wins_selfplay | win_vs_pool | turns | decisions per trajectory | entropy | approx_kl | clip_frac | explained_var | v_loss |
|---|---|---|---|---|---|---|---|---|---|
| 1-50 | 0.224 | 0.649 | 20.5 | 83 | 0.576 | 0.0087 | 0.08 | 0.818 | 0.027 |
| 51-100 | 0.296 | 0.581 | 19.7 | 96 | 0.374 | 0.0127 | 0.10 | 0.836 | 0.027 |
| 101-150 | 0.328 | 0.567 | 20.0 | 102 | 0.351 | 0.0133 | 0.10 | 0.832 | 0.028 |
| 151-200 | 0.350 | 0.559 | 20.2 | 105 | 0.341 | 0.0133 | 0.10 | 0.824 | 0.027 |
| 201-245 | 0.372 | 0.557 | 20.7 | 108 | 0.344 | 0.0140 | 0.11 | 0.820 | 0.028 |

No draws in any iteration (max_turns 100 is never hit). `target_kl` 0.03 never triggered in h128 (0 early stops in 245 iterations). The h512 run hits it in 10 of every 10 iterations from iteration 15 on (approx_kl 0.026, clip_frac 0.17-0.21, 260-380 of ~600 minibatch updates used).

### Head-to-head of h128 checkpoints (400 paired games each, both seats)

| match | learner as Jund | learner as Blue | overall |
|---|---|---|---|
| iter 245 vs iter 100 | 0.350 | 0.780 | 0.565 (0.52-0.61) |
| iter 245 vs iter 150 | 0.335 | 0.710 | 0.522 (0.47-0.57) |
| iter 245 vs iter 200 | 0.365 | 0.685 | 0.525 (0.48-0.57) |
| iter 150 vs iter 100 | 0.330 | 0.735 | 0.532 (0.48-0.58) |

Two things are visible: the learned Blue beats the learned Jund about 2:1 regardless of version (a strong deck asymmetry in self-play, so `win_vs_pool` of 0.55 is roughly 50% after adjusting for seat), and 200k further games produced a policy that is indistinguishable from the one at 300k games.

### Loss anatomy vs the blue bot (iter 245, 400 benchmark games)

| | learner (sampling) | learner (argmax) | Jund bot |
|---|---|---|---|
| Jund win rate | 0.310 | 0.347 | 0.253 |
| won games: turns / Jund life / Blue life / attacks | 25.2 / 13.9 / -2.3 / 7.6 | 24.9 / 14.3 / -2.9 / 8.1 | 20.4 / 8.5 / -3.0 / 9.8 |
| lost games: turns / Jund life / Blue life / attacks | 18.5 / -4.5 / 17.6 / 1.2 | 18.3 / -4.4 / 17.7 / 1.0 | 14.9 / -4.6 / 16.3 / 1.8 |
| end reasons | 274 Blue life, 119 Jund life, 7 decking | 257 / 132 / 11 | 299 / 101 / 0 |

Paired on the same seeds and starting players: both win 49, learner only 75, bot only 52, neither 224. In 56% of games neither the learner nor the Jund bot wins, and lost games end with Blue at 17-18 life after about one Jund attack in the whole game: Jund never gets a board through Delver plus counterspells, and its wins are 25-turn grinds. The learner casts more spells per game than the bot (22.9 vs 20.0 in wins, 13.3 vs 9.5 in losses) and plays more lands, so it is not simply failing to play its cards.

Per decision kind (learner, sampling):

| kind | n | options | entropy | p(max) |
|---|---|---|---|---|
| priority | 43,933 | 2.86 | 0.221 | 0.909 |
| pay_mana | 12,193 | 3.88 | 0.846 | 0.640 |
| target | 2,256 | 3.91 | 0.334 | 0.857 |
| declare_attacker | 2,006 | 2.60 | 0.320 | 0.859 |
| sacrifice | 1,767 | 4.22 | 0.471 | 0.791 |
| declare_blocker | 1,567 | 2.43 | 0.264 | 0.879 |
| choose_card | 1,210 | 3.96 | 0.385 | 0.844 |
| mulligan | 666 | 2.00 | 0.108 | 0.955 |

70% of priority decisions are "Pass priority" (30,899 of 43,933). pay_mana is 18% of all learner decisions and carries the highest entropy by far.

### Mulligans (400 games each)

| policy | games with 0 / 1 / 2 / 3+ mulligans | win rate after 0 / 1 / 2 mulligans |
|---|---|---|
| Jund bot | 286 / 73 / 41 / 0 | 0.276 / 0.247 / 0.098 |
| h128 iter 50 | 270 / 100 / 27 / 3 | 0.185 / 0.110 / 0.074 |
| h128 iter 245 | 211 / 114 / 61 / 14 | 0.355 / 0.377 / 0.098 |
| tf iter 54 | 251 / 109 / 36 / 4 | 0.163 / 0.101 / 0.028 |

P(keep) of the h128 iter 245 policy on 7-card hands, by (number of lands, number of distinct land names):

| lands | 1 name | 2 names | 3 names | 4 names | 5 names |
|---|---|---|---|---|---|
| 1 | 0.13 | | | | |
| 2 | **0.29** (n=34) | **0.70** (n=210) | | | |
| 3 | 0.97 (n=1) | 0.77 (n=35) | 0.93 (n=117) | | |
| 4 | | **0.28** (n=8) | **0.90** (n=45) | **0.99** (n=20) | |
| 5 | | | 0.66 (n=8) | 0.97 (n=18) | 0.96 (n=6) |

The same pattern holds at iteration 50 (2 lands: 0.34 vs 0.91) and in the transformer run (0.40 vs 0.82), so it is a property of the input, not of one checkpoint. The Jund bot keeps every 2-5 land hand with a cheap spell (keep rate 1.00 on 2- and 3-landers).

## Ranked reasons

### 1. The state features hide counts (confirmed, high confidence)

`mtg_ml/encode.py` `state_features` emits one string per card name per zone (`self:hand:Swamp`, `self:bf:Island`, `opponent:gy:Ponder`, `self:bf_type:Land`), and `encode_state` / `features.featurize` hash them into a set (`sorted({...})`), and `EmbeddingBag(mode="sum")` sums unique indices. Verified locally by dumping the features of a turn-7 position: 33 features, 27 unique; a battlefield of Swamp, Twisted Landscape, Ichor Wellspring and Slagwoods Bridge produces one `self:bf_type:Land` token, and a hand with three Swamps is the same vector as a hand with one. There is no turn-number feature either (only `step:` and `active:`), life is bucketed with edges (0,1,2,3,5,8,13,20), so 13-19 life is one bucket, and the opponent's hand size is bucketed the same way. The native engine produces identical features (differentially tested, md5 of the code on the box matches the worktree).

What the policy therefore cannot represent directly: its land count and mana available (central for a ramp deck with Bridges and Cleansing Wildfire), how many creatures or Spawn tokens it or the opponent has, how many instants and sorceries are in Blue's graveyard (Tolarian Terror and Cryptic Serpent cost), how many copies of a card it has drawn, or how late in the game it is. The GRU can in principle reconstruct some of this from the event stream, but the mulligan table shows it does not even for the simplest case, and the first decision of a game has no event history at all.

Evidence that it matters for the benchmark: the learner mulligans 47% of 7-card hands (bot: 36%), starts 19% of games on 5 or fewer cards (bot: 10%) and wins 9% of those. Correcting only the mulligan rate to the bot's would be worth about +2 to +3 points. The in-game cost (mana counting, when to Wildfire a Bridge, blocking math) is not measured here but the same blindness applies to every decision.

Fix: add count features (`self:lands:{n}`, `self:hand_size:{n}`, per-name count buckets `self:hand:{name}:x{k}`, `self:bf:{name}:x{k}`, `opponent:gy_spells:{bucket}`, `turn:{bucket}`, finer life buckets) in `encode.py` and the Rust `state_features`, keep the hashed set otherwise. Cost: engine change in both engines plus a differential test, then a fresh 300k-game h128 run (about 2 hours on the box at the h128 run's 2.7 s rollout + 8.4 s update per iteration). Expected effect: the mulligan curve flattens to a land-count curve immediately; a few points on the benchmark from mulligans alone, probably more from mana and combat decisions. This is the one change that every other experiment should be run on top of.

### 2. Learning has converged for this setup, not just transfer (confirmed stall, medium confidence on the mechanism)

The head-to-head table shows 200k games (iteration 150 to 245) produced no measurable change in strength against itself. Yet the optimizer still moves: approx_kl 0.014 per minibatch, clip_frac 0.11, 600 Adam steps per iteration at a constant 3e-4, 147k gradient steps in total. Policy entropy fell from 0.98 to 0.34 and has been flat since iteration 150 at exactly the point where the benchmark stopped moving. The value function stopped improving earlier (explained_var 0.84 at iteration 50, 0.82 at 245; v_loss flat at 0.027).

Hypothesis for the mechanism: the policy is at a local optimum of the representation in reason 1, and the remaining per-iteration movement is noise churn on decisions that do not affect outcomes. Supporting detail: 70% of priority decisions are passes and 18% of all decisions are mana payments with entropy 0.85, so most of the PPO objective (and the entropy bonus) is spent on choices that rarely matter, while decisive choices (mulligan, attacks, targets) are 5% of the data. Sampling noise alone costs 3-4 points: argmax play scores 0.347 vs 0.310 for sampling on the same 400 seeds (CIs overlap, suggestive).

Fixes to test, cheapest first: (a) report the benchmark with argmax or low-temperature play as well as sampling, this is free; (b) learning-rate decay or a lower LR after 100 iterations plus a value-loss coefficient sweep; (c) auto-resolve or exclude trivial pay_mana decisions from the trajectory when all options are colour-equivalent (or weight them down), so the advantage signal concentrates on real decisions; (d) gamma 1.0 with lam 0.95: trajectories are 105 decisions long, so gamma 0.995 discounts the terminal reward to 0.59 at the mulligan decision and makes early-game credit depend on the value function. Each is a one-line config change except (c); a 300k-game h128 run per variant costs about 2 hours.

### 3. The learner never trains against the opponent it is benchmarked on (confirmed configuration, medium confidence on impact)

`bot_frac` is 0.0, so all training games are learner vs learner or learner vs pool. Three observations suggest this costs real points: the self-play Jund rate rose from 0.33 to 0.39 between iterations 100 and 245 while the benchmark did not move, so whatever Jund learned recently is specific to the learned Blue; the learned Blue beats the learned Jund 2:1 whereas the scripted Blue beats the learned Jund about 7:3, so the learned Blue is a different and harsher opponent (it plays differently, so Jund's counterplay to it need not transfer); and the learner's loss profile against the bot (Blue untouched at 17 life after one attack) is the same as the scripted Jund bot's, suggesting both fall into the same structural losses against the bot's counter and Delver plan rather than the learner having found exploits.

Against: self-play is what should produce a Jund that is good in general; training only vs the bot risks overfitting to it, and the real-world 55-61% figure is for humans against humans, not against this bot. A quick way to measure the gap between "general play" and "this bot" is to set bot_frac to 0.25-0.5 (the games are taken out of the pool share) and watch whether the benchmark rises past 0.35 within 100 iterations. Cost: config change, 200k games, about 1.5 hours. If the benchmark jumps, the bot is exploitable and the current plateau is partly a transfer gap; if it does not, the ceiling is in reasons 1 and 2 or in the matchup itself.

### 4. Wider networks have not been tested fairly yet (confirmed from metrics, high confidence)

At equal games the h512 MLP is behind h128 (0.133 vs 0.194 at 184k games; 0.177 vs 0.244 at 240k games) and the transformer is level (0.136 vs 0.133 at 102k games). The h512 run early-stops every PPO update from iteration 15 on (approx_kl 0.026 vs target 0.03 averaged over the epoch, clip_frac 0.17-0.21, only 260-380 of ~600 minibatch updates used), so it is both taking larger policy steps and using fewer of its data. The transformer's approx_kl is rising (0.008 to 0.016) and will reach the same regime. This is the standard symptom of a learning rate tuned for the smaller network. The brief's "slightly ahead at equal games" is not supported by the metrics; the mulligan diagnostics show the transformer has the same count blindness (2 lands: keep 0.40 vs 0.82 by name count), as expected since the input is the same.

Fix: run h512 and transformer with lr 1e-4 (or 1.5e-4) and minibatch 4096, or with per-decision-kind KL monitoring; compare at 200k games. Cost: two 200k-game runs, 1.5 hours (MLP) and about 6 hours (transformer, whose update takes 150 s per iteration with the server). Without fixing reason 1 capacity cannot help with counting anyway.

### 5. Matchup structure: 30% may be close to what this Jund list can do against this bot without search (hypothesis, low-medium confidence)

Evidence for a low ceiling: the scripted Jund bot gets 0.253 on the same seeds, an 8-playout determinized-search Jund got 0.27 (README), and the learner, the bot and the search bot all lose the same way. 56% of seeds are lost by both learner and bot. The blue bot never misplays combat, always holds UU when it has a threat, and Force Spikes a tapped-out Jund; a human Blue player does not play this consistently, which is one reason the 55-61% real-world figure may not apply.

Evidence against: the learner already wins 75 seeds the bot loses, and argmax play adds 3-4 more points, so the ceiling is at least in the high 30s. The decisive test is reason 3 (train against the bot) or a search baseline with 64-256 playouts on ~200 games (CPU heavy: 20 ms per playout, so ~1-4 hours on 32 cores); whichever is cheaper to schedule. If neither gets past 0.40, the benchmark itself needs re-examining (card rules audit is done, but the Blue bot may be stronger than realistic, or Jund's list and sideboard plan may be weak against it).

### 6. Opponent pool dynamics (not a cause, confirmed)

The pool has 25 snapshots, 50% of pool games are against the newest. The learner beats pool0 (iteration 0) 98-100% in both seats, beats pool checkpoints 100 iterations old only 56%, and there is no sign of cycling (no checkpoint beats a later one in the head-to-head table). The learner's deck asymmetry in self-play (Jund 37%) is the strongest pool-related signal and belongs to reason 3. Nothing to change here before reasons 1-3.

### 7. Shaping anneal (not a cause, confirmed)

Shaping reached 0 at iteration 100 (205k games). The benchmark rose from 0.210 at iteration 100 to 0.290 at iteration 150 after that, and the dip to 0.180 at iteration 110 is within 2 sd of its neighbours (0.209, 0.198). Self-play turn length and value loss did not change at iteration 100 either.

### 8. Evaluation artefacts (confirmed, small)

Starting player makes no difference for the learner (0.315 on the play vs 0.305 on the draw), but the Jund bot is much better on the play (0.295 vs 0.210). No max_turns draws anywhere. The benchmark uses sampled actions; argmax is 3-4 points better and would be the more faithful measure of the policy. `eval/bot/jund` (50 games) is far too noisy to read trends from; the 1000-game `bench/*` is the one to use. The transformer run's `latest.pt` is rewritten every iteration while the run is live, so any ad-hoc evaluation should copy the file first.

## Recommended next 3 runs

1. **Count features, h128, otherwise identical** (`--hidden 128 --value-net shared`, 300k games, ~2 h). Prerequisite: add count/turn/finer-life features to `encode.py` and the Rust `state_features`, keep the differential test green. Success criterion: mulligan P(keep) becomes a function of land count (2-landers kept ~equally regardless of names), benchmark above 0.33 at 300k games (current run: 0.29). Also log the argmax benchmark alongside the sampled one.

2. **Same features, bot_frac 0.33** (300k games, ~2 h, in parallel with run 1 on another free GPU and the other half of the cores). Compares directly with run 1: the difference is the transfer gap between self-play and this bot. If run 2 is above 0.40 the bot is exploitable and a mixed schedule (bot_frac 0.2 plus self-play) is the way to climb; if both are equal, concentrate on optimization (reason 2).

3. **Same features, h512 separate value net, lr 1e-4, minibatch 4096, gamma 1.0** (300k games, ~3 h). Tests whether capacity helps once the LR no longer trips the KL stop every iteration and once the inputs contain counts. Compare with run 1 at equal games, not at equal wall time.

Cheap additions that need no new run: evaluate existing checkpoints with argmax; add per-decision-kind entropy and approx_kl to `metrics.jsonl` so that the pass/pay_mana churn can be separated from decisive decisions in future runs.

## Scripts

`~/plateau-analysis/diag.py` (per-decision-kind stats, loss anatomy, Jund bot comparison on the same seeds, `--greedy`) and `~/plateau-analysis/mull.py` (mulligan table by land count and distinct land names) on `h100-private`; both run CPU-only in about 10 s per 400 games with the native engine. Raw outputs are in the same directory (`h128_sample.txt`, `h128_greedy.txt`, `h128_i150.txt`, `h2h.txt`, `mull_h128.txt`, `mull_tf.txt`, `tf_sample.txt`, `h512_sample.txt`).
