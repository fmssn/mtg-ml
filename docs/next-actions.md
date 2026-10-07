# Next actions after the overnight A/B run (2026-10-07)

Source: the overnight A/B (entity h128, self-play + pool vs. + 10% bot games, ~8.3M games each, h100-private), a review of 6 sampled and 5 greedy model games, and four audits of the training setup (reward, exploration, opponents, representation). Items marked **confirmed** were checked in the code; the rest are hypotheses with a cheapest test each.

## Where we stand

- Both runs plateaued at ~70% game-1 vs the blue bot from ~5M games on (last-six mean: self-play 70.8%, +bot 70.3%). The 500k run was at 52%.
- Nearly all measurable change happens in the first ~1M games: entropy 0.98 → 0.35 by 0.6M then 0.30 at 8M; explained variance ~0.85 from 0.1M on; win rate against past snapshots ~0.53.
- Greedy play is visibly better in places (KCS + Toxin sweep found at p=1.00, good removal baiting), but makes confident mistakes: missed lethal and alpha strikes, Krark-Clan Shaman never used as removal (p=0.00 for 10 turns against 1/1 Delvers), idle mana / held threats / skipped land drops, Toxin Analysis spent as a cantrip, Blue blocking into deathtouch.

## 1. Correctness fixes (small, do first; one PR, both engines, golden digests will change)

1. **Attacker declaration trap — confirmed.** `game.py:1372` sets `gi_min = gi` after each pick, so groups earlier in the canonical order are never offered again, and pending attackers are not in the state. Picking the big creature first silently drops the Familiars (missed lethal in s301 f362, missed swings in s301/s302/s303). About 26% of multi-group declarations end with eligible attackers left out.
   - Change: one yes/no decision per eligible creature in a fixed order (or offer all remaining groups), and mark pending attackers (`e:attacking`) during the loop. Python engine + `native/src`, test in an `ENGINE_MODULES` module, `make difftest`.
   - Done when: a test position with a later-group creature picked first can still add earlier ones; replaying the overnight checkpoint over 500 benchmark games shows 0 forced Done with eligible attackers left.
2. **Unbounded value head — confirmed.** `model.py:662` is `nn.Linear(hidden, 1)`; values reach +1.41 / −1.12 in reviewed games, which biases the terminal advantages (winning looks slightly worse, losing slightly better).
   - Change: clamp `traj.values` to [−1, 1] before the deltas in `rollout.py` `_finish` (keep raw values for logging); optionally bound the head.
3. **Sleep of the Dead untap skips stack — confirmed.** `cards.py:101` and `native/src/engine.rs:1004` use `skip_untap +=`; per the rules all such effects point at the same next untap step. Decided s103; the policy exploits it.
   - Done on main (PR fmssn/mtg-ml#13, non-stacking skip_untap).
4. **Infra fixes from the run** (already in draft PR fmssn/mtg-ml#12): `--server-request-ints`, `--server-policy-slots`, fresh graph pool on stack growth, PPO graph-shape margin 1.25. Request sizing is now automatic (`--server-request-ints 0`, from games per request).

## 2. Measurement (needed before any A/B is meaningful)

5. **Benchmark that can see 5-point effects.** 100 sampled games give ±9 pp; all late movement between 63% and 77% is noise, and 3 of 4 eval blocks are saturated (random, bot/blue, pool0 = random init).
   - Change: `--greedy` in `evaluate`; benchmark 1000 games (greedy and sampled) every ~250k games; drop or shrink the saturated blocks; replace `pool0` with a ladder of fixed reference checkpoints (e.g. 1M, 3M, 5M, 8M of the overnight run) at 200 paired games each, Elo fit.
   - First run: re-evaluate both overnight final checkpoints at 1000 games, greedy and sampled (expect within ±3 pp of each other, greedy a few points higher).
6. **Logging.** Per-kind entropy and KL over non-trivial decisions (p_max < 0.99), epoch-1 vs epoch-4 KL, win rate per pool opponent, bot games excluded from `win_vs_pool` (they are counted today, `train.py:422`), decisions per game and pay_mana share.

## 3. Fine-tune experiments from the overnight checkpoint (~1M games each, judged by item 5)

7. **Learning-rate anneal.** lr is constant 3e-4 (`ppo.py:42`); KL 0.015/iteration and 10% clipping for 4000 iterations, target_kl (0.03, mean over all decisions) fired once in 8100 iterations. Anneal 3e-4 → 3e-5 over ~1M games; try epochs 2 / minibatch 4096. Expect KL to drop to ~0.004 and +3-6 pp if the plateau is noise-limited.
8. **Discount per game turn, not per decision.** GAE runs over every decision (`rollout.py:203`, gamma 0.995, lam 0.95): ~6 decisions per turn, mostly passes and mana payments, so the credit horizon is ~18 decisions ≈ 3 turns and the cost of acting depends on how many clicks a line takes. Record the turn per step and use gamma_turn ≈ 0.97 per game turn. Offline check first: recompute advantages on one recorded iteration and compare "play land" vs "pass" advantages under both schemes.
9. **Separate, larger critic.** Explained variance flat at 0.85 from 0.1M to 8M games; the shared Linear head on the h128 policy trunk doesn't price board swings (losing the whole board: −0.67 → −0.75). Try `--value-net separate --value-hidden 256`, higher lambda or Monte Carlo value targets. Offline check first: fit shared head / separate h128 / separate h256 on MC returns from 500k decisions.
10. **Lethal and readiness features.** `encode.py:100-110` sums all creature power including tapped and summoning-sick. Add `self:ready_power`, `self:ready_evasive_power`, `opp:potential_blockers`, `lethal_on_board` / `evasive_lethal_on_board` (and the opponent's), untapped mana, in both engines. Probe first: logistic probe from the core vector to `evasive_lethal_on_board` on 50k states.
11. **Missing known state.** Emit `skip_untap`, stack-item targets and X, and positions of known library cards (all available in `view.py`, dropped in `encode.py` / `features.rs`). s305: Sleep target was a 0.50/0.50 coin flip between a locked and an unlocked Familiar.
12. **Effect previews on options.** KCS activation points only at the Shaman; cast options point at nothing; hand cards are not entities. Add engine-computed tokens: `kills_opp:k`, `kills_self:k`, `damage_lethal_to_target`, `target_ward:N` / `ward_payable`, and for casts `mana_left_after:k` / `colors_left`. Probe first: P(activate KCS) on 200 positions where it kills X/1s vs 200 where it kills nothing (expect no difference today).
13. **Less game-1 dilution.** `postboard_frac` 0.5 removes Toxin Analysis, Sleep of the Dead and Force Spike from half the games while the benchmark is game 1. Try 0.2.

## 4. Larger projects

14. **League instead of near-copies.** The learner beats a 122-iteration-old snapshot only ~51%; shared blind spots are never punished. Prioritised fictitious self-play over the pool (weight by how often the opponent beats the learner), plus per-deck exploiters trained against the frozen main policy. Cheapest test: a Jund-only and a Blue-only exploiter from the final checkpoint, 150 iterations each; ≥60% against the frozen main policy confirms exploitable gaps.
15. **Bot games done properly.** The A/B was underdosed: only ~3% of training data was the benchmark matchup, because half the bot games put the learner on Blue against a Jund bot it beats ~100%. Fix the learner seat to Jund vs the blue bot, 0.25-0.3 of games, taken from the uniform-pool share; or make the bot one league member under item 14.
16. **Search as the explorer.** Multi-step lines (KCS on Delvers before they flip, holding Toxin, the Map explore, the right alpha strike) sit at p≈0 and won't be sampled. Use the own-turn Gumbel search (`mtg_ml/rl/search.py`, budget 32-64) on 5-10% of learner turns and distil it. Offline check first: run the search on the reviewed positions (s304 f147/f305, s302 f256, s301 f362, s303 f92) and see whether it finds the better move.
17. **Action decomposition and capacity.** Auto-pay mana with a colour-preserving payer (keep only strategic payments like sacrificing a Spawn), collapse uneventful passes; entity self-attention (1-2 layers, 4 heads at h128) and pointer slots in event tokens; a capacity arm at h192/h256 to separate capacity from architecture.

## Suggested order

1 → 2 → 3 (items 1-4, one PR) · 5-6 (measurement) · 7, 8, 9 as parallel fine-tunes, then 10-12 · 14-16 · 17.

Evidence per item (file:line, metric numbers, frames): the full review and audit outputs of 2026-10-07 are kept outside the repo (Fabsi's notes, MTG-ML project). Replays: `~/mtg-ml-overnight/replays-{morning,greedy}` on h100-private (record more with `python -m mtg_ml.replay record --greedy`). Overnight run metrics: `~/mtg-ml-overnight/runs/overnight-{selfplay,bot10}/metrics.jsonl`.
