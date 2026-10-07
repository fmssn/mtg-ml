# Handoff: feature set 4, closing the gaps the game review found (2026-10-07)

Read this first. Task: **one PR that adds feature set 4** and fixes the observation and representation gaps listed below. The gaps were found by reviewing 50 games and checking each finding against the code. Engine rules are not part of this task: the review found no engine bugs.

## Where the evidence is

- `/Users/fabsi/repos/mtg-ml-reviews/2026-10-07-r4-control-v07714/` (outside the repo):
  - `games/*.json`: 50 greedy self-play games of `r4-control` policy v07714, about 15.8M training games, feature set 3, entity trunk h128, GRU memory, no entity attention. Recorded on the Python engine with PR fmssn/mtg-ml#28's code. The format is replay files plus `meta.review` (PR fmssn/mtg-ml#31, `mtg_ml.review`).
  - `games/*.prompt.md`: the rendered transcript of each game.
  - `games/*.reply.txt`: the Opus review of each game.
  - `flaws.json`: the clustered flaws, each with a verifier's verdict, evidence, root cause and proposed fix. Read the `check.evidence` of the items below before you design anything.
  - `r4-control-v07714.pt`: the policy, for reproducing the cases.
- Rebuild any cited decision with `mtg_ml.review.verify.prefixes(rep, [d])` (PR #31). It yields the game at decision `dN`. Then call `mtg_ml.encode.entity_features` / `featurize` on it to see what the policy saw.

## Base branch

Feature set 3 (`opp:deck:`) lives in PR fmssn/mtg-ml#28 (`claude/multi-matchup`), which is not merged yet. Build on it: branch from `claude/multi-matchup`, or rebase once it merges. `encode.FEATURES` must become 4, with `FEATURE_VERSIONS = (1, 2, 3, 4)`.

## Ground rules (from CLAUDE.md and docs/features.md)

- Gate everything with `features >= 4`. Sets 1 to 3 must stay byte-identical; existing checkpoints featurize with their recorded version. Add a set-3 digest test like `tests/data/features_v1_digests.json` before changing code, so you can prove set 3 did not move.
- Implement every change in both engines: `mtg_ml/encode.py` / `mtg_ml/rl/features.py` and `native/src/features.rs` (plus `py.rs` bindings for new previews). `tests/test_difftest.py` and `make difftest` must pass, and `test_entity_and_preview_strings_identical` must cover the new tokens.
- No rule changes, so golden digests stay as they are.
- Tests that touch featurization of real games go in an `ENGINE_MODULES` test module.
- Document every new token in `docs/features.md` (version table and per-feature section).

## The gaps, by priority

### P1. Combat relations (verdict: confirmed; seen in 12 of 50 games, often lethal)

What the policy cannot see today:
- A blocker entity gets only `e:blocking`, not which attacker it blocks.
- An attacker gets nothing about whether it is already blocked, how many creatures block it, or their total power.
- `declare_blocker` option keys are `('block', blocker_name, attacker_name)`, names only, and `option_preview` returns nothing for that decision kind.
- Two same-name attackers (two Tolarian Terrors, two Cryptic Serpents) therefore point to bit-identical entities. The policy splits exactly 0.50/0.50 and stacks a second blocker on the already-blocked one while the other deals lethal damage.
- It also cannot add blocker power against toughness: s1010 d359 puts two 3-power blockers on a 6/7, which kills nothing.
- GRU events don't help either: they carry names only.

Add:
1. **Attacker entities, during combat:**
   - `e:blocked` / `e:unblocked` (from `Game.blocked` / `blocks`);
   - `e:blockers>=k`;
   - `e:block_power>=k` (summed power of its blockers);
   - `e:block_lethal` (blockers' power plus damage >= toughness, or a deathtouch blocker).
2. **Blocker entities:** a relation to the blocked attacker. Either `e:blocking:name:<attacker>` plus the attacker's P/T tiers, or, better, an extra entity pointer such as the option pointers in `featurize`. Pick one and say why in the PR.
3. **`option_preview` for `declare_blocker`:**
   - `pv:attacker_already_blocked`;
   - `pv:attacker_dies` (existing blockers plus this one vs remaining toughness, counting deathtouch);
   - `pv:blocker_dies`;
   - `pv:unblocked_damage_left>=k` (attackers still unblocked if this option is taken);
   - `pv:lethal_left` (that damage >= the defender's life).

Fixtures: s1000 d153, s1035 d102, s1047 d229 and d279 (ties of exactly 0.50/0.50, or 0.486/0.486 at d229), s1010 d359 (failed double block). A test should assert that at these states the two candidate options' token sets differ.

### P2. Incoming combat damage (verdict: confirmed; seen in 3 games, and in many chump-block findings)

`_board_features` counts only creatures that could attack (`_ready`: untapped, not sick). During the opponent's combat the attackers are tapped, so `opponent:ready_power` and `opponent:lethal_on_board` drop to about zero exactly when damage is coming. That is misleading, not just missing.

Add, for the attacking side, while there are attackers:
- `{side}:attacking_power>=k`;
- `{side}:unblocked_power>=k` (from the current partial block assignment);
- `{side}:incoming_lethal` (unblocked power >= the defender's life);
- optionally a thermometer of life after unblocked damage.

Fixtures: s1011 d216, s1049 d306, s1000 d153.

### P3. Decision kinds with no preview (verdict: partly confirmed; each a few games)

The pattern: `option_preview` handles only priority cast/activate and target. Every other decision kind is an opaque hashed key.

- **`choose_x`** (Nyxborn Hydra cast with X=0 while mana is spare, 6 games: s1003 d250, s1015 d142, s1031 d152, s1049 d146). Add:
  - `pv:x>=k` thermometer;
  - `pv:x_is_max`;
  - `pv:mana_left_after:{k}`, X-inclusive (reuse `_item_cost` on the stack item);
  - for `etb_x_counters`, `pv:enters_power>=k`;
  - an entity pointer from each X option to the spell on the stack (extend `option_object_ids` in both engines).
- **Colours** (8 games of mana payment, 5 of fetching the wrong basic). Nothing says what colours a source makes or what the hand still needs. Add:
  - state `self:sources:{C}>=k`, `self:hand_needs:{C}` and `self:hand_missing:{C}`;
  - `pv:adds_color:{C}` / `pv:adds_missing_color` on land plays and `('search', <basic>)` options;
  - optionally `e:produces:{C}` on land entities;
  - on `pay_mana` options, `pv:colors_left:{C}` after paying that unit (the same helper as the cast preview).
  - (Alternative for payment only: train with `--auto-mana 1`; not part of this PR.)
- **Delver reveal** (`yes_no`, 4 games: s1006 d9, s1045 d71, s1023 d133, s1042 d93). Add `pv:reveal_type:<type>` and `pv:transforms` on the reveal-yes option.

### P4. Combat projection (verdict: partly confirmed; 3 to 5 games). Optional, take only if P1 to P3 land cleanly.

- `{side}:lethal_through_blocks`: ready attackers against the defender's potential chump blockers, the greedy assignment that `flaws.json` (`no-next-turn-threat-feature`) describes.
- Previews on declare-attacker and creature-cast options for the crack-back: does the opponent have lethal if this creature taps or once this blocker is added?
- `self:lethal_after_stack`, `self:lethal_with_reach`: own pending pumps on the stack, sacrifice outlets plus "whenever you sacrifice" triggers, Munitions-style reach (`pending-pump-lethal-blind`).

## Not in this PR (training-side; noted so they aren't re-diagnosed)

- Sequencing the land drop before spells (loses to Force Spike).
- Passing whole turns with castable spells.
- Flyers not attacking into empty boards.
- Chump-blocking at high life. Shaping was already 0, so this is not reward shaping; P2 should help.
- Mulligans: refuted, it's a deliberate strategy.
- "Cheap spells not cast": refuted, cost reduction works.
- Masking claims about merged same-state options: refuted, `equiv_key` merges only fully equivalent objects.
- Minor engine question to check separately: Cleansing Wildfire's "Find nothing" still shuffles.

## Done when

1. Set 1 to 3 digests are unchanged, set-4 tokens are present and identical on both engines, and `make test` / `make difftest` / `make lint` pass.
2. At the P1 fixtures the two block options have different token sets, and at the P2 fixtures `incoming_lethal` is set.
3. `docs/features.md` documents set 4.
4. Follow-up, after merge: fine-tune `r4-control` or its successor with `--features 4` (new rows start untrained). Then re-run the `game-review` skill on the new checkpoint; the 0.50/0.50 block ties and the X=0 Hydras should be gone. Record the run in `docs/experiments/`.

Consider entity attention (`entity_attn >= 1`) for the next fresh run. On its own it can't fix P1, because the relation is missing from the inputs, but with P1's pointer in place it can use it.
