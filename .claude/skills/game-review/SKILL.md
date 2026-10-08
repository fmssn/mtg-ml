---
name: game-review
description: Review a trained checkpoint's games with Opus agents to find setup flaws (engine bugs, masking gaps, missing features, architecture limits) before studying play quality. Use when the user asks to review games, audit a new checkpoint or feature set, or "find what the model can't see". Records greedy self-play, has one Opus agent review each game, clusters findings across games and verifies each against the code.
---

# Game review of a checkpoint

Built on `mtg_ml.review` (docs/game-review.md). One run: about 50 games, 67 agents, about 9M subagent tokens and 35-40 minutes. The first run (2026-10-07, `r4-control` v07714) is in `/Users/fabsi/repos/mtg-ml-reviews/2026-10-07-r4-control-v07714/`; compare against it.

## 1. Pick the checkpoint and its code

- Default: the newest policy snapshot of the run the user names. If they name none, use the newest jund_blue run on h100-private.
  - `ssh h100-private 'ls -td ~/mtg-ml-*/runs/*/'` lists the runs.
  - `ls -t <run>/policy | head -1` gives the snapshot.
  - Archived runs are in `~/mtg-ml-checkpoints/<id>/`.
  - Copy the `policy/vNNNNN.pt` file (about 68 MB), not `latest.pt` (about 200 MB, with Adam state).
- Read its config: `torch.load(p, map_location="cpu", weights_only=False)["config"]` gives `features`, `trunk`, `hidden`, `memory`, `entity_attn`. Policy files hold no iteration or game count. Take those from the run's `metrics.jsonl` or the snapshot number, and say them in `--note`.
- The code must know that feature set. Check that `FEATURE_VERSIONS` in `mtg_ml/encode.py` includes it. If not, check out the branch the run trained from into a scratch worktree (`git worktree add --detach <scratch>/code origin/<branch>`). If that branch lacks `mtg_ml/review`, copy it in. Run every command below from that code directory.
- Play locally if the box is loaded: `ssh h100-private uptime`, load above about 60 means busy. Recording 50 games takes about 15 s on the Python engine.

## 2. Record greedy games

```bash
M=model:<path>/vNNNNN.pt
python -m mtg_ml.review record --agents $M,$M --games 50 --seed 1000 --greedy --engine python \
  --matchup jund_blue --out <dir> \
  --note "<run> policy vNNNNN (iteration N, about X M training games; feature set F; trunk T, hidden H, memory M, entity_attn A)"
```

- `--greedy`: argmax play takes sampling noise out, so the remaining errors are the policy's, the features' or the engine's.
- Keep seeds 1000-1049 for comparable reruns.
- Python engine, because it is the reference and `verify` rebuilds games with it.
- `<dir>`: `/Users/fabsi/repos/mtg-ml-reviews/<date>-<run>-v<NNNNN>/games`. Never inside the repo: about 50 MB.

## 3. Write the prompts

```bash
python -m mtg_ml.review review <dir>/*.json --backend prompt --focus setup
```

`--focus setup` adds the "what the policy observes" sheet and the setup-first instruction. New recordings carry structured checkpoint configuration; the sheet selects the actual feature set (1–6), trunk, memory and attention layers for each model seat. Check it against the checkpoint before reviewing. Legacy recordings without configuration get a versioned reference marked unknown: resolve the actual configuration before confirming a gap. Keep `observation_sheet` in `mtg_ml/review/prompt.py` aligned with new feature versions. A wrong sheet makes reviewers report gaps already fixed, or miss new ones.

## 4. Run the workflow

Call `Workflow` with the saved workflow `game-review` (`.claude/workflows/game-review.js`) and args:

```json
{"dir": "<dir>", "code": "<code dir>", "games": ["<file stem without .json>", "..."],
 "previous": "/Users/fabsi/repos/mtg-ml-reviews/<earlier review>/flaws.json"}
```

List the stems with `ls <dir>/*.json | grep -v findings | xargs -n1 basename | sed 's/.json$//'`. `previous` is optional; with it, each flaw is marked new or recurring, and fixed flaws are listed in `gone`. Opus is the reviewer: it caught 22/27 seeded faults with no false engine claims. Sonnet 5.5 caught 16, deepseek-reasoner 14, and deepseek-chat 5 with false engine bugs (PR fmssn/mtg-ml#31).

## 5. Save and report

- Save the workflow result as `<review dir>/flaws.json`. Copy the policy file and the workflow script next to it.
- Report to the user, grouped by verdict (confirmed, partly, refuted), each with the number of games, one or two example decisions, the root cause and the fix. Lead with engine bugs and confirmed feature or architecture gaps. Put training-side patterns last, and say they are not feature flaws.
- If the user wants the fixes done, write a handoff note in the shape of `docs/handoff-2026-10-07-feature-set-4.md`: evidence paths, base branch, ground rules (version gate, both engines, difftest, docs/features.md), gaps by priority with fixture decisions, out-of-scope items, and done-when.
- Run `python -m mtg_ml.audit --features N` on the checkpoint's feature set (see `docs/audit.md`); report its per-decision-kind table next to the flaws.
- Every flaw comes with fixture decisions. Rebuild any of them with `mtg_ml.review.verify.prefixes(rep, [d])`, then call `encode.entity_features` / `rl.features.featurize` to show exactly what the policy saw.

## Gotchas

- A reviewer that sees merged options ("Spawn#211 blocks Serpent#195" covering two identical Spawns) may call it a masking gap. The engine merges only fully equivalent objects (`equiv_key`); the verifiers refute these.
- Single-option decisions show only as log lines. When a fault leaves one option, that line is the whole action mask.
- "Option B was better" claims are opinions until `verify.compare` (paired rollouts) confirms them. In the first calibration run, one of four claims was refuted.
- Option faults (`calibrate`) need the Python engine. Run `python -m mtg_ml.review calibrate` when you change the prompt or the reviewer model, so recall stays known.
