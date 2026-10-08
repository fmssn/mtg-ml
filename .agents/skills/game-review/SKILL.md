---
name: game-review
description: Review a trained mtg-ml checkpoint's games for engine, masking, observation and architecture flaws; record greedy games, verify findings and compare with earlier reviews. Use for checkpoint or game reviews, not general code review.
---

# Game review of a checkpoint

Read `docs/game-review.md` for the recording, prompt, reply-import, verification
and calibration commands. Use `.venv/bin/python` in the code checkout compatible
with the checkpoint. `docs/context.md` points to current run ownership, external
review evidence and the append-only checkpoint archive.

1. Select the named checkpoint and code ref. Inspect its `config` (especially
   `features`, `trunk`, `memory`, `entity_attn`) and keep the recorded feature
   version. Record greedy Python-reference games, normally 50 seeds 1000–1049,
   outside Git; policy snapshots are smaller than resume checkpoints.
2. Generate `--backend prompt --focus setup`. Check that each prompt's observation
   sheet describes the recorded policy. Legacy replay files without structured
   configs provide a versioned reference; resolve the actual config before
   confirming an observation or architecture gap.
3. Review each prompt independently and save the requested JSON as
   `<game>.reply.txt`. Use Codex subagents when delegation is available and
   authorized, in bounded batches that fit the available concurrency. Otherwise
   review sequentially. Reviewers read the whole prompt without inspecting code
   or other reviewers' answers. Codex does not execute the Claude Workflow JS.
4. Import replies with `--backend file`, optionally `--model codex` for separately
   named reviewer artifacts (`.reply-codex.txt`). Run counterfactual verification
   for findings with a better option, then cluster underlying setup causes.
   Verify each cluster against code and cited replay prefixes, trying to refute
   it. Equivalent objects intentionally share options; that alone is not a mask
   bug. Check `docs/features.md` before asserting that a feature is absent.
5. Save `<review root>/flaws.json` using the existing workflow shape: `reviewed`,
   `flaws` (id, cause, title, description, games, examples, check), and `gone`.
   Each `check` has verdict (`confirmed`, `partly`, `refuted`), evidence,
   root_cause, fix and files. Compare with the named earlier review, mark new or
   recurring causes, and identify earlier confirmed flaws no longer observed.
6. Run `python -m mtg_ml.audit` on the checkpoint's feature set for the mechanical
   blind-spot measurement. Report confirmed engine/feature/architecture gaps
   first, training patterns last, with game counts and reproducible decisions.

Claude's Opus workflow remains at `.claude/workflows/game-review.js`; its measured
recall does not transfer to another model. Calibrate a changed reviewer/prompt
with seeded faults before treating recall as established. Prompt/file mode uses
the coding agent directly and needs no additional API key. Do not launch training
or mark related PRs ready merely because a review found possible fixes.
