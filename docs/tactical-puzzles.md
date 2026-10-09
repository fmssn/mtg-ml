# Tactical puzzles: runner and development corpus

This is the agent-driven part of [benchmark plan](benchmark-plan.md) row 4. An
evaluated agent chooses **every** decision from a reviewed position: targets,
payments, order and passes. The pinned specialist plays the other seat. The
attempt is scored by the resulting outcome, not by matching a recorded line.
The development corpus starts at seven puzzles. The remaining corpus (100
puzzles, with dev and final groups), paired intervals and the public
`run`/`compare` commands are later PRs.

## Run it

```bash
make native                                   # required for --engine native
.venv/bin/python tools/puzzle_manifest.py --out .context/benchmark/dev.json
.venv/bin/python -m mtg_ml.benchmark validate --manifest .context/benchmark/dev.json \
  --engines python,native --out .context/benchmark/validation.json
.venv/bin/python -m mtg_ml.benchmark puzzles --manifest .context/benchmark/dev.json \
  --checkpoint runs/candidate.pt --engine native --modes greedy --out .context/benchmark/puzzles.json
```

- `puzzle_manifest.py` pins HEAD, so `mtg_ml/` and `native/` must be committed.
  A manifest cannot pin the commit that contains it, so it is written locally,
  not committed.
- `--agent benchmark-jund@1` (or `benchmark-blue@1`) attempts that deck's puzzles
  with a scripted specialist, which is useful as a baseline.
- Checkpoints with feature sets below 7 see hidden information and need
  `--contract diagnostic`.
- The report (`BenchmarkPuzzleReport`) is descriptive. It contains rows,
  per-puzzle and per-category success, `first_action_correct` and error counts.
  It has no intervals and is not a released `BenchmarkResult`. Its rows have the
  shape `results.build_result(puzzle_rows=...)` expects. Existing outputs are
  never overwritten. Technical errors exit nonzero.

## Classification

| status | reason | meaning |
|---|---|---|
| success | `objective_met` | first stopping boundary (or game end) reached with the objective true |
| failure | `objective_failed` | boundary or game end reached, objective false |
| failure | `horizon_exhausted` | the case's decision cap (≤ 64, both players) ran out first, including repeated passes |
| error | `illegal_action` | the learner returned an index outside the legal options |
| error | `agent_exception` | the learner raised (or a scripted line no longer matched) |
| error | `response_policy_error` | the response bot raised or returned an illegal index |
| error | `engine_error` | compile failure, missing-history rejection or engine fault |

Errors are not attempts: a puzzle with any error row is unscored and the run is
incomplete. A puzzle succeeds only if every one of its cases succeeds.
An objective that is true initially is credited only if it is still true at the
boundary.

## Authoring a puzzle

Puzzles are defined in [`tools/puzzle_corpus.py`](../tools/puzzle_corpus.py), and
the JSON under `benchmark/puzzles/dev/` is generated from it. Run the tool after
editing; `tests/test_benchmark_tactics.py` fails if the artifacts are stale.

1. **Position.** Use setup cards plus legal `setup_actions` (for example the
   opponent casting a spell, or declaring attackers). Never inject state. Keep
   each deck within its decklist counts across all zones. Hidden libraries hold
   blanks, and the case's `hidden_completion` names that assumption.
2. **No missing history.** Reset mode feeds the learner no earlier events, so
   the start must not show revealed hidden cards (`hand_known`,
   `library_known`). Such cases are rejected.
3. **Objective.** A conjunction over public observation paths or named-object
   presence in public zones (`battlefield`, `graveyard`, `exile`). Private
   paths such as `self.hand` are refused. To require that a card was kept, name
   it and require it absent from the graveyard.
4. **Boundary.** Learner priority with an empty stack at a given turn and step,
   or game end.
5. **Witnesses** (`learner_only: true`). These are the learner's selectors only.
   After a line runs out, the learner passes priority, declares no attackers and
   declares no blocks. Witnesses must succeed against the real response policy
   on both engines. Add equivalent lines wherever a different sequence is
   equally good.
6. **Mistakes.** Known bad lines with the failure they must produce.
7. **Review.** An independent reviewer accepts each puzzle against the digest
   printed by `tools/puzzle_corpus.py --keys`. The review is recorded in
   `reviews.json`. Editing a puzzle invalidates its review, and the puzzle is
   written as `pending`, which validation rejects.

Validation also runs a pass learner and four seeded random learners against the
response policy. Every one must finish without a technical error and agree
across engines. Their outcomes are reported, so a puzzle that random play
usually solves is visible as a weak discriminator.

## Development corpus

| id | learner | category | point |
|---|---|---|---|
| blue-sleep-redundant | Blue | resource-preservation | don't re-escape Sleep of the Dead onto an already locked creature (r7 playtest flag 4) |
| blue-sleep-productive | Blue | combat | escape Sleep onto the untapped attacker, not the locked one |
| blue-counter-untapped | Blue | stack-interaction | Force Spike gets paid; Counterspell is needed |
| blue-counter-tapped-out | Blue | stack-interaction | Force Spike suffices; keep Counterspell |
| jund-shaman-toxin-sweep | Jund | deck-synergy | Toxin Analysis on Krark-Clan Shaman, then sacrifice the Clue for a deathtouch sweep |
| jund-survive-attack | Jund | combat | chump block; Cast Down cannot pay Terror's ward |
| jund-color-sequencing | Jund | mana-sequencing | pay Cast Down's generic with the Forest to keep red for the Shaman |

All are synthetic, development split. Their claim is success against the
declared responses, not a forced solution.
