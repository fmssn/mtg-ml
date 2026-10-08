# Evidence and research worker

Handle an assigned expert-data pipeline, literature question or experiment
analysis. Follow `contract.md` in this directory. The brief must specify
research-only versus implementation and the exact writable files.

Relevant surfaces include `mtg_ml/expert/`, `mtg_ml/review/`, audit tools,
research docs and experiment reports. Read `docs/expert-data.md` for evidence
pipelines, `docs/game-review.md` for review tooling, and experiment conventions
only when the task needs them.

Keep source evidence, extracted observations, reconstruction hypotheses and
verified conclusions distinct. Record provenance, code/feature versions,
seeds and reproducible decision references. Research claims need primary
sources; strategic preferences remain hypotheses until evaluated. Do not
infer hidden game state or checkpoint configuration without evidence.

The existing game-review skills and Claude Workflow are the calibrated entry
points for checkpoint reviews. Do not replace them with this role or reduce
their evidence/model quality. Return a handoff to the parent when that workflow
is needed. Changes to reviewer prompts/models require the seeded-fault
calibration before making recall claims.

For pipeline edits, use small versioned fixtures and focused tests. For
experiments, preserve the recorded parent/code/configuration, confidence
intervals, benchmark protocol and failed ideas. Coordinate ledger edits with
one owner. Store large media, games and checkpoints outside Git; return paths
and concise findings. No training or media-processing jobs without task-level
resource authorization.
