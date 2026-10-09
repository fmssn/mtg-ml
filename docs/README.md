# Documentation index

Start with [development setup](development.md), the [shared context index](context.md)
and the project rules in [CLAUDE.md](../CLAUDE.md). Current work ownership and PR
observations live in the external tracker linked from the context index; this
directory is reference material and evidence, not a live task board.

## Current reference

| Area | Guides |
|---|---|
| Development and coordination | [Setup](development.md), [agents](agents.md), [workspace ownership and handoffs](workspace-coordination.md), [PR autopilot](pr-autopilot.md) |
| Rules and decks | [Adding cards](adding-cards.md), [native engine](native-engine.md), [sideboarding](sideboarding.md), [deck variants](deck-variants.md) |
| Model inputs and training | [Feature versions](features.md), [belief head and BO3 knowledge](belief-head.md), [training options](training-options.md), [inference server](inference-server.md), [evaluation](evaluation.md) |
| Review and expert evidence | [Game review](game-review.md), [expert data](expert-data.md), [log reconstruction](expert-reconstruction.md) |
| Playing and hosting | [Play UI](play-ui.md), [hosting](hosting.md), [playtest feedback](play-feedback.md) |
| Experiments | [Recording requirements](experiments/README.md), [ledger](experiments/ledger.md), [reference ladder](experiments/ladder.md) |

Feature set 7 is the default for new runs. Feature set 8 is opt-in; existing
checkpoints retain their recorded feature version. Use the feature and belief
guides for the compatibility contracts.

## Current roadmaps

- [Benchmark plan](benchmark-plan.md): implementation stages and release gates;
  consult its delivery table before choosing benchmark work.
- [Representation plan](representation-plan.md): research direction and rationale.
  Its dated implementation proposals must be read alongside the current feature
  contracts, experiment ledger and benchmark plan.
- [Width-256 attention training](training-256-attention.md): optimization design
  and validation requirements; measured results and unfinished experiments belong
  in the ledger and shared tracker.

Roadmaps do not authorize training jobs, replace write claims, or establish that
a PR is ready. Verify live ownership and GitHub state before implementation.

## Historical evidence

Preserve these documents as records of the code, measurements and hypotheses at
their stated dates; do not treat their next-step lists as current assignments.

- [October 6 handoff](handoff-2026-10-06.md), [October 7 feature-set-4 handoff](handoff-2026-10-07-feature-set-4.md).
- [October 7 next actions](next-actions.md) and [plateau analysis](plateau-analysis.md).
- [First training-speed plan](training-speed-plan.md) and [second training-speed plan](training-speed-plan-2.md).
- [Agent validation](agent-validation.md), [autopilot capacity validation](autopilot-capacity-validation.md), and [generalized Pauper review](generalized-pauper-review.md).
- [October 9 playtest triage](playtest-reviews/2026-10-09.md), [benchmark evidence](benchmark-jund-results.md), and the dated entries in the [experiment ledger](experiments/ledger.md).

Historical source references may describe code that has since changed or remained
on a rejected branch. Keep their evidence intact and use current reference guides
for commands and behavior.
