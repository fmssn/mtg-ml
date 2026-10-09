# Scripted bot and tactical puzzle benchmark plan

**Status: partly implemented.** The foundation (#64,
[benchmark-foundation.md](benchmark-foundation.md)), the Blue specialist (#65,
[benchmark-blue.md](benchmark-blue.md)) and the Jund specialist (#66,
[benchmark-jund.md](benchmark-jund.md)) have merged, and so has the tactical
puzzle runner with seven reviewed development puzzles (#81,
[tactical-puzzles.md](tactical-puzzles.md)). The release tooling (#84,
[benchmark-release.md](benchmark-release.md)) adds `run`, `compare`,
`calibrate`, `turn-limit` and `archive` to `python -m mtg_ml.benchmark`, with
paired statistics and recomputed aggregates. Still to deliver: the remaining
reviewed puzzle corpus (100 puzzles, PR 4), the baseline, turn-limit and
power-pilot campaigns with their archived evidence, and the reserved final
assessment (PR 5). See [Delivery and acceptance](#delivery-and-acceptance).

The first release will measure preboard Jund Wildfire and Mono Blue Terror play
against two deterministic scripted specialists, alongside reviewed tactical
puzzles. This implements a bounded first part of
[PR57's evaluation programme](generalized-pauper-review.md#74-x--is-apparent-strength-robust-priority-3),
not its full six-archetype or human-strength panel. Game score measures performance
against the frozen opponents; puzzle score measures completion of stated tactical
objectives. Neither alone establishes general Pauper proficiency.

## Release scope and reproducibility

The four game cells are explicit learner/opponent pairs, independent of physical
seat. A learner must support both deck roles to receive the overall game score.
A specialist can run its two cells, labeled as a partial panel.

| Cell | Learner deck | Scripted opponent |
|---|---|---|
| `jund_vs_jund` | Jund Wildfire | `benchmark-jund@1` |
| `jund_vs_blue` | Jund Wildfire | `benchmark-blue@1` |
| `blue_vs_jund` | Mono Blue Terror | `benchmark-jund@1` |
| `blue_vs_blue` | Mono Blue Terror | `benchmark-blue@1` |

Game one uses the exact current 60-card maindecks, without sideboarding. Freeze
their card-count maps in the release manifest, rather than looking up a mutable
`DECKS` entry at evaluation time. Initial proposed identities come from commit
`82a2eb1`, which merged PR57:

| Deck ID | Cards | SHA-256 of canonical card-count map |
|---|---|---|
| `jund_wildfire` | 60 | `998d97f37438776c175fe75ca50926eaa2490df3b869e627ea88b2391bc4b6bd` |
| `mono_blue_terror` | 60 | `7d7681cf282f5cf7d79ca2cc8629e3562d43af70078d13b23835afcc02fd17c8` |

For JSON artifacts, canonical bytes are UTF-8 with sorted object keys, no extra
spaces (`separators=(",", ":")`), `ensure_ascii=False`, preserved array order
and no trailing newline. Hash those bytes, not the filename or pretty printing.
File references also carry a SHA-256 of their actual bytes. The manifest embeds
the deck maps. The new runner expands each map in sorted card-name order before
shuffling, so JSON object insertion order cannot change seeded games. This rule
applies only to the new suite, not the legacy deck expansion. The manifest pins
the card specification, bot source/rule versions and
parameters, puzzle bundle, engine code and information contract. A changed
component creates a new suite version; results from different freezes are not
silently pooled. Engine selection is a run option, recorded in the result;
Python/native comparisons require the same engine code revision and parity checks.

Each cell and block index determines one simulator seed. A block contains:

| Slot | Learner seat | Starting player |
|---|---|---|
| 0 | 0 | 0 (learner plays first) |
| 1 | 0 | 1 (learner draws first) |
| 2 | 1 | 1 (learner plays first) |
| 3 | 1 | 0 (learner draws first) |

Construct the deck array and agents from the cell and slot; do not infer a deck
from seat 0/1 or rely on the existing matchup registry's fixed deck ordering.
Use the same simulator seed in all four slots. Each candidate repeats exactly
the same seed, deck ordering and starting player in a corresponding slot. Seat
swaps balance position; they do **not** promise the same per-deck shuffle across
slots, since the engine shuffles seat libraries in order.

Derive the seed from SHA-256 of canonical JSON
`[stream, cell, block_index]`, using the first eight digest bytes as an unsigned
big-endian integer modulo `2**31`. Streams are
`benchmark-v1/dev`, `benchmark-v1/power-pilot` and
`benchmark-v1/final/<assessment-id>`. Changing `suite_version` alone does not
change any seed, so every final assessment names itself: a final manifest
carries an `assessment` record (`id`, `exposed.checkpoints` as checkpoint
SHA-256s, `exposed.puzzle_groups` as leakage-group IDs, and `supersedes`), and
its stream must embed that ID. The ID therefore enters every simulator, actor,
puzzle and bootstrap seed below. Development and power-pilot manifests carry no
assessment and keep their streams, so existing development evidence is
unaffected. The foundation validator enforces this.
Use separate tagged hashes for policy sampling (`[stream, cell, block_index,
slot, mode, "actor"]`), puzzle sampling (`[stream, puzzle_id, repetition, mode,
"actor"]`) and bootstrap resampling (`[stream, suite_id, mode, "bootstrap"]`).
Puzzle cases share the actor seed within a repetition so an unchanged initial
information state does not acquire a different sampling policy from its hidden
completion. Policy random
generators are local to each episode, independent of worker scheduling; do not
pass simulator seeds or runner identifiers into the scripted bot's observation.

Sampled checkpoint play is primary; greedy is a separately reported mode.
Scripted bots remain deterministic in both modes. Default full-game settings
are `max_turns=100`, `auto_single=false`, `auto_mana=false`, `auto_pass=false`,
with a runner cap of 10,000 environment decisions per game. Count every engine
decision, including opponent and forced decisions. An engine turn-limit draw
scores one half with its reason recorded; hitting the runner cap is a technical
error, not a draw. Alternate action-decomposition settings require a separately
identified configuration and cannot be mixed into the primary panel.

Start size planning with 400 development pilot blocks per cell (1,600 games per
cell and mode). This is a proposed pilot budget, not a power guarantee. Freeze
the final block count from development paired variance and the intended effect
size before final evaluation, following
[PR57's uncertainty protocol](generalized-pauper-review.md#71-common-contract-for-every-experiment).
Final streams and puzzle groups are reserved for a frozen release assessment;
checkpoint/bot tuning uses development only. Retuning after final inspection
requires a new assessment ID (hence fresh seeds), fresh final puzzle groups and
a `supersedes` link to the inspected assessment. Its `exposed` lists record
every checkpoint and puzzle group whose final results were seen before the
retune; exposed puzzle groups are retired from later final splits.

## Proposed bot and adapter interface

The new scripted specialists implement this proposed protocol, shown as a
signature sketch rather than importable code:

```python
reset(own_deck: Mapping[str, int]) -> None
observe(visible_event: VisibleEvent) -> None
choose(player_view: PlayerView, legal_actions: Sequence[LegalAction]) -> int
```

`reset` starts each game or puzzle information state with the exact own-list
counts and clears all memory. An immutable `PlayerView` contains the deciding
player's permitted information, building on
[`observe`](../mtg_ml/engine/view.py): their hand, public zones and game state,
mana, publicly known or personally revealed cards, and the current decision.
It must also expose permitted own-card identities and public rule facts needed
to resolve structured actions. Unknown opponent cards and either unknown library
order remain hidden. Own-deck counts are allowed; an actual opponent deck ID,
list, private hand or future draw is not an observation.

`LegalAction` has a current `index`, decision `kind`, structured `key`, display
`label` and permitted object/stack references and choice data (such as mode,
target, mana source/color or damage allocation). The adapter derives these from
the legal options and visible rule facts; it never passes engine `Card` objects,
`Option.value` objects or a callable path back to the complete game. Only cards
actually revealed by a legal library/hand-choice decision may appear in its
choice data. Semantic data must cover all decision kinds; labels are for display,
not the source of strategy or object identity.

`choose` returns an index into the current action sequence. It must not mutate
the view or game. Equal rule scores choose the lowest current index; that order
must agree between engines. `observe` receives each visible action/reveal event
after it occurs, before the bot's next choice, from either player's actions.
Use the existing event visibility rules, including self-only choices, and never
forward the opponent's private decision menu. No answer, objective, fixture ID,
response-policy ID or release seed is delivered to an agent.

Trusted adapters own the `Game` and connect the protocol to the existing
[`act(game)` runners](../mtg_ml/agents.py) and
[`ModelAgent`](../mtg_ml/rl/agent.py). Checkpoint adapters preserve each network's
recorded features, sampling behavior and recurrent event handling. They must
satisfy the fair input contract before admission; do not silently mask or
reinterpret old checkpoint features to obtain a primary result. Existing
privileged-input checkpoints can be evaluated under a separately labeled
`privileged-diagnostic` contract. Such results never enter the fair aggregate.

Register `benchmark-jund@1` and `benchmark-blue@1` separately from `make_bot`
and the existing `bot` rollout token. Metadata includes deck ID, bot version,
source revision, rule revision, parameter digest and information contract
`own-list-hidden-opponent-v1`. Improvements to shared helpers must not change
legacy bot behavior; copy or isolate behavior where inheritance would alter it.
Historical `bench/*` metrics, existing training and the six legacy bots retain
their current meaning.

### Strategy specification and quality requirements

These are complete deck pilots, not scripts matching a known game or puzzle ID.
Use ordered expert rules and explicit mana, combat and stack calculations. Bot
strategy does not use learned policies, determinization or game-tree search.
The later bot PRs record their concrete priorities and parameters with reviewed
development fixtures covering favorable and unfavorable uses of each rule.

| Area | Jund Wildfire | Mono Blue Terror |
|---|---|---|
| Opening and mana | Keep functional colored mana and an actionable curve; bottom redundant cards; choose land/fetch/payment lines that preserve planned plays | Account for cantrip-supported hands; bottom excess lands or redundant expensive threats; preserve required blue mana |
| Sequencing | Ramp with Wildfire on an indestructible Bridge; sequence enters-tapped lands and card-draw artifacts around available mana | Use cantrips to find necessary mana/interaction and build graveyard discounts; avoid milling known valuable top cards |
| Card advantage | Value Wellspring/Lembas sacrifice triggers, Offering/Insight, Spawn and Munitions costs without giving away necessary mana or bodies | Account for Brainstorm put-backs and available shuffles; Ponder ordering/shuffling and self-mill use only known information |
| Synergies and deployment | Execute Shaman–Toxin in the correct order, maintain deathtouch/lifelink through relevant activations and account for its collateral losses | Deploy Delver and discounted Terror/Serpent while preserving justified counter mana; consider ward when exposing or protecting a threat |
| Interaction | Choose legal valuable removal targets, avoid irrelevant spells, reserve needed mana and respond to threatening combat/stack situations | Select the correct counter target, let already-handled or harmless spells resolve, calculate Force Spike/ward payments and avoid redundant counters |
| Combat and lethal | Evaluate combined attacks/blocks, survival, sacrifice outlets and full damage allocations; take available lethal | Evaluate evasive pressure, bounce/tap sequencing, combined combat, lethal and necessary defensive blocks |

All reachable decisions require an explicit handler or a documented generic
rule. Missing handlers are reported as unsupported behavior during validation,
not silently converted to option 0. This includes mulligan/bottoming, priority,
targets, mana and nonmana costs, card/order choices, modes/X, attack/block and
combat-damage decisions. Shared combat rules must use the complete legal action
space supplied by the engine correctness follow-up.

## Proposed artifact contracts

All three proposed formats use `format` and integer `version=1`. A manifest is
configuration; an executable scenario constructs a position; a puzzle adds the
assessment conditions; a result records what actually happened. References are
relative to their containing artifact and carry file digests. Unknown major
versions, missing components or hash mismatches fail validation rather than
falling back to current defaults.

| Contract | Required content |
|---|---|
| `BenchmarkManifest` | `id`, `suite_version`, split/seed stream, final-only `assessment` (ID, exposed checkpoints/puzzle groups, superseded assessments), code/card-spec freeze, embedded deck maps and hashes, pinned bots, four cells, puzzle bundle reference, input contract, modes, block count, puzzle repetitions, engine settings/limits and bootstrap configuration |
| `TacticalPuzzle` | `id`, deck, category, template and source group IDs, split, accepted review/rationale, memory mode, case references to scenario/evidence and hidden-completion identity, pinned response policy, objective, stopping boundary, decision cap and optional acceptable first-action selectors |
| `BenchmarkResult` | Manifest reference/hash, candidate checkpoint hash/path, recorded feature version/input contract and code, engine/build/runtime/hardware, modes/seeds/settings, planned/completed counts, game and puzzle records, summaries/interval method and replay references |

The manifest's four cells have equal primary weight. Result rows identify a mode,
cell, block, slot, simulator/actor seed and learner seat so comparisons can join
them exactly. The checkpoint path is descriptive; its hash is its identity. The
native extension must be built from the recorded checkout's code. Hardware and
timing settings accompany latency measurements.

### Proposed manifest example

This JSON is a configuration sketch: `<...>` digests identify artifacts to be
created and frozen in later PRs, not existing files. A release manifest embeds
both complete card-count maps; the `decks` value here is a separate excerpt to
show the exact Blue map and its real initial digest.

```json
{
  "format": "BenchmarkManifest",
  "version": 1,
  "id": "jund-blue-tactics",
  "suite_version": "1.0.0",
  "split": "dev",
  "stream": "benchmark-v1/dev",
  "freeze": {
    "code_revision": "<full-engine-and-runner-commit>",
    "card_spec_sha256": "<cards.toml-sha256>",
    "deck_bundle_sha256": "<embedded-deck-bundle-sha256>"
  },
  "information_contract": "own-list-hidden-opponent-v1",
  "bots": [
    {"id": "benchmark-jund@1", "deck": "jund_wildfire", "source_revision": "<bot-commit>", "rules_revision": 1, "parameters_sha256": "<jund-parameters-sha256>"},
    {"id": "benchmark-blue@1", "deck": "mono_blue_terror", "source_revision": "<bot-commit>", "rules_revision": 1, "parameters_sha256": "<blue-parameters-sha256>"}
  ],
  "cells": [
    {"id": "jund_vs_jund", "learner_deck": "jund_wildfire", "opponent": "benchmark-jund@1"},
    {"id": "jund_vs_blue", "learner_deck": "jund_wildfire", "opponent": "benchmark-blue@1"},
    {"id": "blue_vs_jund", "learner_deck": "mono_blue_terror", "opponent": "benchmark-jund@1"},
    {"id": "blue_vs_blue", "learner_deck": "mono_blue_terror", "opponent": "benchmark-blue@1"}
  ],
  "puzzles": {"path": "puzzles/dev.bundle.json", "sha256": "<puzzle-bundle-sha256>"},
  "modes": ["sampled", "greedy"],
  "blocks_per_cell": 400,
  "puzzle_repetitions": 1,
  "engine_settings": {"max_turns": 100, "max_decisions": 10000, "auto_single": false, "auto_mana": false, "auto_pass": false},
  "bootstrap": {"replicates": 10000, "confidence": 0.95}
}
```

```json
{
  "decks": {
    "mono_blue_terror": {
      "sha256": "7d7681cf282f5cf7d79ca2cc8629e3562d43af70078d13b23835afcc02fd17c8",
      "cards": {
        "Island": 16, "Delver of Secrets": 4, "Tolarian Terror": 4,
        "Cryptic Serpent": 4, "Brainstorm": 4, "Ponder": 3,
        "Thought Scour": 4, "Mental Note": 4, "Counterspell": 4,
        "Lorien Revealed": 4, "Force Spike": 3, "Deem Inferior": 3,
        "Sleep of the Dead": 2, "Plunder the Trollshaws": 1
      }
    }
  }
}
```

A final manifest replaces the split and stream above and adds its assessment
record. This excerpt shows a retune after the first final assessment exposed
one checkpoint and one puzzle group:

```json
{
  "split": "final",
  "stream": "benchmark-v1/final/release-1-b",
  "assessment": {
    "id": "release-1-b",
    "exposed": {"checkpoints": ["<inspected-checkpoint-sha256>"], "puzzle_groups": ["jund-removal-before-endstep"]},
    "supersedes": ["release-1-a"]
  }
}
```

## Tactical puzzle construction and scoring

Target 100 accepted puzzles, 50 per deck. For each deck allocate ten to each of
five categories: `mana-sequencing`, `combat`, `stack-interaction`,
`resource-preservation`, `deck-synergy`. Each category has five development and
five final puzzles, giving 25 development and 25 final puzzles per deck. Group
by strategic template and source game/video before assigning a split. Renamed
cards, shuffled object IDs, adjacent frames and small variants of one solution
stay in the same group. Merge intersecting template/source groups transitively;
the resulting leakage groups are also the puzzle bootstrap units. Keep the
final groups out of bot development fixtures
and routine CI; public repository storage does not itself prevent contamination.

Fifty final puzzles per deck, five per deck/category, may collapse into fewer
independent leakage groups, and the game-count power calculation says nothing
about them. Before the final assessment, predeclare the smallest puzzle-success
difference the release should detect, and size the corpus from development
group-level variance. Minimum coverage, counted in independent leakage groups:
at least 20 per deck before reporting a deck-level puzzle interval, and at
least 5 per deck/category before a category-level claim. Below those minimums
the interval is `unavailable` and the score is reported as descriptive only.
Category scores stay descriptive until a predeclared analysis supports them.

Reuse [`ExecutableScenario` version 1](expert-data.md#review-and-compile),
its evidence/assumption accounting, accepted reviews, legal `setup_actions`,
object bindings and unambiguous selectors. The compiler already constructs
positions and verifies recorded continuations on both engines. It does **not**
currently evaluate an agent's line; the tactical runner is new work.
Use the existing review shape, `status`, `reviewer` and `note`; `note` contains
the review rationale.

Synthetic tactical fixtures are allowed with explicit synthetic provenance and
review. They are not recorded human demonstrations. For sourced positions,
respect evidence availability and declared hidden completions. Damage,
attachments, combat, mana pools and arbitrary stacks cannot be injected into the
current initial format; build supported positions with legal setup actions.
Do not silently broaden the scenario schema or substitute unsupported cards.

Each puzzle case pins a compiled scenario/evidence pair, named hidden completion
and deterministic response policy. Multiple cases can vary hidden worlds or
defenses while preserving the initial player information. Use the appropriate
frozen benchmark bot as the default response policy; puzzle-specific response
rules, when needed, are separately versioned and reviewed. Every response policy
must handle every legal branch reachable in that fixture, including a wrong
learner action. A fixed recorded continuation cannot stand in for a response
policy that must react to the learner.

The evaluated agent chooses **all** its decisions, including targets, payments,
ordering and passes. The runner resets memory at the target position, then
feeds visible events throughout the attempt. Setup actions establish the
position, not fabricated recurrent history. Require a self-contained initial
information state; exclude puzzles whose decision depends on missing earlier
reveals or action history. Replaying a real prefix into memory is a later
extension, distinct from v1's reset mode.

Objectives are declarative predicates over `observe(game, perspective)` and
named visible object bindings. V1 supports conjunctions of path comparisons
(`eq`, `gte`, `lte`, `contains`) and named-object presence/absence in public
zones. It must not score access to hidden facts. Record both the tactical reason
for the objective and the reviewed successful witness lines; witnesses verify
solvability but are never supplied to the evaluated agent or used to force its
actions. Equivalent lines are accepted by their outcome, not exact sequence.

An objective is checked at the first declared stopping boundary **after at least
one environment decision**, or at game termination if that occurs first. V1
boundaries are terminal state or a specified turn/step at the learner's decision
with an empty stack and resolved entry triggers. A goal that happens to be true
initially or briefly mid-resolution does not complete the puzzle. Each case has
an explicit cap no greater than 64 environment decisions. Failure to achieve
the objective by that cap is a strategic failure (`horizon_exhausted`), including
repeated legal passes; an illegal index/agent exception is a technical error.

One attempt per case and mode is the default. A puzzle succeeds only if every
declared completion/response case succeeds. Report individual case outcomes as
well, and never give a multi-case puzzle more aggregate weight. If a manifest
requests repeated sampled attempts, average repetition-level puzzle successes;
keep those attempts clustered with their puzzle/template. Optional
acceptable-first-action selectors measure the labeled first decision separately,
never substitute for achieving the objective.

The label is conditional on the stated hidden completions and responses. Call a
puzzle **forced** only after verifying success against all legal defenses inside
its declared horizon; a win against one scripted response is not that proof.
Record the verification coverage, review rationale and scope of that claim.

### Proposed puzzle example and worked outcome

This proposed development puzzle uses a synthetic Jund removal position. Its
future scenario file wraps the initial-state excerpt below in a complete
`ExecutableScenario` with field accounting, assumptions and review. The response
bot has no card in hand and can pass priority. Cast Down can resolve before the
learner reaches their end step. This is an illustrative fixture, not one of the
100 accepted puzzles; both reviews remain pending until independently checked.

```json
{
  "format": "TacticalPuzzle",
  "version": 1,
  "id": "jund-remove-delver",
  "deck": "jund_wildfire",
  "category": "stack-interaction",
  "template_id": "jund-removal-before-endstep",
  "source_group_id": "synthetic-jund-removal",
  "split": "dev",
  "review": {"status": "pending", "reviewer": "", "note": ""},
  "episode_mode": "reset",
  "cases": [{
    "id": "empty-opponent-hand",
    "scenario": {"path": "scenarios/jund-remove-delver.scenario.json", "sha256": "<scenario-sha256>"},
    "evidence": {"path": "evidence/jund-remove-delver.evidence.json", "sha256": "<evidence-sha256>"},
    "hidden_completion": "no-draw-before-boundary",
    "response_policy": {"id": "benchmark-blue@1", "parameters_sha256": "<blue-parameters-sha256>"}
  }],
  "objective": {"all": [
    {"path": "opponent.graveyard", "op": "contains", "value": "Delver of Secrets"},
    {"path": "self.life", "op": "eq", "value": 20}
  ]},
  "stop": {"kind": "decision_boundary", "turn": 8, "step": "end", "active": "self", "require_empty_stack": true},
  "max_decisions": 64,
  "acceptable_first_actions": [{"player": 0, "kind": "priority", "key": ["cast", "Cast Down", "hand", "normal"]}],
  "claim": "success-against-declared-responses"
}
```

```json
{
  "initial": {
    "active": 0, "step": "main1", "seed": 5, "match_game": 1,
    "turn": 8, "lands_played": 1, "spells_cast_this_turn": 0,
    "players": [
      {"life": 20, "drawn": 0, "hand": ["Cast Down"], "hand_count": 1,
       "library": ["Swamp", "Swamp"], "library_count": 2,
       "graveyard": [], "exile": [], "battlefield": ["Swamp", "Swamp"]},
      {"life": 20, "drawn": 0, "hand": [], "hand_count": 0,
       "library": ["Island", "Island"], "library_count": 2,
       "graveyard": [], "exile": [],
       "battlefield": [{"name": "Delver of Secrets", "id": "delver", "sick": false}]}
    ]
  },
  "setup_actions": [],
  "preferred": [{"player": 0, "kind": "priority", "key": ["cast", "Cast Down", "hand", "normal"]}]
}
```

The successful witness casts Cast Down, pays with the two Swamps, targets Delver
and passes until resolution and the stopping boundary. Either Swamp payment
order succeeds. Selecting the cast action alone is not a completed solution;
passing through the turn without removing Delver fails the outcome test. In
more complex puzzles a correct cast followed by the wrong target or sequencing
can pass the first-action test while failing the objective. A selector key
resolves against current legal
options; indices are not durable solution labels. Duplicate copies require
the scenario's named object bindings, with ambiguity rejected.

## Results and comparisons

Write a versioned `BenchmarkResult` with complete game/case rows and summaries.
Replays use the existing viewer-compatible format where supported; they show
the evaluated player's information, not synthetic opponent hands or private
decision echoes. Record attempted semantic actions and public outcomes in failure
rows even when a full replay cannot be produced.

For games, report wins/draws/losses, `score=(wins+0.5*draws)/n`, termination
reasons, turn/decision counts, and decision latency median/p95 by cell, seat,
play/draw and mode. Overall game score is the mean of the four cell scores,
with equal weights. For puzzles, report success by deck/category, template
coverage, case failures, first-action accuracy where labeled, and attempted,
completed and technically failed counts. There is no combined game/puzzle scalar.

Use `status=complete|incomplete|invalid` at run level and
`status=success|failure|error` at puzzle case level. Artifact/review/compile
failures make the run invalid. Agent errors, full-game runner caps or missing
planned rows make it incomplete. A valid engine draw is a completed game; a
puzzle horizon miss is a completed strategic failure. Technical errors cannot
be dropped to improve a denominator: incomplete/invalid runs retain descriptive
partial rows, but primary aggregate scores and intervals are null. No automatic
retries select a better stochastic attempt; an infrastructure rerun uses the
same frozen inputs and seeds and records its provenance.

### Proposed result excerpt

The following numbers are invented to illustrate interpretation; they are not
measurements. This excerpt represents one completed greedy pilot cell and one
puzzle row from an incomplete full panel. The real result also carries all
provenance, settings, seeds and rows listed above.

```json
{
  "format": "BenchmarkResult",
  "version": 1,
  "manifest": {"path": "benchmark.dev.json", "sha256": "<manifest-sha256>"},
  "candidate": {"checkpoint": "runs/candidate.pt", "sha256": "<checkpoint-sha256>", "features": 7, "information_contract": "own-list-hidden-opponent-v1"},
  "engine": "native",
  "code_revision": "<full-engine-and-runner-commit>",
  "status": "incomplete",
  "counts": {
    "greedy": {"planned_games": 6400, "completed_games": 1600, "planned_puzzles": 50, "completed_puzzles": 1},
    "sampled": {"planned_games": 6400, "completed_games": 0, "planned_puzzles": 50, "completed_puzzles": 0}
  },
  "partial_cells": [{"cell": "jund_vs_blue", "mode": "greedy", "blocks": 400, "games": 1600, "wins": 960, "draws": 16, "losses": 624, "score": 0.605}],
  "puzzle_rows": [{"puzzle": "jund-remove-delver", "case": "empty-opponent-hand", "mode": "greedy", "repetition": 0, "status": "success", "objective_met": true, "first_action_correct": true, "decisions": 17}],
  "aggregate": {"game_score": null, "puzzle_success": null, "ci95": null}
}
```

Feature version 7 in this example is illustrative, not an automatic checkpoint
migration or a guarantee of fair inputs. Admission checks the actual input
contract and recorded encoder. Planned counts are per mode; the full result
keeps sampled and greedy counts/aggregates separate.

For a candidate comparison, require the same freeze, input contract, split,
modes, seed blocks, action settings and puzzle cases/repetitions. Hardware may
differ for gameplay score but must be matched to compare latency. Compare only
complete runs and reject missing/duplicate/mismatched rows. Within each game
block, average its four slots and form treatment minus baseline. Report that
paired difference, not the overlap of two independently calculated Wilson
intervals.

Use 10,000 paired bootstrap replicates with 95% percentile intervals. Resample
whole game blocks within each fixed cell, applying one shared resampled index
list across cells that share a block index, then apply uniform cell weights.
For puzzles, resample whole template/source leakage groups, carrying all their
puzzles/cases/repetitions in both arms together. Each puzzle has equal weight;
cases/repetitions remain nested, not extra independent samples. Show per-deck
and per-category results. A one-checkpoint comparison measures evaluation
uncertainty only; independent training-root comparisons add the hierarchical
outer resampling in PR57 and report root-specific effects. Do not infer training
variance from thousands of games against one checkpoint. When there are too
few independent groups to estimate an interval, report it as unavailable.

## CLI

The entry point is `mtg_ml.benchmark`, separate from the existing evaluator and
trainer. `validate` (#64), `puzzles` ([tactical-puzzles.md](tactical-puzzles.md),
#81) and `run`, `compare`, `calibrate`, `turn-limit` and `archive` (#84) exist;
[benchmark-release.md](benchmark-release.md) is the command reference and
explains how to read each artifact. `validate` checks reviews/digests,
constructs and verifies scenarios/witnesses and compares both-engine
inputs/actions/outcomes without evaluating a checkpoint. `run` defaults to the
manifest's complete panel and modes. `compare` writes paired differences from
two complete compatible runs.

```bash
# Validate the freeze and engine parity before running candidates.
.venv/bin/python -m mtg_ml.benchmark validate \
  --manifest benchmark.dev.json --engines python,native \
  --out .context/benchmark/validation.json

# A development run; checkpoint features come from its metadata.
.venv/bin/python -m mtg_ml.benchmark run \
  --manifest benchmark.dev.json --checkpoint runs/candidate.pt \
  --engine native --workers 4 --out .context/benchmark/candidate.json

# Treatment-minus-baseline comparisons, separated by mode.
.venv/bin/python -m mtg_ml.benchmark compare \
  --baseline .context/benchmark/baseline.json \
  --candidate .context/benchmark/candidate.json \
  --out .context/benchmark/comparison.json
```

An explicit `run --cells jund_vs_jund,jund_vs_blue` supports specialist
diagnostics, selecting matching learner-deck puzzles as well as game cells.
Record the selected rows as the run's planned counts and its panel as partial;
completion of that selection does not authorize a full-suite headline score.
An optional `--modes sampled` selects manifest modes; it does not relabel or
pool results. Emit JSON atomically, protect existing outputs from overwrite,
and return nonzero for invalid/incomplete runs or incompatible comparisons.
Workers affect scheduling only, not row identity or gameplay randomness.

## Delivery and acceptance

| Follow-up PR | Deliverable | Acceptance gate |
|---|---|---|
| 1. Benchmark foundation (merged, #64) | Versioned loaders/validation, visibility and checkpoint adapters, explicit deck/seat mapping, deterministic seed blocks and result writer in a new benchmark package | Synthetic development fixtures validate contracts, both-engine parity, all four slots, memory resets, hash/version errors and repeatability across worker counts; no new bot strategy |
| 2. Jund specialist (merged, #66) | Separate `benchmark-jund@1` registration, ordered rules/parameters and reviewed development fixtures | Every reachable decision has a handler; all reviewed critical development fixtures pass; full-game smoke panel finishes legally; paired comparison against the legacy Jund pilot is recorded |
| 3. Blue specialist (merged, #65) | Separate `benchmark-blue@1` registration, ordered rules/parameters and reviewed development fixtures | Same gate for Blue, including cantrip/library knowledge, counter/ward interactions and mana reservation |
| 4. Tactical runner and corpus (runner and 7 development puzzles merged, #81; the 100-puzzle corpus is still to do) | Agent-driven attempts, declarative objectives/boundaries, total response policies, case aggregation and 100 reviewed puzzles split by groups | Both-engine verification of witnesses and adverse responses; equivalent solutions succeed; wrong targets/sequencing, missing history, ambiguous selectors and horizon misses are correctly classified |
| 5. Frozen benchmark release (tooling merged, #84; campaigns, archive and final assessment still to do) | Validated suite freeze, paired statistics/CLI, existing-bot/checkpoint baselines, power-pilot sizing and reserved final assessment | Complete provenance and rows; no hidden-information dependence; reported scores/intervals match raw rows; calibration controls and turn-limit sensitivity pass; archived release bundle; baseline results and machine-qualified latency published |

The foundation may use synthetic development fixtures and legacy adapters for
smoke checks; it cannot advertise the new benchmark as released before the two
specialists and reviewed corpus exist. Jund and Blue bot PRs can proceed
independently after the foundation. The tactical runner can be developed with
development fixtures in parallel, but release response identities are frozen
only after both specialists are ready.

For each bot, the smoke panel is 40 development blocks in each of its two
opponent-deck cells (320 games), with both seats/starts. Freeze the legacy bots
from the baseline source revision. Compare the new and legacy **same-deck
pilots** against an identical fixed legacy Jund/Blue opponent panel, starting
with the 400-block-per-cell pilot; report per-cell and equal-weight mean paired
differences and intervals. A deterministic fixture pass and legal game completion
are necessary, but a claim of stronger play additionally needs a positive paired
mean difference with a 95% lower bound above zero **and** a 95% lower bound
above −3 percentage points in every cell. "No confirmed loss" is not enough: a
cell interval of −10 to +2pp has not ruled out a large regression, so it makes
the claim inconclusive. A cell upper bound below −3pp is a regression. The
merged #65/#66 tools apply the weaker gate, but their published results also
clear this one; later comparisons and the release use it. If evidence is inconclusive,
report it and size further development evaluation from the pilot rather than
claiming expert strength from a bot-versus-bot win rate. Two asymmetric decks
need not have a 50% equilibrium score.

Final release validation covers hidden-hand and unknown-library resampling,
opponent metadata changes with identical observations, deterministic bot action
and observation traces, all reachable decision kinds, legal full lines,
equivalent solutions, first-action/outcome disagreement, technical errors,
incomplete artifacts, whole-block/group bootstrap and matched worker-count
reproducibility. Hidden-information tests compare choices at the same information
state; later choices may change after a legitimately revealed card. Engine-facing
tests go into an engine-parametrized module registered in
[`tests/conftest.py`](../tests/conftest.py). Follow repository commands and
rebuild the local native extension after any later Rust change.

### Release gates

The frozen release (PR 5) also requires these end-to-end checks, run through the
public `run`/`compare` path rather than unit fixtures alone:

- **Calibration controls.** Comparing a checkpoint against itself yields exactly
  zero paired difference and a degenerate interval. Agents scripted to make known
  tactical mistakes (wrong target, wrong order, missed payment, passing through
  the boundary) fail; reviewed witness lines pass. Objective-loophole fixtures
  fail, for example achieving the removal while losing the game or violating a
  life/resource clause. Baseline results (both scripted specialists, the legacy
  bots and the reference checkpoints) report per deck/category whether the suite
  is saturated or near floor; a saturated or floor category is flagged and kept
  out of headline claims.
- **Turn-limit sensitivity.** Following
  [PR57's correctness protocol](generalized-pauper-review.md#72-f--are-we-training-the-intended-game-priority-1),
  rerun the baseline panel at 100, 200 and 400 turns on identical seeds. With a
  predeclared 0.5pp margin, the paired score-change interval between 100 and
  each longer cap must stay within ±0.5pp for each cell and the mean; report the
  cap-draw proportion. Otherwise raise `max_turns` or label results as a
  restricted-game benchmark.
- **Durable evidence.** Archive a release bundle outside any workspace (the
  append-only `~/mtg-ml-checkpoints` archive on h100-private or a release
  asset): manifests, raw game/puzzle rows, reviews, dependency versions/lock,
  native build identity, exact analysis commands and failure replays. Hashes
  alone do not preserve reproducibility after workspace cleanup, so docs cite
  the archived paths, never workspace-local `.context` directories. Existing
  Blue specialist evidence still names `.context` paths; archive it before the
  release cites it.

Admission of fair checkpoint results and release of the combat puzzles depend
on the separate PR57 correctness work: complete combat allocation options,
nontruncating relevant observations, hidden-opponent actor inputs and explicit
deck/seat handling. Verify these capabilities at the selected release revision
instead of assuming a particular future PR number or feature version supplies
them. The benchmark PRs implement none of those changes and require no
training run or remote campaign.

BO3/sideboarding, other archetypes, list-transfer panels, search, human pilots,
curriculum resets and trainer integration are later extensions. Keep final
puzzles out of demonstrations or training datasets. Preserve the existing
[evaluation metrics](evaluation.md) and
[L1 ladder](experiments/ladder.md) as historical regression checks alongside the
new suite.
