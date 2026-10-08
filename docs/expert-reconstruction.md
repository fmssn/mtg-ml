# Log reconstruction from reviewed checkpoints

`mtg_ml.expert reconstruct` builds a `ReconstructionTimeline` v1 from
`EvidenceGame` v1 and a `ReconstructionSpec` v1. It preserves the raw evidence;
interpretations, occurrence corrections, hidden completions and replay coverage
are separate claims. The local review page can load both evidence and timeline,
navigate games/checkpoints, seek a locally selected recording, inspect replay
assumptions and download corrected evidence or a revised reconstruction spec.

```bash
.venv/bin/python -m mtg_ml.expert reconstruct evidence.json \
  --spec reconstruction.spec.json --out timeline.json
.venv/bin/python -m mtg_ml.expert checkpoint-template evidence.json \
  --spec reconstruction.spec.json --checkpoint decision-1 \
  --id decision-1 --out pending.scenario.json
```

## Contract

The spec identifies the source, perspective, player-name/seat mapping, aliases,
reviewed nonoverlapping game intervals, event overrides, checkpoints and windows.
A checkpoint contains `id`, `game_id`, original video `time` and reviewed state
fact IDs. A replay checkpoint also supplies
`position: {event_id: "log-10", side: "before" | "after"}`; an executable
root checkpoint may instead equal its scenario's decision time. Partial
checkpoints remain useful evidence but do not become complete engine states.
Conflicting field bindings and future/private/unreviewed checkpoint inputs fail
validation. Corrections retain reviewer, rationale and original source timing;
`duplicate_of` removes a re-shown occurrence only after independent review.

Each root window has a reviewed `start_scenario`, ordered `event_ids`,
`end_assertions` over the player view and optional `checkpoint_ids` and
`completion_domains`. A successor supplies `start_window_id` instead of a new
scenario. Its retained predecessor traces and endpoint constraints are replayed
from the same root for every completion change. It cannot reset the game to an
unrelated checkpoint. The timeline stores reproducible seeds, full completions,
exact selectors, inferred choices, event alignment and checkpoint agreement.
Exact selector labels are scoped to their persisted root/object allocation;
indices are conveniences and are re-resolved when replaying a predecessor.

Domains use `{path: "players.1.library.0", candidates: ["Island", "Swamp"]}`
and address explicitly synthetic hidden zones only. Named root card IDs survive
replacement. Anonymous opponent-hand slots are searched as multisets; explicit
slot domains retain their distinctions. Optional source deck copy counts are
`source_decklists: {g1: {"0": {Island: 18, Ponder: 4, ...}}}`. Counts constrain
known assignments during backtracking and fill remaining anonymous slots.
Repository decklists are never implicitly substituted for the recorded lists.

Reviewed casts, targets and methods must agree with completed engine casting;
draw counts use engine draw counters rather than treating a library search as
a draw. Zone transfers, counters, tokens, public reveals, turns and combat are
checked against transitions produced by legal choices. Unsupported constraints
remain unresolved. A matched window checks every selected occurrence and its
endpoint in Python/native lockstep, including observations, ordered legal
options, state, features and event hashes. The native build is required to retain
verified paths, even with `--engine python`.

## Bounds, knowledge and export

Defaults are 64 candidates, 50,000 expansions, 200 actions and 30 seconds per
window, with two escalations up to 256 candidates, 200,000 expansions and 120
seconds. All five values have CLI overrides. Domain exploration, beam pruning
and search exhaustion are reported; results never claim exhaustive hidden-state
search. `<timeline-stem>.progress.json` records the latest completed attempt so
an interrupted investigation has a bounded progress receipt.

Timeline events distinguish `verified`, `evidence_only`, `boundary` and
`removed` coverage. A full log audit does not establish one uninterrupted legal
replay. The review page's checkpoint view uses facts available at that time;
retrospective candidate states/completions are displayed separately. A selected
checkpoint exports a **pending** scenario with lineage, not an approved policy
sample. Public state, own hand and decision timing need separate evidence review.
Use the existing `compile`, `verify`, `replay` and `export` commands after review.
Each episode resets memory; guessed mana payments, passes, private choices and
reconstruction traces are never inserted as demonstrations or recurrent history.

## Recovered Kalikaiz pilot

The independent recovered-session audit covers all 390 extracted log occurrences
and six extraction gaps. Sixty-three occurrences are re-shown viewport blocks;
13 initial entries belong to the preceding match. Twelve newly decoded frames
surround the gaps, and 35 partial/complete checkpoints are retained. Two saved
hand facts were corrected because they included a card absent at that exact
frame. Three post-opening scenarios supply reviewed decisions with matching
Python/native observations, options and feature-set-6 exports.

The measured ordered **semantic occurrence** score against the reviewed visible
log annotation is 83.59% raw precision / 99.69% raw recall; after reviewed
corrections and duplicate removal it is 100% / 100%. This is not byte-for-byte
OCR accuracy, private-choice recall or a held-out extraction benchmark. The
reviewed annotation was produced by comparing extraction occurrences against
source panels; its corrected score is an annotation consistency measure. The
95% raw precision target is not met and is not the readiness gate.

The locally delivered evidence, gold annotation, specs, timelines, gap frames,
scenarios, verification reports, replays, episodes and processing receipts live
under `.context/expert-pilot/`. Media and generated datasets remain outside Git.
See that directory's recovery report for current replay coverage, runtimes,
artifact digests and remaining evidence limitations. No training was launched.
