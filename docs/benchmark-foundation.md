# Benchmark foundation

`mtg_ml.benchmark` implements follow-up 1 of [the benchmark plan](benchmark-plan.md).
It depends on the PR63 correctness changes, including hidden-list feature set 7
and complete combat allocations. It is an unreleased foundation: the new Jund
and Blue specialists, agent-driven tactical runner, reviewed release corpus,
campaign commands and paired statistics are separate work.

## Available command and APIs

```bash
.venv/bin/python -m mtg_ml.benchmark validate \
  --manifest benchmark.dev.json --engines python,native \
  --out .context/benchmark/validation.json
```

`validate` admits the manifest freeze, registered bot metadata and accepted
puzzle/scenario reviews; checks referenced file bytes; then compiles and replays
every declared successful witness on the requested engines. It compares complete
scripted inputs, structured legal actions, visible events and final observations.
It checks the objective at the first specified resolved boundary after an action,
or earlier game termination. It does not execute a candidate or certify response
policy coverage over every possible branch. An unavailable engine, mismatched
freeze, missing specialist or bad witness produces an invalid report and exit 1.
Successful foundation validation does not certify a benchmark release.

The package exports:

| API | Purpose |
| --- | --- |
| `load_manifest`, `load_puzzle`, `load_result` | Validate version-1 files; errors name the artifact and field |
| `canonical_bytes`, `digest`, `file_digest` | Canonical content identity and actual referenced-file identity |
| `PlayerView`, `LegalAction`, `VisibleEvent`, `BenchmarkAgent` | Deeply immutable scripted-agent protocol |
| `AgentRegistry`, `AgentMetadata`, `REGISTRY` | Independent factories and frozen metadata |
| `ScriptedAdapter`, `CheckpointAdapter`, `LegacyAdapter`, `take` | Trusted episode/event bridges |
| `episodes`, `puzzle_plan` and the four seed helpers | Explicit seat/deck placement and stable game/case identities |
| `play_episode`, `run_episodes` | Bounded development games; no public `run` command |
| `build_result`, `write_result` | Manifest-derived bookkeeping and exclusive publication |

Factories for development games receive `(physical_seat, mode)` and must create
fresh adapters. `run_episodes(workers=N)` uses spawned processes when `N > 1`;
factories must be importable/picklable. Each episode resets both adapters and uses
its own actor generator. Results are sorted by identity rather than completion
order. Timing fields vary with worker count and hardware; gameplay seeds do not.
Protect process-launching callers with Python's usual `if __name__ == "__main__"`
guard. Scripted-only execution and validation do not require PyTorch.

For a heterogeneous panel, pass learner factories keyed by learner deck ID and
opponent factories keyed by the manifest's agent ID. The runner selects each
factory from the cell before assigning its physical seat. A single callable is
also supported for a shared checkpoint or generic synthetic adapter. Factory
selection never infers a deck from seat 0/1.

Call `reset(own_deck, actor_seed)` on every adapter before using `take`. The shared
stepper captures model events before the engine step to preserve existing
recurrent encoding, then sends scripted agents only visible events after the
step. `PlayerView.state` follows `engine.view.observe`; `own_deck` contains the
exact registered counts; `cards` contains detached permitted rule facts; and
`context` carries payment or combat progress. Actions expose semantic `data`
alongside their index, kind, key and display label. Neither engine objects nor
callbacks cross the protocol. Unknown decision semantics fail explicitly.

PR63 checkpoints reporting `hidden_list` are admitted as
`own-list-hidden-opponent-v1`. Earlier feature versions require explicit
`privileged-diagnostic`; they keep their actual encoder. No feature override or
automatic migration is provided. Legacy bots remain diagnostic development
adapters, with their behavior isolated from the new specialist registrations.

## Artifact details

The manifest/puzzle/result fields follow [PR62's contracts](benchmark-plan.md#proposed-artifact-contracts).
Both complete card-count maps must be embedded. The deck-bundle digest hashes
the complete `decks` object, including individual card-map digests. Canonical
JSON preserves arrays, sorts object keys, uses UTF-8 without ASCII escaping or
extra spaces, and has no trailing newline. Referenced files are hashed as bytes.
Duplicate JSON keys, booleans masquerading as versions/counts and nonfinite
numbers are rejected.

The additional bundle envelope is:

```json
{"format":"BenchmarkPuzzleBundle","version":1,"puzzles":[{"path":"removal.puzzle.json","sha256":"<actual-file-sha256>"}]}
```

References resolve relative to the containing artifact. A puzzle case can supply
`witnesses: [{"actions": [<scenario selectors>], "note": "review rationale"}]`.
Otherwise its reviewed scenario demonstration/continuation supplies the witness.
Preferred first actions alone are insufficient. Multiple witnesses allow
equivalent legal solutions. Named-object objectives use
`{"object":"scenario-binding","zone":"battlefield","present":false}`
or another public zone (`graveyard`, `exile`); identity follows the physical card
across zone changes. Path comparisons use the observed public state and cannot
score private hand/library contents. The foundation accepts only the claim
`success-against-declared-responses`, not a forced-solution claim.

Bot implementations register factories with `REGISTRY.register(metadata,
factory, parameters)` before invoking `validate` (or its CLI `main` in an embedding
program). The later specialist integration will supply the CLI's built-in
registrations. This PR does not install synthetic adapters under the reserved
`benchmark-jund@1` or `benchmark-blue@1` IDs. The generated fixtures in
`tests/benchmark_fixtures.py` use explicit synthetic identities and review notes;
they are not part of the eventual 100-puzzle corpus.

The code freeze compares runtime source against the declared local Git commit,
including uncommitted changes. Native validation additionally compares embedded
Rust/Cargo build inputs with this checkout, so a stale or foreign extension is
rejected. Run `make native` after Rust edits. Card-spec bytes and registration
metadata are checked independently.

## Development results and limits

`build_result` derives the selected game schedule and matching learner-deck
puzzle cases from the manifest. Puzzle rows are supplied by a future tactical
runner; absence of planned cases makes the result incomplete. Rows include their
mode, block/slot or puzzle/case/repetition, seeds and physical mapping. A normal
engine draw completes a game; an illegal action, adapter exception or runner cap
retains an error row. `max_decisions` counts all explicitly stepped environment
decisions, including forced and opponent choices. The primary configuration
disables automatic decomposition; alternatives require a named diagnostic
configuration.

Results carry planned/completed/error counts, per-cell and seat/play-draw
descriptive summaries, latency median/p95 and raw rows. Primary aggregate fields
and confidence intervals remain null in the foundation, with
`interval_method="unavailable-foundation"`; release statistics come later.
Partial, privileged, invalid and incomplete runs cannot carry primary scores.
Record the caller's actual code revision, candidate byte digest and features;
native results additionally require their build revision. Pass `workers=N` and
optional `runtime` details (including the hardware model) when building results;
the defaults capture hostname, architecture, processor, CPU count and parent
thread settings. Process workers use one Torch thread. Match hardware and
thread/worker settings before interpreting latency.

`write_result` returns the stored envelope, converts its manifest reference to
the output's relative path, verifies the manifest-derived schedule and optional
replay references, and publishes finite JSON atomically. Existing destinations
are never replaced, including concurrent publication. `load_result` rechecks the
manifest identity and plan. Development recording uses viewer-compatible frames
filtered to the learner's information, showing permitted learner menus and
public opponent choices. No final puzzles are used for routine tests, and historical `bench/*`
metrics and the legacy evaluator retain their meaning.
