# Frozen benchmark release tooling

Row 5 of the [benchmark plan](benchmark-plan.md): run a candidate, compare two
candidates with paired statistics, check that the benchmark itself behaves, and
archive the evidence. The point is a dependable test for future training
changes: equivalent good lines pass, known mistakes fail, both engines agree,
and a technical failure can never raise a score.

The tools live in `mtg_ml/benchmark/` (`stats.py`, `release.py`, `archive.py`,
`results.py`) and are driven through `python -m mtg_ml.benchmark`. The older
`tools/benchmark_{blue,jund,checkpoint}.py` scripts are unchanged so their
published evidence still reproduces.

## Commands

```bash
make native
M=.context/benchmark/dev.json
.venv/bin/python tools/puzzle_manifest.py --blocks 400 --id release-dev-v1 --out $M   # --split power-pilot for sizing
.venv/bin/python -m mtg_ml.benchmark validate  --manifest $M --engines python,native --out .context/benchmark/validation.json
.venv/bin/python -m mtg_ml.benchmark calibrate --manifest $M --engines python,native --out .context/benchmark/cal.json
.venv/bin/python -m mtg_ml.benchmark run --manifest $M --checkpoint runs/c.pt --engine native --workers 8 --out .context/benchmark/c.json
.venv/bin/python -m mtg_ml.benchmark run --manifest $M --agent benchmark-jund@1 --engine native --out .context/benchmark/jund.json
.venv/bin/python -m mtg_ml.benchmark compare --baseline base.json --candidate c.json --out cmp.json
.venv/bin/python -m mtg_ml.benchmark turn-limit --manifest $M --checkpoint runs/c.pt --caps 100,200,400 --out-dir .context/benchmark/turns
.venv/bin/python -m mtg_ml.benchmark archive build --out ~/archive/release-dev-v1 --manifest $M --result c.json --comparison cmp.json --calibration cal.json
.venv/bin/python -m mtg_ml.benchmark archive verify ~/archive/release-dev-v1
```

- `run` plays the manifest's games (both modes, four-slot blocks) and puzzles.
  `--agent` takes `benchmark-jund@1`, `benchmark-blue@1` (fair) or `legacy-jund`,
  `legacy-blue` (diagnostic); an agent plays only its own deck's cells, so its
  result is a partial panel. Checkpoints with feature sets below 7 need
  `--contract diagnostic`.
- Outputs are written atomically and never overwritten. `run` exits nonzero
  unless the result is `complete`; `compare` exits nonzero on any incompatibility.
- `--resume` reuses the clean chunks of the same run (see below).
- A manifest pins the commit it ran at, so `mtg_ml/` and `native/` must match it.
  `make native` after any Rust change; `validate` checks the installed binary.

## Reading a result (`BenchmarkResult`)

`aggregate[mode]` is recomputed from the raw rows every time a result is loaded,
and must match exactly; an edited score or interval is rejected. An `incomplete`
or `invalid` result reports `null` for every aggregate.

| field | meaning |
|---|---|
| `cells.<cell>` | games, W/D/L, `score`, `ci95` and `flag` (`saturated` ≥ 0.95, `floor` ≤ 0.05) |
| `game_score`, `game_ci95` | equal-weight mean of the cell scores, **full panel only** |
| `puzzle_success`, `puzzle_ci95` | mean over puzzles, each weighing the same; `puzzles` holds per-deck results and descriptive per-category means |
| `fair` | `false` for diagnostic candidates (legacy bots, privileged features): never a release claim |

Intervals are 95% percentile intervals from 10,000 paired bootstrap replicates
(`interval_method = paired-block-bootstrap-v1`). Games resample whole four-slot
blocks, with one shared index list across the cells. Puzzles resample whole
leakage groups (template/source groups merged transitively). A puzzle interval is
`unavailable` below 20 independent leakage groups per deck, so the seven
development puzzles never get one. Flagged cells stay out of headline claims.
The constants are recorded in every output (`statistics`).

Rows with `status = error` are technical failures. They are never scored, never
retried, and make the run `incomplete`; each gets a replay under
`<out>.failures/` (`replay_reference`). Chunks are `<out>.parts/<mode>-<cell>-<offset>.json`,
each stamped with a run key (manifest, candidate, engine, code revision, native
build). `--resume` reuses a chunk only if its key matches and it has no error
rows; an error chunk is kept as evidence and the run stops again.

Runtime provenance: argv, dependency versions, native build identity, machine,
workers and threads.

## Reading a comparison (`BenchmarkComparison`)

`compare` refuses unless both results are `complete` and share manifest, cells,
modes, engine, code revision, native build and information-contract fairness. Per
mode it reports, per cell and for the equal-weight mean, the
candidate-minus-baseline difference with a paired 95% interval, the paired puzzle
result, and a `claim`:

- `stronger`: mean lower bound > 0 **and** every cell lower bound > −3 pp;
- `regression`: any cell upper bound < −3 pp, or mean upper bound < 0;
- `inconclusive` otherwise (including an unavailable interval).

A partial panel is descriptive. A cell with `saturated`/`floor` flags on either
side is listed in `saturated_cells`. Latency is reported for both arms with
`latency_comparable`, true only when machine, processor and workers match.
`blocks_for_effect(var, delta)` in `stats.py` gives the blocks needed for a
paired mean difference `delta` at 80% power: `ceil((1.96 + 0.84)² · var / δ²)`.

## Calibration (`BenchmarkCalibration`)

`calibrate` runs the controls below and passes only if all do. It writes its
intermediate runs to `<out>.work/`.

| check | what must hold |
|---|---|
| `witnesses-pass` | every reviewed learner-only witness of every case, i.e. every equivalent line, gives puzzle success 1.0 on each engine, with identical status, reason, decisions and trace hash across engines |
| `mistakes-fail` | every declared mistake is classified as its declared `expect` and scores puzzle success 0 |
| `passing-through-fails` | a learner that only passes solves no puzzle. A solved puzzle is a corpus defect, unless passing is itself one of the case's reviewed witnesses (a "do nothing" resource-preservation puzzle), which is listed in `reviewed_pass_lines` |
| `self-compare` | a specialist panel (or `--checkpoint`, sampled) run with `workers=1` and `workers=N` has identical rows apart from timing, and `compare` gives exactly 0.0 and `[0.0, 0.0]` in every cell and the mean |
| `engine-games` | a specialist-vs-specialist panel gives identical rows on python and native |
| `errors-cannot-inflate` | an agent that raises at (block 0, slot 0), one that returns an illegal index, and one that raises only when behind on life each leave an `incomplete` result with null aggregates, and `compare` refuses it |

The fault injectors exist only inside `calibrate` and the tests; they are never
registered. A puzzle that fails a control is reported, not edited.

## Turn-limit sensitivity (`BenchmarkTurnLimitSensitivity`)

`turn-limit` derives variant manifests that differ only in `max_turns` and the id
suffix `-t<cap>` (same stream, so the same seeds), runs each, and compares the
shortest cap with each longer one using the paired block bootstrap. Status
`passed` needs every cell and the mean 95% interval within ±0.005; otherwise the
benchmark is `restricted-game` and a `max_turns` increase should be proposed.
The share of games ending on `reason == "turn limit"` is reported per cap, mode
and cell.

## Archive layout

`archive build` creates a new directory (it refuses an existing one):

```
ARCHIVE.json     roles: manifest, results, comparisons, calibration, sensitivity, validation
SHA256SUMS       every other file
commands.txt     each artifact's recorded argv
artifacts/       the named files and everything they reference (manifest, puzzle
                 bundle, puzzles, scenarios, evidence, reviews, chunks, failure
                 replays, calibration work), with relative paths preserved
environment/     pip-freeze.txt, toolchain.txt (rustc, cargo), native-build.json,
                 git.txt, source.tar.gz (git archive of the frozen revision)
```

`archive verify <dir>` rechecks `SHA256SUMS`, then reloads every result (which
recomputes its aggregates), recomputes every comparison from the referenced
results and every sensitivity report from its stored rows. Archives are
append-only: `chmod -R a-w` after verifying.

## What is still open

The 100-puzzle corpus, final-split groups and the reserved final assessment.
Until then puzzle results are descriptive.
