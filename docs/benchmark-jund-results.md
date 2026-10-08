# Jund development comparison — 2026-10-08

`benchmark-jund@1` passes PR62’s stronger-play confidence gates on this fixed development panel. The equal-weight paired improvement over the legacy Jund pilot is **13.34 percentage points**, with a **95% bootstrap interval of [11.53, 15.16] points**. Both opponent cells improve. This measures play against these frozen legacy opponents on development seeds; it is not a reserved-release assessment or evidence of expert or human-level strength.

All **6,400 planned comparison games** and **320 smoke games** completed with no technical failures. The comparison has 400 four-game blocks per opponent and pilot; the smoke uses the first 40 development blocks and overlaps the comparison. No runtime tuning followed the final source freeze. Later commits add tests and result documentation only.

## Scores

Wins/draws/losses score 1/½/0. Each cell has 1,600 games per pilot.

| Opponent | Specialist W/D/L | Legacy W/D/L | Specialist score | Legacy score | Paired difference (95% CI), points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy Jund | 1075/0/525 | 800/0/800 | 67.19% | 50.00% | +17.19 [15.00, 19.38] |
| Legacy Blue | 656/0/944 | 504/0/1096 | 41.00% | 31.50% | +9.50 [6.75, 12.19] |

The overall scores are 54.09% for the specialist and 40.75% for legacy. The bootstrap uses 10,000 draws of whole four-game blocks, preserving pilot pairing and sharing resampled block indices across the two cells. The overall lower bound exceeds zero; neither cell has a confirmed loss greater than three points. These intervals quantify evaluation uncertainty for this fixed pilot, not training variance.

## Seats and starts

| Opponent | Jund seat | Starting player | Specialist score | Legacy score |
| --- | ---: | ---: | ---: | ---: |
| Legacy Jund | 0 | 0 | 66.00% | 50.25% |
| Legacy Jund | 0 | 1 | 67.00% | 45.25% |
| Legacy Jund | 1 | 1 | 68.50% | 54.75% |
| Legacy Jund | 1 | 0 | 67.25% | 49.75% |
| Legacy Blue | 0 | 0 | 45.00% | 33.00% |
| Legacy Blue | 0 | 1 | 36.50% | 28.25% |
| Legacy Blue | 1 | 1 | 45.25% | 37.50% |
| Legacy Blue | 1 | 0 | 37.25% | 27.25% |

## Runtime and latency

The comparison took **318.42 seconds** (5.31 minutes); the smoke took **26.85 seconds**. Machine: Intel Xeon Platinum 8462Y+, 64 physical cores, Linux 5.15.0-164-generic x86_64, Python 3.11.17. Campaigns used the native engine, 32 processes pinned to CPU0–31, `OMP_NUM_THREADS=1`, and no GPU. The full test suite finished before this final campaign.

Latency is measured around the adapter plus choice, pooled over that pilot’s decisions. The specialist includes constructing/freeze-copying the public input; legacy receives the engine through its diagnostic adapter. These numbers are not isolated strategy-function timings.

| Pilot | Opponent | Median, ms | p95, ms |
| --- | --- | ---: | ---: |
| specialist | Legacy Jund | 1.234 | 1.977 |
| specialist | Legacy Blue | 1.336 | 2.043 |
| legacy | Legacy Jund | 0.024 | 0.104 |
| legacy | Legacy Blue | 0.021 | 0.103 |

## Provenance and verification

- Candidate and corrected-engine source: `9334e59c733e8960925633402aa0d4f47f4fb3d8`.
- Legacy pilot/opponents and inherited PR64 baseline: `ac7e95ad583122e8031f051d0ffcac515540d9b2`.
- Parameter SHA256: `ae78cd031937091182d8be444c6f4bdd84c18e1ead382e1bca668fe7f1fe7f95`.
- Card-spec SHA256: `4ad79d116513b02993949cf096848004698568c03990f1de25cfaa4f25b7b211`.
- Tool SHA256: `b9bb664ec1a6b42aba9f75eee6a87996a92b4f21f4261022d63aa68c49697268`.
- Stream: `benchmark-v1/dev`; automatic single/mana/pass actions disabled; 100-turn and 10,000-decision limits.
- Both pilots use the same engine and identical scheduled decks, seeds, seats and starts. Legacy source files were checked against the pinned baseline.
- Sixteen matched games verified full sampled/greedy and Python/native trace/outcome equivalence for both pilots against both opponents. The deterministic campaign was run once in greedy mode.
- Engine-parametrized development fixtures cover critical tactical lines and input contracts; the final focused specialist/comparison suite has 72 passing tests. The final campaign-source foundation selection passed 135 tests with two expected skips.
- `make test-fast` passed 1,547 tests (16 skipped, 23 deselected) on the prior runtime in 912.02 seconds. The subsequent narrow Munitions correction passed the final 72-test specialist suite and 135-test combined selection. The earlier full suite emitted a native CardView cross-thread cleanup warning; no test failed.

Both deck hashes, registry metadata, bootstrap differences and the complete machine/settings/source record are in [the machine-readable summary](benchmark-jund-results.json). Raw reports retain every game row, row score, seed, seat/start, decision/turn count, end reason, elapsed time and per-game median latency. Their compressed copies reproduce the original JSON byte-for-byte:

| Artifact | Compressed size | Original JSON SHA256 |
| --- | ---: | --- |
| [comparison](data/jund-9334e59-comparison.json.gz) | 323,740 bytes | `d30cd2821b4b267ee2e974ab891fa7758eb37c0c9b4c432d00e9e91d4ef3fd0c` |
| [smoke](data/jund-9334e59-smoke.json.gz) | 19,008 bytes | `5e54e8f8d301a9efb9315802d4ca12e53fe24aa39ac2b7e0f54575ddcadd2327` |

The report’s paired statistics were recomputed from its retained raw rows after transfer and matched exactly. To rerun the saved inference:

```sh
.venv/bin/python - <<'PY'
import gzip, json
from tools.benchmark_jund import paired_summary
with gzip.open("docs/data/jund-9334e59-comparison.json.gz", "rt") as f:
    report = json.load(f)
assert paired_summary(report["rows"], 400) == report["comparison"]
PY
```

For a fresh run at the exact freeze, check out the candidate revision in an isolated workspace, run `make setup`, and use the commands in [the strategy and reproduction guide](benchmark-jund.md). The earlier `3d0146b` smoke and `c60e561` comparison are superseded development runs, not additional samples in these estimates. After the first comparison, static code review found a missed Munitions lethal using separate Spawns for mana and fodder. The added full-line regression drove a resource-accounting correction; all 320 smoke and 6,400 paired games were rerun at the new freeze. The earlier compressed `c60e561` reports remain in `docs/data/` for audit. No legacy, training, sideboard or engine-rule implementation changed.
