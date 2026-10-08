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

The comparison took **316.58 seconds** (5.28 minutes); the smoke took **26.85 seconds**. Machine: Intel Xeon Platinum 8462Y+, 64 physical cores, Linux 5.15.0-164-generic x86_64, Python 3.11.17. Campaigns used the native engine, 32 processes pinned to CPU0–31, `OMP_NUM_THREADS=1`, and no GPU. The test suite ran concurrently on CPU32–39.

Latency is measured around the adapter plus choice, pooled over that pilot’s decisions. The specialist includes constructing/freeze-copying the public input; legacy receives the engine through its diagnostic adapter. These numbers are not isolated strategy-function timings.

| Pilot | Opponent | Median, ms | p95, ms |
| --- | --- | ---: | ---: |
| specialist | Legacy Jund | 1.219 | 1.880 |
| specialist | Legacy Blue | 1.336 | 1.969 |
| legacy | Legacy Jund | 0.024 | 0.104 |
| legacy | Legacy Blue | 0.021 | 0.103 |

## Provenance and verification

- Candidate and corrected-engine source: `c60e5611f94d8f5cd2a2f6b4c54969e339c601dd`.
- Legacy pilot/opponents and inherited PR64 baseline: `66345da4c5bff4a1a31ab0047ff8b1abf3869c49`.
- Parameter SHA256: `ae78cd031937091182d8be444c6f4bdd84c18e1ead382e1bca668fe7f1fe7f95`.
- Card-spec SHA256: `4ad79d116513b02993949cf096848004698568c03990f1de25cfaa4f25b7b211`.
- Tool SHA256: `ecfdaa30bc89467f78c3e885b92d83b1f44730a303b116081a8d198bfa5f12a4`.
- Stream: `benchmark-v1/dev`; automatic single/mana/pass actions disabled; 100-turn and 10,000-decision limits.
- Both pilots use the same engine and identical scheduled decks, seeds, seats and starts. Legacy source files were checked against the pinned baseline.
- Sixteen matched games verified full sampled/greedy and Python/native trace/outcome equivalence for both pilots against both opponents. The deterministic campaign was run once in greedy mode.
- Engine-parametrized development fixtures cover critical tactical lines and input contracts; the final focused specialist/comparison suite has 70 passing tests. The campaign-source foundation selection passed 129 tests with two expected skips.
- An intermittent native CardView cleanup warning appeared when the combined fixture suite later collected objects on another thread; it did not cause game, parity or test failures.

Both deck hashes, registry metadata, bootstrap differences and the complete machine/settings/source record are in [the machine-readable summary](benchmark-jund-results.json). Raw reports retain every game row, row score, seed, seat/start, decision/turn count, end reason, elapsed time and per-game median latency. Their compressed copies reproduce the original JSON byte-for-byte:

| Artifact | Compressed size | Original JSON SHA256 |
| --- | ---: | --- |
| [comparison](data/jund-c60e561-comparison.json.gz) | 323,659 bytes | `1119a8db5d80b8670bbb9374a54fdcb37e58a4a4db116c5d1c059db1c902a5db` |
| [smoke](data/jund-c60e561-smoke.json.gz) | 19,062 bytes | `bb2318b4b79e98f6f3696a77ac9b38ab3b6705166c8d397f3210d775d6dff796` |

The report’s paired statistics were recomputed from its retained raw rows after transfer and matched exactly. To rerun the saved inference:

```sh
.venv/bin/python - <<'PY'
import gzip, json
from tools.benchmark_jund import paired_summary
with gzip.open("docs/data/jund-c60e561-comparison.json.gz", "rt") as f:
    report = json.load(f)
assert paired_summary(report["rows"], 400) == report["comparison"]
PY
```

For a fresh run at the exact freeze, check out the candidate revision in an isolated workspace, run `make setup`, and use the commands in [the strategy and reproduction guide](benchmark-jund.md). The initial development smoke on `3d0146b` also completed 320 games legally; it is not included in the final estimates. No legacy, training, sideboard or engine-rule implementation changed.
