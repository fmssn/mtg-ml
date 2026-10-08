# Jund development comparison — 2026-10-08

`benchmark-jund@1` passes PR62’s stronger-play confidence gates on this fixed development panel. The equal-weight paired improvement over the legacy Jund pilot is **14.44 percentage points**, with a **95% bootstrap interval of [12.62, 16.22] points**. Both opponent cells improve. This measures play against these frozen legacy opponents on development seeds; it is not a reserved-release assessment or evidence of expert or human-level strength.

All **6,400 planned comparison games** and **320 smoke games** completed with no technical failures. The comparison has 400 four-game blocks per opponent and pilot; the smoke uses the first 40 development blocks and overlaps the comparison. No runtime tuning followed the final source freeze. Later commits add tests and result documentation only.

## Scores

Wins/draws/losses score 1/½/0. Each cell has 1,600 games per pilot.

| Opponent | Specialist W/D/L | Legacy W/D/L | Specialist score | Legacy score | Paired difference (95% CI), points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy Jund | 1093/0/507 | 800/0/800 | 68.31% | 50.00% | +18.31 [16.12, 20.50] |
| Legacy Blue | 673/0/927 | 504/0/1096 | 42.06% | 31.50% | +10.56 [7.75, 13.31] |

The overall scores are 55.19% for the specialist and 40.75% for legacy. The bootstrap uses 10,000 draws of whole four-game blocks, preserving pilot pairing and sharing resampled block indices across the two cells. The overall lower bound exceeds zero; neither cell has a confirmed loss greater than three points. These intervals quantify evaluation uncertainty for this fixed pilot, not training variance.

## Seats and starts

| Opponent | Jund seat | Starting player | Specialist score | Legacy score |
| --- | ---: | ---: | ---: | ---: |
| Legacy Jund | 0 | 0 | 67.25% | 50.25% |
| Legacy Jund | 0 | 1 | 68.25% | 45.25% |
| Legacy Jund | 1 | 1 | 70.00% | 54.75% |
| Legacy Jund | 1 | 0 | 67.75% | 49.75% |
| Legacy Blue | 0 | 0 | 46.00% | 33.00% |
| Legacy Blue | 0 | 1 | 37.75% | 28.25% |
| Legacy Blue | 1 | 1 | 46.25% | 37.50% |
| Legacy Blue | 1 | 0 | 38.25% | 27.25% |

## Runtime and latency

The comparison took **312.82 seconds** (5.21 minutes); the smoke took **26.77 seconds**. Machine: Intel Xeon Platinum 8462Y+, 64 physical cores, Linux 5.15.0-164-generic x86_64, Python 3.11.17. Campaigns used the native engine, 32 processes pinned to CPU0–31, `OMP_NUM_THREADS=1`, and no GPU. The final full test suite ran concurrently on CPU32–39.

Latency is measured around the adapter plus choice, pooled over that pilot’s decisions. The specialist includes constructing/freeze-copying the public input; legacy receives the engine through its diagnostic adapter. These numbers are not isolated strategy-function timings.

| Pilot | Opponent | Median, ms | p95, ms |
| --- | --- | ---: | ---: |
| specialist | Legacy Jund | 1.227 | 1.921 |
| specialist | Legacy Blue | 1.336 | 2.001 |
| legacy | Legacy Jund | 0.024 | 0.104 |
| legacy | Legacy Blue | 0.021 | 0.103 |

## Provenance and verification

- Candidate and corrected-engine source: `6892b6c64fdad28f7befbb8070892cee424b9941`.
- Legacy pilot/opponents and inherited PR64 baseline: `ac7e95ad583122e8031f051d0ffcac515540d9b2`.
- Parameter SHA256: `ae78cd031937091182d8be444c6f4bdd84c18e1ead382e1bca668fe7f1fe7f95`.
- Card-spec SHA256: `4ad79d116513b02993949cf096848004698568c03990f1de25cfaa4f25b7b211`.
- Tool SHA256: `b9bb664ec1a6b42aba9f75eee6a87996a92b4f21f4261022d63aa68c49697268`.
- Stream: `benchmark-v1/dev`; automatic single/mana/pass actions disabled; 100-turn and 10,000-decision limits.
- Both pilots use the same engine and identical scheduled decks, seeds, seats and starts. Legacy source files were checked against the pinned baseline.
- Sixteen matched games verified full sampled/greedy and Python/native trace/outcome equivalence for both pilots against both opponents. The deterministic campaign was run once in greedy mode.
- Engine-parametrized development fixtures cover critical tactical lines and input contracts; the final focused specialist/comparison suite has 80 passing tests. The final campaign-source foundation selection passed 143 tests with two expected skips.
- `make test-fast` passed 1,547 tests (16 skipped, 23 deselected) on the prior runtime in 912.02 seconds. The subsequent bounded resource/timing corrections passed the final 80-test specialist suite and 143-test combined selection. The earlier full suite emitted a native CardView cross-thread cleanup warning; no test failed.

Both deck hashes, registry metadata, bootstrap differences and the complete machine/settings/source record are in [the machine-readable summary](benchmark-jund-results.json). Raw reports retain every game row, row score, seed, seat/start, decision/turn count, end reason, elapsed time and per-game median latency. Their compressed copies reproduce the original JSON byte-for-byte:

| Artifact | Compressed size | Original JSON SHA256 |
| --- | ---: | --- |
| [comparison](data/jund-6892b6c-comparison.json.gz) | 323,771 bytes | `e7106cf9b117a0215f55876b7396ada30ef515952b79608f50188ae1e7872cca` |
| [smoke](data/jund-6892b6c-smoke.json.gz) | 18,962 bytes | `c849939e8a791f57cdb9e423e84ffc78e5b9c38508e2d5b35b834df435474c3f` |

The report’s paired statistics were recomputed from its retained raw rows after transfer and matched exactly. To rerun the saved inference:

```sh
.venv/bin/python - <<'PY'
import gzip, json
from tools.benchmark_jund import paired_summary
with gzip.open("docs/data/jund-6892b6c-comparison.json.gz", "rt") as f:
    report = json.load(f)
assert paired_summary(report["rows"], 400) == report["comparison"]
PY
```

For a fresh run at the exact freeze, check out the candidate revision in an isolated workspace, run `make setup`, and use the commands in [the strategy and reproduction guide](benchmark-jund.md). The earlier `3d0146b` smoke and `c60e561`/`9334e59` comparisons are superseded development runs, not additional samples in these estimates. Static review and new outcome fixtures after the earlier comparisons found missed Munitions lethal (distinct Spawn costs and valuable fodder) and a false post-damage growth lethal. The resource and timing corrections were driven by those fixtures. All 320 smoke and 6,400 paired games were rerun at the final freeze; none of the earlier games enter its estimates. The earlier compressed `c60e561` and `9334e59` reports remain in `docs/data/` for audit. No legacy, training, sideboard or engine-rule implementation changed.
