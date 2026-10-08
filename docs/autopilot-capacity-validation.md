# DeepSeek review capacity validation — 2026-10-08

Draft delivery: [PR #54](https://github.com/fmssn/mtg-ml/pull/54). Live defaults are
unchanged until this PR is approved and merged.

## Result

The target passed: **11/13 historical reviews completed** within the 15-minute
model budget, with **4/4 targeted PR #30 controls passing**. No capacity-only
Opus escalation occurred. PRs #34 and #45 honestly reported incomplete coverage
or unfinished repair; the production controller would disable auto-merge and
hand them to the developer. All relevant replay checkouts had their own freshly
installed native extension; none accepted a failed build as completion.

| Measure | Audited initial reviews | New shadow reviews |
|---|---:|---:|
| Completed with DeepSeek | 5/13 | 11/13 |
| Capacity-only Opus escalations | 8/13 | 0/13 |
| Incomplete handoffs | Previously routed to Opus | 2/13 |
| Median DeepSeek model step | 2m51s | 6:13 |
| Maximum new model step | — | 9:52 |
| Evidence-preserving continuations | None | 5/13 |

The older eight Opus escalations produced six clean verdicts and two
non-behavioral documentation/whitespace edits. The new historical sample
produced three independently verified fixes in PR #31 and one verified,
unfixed defect in PR #34. No cosmetic edits were observed in the final replay
patches. Native verification and more investigation take longer; this change
improves completion and useful findings, and does not optimize CI runtime.

Historical timings were measured on GitHub's Ubuntu runners. Replays were
measured on the local Mac (macOS-26.6.2-arm64-arm-64bit); timings exclude setup/native
preparation. This is not a matched-hardware performance comparison. CLI dollar
estimates are not provider billing records.

## Historical cases

The original checkout logs confirmed all head/base pairs. The PR #45 dispatch
run's `head_sha` was the workflow commit, so the PR checkout SHA was used.
Recorded bases were merged before review; four early trials that omitted those
merges were superseded and excluded. Replays used `deepseek-flash`, max effort,
Claude Code 2.1.293, 60 initial turns, and one tool-disabled continuation.

| PR | Verdict | Model time | Continued | Denied calls |
|---|---|---:|---|---:|
| [#31](https://github.com/fmssn/mtg-ml/pull/31) | fixed | 6:13 | no | 7 |
| [#32](https://github.com/fmssn/mtg-ml/pull/32) | clean | 1:14 | no | 2 |
| [#33](https://github.com/fmssn/mtg-ml/pull/33) | clean | 6:34 | yes | 5 |
| [#34](https://github.com/fmssn/mtg-ml/pull/34) | incomplete | 9:52 | yes | 4 |
| [#35](https://github.com/fmssn/mtg-ml/pull/35) | clean | 6:07 | yes | 5 |
| [#36](https://github.com/fmssn/mtg-ml/pull/36) | clean | 6:33 | no | 10 |
| [#41](https://github.com/fmssn/mtg-ml/pull/41) | clean | 0:32 | no | 1 |
| [#43](https://github.com/fmssn/mtg-ml/pull/43) | clean | 8:49 | yes | 6 |
| [#45](https://github.com/fmssn/mtg-ml/pull/45) | incomplete | 5:33 | yes | 4 |
| [#47](https://github.com/fmssn/mtg-ml/pull/47) | clean | 0:47 | no | 4 |
| [#48](https://github.com/fmssn/mtg-ml/pull/48) | clean | 8:05 | no | 4 |
| [#50](https://github.com/fmssn/mtg-ml/pull/50) | clean | 8:23 | no | 4 |
| [#51](https://github.com/fmssn/mtg-ml/pull/51) | clean | 3:21 | no | 6 |

The 62 denied calls are retained in the logs, including initial and resumed
phases. Some investigations still spent turns on unsupported shell/probe
commands. In particular, PR #34 exhausted its budget after identifying a
mechanical fix; it was not relabeled clean or escalated just for capacity.

## Findings checked independently

- **PR #31, identical options in counterfactual comparison:** deterministic
  all-win playouts report a win rate of 2.0 before the patch and 1.0 afterward.
- **PR #31, default model calibration key:** naming `deepseek-chat` explicitly
  makes calibration read `findings-deepseek-chat.json`, while review writes
  `findings-deepseek.json`. The replay fix makes the keys agree.
- **PR #31, native mask faults:** hiding the first native option records
  `Mulligan (to 6)` while Rust executes `Keep (7 cards)`. The patch rejects this
  unsupported native fault before recording.
- **PR #34, audit replay flags:** an example recorded with automatic mana/pass
  rebuilds with `Play Island` where the original decision offers
  `Cast Mental Note`, on both engines. Passing the recorded flags reconstructs
  the correct state. The final replay detected this but did not finish fixing it.
- **PR #30, separate seed exposure:** the live response's seed reconstructs the
  opponent's seven-card hand exactly on both engines. Shadow patches withhold
  that seed from normal live responses. Client-selected deterministic seeds
  remain possible; this is not a complete adversarial hardening assessment.

Shadow domain fixes were made only in isolated checkouts, with no GitHub writes.
They are evidence of review quality, not changes included in this PR.

## PR #30 controls

| Defect | Before fix | After fix |
|---|---|---|
| Opponent Delver/top-card prompt | Detected and patched | No corresponding finding |
| Native shutdown on the wrong thread | Detected and patched | No corresponding finding |

Both repaired cases independently found the separate seed exposure. The control
matcher requires the original prompt/top-card or thread/shutdown defect; it does
not count every hidden-information issue as recurrence. Regression tests cover
this distinction. These are targeted controls, not blinded discovery trials:
case directory names identify their defect and before/after position. They
verify the known regressions and do not establish a general bug-catching rate.

## Implementation validation and reproduction

- 35 controller/harness tests pass, including turn exhaustion/continuation,
  unfinished reviews, malformed output, API failure, native failure, conflict
  staging, protected edits, GitHub mutation rejection and process-group deadlines.
- Local full CPU selection passed: 1232 tests, 11 skipped, 16 deselected. The later
  controller additions were tested separately in the 35-test focused run.
- Ruff, shell syntax, diff whitespace and actionlint 1.7.12 pass.
- Required `lint`, `python` and `native` CI checks must pass on the final PR head.
  The CI workflow and merge gates are unchanged.

Exact commits, PR context and original run IDs are in
[`autopilot-replay-cases-2026-10-08.json`](autopilot-replay-cases-2026-10-08.json).
The harness creates clones with no remotes, one environment per case, and never
runs the finish/push/merge controller:

```bash
npm exec --yes --package @anthropic-ai/claude-code@2.1.293 -- claude --version
.venv/bin/python tools/autopilot_replay.py --manifest docs/autopilot-replay-cases-2026-10-08.json --jobs 3
```

Detailed model streams, full diff artifacts, native build logs, patches and
manual reproductions remain in `.context/autopilot-replay/`. Receipts report
cumulative conversation tokens/cost once and per-phase usage separately.
