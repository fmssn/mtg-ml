# Fresh width-256 overnight campaign · October 8–9, 2026

Runtime is frozen at `4a32795fd4ad25e328fda69c210c08be0d40982d`, including pinned
#53 `b80b6306b18354f87bb60ae742c70ad0369ba5b7` and merged engine fixes #63.
The draft implementation PR is [#68](https://github.com/fmssn/mtg-ml/pull/68).
The exact two commands and resolved settings live in
[r7-overnight-launch.json](r7-overnight-launch.json). Subsequent ledger-only
commits do not change the deployed runtime. Keep its checkout clean and frozen.

## Server operations

Host: `h100-private2`, login `taiga-support`; checkout `~/mtg-ml-r7/code`.
Campaign: `~/mtg-ml-r7/campaigns/overnight-20261008`.
Launched at 22:47 Berlin on October 8. The following are the recorded commands;
do not start duplicates while the sessions already exist:

```bash
cd ~/mtg-ml-r7/code
bash tools/overnight_host.sh start ~/mtg-ml-r7/campaigns/overnight-20261008
bash tools/overnight_host.sh dashboard ~/mtg-ml-r7/campaigns/overnight-20261008
bash tools/overnight_host.sh status ~/mtg-ml-r7/campaigns/overnight-20261008
# Explicitly stop just the trainers if needed:
bash tools/overnight_host.sh stop ~/mtg-ml-r7/campaigns/overnight-20261008
```

Detached tmux sessions `mtg-r7-lr150`, `mtg-r7-lr075`, `mtg-r7-watch` and
`mtg-r7-dashboard` have no dependency on the Mac or an open SSH connection.
Each arm tracks its trainer PID in `process.json`; failures are recorded without
respawning. Training logs rotate at 2 MB with three backups. Monitoring service
output is bounded by tmux history. Explicit resume uses `overnight_campaign.py
run --root CAMPAIGN --arm ARM --resume` only after reviewing the failure.

The dashboard binds exclusively to `127.0.0.1:8767`. Tailscale Serve 8443 proxies
it at <https://gpu-server1.tailc02128.ts.net:8443>. Existing ComfyUI 443 proxies
`127.0.0.1:8188`; both endpoints are tailnet-only, with no Funnel. No access-policy
changes were made. The HTML, assets and read-only API are served on the host.

The watcher writes `morning-report.md` and `.json` at 08:00 Europe/Berlin on
October 9. They are available from the dashboard's report link and
`/morning-report.md`. This is a server-generated report, not a chat notification.
The watcher keeps running and training continues toward 20M games per arm.

## Experiment and evaluation

Both fresh models use width 256/entity-attention 1/four heads/GRU/shared linear
value/features 7/seed 8, batch 2048, minibatch 2048, epochs 4, target KL 0.03, stacked
FP32 inference, selective BF16 compiled PPO, pipeline lag 1, 64 resident slots,
50% self-play and 50% recent/uniform historical opponents. Each arm owns its
historical pool. Discount/clipping/entropy/shaping defaults are preserved in
this frozen source. LR is the only learning-setting difference.

| Arm | Learner/inference buses | Trainer / inference CPUs | Rollouts | Evaluation |
|---|---|---|---|---|
| lr150 | 0A /18 | 0–1 /2 | 8–31 | 4–7 |
| lr075 | 87 /90 | 32–33 /34 | 40–63 | 36–39 |

GPU UUIDs were resolved and checked before launch. Monitoring uses CPU 3 at nice 10.
OMP/MKL threads are 1. The full CPU suite was moved to spare CPU 35 during smoke;
it does not share any allocated training or evaluation CPU.

All 15 cross pairings have weight 2; all 6 mirrors have weight 1. Feature 7 randomizes
deck-seat assignment, giving a uniform 36-cell distribution and 1/6 mirrors.
Mirrors remain explicitly selected so existing default mixtures are preserved.
80% main-deck and 20% postboard games use fixed plans, including the user-approved
mirror rows; these are tactical starting assumptions. Learned sideboarding is
outside the experiment.

Resume checkpoints are written every 25 updates; immutable historical snapshots
every 122. Routine evaluation is due each 250k games: 1000 sampled and 1000 greedy
Jund-versus-Blue games, plus 200 per fixed L1 rung. The local L1 copy has unchanged
ratings 0/33.3/50.8/78.8 and validated absolute checkpoint paths. The asynchronous
trainer coalesces queued evaluations if they fall behind.

After both arms cross 1M, a common immutable checkpoint is copied into
`evaluation-checkpoints/`; each mode evaluates 200 games for each of 36 ordered
cells per arm. Mirrors are counted once with balanced starts/seats. Head-to-head
uses 400 games for each of 21 pairings; cross weights 2 and mirror weights1. The
weighted 95% interval resamples complete seed blocks jointly across pairings.
It describes evaluation-seed uncertainty conditional on these two policies,
not training-seed uncertainty. One seed cannot establish a robust LR winner.

Matrix work and routine evaluation cooperate through per-arm file locks and
stay within reserved CPU budgets. At most one panel runs; after completion the
watcher chooses the latest common snapshot. Missing cells remain explicitly
missing, and every result records checkpoint/game count. Evaluator failure is
recorded without affecting trainers or erasing completed results.

## Validation evidence

The full fast suite passed 1,462 assertions with 26 skipped and 19 deselected
in 1,272 seconds. Its interpreter then hung on an abandoned in-process test
rollout and an inspection-only trainer. The stack was captured and only that
pytest process tree was terminated. Test-only cleanup fixes drain tiny fixture
jobs and close the inspection trainer; the deployed process-based collector is
unchanged and its simultaneous smoke/resume exited cleanly. The six targeted
cleanup/resume/crash checks passed and exited cleanly after the test-only fixes.

Raw logs are under `h100-private2:~/mtg-ml-r7/validation/`.

- `make setup` installed isolated Python/native dependencies; Python 3.11.17,
  torch 2.14.1+cu130, CUDA 13.0, NumPy 2.4.6, driver 580.126.09. No Rust changes
  followed the native build. GPU assignments and dependency versions are saved.
- 84/84 full-matrix Python/native lockstep games:21 pairings × main/postboard ×
  both starts, with state/features checks and fork checks; zero divergence.
- 2000/2000 differential fuzz games identical (431s).
- Ten CUDA scaling checks passed; additional width 256/feature7/GRU test crosses
 64-policy residency with stable buffers, eviction and reused IDs. Final test
  also uses 73 entities and 512 candidate actions and matches direct inference.
- Both configurations completed 25 simultaneous updates; five warm-up and 20
  measured. All PPO losses/statistics finite, correct lag 1 and separate resources.
  Initial hashes match. Both resumed 25→26 with schedule preservation and complete
  benchmark/L1 evaluation. Smoke state is separate from production.
- Tiny matrix smoke completed 144 cells and 21 head-to-head pairings; repeated on
  final evaluator code to exercise paired-seed summary publication.
- Focused tests cover mirrors, weighted distribution, evaluation selection,
  balanced seats/starts, paired uncertainty, partial metrics, stale collectors,
  stopped trainers, fresh metadata and incomplete matrices. Lint and whitespace
  checks pass. The original native CI import issue and eviction-fixture cache
  setup issue were repaired; failed logs are retained.
- Mac HTTPS GET checks: HTML/API/health 200, unlisted file 404, POST 501. ComfyUI
  remains 200; loopback listener and tailnet-only Serve/Funnel status verified.
  JavaScript syntax and DOM rendering/15-second refresh/stale state/mode-switch
  checks pass. Native browser automation is unavailable in this session, so no
  visual screenshot inspection was performed.

The 25-update sample yielded 1.248M games/hour(A) and 1.040M(B), with median learning
steps 5.392s/6.418s. This early pool is tiny and policies change quickly. Do not
assume 20M completes by morning; report observed progress and evaluation coverage.

When an arm completes, its supervisor copies checkpoints, history, metrics,
launch metadata and hashes to append-only
`h100-private2:~/mtg-ml-checkpoints/20261008-r7-fs7-h256-ARM/`. Completed archives
must subsequently be transferred to the primary `h100-private` archive without
overwriting existing IDs, and final results reconciled into both ledgers.
