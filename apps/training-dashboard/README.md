# Live training dashboard

A read-only monitor for `tools/training_campaign.py` campaigns. It reads the
campaign manifest, complete metric records, outcome records, bounded log tails,
live trainer processes and `nvidia-smi`. The dashboard distinguishes provisional
numbers from completed screens, excludes warm-up from throughput, and reports
stopped jobs and stale SSH data. It polls every 15 seconds even when the page is
closed; it cannot start, stop or modify training.

Run from the workspace with its Python environment:

```bash
scp apps/training-dashboard/snapshot.py \
  h100-private:/home/taiga-support/mtg-ml-256-opt/dashboard_snapshot.py
.venv/bin/python apps/training-dashboard/server.py \
  --port 55012 \
  --campaign /home/taiga-support/mtg-ml-256-opt/campaigns/screens-20261008-dedicated
```

Open <http://127.0.0.1:55012>. The server binds only to the local loopback
interface and uses the existing `h100-private` SSH connection. No credentials
are stored in the dashboard. `--host`, `--remote-script`, `--campaign`,
`--interval` and `--cache` can be overridden; the default cache is
`.context/training-dashboard/state.json`. The remote interpreter is the
isolated `mtg-ml-256-opt/code/e8662a5/.venv/bin/python` used by this campaign.

Charts use actual recorded learner decisions / iteration wall time. Learner,
collection and publication stages overlap and must not be added together.
Remaining time is an estimate for the selected screen from its last five
iterations. Running GPU-hours use the last recorded elapsed time; completed
GPU-hours use the runner's charged time, including startup. Strength per hour
requires the subsequent paired training/evaluation runs.

## Fresh training campaigns on h100-private2

`tools/overnight_campaign.py prepare` creates an immutable manifest for two
feature-7 runs, including exact commands, GPU UUIDs, CPU placement, source SHA,
fixed L1 ratings and a hash of the common random initialization. `run` supervises
one arm with bounded logs and no automatic restarts. `watch` performs serialized
matrix evaluation and writes the 08:00 Europe/Berlin report independently of the
agent session. Campaign archives are append-only on the training host under
`~/mtg-ml-checkpoints/`; transfer completed archives to the primary archive host
without replacing existing IDs.

The dashboard supports both original screen manifests and `mode: training`
manifests. Training snapshots expose progress, rolling rates, PPO statistics,
benchmark/L1 history and incremental matrix results. Old SSH monitoring remains
supported with `--python` selecting the remote interpreter.

On the training host (inside a detached tmux session):

```bash
taskset -c 3 nice -n 10 .venv/bin/python apps/training-dashboard/server.py \
  --source local --campaign /home/taiga-support/mtg-ml-r7/campaigns/overnight \
  --cache /home/taiga-support/mtg-ml-r7/dashboard-state.json --port 8767
# Add only this port; do not reset the existing ComfyUI configuration.
tailscale serve --bg --https=8443 http://127.0.0.1:8767
```

The Python listener remains loopback-only. Tailscale Serve provides HTTPS within
the tailnet at `https://gpu-server1.tailc02128.ts.net:8443`; do not enable Funnel.
The endpoints are GET-only `/`, `/api/state`, and `/healthz`. No model loading or
GPU allocation occurs in the dashboard. Snapshot failures retain the last data
with a visible error; interrupted trainers are distinguished from delayed updates.

Mirror sideboard rows are fixed tactical assumptions, not validated optimal
choices. Mirrors remain explicit-only; the fresh campaign assigns cross-pair
weight 2 and mirror weight 1 so deck-seat randomization gives a uniform 6×6 mix.
The mirror aggregate score now counts every real game once. Evaluation jobs use
`MTG_EVAL_LOCK` to serialize matrix work with routine evaluation on reserved CPUs.
