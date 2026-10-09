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
