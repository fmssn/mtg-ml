# Training tracker

The one place to follow model training: a read-only web page with a **Live**
view (every running or recent run side by side: progress, games/h, decisions/s,
PPO entropy / KL / clip / explained variance, sampled and greedy benchmark, L1
Elo over time, GPU and host load) and a **History** view (every row of
`docs/experiments/ledger.jsonl`: parent, flags, benchmark, L1 Elo, verdict; a
sortable, filterable table plus Elo and benchmark charts over the ledger).
The page is self-contained (inline CSS/JS, SVG charts, no CDN), light and dark.

It reads versioned files produced by `mtg_ml` only: campaign manifests
(`campaign.json` from `tools/training_campaign.py` / `tools/overnight_campaign.py`),
`<run>/metrics.jsonl`, `process.json`, `train.log` and the ledger. It cannot
start, stop or modify training.

## Discovery

The server rescans on every poll (15 s), so new runs appear without a restart.
`--root GLOB` (repeatable; default `~/mtg-ml-*/campaigns/*` and `~/mtg-ml-*/runs/*`)
matches directories; one with a `campaign.json` is a campaign (each arm is a
run), one with a `metrics.jsonl` is a plain `mtg_ml.rl.train --run` directory.
Runs idle for more than `--recent-hours` (72) are hidden. `--campaign DIR`
(repeatable) names a campaign explicitly and is always shown. A live trainer is
recognised from `/proc` (`mtg_ml.rl.train --run ...`), which also yields its flags.
`metrics.jsonl` is read incrementally, so large logs stay cheap.
`--ledger` defaults to the ledger of the checkout the server runs from.

```bash
python apps/training-dashboard/server.py --port 8768 \
  --root '~/mtg-ml-*/campaigns/*' --root '~/mtg-ml-*/runs/*'
```

Endpoints (GET only): `/`, `/api/state`, `/api/history`,
`/report/<campaign>.md|.json` (morning report of a discovered campaign), `/healthz`.

## Hosting on the training box

Server on the training machine, loopback bind, `tailscale serve` HTTPS, tailnet
only, no credentials, no Funnel. One idempotent command (run on the box):

```bash
DASHBOARD_CPU=3 bash tools/dashboard_host.sh start --port 8768 --https 8444
bash tools/dashboard_host.sh status
bash tools/dashboard_host.sh stop --port 8768 --https 8444
```

`start` creates the detached tmux session `mtg-tracker` (nice 10, `taskset` when
`DASHBOARD_CPU` is set) if missing, waits for `/healthz`, checks
`tailscale serve status` and adds `--https=<https>` only if that port is free
(it refuses a port that proxies elsewhere; 443 and 8188, ComfyUI, are rejected).
`stop` removes only the entry that proxies to this server. Serve changes fall
back to passwordless `sudo` when the user is not the tailscale operator.

URL pattern: `https://<machine>.tailc02128.ts.net:<https-port>`, for example
`https://gpu-server1.tailc02128.ts.net:8444` (h100-private2) or
`https://gpu-server.tailc02128.ts.net:<port>` (h100-private). To deploy a new
checkout, rsync it to a fresh directory on the box and run `stop` then `start`
from there (the server reads code and ledger from its own checkout).

## Legacy mode

`--source ssh --host H --remote-script S --campaign DIR` keeps the old
single-campaign monitor over SSH (the `index.html` screens page, written for the
256-width computational screens). The tracker shows campaigns on the training
host, old screen campaigns included, as plain runs.

## Fresh training campaigns on h100-private2

`tools/overnight_campaign.py prepare` creates an immutable manifest for two
feature-7 runs, including exact commands, GPU UUIDs, CPU placement, source SHA,
fixed L1 ratings and a hash of the common random initialization. `run` supervises
one arm with bounded logs and no automatic restarts. `watch` performs serialized
matrix evaluation and writes the 08:00 Europe/Berlin report independently of the
agent session. Campaign archives are append-only on the training host under
`~/mtg-ml-checkpoints/`; transfer completed archives to the primary archive host
without replacing existing IDs.

Use `bash tools/overnight_host.sh start CAMPAIGN` and then
`bash tools/overnight_host.sh dashboard CAMPAIGN` to create detached server
sessions. `status` reports tracked trainer PIDs; `stop` stops only this campaign's
trainers, leaving monitoring available. `stop-monitoring` stops its dashboard
and watcher. The optional third argument is the tmux prefix (use `mtg-r7-smoke`
for smoke runs). Training logs rotate at 2 MB with three backups; service output
stays in a bounded 2,000-line tmux history. No service automatically respawns.

Mirror sideboard rows are fixed tactical assumptions, not validated optimal
choices. Mirrors remain explicit-only; the fresh campaign assigns cross-pair
weight 2 and mirror weight 1 so deck-seat randomization gives a uniform 6×6 mix.
The mirror aggregate score now counts every real game once. Evaluation jobs use
`MTG_EVAL_LOCK` to serialize matrix work with routine evaluation on reserved CPUs.
