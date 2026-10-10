# Fleet board

One command collects live data from the GPU boxes over ssh and writes a self-contained HTML page
(Artifact rules: no `<html>`/`<head>`/`<body>`, no JS, fonts only from Google Fonts, inline SVG charts, phone-width safe).

```bash
python -m tools.fleet_board                      # writes .context/fleet.html and .context/fleet.json
python -m tools.fleet_board --out PATH           # other output path
python -m tools.fleet_board --from-json .context/fleet.json   # re-render a snapshot, no ssh
```

It prints the output path and one summary line, for example `4/4 machines up · 32 GPUs (14 busy) · 10 runs live`.
Publish the file with the Artifact tool afterwards.

## How it works

- `collect.py` makes one ssh call per machine (parallel, 45 s timeout, `BatchMode`). It pipes `remote.py` to
  `python3` on the box (stdlib only, Python 3.8+). The probe is read-only: `nvidia-smi` queries, `ps`, `/proc`, and reading
  `process.json`, `metrics.jsonl`, `campaign.json`. A box that does not answer shows as "Unreachable"; the page still renders.
- GPU to run mapping: each compute-app pid from `nvidia-smi` is walked up its parent chain to the `mtg_ml.rl.train --run DIR`
  process. The pid itself is the learner, a descendant is the inference server. Other GPU users show as "Other".
- CPU lanes use the union of `Cpus_allowed_list` over a run's process tree (processes with the full mask are ignored).
- Games/h is taken over the trailing `window_minutes` of `metrics.jsonl`.
- `render.py` is pure: snapshot plus config in, HTML out.

## Adapting `fleet.toml`

- `[[machines]]`: `name`, `ssh` alias, `role` label, `color` (`pilot`, `base` or `opt`), optional `cpu_label`.
  `runs` are globs expanded on the box (a directory counts as a run if it has `metrics.jsonl` or `process.json`).
  A glob entry may set its own `color`.
- `[machines.gpus]`: annotate a PCI bus (`"2F" = { label, note, kind }`, kind `foreign` or `dead`). Shown only while none of our runs uses it.
- `[[streams]]`: the workstream cards, the only hand-written text.
- `[[describe]]`: first regex that matches a run name or handle gives the table text and the optional tile label (`short`).
- `[charts]`: every run with `ladder/elo` gets a chart; `include` and `exclude` list run names or handles.
- `[[reference]]`: dashed line on the Elo charts, `match` is a regex on the run (omit for all).
- A new colour needs tokens in `render.py` (`PALETTE` and the CSS rules). A new metric needs a line in `remote.py`.

Tests: `tests/test_fleet_board.py` (fixture snapshot, no ssh).
