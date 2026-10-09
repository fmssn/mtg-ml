"""Read-only, bounded snapshot of a training campaign; run on the training host."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

FIELDS = ('iteration', 'games', 'games_total', 'decisions', 'elapsed_s', 'wall_s', 'rollout_s', 'ppo_s', 'publish_s', 'queue_s', 'wait_s', 'entropy', 'approx_kl', 'updates', 'learner_peak_bytes', 'learner_peak_bytes_by_rank', 'residency', 'early_stop', 'policy_lag')
LABELS = {'legacy': 'Eager attention', 'stacked': 'Stacked attention', 'separate': 'Dedicated inference GPU', 'resident64': '64 resident policies', 'resident128': '128 resident policies', 'bf16': 'BF16 learner', 'learners2': 'Two learner GPUs', 'learners4': 'Four learner GPUs', 'batch8192-lr150': 'Batch 8192 · LR 1.5e−4', 'batch8192-lr300': 'Batch 8192 · LR 3e−4', 'epochs2': 'Two PPO epochs', 'lag2': 'Policy lag two'}


def jsonl(path):
    """A concurrently appended partial last line is not a completed record."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or line.endswith('\n'):
                raise ValueError(f'Invalid complete JSON record in {path.name}, line {i + 1}') from None
    return rows


def tail(path, limit=6000):
    if not path.exists():
        return ''
    with path.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - limit))
        return f.read().decode('utf-8', errors='replace')


def process_inventory():
    active, supervisor = {}, []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().decode().split('\0')
            if 'mtg_ml.rl.train' in args and '--run' in args:
                active[args[args.index('--run') + 1]] = int(p.name)
            if any(a.endswith('/start_computational_checks.sh') for a in args):
                supervisor.append(int(p.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return active, supervisor


def gpu_inventory():
    query = 'pci.bus_id,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw'
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'], text=True, timeout=5)
    out = []
    for fields in csv.reader(raw.splitlines()):
        bus, uuid, name, used, total, util, temp, power = [f.strip() for f in fields]
        number = lambda s: float(s) if re.fullmatch(r'[\d.]+', s) else None
        out.append(dict(bus=bus.split(':')[-2], uuid=uuid, name=name, used_mib=number(used), total_mib=number(total), utilization=number(util), temperature=number(temp), power_w=number(power), project=bus.split(':')[-2] in {'0A', '18', '87', '90', 'C7'}))
    return out


def summarize(rows, warmup):
    timed = rows[warmup:]
    wall = sum(r.get('wall_s', 0) for r in timed)
    if not timed or wall <= 0:
        return None
    return dict(timed_iterations=len(timed), games_per_hour=sum(r['games'] for r in timed) * 3600 / wall,
                decisions_per_s=sum(r['decisions'] for r in timed) / wall,
                median_rollout_s=statistics.median(r['rollout_s'] for r in timed),
                median_ppo_s=statistics.median(r['ppo_s'] for r in timed),
                median_publication_s=statistics.median(r['publish_s'] for r in timed))


def snapshot(root, active=None, supervisor=None, gpus=None):
    root = Path(root).resolve()
    entries = json.loads((root / 'campaign.json').read_text())
    if active is None:
        active, supervisor = process_inventory()
    outcomes = {r['name']: r for r in jsonl(root / 'outcomes.jsonl')}
    arms, alerts = [], []
    for e in entries:
        run = Path(e['run'])
        if run.parent.resolve() != root:
            raise ValueError('Run directories must be children of the selected campaign')
        rows = jsonl(run / 'metrics.jsonl')
        result = outcomes.get(e['name'])
        running = str(run) in active
        if result:
            status = 'failed' if result['verdict'] == 'failed' else ('skipped' if result['verdict'] == 'skipped' else 'complete')
        else:
            status = 'running' if running else ('interrupted' if rows or (run / 'train.log').exists() else 'queued')
        provisional = summarize(rows, e['warmup'])
        last = rows[-1] if rows else {}
        phase = 'initializing' if not rows else ('warm-up' if len(rows) < e['warmup'] else 'timed')
        expected = e['warmup'] + e['timed']
        eta = (expected - len(rows)) * statistics.median(r['wall_s'] for r in rows[-5:]) if len(rows) > e['warmup'] and running else None
        args = dict(zip(e['command'][3::2], e['command'][4::2]))
        projected = [{k: r[k] for k in FIELDS if k in r} for r in rows[-1000:]]
        for p in projected:
            p['decisions_per_s'] = p['decisions'] / p['wall_s'] if p.get('wall_s') else None
        if status == 'failed':
            alerts.append(f"{LABELS.get(e['arm'], e['arm'])}: {result.get('error') or result.get('reason') or 'exit code ' + str(result.get('exit_code'))}")
        arms.append(dict(name=e['name'], arm=e['arm'], label=LABELS.get(e['arm'], e['arm']), status=status,
                         phase=phase, iterations=len(rows), expected=expected, warmup=e['warmup'], timed=e['timed'],
                         latest=projected[-1] if projected else None, metrics=projected, provisional=provisional,
                         result=result, eta_s=max(eta, 0) if eta is not None else None, pid=active.get(str(run)),
                         gpus=e['gpus'], cpus=e['cpus'], code_ref=e['code_ref'], seed=e['seed'],
                         configuration={k.removeprefix('--'): v for k, v in args.items() if k in ('--workers', '--ppo-precision', '--ppo-minibatch', '--ppo-epochs', '--pipeline', '--server-resident-limit', '--learner-devices', '--server-device')},
                         elapsed_s=last.get('elapsed_s', 0), log=tail(run / 'train.log'), log_updated= (run / 'train.log').stat().st_mtime if (run / 'train.log').exists() else None))
    validation = tail(root / 'four-gpu.log')
    passed = re.search(r'(\d+) passed', validation)
    validation_status = 'passed' if passed and 'failed' not in validation else ('failed' if any(x in validation for x in ('FAILED', 'Fatal Python error', 'Error')) else 'pending')
    if validation_status == 'failed':
        alerts.append('Four-GPU validation failed; inspect the validation log')
    supervisor = supervisor or []
    if any(a['status'] == 'running' for a in arms):
        state = 'running'
    elif alerts:
        state = 'failed'
    elif all(a['status'] in ('complete', 'skipped') for a in arms):
        state = 'complete'
    elif supervisor:
        state = 'starting'
    else:
        state = 'stopped'
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return dict(schema_version=1, observed_at=time.time(), host='h100-private', campaign=root.name, directory=str(root),
                status=state, arms=arms, alerts=alerts, supervisor_pids=supervisor, gpus=gpu_inventory() if gpus is None else gpus,
                load=load, cpu_count=os.cpu_count(), validation=dict(status=validation_status, passed=int(passed[1]) if passed else 0, log=validation),
                supervisor_log=tail(root / 'supervisor.log'), runner_log=tail(root / 'screens-runner.log') or tail(root / 'baseline-runner.log'),
                parent_sha256=entries[0]['parent']['parent_sha256'], parent_games=entries[0]['parent']['continuation']['games_total'])


def clean(value):
    """Strict JSON: nonfinite training statistics are represented as missing."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', required=True)
    print(json.dumps(clean(snapshot(ap.parse_args().campaign)), allow_nan=False))
