"""Run discovery, incremental metrics reading and ledger loading for the training tracker.

Read-only. Everything here works on files produced by ``mtg_ml`` (campaign
manifests, ``metrics.jsonl``, ``process.json``, ``docs/experiments/ledger.jsonl``);
nothing imports engine internals.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_ROOTS = ('~/mtg-ml-*/campaigns/*', '~/mtg-ml-*/runs/*')
SERIES_FIELDS = ('iteration', 'games_total', 'elapsed_s', 'wall_s', 'decisions', 'games', 'entropy', 'approx_kl',
                 'clip_frac', 'explained_var', 'pg_loss', 'v_loss', 'lr', 'decisions_per_s', 'win_vs_pool',
                 'rollout_s', 'update_s', 'ppo_s')
MAX_POINTS = 300


def read_jsonl_tail(path, offset=0):
    """Complete JSON lines from ``offset``; returns (rows, new_offset). A partial last line waits."""
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b'\n') + 1
    rows = []
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a damaged line is skipped, not fatal for a monitor
    return rows, offset + end


def slim(row):
    out = {k: row[k] for k in SERIES_FIELDS if k in row and not isinstance(row[k], (dict, list))}
    ev = evaluation(row)
    return out, ev


def evaluation(row):
    """Benchmark / ladder results merged into a metrics row, or None."""
    bench, greedy = {}, {}
    for k, v in row.items():
        if not k.startswith('bench/') or k.endswith(('_ci', '_n')) or isinstance(v, (dict, list)):
            continue
        (greedy if k.endswith('_greedy') else bench)[k[6:].removesuffix('_greedy')] = v
    if not bench and not greedy and 'ladder/elo' not in row:
        return None
    return dict(iteration=row.get('iteration'), games_total=row.get('games_total'), elapsed_s=row.get('elapsed_s'),
                elo=row.get('ladder/elo'), elo_se=row.get('ladder/elo_se'), bench=bench, greedy=greedy)


class RunReader:
    """Incremental reader for metrics.jsonl, so 40 MB logs are not re-parsed every poll.

    The trainer rewrites the file (os.replace) when it merges evaluations into old rows; a new
    inode or a shorter file triggers a full re-read.
    """

    def __init__(self):
        self.cache = {}

    def read(self, path):
        path = Path(path)
        try:
            st = path.stat()
        except OSError:
            return [], []
        c = self.cache.get(str(path))
        if c is None or c['ino'] != st.st_ino or st.st_size < c['offset']:
            c = dict(ino=st.st_ino, offset=0, rows=[], evals=[])
            self.cache[str(path)] = c
        if st.st_size > c['offset']:
            new, c['offset'] = read_jsonl_tail(path, c['offset'])
            for r in new:
                s, ev = slim(r)
                c['rows'].append(s)
                if ev:
                    c['evals'].append(ev)
        return c['rows'], c['evals']


def clean(value):
    """Strict JSON: non-finite numbers become null."""
    if isinstance(value, float) and (value != value or abs(value) == float('inf')):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def tail(path, limit=2500):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def active_trainers():
    """{resolved run dir: (pid, argv)} for live `mtg_ml.rl.train` processes (Linux /proc)."""
    out = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().decode().split('\0')
            if 'mtg_ml.rl.train' in args and '--run' in args:
                run = Path(args[args.index('--run') + 1])
                if not run.is_absolute():
                    run = Path(os.readlink(p / 'cwd')) / run
                out[str(run.resolve())] = (int(p.name), args)
        except (OSError, UnicodeError, IndexError):
            continue
    return out


def flags_of(command):
    """`--flag value` pairs of a command line (flags without a value get 'true')."""
    flags, i = {}, 0
    command = [str(c) for c in command]
    while i < len(command):
        c = command[i]
        if c.startswith('--'):
            if '=' in c:
                k, v = c[2:].split('=', 1)
                flags[k], i = v, i + 1
            elif i + 1 < len(command) and not command[i + 1].startswith('--'):
                flags[c[2:]], i = command[i + 1], i + 2
            else:
                flags[c[2:]], i = 'true', i + 1
        else:
            i += 1
    return flags


def downsample(rows, limit=MAX_POINTS):
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    picked = [rows[int(i * step)] for i in range(limit)]
    if picked[-1] is not rows[-1]:
        picked.append(rows[-1])
    return picked


def number(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float('inf') else None


def run_summary(run, reader, active, meta=None, now=None, campaign=None):
    """One run's state for the UI. ``meta`` carries campaign-manifest fields when there are any."""
    run, meta, now = Path(run), meta or {}, now or time.time()
    rows, evals = reader.read(run / 'metrics.jsonl')
    pid, argv = active.get(str(run.resolve()), (None, None))
    flags = flags_of(meta['command']) if meta.get('command') else {}
    if argv:
        flags.update(flags_of(argv))
    try:
        process = json.loads((run / 'process.json').read_text())
    except (OSError, ValueError):
        process = {}
    last = rows[-1] if rows else {}
    recent = rows[-20:]
    wall = sum(r.get('wall_s', 0) or 0 for r in recent)
    gph = sum(r.get('games', 0) or 0 for r in recent) * 3600 / wall if wall else None
    dps = sum(r.get('decisions', 0) or 0 for r in recent) / wall if wall else None
    target = number(meta.get('target_games')) or number(flags.get('total-games') and float(flags['total-games']))
    games = last.get('games_total', 0) or 0
    try:
        age = now - (run / 'metrics.jsonl').stat().st_mtime
    except OSError:
        age = None
    if pid:
        status = 'running'
    elif target and games >= target:
        status = 'finished'
    elif process.get('status') in ('failed', 'finished', 'complete', 'stopped'):
        status = process['status']
        if status == 'failed' and process.get('exit_code') in (143, -15, 130, -2):
            status = 'stopped'  # SIGTERM / SIGINT: a requested stop, not a crash
    elif not rows:
        status = 'queued'
    elif age is not None and age < 120:
        status = 'running'  # metrics still moving (e.g. process not visible from this user)
    else:
        status = 'stopped'
    eta = (target - games) * 3600 / gph if target and gph and status == 'running' and games < target else None
    series = []
    for r in downsample(rows):
        s = {k: number(r.get(k)) for k in ('iteration', 'games_total', 'elapsed_s', 'entropy', 'approx_kl', 'clip_frac',
                                           'explained_var', 'pg_loss', 'v_loss', 'lr', 'decisions_per_s', 'win_vs_pool')}
        s['gph'] = r['games'] * 3600 / r['wall_s'] if r.get('games') and r.get('wall_s') else None
        series.append(s)
    name = meta.get('name') or run.name
    return dict(id=f'{campaign}/{name}' if campaign else str(run), name=name, label=meta.get('label') or name,
                campaign=campaign, path=str(run), status=status, pid=pid, iteration=last.get('iteration', 0),
                games=games, target_games=target, elapsed_s=last.get('elapsed_s', 0), games_per_hour=gph,
                decisions_per_s=dps, eta_s=eta, metrics_age_s=age, flags=flags, code_ref=meta.get('code_ref'),
                seed=meta.get('seed'), features=last.get('features') or meta.get('features'),
                gpus=meta.get('gpus'), latest={k: number(last.get(k)) for k in ('entropy', 'approx_kl', 'clip_frac', 'explained_var', 'lr', 'win_vs_pool')},
                series=series, evals=evals, log=tail(run / 'train.log'))


def discover(globs, explicit=()):
    """Directories matched by ``globs`` (plus explicit campaign dirs) as [(kind, path)]; kind campaign|run."""
    seen, found = set(), []
    for pattern in list(globs) + list(explicit):
        for m in sorted(glob.glob(os.path.expanduser(pattern))):
            p = Path(m)
            if not p.is_dir() or str(p.resolve()) in seen:
                continue
            if (p / 'campaign.json').exists():
                kind = 'campaign'
            elif (p / 'metrics.jsonl').exists():
                kind = 'run'
            else:
                continue
            seen.add(str(p.resolve()))
            found.append((kind, p))
    return found


def build_state(globs, explicit=(), reader=None, active=None, recent_hours=72, now=None, gpus=None):
    """Live state for every discovered campaign and run. Explicit directories are always shown."""
    reader = reader or RunReader()
    now = now or time.time()
    active = active_trainers() if active is None else active
    campaigns, runs, hidden = [], [], 0
    explicit_set = {str(Path(os.path.expanduser(e)).resolve()) for e in explicit}
    for kind, path in discover(globs, explicit):
        always = str(path.resolve()) in explicit_set
        if kind == 'run':
            s = run_summary(path, reader, active, now=now)
            if always or s['status'] == 'running' or (s['metrics_age_s'] or 1e12) < recent_hours * 3600:
                runs.append(s)
            else:
                hidden += 1
            continue
        try:
            entries = json.loads((path / 'campaign.json').read_text())
        except (OSError, ValueError):
            continue
        members = []
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict) or 'run' not in e:
                continue
            rd = Path(e['run'])
            if not rd.exists():
                rd = path / rd.name
            if not rd.exists():
                continue
            meta = dict(e)
            meta['label'] = e.get('label') or e.get('arm') or e.get('name')
            members.append(run_summary(rd, reader, active, meta, now, campaign=path.name))
        fresh = any(m['status'] == 'running' or (m['metrics_age_s'] or 1e12) < recent_hours * 3600 for m in members)
        if not (fresh or always):
            hidden += 1
            continue
        runs.extend(members)
        alerts = [f"{m['name']}: {m['status']}" for m in members if m['status'] == 'failed']
        campaigns.append(dict(name=path.name, path=str(path), runs=[m['id'] for m in members], alerts=alerts,
                              report_ready=(path / 'morning-report.md').exists() or (path / 'morning-report.json').exists()))
    try:
        disk = shutil.disk_usage(Path.home())
        disk = dict(free=disk.free, total=disk.total)
    except OSError:
        disk = None
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return dict(schema_version=2, mode='tracker', observed_at=now, host=os.uname().nodename, campaigns=campaigns,
                runs=runs, hidden=hidden, gpus=gpus if gpus is not None else gpu_inventory(), load=load,
                cpu_count=os.cpu_count(), disk=disk)


def gpu_inventory():
    import csv
    import subprocess
    query = 'index,pci.bus_id,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw'
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'], text=True, timeout=12, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for f in csv.reader(raw.splitlines()):
        f = [x.strip() for x in f]
        if len(f) < 8:
            continue
        n = lambda s: float(s) if s.replace('.', '', 1).isdigit() else None  # noqa: E731
        out.append(dict(index=int(f[0]), bus=f[1].split(':')[-2], name=f[2], used_mib=n(f[3]), total_mib=n(f[4]),
                        utilization=n(f[5]), temperature=n(f[6]), power_w=n(f[7])))
    return out


# ---- history: docs/experiments/ledger.jsonl ----

def _first(results, prefix, avoid=('old_engine',), exact=None):
    if exact and results.get(exact) is not None:
        return number(results[exact])
    keys = [k for k in results if k.startswith(prefix) and not any(a in k for a in avoid)]
    return number(results[sorted(keys)[0]]) if keys else None


def verdict_class(v):
    v = (v or '').lower()
    if v.startswith(('adopt', 'best', 'works', 'promising')):
        return 'ok'
    if v.startswith(('reject', 'abort', 'no effect', 'fail')):
        return 'bad'
    if v.startswith(('control', 'parent', 'baseline')):
        return 'ref'
    return 'pend'


def load_ledger(path):
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or 'id' not in r:
            continue
        res = r.get('results') or {}
        rows.append(dict(id=r['id'], date=r.get('date'), parent=r.get('parent'), code=r.get('code'), engine=r.get('engine'),
                         change=r.get('change'), flags=r.get('flags'), games=number(r.get('games')), verdict=r.get('verdict'),
                         verdict_class=verdict_class(r.get('verdict')), archive=r.get('archive'), results=res,
                         elo=_first(res, 'elo_L1', exact='elo_L1'), sampled=_first(res, 'bench_sampled'),
                         greedy=_first(res, 'bench_greedy')))
    return rows
