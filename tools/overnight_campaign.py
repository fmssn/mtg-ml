"""Fresh feature-7 campaign: prepare, supervise, evaluate and report on the host.

No automatic restarts. All run directories and source identities are immutable.
The dashboard consumes campaign.json, metrics.jsonl and evaluation.json only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtg_ml.match import MATCHUPS  # noqa: E402

ARMS = ("lr150", "lr075")
DECK_ORDER = ("jund_wildfire", "mono_blue_terror", "red_madness", "grixis_affinity", "elves", "tron")


def matrix_mix():
    return ",".join(f"{name}:{1 if a == b else 2}" for name, (a, b) in MATCHUPS.items())


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def records(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\n"):
            out.append(json.loads(line))
    return out


def tensor_hash(path=None):
    import torch
    from mtg_ml.rl.model import PolicyNet
    if path:
        weights = torch.load(path, map_location="cpu", weights_only=False)["model"]
    else:
        torch.manual_seed(8)
        weights = PolicyNet(hidden=256, trunk="entity", value_net="shared", memory="gru", entity_attn=1, features=7).state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(args):
    root, code, ladder = (Path(x).resolve() for x in (args.root, args.code, args.ladder))
    if (root / "campaign.json").exists():
        raise FileExistsError("Campaign already exists; use a new directory")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
        raise ValueError("Deploy a clean committed source tree")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True)
    gpu_by_bus = {line.split(',')[0].strip().split(':')[-2]: line.split(',')[1].strip() for line in gpu_rows.splitlines()}
    original = json.loads((ladder / "ladder.json").read_text())
    expected = {488: 0.0, 1464: 33.3, 2440: 50.8, 3904: 78.8}
    mapped = {}
    for iteration, rating in expected.items():
        path = ladder / f"iter_{iteration:05d}.pt"
        matches = [v for k, v in original["ratings"].items() if Path(k).name == path.name]
        if matches != [rating] or not path.is_file():
            raise ValueError(f"Missing or changed fixed L1 rung {path}")
        mapped[str(path)] = rating
    root.mkdir(parents=True, exist_ok=True)
    atomic(root / "ladder.json", {**original, "ratings": mapped})
    initial = tensor_hash()
    entries = []
    for i, arm in enumerate(ARMS):
        run = root / arm
        if run.exists():
            raise FileExistsError(run)
        run.mkdir()
        offset = i * 32
        buses = (("0A", "18"), ("87", "90"))[i]
        flags = {
            "run": str(run), "hidden": 256, "trunk": "entity", "entity-attn": 1,
            "value-net": "shared", "value-bound": "none", "memory": "gru", "features": 7,
            "seed": 8, "engine": "native", "device": "cuda:0", "inference": "server",
            "server-device": "cuda:1", "server-cpus": str(offset+2), "trainer-cpus": f"{offset}-{offset+1}",
            "worker-cpus": f"{offset+8}-{offset+31}", "eval-cpus": f"{offset+4}-{offset+7}",
            "workers": 24, "eval-workers": 4, "games-per-iter": 2048,
            "server-policy-slots": 64, "server-resident-limit": 64, "server-stacked-attention": 1,
            "ppo-precision": "bf16", "ppo-capture": 2, "ppo-minibatch": 2048, "ppo-epochs": 4,
            "ppo-target-kl": 0.03, "ppo-lr": (1.5e-4, 7.5e-5)[i], "ppo-lr-final": (1.5e-5, 7.5e-6)[i],
            "lr-anneal-games": 20000000, "lr-schedule": "linear", "pipeline": 1,
            "self-play-frac": 0.5, "pool-recent-frac": 0.5, "pool-sampling": "uniform", "bot-frac": 0,
            "postboard-frac": 0.2, "matchup": matrix_mix(), "checkpoint-every": 25,
            "snapshot-every": 122, "iterations": 25 if args.smoke else 100000000,
            "total-games": 0 if args.smoke else 20000000, "eval-every": 0,
            "eval-every-games": 0 if args.smoke else 250000, "eval-extra-matchups": 0,
            "eval-final": 1, "eval-games": 0, "eval-blocks": "",
            "bench-games": 8 if args.smoke else 1000, "bench-greedy-games": 8 if args.smoke else 1000,
            "bench-bo3-matches": 0, "eval-bo3-matches": 0,
            "ladder-games": 8 if args.smoke else 200, "ladder": ",".join(mapped), "ladder-ratings": str(root / "ladder.json"),
        }
        command = [str(code / ".venv/bin/python"), "-m", "mtg_ml.rl.train"]
        command += [part for k, v in flags.items() for part in ("--" + k, str(v))]
        env = dict(CUDA_VISIBLE_DEVICES=",".join(gpu_by_bus[b] for b in buses), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TORCHINDUCTOR_COMPILE_THREADS="1", MTG_EVAL_LOCK=str(run / "evaluation.lock"), PYTHONUNBUFFERED="1")
        entries.append(dict(name=arm, arm=arm, mode="training", run=str(run), code=str(code), code_ref=source,
                            command=command, environment=env, gpus=env["CUDA_VISIBLE_DEVICES"].split(","),
                            cpus=[offset, offset+1, offset+2, *range(offset+4, offset+32)], eval_cpus=list(range(offset+4, offset+8)),
                            seed=8, features=7, target_games=51200 if args.smoke else 20000000,
                            warmup=5, initial_tensor_sha256=initial, parent=None, created_at=time.time(), smoke=args.smoke))
    atomic(root / "campaign.json", entries)
    print(root / "campaign.json")


def entries(root):
    return json.loads((Path(root) / "campaign.json").read_text())


def preflight(entry):
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    occupied = {line.split(',')[0].strip() for line in apps.splitlines()}
    if occupied & set(entry["gpus"]):
        raise RuntimeError(f"Allocated GPUs occupied: {occupied & set(entry['gpus'])}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=entry["code"], text=True).strip() != entry["code_ref"]:
        raise RuntimeError("Source revision changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=entry["code"], text=True).strip():
        raise RuntimeError("Source checkout is dirty")
    for p in Path('/proc').glob('[0-9]*'):
        try:
            command = (p / 'cmdline').read_bytes().split(b'\0')
            if b'mtg_ml.rl.train' in command and set(os.sched_getaffinity(int(p.name))) & set(entry['cpus']):
                raise RuntimeError(f"CPU allocation overlaps trainer {p.name}")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def run_arm(args):
    entry = next(e for e in entries(args.root) if e['name'] == args.arm)
    run = Path(entry['run'])
    import fcntl
    with (run / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preflight(entry)
            if (run / 'latest.pt').exists() and not args.resume:
                raise ValueError('Use --resume explicitly for a previous run')
        except Exception as exc:
            atomic(run / 'process.json', dict(status='failed', error=str(exc), finished_at=time.time(), code_ref=entry['code_ref']))
            raise
        command = ['taskset', '-c', ','.join(map(str, entry['cpus'])), *entry['command']]
        p = subprocess.Popen(command, cwd=entry['code'], env=os.environ | entry['environment'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        state = dict(status='running', pid=p.pid, started_at=time.time(), code_ref=entry['code_ref'])
        atomic(run / 'process.json', state)
        def stop(sig, _):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        log = run / 'train.log'
        with log.open('ab', buffering=0) as output:
            while chunk := p.stdout.read1(65536):
                if output.tell() + len(chunk) > 2_000_000:
                    for n in (3, 2, 1):
                        older = run / f'train.log.{n}'
                        if older.exists():
                            if n == 3:
                                older.unlink()
                            else:
                                older.replace(run / f'train.log.{n+1}')
                    output.close()
                    log.replace(run / 'train.log.1')
                    output = log.open('ab', buffering=0)
                output.write(chunk)
            output.close()
        rc = p.wait()
        # Clean only this trainer's remaining descendants, never another run.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        initial_path = run / 'pool/iter_00000.pt'
        observed = tensor_hash(initial_path) if initial_path.exists() else None
        rows = records(run / 'metrics.jsonl')
        complete = rc == 0 and bool(rows) and rows[-1]['games_total'] >= entry['target_games'] and observed == entry['initial_tensor_sha256']
        state.update(status='complete' if complete else 'failed', exit_code=rc, finished_at=time.time(), initial_tensor_sha256=observed)
        atomic(run / 'process.json', state)
        if complete and not entry['smoke']:
            archive(entry)
        if not complete:
            raise SystemExit(1)


def archive(entry):
    import shutil
    dest = Path.home() / 'mtg-ml-checkpoints' / ('20261008-r7-fs7-h256-' + entry['name'])
    if (dest / 'COMPLETE').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    run = Path(entry['run'])
    for name in ('latest.pt', 'metrics.jsonl', 'process.json', 'train.log'):
        target = dest / ('final.pt' if name == 'latest.pt' else name)
        if not target.exists():
            shutil.copy2(run / name, target)
    for folder in ('pool', 'policy'):
        shutil.copytree(run / folder, dest / folder, dirs_exist_ok=True)
    atomic(dest / 'launch.json', entry)
    (dest / 'SHA256').write_text(hashlib.file_digest((dest / 'final.pt').open('rb'), 'sha256').hexdigest() + '  final.pt\n')
    (dest / 'COMPLETE').write_text(datetime.now().isoformat() + '\n')


def choose_common(arms):
    sets = [{p.name for p in (Path(e['run']) / 'pool').glob('iter_*.pt')} for e in arms]
    common = sorted(set.intersection(*sets))
    return common[-1] if common else None


def panel(args):
    import torch
    from mtg_ml.backend import ENV_VAR
    from mtg_ml.rl.collect import PoolProcess
    from mtg_ml.rl.evaluate import DECK_KEYS, benchmark, evaluation_lock
    os.environ[ENV_VAR] = 'native'
    torch.set_num_threads(1)
    arms = entries(args.root)
    path = Path(args.root) / 'evaluation.json'
    state = json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'cells': [], 'h2h': []}
    checkpoint = args.checkpoint or choose_common(arms)
    if not checkpoint:
        return
    if state.get('completed_checkpoint') == checkpoint:
        return
    import shutil
    frozen = Path(args.root) / 'evaluation-checkpoints' / checkpoint.removesuffix('.pt')
    frozen.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        dest = frozen / (arm['name'] + '.pt')
        if not dest.exists():
            tmp = dest.with_suffix('.tmp')
            shutil.copy2(Path(arm['run']) / 'pool' / checkpoint, tmp)
            tmp.replace(dest)
    iteration = int(Path(checkpoint).stem.split('_')[1])
    state.update(checkpoint=checkpoint, games=iteration*2048, status='running', updated_at=time.time())
    atomic(path, state)
    # Resume exact cells from the pinned common checkpoint, never silently mix snapshots.
    existing = {(r['arm'], r['matchup'], r['seat'], r['greedy']) for r in state['cells'] if r['checkpoint'] == checkpoint}
    for entry in arms:
        policy = str(frozen / (entry['name'] + '.pt'))
        cpus = tuple(entry['eval_cpus'])
        os.sched_setaffinity(0, set(cpus))
        pool = PoolProcess(len(cpus), 'local', worker_cpus=cpus, cpus=cpus, nice=10, name='matrix')
        try:
            for matchup, (a, b) in MATCHUPS.items():
                for seat in ((0,) if a == b else (0, 1)):
                    for greedy in (False, True):
                        key = (entry['name'], matchup, seat, greedy)
                        if key in existing:
                            continue
                        with evaluation_lock(str(Path(entry['run']) / 'evaluation.lock')):
                            result = pool.call(benchmark, policy, 0 if greedy else args.games, 0, len(cpus), 0, 100,
                                               'local', args.games if greedy else 0, False, False, matchup, seat).wait()
                        deck = DECK_KEYS[(a, b)[seat]]
                        k = f"bench/{deck}_vs_bot" + ('_greedy' if greedy else '')
                        state['cells'].append(dict(arm=entry['name'], matchup=matchup, seat=seat, greedy=greedy,
                                                   row=(a,b)[seat], column=(a,b)[1-seat], score=result[k], ci=result[k+'_ci'],
                                                   n=result[k+'_n'], checkpoint=checkpoint, games=iteration*2048))
                        state['updated_at'] = time.time()
                        atomic(path, state)
        finally:
            pool.close()
    cpus = tuple(arms[0]['eval_cpus'])
    os.sched_setaffinity(0, set(cpus))
    pool = PoolProcess(len(cpus), 'local', worker_cpus=cpus, cpus=cpus, nice=10, name='h2h')
    try:
        for matchup, (a, b) in MATCHUPS.items():
            if any(r['matchup'] == matchup and r['checkpoint'] == checkpoint for r in state['h2h']):
                continue
            with evaluation_lock(str(Path(arms[0]['run']) / 'evaluation.lock')):
                result = pool.call(h2h_cell, str(frozen / (arms[0]['name'] + '.pt')),
                                   str(frozen / (arms[1]['name'] + '.pt')), args.h2h_games, len(cpus), matchup).wait()
            state['h2h'].append(dict(matchup=matchup, score=result['all'][0], ci=result['all'][1], n=result['all'][2],
                                     weight=1 if a == b else 2, checkpoint=checkpoint, seed_scores=result['seed_scores']))
            state['h2h_summary'] = paired_summary([r for r in state['h2h'] if r['checkpoint'] == checkpoint])
            state['updated_at'] = time.time()
            atomic(path, state)
    finally:
        pool.close()
    state.update(status='complete', completed_checkpoint=checkpoint, updated_at=time.time())
    atomic(path, state)


def h2h_cell(pool, a, b, games, jobs, matchup):
    from collections import defaultdict
    from mtg_ml.rl.evaluate import paired_specs, score
    from mtg_ml.rl.rollout import Job, LEARNER, play
    specs = paired_specs(b, games, matchup=matchup, balance_seats=True)
    played = play(pool, specs, Job([], a, 0, record=False, max_turns=100, inference='local'), jobs).games
    result = score(played, matchup)
    blocks = defaultdict(list)
    for seats, winner, _, _, _, seed in played:
        seat = 0 if seats[0] == LEARNER else 1
        blocks[str(seed)].append(.5 if winner is None else float(winner == seat))
    result['seed_scores'] = {seed: sum(xs)/len(xs) for seed, xs in blocks.items()}
    return result


def paired_summary(rows):
    """Resample complete seed blocks jointly across matchups, preserving pairing."""
    import random
    if not rows or any(not r.get('seed_scores') for r in rows):
        return None
    seeds = sorted(set.intersection(*(set(r['seed_scores']) for r in rows)))
    if not seeds:
        return None
    weight = sum(r['weight'] for r in rows)
    xs = [sum(r['weight']*r['seed_scores'][seed] for r in rows)/weight for seed in seeds]
    rng = random.Random(8)
    samples = sorted(sum(rng.choices(xs, k=len(xs)))/len(xs) for _ in range(2000))
    return dict(score=sum(xs)/len(xs), ci=[samples[49], samples[1949]], seed_blocks=len(xs),
                pairings=len(rows), complete=len(rows)==21,
                method='paired-seed block bootstrap; conditional on these two trained policies')


def report(root):
    root = Path(root)
    result = dict(generated_at=datetime.now(ZoneInfo('Europe/Berlin')).isoformat(), arms=[])
    lines = ['# Overnight training report', '', result['generated_at'], '']
    for entry in entries(root):
        rows = records(Path(entry['run']) / 'metrics.jsonl')
        last = rows[-1] if rows else {}
        evaluated = [r for r in rows if 'ladder/elo' in r]
        recent = rows[-20:]
        wall = sum(r.get('wall_s', 0) for r in recent)
        speed = sum(r.get('games', 0) for r in recent)*3600/wall if wall else None
        process_file = Path(entry['run']) / 'process.json'
        process = json.loads(process_file.read_text()) if process_file.exists() else {'status': 'queued'}
        outcome = dict(arm=entry['name'], latest=last, evaluation=evaluated[-1] if evaluated else None,
                       games_per_hour=speed, process=process, code_ref=entry['code_ref'])
        result['arms'].append(outcome)
        lines += [f"## {entry['name']}", f"Games: {last.get('games_total', 0):,}; learning rate: {last.get('lr', 'pending')}"]
        lines += [f"Status: {process['status']}; rolling games/hour: {round(speed) if speed else 'pending'}; elapsed hours: {last.get('elapsed_s', 0)/3600:.2f}.",
                  f"Policy loss: {last.get('pg_loss')}; value loss: {last.get('v_loss')}; entropy: {last.get('entropy')}; KL: {last.get('approx_kl')}; clipping: {last.get('clip_frac')}."]
        if evaluated:
            e = evaluated[-1]
            lines += [f"Evaluated at {e['games_total']:,} games: sampled {e.get('bench/jund_vs_bot')}; greedy {e.get('bench/jund_vs_bot_greedy')}; L1 Elo {e['ladder/elo']}."]
            lines += [f"Sampled 95% CI: {e.get('bench/jund_vs_bot_ci')}; greedy 95% CI: {e.get('bench/jund_vs_bot_greedy_ci')}; L1 SE: {e.get('ladder/elo_se')}."]
        else:
            lines += ['Routine evaluation pending.']
        lines += ['']
    if (root / 'evaluation.json').exists():
        panel_state = json.loads((root / 'evaluation.json').read_text())
        result['matrix'] = panel_state
        checkpoint = panel_state.get('checkpoint')
        cells = [c for c in panel_state.get('cells', []) if c['checkpoint'] == checkpoint]
        h2h = [c for c in panel_state.get('h2h', []) if c['checkpoint'] == checkpoint]
        lines += [f"Matrix: {panel_state.get('status')}; checkpoint {checkpoint} ({panel_state.get('games')} games); {len(cells)}/144 cells; {len(h2h)}/21 head-to-head pairings."]
        if h2h:
            weight = sum(c['weight'] for c in h2h)
            lines += [f"Descriptive weighted A score: {sum(c['weight']*c['score'] for c in h2h)/weight:.3f}. Coverage {'complete' if len(h2h) == 21 else 'incomplete'}; per-pair Wilson intervals below are diagnostic, not a paired-seed aggregate confidence interval."]
            lines += [f"- {c['matchup']}: {c['score']:.3f}, 95% CI {c['ci']}, n={c['n']}" for c in h2h]
            summary = panel_state.get('h2h_summary')
            if summary:
                lines += [f"Weighted paired-seed bootstrap 95% CI: {summary['ci']} across {summary['seed_blocks']} seed blocks; conditional on these two trained policies, not training-seed uncertainty."]
    lines += ['', 'One seed per arm. Incomplete evaluations are pending, not losses. Healthy training continues to 20M games.',
              '', 'Live dashboard: https://gpu-server1.tailc02128.ts.net:8443']
    atomic(root / 'morning-report.json', result)
    tmp = root / 'morning-report.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n')
    tmp.replace(root / 'morning-report.md')


def watch(args):
    arms = entries(args.root)
    child = None
    reported = False
    due = datetime(2026, 10, 9, 8, tzinfo=ZoneInfo('Europe/Berlin')).timestamp()
    while True:
        if not reported and time.time() >= due:
            report(args.root)
            reported = True
        if child is not None and child.poll() == 0:
            evaluation = json.loads((Path(args.root) / 'evaluation.json').read_text())
            if choose_common(arms) != evaluation.get('completed_checkpoint'):
                child = None
        if child is None:
            counts = [records(Path(e['run']) / 'metrics.jsonl') for e in arms]
            if all(rows and rows[-1]['games_total'] >= 1000000 for rows in counts):
                chosen = choose_common(arms)
                cmd = [sys.executable, __file__, 'panel', '--root', str(args.root), '--checkpoint', chosen]
                child = subprocess.Popen(cmd, env=os.environ | {'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}, start_new_session=True)
        if child is not None and child.poll() not in (None, 0):
            atomic(Path(args.root) / 'evaluation-failure.json', dict(exit_code=child.returncode, observed_at=time.time()))
        states = [json.loads(p.read_text()) if p.exists() else {} for p in (Path(e['run']) / 'process.json' for e in arms)]
        if reported and all(s.get('status') in ('complete', 'failed') for s in states) and (child is None or child.poll() is not None):
            report(args.root)
            return
        time.sleep(15)



def status_or_stop(args):
    for entry in entries(args.root):
        state_path = Path(entry['run']) / 'process.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status':'queued'}
        print(entry['name'], json.dumps(state))
        if args.action == 'stop' and state.get('status') == 'running':
            pid = state['pid']
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                if b'mtg_ml.rl.train' in command and entry['run'].encode() in command:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    raise RuntimeError(f'PID {pid} no longer belongs to this run')
            except FileNotFoundError:
                pass


def validate_scenario(spec):
    from mtg_ml.difftest import run_lockstep
    failure = run_lockstep(spec, full_every=17, fork_every=97)
    return None if failure is None else failure.to_json()


def validate_engines(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from mtg_ml.trace import Scenario
    specs = [Scenario(seed=71000+i, agents=('bot', 'bot'), matchup=name,
                      match_game=game, starting_player=start)
             for i, (name, game, start) in enumerate((n,g,s) for n in MATCHUPS for g in (1,2) for s in (0,1))]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn')) as pool:
        failures = [f for f in pool.map(validate_scenario, specs) if f is not None]
    print(json.dumps({'games':len(specs), 'failures':failures}), flush=True)
    if failures:
        raise SystemExit(1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('validate')
    p = sub.add_parser('prepare')
    for name in ('root', 'code', 'ladder'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('panel')
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--h2h-games', type=int, default=400)
    for name in ('watch', 'report', 'status', 'stop'):
        p = sub.add_parser(name)
        p.add_argument('--root', required=True)
    args = ap.parse_args()
    {'validate': validate_engines, 'prepare': prepare, 'run': run_arm, 'panel': panel, 'watch': watch, 'report': lambda a: report(a.root), 'status': status_or_stop, 'stop': status_or_stop}[args.action](args)


if __name__ == '__main__':
    main()
