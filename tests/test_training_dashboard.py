"""The monitor must not present stale, partial or failed work as completed."""
import importlib.util
import json
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / 'apps' / 'training-dashboard'
spec = importlib.util.spec_from_file_location('dashboard_snapshot', APP / 'snapshot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def campaign(tmp_path):
    entries = []
    for arm in ('legacy', 'stacked'):
        run = tmp_path / (arm + '-seed0')
        run.mkdir()
        entries.append(dict(name=run.name, arm=arm, run=str(run), command=['python', '-m', 'mtg_ml.rl.train', '--ppo-precision', 'fp32', '--workers', '26'], warmup=5, timed=20, gpus=['GPU-test'], cpus=[0], code_ref='test', seed=0, parent={'parent_sha256': 'test', 'continuation': {'games_total': 4997120}}))
    (tmp_path / 'campaign.json').write_text(json.dumps(entries))
    return entries


def rows(n):
    return [dict(iteration=i+1, games=2048, decisions=300000, wall_s=100 if i<5 else 10, rollout_s=6, ppo_s=7, publish_s=.1, elapsed_s=(i+1)*10) for i in range(n)]


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))


def test_partial_record_does_not_change_progress(tmp_path):
    entries = campaign(tmp_path)
    metrics = Path(entries[0]['run']) / 'metrics.jsonl'
    write_jsonl(metrics, rows(6))
    with metrics.open('a') as f:
        f.write('{"iteration":7')
    result = module.snapshot(tmp_path, active={entries[0]['run']:123}, supervisor=[100], gpus=[])
    a = result['arms'][0]
    assert result['status'] == 'running'
    assert a['iterations'] == 6 and a['status'] == 'running'
    assert a['provisional']['timed_iterations'] == 1
    assert a['provisional']['decisions_per_s'] == 30000
    assert a['provisional']['games_per_hour'] == 737280
    assert a['result'] is None  # partial timing never becomes a final comparison


def test_dead_trainer_is_interrupted_and_completed_outcome_is_authoritative(tmp_path):
    entries = campaign(tmp_path)
    write_jsonl(Path(entries[0]['run']) / 'metrics.jsonl', rows(25))
    result = module.snapshot(tmp_path, active={}, supervisor=[], gpus=[])
    assert result['status'] == 'stopped'
    assert result['arms'][0]['status'] == 'interrupted'
    write_jsonl(tmp_path / 'outcomes.jsonl', [dict(name=entries[0]['name'], verdict='inconclusive', games_per_hour=123)])
    result = module.snapshot(tmp_path, active={}, supervisor=[100], gpus=[])
    assert result['arms'][0]['status'] == 'complete'
    assert result['arms'][0]['result']['games_per_hour'] == 123


def test_failures_and_skips_are_visible(tmp_path):
    entries = campaign(tmp_path)
    write_jsonl(tmp_path / 'outcomes.jsonl', [dict(name=entries[0]['name'], verdict='failed', exit_code=-9), dict(name=entries[1]['name'], verdict='skipped', reason='Learning no longer limits throughput')])
    result = module.snapshot(tmp_path, active={}, supervisor=[], gpus=[])
    assert result['status'] == 'failed'
    assert result['arms'][1]['status'] == 'skipped'
    assert 'exit code -9' in result['alerts'][0]


def test_corrupted_complete_record_is_reported_and_nan_is_strict_json(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{invalid}\n')
    with pytest.raises(ValueError, match='Invalid complete JSON record'):
        module.jsonl(path)
    assert json.dumps(module.clean({'x': float('nan')}), allow_nan=False) == '{"x": null}'


def test_ssh_failure_preserves_last_snapshot_with_visible_error(tmp_path, monkeypatch):
    import subprocess

    spec = importlib.util.spec_from_file_location('dashboard_server', APP / 'server.py')
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monitor = server.Monitor('host', '/script.py', '/campaign', tmp_path / 'cache.json', 15)
    observed = {'schema_version': 1, 'observed_at': 123, 'arms': []}
    monitor.value['snapshot'] = observed

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ssh', 20)

    monkeypatch.setattr(server.subprocess, 'run', timeout)
    monitor.poll()
    value = json.loads(monitor.state())
    assert value['snapshot'] == observed
    assert 'timed out' in value['error']
    assert value['last_attempt'] > observed['observed_at']
