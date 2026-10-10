"""docs/experiments/models.json is consistent, matches models.md, and the dashboard resolves handles from it."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'docs' / 'experiments'
REGISTRY = json.loads((EXP / 'models.json').read_text())
MODELS = REGISTRY['models']

spec = importlib.util.spec_from_file_location('dashboard_tracker', ROOT / 'apps' / 'training-dashboard' / 'tracker.py')
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)

ID = re.compile(r'^\d{8}-[a-z0-9]+-fs\d+h\d+-(all|mix\d+|mu-[a-z]+-[a-z]+|dk-[a-z]+)-(scratch|ft-[\w.-]+)(-[a-z0-9]+)?-s\d+(\.x\d+)?$')
STATUS = {'planned', 'finished', 'running', 'stopped', 'archived', 'superseded', 'rejected', 'crashed'}


def test_unique_handles_and_run_ids():
    handles = [m['handle'] for m in MODELS]
    ids = [m['run_id'] for m in MODELS]
    assert len(set(handles)) == len(handles)
    assert len(set(ids)) == len(ids)


def test_run_ids_follow_the_grammar_and_the_fields():
    for m in MODELS:
        if m['init'] != 'scratch' and not m['init'].startswith('model_v3'):
            assert m['init'] in {x['handle'] for x in MODELS}, m['handle']
        if m['init'].startswith('model_v3'):
            continue  # search64: pre-ledger parent, written as its name
        assert ID.match(m['run_id']), m['run_id']
        assert m['run_id'].startswith(m['date'] + '-' + m['campaign'] + '-' + m['arch'] + '-' + m['scope'])
        assert m['run_id'].split('.x')[0].endswith('-s%d' % m['seed'])
        assert m['status'] in STATUS


def test_parents_and_roles_point_to_registered_handles():
    handles = {m['handle'] for m in MODELS}
    assert all(m['parent'] in handles for m in MODELS if m['parent'])
    for role, v in REGISTRY['roles'].items():
        assert re.match(r'^(best|play)/[a-z_]+$', role), role
        assert v['handle'] in handles, role
    for a in REGISTRY['in_flight']:
        assert a['handle'] in handles and a['vs'] in handles


def test_every_ledger_id_is_in_the_registry_or_not_a_model():
    """New ledger entries that train a model must add it to models.json (evaluation-only entries are exempt)."""
    known = {m['ledger_id'] for m in MODELS}
    exempt = {'20261009-specialist-benchmark', '20261009-r8-evals-1', '20261008-h256-attention-scaling', '20261010-r8-pilot-matrix', '20261010-interference-diag'}
    for line in (EXP / 'ledger.jsonl').read_text().splitlines():
        i = json.loads(line)['id']
        assert i in known or i in exempt, i


def test_models_md_lists_exactly_the_json_rows():
    md = (EXP / 'models.md').read_text()
    for m in MODELS:
        assert md.count('| `%s` | `%s` |' % (m['handle'], m['run_id'])) == 1, m['handle']
    for m in MODELS:
        for name in m['legacy']:
            assert name in md, name


def test_dashboard_resolves_handles_from_legacy_names(tmp_path):
    handles = tracker.load_handles(EXP / 'models.json')
    assert tracker.handle_for(handles, 'r8-jund-blue-ft', 'r8-20261009') == 'r8-jund-blue'
    assert tracker.handle_for(handles, 'r8-fs8-belief', 'r8b-20261009') == 'r8-belief'
    assert tracker.handle_for(handles, 'r8-fs8-belief', 'r8-20261009') == 'r8-belief-x1'
    assert tracker.handle_for(handles, 'r8-jund-mirror-ft', 'r8e-20261009') == 'r8-jund-mirror-x1'
    assert tracker.handle_for(handles, '20261008-r7-fs7-h256-lr075') == 'r7-lr075'
    assert tracker.handle_for(handles, 'unknown-run', 'x') is None
    assert tracker.handle_for({}, 'r7-lr075') is None
    assert tracker.load_handles(tmp_path / 'missing.json') == {}
    ledger = tmp_path / 'ledger.jsonl'
    ledger.write_text(json.dumps({'id': '20261009-r8-jund-pilot', 'results': {}}) + '\n' + json.dumps({'id': 'zzz'}) + '\n')
    rows = tracker.load_ledger(ledger, handles)
    assert [r['handle'] for r in rows] == ['r8-jund-pilot', None]
