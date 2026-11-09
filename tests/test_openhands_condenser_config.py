import copy
import csv
import hashlib
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.config import Agent, condenser_token_threshold, resolve
from core.io import hash_json, read_json, sha256, write_json
from runtime.agents.condenser import MEMBER, POLICY, adapt_source, derive_snapshot, verify_derivation
from runtime.agents.dependencies import build_snapshot, verify_snapshot
from runtime.agents.openhands import CLI_REVISION, condenser_context, driver_environment


@pytest.mark.parametrize('size,keep', [(80, 4), (120, 6), (4, 0)])
def test_root_config_accepts_native_settings(cfg, size, keep):
    path = Path(cfg['config_source'])
    raw = yaml.safe_load(path.read_text())
    raw['agent'].update(condenser_max_size=size, condenser_keep_first=keep)
    path.write_text(yaml.safe_dump(raw))
    actual = resolve(path)
    assert actual['agent']['condenser_max_size'] == size
    assert actual['agent']['condenser_keep_first'] == keep


@pytest.mark.parametrize('size,keep', [(0, 4), (80, -1), (10, 4), (80, 39), (True, 4), (80, 1.5)])
def test_invalid_native_setting_combinations_are_rejected(cfg, size, keep):
    with pytest.raises(ValueError):
        Agent.model_validate({**cfg['agent'], 'condenser_max_size': size, 'condenser_keep_first': keep})


def test_absent_fields_preserve_legacy_config_and_context(cfg):
    path = Path(cfg['config_source'])
    raw = yaml.safe_load(path.read_text())
    raw['agent'].pop('condenser_max_size', None)
    raw['agent'].pop('condenser_keep_first', None)
    raw['agent'].pop('condenser_max_tokens', None)
    path.write_text(yaml.safe_dump(raw))
    actual = resolve(path)
    assert 'condenser_max_size' not in actual['agent']
    assert 'condenser_keep_first' not in actual['agent']
    assert 'condenser_max_tokens' not in actual['agent']
    assert condenser_context(actual, {}) == {}
    # Only the configured field is added by the standard configuration resolver.
    changed = resolve(path, ['agent.condenser_max_size=120'])
    assert changed['agent']['condenser_max_size'] == 120
    assert 'condenser_keep_first' not in changed['agent']


def publication(root):
    source = root / 'runtime/dependency-snapshots/parent'
    wheels = source / 'wheels'
    wheels.mkdir(parents=True)
    wheel = wheels / 'openhands-1.16.0-py3-none-any.whl'
    text = '''class AgentStore:
    def _maybe_build_condenser(self, agent, *, session_id):
        import os
        condenser = agent.condenser
        threshold = os.environ.get("OPENHANDS_CONDENSER_MAX_TOKENS")
        if threshold:
            condenser = condenser.model_copy(update={"max_tokens": int(threshold)})
        return condenser
'''
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(MEMBER, text)
        archive.writestr('openhands-1.16.0.dist-info/RECORD', '')
    (source / 'requirements.txt').write_text('openhands==1.16.0 --hash=sha256:' + sha256(wheel) + '\n')
    (wheels / 'unrelated.whl').write_bytes(b'unchanged-sdk')
    manifest = {'schema_version': 1, 'cli_revision': CLI_REVISION,
                'source_adaptations': [{'policy': 'stock_condenser_for_minimal_BenchFlow_settings_v1'}],
                'files': [{'path': p.relative_to(source).as_posix(), 'size': p.stat().st_size,
                           'sha256': sha256(p)} for p in sorted(source.rglob('*')) if p.is_file()]}
    manifest['receipt_sha256'] = hash_json(manifest)
    write_json(source / 'manifest.json', manifest)
    return source


def test_derived_publication_is_immutable_hash_locked_and_exact(tmp_path):
    source = publication(tmp_path)
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result = build_snapshot({'project_root': str(tmp_path)}, 'events', from_snapshot=source)
    target = Path(result['path'])
    assert verify_snapshot(target) == result['manifest']
    assert verify_derivation(source, target)['policy'] == POLICY
    assert build_snapshot({'project_root': str(tmp_path)}, 'events', from_snapshot=source) == result
    for relative, contents in original.items():
        assert (source / relative).read_bytes() == contents
    wheel = next(target.glob('wheels/openhands-*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        rows = csv.reader(io.StringIO(archive.read('openhands-1.16.0.dist-info/RECORD').decode()))
        for name, digest, size in rows:
            if digest:
                assert len(archive.read(name)) == int(size)
                import base64
                actual = base64.urlsafe_b64encode(hashlib.sha256(archive.read(name)).digest()).decode().rstrip('=')
                assert digest == 'sha256=' + actual
    # A valid new manifest cannot make an unrelated dependency change eligible
    # for automatic qualification transfer.
    (target / 'wheels/unrelated.whl').write_bytes(b'changed-sdk')
    manifest = read_json(target / 'manifest.json')
    for row in manifest['files']:
        path = target / row['path']
        row.update(size=path.stat().st_size, sha256=sha256(path))
    manifest['receipt_sha256'] = hash_json({k: v for k, v in manifest.items() if k != 'receipt_sha256'})
    write_json(target / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='unrelated dependency'):
        verify_derivation(source, target)


def test_frozen_settings_reach_driver_with_zero_keep_first(cfg, tmp_path):
    source = publication(tmp_path)
    target = derive_snapshot({'project_root': str(tmp_path)}, 'events', source)['path']
    cfg['agent'].update(condenser_max_size=20, condenser_keep_first=0)
    profile = {'dependency_snapshot': target, 'condenser_max_tokens': 24000}
    cfg['execution_context'] = {'profile': profile, **condenser_context(cfg, profile)}
    cfg['agent'].update(condenser_max_size=80, condenser_keep_first=4)
    class Deadline:
        def remaining(self, seconds):
            return seconds
    env = driver_environment(cfg, 'http://127.0.0.1/v1', 'test', Deadline())
    assert env['OPENHANDS_CONDENSER_MAX_SIZE'] == '20'
    assert env['OPENHANDS_CONDENSER_KEEP_FIRST'] == '0'
    assert env['OPENHANDS_CONDENSER_MAX_TOKENS'] == '24000'


def test_nondefault_settings_reject_old_publication(cfg, tmp_path):
    source = publication(tmp_path)
    cfg['agent'].update(condenser_max_size=120, condenser_keep_first=4)
    with pytest.raises(ValueError, match='configurable_condenser_events_v1'):
        condenser_context(cfg, {'dependency_snapshot': str(source)})


def test_new_settings_are_frozen_and_old_run_is_unchanged(cfg):
    from application.execution.run_plan import create_run, load_plan
    from catalog.store import Catalog
    from runtime.agents.openhands import validate_profile
    root = Path(cfg['project_root'])
    source = publication(root)
    dependency = derive_snapshot(cfg, 'events', source)['path']
    profile = validate_profile({'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1',
                                'dependency_snapshot': dependency})
    write_json(root / 'profile.json', profile)
    profile_before = (root / 'profile.json').read_bytes()
    task = {'benchmark': 'skillsbench', 'revision': 'a' * 40, 'instance_id': 'condenser-task',
            'partition': 'train', 'instance': {'instance_id': 'condenser-task',
            'task_path': '/source/condenser-task', 'files': [], 'package_sha256': hash_json([]),
            'metadata': {}, 'images': []}}
    Catalog(root).register_tasks([task])
    key = hash_json({k: task[k] for k in ('benchmark', 'revision', 'instance_id')})
    write_json(root / 'snapshot.json', {'schema_version': 1, 'benchmark': 'skillsbench',
               'revision': 'a' * 40, 'tasks': {key: task}})
    cfg['benchmark'].update(backend='skillsbench', snapshot_ref='snapshot.json')
    cfg['agent'].update(backend='openhands', profile_ref='profile.json')
    cfg['agent'].pop('condenser_max_size', None)
    cfg['agent'].pop('condenser_keep_first', None)
    create_run(cfg, 'old', operation='rollout', task_ids=['condenser-task'])
    old = load_plan(root, 'old')
    before = (old['run'] / 'resolved-config.json').read_bytes()
    cfg['agent'].update(condenser_max_size=120, condenser_keep_first=6, condenser_max_tokens=None)
    create_run(cfg, 'new', operation='rollout', task_ids=['condenser-task'])
    fresh = load_plan(root, 'new')
    cfg['agent'].update(condenser_max_size=160, condenser_keep_first=8)
    expected = cfg['serving']['max_model_len'] - cfg['generation']['max_tokens'] - 1
    cfg['generation']['max_tokens'] = 1024
    assert fresh['config']['execution_context']['condenser']['max_size'] == 120
    assert fresh['config']['execution_context']['condenser']['keep_first'] == 6
    assert fresh['config']['execution_context']['condenser']['max_tokens'] == expected
    env = driver_environment(fresh['config'], 'http://localhost/v1', 'test',
                             SimpleNamespace(remaining=lambda seconds: seconds))
    assert env['OPENHANDS_CONDENSER_MAX_TOKENS'] == str(expected)
    assert load_plan(root, 'new')['config']['execution_context'] == fresh['config']['execution_context']
    assert 'condenser' not in load_plan(root, 'old')['config']['execution_context']
    assert (old['run'] / 'resolved-config.json').read_bytes() == before
    assert (root / 'profile.json').read_bytes() == profile_before


@pytest.mark.parametrize('configured,expected,source', [
    (None, 163839, 'auto_context_budget'), (24000, 24000, 'root_config'),
])
def test_root_token_threshold_overrides_profile_and_reaches_both_drivers(cfg, tmp_path, configured, expected, source):
    from runtime.agents.swebench_openhands import agent_environment
    cfg['serving']['max_model_len'] = 262144
    cfg['generation']['max_tokens'] = 98304
    cfg['agent']['condenser_max_tokens'] = configured
    # Token-only policy is independent of the event-count adaptation.
    cfg['agent'].pop('condenser_max_size', None)
    cfg['agent'].pop('condenser_keep_first', None)
    profile = {'dependency_snapshot': str(publication(tmp_path)), 'condenser_max_tokens': 12000}
    before = copy.deepcopy(profile)
    context = condenser_context(cfg, profile)
    assert context == {'condenser': {'max_tokens': expected, 'max_tokens_source': source}}
    assert profile == before
    cfg['execution_context'] = {'profile': profile, **context}
    # Runtime changes cannot override the frozen value or recompute the formula.
    cfg['agent']['condenser_max_tokens'] = 32000
    cfg['serving']['max_model_len'] = 524288
    deadline = SimpleNamespace(remaining=lambda seconds: seconds)
    for driver in (driver_environment, agent_environment):
        assert driver(cfg, 'http://localhost/v1', 'test', deadline)['OPENHANDS_CONDENSER_MAX_TOKENS'] == str(expected)


def test_legacy_context_still_uses_profile_threshold(cfg):
    cfg['execution_context'] = {'profile': {'condenser_max_tokens': 24000}}
    cfg['agent']['condenser_max_tokens'] = None
    env = driver_environment(cfg, 'http://localhost/v1', 'test',
                             SimpleNamespace(remaining=lambda seconds: seconds))
    assert env['OPENHANDS_CONDENSER_MAX_TOKENS'] == '24000'


def test_auto_threshold_requires_dependency_publication(cfg):
    cfg['agent']['condenser_max_tokens'] = None
    with pytest.raises(ValueError, match='qualified dependency snapshot'):
        condenser_context(cfg, {})


def test_root_threshold_rejects_snapshot_without_native_condenser_support(cfg, tmp_path):
    source = publication(tmp_path)
    manifest = read_json(source / 'manifest.json')
    manifest['source_adaptations'] = []
    write_json(source / 'manifest.json', manifest)
    cfg['agent']['condenser_max_tokens'] = None
    with pytest.raises(ValueError, match='qualified native-condenser dependency snapshot'):
        condenser_context(cfg, {'dependency_snapshot': str(source)})


@pytest.mark.parametrize('value', [0, -1, 1023, True, 24000.5, 'none'])
def test_root_token_threshold_rejects_invalid_values(cfg, value):
    with pytest.raises(ValueError):
        Agent.model_validate({**cfg['agent'], 'condenser_max_tokens': value})


def test_root_null_is_preserved_and_can_be_overridden(cfg):
    path = Path(cfg['config_source'])
    raw = yaml.safe_load(path.read_text())
    raw['agent']['condenser_max_tokens'] = None
    path.write_text(yaml.safe_dump(raw))
    actual = resolve(path)
    assert actual['agent']['condenser_max_tokens'] is None
    assert condenser_token_threshold(actual) == actual['serving']['max_model_len'] - actual['generation']['max_tokens'] - 1
    assert resolve(path, ['agent.condenser_max_tokens=24000'])['agent']['condenser_max_tokens'] == 24000
    raw['agent'].pop('condenser_max_tokens')
    path.write_text(yaml.safe_dump(raw))
    assert condenser_token_threshold(resolve(path)) is None
    assert resolve(path, ['agent.condenser_max_tokens=null'])['agent']['condenser_max_tokens'] is None


@pytest.mark.parametrize('max_model_len,max_tokens,valid', [(2048, 1023, True), (2048, 1024, False), (2048, 2047, False)])
def test_auto_threshold_minimum_is_checked_during_config_resolution(cfg, max_model_len, max_tokens, valid):
    path = Path(cfg['config_source'])
    raw = yaml.safe_load(path.read_text())
    raw['agent'].update(backend='openhands', condenser_max_tokens=None)
    raw['serving']['max_model_len'] = max_model_len
    raw['generation']['max_tokens'] = max_tokens
    path.write_text(yaml.safe_dump(raw))
    if valid:
        assert condenser_token_threshold(resolve(path)) == 1024
    else:
        with pytest.raises(ValueError, match='at least 1024'):
            resolve(path)


def test_native_condenser_validates_and_consumes_environment(monkeypatch):
    sdk = pytest.importorskip('openhands.sdk')
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    source = '''def configure(condenser):
        import os
        threshold = os.environ.get("OPENHANDS_CONDENSER_MAX_TOKENS")
        return condenser
'''
    namespace = {}
    exec(adapt_source(source), namespace)
    condenser = LLMSummarizingCondenser(llm=sdk.LLM(model='openai/test', api_key='test'))
    monkeypatch.setenv('OPENHANDS_CONDENSER_MAX_SIZE', '40')
    monkeypatch.setenv('OPENHANDS_CONDENSER_KEEP_FIRST', '0')
    changed = namespace['configure'](condenser)
    assert (changed.max_size, changed.keep_first) == (40, 0)
    monkeypatch.setenv('OPENHANDS_CONDENSER_KEEP_FIRST', '19')
    with pytest.raises(ValueError):
        namespace['configure'](condenser)
