from core.io import hash_json
from runtime.agents.skillsbench import environment_contract
from runtime.agents import verifier_dependencies


def test_resource_implementation_invalidates_only_resource_policy_contract(monkeypatch, tmp_path):
    source = tmp_path / 'verifier_dependencies.py'
    source.write_text('first implementation')
    monkeypatch.setattr(verifier_dependencies, '__file__', str(source))
    cfg = {'run_id': 'test', 'config_hash': 'config', 'model': {'fingerprint': 'model'},
           'execution_context': {'context_sha256': 'context', 'profile': {}}}
    task = {'key': 'task', 'instance': {'package_sha256': 'package', 'images': []}}
    legacy_before = environment_contract(cfg, task)
    cfg['execution_context']['profile']['verifier_resource_policy'] = 'skillsbench-verifier-resources-v1'
    resource_before = environment_contract(cfg, task)
    source.write_text('changed implementation')
    resource_after = environment_contract(cfg, task)
    assert hash_json(resource_before) != hash_json(resource_after)
    assert set(resource_after['code']) - set(legacy_before['code']) == {'verifier_dependencies.py'}
    cfg['execution_context']['profile'].pop('verifier_resource_policy')
    assert environment_contract(cfg, task) == legacy_before
