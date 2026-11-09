import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.execution.config_revision import revise_config
from application.execution.run_plan import load_plan
from core.io import hash_json, read_json, sha256, write_json
from core.processes import identity
from runtime.agents.openhands import validate_profile
from tests.test_capture_budget_revision import make_run


def test_memory_revision_keeps_samples_and_audits_identity(cfg):
    run = make_run(cfg)
    before = read_json(run / 'resolved-config.json')
    stable = {name: sha256(run / name) for name in ['selection.json','tasks.json','rollout/work-items.json']}
    kwargs = dict(revision_id='64g', updates={'sandbox.memory_mb': 65536}, reason='User requested all task containers at 64 GiB')
    receipt = revise_config(cfg['project_root'], run.name, **kwargs)
    current = load_plan(cfg['project_root'], run.name)['config']
    assert current['sandbox']['memory_mb'] == 65536
    assert all(sha256(run / k) == v for k,v in stable.items())
    assert read_json(run / 'config-revisions/64g/before/resolved-config.json') == before
    assert revise_config(cfg['project_root'], run.name, **kwargs) == receipt


@pytest.mark.parametrize('active', [True, False])
def test_memory_revision_rejects_active_or_unclean_execution(cfg, active):
    run = make_run(cfg)
    attempt = run / 'execution/attempts/001'
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    if active:
        write_json(attempt / 'process.json', identity(os.getpid()))
    else:
        write_json(attempt / 'exit.json', {'exit_code': -15})
        write_json(attempt / 'termination.json', {'cleanup_complete': False})
    with pytest.raises(RuntimeError):
        revise_config(cfg['project_root'], run.name, revision_id='64g', updates={'sandbox.memory_mb': 65536}, reason='test')


def test_memory_revision_rejects_unknown_field_without_snapshot_change(cfg):
    run = make_run(cfg)
    digest = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError):
        revise_config(cfg['project_root'], run.name, revision_id='64g',
                      updates={'sandbox.other_run_task': 65536}, reason='test')
    assert sha256(run / 'resolved-config.json') == digest


def test_profile_memory_cli_and_invalid_values(cfg, monkeypatch):
    from core import cli
    monkeypatch.setattr(cli, 'resolve', lambda *args: cfg)
    args = ['skillsbench','profile','memory-cli','--bridge-advertise','127.0.0.1',
        '--memory','task-a:65536','--memory','task-b:32768']
    cli.main(args)
    profile = read_json(Path(cfg['project_root']) / 'runtime/agent-profiles/memory-cli/manifest.json')
    assert profile['memory_overrides'] == {'task-a':65536,'task-b':32768}
    with pytest.raises(SystemExit):
        cli.main(args + ['--memory','task-c:0'])
    with pytest.raises(ValueError):
        validate_profile({**profile,'memory_overrides':{'task-a':-1}})


@pytest.mark.parametrize('cfs_supported', [True, False])
def test_memory_override_reaches_compose_independent_of_cpu_support(tmp_path, cfs_supported):
    from runtime.agents.skillsbench import OwnedPlanes
    env = SimpleNamespace(stop=lambda **kw:None, _env_vars=SimpleNamespace(memory='8192M', cpus=1))
    key = hash_json({'benchmark':'skillsbench','revision':'v1','instance_id':'task'})
    cfg = {'project_root':str(tmp_path), 'skillsbench_daemon_capabilities':{'cpu_quota_supported':cfs_supported},
        'execution_context':{'revision':'v1','profile':{'memory_overrides':{'task':65536},
            'cpu_quota':'strict' if cfs_supported else 'unenforced_with_receipt'}},
        'selection':{'tasks':{key:{'instance':{'metadata':{'sandbox':{'cpus':1,'memory_mb':8192}}}}}}}
    planes = OwnedPlanes(SimpleNamespace(create_environment=lambda:env),cfg,'run',
        SimpleNamespace(instance_id='task',sample_key='s'),'attempt',{},tmp_path)
    planes.create_environment()
    assert env._env_vars.memory == '65536M'
    assert env._env_vars.cpus == (1 if cfs_supported else 0)
    receipt = read_json(tmp_path/'resource-adaptation.json')
    assert receipt['requested_memory_mb'] == 8192 and receipt['applied_memory_mb'] == 65536


def test_root_sandbox_memory_reaches_compose(tmp_path):
    from runtime.agents.skillsbench import OwnedPlanes
    env = SimpleNamespace(stop=lambda **kw: None, _env_vars=SimpleNamespace(memory='8192M', cpus=1))
    key = hash_json({'benchmark': 'skillsbench', 'revision': 'v1', 'instance_id': 'task'})
    cfg = {
        'project_root': str(tmp_path),
        'sandbox': {'memory_mb': 16},
        'skillsbench_daemon_capabilities': {'cpu_quota_supported': True},
        'execution_context': {'revision': 'v1', 'profile': {'memory_overrides': {}, 'cpu_quota': 'strict'}},
        'selection': {'tasks': {key: {'instance': {'metadata': {'sandbox': {'cpus': 1, 'memory_mb': 8192}}}}}},
    }
    planes = OwnedPlanes(SimpleNamespace(create_environment=lambda: env), cfg, 'run',
        SimpleNamespace(instance_id='task', sample_key='s'), 'attempt', {}, tmp_path)
    planes.create_environment()
    assert env._env_vars.memory == '16M'
    receipt = read_json(tmp_path / 'resource-adaptation.json')
    assert receipt['requested_memory_mb'] == 8192 and receipt['applied_memory_mb'] == 16
