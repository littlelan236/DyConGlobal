import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip('benchflow')
from core.io import hash_json, read_json, write_json
from runtime.agents.skillsbench import OwnedPlanes, _oracle_attempt, _oracle_valid, run_oracles


@pytest.mark.parametrize('rewards,error,verifier,cleanup,valid', [
    ({'reward': 1.0}, None, None, True, True),
    ({'reward': True}, None, None, True, False),
    ({'reward': 0.9}, None, None, True, False),
    ({'reward': 1}, 'oracle failed', None, True, False),
    ({'reward': 1}, None, 'verifier failed', True, False),
    ({'reward': 1}, None, None, False, False),
    ({}, None, None, True, False),
])
def test_oracle_success_requires_official_full_reward_and_clean_lifecycle(rewards,error,verifier,cleanup,valid):
    assert _oracle_valid(rewards,error,verifier,cleanup) is valid


def test_failed_retry_keeps_original_receipt_and_uses_new_directory(tmp_path):
    row = {'task_key': 'k', 'task_package_sha256': 'p', 'execution_context_sha256': 'c', 'valid': False}
    path = tmp_path / 'result.json'
    write_json(path, row)
    original = path.read_bytes()
    assert _oracle_attempt(tmp_path, 'k', 'p', 'c', False) == (tmp_path, row)
    target, previous = _oracle_attempt(tmp_path, 'k', 'p', 'c', True)
    assert target == tmp_path / 'attempts/attempt-001' and previous is None
    assert path.read_bytes() == original
    write_json(target / 'result.json', {**row, 'valid': True})
    assert _oracle_attempt(tmp_path, 'k', 'p', 'c', True)[1]['valid']
    with pytest.raises(ValueError, match='another snapshot'):
        _oracle_attempt(tmp_path, 'k', 'changed', 'c', True)


def test_sidecar_has_leased_image_and_ownership_without_replacing_native_topology(tmp_path):
    env = SimpleNamespace(stop=lambda **kw: None, start=lambda **kw: None)
    cfg = {'project_root':str(tmp_path), 'execution_context':{'profile':{}},
           'skillsbench_service_images':{'main':'leased-main', 'api':'leased-api'}}
    sample = SimpleNamespace(instance_id='task',sample_key='sample')
    planes = OwnedPlanes(SimpleNamespace(create_environment=lambda:env),cfg,'run',sample,'attempt',{},tmp_path/'out')
    task = tmp_path/'task'
    (task/'environment').mkdir(parents=True)
    (task/'environment/docker-compose.yaml').write_text('services:\n  main: {}\n  api: {}\nnetworks:\n  task-net: {}\n')
    planes.task_source = task
    planes.create_environment()
    override = read_json(env._mounts_compose_path)
    assert override['services']['api']['image'] == 'leased-api'
    assert override['services']['api']['labels']['dycon.run'] == 'run'
    assert override['services']['main']['image'] == 'leased-main'
    assert override['networks']['task-net']['labels']['dycon.run'] == 'run'
    assert 'ports' not in override['services']['api']


def test_sweep_bounds_concurrency_and_collects_every_selected_task(tmp_path,monkeypatch):
    import runtime.agents.skillsbench as module
    tasks = {str(i): {'instance_id': f'task-{i}'} for i in range(5)}
    cfg = {'benchmark':{'backend':'skillsbench'},'agent':{'backend':'openhands'},
           'selection':{'tasks':tasks}, 'execution_context':{'profile':{'use_host_proxy':True}}}
    monkeypatch.setattr('application.execution.run_plan.load_plan',lambda *a:{'config':cfg,'run':tmp_path})
    monkeypatch.setattr('runtime.agents.openhands.check_version',lambda:None)
    network_events = []
    class Factory:
        def __init__(self,cfg,run): self.cfg=cfg
        def lock_images(self,tasks):
            assert tasks == []
            network_events.append('start')
            self.cfg['skillsbench_network_proxy']='shared-proxy'
        def cleanup_owned(self,run): network_events.append('stop')
    monkeypatch.setattr(module,'SkillsBenchFactory',Factory)
    active = peak = 0
    async def execute(cfg,run,key,task,base,**kw):
        nonlocal active,peak
        assert cfg['skillsbench_network_proxy']=='shared-proxy'
        active += 1
        peak = max(active,peak)
        await asyncio.sleep(0.01)
        active -= 1
        return {'task_id':task['instance_id'],'valid':key != '2'}
    monkeypatch.setattr(module,'_run_oracle_task',execute)
    summary = run_oracles({'project_root':str(tmp_path)},'run',workers=2)
    assert peak == 2
    assert network_events == ['start','stop']
    assert summary['oracle_count'] == 5 and summary['valid_oracles'] == 4
    assert read_json(tmp_path/'validation/oracle/progress.json')['completed_count'] == 5
    with pytest.raises(ValueError,match='absent'):
        run_oracles({'project_root':str(tmp_path)},'run',task_ids=['unknown'])
    with pytest.raises(ValueError,match='between'):
        run_oracles({},'run',workers=0)
    # The oracle and empty control own the same run-level proxy. Holding
    # either section's lifecycle lock must reject the other before launch.
    from core.locking import file_lock
    with file_lock(tmp_path / 'validation/.oracle-lifecycle.lock'):
        with pytest.raises(BlockingIOError):
            run_oracles({'project_root': str(tmp_path)}, 'run', empty_control=True)


def test_validation_only_snapshot_is_bound_in_execution_context(tmp_path):
    from core.backends import execution_context
    from runtime.agents.openhands import OpenHandsProfile
    profile = OpenHandsProfile(bridge_listen='127.0.0.1',bridge_advertise='127.0.0.1').model_dump()
    write_json(tmp_path/'profile.json',profile)
    task = {'benchmark':'skillsbench','revision':'a'*40,'instance_id':'task'}
    snapshot = {'schema_version':1,'benchmark':'skillsbench','revision':'a'*40,'tasks':{'k':task},'validation_only':True}
    write_json(tmp_path/'snapshot.json',snapshot)
    cfg = {'project_root':str(tmp_path),'benchmark':{'backend':'skillsbench','snapshot_ref':'snapshot.json'},
           'agent':{'backend':'openhands','profile_ref':'profile.json'},'generation':{},'parallel':{}}
    result = execution_context(cfg,{'k':task})
    assert result['validation_only'] is True
    assert result['context_sha256'] == hash_json({k:v for k,v in result.items() if k!='context_sha256'})


def test_published_tools_preserve_task_image_virtualenv_path(tmp_path, monkeypatch):
    import runtime.agents.dependencies as dependencies
    monkeypatch.setattr(dependencies, 'verify_snapshot', lambda p: None)
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (tmp_path / 'runtime/skillsbench-tools').mkdir(parents=True)
    env = SimpleNamespace(stop=lambda **kw: None)
    cfg = {'project_root': str(tmp_path), 'execution_context': {'profile': {'dependency_snapshot': str(wheels)}}}
    planes = OwnedPlanes(SimpleNamespace(create_environment=lambda: env), cfg, 'run',
        SimpleNamespace(instance_id='task', sample_key='sample'), 'attempt',
        {'environment_path': '/opt/venv/bin:/usr/bin:/bin'}, tmp_path / 'out')
    planes.create_environment()
    actual = read_json(env._mounts_compose_path)['services']['main']['environment']['PATH']
    assert actual.endswith('/opt/venv/bin:/usr/bin:/bin')
    assert actual.index('/opt/dycon-skillsbench-tools') < actual.index('/opt/venv/bin')


@pytest.mark.parametrize('oracle', [False, True])
def test_oracle_dependency_index_survives_compose_proxy_without_changing_agent(tmp_path, oracle):
    env = SimpleNamespace(stop=lambda **kw: None)
    cfg = {'project_root': str(tmp_path), 'skillsbench_oracle_validation': oracle,
        'skillsbench_network_proxy': 'http://proxy:8888',
        'execution_context': {'context_sha256': 'context', 'profile': {
            'network': 'public', 'bridge_advertise': '127.0.0.1',
            'verifier_python_index': 'https://index.example/simple/'}}}
    out = tmp_path / 'out'
    planes = OwnedPlanes(SimpleNamespace(create_environment=lambda: env), cfg, 'run',
        SimpleNamespace(instance_id='task', sample_key='sample'), 'attempt', {}, out)
    planes.create_environment()
    actual = read_json(env._mounts_compose_path)['services']['main']['environment']
    assert actual['HTTPS_PROXY'] == 'http://proxy:8888'
    if oracle:
        assert actual['UV_DEFAULT_INDEX'] == 'https://index.example/simple/'
        assert read_json(out / 'oracle-environment.json')['execution_context_sha256'] == 'context'
    else:
        assert 'UV_DEFAULT_INDEX' not in actual
        assert not (out / 'oracle-environment.json').exists()


@pytest.mark.parametrize('text,expected', [
    ('/verifier/test.sh: line 4: pytest: command not found\n', 'VerifierError'),
    ('=================== 1 failed, 2 passed in 1.00s ===================\n', 'ValidZero'),
    ('Nothing was executed\n', 'Review'),
])
def test_empty_control_requires_actual_test_execution(tmp_path, text, expected):
    from runtime.agents.skillsbench import _oracle_zero_audit
    folder = tmp_path / 'benchflow/job/rollout/verifier'
    folder.mkdir(parents=True)
    (folder / 'test-stdout.txt').write_text(text)
    assert _oracle_zero_audit(tmp_path, {}, {'reward': 0})['classification'] == expected
