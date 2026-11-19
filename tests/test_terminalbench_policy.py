"""Terminal policy contracts and real Linux process/service lifecycle checks."""
import asyncio
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.request import urlopen

import pytest

from core.config import Config, resolve
from core.io import read_json
from runtime.agents import dependencies, model_socket
from runtime.backends.harbor.adapter import worker_request
from runtime.backends.harbor.policy import TERMINAL_POLICY, configured_policy, execution_policy


def terminal_cfg(cfg):
    result = copy.deepcopy(cfg)
    result['benchmark']['backend'] = 'terminalbench_2_1'
    result['agent']['backend'] = 'openhands'
    result['harbor'] = {'runtime_ref': 'runtime/pinned', 'image_store': 'terminal', 'cpu_quota': 'strict',
                        'terminal_policy': 'terminalbench-live-v1'}
    result['execution_context'] = {'profile': {'dependency_snapshot': '/offline', 'sandbox_user': 'agent'},
                                   'terminal_policy': copy.deepcopy(TERMINAL_POLICY)}
    return result


def test_policy_is_explicit_versioned_and_backend_scoped(cfg):
    assert configured_policy(cfg) is None
    current = terminal_cfg(cfg)
    assert configured_policy(current) == TERMINAL_POLICY
    assert execution_policy(current) == TERMINAL_POLICY
    current['execution_context']['terminal_policy']['uid'] = 123
    with pytest.raises(ValueError, match='frozen'):
        execution_policy(current)
    current['benchmark']['backend'] = 'swebench_pro_v2'
    with pytest.raises(ValueError, match='requires'):
        configured_policy(current)


def test_optional_policy_does_not_change_legacy_serialized_config(cfg, tmp_path):
    import yaml
    raw = {k: copy.deepcopy(v) for k, v in cfg.items() if k in Config.model_fields}
    raw['benchmark']['backend'] = 'terminalbench_2_1'
    raw['agent']['backend'] = 'openhands'
    raw['harbor'] = {'runtime_ref': 'pinned', 'image_store': 'pinned'}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    legacy = resolve(path)
    assert 'terminal_policy' not in legacy['harbor']
    raw['harbor']['terminal_policy'] = 'terminalbench-live-v1'
    path.write_text(yaml.safe_dump(raw))
    current = resolve(path)
    assert current['harbor']['terminal_policy'] == 'terminalbench-live-v1'
    assert current['config_hash'] != legacy['config_hash']
    raw['benchmark']['backend'] = 'swebench_pro_v2'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='terminal_policy requires'):
        resolve(path)


def test_full_official_budget_and_legacy_budget_are_independent(cfg, tmp_path):
    from tests.test_harbor_contract import harbor_task
    current = terminal_cfg(cfg)
    factory = SimpleNamespace(cpu_supported=True, owner_id='owner', images={})
    binding = {'sample_key': 'sample', 'attempt_id': 'attempt'}
    task = harbor_task()
    task['instance']['metadata']['agent']['timeout_sec'] = 12000
    result = worker_request(current, 'run', factory, task, binding, tmp_path)
    assert result['driver']['agent_seconds'] == 12000
    assert result['driver']['phase_budgets'] == {'environment': 600, 'setup': 600, 'agent': 12000,
                                                'verifier': 600, 'cleanup': 120}
    assert sum(result['driver']['phase_budgets'].values()) == 13920
    assert result['environment']['terminal_policy'] == TERMINAL_POLICY
    assert result['environment']['resources']['cpu_deviation'] is False
    assert not result['disable_verifier']
    del current['execution_context']['terminal_policy']
    legacy = worker_request(current, 'run', factory, task, binding, tmp_path)
    assert legacy['driver']['agent_seconds'] == 11940
    assert legacy['environment']['terminal_policy'] is None
    assert legacy['driver']['phase_budgets'] is None


def test_shared_root_probe_requires_frozen_policy(cfg, tmp_path):
    env = SimpleNamespace(exec=AsyncMock())
    cfg['execution_context'] = {'profile': {'sandbox_user': 'agent'}}
    with pytest.raises(ValueError, match='frozen Terminal'):
        asyncio.run(dependencies.probe_runtime(env, cfg, tmp_path, {}, '/app',
                                              expected_user='root', expected_home='/root'))
    env.exec.assert_not_called()


def test_probe_uses_same_root_home_and_official_workspace(cfg, tmp_path):
    current = terminal_cfg(cfg)
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=0, stdout=json.dumps({'verified': True, 'uid': 0, 'home': '/root'}), stderr='')))
    result = asyncio.run(dependencies.probe_runtime(env, current, tmp_path, {}, '/official/work',
                                                   expected_user='root', expected_home='/root'))
    call = env.exec.call_args
    assert call.kwargs['user'] == 'root'
    assert 'HOME=/root' in call.args[0]
    assert 'cd /root' in call.args[0]
    assert call.args[0].endswith('/official/work ' + str(current['timeouts']['model_seconds']) + ' root /root')
    assert result['verified']


def test_pid_identity_mismatch_never_signals_an_unrelated_process():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        identity = model_socket.process_identity(process.pid)
        identity['start_ticks'] = str(int(identity['start_ticks']) + 1)
        assert not model_socket.signal_owned(identity, signal.SIGTERM)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def wait_until(check, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(.05)
    raise AssertionError('Timed out waiting for real process fixture')


@pytest.mark.parametrize('end_mode', ['transport_exit', 'explicit_cleanup', 'agent_timeout'])
def test_owned_runtime_cleanup_preserves_http_service_until_verifier(tmp_path, end_mode):
    receipt = tmp_path / 'processes.json'
    service_info = tmp_path / 'service.json'
    ready = tmp_path / 'ready'
    service = tmp_path / 'service.py'
    service.write_text('''import http.server,json,os,sys
from pathlib import Path
server=http.server.HTTPServer(('127.0.0.1',0), http.server.SimpleHTTPRequestHandler)
Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'port':server.server_port}))
server.serve_forever()
''')
    acp = tmp_path / 'acp.py'
    acp.write_text('''import subprocess,sys,time
from pathlib import Path
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]], stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
Path(sys.argv[3]).write_text('ready')
time.sleep(0.5 if sys.argv[4]=='transport_exit' else 30)
''')
    parent = subprocess.Popen([sys.executable, '-c', '''import os,sys,time
from runtime.agents.model_socket import supervise_acp
relay=os.fork()
if relay==0:
    time.sleep(60)
    os._exit(0)
raise SystemExit(supervise_acp([sys.executable,*sys.argv[2:]],relay,sys.argv[1]))
''', str(receipt), str(acp), str(service), str(service_info), str(ready), end_mode])
    info = None
    try:
        wait_until(lambda: ready.exists())
        info = wait_until(lambda: json.loads(service_info.read_text()))
        identities = wait_until(lambda: (r if (r := read_json(receipt)).get('acp') else None))
        url = f"http://127.0.0.1:{info['port']}/"
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
        if end_mode == 'transport_exit':
            assert parent.wait(timeout=10) == 0
        elif end_mode == 'agent_timeout':
            parent.terminate()
            parent.wait(timeout=10)
        result = model_socket.stop_owned(receipt)
        assert result['verified']
        parent.wait(timeout=10)
        for key in ('supervisor', 'relay', 'acp'):
            assert model_socket.process_identity(identities[key]['pid']) != identities[key]
        # This HTTP request stands in for the live-state verifier after ACP exit.
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
        assert read_json(receipt)['children_stopped']
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if info:
            try:
                os.kill(info['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_acp_prompt_timeout_is_total_even_when_updates_keep_arriving(tmp_path):
    from runtime.agents.openhands_acp import run_session
    source = tmp_path / 'acp.py'
    source.write_text('''import json,sys,time
for line in sys.stdin:
    msg=json.loads(line)
    if msg['method']=='session/prompt':
        while True:
            print(json.dumps({'method':'session/update','params':{'sessionId':'s','update':{}}}),flush=True)
            time.sleep(.03)
    else:
        print(json.dumps({'id':msg['id'],'result':{'sessionId':'s'}}),flush=True)
''')
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(run_session(sys.executable, [str(source)], os.environ.copy(), None, 'task',
                                seconds=.25, session_cwd='/official/work'))
    assert time.monotonic() - start < 4


@pytest.mark.parametrize('long_checks', [True, False])
def test_new_context_freezes_terminal_policy_and_shared_condenser(cfg, tmp_path, monkeypatch, long_checks):
    from catalog.harbor import REVISIONS, execution_context
    from core.io import sha256, write_json
    from runtime.agents.condenser import POLICY
    from runtime.agents.openhands import OpenHandsDriverProfile
    from runtime.backends.harbor import dependencies as worker_dependencies
    from runtime.images import harbor as images
    current = terminal_cfg(cfg)
    current['execution']['preflight'] = {'long_checks': long_checks}
    current['project_root'] = str(tmp_path)
    current['benchmark']['snapshot_ref'] = 'snapshot.json'
    current['agent'].update(profile_ref='profile.json', condenser_max_size=12, condenser_keep_first=1)
    current['harbor']['runtime_ref'] = 'runtime.json'
    publication = tmp_path / 'dependencies'
    write_json(publication / 'manifest.json', {'source_adaptations': [{'policy': POLICY}]})
    profile = OpenHandsDriverProfile(bridge_listen='127.0.0.1', bridge_advertise='127.0.0.1',
        dependency_snapshot=str(publication), dependency_snapshot_sha256=sha256(publication / 'manifest.json'))
    write_json(tmp_path / 'profile.json', profile.model_dump())
    write_json(tmp_path / 'runtime.json', {})
    write_json(tmp_path / 'snapshot.json', {'benchmark': 'terminalbench_2_1',
        'revision': REVISIONS['terminalbench_2_1'], 'tasks': {}})
    depths = []
    monkeypatch.setattr(dependencies, 'verify_snapshot', lambda path, **kwargs: depths.append(kwargs['deep']) or {})
    monkeypatch.setattr(worker_dependencies, 'verify_runtime', lambda path, **kwargs: {'worker_python': '/isolated/python',
                                                                         'worker_protocol': 1})
    monkeypatch.setattr(images, 'references', lambda *args: {})
    frozen = execution_context(current, {})
    assert depths == [long_checks]
    assert frozen['terminal_policy'] == TERMINAL_POLICY
    assert frozen['profile']['sandbox_user'] == 'agent'
    assert frozen['condenser'] == {'max_size': 12, 'keep_first': 1, 'runtime_configurable': True}
    del current['harbor']['terminal_policy']
    legacy = execution_context(current, {})
    assert 'terminal_policy' not in legacy and 'condenser' not in legacy
    assert frozen['context_sha256'] != legacy['context_sha256']


def test_terminal_model_health_probe_authenticates_without_generation(cfg, tmp_path, monkeypatch):
    from core.deadline import Deadline
    from rollout.adapters.openhands_bridge import OpenHandsBridge
    profile = {'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1', 'request_roles': 'unqualified'}
    with OpenHandsBridge(cfg, {'attempt_id': 'probe'}, profile, tmp_path / 'bridge', Deadline(10)) as bridge:
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('LLM_API_KEY', bridge.secret)
        monkeypatch.setenv('LLM_MODEL', 'openai/test')
        monkeypatch.setenv('DYCON_PROBE_CHANNEL', '1')
        monkeypatch.setenv('DYCON_PROCESS_RECEIPT', str(tmp_path / 'processes.json'))
        monkeypatch.setattr(model_socket, 'supervise_acp', lambda *args: 0)
        with pytest.raises(SystemExit) as exit_info:
            model_socket._configure_and_launch(bridge.server.server_port, 0)
        assert exit_info.value.code == 0
        assert read_json(tmp_path / 'dycon-model-channel-probe.json') == {
            'authenticated': True, 'unauthenticated': True, 'verified': True}
        assert bridge.records == []


def test_relay_is_reaped_if_managed_launch_configuration_fails(tmp_path, monkeypatch):
    child = []
    fork = os.fork

    def tracked_fork():
        pid = fork()
        if pid:
            child.append(pid)
        return pid

    def fail(*args):
        raise ValueError('injected initialization failure')

    monkeypatch.setenv('DYCON_PROCESS_RECEIPT', str(tmp_path / 'processes.json'))
    monkeypatch.setattr(os, 'fork', tracked_fork)
    monkeypatch.setattr(model_socket, '_configure_and_launch', fail)
    with pytest.raises(ValueError, match='initialization failure'):
        model_socket.launch_acp()
    assert len(child) == 1 and model_socket.process_identity(child[0]) is None
    with pytest.raises(ChildProcessError):
        os.waitpid(child[0], os.WNOHANG)


def test_long_official_control_keeps_full_budget_and_valid_reward(cfg, tmp_path, monkeypatch):
    from runtime.backends.harbor import operations
    from tests.test_harbor_contract import harbor_task
    current = terminal_cfg(cfg)
    current['timeouts']['container_lifetime_seconds'] = 120
    task = harbor_task()
    task['instance']['metadata']['agent']['timeout_sec'] = 12000
    current['selection'] = {'tasks': {'task-key': task}}
    execute = Mock(return_value={
        'result': {'verifier_result': {'rewards': {'reward': 1}}, 'reward_valid': True}})
    factory = SimpleNamespace(cpu_supported=True, owner_id='owner', images={}, lock_images=lambda tasks: None,
                              execute_worker=execute)
    adapter = SimpleNamespace(binding=SimpleNamespace(backend_family='harbor'),
        open_run=lambda *args: SimpleNamespace(factory=factory), close_run=lambda context: None)
    monkeypatch.setattr(operations, 'load_plan', lambda *args: {'config': current, 'run': tmp_path})
    monkeypatch.setattr(operations, 'adapter_for', lambda cfg: adapter)
    result = operations.controls(tmp_path, 'run', 'long-oracle', 'oracle')
    assert result['tasks']['task-key']['resolved'] is True
    assert result['tasks']['task-key']['model_requests'] == 0
    assert execute.call_args.kwargs['seconds'] == 13920
    assert read_json(tmp_path / 'validation/harbor-controls/long-oracle/summary.json')['agent'] == 'oracle'
