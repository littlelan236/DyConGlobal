"""Opt-in checks against the real pinned Harbor worker, without Docker/images."""
import os
import subprocess

import pytest


def test_real_harbor_setup_launch_identity_and_legacy_compatibility(tmp_path):
    python = os.environ.get('DYCON_HARBOR_TEST_PYTHON')
    if not python:
        pytest.skip('Set DYCON_HARBOR_TEST_PYTHON to the isolated pinned Harbor interpreter')
    source = r'''
import asyncio, copy, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import TrialConfig
from runtime.backends.harbor.environment import DyConDockerEnvironment
from runtime.backends.harbor.openhands_acp import DyConOpenHandsACP
from runtime.backends.harbor.policy import TERMINAL_POLICY

async def main():
    root = Path(sys.argv[1])
    cfg = {'benchmark': {'backend': 'terminalbench_2_1'},
           'execution_context': {'terminal_policy': copy.deepcopy(TERMINAL_POLICY)}}
    driver = {'cfg': cfg, 'ownership': {}, 'output': str(root),
              'agent_environment': {}, 'agent_seconds': 12000}
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='{}', stderr='')),
        terminal_identity=AsyncMock(return_value={'cwd': '/official/work', 'path': '/task/python/bin:/usr/bin:/bin',
                                                 'uid': 0, 'home': '/root'}),
        upload_file=AsyncMock(),
        acp_command=AsyncMock(return_value=('docker', ['exec'], {})))
    agent = DyConOpenHandsACP.__new__(DyConOpenHandsACP)
    agent.dycon = driver
    with patch('runtime.backends.harbor.openhands_acp.install_runtime', new_callable=AsyncMock), \
         patch('runtime.backends.harbor.openhands_acp.probe_runtime', new_callable=AsyncMock) as probe, \
         patch('runtime.backends.harbor.openhands_acp.run_session', new_callable=AsyncMock) as session:
        await agent.setup(env)
        assert not any('chown' in c.args[0] or 'useradd' in c.args[0] for c in env.exec.call_args_list)
        assert probe.call_args.kwargs == {'expected_user': 'root', 'expected_home': '/root',
                                         'task_path': '/task/python/bin:/usr/bin:/bin'}
        os.environ['DYCON_MODEL_CHANNEL_SECRET'] = 'test-only'
        await agent.run('official instruction', env, None)
        launched = env.acp_command.call_args.args[0]
        assert launched['HOME'] == '/root'
        assert launched['PATH'] == '/task/python/bin:/usr/bin:/bin'
        assert session.call_args.kwargs['session_cwd'] == '/official/work'
        assert session.call_args.kwargs['seconds'] == 12000
        assert any('--stop /tmp/dycon-acp-processes.json' in c.args[0] for c in env.exec.call_args_list)
        assert not any('pkill' in c.args[0] for c in env.exec.call_args_list)
        # Cleanup failure is an infrastructure error, never a normal zero reward.
        env.exec.return_value = SimpleNamespace(return_code=1, stdout='failure', stderr='')
        try:
            await agent.run('instruction', env, None)
        except ValueError as exc:
            assert 'cleanup' in str(exc)
        else:
            raise AssertionError('Unconfirmed cleanup was accepted')
    # Exercise actual environment command builder with both identities.
    docker = DyConDockerEnvironment.__new__(DyConDockerEnvironment)
    docker.dycon = {'terminal_policy': copy.deepcopy(TERMINAL_POLICY)}
    docker.container_id = AsyncMock(return_value='a'*64)
    command, args, process_env = await docker.acp_command(launched)
    assert args[args.index('--user')+1] == 'root'
    assert args[args.index('--workdir')+1] == '/official/work'
    assert args[-2:] == ['/opt/dycon-openhands/bin/python', '/opt/dycon-model-socket.py']
    docker.dycon = {}
    _, args, _ = await docker.acp_command(launched)
    assert args[args.index('--user')+1] == 'agent'
    docker.task_env_config = TaskConfig.model_validate({'version': '1.1'}).environment
    docker._docker = AsyncMock(side_effect=[json.dumps([{'Config': {'WorkingDir': '/official/work'}}]),
        json.dumps({'cwd': '/official/work', 'path': '/task/bin', 'uid': 0, 'home': '/root'})])
    identity = await docker.terminal_identity()
    assert identity['path'] == '/task/bin'
    assert 'bash' not in docker._docker.call_args.args
    # Legacy ACP user/PATH/cleanup retain their existing behavior.
    legacy = DyConOpenHandsACP.__new__(DyConOpenHandsACP)
    legacy.dycon = {**driver, 'cfg': {'benchmark': {'backend': 'terminalbench_2_1'}, 'execution_context': {}}}
    legacy.policy, legacy.cwd = None, '/app'
    env.exec.return_value = SimpleNamespace(return_code=0, stdout='', stderr='')
    env.exec.reset_mock()
    with patch('runtime.backends.harbor.openhands_acp.run_session', new_callable=AsyncMock):
        await legacy.run('instruction', env, None)
    assert any('pkill -TERM -u agent' in c.args[0] for c in env.exec.call_args_list)
    assert env.acp_command.call_args.args[0]['HOME'] == '/home/agent'
    assert env.acp_command.call_args.args[0]['PATH'].startswith('/opt/dycon-openhands/bin:')
    parsed = TrialConfig.model_validate({'task': {'path': '/test'}, 'agent': {
        'import_path': 'runtime.backends.harbor.openhands_acp:DyConOpenHandsACP',
        'override_setup_timeout_sec': TERMINAL_POLICY['setup_seconds']}})
    assert parsed.agent.override_setup_timeout_sec == 600
    print('real_harbor_worker_contracts_passed')

asyncio.run(main())
'''
    result = subprocess.run([python, '-c', source, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'real_harbor_worker_contracts_passed' in result.stdout


def test_pinned_openhands_terminal_preserves_task_path_and_background_service(tmp_path):
    python = os.environ.get('DYCON_OPENHANDS_TEST_PYTHON')
    if not python:
        pytest.skip('Set DYCON_OPENHANDS_TEST_PYTHON to an isolated pinned OpenHands interpreter')
    source = r'''
import json, os, shlex, signal, sys, time
from pathlib import Path
from urllib.request import urlopen
from openhands.tools.terminal import TerminalAction, TerminalExecutor

root = Path(sys.argv[1])
home = root / 'home'
home.mkdir()
os.environ['HOME'] = str(home)
official_path = os.environ['PATH']
service = root / 'service.py'
service.write_text("import http.server,json,os,sys\nfrom pathlib import Path\n"
    "s=http.server.HTTPServer(('127.0.0.1',0),http.server.SimpleHTTPRequestHandler)\n"
    "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'port':s.server_port}))\n"
    "s.serve_forever()\n")
executor = TerminalExecutor(working_dir=str(root), terminal_type='subprocess')
identity = root / 'identity.json'
info = None
try:
    code = "import json,os; from pathlib import Path; Path(" + repr(str(identity)) + ").write_text(json.dumps({'path':os.environ['PATH'],'cwd':os.getcwd(),'home':os.environ['HOME'],'uid':os.getuid()}))"
    action = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
    observation = executor.session.execute(TerminalAction(command=action, timeout=5))
    actual = json.loads(identity.read_text())
    assert actual == {'path':official_path,'cwd':str(root),'home':str(home),'uid':os.getuid()}, actual
    action = ' '.join(shlex.quote(x) for x in [sys.executable, str(service), str(root / 'service.json')])
    executor.session.execute(TerminalAction(command=action + ' >service.log 2>&1 &', timeout=5))
    deadline = time.monotonic() + 5
    while not (root / 'service.json').exists() and time.monotonic() < deadline:
        time.sleep(.05)
    info = json.loads((root / 'service.json').read_text())
    executor.close()
    with urlopen('http://127.0.0.1:' + str(info['port']), timeout=3) as response:
        assert response.status == 200
    print('pinned_native_terminal_identity_and_live_service_passed')
finally:
    executor.close()
    if info:
        try:
            os.kill(info['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
'''
    result = subprocess.run([python, '-c', source, str(tmp_path)], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'pinned_native_terminal_identity_and_live_service_passed' in result.stdout
