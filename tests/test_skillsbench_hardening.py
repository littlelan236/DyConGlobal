import asyncio
from types import SimpleNamespace

import pytest

from runtime.agents.openhands import OpenHandsProfile


def test_openhands_profile_never_allows_root_as_agent():
    arguments = {'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1'}
    assert OpenHandsProfile(**arguments).sandbox_user == 'agent'
    with pytest.raises(ValueError, match='sandbox_user'):
        OpenHandsProfile(**arguments, sandbox_user='root')


def test_official_hardening_keeps_the_nonroot_user_and_exports_afterward(tmp_path):
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import OwnedPlanes
    calls = []

    class Inner:
        async def harden_before_verify(self, env, task, user, *, workspace):
            calls.append(('official_hardening', user, workspace))
    class Env:
        async def exec(self, command, **kwargs):
            calls.append(('tool_preinstall', command))
            return SimpleNamespace(return_code=0, stdout='sha256  uv 0.9.7', stderr='')

    sample = SimpleNamespace(instance_id='task', sample_key='sample', task_key='key')
    planes = OwnedPlanes(Inner(), {'project_root': str(tmp_path)}, 'run', sample,
                         'attempt', {}, tmp_path)
    async def export():
        calls.append(('export', planes.workspace))
    planes._export_workspace = export
    asyncio.run(planes.harden_before_verify(Env(), object(), 'agent', workspace='/root'))
    assert [row[0] for row in calls] == ['official_hardening', 'tool_preinstall', 'export']
    assert calls[0] == ('official_hardening', 'agent', '/root')
    assert calls[-1] == ('export', '/root')


def test_root_hardening_is_rejected_before_destructive_upstream_call(tmp_path):
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import OwnedPlanes
    class Inner:
        async def harden_before_verify(self, *args, **kwargs):
            pytest.fail('Root pkill must never be delegated')
    sample = SimpleNamespace(instance_id='task', sample_key='sample')
    planes = OwnedPlanes(Inner(), {'project_root': str(tmp_path)}, 'run', sample,
                         'attempt', {}, tmp_path)
    with pytest.raises(ValueError, match='terminate the sandbox'):
        asyncio.run(planes.harden_before_verify(object(), object(), 'root', workspace='/root'))


def test_official_normalized_partial_reward_is_valid_but_not_resolved():
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import _map_evaluation
    status,resolved,policy=_map_evaluation({}, {'completed':True,'rewards':{'reward':0.75}},'complete')
    assert status=='AgentCompleted' and resolved is False
    assert policy=='normalized_all_dimensions_equal_one_v1'
    assert _map_evaluation({}, {'completed':True,'rewards':{'reward':1.25}},'complete')[1] is None


def test_agent_protocol_error_is_not_mislabeled_as_verifier_failure():
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import _map_evaluation
    assert _map_evaluation({'error':'ACP internal error'}, {'completed':False},'complete')[0]=='OpenHandsError'
    assert _map_evaluation({'error':'timeout','error_category':'agent_timeout'},
                           {'completed':False},'complete')[0]=='AgentTimeout'


def test_verifier_uvx_and_env_work_without_a_successful_installer(tmp_path):
    import shlex
    import subprocess

    from runtime.agents.verifier_tools import verifier_tool_command

    wheels, published, local, trusted = [tmp_path / n for n in ['wheels', 'published', 'local', 'trusted']]
    for p in [wheels, published, local, trusted]:
        p.mkdir()
    uv = wheels / 'uv'
    uv.write_text('#!/bin/sh\nprintf "PINNED_UV:%s\\n" "$*"\n')
    uv.chmod(0o755)
    curl = published / 'curl'
    curl.write_text('#!/bin/sh\nexit 35\n')
    curl.chmod(0o755)
    before = uv.read_bytes(), curl.read_bytes()
    command = verifier_tool_command()
    for source, target in [('/opt/dycon-wheels', wheels), ('/opt/dycon-skillsbench-tools', published),
                           ('/root/.local/bin', local), ('/usr/local/bin', trusted)]:
        command = command.replace(source, str(target))
    command += '\n' + f'. {shlex.quote(str(local / "env"))}\nuvx --with pytest==8.4.1 pytest\n'
    result = subprocess.run(['/bin/sh', '-c', command], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'PINNED_UV:tool run --with pytest==8.4.1 pytest' in result.stdout
    assert (uv.read_bytes(), curl.read_bytes()) == before
    assert (trusted / 'curl').read_bytes() == curl.read_bytes()
