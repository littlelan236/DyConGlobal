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


def test_verifier_wheel_links_expose_requires_python_without_mutating_snapshot(tmp_path):
    import subprocess
    import zipfile
    from html.parser import HTMLParser
    from runtime.agents.verifier_tools import verifier_wheel_links_command

    source = tmp_path / 'snapshot'
    (source / 'wheels').mkdir(parents=True)
    expected = {}
    for version, requirement in [('9.0.2', '>=3.10'), ('8.3.5', '>=3.8')]:
        wheel = source / 'wheels' / ('pytest-' + version + '-py3-none-any.whl')
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('pytest/_vendor/other-1.dist-info/METADATA', 'Name: other\n')
            archive.writestr('pytest-' + version + '.dist-info/METADATA',
                'Metadata-Version: 2.1\nName: pytest\nVersion: ' + version + '\nRequires-Python: ' + requirement + '\n\n')
        expected[wheel.name] = requirement
    before = {p.name: p.read_bytes() for p in (source / 'wheels').iterdir()}
    target = tmp_path / 'links'
    command = verifier_wheel_links_command().replace('/opt/dycon-wheels', str(source)).replace('/opt/dycon-verifier-links', str(target))
    result = subprocess.run(['/bin/sh', '-c', command], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    links = []
    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == 'a': links.append(dict(attrs))
    Parser().feed((target / 'index.html').read_text())
    assert {link['href'].rsplit('/', 1)[-1]: link['data-requires-python'] for link in links} == expected
    assert {p.name: p.read_bytes() for p in (source / 'wheels').iterdir()} == before


@pytest.mark.parametrize('proxy', [None, 'http://proxy.test:7890'])
def test_public_verifier_index_is_injected_with_or_without_proxy(tmp_path, proxy):
    from core.io import read_json
    from runtime.agents.skillsbench import OwnedPlanes
    index = 'https://mirrors.aliyun.com/pypi/simple/'
    verifier = SimpleNamespace(env={'TASK_VARIABLE': 'preserved'}, timeout_sec=10)
    task = SimpleNamespace(config=SimpleNamespace(verifier=verifier))
    cfg = {'project_root': str(tmp_path), 'execution_context': {'profile': {
        'network': 'public', 'verifier_python_index': index,
        'verifier_timeout_overrides': {'task': 123}}}}
    if proxy:
        cfg['skillsbench_network_proxy'] = proxy
    class Inner:
        async def harden_before_verify(self, env, task, user, *, workspace):
            assert task.config.verifier.env['PIP_INDEX_URL'] == index
            assert task.config.verifier.env['UV_INDEX_URL'] == index
    class Env:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')
    sample = SimpleNamespace(instance_id='task', sample_key='sample', task_key='key')
    planes = OwnedPlanes(Inner(), cfg, 'run', sample, 'attempt', {}, tmp_path)
    async def export():
        pass
    planes._export_workspace = export
    asyncio.run(planes.harden_before_verify(Env(), task, 'agent', workspace='/root'))
    assert verifier.env['TASK_VARIABLE'] == 'preserved'
    assert verifier.timeout_sec == 123
    receipt = read_json(tmp_path / 'verifier-environment.json')
    assert receipt['verifier_python_index'] == index
    assert receipt['proxy_mode'] == ('configured' if proxy else 'none')
    assert receipt['configured'] is True
    assert receipt['run_id'] == 'run' and receipt['attempt_id'] == 'attempt'
    assert receipt['phase'] == 'after_hardening_and_tools'
    assert receipt['effective_variables']['UV_DEFAULT_INDEX'] == index


def test_index_policy_preserves_legacy_and_rejects_task_conflicts():
    from runtime.agents.skillsbench import verifier_environment_policy
    original = {'PIP_INDEX_URL': 'https://dedicated.test/simple/', 'TASK': 'keep'}
    assert verifier_environment_policy({'network': 'public'}, original) == (original, {})
    index = 'https://mirror.test/simple/'
    with pytest.raises(ValueError, match='dedicated index'):
        verifier_environment_policy({'network': 'public', 'verifier_python_index': index}, original)
    with pytest.raises(ValueError, match='offline policy'):
        verifier_environment_policy({'network': 'public', 'verifier_python_index': index}, {'PIP_NO_INDEX': '1'})
    effective, _ = verifier_environment_policy({'network': 'none'}, {'TASK': 'keep'})
    assert effective['TASK'] == 'keep'
    for key in ('PIP_NO_INDEX', 'UV_NO_INDEX', 'UV_OFFLINE'):
        assert effective[key] == '1'
    assert effective['UV_PYTHON_DOWNLOADS'] == 'never'
    assert effective['HTTP_PROXY'] == ''


def test_environment_receipt_does_not_export_credentials():
    from runtime.agents.skillsbench import _verifier_safe_environment
    result = _verifier_safe_environment({'HTTP_PROXY': 'http://user:password@proxy.test/',
                                        'API_KEY': 'secret', 'TASK': 'private'})
    assert result == {'HTTP_PROXY': 'http://[redacted]@proxy.test/'}


def test_simpo_keeps_legacy_explicit_cpu_index_precedence():
    from runtime.agents.skillsbench import verifier_environment_policy
    profile = {'network': 'public', 'verifier_python_index': 'https://mirror.test/simple/'}
    effective, _ = verifier_environment_policy(profile, {}, task_id='simpo-code-reproduction')
    assert effective['UV_DEFAULT_INDEX'] == ''
    assert effective['UV_INDEX_URL'] == profile['verifier_python_index']
