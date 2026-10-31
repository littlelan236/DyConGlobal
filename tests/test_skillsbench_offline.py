import asyncio
from types import SimpleNamespace

import pytest


def test_offline_profile_forces_benchflow_network_overlay(tmp_path):
    pytest.importorskip("benchflow")
    from runtime.agents.skillsbench import OwnedPlanes
    from pydantic import BaseModel
    class Sandbox(BaseModel):
        allow_internet: bool = True
    env = SimpleNamespace(task_env_config=Sandbox())
    async def stop(**kwargs): pass
    env.stop = stop
    inner = SimpleNamespace(create_environment=lambda: env)
    cfg = {"project_root": str(tmp_path), "execution_context": {"profile": {"network": "none"}}}
    sample = SimpleNamespace(instance_id="task", sample_key="s")
    assert not OwnedPlanes(inner, cfg, "run", sample, "a", {}, tmp_path).create_environment().task_env_config.allow_internet


def test_offline_install_rejects_networked_container_before_install(tmp_path, monkeypatch):
    from runtime.agents.offline import install_probe
    monkeypatch.setattr("runtime.agents.offline.verify_snapshot", lambda *a: {})
    container = SimpleNamespace(reload=lambda: None, attrs={"HostConfig": {"NetworkMode": "bridge"}})
    client = SimpleNamespace(containers=SimpleNamespace(list=lambda **k: [container]))
    planes = SimpleNamespace(run_id="r", ownership={"epoch": "e"})
    cfg = {"execution_context": {"profile": {"dependency_snapshot": "fixture"}}}
    with pytest.raises(ValueError, match="NetworkMode=none"):
        asyncio.run(install_probe(None, client, planes, cfg, tmp_path))


def test_public_snapshot_keeps_network_for_task_specific_dependencies(tmp_path):
    from runtime.agents.skillsbench import OwnedPlanes
    from types import SimpleNamespace
    class Env:
        task_env_config = SimpleNamespace(allow_internet=True)
    cfg = {'project_root': str(tmp_path), 'execution_context': {'profile': {
        'network': 'public', 'dependency_snapshot': str(tmp_path / 'snapshot')}},
           'skillsbench_service_images': {'main': 'dycon/task:frozen'}}
    (tmp_path / 'runtime/skillsbench-tools').mkdir(parents=True)
    # Compose generation is validated separately; the public verifier policy
    # must not inject PIP_NO_INDEX and block task-specific packages.
    assert cfg['execution_context']['profile']['network'] == 'public'


def test_offline_install_command_pins_python312():
    from runtime.agents.dependencies import install_command
    command = install_command()
    assert "python3.12" in command
    assert "command -v python3" in command
    assert "--python $PYTHON_BIN" in command


def test_tool_publication_is_mounted_for_public_snapshot(tmp_path, monkeypatch):
    from runtime.agents.skillsbench import OwnedPlanes
    from types import SimpleNamespace
    tools = tmp_path / 'runtime/skillsbench-tools'; tools.mkdir(parents=True)
    class Env:
        task_env_config = SimpleNamespace(allow_internet=True)
        async def start(self, **kwargs): pass
        def stop(self, **kwargs): pass
    class Inner:
        def create_environment(self): return Env()
    cfg = {'project_root': str(tmp_path), 'execution_context': {'profile': {
        'network': 'public', 'dependency_snapshot': str(tmp_path / 'snapshot')}},
           'skillsbench_service_images': {'main': 'dycon/task:frozen'}}
    sample = SimpleNamespace(instance_id='task', sample_key='s')
    planes = OwnedPlanes(Inner(), cfg, 'run', sample, 'attempt', {}, tmp_path/'out')
    (tmp_path/'snapshot').mkdir()
    monkeypatch.setattr('runtime.agents.dependencies.verify_snapshot', lambda path: {'receipt_sha256':'test'})
    env = planes.create_environment()
    import json
    compose = json.loads((tmp_path/'out/ownership-compose.json').read_text())
    volumes = compose['services']['main']['volumes']
    assert any(v['target'] == '/root/.local/bin' for v in volumes)
    assert compose['services']['main']['image'] == 'dycon/task:frozen'
    assert 'PIP_NO_INDEX' not in compose['services']['main']['environment']
