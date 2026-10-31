import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('mode,visible,valid', [('no-skill', '', True), ('no-skill', None, True),
    ('no-skill', 'a' * 64 + ' /skills/leak/SKILL.md', False)])
def test_no_skill_visibility_uses_official_deploy_and_actual_user(tmp_path, mode, visible, valid):
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import OwnedPlanes
    from core.io import read_json
    called = []

    async def deploy(*args, **kwargs):
        called.append('official')

    async def execute(command, **kwargs):
        assert called == ['official']
        assert kwargs['user'] == 'agent' and '/skills' in command
        return SimpleNamespace(return_code=0, stdout=visible)

    cfg = {'project_root': str(tmp_path), 'execution_context': {'profile': {'skill_mode': mode}}}
    sample = SimpleNamespace(instance_id='task', sample_key='s')
    planes = OwnedPlanes(SimpleNamespace(deploy_skills=deploy), cfg, 'run', sample, 'a', {}, tmp_path)
    call = planes.deploy_skills(SimpleNamespace(exec=execute), tmp_path, None, None, 'agent', '/app')
    if valid:
        asyncio.run(call)
    else:
        with pytest.raises(ValueError, match='skill visibility'):
            asyncio.run(call)
    assert read_json(tmp_path / 'skills-visibility.json')['verified'] is valid


def test_cli_reports_oracle_failure_as_nonzero(cfg, monkeypatch):
    pytest.importorskip('benchflow')
    from core import cli
    from runtime.agents import skillsbench
    monkeypatch.setattr(cli, 'resolve', lambda *args: cfg)
    monkeypatch.setattr(skillsbench, 'run_oracles', lambda *args: {'oracle_count': 1, 'valid_oracles': 0})
    with pytest.raises(SystemExit) as result:
        cli.main(['skillsbench', 'oracle', 'run'])
    assert result.value.code == 2
