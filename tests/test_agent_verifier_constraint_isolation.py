import asyncio
import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('resource_policy', [False, True])
def test_private_install_ignores_only_new_verifier_constraints(tmp_path, monkeypatch, resource_policy):
    from runtime.agents import dependencies
    monkeypatch.setattr(dependencies, 'verify_snapshot', lambda *args: {})
    names = ('UV_CONSTRAINT', 'PIP_CONSTRAINT', 'UV_BUILD_CONSTRAINT', 'PIP_BUILD_CONSTRAINT')
    inherited = {key: '/verifier/incompatible-abi-constraints.txt' for key in names}
    code = 'import json,os; print(json.dumps({k:os.environ[k] for k in ' + repr(names) + ' if k in os.environ}))'
    monkeypatch.setattr(dependencies, 'install_command', lambda: shlex.quote(sys.executable) + ' -c ' + shlex.quote(code))

    class Environment:
        async def exec(self, command, **kwargs):
            result = subprocess.run(['bash', '-c', command], env={**os.environ, **inherited},
                                    capture_output=True, text=True, timeout=10)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'manifest.json').write_text('{}')
    profile = {'dependency_snapshot': str(snapshot)}
    if resource_policy:
        profile['verifier_resource_policy'] = 'skillsbench-verifier-resources-v1'
    cfg = {'config_hash': 'config', 'execution_context': {'profile': profile, 'profile_sha256': 'profile'}}
    asyncio.run(dependencies.install_runtime(Environment(), cfg, tmp_path, {}))
    proof = json.loads((tmp_path / 'dependency-install.json').read_text())
    assert json.loads(proof['stdout']) == ({} if resource_policy else inherited)
    assert proof['return_code'] == 0
    assert inherited == {key: '/verifier/incompatible-abi-constraints.txt' for key in names}
