"""Harbor worker integration tests; run with the pinned Harbor interpreter."""
import asyncio
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

try:
    from harbor.environments.docker.docker import DockerEnvironment
    from runtime.backends.harbor.environment import DyConDockerEnvironment
except ModuleNotFoundError as exc:
    if exc.name == 'harbor':
        raise unittest.SkipTest('Requires the isolated pinned Harbor interpreter')
    raise


class ExecutionContract(unittest.IsolatedAsyncioTestCase):
    def environment(self, output, affinity=None):
        box = DyConDockerEnvironment.__new__(DyConDockerEnvironment)
        box._affinity_cpus = affinity
        box.dycon = {'benchmark': 'terminalbench_2_1', 'attempt_id': 'test', 'output': output}
        # Use the actual platform contract from the pinned Harbor publication.
        from harbor.environments.docker.docker_unix import UnixOps
        box._platform = UnixOps(box)
        return box

    async def test_affinity_preserves_fallback_and_explicit_interpreters(self):
        cpu = str(min(os.sched_getaffinity(0)))
        with tempfile.TemporaryDirectory() as output:
            box = self.environment(output, cpu)
            cases = [
                ('# comment\nset -euo pipefail\n[[ 1 == 1 ]]\nprintf bash-ok\n', 'bash-ok'),
                ('# comment\n#!/bin/bash\nsource /dev/null\nprintf source-ok\n', 'source-ok'),
                ('#!/bin/sh\n[ -n "${BASH_VERSION:-}" ] && exit 9\nprintf explicit-sh\n', 'explicit-sh'),
            ]
            async def execute(command, **kwargs):
                result = subprocess.run(box._platform.exec_shell_args(command), capture_output=True, text=True, timeout=5)
                return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            with patch.object(DockerEnvironment, 'exec', side_effect=execute):
                for index, (script, expected) in enumerate(cases):
                    path = Path(output) / str(index)
                    path.write_text(script); path.chmod(0o755)
                    result = await box.exec(shlex.quote(str(path)))
                    self.assertEqual(result.return_code, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    async def test_cancellation_recovers_partial_logs_and_preserves_cancellation(self):
        await self.check_interruption(asyncio.CancelledError())

    async def test_error_recovers_partial_logs_and_preserves_error(self):
        await self.check_interruption(RuntimeError('original failure'))

    async def check_interruption(self, error):
        with tempfile.TemporaryDirectory() as output:
            box = self.environment(output)
            result = SimpleNamespace(return_code=0, stdout='partial verifier log', stderr='')
            async def execute(command, **kwargs):
                if command.startswith('(/tests/test.sh)'):
                    raise error
                self.assertLessEqual(kwargs['timeout_sec'], 10)
                return result
            with patch.object(DockerEnvironment, 'exec', side_effect=execute):
                with self.assertRaises(type(error)):
                    await box.exec('(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1')
            self.assertEqual((Path(output) / 'verifier-interrupted-stdout.txt').read_text(), result.stdout)
            self.assertTrue((Path(output) / 'verifier-interrupted-processes.json').exists())
            import json
            receipt = json.loads((Path(output) / 'verifier-execution.json').read_text())
            self.assertEqual(receipt['exception_type'], type(error).__name__)
            self.assertTrue(receipt['interrupted_logs']['partial'])
            self.assertIsNone(receipt['return_code'])

    async def test_recovery_failure_does_not_mask_original(self):
        with tempfile.TemporaryDirectory() as output:
            box = self.environment(output)
            with patch.object(DockerEnvironment, 'exec', side_effect=RuntimeError('original failure')):
                with self.assertRaisesRegex(RuntimeError, 'original failure'):
                    await box.exec('(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1')
            import json
            receipt = json.loads((Path(output) / 'verifier-execution.json').read_text())
            self.assertEqual(receipt['interrupted_logs']['error'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()

