import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.execution.harness import run
from core.io import hash_json, write_json


class WheelhouseBackendTest(unittest.TestCase):
    def check_backend(self, current_backend, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = {'rollout_ids': ['fixture'], 'tasks': {}, 'rollouts': {}}
            write_json(root/'harness/selection.json', selection)
            write_json(root/'harness/selection-receipt.json', {'selection_sha256': hash_json(selection)})
            frozen = {'benchmark': {'backend': 'swebench'}, 'environment': {'wheelhouse': 'frozen-swe'}}
            current = {'benchmark': {'backend': current_backend}, 'environment': {'wheelhouse': 'current'}}
            with patch('application.execution.harness.load_plan', return_value={'run': root, 'config': frozen}), \
                 patch('application.execution.harness.load_snapshot', return_value=frozen), \
                 patch('core.config.resolve', return_value=current), \
                 patch('runtime.wheelhouse.verify') as verify, \
                 patch('benchmark_evaluation.harness.evaluate') as evaluate:
                run(root, 'fixture-run', 'gold', gold=True)
                self.assertEqual(evaluate.call_args.args[0]['environment']['wheelhouse'], expected)
                self.assertEqual(evaluate.call_args.args[3], True)
                self.assertEqual(verify.call_count, int(current_backend == 'swebench'))

    def test_other_backend_does_not_replace_swe_wheelhouse(self):
        self.check_backend('terminalbench_2_1', 'frozen-swe')

    def test_same_backend_retains_existing_refresh_behavior(self):
        self.check_backend('swebench', 'current')


if __name__ == '__main__':
    unittest.main()
