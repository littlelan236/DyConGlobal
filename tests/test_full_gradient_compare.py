"""Independent statistics checks and opt-in real FSDP2 diagnostic lifecycle."""
import json
import math
import os
from pathlib import Path
import subprocess

import pytest


def test_complete_gradient_statistics_and_disjoint_shards():
    import torch
    from RL.full_gradient_compare import statistics, combine, describe
    actual = torch.tensor([1., -3., 4., 0., 2.])
    reference = torch.tensor([2., -2., 4., 0., -1.])
    whole = describe(statistics(actual, reference, chunk_elements=2))
    parts = describe(combine([statistics(actual[:3], reference[:3]), statistics(actual[3:], reference[3:])]))
    assert whole == parts
    assert whole['elements'] == 5 and whole['max_abs'] == 3
    assert whole['reference_sq'] == 25 and whole['difference_sq'] == 11
    assert whole['relative_l2'] == pytest.approx(math.sqrt(11/25))
    assert whole['cosine'] == pytest.approx(22 / math.sqrt(30*25))
    zero = describe(statistics(torch.ones(2), torch.zeros(2)))
    assert zero['relative_l2'] is None and zero['relative_l2_undefined']
    with pytest.raises(ValueError, match='Nonfinite'):
        statistics(torch.tensor([float('nan')]), torch.ones(1))


@pytest.mark.skipif(os.environ.get('DYCON_FULL_GRADIENT_GPU') != '1', reason='Opt-in idle-GPU diagnostic')
def test_tiny_full_gradient_comparison_cli(tmp_path):
    import yaml
    from tests.lora_helpers import raw_model
    from tests.test_full_policy import full_values
    from RL.config import Config
    from RL.grpo.prepared import materialize, binding
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    root = Path(__file__).parents[1]
    # Diagnostic directories follow the production temp/ ownership restriction.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='gradient-compare-test-', dir=root/'temp') as work:
        folder = Path(work)
        base = folder/'base'
        raw_model().save_pretrained(base)
        PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i):i for i in range(64)}, unk_token='0')),
            unk_token='0', pad_token='0', eos_token='1').save_pretrained(base)
        values = full_values()
        values['model'].update(path=str(base), dtype='bfloat16', gdn_backend='fla_chunk')
        values['runtime'].update(gpus=[int(i) for i in os.environ['DYCON_FULL_GPUS'].split(',')][:2], gpu=None, output_root=str(folder))
        values['run'].update(mode='smoke', attempt_id='gradient-comparison-test')
        reader = materialize(Config.model_validate(values), root, 'gradient-comparison-test-prepared')
        values['prepared_data'] = binding(reader.path)
        training = folder/'training.yaml'
        training.write_text(yaml.safe_dump(values))
        comparison = folder/'comparison.yaml'
        output = folder/'result'
        comparison.write_text(yaml.safe_dump(dict(training_config=str(training), group_id=reader.ids[0], output=str(output))))
        result = subprocess.run([str(root/'.venvs/runner/bin/dycon'), 'rl-gradient-compare', '--comparison-config', str(comparison)],
                                cwd=root, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-10000:]
        from RL.artifacts import verify_receipt
        verify_receipt(output, 'rl-full-gradient-comparison-v1')
        report = json.loads((output/'comparison.json').read_text())
        assert report['status'] == 'completed'
        reference, actual = (report['results'][k] for k in ('reference_pytorch','fla_chunk'))
        assert reference['initial_parameter_shards'] == actual['initial_parameter_shards']
        for mode in actual['modes'].values():
            assert mode['all_parameters']['elements'] == actual['inventory']['trainable_parameters']
            assert len(mode['per_parameter']) == len(actual['inventory']['parameters'])
            assert mode['all_parameters']['relative_l2'] < .05
        assert not list((output/'reference_pytorch').glob('gradients-*'))
        assert json.loads((output/'gradient-cleanup.json').read_text())['deleted_bytes'] > 0
