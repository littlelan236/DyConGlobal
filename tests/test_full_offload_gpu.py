"""Opt-in NCCL full-policy activation transfers, numerical oracle and exact resume.

All comparisons are within one backend and fixed world size. Pressure's 200-GiB
headroom intentionally selects every boundary on these 96-GiB devices; this is
policy coverage, not a natural memory-pressure or throughput measurement.
"""
import copy
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from core.io import read_json
from tests.test_full_offload import managed_values
from tests.test_full_fsdp2_gpu import cli, checkpoint_tensors

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_FULL_OFFLOAD_GPU') != '1',
                               reason='Requires explicit idle GPU pair')


def identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            identical(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            identical(x, y)
    else:
        assert a == b


def compare_checkpoint(a, b, world):
    for name, tensor in checkpoint_tensors(a).items():
        torch.testing.assert_close(tensor, checkpoint_tensors(b)[name], atol=0, rtol=0, msg=name)
    # Each local weight and FP32 m/v tensor must agree, not only loss/samples.
    left, right = (read_json(p / 'receipt.json')['files'] for p in (a, b))
    for name in left:
        if name.startswith('ranks/') and name.endswith('.safetensors'):
            assert left[name]['sha256'] == right[name]['sha256'], name
    for rank in range(world):
        identical(*(torch.load(p / f'ranks/{rank}/runtime.pt', weights_only=True) for p in (a, b)))


@pytest.mark.parametrize('backend', ['reference_pytorch', 'fla_chunk'])
def test_full_offload_transfers_oracle_resume_and_refusal(tmp_path, backend):
    from tests.lora_helpers import raw_model
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.set_num_threads(1)
    model_path = tmp_path / 'tiny-base'
    raw_model().to(torch.bfloat16).save_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token='0')),
        unk_token='0', pad_token='0', eos_token='1')
    tokenizer.save_pretrained(model_path)
    gpus = [int(x) for x in os.environ['DYCON_FULL_GPUS'].split(',')]
    world = len(gpus)
    assert world == 2
    values = managed_values()
    values['run'].update(mode='smoke')
    values['model'].update(path=str(model_path), dtype='bfloat16', gdn_backend=backend)
    values['runtime'] = {'gpus': gpus, 'output_root': str(tmp_path), 'distributed_timeout_seconds': 60}
    values['training'].update(learning_rate=1e-3, checkpoint_every_updates=2)
    values['algorithm']['dataset_root'] = str(tmp_path)
    values['preflight'] = {'verify_identity': True, 'check_resources': True}
    values['grpo'].update(top_k=1, k3_cap=1.)

    def run(label, current, expected=0):
        current['run']['attempt_id'] = label
        config = tmp_path / (label + '.yaml')
        config.write_text(yaml.safe_dump(current))
        cli(tmp_path, label, ['rl-train', '--training-config', str(config)], timeout=240, expected=expected)
        return tmp_path / 'RL/checkpoints' / label

    baseline = copy.deepcopy(values)
    baseline['training']['memory'].update(activation_offload='none', host_offload={'enabled': False})
    oracle = run('oracle', baseline) / 'checkpoints/step-000002'
    forced = None
    evidence = []
    for mode in ['checkpoint_boundary', 'auto', 'memory_pressure', 'none']:
        current = copy.deepcopy(values)
        current['training']['memory']['activation_offload'] = mode
        current['training']['memory']['auto_offload'] = {'prompt_tokens_threshold': 1, 'sequence_tokens_threshold': 1}
        current['training']['memory']['memory_pressure'] = {'device_headroom_gib': 200.}
        attempt = run(mode, current)
        checkpoint = attempt / 'checkpoints/step-000002'
        compare_checkpoint(checkpoint, oracle, world)
        metrics = [json.loads(s) for s in (attempt / 'metrics.jsonl').read_text().splitlines()]
        for row in metrics:
            assert row['policy_changed']
            for rank in row['ranks']:
                offload = rank['activation_offload']
                assert offload['offload_live_bytes'] == 0
                assert offload['host_pool_reserved_bytes'] == 0
                if mode != 'none':
                    assert offload['offload_d2h_bytes'] > 0
                    assert offload['offload_h2d_bytes'] == offload['offload_d2h_bytes']
                else:
                    assert offload['offload_d2h_bytes'] == offload['offload_h2d_bytes'] == 0
        evidence.append({'mode': mode, 'metrics': metrics})
        if mode == 'checkpoint_boundary':
            forced = (current, checkpoint)
    pause = copy.deepcopy(forced[0])
    pause['validation']['stop_after_update'] = 1
    source = run('pause', pause) / 'checkpoints/step-000001'
    resume = copy.deepcopy(forced[0])
    resume['run']['start'] = {'mode': 'resume', 'checkpoint': str(source)}
    final = run('resume', resume) / 'checkpoints/step-000002'
    compare_checkpoint(final, forced[1], world)
    cli(tmp_path, 'verify', ['artifact', 'verify', 'resume', '--scope', 'rl-full-attempt', '--storage-root', str(tmp_path)])
    wrong = copy.deepcopy(resume)
    wrong['training']['memory']['host_offload']['allocator_budget_gib'] = .25
    run('wrong-policy', wrong, expected=2)
    refusal = copy.deepcopy(values)
    refusal['training']['memory']['host_offload']['allocator_budget_gib'] = .0005
    attempt = run('refused', refusal, expected=None)
    assert read_json(attempt / 'attempt.json')['status'] == 'failed'
    assert not (attempt / 'receipt.json').exists()
    assert not list((attempt / 'checkpoints').glob('*/receipt.json'))
    (tmp_path / 'offload-evidence.json').write_text(json.dumps(evidence))
