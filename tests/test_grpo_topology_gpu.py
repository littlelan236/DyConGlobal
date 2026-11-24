"""Explicitly opt-in Qwen/NCCL topology-switch acceptance via the formal CLI."""
import os
from pathlib import Path

import pytest
import torch

from core.io import read_json, write_json, sha256
from tests.test_grpo_execution_gpu import _fixture, _training

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_GRPO_TOPOLOGY_GPU') != '1',
    reason='requires eight explicitly selected idle GPUs and local Qwen3.5-9B')


@pytest.mark.parametrize('split,streaming',[(False,False),(False,True),(True,False),(True,True)])
def test_public_cli_bidirectional_resume_and_artifact_verification(tmp_path, split, streaming):
    from safetensors.torch import load_file
    evidence = Path(os.environ.get('DYCON_GRPO_TOPOLOGY_EVIDENCE',str(tmp_path)))/f'whole-{not split}-stream-{streaming}'
    evidence.mkdir(parents=True,exist_ok=False)
    cfg = _fixture(evidence,9 if split else 13)
    devices = list(cfg.runtime.gpus)
    cfg.runtime.data_preparation_async = streaming
    if split:
        counts = (8,2,4,8)
    else:
        counts = (2,8,4,2)
        cfg.grpo.group_batch_size = 4
        cfg.grpo.parallel_branch_chunk_size = None
        cfg.grpo.parallel_branch_schedule = None
    cfg.runtime.gpus = devices[:counts[0]]
    full,_ = _training(cfg,evidence,'fixed-topology')
    previous = None
    receipts = {}
    for step,world in enumerate(counts,1):
        item = cfg.model_copy(deep=True)
        item.runtime.gpus = devices[:world]
        item.runtime.topology_policy = 'exact' if previous is None else 'preserve_updates'
        current,_ = _training(item,evidence,f'world-{world}-step-{step}',parent=previous,stop=step if step<4 else None)
        previous = current/'checkpoints'/f'step-{step:06d}'
        want = full/'checkpoints'/f'step-{step:06d}'
        a,b = [torch.load(p/'training-state.pt',map_location='cpu',weights_only=True) for p in (want,previous)]
        assert a['metadata']['state'] == b['metadata']['state']
        assert a['metadata']['contract'] == b['metadata']['contract']
        assert len(b['rank_rng']) == world
        assert a['scheduler'] == b['scheduler']
        expected = load_file(str(want/'adapter/adapter_model.safetensors'))
        actual = load_file(str(previous/'adapter/adapter_model.safetensors'))
        for name in expected:
            torch.testing.assert_close(actual[name],expected[name],atol=2e-4,rtol=2e-3)
        for index, state in a['optimizer']['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(b['optimizer']['state'][index][name],value,atol=2e-5,rtol=2e-3)
        for path,digest in receipts.items():
            assert sha256(path) == digest
        receipts[previous/'receipt.json'] = sha256(previous/'receipt.json')
    write_json(evidence/'acceptance.json',{'status':'passed','counts':counts,'streaming':streaming,
        'logical_schedule':'two_pass' if split else 'whole_parent','baseline':str(full),'resumed':str(current),
        'long_context_capacity_verified':False,'throughput_verified':False,'quality_verified':False})
