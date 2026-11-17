"""CPU integration tests for serving export and tensor inventory preservation."""
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from RL.export import merge_weight, export


def test_merge_matches_live_lora_forward():
    torch.manual_seed(13)
    weight = torch.randn(7, 9)
    a, b = torch.randn(3, 9), torch.randn(7, 3)
    x = torch.randn(5, 9)
    expected = x @ weight.T + ((x @ a.T) @ b.T) * 2
    merged = merge_weight(weight, a, b, 2, torch.float32)
    torch.testing.assert_close(x @ merged.T, expected)


def test_reject_shape_and_nonfinite():
    with pytest.raises(ValueError, match='shapes'):
        merge_weight(torch.zeros(2, 3), torch.zeros(1, 4), torch.zeros(2, 1), 2, torch.float16)
    with pytest.raises(ValueError, match='Nonfinite'):
        merge_weight(torch.zeros(2, 3), torch.full((1, 3), float('nan')), torch.zeros(2, 1), 2, torch.float16)


def test_export_preserves_non_adapter_tensors_and_binds_source(tmp_path, monkeypatch):
    from RL import checkpoints, artifacts
    from core import model_identity
    base = tmp_path / 'base'
    ckpt = tmp_path / 'step-000001'
    adapter = ckpt / 'adapter'
    base.mkdir()
    adapter.mkdir(parents=True)
    write = lambda p, value: p.write_text(json.dumps(value))
    key = 'model.language_model.layers.0.mlp.down_proj.weight'
    other = 'mtp.special.weight'
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    untouched = torch.ones(2)
    save_file({key: weight, other: untouched}, base / 'model-01.safetensors')
    write(base / 'model.safetensors.index.json', {'weight_map': {key: 'model-01.safetensors', other: 'model-01.safetensors'}})
    write(base / 'config.json', {'text_config': {}})
    write(base / 'tokenizer.json', {'test': True})
    write(ckpt / 'receipt.json', {'global_step': 1})
    write(adapter / 'identity.json', {'contract': {'base_fingerprint': 'basehash', 'targets': [{'path': key[:-7]}]}})
    write(adapter / 'adapter_config.json', {'peft_type': 'LORA', 'bias': 'none', 'r': 1, 'lora_alpha': 2})
    prefix = 'base_model.model.' + key[:-7]
    save_file({prefix + '.lora_A.weight': torch.ones(1, 4), prefix + '.lora_B.weight': torch.ones(3, 1)}, adapter / 'adapter_model.safetensors')
    monkeypatch.setattr(checkpoints, 'verify_checkpoint', lambda p: {'global_step': 1})
    monkeypatch.setattr(artifacts, 'base_identity', lambda *a, **k: {'fingerprint': 'basehash'})
    monkeypatch.setattr(model_identity, 'register', lambda *a: 'mergedhash')
    out = tmp_path / 'merged'
    result = export({'project_root': str(tmp_path)}, checkpoint=ckpt, base_model=base, output=out)
    assert result['fingerprint'] == 'mergedhash'
    with safe_open(out / 'model-01.safetensors', framework='pt') as f:
        assert set(f.keys()) == {key, other}
        torch.testing.assert_close(f.get_tensor(key), (weight + 2).half())
        torch.testing.assert_close(f.get_tensor(other), untouched.half())
    assert (out / 'tokenizer.json').read_bytes() == (base / 'tokenizer.json').read_bytes()
    receipt = json.loads((out / 'lora-export-receipt.json').read_text())
    assert receipt['global_step'] == 1 and receipt['base_fingerprint'] == 'basehash'
    assert 'model-01.safetensors' in receipt['files']
    with pytest.raises(FileExistsError):
        export({'project_root': str(tmp_path)}, checkpoint=ckpt, base_model=base, output=out)

