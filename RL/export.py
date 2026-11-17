"""Publish a full serving checkpoint from a verified project LoRA checkpoint."""

from pathlib import Path
import os
import shutil
import tempfile

from core.io import read_json, write_json, sha256, now


def merge_weight(weight, a, b, scale, dtype):
    """CPU FP32 merge with a single final serving-dtype rounding."""
    import torch
    if weight.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError('Only linear LoRA weights are supported')
    if a.shape[1] != weight.shape[1] or b.shape != (weight.shape[0], a.shape[0]):
        raise ValueError('LoRA matrix shapes do not match the base weight')
    if not all(torch.isfinite(t).all().item() for t in (weight, a, b)):
        raise ValueError('Nonfinite input weight')
    merged = (weight.float() + (b.float() @ a.float()) * scale).to(dtype)
    if not torch.isfinite(merged).all().item():
        raise ValueError('Nonfinite merged weight')
    return merged.contiguous()


def export(root_config, *, checkpoint, base_model, output, dtype='float16', dry_run=False):
    from RL.checkpoints import verify_checkpoint
    from RL.artifacts import base_identity, inventory
    from core.model_identity import register
    checkpoint = Path(checkpoint).resolve()
    base_model = Path(base_model).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    receipt = verify_checkpoint(checkpoint)
    adapter = read_json(checkpoint / 'adapter/identity.json')
    settings = read_json(checkpoint / 'adapter/adapter_config.json')
    identity = base_identity(base_model, root_config['project_root'], verify_content=True)
    if identity['fingerprint'] != adapter['contract']['base_fingerprint']:
        raise ValueError('Checkpoint adapter belongs to a different base model')
    unsupported = ('fan_in_fan_out', 'modules_to_save', 'use_dora', 'use_rslora',
                   'rank_pattern', 'alpha_pattern', 'layer_replication', 'alora_invocation_tokens',
                   'lora_bias', 'target_parameters', 'ensure_weight_tying')
    if settings.get('peft_type') != 'LORA' or settings.get('bias') != 'none' or any(settings.get(k) for k in unsupported):
        raise ValueError('Only plain fixed-rank, bias-free linear LoRA is supported')
    index = read_json(base_model / 'model.safetensors.index.json')
    targets = adapter['contract']['targets']
    pairs = {}
    for target in targets:
        key = target['path'] + '.weight'
        if key not in index['weight_map'] or key in pairs:
            raise ValueError(f'Missing or duplicate target weight: {key}')
        prefix = 'base_model.model.' + target['path']
        pairs[key] = (prefix + '.lora_A.weight', prefix + '.lora_B.weight')
    metadata = {
        'schema_version': 1, 'kind': 'lora-merged-serving-v1', 'created_at': now(),
        'checkpoint': str(checkpoint), 'checkpoint_receipt_sha256': sha256(checkpoint / 'receipt.json'),
        'global_step': receipt['global_step'], 'base_model': str(base_model),
        'base_fingerprint': identity['fingerprint'], 'adapter_identity_sha256': sha256(checkpoint / 'adapter/identity.json'),
        'merge_compute_dtype': 'float32', 'output_dtype': dtype, 'device': 'cpu',
        'target_count': len(pairs), 'scale': settings['lora_alpha'] / settings['r'],
    }
    if dry_run:
        return {**metadata, 'dry_run': True, 'output': str(output)}
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    target_dtype = getattr(torch, dtype)
    with safe_open(checkpoint / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as source:
        expected = {k for pair in pairs.values() for k in pair}
        if set(source.keys()) != expected:
            raise ValueError('Adapter tensor set differs from the exact frozen target list')
        adapters = {k: source.get_tensor(k) for k in expected}
    for key, (ak, bk) in pairs.items():
        if adapters[ak].shape[0] != settings['r'] or adapters[bk].shape[1] != settings['r']:
            raise ValueError(f'Unexpected adapter rank for {key}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-staging-', dir=output.parent))
    try:
        seen = set()
        total_size = 0
        for name in sorted(set(index['weight_map'].values())):
            if Path(name).name != name:
                raise ValueError('Unsafe base shard path')
            tensors = {}
            with safe_open(base_model / name, framework='pt', device='cpu') as source:
                for key in source.keys():
                    if key in seen or index['weight_map'].get(key) != name:
                        raise ValueError('Base index and actual tensor inventory disagree')
                    seen.add(key)
                    weight = source.get_tensor(key)
                    if key in pairs:
                        ak, bk = pairs[key]
                        weight = merge_weight(weight, adapters[ak], adapters[bk], metadata['scale'], target_dtype)
                    elif weight.is_floating_point():
                        weight = weight.to(target_dtype)
                    tensors[key] = weight.contiguous()
                    total_size += weight.numel() * weight.element_size()
                save_file(tensors, staging / name, metadata={'format': 'pt'})
            del tensors
        if seen != set(index['weight_map']) or not set(pairs).issubset(seen):
            raise ValueError('Incomplete merged tensor inventory')
        index.setdefault('metadata', {})['total_size'] = total_size
        write_json(staging / 'model.safetensors.index.json', index)
        for path in base_model.iterdir():
            if path.is_file() and path.name not in {'tact-identity.json', 'model.safetensors.index.json'} and path.suffix != '.safetensors':
                if path.name.endswith(('.bin', '.pt')):
                    raise ValueError('Unexpected binary base checkpoint format')
                shutil.copyfile(path, staging / path.name)
        config = read_json(staging / 'config.json')
        config['dtype'] = dtype
        config['torch_dtype'] = dtype
        if 'text_config' in config:
            config['text_config']['dtype'] = dtype
            config['text_config']['torch_dtype'] = dtype
        write_json(staging / 'config.json', config)
        metadata['files'] = inventory(staging)
        write_json(staging / 'lora-export-receipt.json', metadata)
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    fingerprint = register(root_config['project_root'], str(output))
    return {**{k: v for k, v in metadata.items() if k != 'files'},
            'output': str(output), 'fingerprint': fingerprint,
            'receipt': str(output / 'lora-export-receipt.json')}
