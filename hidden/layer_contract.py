"""Public capture sites and immutable on-disk native site bindings.

The last public site denotes final norm, native site N rather than N-1.
Legacy online NPZs retain native names; readers expose public names only when
a pointwise receipt explicitly binds the complete storage key set.
"""
from pathlib import Path

from core.io import read_json, sha256
from hidden.backends.vllm.capture import native_layers

POINTWISE_CONTRACT = 'public_layers_receipt_roots_v2'


def storage_binding(cfg, names):
    public = ['layer_' + str(i) for i in cfg['capture']['layers']]
    if set(names) == set(public):
        return None
    model = cfg.get('model', {})
    config = Path(model.get('path', '')) / 'config.json'
    if not model.get('path') or not config.is_file():
        raise ValueError('Incomplete tensor layer count; no model config for native site binding')
    data = read_json(config)
    count = data.get('text_config', data).get('num_hidden_layers')
    if type(count) is not int or count <= 0:
        raise ValueError('Model config has no valid text decoder layer count')
    native = ['layer_' + str(i) for i in native_layers(cfg['capture']['layers'], count)]
    if set(names) != set(native):
        raise ValueError('Incomplete tensor layer count: public/native site set mismatch')
    return {
        'schema_version': 1, 'convention': 'public_final_norm_native_v1',
        'public_to_storage': dict(zip(public, native)),
        'num_hidden_layers': count, 'model_fingerprint': model.get('fingerprint'),
        'model_config': str(config.resolve()), 'model_config_sha256': sha256(config),
    }


def public_arrays(arrays, row):
    binding = row.get('layer_binding')
    if binding is None:
        return arrays
    if binding.get('schema_version') != 1 or binding.get('convention') != 'public_final_norm_native_v1':
        raise ValueError('Unknown hidden layer binding contract')
    mapping = binding['public_to_storage']
    if (not mapping or len(set(mapping.values())) != len(mapping)
            or set(arrays) != set(mapping.values())):
        raise ValueError('Hidden layer binding differs from stored tensor sites')
    count = binding['num_hidden_layers']
    public = [int(name.removeprefix('layer_')) for name in mapping]
    expected = dict(zip(mapping, ['layer_' + str(i) for i in native_layers(public, count)]))
    if mapping != expected:
        raise ValueError('Hidden layer binding violates public/native site convention')
    return {name: arrays[stored] for name, stored in mapping.items()}
