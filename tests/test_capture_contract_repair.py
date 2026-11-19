"""Regression: final-norm naming, immutable retries and staged receipt roots."""
import copy
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from core.io import read_json, sha256, write_json
from hidden.artifacts import read_arrays, iter_arrays_grouped
from hidden.layer_contract import POINTWISE_CONTRACT, public_arrays, storage_binding
from hidden.repair_points import _tensor
from application.execution.artifact import receipt_roots
from application.execution.capture_publication import finalize_capture_publication
from test_capture_publication_finalizer import setup_finalizer


def native_fixture(tmp_path):
    model = tmp_path / 'model'
    write_json(model / 'config.json', {'text_config': {'num_hidden_layers': 32}})
    cfg = {'capture': {'layers': [0, 31]}, 'model': {'path': str(model), 'fingerprint': 'model'}}
    arrays = {'layer_0': np.arange(4, dtype=np.float32), 'layer_32': np.ones(4, dtype=np.float32)}
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = buf.getvalue()
    path = tmp_path / 'raw.npz'
    path.write_bytes(payload)
    row = {'sample_id': 's', 'session_id': 'session', 'run_id': 'run', 'tensor': path.name,
           'sha256': sha256(path), 'storage_shapes': {k: list(v.shape) for k, v in arrays.items()},
           'storage_dtypes': {k: str(v.dtype) for k, v in arrays.items()}}
    return cfg, row, arrays


def test_native_final_norm_readers_preserve_bytes_and_expose_public_sites(tmp_path):
    cfg, row, arrays = native_fixture(tmp_path)
    original = copy.deepcopy(row)
    _, _, _, repaired = _tensor([(tmp_path, row)], tmp_path, cfg, {})
    assert row == original
    assert repaired['layer_binding']['public_to_storage'] == {'layer_0': 'layer_0', 'layer_31': 'layer_32'}
    assert sha256(tmp_path / 'raw.npz') == row['sha256']
    assert set(read_arrays(tmp_path, repaired)) == {'layer_0', 'layer_31'}
    np.testing.assert_array_equal(read_arrays(tmp_path, repaired)['layer_31'], arrays['layer_32'])
    assert set(next(iter_arrays_grouped([('point', tmp_path, repaired)]))[1]) == {'layer_0', 'layer_31'}
    shard = tmp_path / 'tensor.tar'
    with tarfile.open(shard, 'w:') as out:
        out.add(tmp_path / 'raw.npz', arcname='s.npz')
    repaired['tensor_locator'] = {'shard': shard.name, 'shard_sha256': sha256(shard),
                                 'member': 's.npz', 'member_sha256': row['sha256']}
    assert set(next(iter_arrays_grouped([('point', tmp_path, repaired)]))[1]) == {'layer_0', 'layer_31'}


def test_native_contract_rejects_missing_extra_and_false_final_norm(tmp_path):
    cfg, row, _ = native_fixture(tmp_path)
    for names in [{'layer_0'}, {'layer_0', 'layer_31', 'layer_32'}, {'layer_0', 'layer_33'}]:
        with pytest.raises(ValueError, match='layer count'):
            storage_binding(cfg, names)
    bad = storage_binding(cfg, row['storage_shapes'])
    bad['public_to_storage']['layer_31'] = 'layer_30'
    with pytest.raises(ValueError):
        public_arrays({'layer_0': [], 'layer_30': []}, {'layer_binding': bad})


def test_repair_receipt_scope_is_payload_root_and_rejects_identity_or_escape(tmp_path):
    run = tmp_path / 'runs/run'
    payload = run / 'hidden/capture-repairs/fix/payload'
    receipt = payload / 'capture-repairs/fix/sessions/s/receipt.json'
    value = {'repair_id': 'fix', 'run_id': 'run'}
    cfg = {'paths': {'nas_root': str(tmp_path / 'nas')}, 'capture': {'export': {'root': 'online'}}}
    assert receipt_roots(run, receipt, value, cfg) == (payload, tmp_path / 'nas/online/run')
    top = payload / 'capture-repairs/fix/receipt.json'
    assert receipt_roots(run, top, value, cfg)[0] == payload
    ordinary = run / 'rollout/receipt.json'
    assert receipt_roots(run, ordinary, {}, cfg) == (ordinary.parent, None)
    with pytest.raises(ValueError, match='identity'):
        receipt_roots(run, receipt, {**value, 'repair_id': 'wrong'}, cfg)
    from core.io import owned_path
    with pytest.raises(ValueError):
        owned_path(payload, '../outside')


def test_run_verify_checks_legacy_staged_repair_and_nas_dependencies(tmp_path, monkeypatch):
    import application.execution.artifact as artifact
    from types import SimpleNamespace
    run = tmp_path / 'runs/run'
    payload = run / 'hidden/capture-repairs/fix/payload'
    cfg = {'parallel': {}, 'paths': {'nas_root': str(tmp_path / 'nas')},
           'capture': {'export': {'root': 'online'}}}
    nas = tmp_path / 'nas/online/run'
    write_json(nas / 'requests/native.json', {'native': True})
    rel = 'capture-repairs/fix/sessions/s/point-index.json'
    write_json(payload / rel, {'points': []})
    write_json(payload / 'capture-repairs/fix/sessions/s/receipt.json', {
        'repair_id': 'fix', 'run_id': 'run', 'files': {rel: sha256(payload / rel)},
        'dependencies': {'requests/native.json': sha256(nas / 'requests/native.json')}})
    write_json(run / 'artifacts.json', {'artifacts': []})
    monkeypatch.setattr(artifact, 'load_plan', lambda *a: {'run': run, 'config': cfg})
    monkeypatch.setattr(artifact, 'RunEvidence', lambda *a: SimpleNamespace(
        verify=lambda: None, read=lambda: {'rollouts': {}}))
    assert artifact.verify(tmp_path, 'run')['external_metadata_checked'] == 1
    write_json(nas / 'requests/native.json', {'native': 'changed'})
    with pytest.raises(ValueError, match='dependency checksum'):
        artifact.verify(tmp_path, 'run')


def test_contract_revision_uses_new_ids_and_keeps_legacy_failed_receipts(tmp_path, monkeypatch):
    cfg, evidence, item, _ = setup_finalizer(tmp_path, monkeypatch, None)
    sentinel = tmp_path / 'runs/run/hidden/capture-repairs/legacy/receipt.json'
    write_json(sentinel, {'legacy': 'immutable'})
    original = sha256(sentinel)
    first = finalize_capture_publication(cfg, 'run', items=[item])
    assert first['status'] == 'complete', first
    selection = read_json(tmp_path / 'runs/run/hidden/capture-repairs' / first['repair_id'] / 'publication-selection.json')
    assert selection['selection']['repair_contract'] == POINTWISE_CONTRACT
    assert sha256(sentinel) == original
    again = finalize_capture_publication(cfg, 'run', items=[item])
    assert again['publication_id'] == first['publication_id']


def test_finalizer_publishes_native_named_npz_with_public_layer_binding(tmp_path, monkeypatch):
    cfg, evidence, item, _ = setup_finalizer(tmp_path, monkeypatch, None)
    model = tmp_path / 'model'
    write_json(model / 'config.json', {'num_hidden_layers': 32})
    cfg['model'] = {'path': str(model), 'fingerprint': 'model'}
    cfg['capture']['layers'] = [0, 31]
    write_json(tmp_path / 'runs/run/resolved-config.json', cfg)
    for path in (tmp_path / 'spool/run/index').glob('*.json'):
        row = read_json(path)
        tensor = tmp_path / 'spool/run' / row['tensor']
        np.savez(tensor, layer_0=np.ones(4, dtype=np.float32), layer_32=np.arange(4, dtype=np.float32))
        # sequence() has already copied the fixture's loose tensor to its NAS.
        (tmp_path / 'nas/online/run' / row['tensor']).write_bytes(tensor.read_bytes())
        row.update(sha256=sha256(tensor), storage_shapes={'layer_0': [4], 'layer_32': [4]},
                   storage_dtypes={'layer_0': 'float32', 'layer_32': 'float32'})
        write_json(path, row)
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    assert summary['status'] == 'complete', summary
    output = tmp_path / 'runs/run/enrichment/online-publications' / summary['publication_id']
    rows = read_json(output / 'index.json')
    assert len(rows) == 3
    assert rows[0]['layer_binding']['public_to_storage']['layer_31'] == 'layer_32'
    nas = tmp_path / 'nas/online/run'
    assert set(read_arrays(nas, rows[0])) == {'layer_0', 'layer_31'}
