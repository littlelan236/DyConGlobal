"""Selected cohort identity, frozen digest resume and image integrity contracts."""
import copy
from collections import Counter
import json

import pytest

from core.io import hash_json, read_json, sha256, write_json
from runtime.images import webarena as wav
from runtime.images.registry import persist_bytes


def source_tasks():
    return [dict(task_id=i * 100 + j, revision=2, sites=['shopping'],
                 intent_template_id=j, intent='Task',
                 eval=[dict(evaluator='AgentResponseEvaluator', expected=dict(task_type=typ))])
            for i, typ in enumerate(wav.TASK_TYPES) for j in range(60)]


def test_quota_selection_is_stable_and_without_replacement():
    tasks = source_tasks()
    hard = {t['task_id'] for t in tasks if t['task_id'] % 100 < 20}
    options = dict(hard_per_type=15, non_hard_per_type=35, seed=20261031)
    selected, rows, _ = wav.select_tasks(tasks, hard, **options)
    assert len(selected) == len({t['task_id'] for t in selected}) == 150
    assert Counter((r['task_type'], r['difficulty']) for r in rows) == {
        (typ, diff): n for typ in wav.TASK_TYPES for diff, n in [('hard', 15), ('non-hard', 35)]}
    assert wav.select_tasks(list(reversed(tasks)), hard, **options) == (selected, rows, _)
    options['seed'] += 1
    assert wav.select_tasks(tasks, hard, **options)[1] != rows


def test_insufficient_stratum_and_duplicate_ids_rejected():
    tasks = source_tasks()
    with pytest.raises(ValueError, match='Insufficient'):
        wav.select_tasks(tasks, set(), hard_per_type=15, non_hard_per_type=35, seed=1)
    with pytest.raises(ValueError, match='Duplicate'):
        wav.select_tasks(tasks + [tasks[0]], set(), hard_per_type=15, non_hard_per_type=35, seed=1)


def test_registry_metadata_fetches_only_requested_platform(tmp_path, monkeypatch):
    import hashlib
    import runtime.images.registry as registry
    documents, requested = {}, []
    def put(value):
        raw = json.dumps(value).encode()
        digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        documents[digest] = raw
        return dict(digest=digest, size=len(raw), mediaType=value.get('mediaType', 'application/vnd.oci.image.config.v1+json'))
    config = put(dict(os='linux', architecture='amd64'))
    manifest = put(dict(schemaVersion=2, mediaType='application/vnd.oci.image.manifest.v1+json', config=config, layers=[]))
    index = put(dict(schemaVersion=2, mediaType='application/vnd.oci.image.index.v1+json', manifests=[
        dict(**manifest, platform=dict(os='linux', architecture='amd64')),
        dict(digest='sha256:' + '0' * 64, size=123, mediaType=manifest['mediaType'], platform=dict(os='linux', architecture='arm64'))]))
    class Response:
        def __init__(self, content): self.content = content
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Client:
        def __init__(self, *args, **kwargs): pass
        def get(self, repository, endpoint):
            requested.append(endpoint)
            return Response(documents[index['digest'] if endpoint == 'manifests/latest' else endpoint.split('/', 1)[1]])
        def close(self): pass
    monkeypatch.setattr(registry, 'Registry', Client)
    root, descriptors = registry.image_metadata('org/image', tmp_path, platform='linux/amd64')
    assert root['digest'] == index['digest']
    assert len(descriptors) == 3
    assert 'manifests/sha256:' + '0' * 64 not in requested


def test_pinned_resume_and_nas_integrity(tmp_path, monkeypatch):
    store, state = tmp_path / 'nas' / 'publication', tmp_path / 'state'
    monkeypatch.setattr(wav, '_paths', lambda cfg: (store, state))
    monkeypatch.setattr(wav.os.path, 'ismount', lambda path: path == tmp_path / 'nas')
    cfg = dict(project_root=str(tmp_path), paths=dict(nas_root=str(tmp_path / 'nas')),
               environment=dict(platform='linux/amd64'), webarena_images=dict(workers=2, host_proxy=None))
    tasks = source_tasks()
    selected, rows, strata = wav.select_tasks(tasks, {i * 100 for i in range(3)},
                                             hard_per_type=1, non_hard_per_type=1, seed=1)
    selection = dict(task_count=6, tasks=rows, strata=strata, site_counts=dict(shopping=6),
                     images=[dict(site='shopping', source_image='am1n3e/site:latest')], scope='images only')
    monkeypatch.setattr(wav, 'inventory', lambda cfg: (copy.deepcopy(selection), selected))
    requests, transfers = [], []
    def metadata(repository, directory, **kwargs):
        requests.append(repository)
        layer = b'compressed layer bytes'
        config = json.dumps(dict(os='linux', architecture='amd64')).encode()
        def blob(raw, media):
            return dict(digest=persist_bytes(directory, raw), size=len(raw), mediaType=media, repository=repository)
        config_d = blob(config, 'application/vnd.oci.image.config.v1+json')
        layer_d = blob(layer, 'application/vnd.oci.image.layer.v1.tar+gzip')
        manifest = dict(schemaVersion=2, mediaType='application/vnd.oci.image.manifest.v1+json',
                        config=config_d, layers=[layer_d])
        root = blob(json.dumps(manifest).encode(), manifest['mediaType'])
        return root, [root, config_d, layer_d]
    monkeypatch.setattr(wav, 'image_metadata', metadata)
    def transfer(item, directory, **kwargs):
        transfers.append(item['digest'])
        assert sha256(directory / item['digest'][7:]) == item['digest'][7:]
    monkeypatch.setattr(wav, 'transfer_blob', transfer)
    wav.plan(cfg)
    assert wav.prepare(cfg)['status'] == 'complete'
    locked = read_json(store / 'source-locks/shopping.json')
    assert wav.prepare(cfg)['status'] == 'complete'
    assert len(requests) == 1  # floating source tag must not be re-resolved on resume
    assert wav.status(cfg, deep=True)['deep_verified'] is True
    assert read_json(store / 'source-locks/shopping.json') == locked
    # A changed quota/selection may not overwrite the existing publication.
    selection['task_count'] = 7
    with pytest.raises(ValueError, match='Frozen selection'):
        wav.plan(cfg)
    selection['task_count'] = 6
    # Same-size corruption must be caught by explicit deep verification.
    image = read_json(store / 'images/shopping.json')
    layer_path = store / 'blobs/sha256' / image['layers'][0]['digest'][7:]
    layer_path.write_bytes(b'!' * layer_path.stat().st_size)
    with pytest.raises(ValueError, match='Corrupt OCI blob'):
        wav.status(cfg, deep=True)


def test_verify_reads_derived_image_from_its_own_nas_store(tmp_path, monkeypatch):
    store, derived = tmp_path / 'main', tmp_path / 'derived'
    monkeypatch.setattr(wav, '_paths', lambda cfg: (store, tmp_path / 'state'))
    monkeypatch.setattr(wav.os.path, 'ismount', lambda path: path == tmp_path)
    cfg = dict(paths=dict(nas_root=str(tmp_path)), environment=dict(platform='linux/amd64'),
               webarena_images=dict(workers=2))
    task = source_tasks()[0]
    selection = dict(task_count=1, revision='pinned', tasks=[dict(task_id=task['task_id'], task_sha256=hash_json(task))],
                     strata=[], site_counts=dict(wikipedia=1), images=[dict(site='wikipedia')], scope='images only')
    write_json(store / 'selection.json', selection)
    write_json(store / 'evaluator/tasks.json', [task])
    selection_sha = sha256(store / 'selection.json')
    write_json(store / 'selection-receipt.json', dict(selection_sha256=selection_sha, task_count=1,
                evaluator_tasks_sha256=sha256(store / 'evaluator/tasks.json')))
    blobs = derived / 'blobs/sha256'
    blobs.mkdir(parents=True)
    def blob(raw, media):
        return dict(digest=persist_bytes(blobs, raw), size=len(raw), mediaType=media)
    config = blob(json.dumps(dict(os='linux', architecture='amd64')).encode(), 'application/vnd.oci.image.config.v1+json')
    layer = blob(b'layer', 'application/vnd.oci.image.layer.v1.tar+gzip')
    manifest = dict(schemaVersion=2, mediaType='application/vnd.oci.image.manifest.v1+json', config=config, layers=[layer])
    root = blob(json.dumps(manifest).encode(), manifest['mediaType'])
    write_json(derived / 'images/wiki.json', dict(manifest_digest=root['digest'], manifest_size=root['size'],
        platform='linux/amd64', provenance=dict(selection_sha256=selection_sha, revision='pinned')))
    reference = dict(store='derived', manifest='images/wiki.json',
                     manifest_sha256=sha256(derived / 'images/wiki.json'), image_id=root['digest'])
    write_json(store / 'images-manifest.json', dict(selection_sha256=selection_sha, images=dict(wikipedia=reference)))
    assert wav.status(cfg, deep=True)['deep_verified'] is True
    (blobs / layer['digest'][7:]).write_bytes(b'bad!!')
    with pytest.raises(ValueError, match='Corrupt OCI blob'):
        wav.status(cfg, deep=True)
