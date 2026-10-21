"""Read frozen Ridge publications and retain their complete provenance without refitting."""

import shutil
from pathlib import Path

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256
from difficulty.models.ridge_grid import load_model

RIDGE_MODEL_KEYS = ('coef', 'intercept', 'feature_mean', 'feature_scale')


def read_publication(path, *, target, site, width, fingerprint, live=True):
    path = Path(path)
    if live:
        from runtime.serving.parallel import artifact_contract
        manifest = artifact_contract(path, model_fingerprint=fingerprint)
    else:
        manifest = read_json(path)
        for relative, digest in manifest['files'].items():
            if sha256(owned_path(path.parent, relative)) != digest:
                raise ValueError('Ridge publication snapshot file changed: ' + relative)
    if (manifest.get('schema_version') != 1 or manifest.get('loader') != 'runtime.serving.ridge_provider:load'
            or manifest.get('target') != target or manifest.get('sites') != [site]
            or manifest.get('hidden_size') != width or manifest.get('endpoint') != 'submission'
            or manifest.get('model_fingerprint') != fingerprint):
        raise ValueError('Ridge publication target/site/width/fingerprint contract mismatch')
    filename = manifest['model_file']
    if filename not in manifest['files']:
        raise ValueError('Ridge publication must hash-bind its model')
    model_path = owned_path(path.parent, filename)
    model = load_model(model_path)
    with np.load(model_path, allow_pickle=False) as archive:
        expected = {'target': target, 'layer': site, 'model_kind': 'ridge',
                    'target_transform': 'log1p', 'model_fingerprint': fingerprint,
                    'input_width': width}
        if len(model['coef']) != width or any(
                key not in archive or archive[key].item() != value for key, value in expected.items()):
            raise ValueError('Ridge checkpoint disagrees with publication contract')
        scope = str(archive['fit_scope'].item())
        if scope not in {'train', 'train_validation'} or manifest['provenance'].get('fit_scope') != scope:
            raise ValueError('Ridge publication has inconsistent fit scope')
        # Published experiment evidence is authoritative; do not consult a live Catalog.
        if 'source-split.json' not in manifest['files']:
            raise ValueError('Ridge publication must include source-split.json to audit held-out overlap')
        source_split_path = owned_path(path.parent, 'source-split.json')
        split = read_json(source_split_path)
        if archive['split_sha256'].item() != sha256(source_split_path):
            raise ValueError('Ridge checkpoint/source split SHA mismatch')
        roles = ['train'] if scope == 'train' else ['train', 'validation']
        trained_ids = set().union(*(set(split['sample_ids'][role]) for role in roles))
    info = {'manifest': str(path), 'manifest_sha256': sha256(path),
            'checkpoint_sha256': sha256(model_path), 'target': target, 'public_site': site,
            'width': width, 'model_fingerprint': fingerprint, 'fit_scope': scope,
            'trained_sample_ids_sha256': hash_json(sorted(trained_ids))}
    return model, manifest, info, trained_ids


def load_sources(root, settings, fingerprint, split):
    models, sources = {}, {}
    for target, reference in settings.ridge_publications.items():
        path = (Path(root) / reference).resolve()
        model, _, info, trained_ids = read_publication(
            path, target=target, site=settings.public_site, width=4096, fingerprint=fingerprint)
        overlaps = {role: len(trained_ids & set(ids)) for role, ids in split['sample_ids'].items()}
        if overlaps.get('validation', 0) or overlaps.get('test', 0):
            raise ValueError(f'Ridge base trained on current held-out samples: {target}: {overlaps}')
        sources[target] = {**info, 'current_split_training_overlap': overlaps}
        models[target] = model
    return models, sources


def snapshot_publication(source, destination):
    path = Path(source['manifest'])
    if sha256(path) != source['manifest_sha256']:
        raise ValueError('Ridge publication manifest changed during attempt')
    manifest = read_json(path)
    for relative, digest in manifest['files'].items():
        original = owned_path(path.parent, relative)
        if sha256(original) != digest:
            raise ValueError('Ridge publication payload changed during attempt')
        copied = owned_path(destination, relative)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        if sha256(copied) != digest:
            raise ValueError('Ridge publication snapshot copy changed')
    shutil.copyfile(path, Path(destination) / 'manifest.json')
    if sha256(Path(destination) / 'manifest.json') != source['manifest_sha256']:
        raise ValueError('Ridge publication manifest snapshot changed')


def verify_source_snapshot(directory, target, source, ridge_model, split):
    path = directory / f'models/{target}/ridge-publication/manifest.json'
    model, _, info, trained_ids = read_publication(
        path, target=target, site=source['public_site'], width=source['width'],
        fingerprint=source['model_fingerprint'], live=False)
    for key in ('manifest_sha256', 'checkpoint_sha256', 'target', 'public_site', 'width',
                'model_fingerprint', 'fit_scope', 'trained_sample_ids_sha256'):
        if info[key] != source[key]:
            raise ValueError('Ridge source snapshot disagrees with attempt: ' + key)
    overlaps = {role: len(trained_ids & set(ids)) for role, ids in split['sample_ids'].items()}
    if (overlaps != source['current_split_training_overlap'] or overlaps.get('validation', 0)
            or overlaps.get('test', 0)):
        raise ValueError('Ridge snapshot has incompatible held-out sample overlap')
    if any(not np.array_equal(model[k], ridge_model[k]) for k in RIDGE_MODEL_KEYS):
        raise ValueError('Combined checkpoint Ridge arrays differ from published checkpoint')
