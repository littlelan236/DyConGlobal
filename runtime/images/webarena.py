"""Freeze a stratified Verified cohort and cache its shared site images in NAS.

Image preparation only: no catalog registration, Docker load or benchmark execution.
Registry transfer, OCI verification and NAS image references use the shared modules.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.nas import NasImageStore
from runtime.images.oci import blob_path, descriptor_graph, platform_manifest, verify_blobs
from runtime.images.registry import image_metadata, transfer_blob

SOURCE_FILES = (
    'assets/dataset/webarena-verified.json',
    'assets/dataset/subsets/webarena-verified-hard.json',
    'docker-compose.yml',
    'src/webarena_verified/environments/container/config.py',
)
TASK_TYPES = ('RETRIEVE', 'NAVIGATE', 'MUTATE')


def select_tasks(tasks, hard_ids, *, hard_per_type, non_hard_per_type, seed):
    """Uniform, deterministic hash sampling without replacement in each stratum."""
    groups = defaultdict(list)
    seen = set()
    for task in tasks:
        task_id = task['task_id']
        if type(task_id) is not int or task_id in seen:
            raise ValueError('Duplicate or invalid source task ID')
        seen.add(task_id)
        labels = [e['expected']['task_type'].upper() for e in task['eval']
                  if e['evaluator'] == 'AgentResponseEvaluator']
        if len(labels) != 1 or labels[0] not in TASK_TYPES:
            raise ValueError('Expected one official task type per task')
        difficulty = 'hard' if task_id in hard_ids else 'non-hard'
        groups[(labels[0], difficulty)].append(task)
    if not set(hard_ids).issubset(seen):
        raise ValueError('Hard subset refers to unknown task IDs')
    selected, rows, strata = [], [], []
    for task_type in TASK_TYPES:
        for difficulty, count in (('hard', hard_per_type), ('non-hard', non_hard_per_type)):
            candidates = groups[(task_type, difficulty)]
            if type(count) is not int or count <= 0 or len(candidates) < count:
                raise ValueError(f'Insufficient tasks for {task_type}/{difficulty}: {len(candidates)} < {count}')
            def rank(task):
                return hashlib.sha256(f'{seed}:{task_type}:{difficulty}:{task["task_id"]}'.encode()).digest()
            cohort = sorted(candidates, key=rank)[:count]
            strata.append(dict(task_type=task_type, difficulty=difficulty,
                               available=len(candidates), selected=len(cohort)))
            selected.extend(cohort)
            for task in cohort:
                rows.append(dict(task_id=task['task_id'], revision=task['revision'],
                                 task_type=task_type, difficulty=difficulty,
                                 sites=task['sites'], intent_template_id=task['intent_template_id'],
                                 task_sha256=hash_json(task)))
    return selected, rows, strata


def inventory(cfg):
    options = cfg['webarena_images']
    project = Path(cfg['project_root']).resolve()
    source = (project / options['source_root']).resolve()
    if not source.is_relative_to(project):
        raise ValueError('WebArena source must be inside the project')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != options['revision']:
        raise ValueError('Verified checkout differs from pinned revision')
    # Compare actual bytes with the Git objects, including untracked replacements.
    for name in SOURCE_FILES:
        expected = subprocess.check_output(['git', '-C', str(source), 'show', revision + ':' + name])
        if (source / name).read_bytes() != expected:
            raise ValueError('Modified pinned source file: ' + name)
    tasks = read_json(source / SOURCE_FILES[0])
    hard_ids = read_json(source / SOURCE_FILES[1])['task_ids']
    selected, rows, strata = select_tasks(tasks, hard_ids,
        hard_per_type=options['hard_per_type'], non_hard_per_type=options['non_hard_per_type'], seed=options['seed'])
    sites = sorted({site for row in rows for site in row['sites']})
    compose = yaml.safe_load((source / 'docker-compose.yml').read_text())['services']
    units = []
    for site in sites:
        matches = [s['image'] for s in compose.values()
                   if s.get('image', '').split(':')[0] == 'am1n3e/webarena-verified-' + site]
        if len(matches) != 1:
            raise ValueError('Expected one official site image: ' + site)
        units.append(dict(site=site, source_image=matches[0],
                          task_ids=[row['task_id'] for row in rows if site in row['sites']]))
    result = dict(schema_version=1, benchmark='webarena_verified',
        repository='https://github.com/ServiceNow/webarena-verified', revision=revision,
        source_sha256={name: sha256(source / name) for name in SOURCE_FILES},
        sampling=dict(method='sha256-ranked-uniform-within-stratum-v1', seed=options['seed'],
                      hard_per_type=options['hard_per_type'], non_hard_per_type=options['non_hard_per_type']),
        platform=cfg['environment']['platform'], task_count=len(rows), strata=strata,
        site_counts=dict(sorted(Counter(site for row in rows for site in row['sites']).items())),
        tasks=rows, images=units,
        scope='selected task definitions and shared site images only; browser/data setup and rollout are not prepared')
    return result, selected


def _paths(cfg):
    options = cfg['webarena_images']
    nas = Path(cfg['paths']['nas_root']).resolve()
    if not any(p != Path('/') and os.path.ismount(p) for p in (nas, *nas.parents)):
        raise ValueError('Configured NAS is not mounted')
    store = nas / identifier(options['store'])
    state = Path(cfg['project_root']) / 'state/webarena-images' / store.name
    return store, state


def _immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable publication differs; use a new store: ' + str(path))
    else:
        write_json(path, value)


def _freeze(cfg, store):
    plan, selected = inventory(cfg)
    # Check compatibility before writing any new files to an existing publication.
    if (store / 'selection.json').exists() and read_json(store / 'selection.json') != plan:
        raise ValueError('Frozen selection differs; use a new publication store')
    _immutable(store / 'selection.json', plan)
    evaluator = store / 'evaluator/tasks.json'
    _immutable(evaluator, selected)
    evaluator.parent.chmod(0o700)
    evaluator.chmod(0o600)
    _immutable(store / 'selection-receipt.json', dict(selection_sha256=sha256(store / 'selection.json'),
               evaluator_tasks_sha256=sha256(evaluator), task_count=len(selected)))
    return plan


def plan(cfg):
    store, _ = _paths(cfg)
    with file_lock(store / '.prepare.lock', nonblocking=True):
        return _freeze(cfg, store)


def _read_selection(store):
    selection = read_json(store / 'selection.json')
    receipt = read_json(store / 'selection-receipt.json')
    tasks = read_json(store / 'evaluator/tasks.json')
    if sha256(store / 'selection.json') != receipt['selection_sha256'] or sha256(store / 'evaluator/tasks.json') != receipt['evaluator_tasks_sha256']:
        raise ValueError('Selection publication checksum mismatch')
    if len(tasks) != selection['task_count'] or len(tasks) != receipt['task_count']:
        raise ValueError('Selected task count mismatch')
    if [(t['task_id'], hash_json(t)) for t in tasks] != [(t['task_id'], t['task_sha256']) for t in selection['tasks']]:
        raise ValueError('Selected tasks differ from frozen metadata')
    return selection


def _source_lock(cfg, store, unit, selection_hash):
    path = store / 'source-locks' / (unit['site'] + '.json')
    if path.exists():
        lock = read_json(path)
        if lock['selection_sha256'] != selection_hash or lock['requested'] != unit['source_image']:
            raise ValueError('Source lock belongs to another selection')
        return lock
    override = store / 'overrides' / (unit['site'] + '.json')
    if override.exists():
        reference = read_json(override)
        image = NasImageStore(cfg, reference)
        image.verify(deep=True)
        provenance = image.record['provenance']
        if (unit['site'] != 'wikipedia' or provenance['selection_sha256'] != selection_hash
                or provenance['revision'] != cfg['webarena_images']['revision']
                or image.record['platform'] != cfg['environment']['platform']):
            raise ValueError('Derived image does not match this selection/platform')
        lock = dict(schema_version=1, selection_sha256=selection_hash, requested=unit['source_image'],
                    external_reference=reference, override_sha256=sha256(override),
                    descriptors=[], metadata_sha256=hash_json([]), platform=image.record['platform'])
        _immutable(path, lock)
        return lock
    repository, tag = unit['source_image'].rsplit(':', 1)
    blobs = store / 'blobs/sha256'
    blobs.mkdir(parents=True, exist_ok=True)
    root, descriptors = image_metadata(repository, blobs, reference=tag,
                                       proxy=cfg['webarena_images']['host_proxy'])
    runtime, manifest = platform_manifest(blobs, root, cfg['environment']['platform'])
    by_digest = {d['digest']: d for d in descriptors}
    required = [by_digest[d['digest']] for d in (runtime, manifest['config'], *manifest['layers'])]
    lock = dict(schema_version=1, selection_sha256=selection_hash, requested=unit['source_image'],
                registry='docker.io', repository=repository, resolved_at=now(),
                source_root=root, runtime_root=runtime, descriptors=required,
                metadata_sha256=hash_json(required), platform=cfg['environment']['platform'])
    _immutable(path, lock)
    print(f"Locked {unit['site']}: {runtime['digest']} ({sum(d['size'] for d in required):,} bytes)", flush=True)
    return lock


def _record(store, unit, lock):
    if 'external_reference' in lock:
        return lock['external_reference']
    blobs = store / 'blobs/sha256'
    root = lock['runtime_root']
    graph = descriptor_graph(blobs, root)
    manifest = read_json(blob_path(blobs, root['digest']))
    record = dict(schema_version=2, requested=unit['source_image'],
        manifest_digest=root['digest'], manifest_size=root['size'], root_media_type=root['mediaType'],
        runtime_manifest_digest=root['digest'], config=manifest['config'], layers=manifest['layers'],
        compressed_bytes=sum(d['size'] for d in manifest['layers']),
        repo_digests=[lock['repository'] + '@' + lock['source_root']['digest']],
        platform=lock['platform'], descriptor_count=len(graph), deep_verified=True,
        provenance=dict(selection_sha256=lock['selection_sha256'], site=unit['site'],
                        source_root=lock['source_root'], source_lock_sha256=sha256(store / 'source-locks' / (unit['site'] + '.json'))))
    path = store / 'images' / (unit['site'] + '.json')
    _immutable(path, record)
    return dict(store=store.name, manifest='images/' + unit['site'] + '.json',
                manifest_sha256=sha256(path), image_id=root['digest'])


def prepare(cfg):
    store, state = _paths(cfg)
    with file_lock(store / '.prepare.lock', nonblocking=True):
        selection = _freeze(cfg, store)
        _read_selection(store)
        selection_hash = sha256(store / 'selection.json')
        locks, source_failures = {}, []
        for unit in selection['images']:
            try:
                locks[unit['site']] = _source_lock(cfg, store, unit, selection_hash)
            except Exception as exc:
                source_failures.append(dict(site=unit['site'], error=str(exc)))
                print(f"Source unavailable {unit['site']}: {exc}", flush=True)
        descriptors = {}
        for lock in locks.values():
            if hash_json(lock['descriptors']) != lock['metadata_sha256']:
                raise ValueError('Source lock descriptor checksum mismatch')
            for d in lock['descriptors']:
                prior = descriptors.setdefault(d['digest'], d)
                if prior['size'] != d['size']:
                    raise ValueError('Conflicting shared blob size')
        progress = dict(status='downloading', started_at=now(),
                        selection_sha256=selection_hash, task_count=selection['task_count'],
                        blob_count=len(descriptors), compressed_bytes=sum(d['size'] for d in descriptors.values()),
                        complete_blobs=0, complete_bytes=0, failures=[], source_failures=source_failures)
        write_json(state / 'progress.json', progress)
        blobs = store / 'blobs/sha256'
        workers = cfg['webarena_images']['workers']
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(transfer_blob, d, blobs, proxy=cfg['webarena_images']['host_proxy']): d
                       for d in sorted(descriptors.values(), key=lambda d: d['size'], reverse=True)}
            for future in as_completed(pending):
                d = pending[future]
                try:
                    future.result()
                    progress['complete_blobs'] += 1
                    progress['complete_bytes'] += d['size']
                    print(f"Blob {progress['complete_blobs']}/{len(descriptors)}: {d['digest']} ({d['size']:,} bytes)", flush=True)
                except Exception as exc:
                    progress['failures'].append(dict(digest=d['digest'], error=str(exc)))
                    print(f"Failed {d['digest']}: {exc}", flush=True)
                write_json(state / 'progress.json', progress)
        if progress['failures']:
            progress['status'] = 'incomplete'
            write_json(state / 'progress.json', progress)
            return progress
        # Transfer hashes every byte. Validate each selected graph before publishing references.
        references = {unit['site']: _record(store, unit, locks[unit['site']])
                      for unit in selection['images'] if unit['site'] in locks}
        if source_failures:
            progress.update(status='incomplete', finished_at=now(), available_images=references)
            write_json(state / 'progress.json', progress)
            return progress
        publication = dict(schema_version=1, benchmark='webarena_verified',
            selection_sha256=selection_hash, task_count=selection['task_count'],
            images=references, unique_blob_count=len(descriptors),
            compressed_bytes=sum(d['size'] for d in descriptors.values()),
            scope=selection['scope'])
        for reference in references.values():
            if reference['store'] != store.name:
                external = NasImageStore(cfg, reference)
                graph = external.verify()
                publication['unique_blob_count'] += len(graph)
                publication['compressed_bytes'] += sum(d['size'] for d in graph)
        _immutable(store / 'images-manifest.json', publication)
        progress.update(status='complete', finished_at=now(), manifest_sha256=sha256(store / 'images-manifest.json'))
        write_json(state / 'progress.json', progress)
        return progress


def status(cfg, *, deep=False):
    store, state = _paths(cfg)
    selection = _read_selection(store)
    publication = read_json(store / 'images-manifest.json') if (store / 'images-manifest.json').exists() else None
    if publication and (publication['selection_sha256'] != sha256(store / 'selection.json') or
                        set(publication['images']) != {u['site'] for u in selection['images']}):
        raise ValueError('Image publication does not match selection')
    checks, descriptors, graphs = {}, {}, {}
    for unit in selection['images']:
        site = unit['site']
        try:
            if publication:
                reference = publication['images'][site]
            elif (store / 'overrides' / (site + '.json')).exists():
                reference = read_json(store / 'overrides' / (site + '.json'))
            else:
                path = store / 'images' / (site + '.json')
                reference = dict(store=store.name, manifest='images/' + site + '.json',
                                 manifest_sha256=sha256(path))
            image = NasImageStore(cfg, reference)
            graph = image.verify()
            if image.record['provenance']['selection_sha256'] != sha256(store / 'selection.json'):
                raise ValueError('Image is bound to another selection')
            lock_path = store / 'source-locks' / (site + '.json')
            if 'source_lock_sha256' in image.record['provenance']:
                if sha256(lock_path) != image.record['provenance']['source_lock_sha256']:
                    raise ValueError('Source lock checksum mismatch')
            elif site == 'wikipedia':
                provenance = image.record['provenance']
                if (provenance['revision'] != selection['revision'] or
                        image.record['platform'] != cfg['environment']['platform']):
                    raise ValueError('Derived Wikipedia source/platform mismatch')
            else:
                raise ValueError('Image source provenance is missing')
            descriptors.update({(str(image.root), d['digest']): d for d in graph})
            graphs.setdefault(image.root / 'blobs/sha256', {}).update({d['digest']: d for d in graph})
            checks[site] = dict(available=True, image_id=image.image_id)
        except (OSError, ValueError, KeyError) as exc:
            checks[site] = dict(available=False, error=str(exc))
    complete = all(c['available'] for c in checks.values())
    if deep and complete:
        for root, graph in graphs.items():
            verify_blobs(root, graph.values(), workers=cfg['webarena_images']['workers'])
    result = dict(status='complete' if complete else 'incomplete', store=str(store),
        task_count=selection['task_count'], strata=selection['strata'], site_counts=selection['site_counts'],
        images=checks, deep_verified=bool(deep and complete),
        compressed_bytes=sum(d['size'] for d in descriptors.values()),
        progress=read_json(state / 'progress.json') if (state / 'progress.json').exists() else None,
        scope=selection['scope'])
    if deep:
        write_json(state / 'verification.json', dict(checked_at=now(), **result))
    return result


def build_wikipedia(cfg):
    """Rebuild the pinned official Wikipedia Dockerfile for the host platform.

    Reuses the existing build infrastructure, base-image NAS cache/lease and Docker
    OCI exporter. Publishes a separate derived store and explicit override reference;
    frozen task selection and original registry digests are never rewritten.
    """
    import copy
    import shutil
    import docker
    from runtime.images.skillsbench import _cache_source, _command, _record as publish_image, build_proxy
    from runtime.images.nas import ImageLeaseManager
    from runtime.scripts.export_task_images_to_nas import import_archive

    if not cfg['webarena_images'].get('wikipedia_build'):
        raise ValueError('Enable webarena_images.wikipedia_build in root config.yaml')
    build_cfg = copy.deepcopy(cfg)
    build_cfg['skillsbench_images']['host_proxy'] = cfg['webarena_images']['host_proxy']
    store, state = _paths(cfg)
    with file_lock(store / '.wikipedia-build.lock', nonblocking=True):
        selection = _read_selection(store)
        if 'wikipedia' not in selection['site_counts']:
            raise ValueError('Selected tasks do not require Wikipedia')
        source = (Path(cfg['project_root']) / cfg['webarena_images']['source_root']).resolve()
        inventory(cfg)  # Verify source/task commit and unmodified official configuration.
        relative = 'dev/environments/docker/sites/wikipedia/Dockerfile'
        names = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '--name-only',
            selection['revision'], '--', 'packages/environment_control/environment_control',
            'dev/environments/docker/sites/wikipedia'], text=True).splitlines()
        if relative not in names:
            raise ValueError('Official Wikipedia Dockerfile is missing')
        hashes = {}
        for name in names:
            expected = subprocess.check_output(['git', '-C', str(source), 'show', selection['revision'] + ':' + name])
            if (source / name).read_bytes() != expected:
                raise ValueError('Modified Wikipedia build source: ' + name)
            hashes[name] = sha256(source / name)
        dockerfile = (source / relative).read_text()
        base = 'ghcr.io/kiwix/kiwix-serve:3.3.0'
        if dockerfile.splitlines()[0].strip() != 'FROM ' + base:
            raise ValueError('Unexpected official Wikipedia base image')
        derived = store.with_name(store.name + '-wikipedia-amd64')
        build_state = state / 'wikipedia-build'
        build_state.mkdir(parents=True, exist_ok=True)
        (derived / 'blobs/sha256').mkdir(parents=True, exist_ok=True)
        spec = dict(selection_sha256=sha256(store / 'selection.json'), revision=selection['revision'],
                    source_sha256=hashes, platform=cfg['environment']['platform'], base_requested=base,
                    infrastructure={k: cfg['skillsbench_images'][k] for k in
                        ('builder_name', 'docker_context', 'build_proxy', 'manage_build_proxy', 'build_seconds')})
        _immutable(derived / 'build-spec.json', spec)
        output = derived / 'images/wikipedia-amd64.json'
        if output.exists():
            reference = dict(store=derived.name, manifest='images/wikipedia-amd64.json',
                             manifest_sha256=sha256(output), image_id=read_json(output)['manifest_digest'])
            NasImageStore(cfg, reference).verify(deep=True)
            _immutable(store / 'overrides/wikipedia.json', reference)
            return dict(status='complete', image=reference, reused=True)
        base_reference = _cache_source(build_cfg, derived, build_state, base, platform=cfg['environment']['platform'])
        base_store = NasImageStore(cfg, base_reference)
        base_store.verify(deep=True)
        client = docker.DockerClient(base_url=cfg['environment']['docker_host'], timeout=120)
        try:
            if shutil.disk_usage(client.info()['DockerRootDir']).free < cfg['skillsbench_images']['reserve_bytes']:
                raise RuntimeError('Insufficient build disk space; refusing global prune')
            options = copy.deepcopy(cfg)
            options['images']['cleanup'] = False
            manager = ImageLeaseManager(options, client)
            tag = 'dycon/webarena-wikipedia-amd64:' + selection['revision'][:12]
            progress = dict(status='building', started_at=now(), spec_sha256=sha256(derived / 'build-spec.json'))
            write_json(build_state / 'progress.json', progress)
            with manager.lease(base_store), build_proxy(build_cfg, build_state):
                client.images.get(base_store.image_id).tag(base)
                command = ['buildx', 'build', '--builder', cfg['skillsbench_images']['builder_name'],
                    '--load', '--provenance=false', '--platform', cfg['environment']['platform'],
                    '--tag', tag, '--file', str(source / relative),
                    '--label', 'org.dycon.webarena.revision=' + selection['revision']]
                for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
                    command.extend(['--build-arg', name + '=' + cfg['skillsbench_images']['build_proxy']])
                command.append(str(source))
                _command(build_cfg, command, log=build_state / 'build.log', timeout=cfg['skillsbench_images']['build_seconds'])
                image = client.images.get(tag)
                if image.attrs.get('Architecture') != 'amd64' or image.attrs.get('Os') != 'linux':
                    raise ValueError('Built Wikipedia image is not linux/amd64')
                import_archive(cfg, [tag], derived / 'blobs/sha256')
                provenance = dict(**spec, site='wikipedia', base_image=base_reference,
                    build_spec_sha256=sha256(derived / 'build-spec.json'),
                    official_dockerfile_unchanged=True,
                    reason='Official Wikipedia registry tags are ARM64-only; rebuild for linux/amd64')
                reference = publish_image(cfg, derived, 'wikipedia-amd64', tag, image, provenance)
                NasImageStore(cfg, reference).verify(deep=True)
                _immutable(store / 'overrides/wikipedia.json', reference)
                progress.update(status='complete', finished_at=now(), image=reference,
                                docker_image_id=image.id, official_dockerfile_sha256=hashes[relative])
                write_json(build_state / 'progress.json', progress)
                return progress
        finally:
            client.close()
