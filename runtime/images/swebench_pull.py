"""Resume a frozen official Verified or dev image completion plan into a new NAS store.

Reuses the registry transport and OCI publication format of benchmark-pull.
This prepares images only: task/catalog registration is a separate operation.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import shutil
import threading
import time
from pathlib import Path

from catalog.benchmark import _repository, _write_image_record
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.oci import descriptor_graph
from runtime.images.registry import image_metadata, transfer_blob

SOURCES = {
    ('SWE-bench/SWE-bench_Verified', '78f471bf655a3137b2e8a75af1501690ec009ec3', 'test'):
        (500, 'a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9'),
    ('SWE-bench/SWE-bench', 'c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2', 'dev'):
        (225, '4937dd07813f8749c2a010b0ef6d964190b01730c6549a9a9c33f8bc339e453d'),
}


def _validate_plan(plan, nas):
    source = plan['source']
    images = plan['official_images']
    digest = hashlib.sha256(('\n'.join(sorted(images)) + '\n').encode()).hexdigest()
    expected = SOURCES.get((source.get('dataset'), source.get('revision'), source.get('split')))
    if plan.get('schema_version') != 1 or expected != (len(images), digest):
        raise ValueError('Plan must describe the pinned official Verified-500 or dev-225 ID set')
    missing = plan['missing_ids']
    existing = plan['existing']
    if (len(missing) != len(set(missing)) or set(missing) & set(existing)
            or set(missing) | set(existing) != set(images)):
        raise ValueError('Existing and missing IDs must partition the complete official source split')
    for iid, image in images.items():
        identifier(iid)
        repository, reference = _repository(image)
        if not repository.startswith('swebench/sweb.eval.x86_64.') or reference != 'latest':
            raise ValueError('Expected official SWE-bench x86_64 image names')
    for item in existing.values():
        store, path = Path(item['store']).resolve(), Path(item['path']).resolve()
        if not store.is_relative_to(nas) or not path.is_relative_to(store / 'images'):
            raise ValueError('Existing image record escapes configured NAS namespace')
        if sha256(path) != item['manifest_sha256']:
            raise ValueError(f'Existing image record changed after inventory: {path}')
        record = read_json(path)
        descriptor_graph(store / 'blobs/sha256', {
            'digest': record['manifest_digest'], 'size': record['manifest_size']})


class _Transfer:
    def __init__(self, store, seed, proxy):
        self.store, self.seed, self.proxy = store, seed, proxy
        self.blobs = store / 'blobs/sha256'
        self.guard = threading.Lock()
        self.locks = {}
        self.verified = {}
        self.received = 0
        self.last_progress = 0.0

    def progress(self, size):
        with self.guard:
            self.received += size
            data = {'pid': os.getpid(), 'updated_at': now(), 'received_bytes': self.received}
            started = self.store / 'dataset/transfer-started.json'
            if not started.exists():
                write_json(started, data)
                print(f'TRANSFER_STARTED received_bytes={self.received}', flush=True)
            if time.monotonic() - self.last_progress >= 30:
                write_json(self.store / 'dataset/transfer-progress.json', data)
                self.last_progress = time.monotonic()

    def blob(self, item):
        with self.guard:
            lock = self.locks.setdefault(item['digest'], threading.Lock())
        with lock:
            if item['digest'] in self.verified:
                if self.verified[item['digest']] != item['size']:
                    raise ValueError('Conflicting OCI descriptor sizes')
                return
            transfer_blob(item, self.blobs, self.seed, self.proxy, self.progress)
            self.verified[item['digest']] = item['size']

    def image(self, iid, source_image):
        # Freeze the first successfully resolved root digest; resume never re-resolves latest.
        path = self.store / 'dataset/metadata' / f'{iid}.json'
        repository, reference = _repository(source_image)
        if path.exists():
            metadata = read_json(path)
            if metadata['source_image'] != source_image:
                raise ValueError('Frozen image source differs from plan')
            root, descriptors = metadata['root'], metadata['descriptors']
        else:
            root, descriptors = image_metadata(repository, self.blobs, reference=reference, proxy=self.proxy)
            write_json(path, {'source_image': source_image, 'root': root, 'descriptors': descriptors})
        missing_bytes = sum(d['size'] for d in descriptors if not (self.blobs / d['digest'][7:]).exists())
        if shutil.disk_usage(self.store).free < missing_bytes + 100 * 1024**3:
            raise BufferError('Insufficient NAS space plus 100 GiB reserve')
        for item in descriptors:
            self.blob(item)
        # No public image record exists until its complete graph has passed SHA checks.
        descriptor_graph(self.blobs, root)
        record_path = self.store / 'images' / f'{iid}.json'
        if record_path.exists():
            previous = read_json(record_path)
            if previous['manifest_digest'] != root['digest']:
                raise ValueError('Published image digest differs from frozen metadata')
            descriptor_graph(self.blobs, root)
            return
        record = _write_image_record(record_path, self.blobs, source_image, repository, root, descriptors)
        record.update(instance_id=iid, image_id=root['digest'], deep_verified=True)
        write_json(record_path, record)


def pull(cfg, store_id, *, plan_path, proxy=None, workers=4, attempts=3, check_only=False):
    identifier(store_id)
    if not 1 <= workers <= 8 or not 1 <= attempts <= 5:
        raise ValueError('Expected 1..8 workers and 1..5 task attempts')
    nas = Path(cfg['paths']['nas_root']).resolve()
    if not any(p != Path('/') and os.path.ismount(p) for p in (nas, *nas.parents)):
        raise RuntimeError('Configured NAS namespace is not mounted')
    plan = read_json(plan_path)
    _validate_plan(plan, nas)
    store = nas / store_id
    if store.is_symlink():
        raise ValueError('Image completion store must not be a symlink')
    summary = {'official_total': len(plan['official_images']), 'existing': len(plan['existing']),
               'to_download': len(plan['missing_ids']), 'store': str(store)}
    if check_only:
        return {**summary, 'checked': True}
    store.mkdir(parents=True, exist_ok=True)
    with file_lock(store / 'dataset/pull.lock', nonblocking=True):
        for relative in ('blobs/sha256', 'images', 'dataset/metadata', 'dataset/attempts'):
            (store / relative).mkdir(parents=True, exist_ok=True)
        frozen = store / 'dataset/plan.json'
        if frozen.exists() and hash_json(read_json(frozen)) != hash_json(plan):
            raise ValueError('Store is bound to a different immutable plan')
        if not frozen.exists():
            write_json(frozen, plan)
        seed = [p / 'blobs/sha256' for p in nas.iterdir()
                if p != store and p.is_dir() and (p / 'blobs/sha256').is_dir()]
        transfer = _Transfer(store, seed, proxy)
        status = {**summary, 'pid': os.getpid(), 'started_at': now(), 'state': 'running',
                  'completed_ids': [], 'errors': {}, 'plan_sha256': sha256(frozen)}
        write_json(store / 'dataset/download-status.json', status)
        print(f'PULL_STARTED existing={summary["existing"]} missing={summary["to_download"]} workers={workers}', flush=True)
        pending = list(plan['missing_ids'])
        completed = set()
        try:
            for attempt in range(1, attempts + 1):
                errors = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(transfer.image, iid, plan['official_images'][iid]): iid for iid in pending}
                    for future in concurrent.futures.as_completed(futures):
                        iid = futures[future]
                        try:
                            future.result()
                            completed.add(iid)
                            print(f'IMAGE_COMPLETE {iid} {len(completed)}/{len(plan["missing_ids"])}', flush=True)
                        except Exception as exc:
                            errors[iid] = repr(exc)
                            print(f'IMAGE_ERROR attempt={attempt} {iid} {exc!r}', flush=True)
                        status.update(attempt=attempt, updated_at=now(), completed_ids=sorted(completed), errors=errors)
                        write_json(store / 'dataset/download-status.json', status)
                write_json(store / 'dataset/attempts' / f'{time.time_ns()}.json', {
                    'attempt': attempt, 'completed_ids': sorted(completed), 'errors': errors, 'finished_at': now()})
                if not errors:
                    break
                pending = sorted(errors)
                if attempt < attempts:
                    time.sleep(60 * attempt)  # Retry backoff, not a progress/status polling loop.
            if errors:
                raise RuntimeError(f'{len(errors)} image(s) failed after {attempts} task attempts; resume same command')
            # Recheck prior references at completion; never claim coverage from a stale inventory.
            _validate_plan(plan, nas)
            coverage = {iid: dict(item) for iid, item in plan['existing'].items()}
            for iid in plan['missing_ids']:
                path = store / 'images' / f'{iid}.json'
                record = read_json(path)
                descriptor_graph(transfer.blobs, {'digest': record['manifest_digest'], 'size': record['manifest_size']})
                coverage[iid] = {'store': str(store), 'path': str(path), 'manifest_sha256': sha256(path)}
            write_json(store / 'dataset/coverage.json', {
                'source': plan['source'], 'covered_count': len(coverage), 'images': coverage,
                'new_images_sha256_verified': True, 'previous_images_check': 'record hash and graph existence/size',
                'runtime_gold_validated': False, 'completed_at': now()})
            status.update(state='complete', completed_at=now(), official_coverage=len(coverage), errors={})
        except BaseException as exc:
            status.update(state='failed', failed_at=now(), error=repr(exc))
            raise
        finally:
            write_json(store / 'dataset/download-status.json', status)
        write_json(store / 'dataset/completion.json', status)
        print(f'IMAGES_COMPLETE coverage={len(coverage)}', flush=True)
        return {**summary, 'state': 'complete', 'official_coverage': len(coverage)}
