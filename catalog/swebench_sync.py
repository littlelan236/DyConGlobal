"""Transactionally reconcile NAS-backed SWE-bench tasks into official groups."""
from __future__ import annotations

import collections
import copy
import hashlib
from pathlib import Path

from catalog.store import Catalog, Manifest, Task
from core.backends import benchmark_family
from core.io import hash_json, now, read_json, sha256, write_json
from runtime.images.nas import NasImageStore

OFFICIAL_GROUPS = {name: 'swebench_' + name for name in ('verified', 'dev', 'nonverified')}
FAILED_GROUPS = {name: name + '-gold-patch-failed' for name in OFFICIAL_GROUPS}
GROUPS = {**OFFICIAL_GROUPS, **{name: name for name in FAILED_GROUPS.values()}}
SOURCES = {
    'verified': ('SWE-bench/SWE-bench_Verified', '78f471bf655a3137b2e8a75af1501690ec009ec3', 'test',
                 500, 'a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9'),
    'dev': ('SWE-bench/SWE-bench', 'c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2', 'dev',
            225, '4937dd07813f8749c2a010b0ef6d964190b01730c6549a9a9c33f8bc339e453d'),
    'test': ('SWE-bench/SWE-bench', 'c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2', 'test',
             2294, 'ddfea339bcfbdd66f03b4dd27f0a31170e71a2169f7e73c9bbae5b5f2f29f84b'),
}


def load_sources(path):
    source = read_json(path)
    sets = {}
    if source.get('schema_version') != 1:
        raise ValueError('Unsupported SWE source schema')
    for group, expected in SOURCES.items():
        data = source['sources'][group]
        ids = data['ids']
        digest = hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()
        actual = (data['dataset'], data['revision'], data['split'], len(ids), digest)
        if actual != expected or len(set(ids)) != len(ids):
            raise ValueError(f'Official source identity mismatch: {group}')
        sets[group] = set(ids)
    if not sets['verified'] <= sets['test'] or sets['test'] & sets['dev']:
        raise ValueError('Official split relationships disagree')
    rows = {}
    for group in ('verified', 'dev'):
        values = source['rows'][group]
        mapped = {row['instance_id']: row for row in values}
        if len(mapped) != len(values) or set(mapped) != sets[group]:
            raise ValueError('Source rows must match official ID set: ' + group)
        rows.update(mapped)
    return source, sets, rows


def category(iid, sets):
    if iid in sets['verified']:
        return 'verified'
    if iid in sets['dev']:
        return 'dev'
    if iid in sets['test']:
        return 'nonverified'
    raise ValueError('Task is outside pinned official SWE-bench test/dev: ' + iid)


def _reference(cfg, store_path, record_path, expected_hash=None):
    nas = Path(cfg['paths']['nas_root']).resolve()
    store, path = Path(store_path).resolve(), Path(record_path).resolve()
    if not store.is_relative_to(nas) or not path.is_relative_to(store / 'images'):
        raise ValueError('NAS reference escapes configured image namespace')
    checksum = sha256(path)
    if expected_hash and checksum != expected_hash:
        raise ValueError('NAS record changed after frozen image plan: ' + str(path))
    record = read_json(path)
    reference = {'store': str(store.relative_to(nas)), 'manifest': str(path.relative_to(store)),
                 'manifest_sha256': checksum, 'image_id': record['manifest_digest']}
    NasImageStore(cfg, reference).verify(deep=False)
    return reference


def _plan_references(cfg, paths, source, sets):
    references = {}
    supplied = set()
    nas = Path(cfg['paths']['nas_root']).resolve()
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_relative_to(nas) or path.parent.name != 'dataset' or path.name != 'plan.json':
            raise ValueError('Image plan must be a NAS store dataset/plan.json')
        plan = read_json(path)
        matches = [g for g in ('verified', 'dev') if all(
            plan['source'][key] == source['sources'][g][key] for key in ('dataset', 'revision', 'split'))]
        if len(matches) != 1 or matches[0] in supplied:
            raise ValueError('Provide exactly one compatible image plan per source')
        group = matches[0]
        expected_images = {r['instance_id']: r['image'] for r in source['rows'][group]}
        if plan['official_images'] != expected_images:
            raise ValueError('Image plan image names disagree with pinned official source rows')
        supplied.add(group)
        existing, missing = plan['existing'], plan['missing_ids']
        if (set(existing) & set(missing) or len(missing) != len(set(missing))
                or set(existing) | set(missing) != sets[group]
                or set(plan['official_images']) != sets[group]):
            raise ValueError('Image plan does not partition its official source')
        for iid, item in existing.items():
            references[iid] = _reference(cfg, item['store'], item['path'], item['manifest_sha256'])
        # Only public, complete image records count; metadata/part files do not.
        store = path.parent.parent
        for iid in missing:
            record_path = store / 'images' / (iid + '.json')
            if record_path.is_file():
                references[iid] = _reference(cfg, store, record_path)
    if supplied != {'verified', 'dev'}:
        raise ValueError('Both Verified and dev image plans are required')
    return references


def build_tasks(cfg, before, source, sets, rows, references):
    tasks, old_by_id = {}, {}
    for key, task in before['tasks'].items():
        if benchmark_family(task['benchmark']) != 'swebench' or task['benchmark'] == 'fixture':
            tasks[key] = copy.deepcopy(task)
            continue
        iid = task['instance_id']
        category(iid, sets)
        if iid in old_by_id:
            raise ValueError('Cannot merge ambiguous existing SWE task revisions: ' + iid)
        old_by_id[iid] = (key, copy.deepcopy(task))
    mappings = []
    for iid in sorted(set(old_by_id) | set(references)):
        group = category(iid, sets)
        if iid in old_by_id:
            old_key, task = old_by_id[iid]
            # Preserve task text, revision, partition, all environment overrides and image pins.
            NasImageStore(cfg, task['instance']['nas_image']).verify(deep=False)
        else:
            old_key = None
            row = copy.deepcopy(rows[iid])
            row['image_name'] = row.get('image', '')
            row['nas_image'] = references[iid]
            row.setdefault('environment_overrides', {})
            task = {'benchmark': GROUPS[group], 'revision': 'hf-' + source['sources'][group]['revision'],
                    'instance_id': iid, 'repo': row['repo'], 'base_commit': row['base_commit'],
                    'partition': 'train', 'exclusion_reason': None, 'instance': row}
        # Keep explicit Gold Patch failure groups when refreshing official sources.
        if task['benchmark'] not in FAILED_GROUPS.values():
            task['benchmark'] = OFFICIAL_GROUPS[group]
        elif task['benchmark'] != FAILED_GROUPS[group]:
            raise ValueError('Gold Patch failure group disagrees with official source: ' + iid)
        task = Task.model_validate(task).model_dump()
        key = hash_json({field: task[field] for field in ('benchmark', 'revision', 'instance_id')})
        if key in tasks:
            raise ValueError('Task key collision')
        tasks[key] = task
        mappings.append({'instance_id': iid, 'group': group, 'old_key': old_key, 'new_key': key})
    counts = collections.Counter(t['benchmark'] for t in tasks.values())
    missing = {group: sorted(sets[group] - {t['instance_id'] for t in tasks.values()
                                         if t['benchmark'] in {OFFICIAL_GROUPS[group], FAILED_GROUPS[group]}})
               for group in ('verified', 'dev')}
    return tasks, mappings, dict(counts), missing


def sync(cfg, *, source_path, image_plans, output, apply=False):
    catalog = Catalog(cfg['project_root'])
    before = catalog.read()
    before_file_hash = sha256(catalog.path)
    source, sets, rows = load_sources(source_path)
    references = _plan_references(cfg, image_plans, source, sets)
    tasks, mappings, counts, missing = build_tasks(cfg, before, source, sets, rows, references)
    candidate = {**before, 'tasks': tasks}
    Manifest.model_validate(candidate)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    changed = tasks != before['tasks']
    report = {'schema_version': 1, 'created_at': now(), 'applied': False, 'changed': changed,
              'before_catalog_revision': before['catalog_revision'], 'before_file_sha256': before_file_hash,
              'counts': counts, 'swebench_counts': {g: counts.get(n, 0) for g, n in GROUPS.items()},
              'added': sum(x['old_key'] is None for x in mappings),
              'rekeyed': sum(x['old_key'] is not None and x['old_key'] != x['new_key'] for x in mappings),
              'missing_official_ids': missing, 'source_sha256': sha256(source_path),
              'image_plans': {str(p): sha256(p) for p in image_plans},
              'image_validation': 'manifest SHA256 and full graph existence/size; no Docker/gold',
              'new_task_partition': 'train (project partition, separate from official split)'}
    write_json(output / 'task-key-mapping.json', mappings)
    write_json(output / 'report.json', report)
    if apply:
        with catalog.transaction() as data:
            if hash_json(data) != hash_json(before) or sha256(catalog.path) != before_file_hash:
                raise RuntimeError('Catalog changed during preparation; retry from current catalog')
            (output / 'catalog-before.json').write_bytes(catalog.path.read_bytes())
            write_json(output / 'official-source.json', source)
            data['tasks'] = tasks
        after = catalog.read()
        report.update(applied=True, after_catalog_revision=after['catalog_revision'],
                      after_file_sha256=sha256(catalog.path), completed_at=now())
        write_json(output / 'catalog-after.json', after)
        write_json(output / 'report.json', report)
    return {k: v for k, v in report.items() if k != 'missing_official_ids'} | {
        'missing_counts': {g: len(ids) for g, ids in missing.items()}, 'report': str(output / 'report.json')}


def list_tasks(cfg, *, group=None, output=None):
    catalog = Catalog(cfg['project_root'])
    selection = catalog.select_tasks(benchmark='swebench')
    tasks = selection['tasks']
    counts = {g: sum(t['benchmark'] == name for t in tasks.values()) for g, name in GROUPS.items()}
    if group:
        tasks = {k: t for k, t in tasks.items() if t['benchmark'] == GROUPS[group]}
    ids = sorted(t['instance_id'] for t in tasks.values())
    if len(ids) != len(set(ids)):
        raise ValueError('Ambiguous IDs across SWE catalog revisions')
    if output:
        path = Path(output)
        if path.exists():
            raise FileExistsError('Refusing to overwrite existing selection file')
        write_json(path, {'tasks': ids, 'catalog_revision': selection['catalog_revision'],
                          'catalog_sha256': selection['catalog_sha256'], 'group': group})
    return {'catalog_revision': selection['catalog_revision'], 'groups': counts, 'selected': len(ids),
            'instance_ids_sha256': hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest(),
            'output': str(output) if output else None}
