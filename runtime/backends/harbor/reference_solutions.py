"""Explicit versioned oracle revisions; original tasks and verifier bytes stay intact."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from application.execution.run_plan import load_plan
from catalog.harbor import verify_package
from catalog.skillsbench import package_manifest
from core.io import hash_json, identifier, read_json, sha256, write_json


def publish_reference(root, run_id, task_id, reference_id, source, reason):
    root = Path(root).resolve()
    cfg = load_plan(root, run_id)['config']
    if cfg['benchmark']['backend'] != 'terminalbench_2_1' or not reason.strip():
        raise ValueError('Reference publication requires Terminal-Bench and an explicit reason')
    tasks = [t for t in cfg['selection']['tasks'].values() if t['instance_id'] == task_id]
    if len(tasks) != 1:
        raise ValueError('Select exactly one frozen task')
    task = tasks[0]
    verify_package(task)
    source = Path(source).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError('Expected a reference solve.sh file')
    target = root / 'catalog/reference-solutions' / identifier(reference_id)
    if target.exists():
        raise FileExistsError('Reference publications are immutable')
    package = target / 'packages' / identifier(task_id)
    shutil.copytree(task['instance']['task_path'], package)
    shutil.copyfile(source, package / 'solution/solve.sh')
    derived = copy.deepcopy(task)
    derived['instance']['task_path'] = str(package)
    derived['instance']['files'] = package_manifest(package)
    derived['instance']['package_sha256'] = hash_json(derived['instance']['files'])
    before = {r['path']: r['sha256'] for r in task['instance']['files']}
    after = {r['path']: r['sha256'] for r in derived['instance']['files']}
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed != ['solution/solve.sh']:
        raise ValueError('Reference revision must change only solution/solve.sh')
    verify_package(derived)
    manifest = {'schema_version': 1, 'scope': 'derived reference solution; original verifier',
                'source_run': run_id, 'source_task': task_id,
                'source_package_sha256': task['instance']['package_sha256'],
                'source_solution_sha256': before['solution/solve.sh'],
                'solution_sha256': after['solution/solve.sh'], 'reason': reason,
                'changed_files': changed, 'task': derived}
    manifest['manifest_sha256'] = hash_json(manifest)
    path = target / 'manifest.json'
    write_json(path, manifest)
    return {'reference_ref': str(path.relative_to(root)), 'manifest_sha256': manifest['manifest_sha256']}


def reference_task(root, ref, official):
    root = Path(root).resolve()
    path = (root / ref).resolve()
    if not path.is_relative_to(root / 'catalog/reference-solutions') or path.name != 'manifest.json':
        raise ValueError('Expected a published reference solution manifest')
    item = read_json(path)
    if item['manifest_sha256'] != hash_json({k: v for k, v in item.items() if k != 'manifest_sha256'}):
        raise ValueError('Reference publication hash changed')
    verify_package(official)
    if (item['source_task'] != official['instance_id'] or
            item['source_package_sha256'] != official['instance']['package_sha256']):
        raise ValueError('Reference source differs from frozen task')
    derived = item['task']
    if Path(derived['instance']['task_path']).resolve() != path.parent / 'packages' / identifier(official['instance_id']):
        raise ValueError('Derived task is outside its reference publication')
    verify_package(derived)
    expected = copy.deepcopy(official)
    expected['instance'].update({k: derived['instance'][k] for k in ('task_path', 'files', 'package_sha256')})
    if expected != derived:
        raise ValueError('Reference changes task identity or configuration')
    before = {r['path']: r['sha256'] for r in official['instance']['files']}
    after = {r['path']: r['sha256'] for r in derived['instance']['files']}
    if sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)) != ['solution/solve.sh']:
        raise ValueError('Reference changes files outside solution/solve.sh')
    if (sha256(Path(derived['instance']['task_path']) / 'solution/solve.sh') != item['solution_sha256']
            or before['solution/solve.sh'] != item['source_solution_sha256']):
        raise ValueError('Reference solution hash differs')
    return derived, {'path': str(path.relative_to(root)), 'sha256': sha256(path),
                     'manifest_sha256': item['manifest_sha256'], 'scope': item['scope']}
