"""Immutable, hash-guarded generic PyPI adaptations for pinned verifier files."""
from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path

from core.io import hash_json, identifier, read_json, sha256, write_json

POLICY = 'generic-pypi-v1'
# These rules bind the exact upstream bytes, including both AgentOps layers.
RULES = {
    'fix-build-agentops': {
        'verifier/test_outputs.py': (
            '176f3fb3e55bb85a2ae886df604ea2788ada0a9dd69c0a458ed6c4a490275170',
            '    build_env["PIP_INDEX_URL"] = "https://pypi.org/simple"',
            '    build_env["PIP_INDEX_URL"] = os.environ.get("DYCON_VERIFIER_PYTHON_INDEX", "https://pypi.org/simple")'),
        'verifier/run_passed.sh': (
            '2c38caede709c2b03a17a412f242656fd63c563a3c6e5ae52bf129a69a592783',
            'export PIP_INDEX_URL=https://pypi.org/simple',
            'export PIP_INDEX_URL="${DYCON_VERIFIER_PYTHON_INDEX:-https://pypi.org/simple}"'),
    },
    'fix-build-google-auto': {
        'verifier/test_outputs.py': (
            'f8bd36c59867da8e5073979e263d4fea12ead8ce5a247674a8ba153f871ef685',
            '    build_env["PIP_INDEX_URL"] = "https://pypi.org/simple"',
            '    build_env["PIP_INDEX_URL"] = os.environ.get("DYCON_VERIFIER_PYTHON_INDEX", "https://pypi.org/simple")'),
    },
    'simpo-code-reproduction': {
        'verifier/test.sh': (
            'f928e6fe316f52fd90700fe9e26a290047638269fe8436409e2ffdfe242bf475',
            '    --extra-index-url https://pypi.org/simple \\',
            '    --extra-index-url "${DYCON_VERIFIER_PYTHON_INDEX:-https://pypi.org/simple}" \\'),
    },
}

CPU_FIRST_POLICY = 'generic-pypi-cpu-first-v2'
CPU_FIRST_RULES = copy.deepcopy(RULES)
CPU_FIRST_RULES['simpo-code-reproduction']['verifier/test.sh'] = (
    RULES['simpo-code-reproduction']['verifier/test.sh'][0],
    '    --index-url https://download.pytorch.org/whl/cpu \\\n    --extra-index-url https://pypi.org/simple \\',
    '    --index-url "${DYCON_VERIFIER_PYTHON_INDEX:-https://pypi.org/simple}" \\\n    --extra-index-url https://download.pytorch.org/whl/cpu \\')
POLICIES = {POLICY: RULES, CPU_FIRST_POLICY: CPU_FIRST_RULES}


def scan_task(task):
    """Classify bounded source lines; this is evidence, never a rewrite rule."""
    root = Path(task['instance']['task_path'])
    rows = []
    pattern = re.compile(r'(?:PIP|UV)_(?:DEFAULT_|EXTRA_)?INDEX|--(?:extra-)?index-url|--no-index|PIP_NO_INDEX|UV_OFFLINE|pypi\.org|download\.pytorch\.org')
    for entry in task['instance']['files']:
        path = root / entry['path']
        if path.suffix not in {'.sh', '.py', '.toml', '.yaml', '.yml'} and path.name != 'Dockerfile':
            continue
        if path.stat().st_size > 2_000_000:
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if pattern.search(line):
                rows.append({'path': entry['path'], 'line': number,
                             'kind': ('specialized_index' if 'pytorch.org' in line else
                                      'offline' if 'no-index' in line or 'OFFLINE' in line else
                                      'generic_index'),
                             'line_sha256': __import__('hashlib').sha256(line.encode()).hexdigest(),
                             'allowlisted': entry['path'] in RULES.get(task['instance_id'], {})})
    return rows


def publish_variant(cfg, snapshot_id, source_ref, policy=POLICY):
    from catalog.skillsbench import package_manifest, verify_package
    if policy not in POLICIES:
        raise ValueError('Unknown verifier index policy')
    ruleset = POLICIES[policy]
    root = Path(cfg['project_root']).resolve()
    source_path = (root / source_ref).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError('Source snapshot must be project-owned')
    source = read_json(source_path)
    if source.get('benchmark') != 'skillsbench' or source.get('verifier_index_compatibility'):
        raise ValueError('Index variants require an unadapted SkillsBench snapshot')
    publication = root / 'catalog/task-variants' / identifier(snapshot_id)
    target = root / 'catalog/snapshots' / snapshot_id / 'manifest.json'
    if publication.exists() or target.exists():
        raise FileExistsError('Index variant publication is immutable; choose a new ID')
    # Validate every source and rule before creating the publication.
    for task in source['tasks'].values():
        verify_package(task)
        for name, (digest, before, after) in ruleset.get(task['instance_id'], {}).items():
            path = Path(task['instance']['task_path']) / name
            if sha256(path) != digest or path.read_text().count(before) != 1:
                raise ValueError('Verifier index rule source hash/location mismatch: ' + task['instance_id'] + '/' + name)
    result = copy.deepcopy(source)
    scan = {}
    for key, task in result['tasks'].items():
        scan[task['instance_id']] = scan_task(task)
        rules = ruleset.get(task['instance_id'], {})
        if not rules:
            continue
        original = copy.deepcopy(task['instance'])
        effective = publication / 'tasks' / task['instance_id']
        shutil.copytree(original['task_path'], effective)
        patches = []
        for name, (digest, before, after) in rules.items():
            path = effective / name
            text = path.read_text()
            line = text[:text.index(before)].count('\n') + 1
            path.write_text(text.replace(before, after, 1))
            patches.append({'path': name, 'line': line, 'original_sha256': digest,
                            'effective_sha256': sha256(path), 'before': before, 'after': after,
                            'reason': 'Read explicitly frozen generic verifier index; preserve scoring and dependency pins'})
        files = package_manifest(effective)
        receipt = {'schema_version': 1, 'policy': policy, 'task_id': task['instance_id'],
                   'source_snapshot_sha256': sha256(source_path), 'upstream': original,
                   'effective_task_path': str(effective), 'effective_files': files,
                   'effective_package_sha256': hash_json(files), 'patches': patches,
                   'verifier_identity': 'official-tests-with-generic-index-adaptation'}
        receipt_path = publication / 'receipts' / (task['instance_id'] + '.json')
        write_json(receipt_path, receipt)
        task['instance'].update(task_path=str(effective), files=files, package_sha256=hash_json(files),
            verifier_index_adaptation={'policy': policy, 'receipt_path': str(receipt_path),
                                      'receipt_sha256': sha256(receipt_path),
                                      'upstream_package_sha256': original['package_sha256']})
    result['verifier_index_compatibility'] = policy
    result['source_snapshot'] = {'path': str(source_path), 'sha256': sha256(source_path)}
    write_json(publication / 'index-scan.json', {'schema_version': 1, 'policy': policy, 'tasks': scan})
    result['index_scan'] = {'path': str(publication / 'index-scan.json'),
                            'sha256': sha256(publication / 'index-scan.json')}
    write_json(target, result)
    return {'snapshot_ref': str(target.relative_to(root)), 'snapshot_sha256': sha256(target),
            'tasks': len(result['tasks']), 'adapted_tasks': sum(bool(t['instance'].get('verifier_index_adaptation'))
                                                            for t in result['tasks'].values()),
            'policy': policy, 'index_scan': result['index_scan']}


def verify_adaptation(task):
    from catalog.skillsbench import package_manifest
    payload = task['instance']
    adaptation = payload.get('verifier_index_adaptation')
    if not adaptation:
        return
    receipt_path = Path(adaptation['receipt_path'])
    if sha256(receipt_path) != adaptation['receipt_sha256']:
        raise ValueError('Verifier adaptation receipt changed')
    receipt = read_json(receipt_path)
    policy = adaptation['policy']
    if (policy not in POLICIES or receipt['policy'] != policy
            or receipt['task_id'] != task['instance_id']
            or receipt['effective_task_path'] != payload['task_path']
            or receipt['effective_files'] != payload['files']
            or receipt['effective_package_sha256'] != payload['package_sha256']
            or receipt['upstream']['package_sha256'] != adaptation['upstream_package_sha256']):
        raise ValueError('Verifier adaptation identity mismatch')
    upstream = receipt['upstream']
    if hash_json(upstream['files']) != upstream['package_sha256']:
        raise ValueError('Verifier adaptation upstream identity changed')
    if package_manifest(upstream['task_path']) != upstream['files']:
        raise ValueError('Verifier adaptation upstream package changed')
    expected = copy.deepcopy(upstream['files'])
    patches = receipt['patches']
    ruleset = POLICIES[policy]
    if {patch['path'] for patch in patches} != set(ruleset.get(task['instance_id'], {})):
        raise ValueError('Verifier adaptation does not cover its exact allowlist')
    for patch in patches:
        digest, before, after = ruleset[task['instance_id']][patch['path']]
        if (patch['original_sha256'], patch['before'], patch['after']) != (digest, before, after):
            raise ValueError('Verifier adaptation is outside the allowlist')
        source = (Path(upstream['task_path']) / patch['path']).read_text()
        effective = (Path(payload['task_path']) / patch['path']).read_text()
        if (sha256(Path(upstream['task_path']) / patch['path']) != digest
                or source.count(before) != 1 or effective != source.replace(before, after, 1)
                or sha256(Path(payload['task_path']) / patch['path']) != patch['effective_sha256']):
            raise ValueError('Verifier adaptation modified scoring or unapproved content')
        next(row for row in expected if row['path'] == patch['path'])['sha256'] = patch['effective_sha256']
    if expected != payload['files']:
        raise ValueError('Verifier adaptation changed files outside the allowlist')


def task_package_path(cfg, task):
    """Authorize official packages or receipt-bound project variant packages."""
    path = Path(task['instance']['task_path']).resolve()
    official = (Path(cfg['project_root']) / cfg['skillsbench_images']['source_root'] / 'tasks').resolve()
    adaptation = task['instance'].get('verifier_index_adaptation')
    if adaptation:
        policy = cfg['execution_context']['profile'].get('verifier_index_compatibility')
        variants = (Path(cfg['project_root']) / 'catalog/task-variants').resolve()
        if (policy != adaptation['policy'] or not path.is_relative_to(variants)
                or path.name != task['instance_id']):
            raise ValueError('Task index variant is not selected by the frozen profile')
        verify_adaptation(task)
    elif path.parent != official:
        raise ValueError('Task package escaped the pinned official task root')
    return path
