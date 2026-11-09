import copy
import subprocess
from pathlib import Path

import pytest

from catalog.skillsbench import package_manifest, verify_package
from catalog.skillsbench_index import POLICY, RULES, publish_variant, verify_adaptation
from core.io import hash_json, read_json, sha256, write_json


def source_snapshot(tmp_path):
    project = Path(__file__).parents[1]
    tasks = {}
    for iid, rules in RULES.items():
        target = tmp_path / 'benchmarks/skillsbench/tasks' / iid
        for name in rules:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project / 'benchmarks/skillsbench/tasks' / iid / name).read_bytes())
        files = package_manifest(target)
        task = {'benchmark': 'skillsbench', 'revision': 'a' * 40, 'instance_id': iid,
                'partition': 'train', 'instance': {'instance_id': iid, 'task_path': str(target),
                'package_sha256': hash_json(files), 'files': files, 'metadata': {}, 'images': []}}
        tasks[iid] = task
    source = tmp_path / 'source.json'
    write_json(source, {'schema_version': 1, 'benchmark': 'skillsbench', 'revision': 'a' * 40,
                        'tasks': tasks})
    return source, tasks


def test_variant_pins_both_identities_and_keeps_torch_and_pins(tmp_path):
    source, originals = source_snapshot(tmp_path)
    before = source.read_bytes()
    result = publish_variant({'project_root': str(tmp_path)}, 'adapted', source)
    manifest = read_json(tmp_path / result['snapshot_ref'])
    assert result['adapted_tasks'] == 3 and manifest['verifier_index_compatibility'] == POLICY
    assert source.read_bytes() == before
    for iid, task in manifest['tasks'].items():
        verify_package(task)
        adaptation = task['instance']['verifier_index_adaptation']
        assert adaptation['upstream_package_sha256'] == originals[iid]['instance']['package_sha256']
        assert adaptation['upstream_package_sha256'] != task['instance']['package_sha256']
        receipt = read_json(adaptation['receipt_path'])
        assert len(receipt['patches']) == len(RULES[iid])
        for patch in receipt['patches']:
            assert len(patch['original_sha256']) == 64
    simpo = Path(manifest['tasks']['simpo-code-reproduction']['instance']['task_path']) / 'verifier/test.sh'
    text = simpo.read_text()
    assert '--index-url https://download.pytorch.org/whl/cpu' in text
    assert '--with torch==2.1.2+cpu' in text
    assert '--extra-index-url "${DYCON_VERIFIER_PYTHON_INDEX:-https://pypi.org/simple}"' in text
    with pytest.raises(FileExistsError):
        publish_variant({'project_root': str(tmp_path)}, 'adapted', source)
    task = manifest['tasks']['fix-build-agentops']
    member = Path(task['instance']['task_path']) / 'verifier/test_outputs.py'
    member.write_text(member.read_text() + '\nassert True\n')
    with pytest.raises(ValueError, match='modified scoring'):
        verify_adaptation(task)


def test_rule_hash_mismatch_refuses_before_publication(tmp_path):
    source, originals = source_snapshot(tmp_path)
    path = Path(originals['fix-build-agentops']['instance']['task_path']) / 'verifier/run_passed.sh'
    path.write_text(path.read_text() + '\n# changed upstream\n')
    data = read_json(source)
    files = package_manifest(path.parent.parent)
    data['tasks']['fix-build-agentops']['instance'].update(files=files, package_sha256=hash_json(files))
    write_json(source, data)
    with pytest.raises(ValueError, match='source hash/location mismatch'):
        publish_variant({'project_root': str(tmp_path)}, 'refused', source)
    assert not (tmp_path / 'catalog/task-variants/refused').exists()


def test_cpu_first_policy_preserves_old_variants_and_all_non_index_bytes(tmp_path):
    from catalog.skillsbench_index import CPU_FIRST_POLICY
    source, originals = source_snapshot(tmp_path)
    old = publish_variant({'project_root': str(tmp_path)}, 'v1', source)
    old_bytes = (tmp_path / old['snapshot_ref']).read_bytes()
    new = publish_variant({'project_root': str(tmp_path)}, 'v2', source, CPU_FIRST_POLICY)
    assert (tmp_path / old['snapshot_ref']).read_bytes() == old_bytes
    manifest = read_json(tmp_path / new['snapshot_ref'])
    for task in manifest['tasks'].values():
        verify_package(task)
    simpo = manifest['tasks']['simpo-code-reproduction']
    before = (Path(originals['simpo-code-reproduction']['instance']['task_path']) / 'verifier/test.sh').read_text()
    after = (Path(simpo['instance']['task_path']) / 'verifier/test.sh').read_text()
    assert after == before.replace(
        '--index-url https://download.pytorch.org/whl/cpu',
        '--index-url "${DYCON_VERIFIER_PYTHON_INDEX:-https://pypi.org/simple}"').replace(
        '--extra-index-url https://pypi.org/simple',
        '--extra-index-url https://download.pytorch.org/whl/cpu')


@pytest.mark.parametrize('iid', ['fix-build-agentops', 'fix-build-google-auto'])
def test_python_subprocess_assignment_uses_explicit_index_and_keeps_default(iid):
    before, after = RULES[iid]['verifier/test_outputs.py'][1:]
    for index in (None, 'https://mirror.test/simple/'):
        scope = {'build_env': {}, 'os': type('OS', (), {'environ': {} if index is None else
                 {'DYCON_VERIFIER_PYTHON_INDEX': index}})}
        exec(after.strip(), scope)
        assert scope['build_env']['PIP_INDEX_URL'] == (index or 'https://pypi.org/simple')


def test_shell_second_assignment_uses_explicit_index():
    import os
    line = RULES['fix-build-agentops']['verifier/run_passed.sh'][2]
    for index in (None, 'https://mirror.test/simple/'):
        env = {k: v for k, v in os.environ.items() if k != 'DYCON_VERIFIER_PYTHON_INDEX'}
        if index:
            env['DYCON_VERIFIER_PYTHON_INDEX'] = index
        result = subprocess.run(['bash', '-c', line + '\nprintf "%s" "$PIP_INDEX_URL"'],
                                env=env, capture_output=True, text=True, timeout=5, check=True)
        assert result.stdout == (index or 'https://pypi.org/simple')
