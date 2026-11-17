import copy
import json
import subprocess
from pathlib import Path

import pytest

from core.io import hash_json, write_json
from catalog.skillsbench import package_manifest
from runtime.backends.harbor.reference_solutions import publish_reference, reference_task
from runtime.backends.harbor.resource_store import public_stateless_response


def test_reference_revision_changes_solution_only_and_rejects_tampering(tmp_path, monkeypatch):
    package = tmp_path / 'original'
    (package / 'solution').mkdir(parents=True)
    (package / 'tests').mkdir()
    (package / 'solution/solve.sh').write_text('#!/bin/sh\nfalse\n')
    (package / 'tests/test.sh').write_text('original tests')
    task = {'benchmark': 'terminalbench_2_1', 'revision': '7131e4375048a0e408a8fb404b5f499d726b695b',
            'instance_id': 'task', 'instance': {'instance_id': 'task', 'task_path': str(package),
            'files': package_manifest(package), 'metadata': {'schema_version': '1.1'},
            'instruction': 'original', 'images': {'agent': 'original-image'}}}
    task['instance']['package_sha256'] = hash_json(task['instance']['files'])
    cfg = {'benchmark': {'backend': 'terminalbench_2_1'}, 'selection': {'tasks': {'key': task}}}
    monkeypatch.setattr('runtime.backends.harbor.reference_solutions.load_plan', lambda *a: {'config': cfg})
    solve = tmp_path / 'fixed.sh'; solve.write_text('#!/bin/sh\ntrue\n')
    result = publish_reference(tmp_path, 'source-run', 'task', 'ref-v1', solve, 'compatibility fix')
    derived, receipt = reference_task(tmp_path, result['reference_ref'], task)
    assert receipt['scope'] == 'derived reference solution; original verifier'
    assert (package / 'solution/solve.sh').read_text().endswith('false\n')
    assert (Path(derived['instance']['task_path']) / 'tests/test.sh').read_text() == 'original tests'
    with pytest.raises(FileExistsError):
        publish_reference(tmp_path, 'source-run', 'task', 'ref-v1', solve, 'compatibility fix')
    manifest_path = tmp_path / result['reference_ref']
    manifest = json.loads(manifest_path.read_text())
    tests = Path(derived['instance']['task_path']) / 'tests/test.sh'
    tests.write_text('changed verifier')
    # Even an attacker updating the bundle hashes cannot broaden the allowed diff.
    manifest['task']['instance']['files'] = package_manifest(tests.parent.parent)
    manifest['task']['instance']['package_sha256'] = hash_json(manifest['task']['instance']['files'])
    manifest['manifest_sha256'] = hash_json({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='outside solution'):
        reference_task(tmp_path, result['reference_ref'], task)


@pytest.mark.parametrize('method,url,headers,status,expected', [
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT', {}, 200, True),
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/Nc%28O%29/cids/TXT', {}, 200, True),
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT', {'cookie': 'x'}, 200, False),
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT', {'authorization': 'x'}, 200, False),
    ('POST', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT', {}, 200, False),
    ('GET', 'https://other.invalid/rest/pug/compound/cid/4578/synonyms/TXT', {}, 200, False),
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT?x=1', {}, 200, False),
    ('GET', 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4578/synonyms/TXT', {}, 503, False),
])
def test_public_api_cookie_exception_is_narrow(method, url, headers, status, expected):
    assert public_stateless_response(method, url, headers, status) is expected


def test_graphql_accepts_only_parsed_query_documents_on_the_public_data_endpoint():
    python = Path('runtime/resource-cache-runtime/venv/bin/python')
    if not python.exists():
        pytest.skip('requires isolated cache runtime')
    program = '''
import json
from runtime.backends.harbor.resource_store import readonly_api_post
url = 'https://data.rcsb.org/graphql'
h = {'content-type': 'application/json'}
def ok(query, **kw):
 return readonly_api_post(kw.get('method','POST'), kw.get('url',url), h, json.dumps({'query':query}).encode())
assert ok('query Schema { __schema { queryType { name } } }')
assert ok('{ entry(entry_id:"5WJ2") { rcsb_id } }')
assert ok('query Q { entry(entry_id:"5WJ2") { ...Fields } } fragment Fields on CoreEntry { rcsb_id }')
assert not ok('mutation { deleteAll }')
assert not ok('query Good { __typename } mutation Bad { deleteAll }')
assert not ok('subscription { events }')
assert not ok('fragment X on Entry { rcsb_id }')
assert not ok('query { mutation }', url='https://other.invalid/graphql')
assert not ok('{ __typename }', url=url+'?x=1')
assert not ok('{ __typename }', method='GET')
assert not ok('not valid graphql')
assert not readonly_api_post('POST',url,h,b'[{"query":"{ __typename }"}]')
print('passed')
'''
    result = subprocess.run([str(python), '-c', program], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_preparation_rejects_shell_input_and_sets_shared_installer_constraints():
    from core.config import HarborTaskPreparation
    from runtime.backends.harbor.task_dependencies import task_dependency_overlay
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        HarborTaskPreparation(system_packages=['cmake; id'])
    with pytest.raises(ValidationError):
        HarborTaskPreparation(python_constraints=['planarity>=0.6'])
    env = task_dependency_overlay({'task_preparation': {'python_constraints': ['planarity==0.6']}}, 'public')[1]
    assert env['PIP_CONSTRAINT'] == env['UV_CONSTRAINT'] == '/opt/dycon-task-constraints.txt'


@pytest.mark.parametrize('name', ['admissions.jsonl', 'response-sanitization.jsonl'])
def test_sealed_cache_rejects_changed_provenance_receipt(tmp_path, name):
    from core.io import write_json
    from runtime.backends.harbor import resource_cache
    write_json(tmp_path / 'record.json', {'mode': 'record'})
    receipt = tmp_path / name
    receipt.write_text('{"source_sha256":"original"}\n')
    resource_cache.seal(tmp_path)
    assert resource_cache.verify(tmp_path)['verified']
    receipt.write_text('{"source_sha256":"modified"}\n')
    with pytest.raises(ValueError, match='provenance receipt changed'):
        resource_cache.verify(tmp_path)
