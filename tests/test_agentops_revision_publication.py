import copy
from pathlib import Path

from core.io import read_json, sha256, write_json
from catalog import verifier_repairs
from catalog.skillsbench import verify_package
from test_verifier_repair_publication import source


def test_revision_publishes_new_protected_validator_and_contract_without_changing_original(source):
    root, snapshot, original = source
    original_hash = original['instance']['package_sha256']
    rules = read_json(verifier_repairs.RULES)
    validator = verifier_repairs.RULES.parent/'matrix.py'
    validator.write_text('REQUIRED = ("py38", "py39", "py310", "py311", "py312")\n')
    rule = rules['tasks']['task']
    rule['files']['verifier/matrix.py'] = {'source_sha256': None, 'effective_sha256': sha256(validator),
        'patch': 'matrix.py', 'mode': 0o644}
    rule['contract'] = {'capabilities': ['agentops_matrix_v1'],
                        'required_environments': ['py38', 'py39', 'py310', 'py311', 'py312']}
    revised = verifier_repairs.RULES.with_name('revision.json');write_json(revised, rules)
    result = verifier_repairs.publish_variant({'project_root': str(root)}, 'revised', snapshot,
                                               rules_path=revised)
    task = read_json(root/result['snapshot_ref'])['tasks']['key']
    verify_package(task);verify_package(original)
    assert original['instance']['package_sha256'] == original_hash
    assert (Path(task['instance']['task_path'])/'verifier/matrix.py').read_bytes() == validator.read_bytes()
    contracts = read_json(root/'runtime/agents/verifier_contracts.json')['contracts']
    assert contracts[0]['capabilities'] == []
    assert contracts[-1]['required_environments'] == rule['contract']['required_environments']
