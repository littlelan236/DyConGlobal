import copy
import pytest

from core.io import hash_json
from runtime.agents.agentops_matrix import REQUIRED, VERSIONS, validate_matrix
from runtime.agents.verifier_stages import agentops_matrix_evidence, agentops_dependency_failures
from test_verifier_stage_repair import evidence


def report(failed=False):
    environments = {}
    for name in REQUIRED:
        code = int(failed)
        tests = [{'command': ['/venv/bin/coverage', 'run', '--source', 'agentops', '-m', 'pytest'],
                  'retcode': code, 'output': '=== test session starts ===\n=== '+
                  ('1 failed, 2 passed' if failed else '3 passed')+' in 0.50s ===\n'}]
        if not failed:
            tests.append({'command': ['/venv/bin/coverage', 'report', '-m'], 'retcode': 0})
        environments[name] = {'python': {'version_info': list(VERSIONS[name])+['final', 0]},
            'setup': [{'run_id': stage, 'retcode': 0} for stage in
                      ('install_deps', 'install_package_deps', 'install_package', 'freeze')],
            'test': tests, 'result': {'success': not failed, 'exit_code': code}}
    return {'testenvs': environments}


@pytest.mark.parametrize('failed', [False, True])
def test_complete_matrix_distinguishes_pass_and_real_assertion_failure(failed):
    result = validate_matrix(report(failed))
    assert result['complete'] and result['passed'] is not failed
    assert len(result['environments']) == 5


@pytest.mark.parametrize('damage', ['environment', 'interpreter', 'version', 'dependency',
    'setup', 'command', 'summary', 'all_skipped', 'coverage', 'result', 'exit'])
def test_incomplete_execution_never_becomes_normal_zero(damage):
    data = report();env = data['testenvs']['py38']
    if damage == 'environment': del data['testenvs']['py38']
    elif damage == 'interpreter': env.pop('python')
    elif damage == 'version': env['python']['version_info'][2] = 18
    elif damage == 'dependency': env['setup'][0]['retcode'] = 1
    elif damage == 'setup': env['setup'] = []
    elif damage == 'command': env['test'][0]['command'].append('tests/test_one.py')
    elif damage == 'summary': env['test'][0]['output'] = 'started but interrupted'
    elif damage == 'all_skipped': env['test'][0]['output'] = '=== test session starts ===\n=== 3 skipped in 1.0s ==='
    elif damage == 'coverage': env['test'].pop()
    elif damage == 'result': env['result']['success'] = False
    elif damage == 'exit': env['test'][0]['retcode'] = 2
    assert not validate_matrix(data)['complete']


def snapshot(data):
    contract = {'package_sha256': 'package', 'capabilities': ['agentops_ci', 'agentops_matrix_v1'],
        'required_environments': list(REQUIRED),
        'files': [{'path': 'verifier/test_outputs.py', 'sha256': 'source'}]}
    artifacts = {'agentops-tox-result.json': {'data': data},
        'agentops-build.json': {'data': {'return_code': 1}},
        'agentops-ci-stages.jsonl': {'data': [
            {'phase': 'setup-python', 'return_code': 0},
            {'phase': 'pip-install-tox', 'return_code': 0},
            {'phase': 'tox-matrix', 'return_code': 1}]}}
    receipt = {'source': 'trusted_executor_download_after_verifier', 'call_id': 'call',
        'ownership': {'attempt': 'a'}, 'task_package_sha256': 'package',
        'verifier_source_sha256': 'source', 'dependency_snapshot_sha256': 'snapshot',
        'artifacts': artifacts}
    receipt['receipt_sha256'] = hash_json(receipt)
    return {'contract': contract, 'documents': {'diagnostics/verifier-stages.json': receipt,
        'diagnostics/verifier-execution.json': {'ownership': {'attempt': 'a'},
            'attempts': [{'call_id': 'call', 'dependency_snapshot_sha256': 'snapshot'}]}}}


@pytest.mark.parametrize('damage', ['none', 'missing_report', 'corrupt_report', 'missing_ci',
    'missing_build', 'dependency', 'interpreter', 'unbound', 'snapshot', 'call'])
def test_protected_stage_faults_are_identified_without_matching_exact_dependency_arguments(damage):
    snap = snapshot(report(True));receipt = snap['documents']['diagnostics/verifier-stages.json']
    artifacts = receipt['artifacts']
    if damage == 'missing_report': del artifacts['agentops-tox-result.json']
    elif damage == 'corrupt_report': artifacts['agentops-tox-result.json'] = {'error': 'JSONDecodeError'}
    elif damage == 'missing_ci': del artifacts['agentops-ci-stages.jsonl']
    elif damage == 'missing_build': del artifacts['agentops-build.json']
    elif damage == 'dependency': artifacts['agentops-tox-result.json']['data']['testenvs']['py38']['setup'][0]['retcode'] = 1
    elif damage == 'interpreter': del artifacts['agentops-tox-result.json']['data']['testenvs']['py38']['python']
    elif damage == 'snapshot': receipt['dependency_snapshot_sha256'] = 'different'
    elif damage == 'call': receipt['call_id'] = 'different'
    receipt['receipt_sha256'] = hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    if damage == 'unbound': receipt['receipt_sha256'] = 'wrong'
    evidence = agentops_matrix_evidence(snap)
    assert evidence['complete'] is (damage == 'none')
    assert bool(agentops_dependency_failures(snap)) is (damage != 'none')


def test_original_contract_does_not_inherit_revised_environment_scope():
    snap = snapshot(report(True));snap['contract']['capabilities'] = ['agentops_ci']
    assert agentops_matrix_evidence(snap) is None


@pytest.mark.parametrize('damage', ['none', 'missing_report', 'corrupt_report',
    'dependency', 'interpreter', 'missing_environment', 'call'])
def test_shared_classifier_rejects_environment_and_report_faults(evidence, monkeypatch, damage):
    from pathlib import Path
    from core.io import read_json, sha256, write_json
    from runtime.agents import verifier_evidence
    from runtime.agents.verifier_audit import classify_attempt
    attempt, task, _ = evidence
    files = task['instance']['files']
    contract = {'instance_id': 'fixture', 'package_sha256': task['instance']['package_sha256'],
        'files': {r['path']: r['sha256'] for r in files}, 'score_mode': 'pytest_exit_binary',
        'capabilities': ['agentops_ci', 'agentops_matrix_v1'], 'required_environments': list(REQUIRED)}
    monkeypatch.setattr(verifier_evidence, 'registry', lambda: {contract['package_sha256']: contract})
    p = attempt/'diagnostics/verifier-stages.json';receipt = read_json(p)
    receipt['artifacts'] = {}
    artifacts = snapshot(report(True))['documents']['diagnostics/verifier-stages.json']['artifacts']
    if damage == 'missing_report': del artifacts['agentops-tox-result.json']
    elif damage == 'dependency': artifacts['agentops-tox-result.json']['data']['testenvs']['py38']['setup'][0]['retcode'] = 1
    elif damage == 'interpreter': del artifacts['agentops-tox-result.json']['data']['testenvs']['py38']['python']
    elif damage == 'missing_environment': del artifacts['agentops-tox-result.json']['data']['testenvs']['py38']
    for name, entry in artifacts.items():
        dest = attempt/'diagnostics/verifier-stage-artifacts/call'/name
        if name.endswith('.jsonl'):
            import json
            dest.write_text('\n'.join(json.dumps(x) for x in entry['data'])+'\n')
        else:
            write_json(dest, entry['data'])
        receipt['artifacts'][name] = {'path': dest.relative_to(attempt).as_posix(),
            'sha256': sha256(dest), 'data': entry['data']}
    if damage == 'corrupt_report':
        entry = receipt['artifacts']['agentops-tox-result.json']
        dest = attempt/entry['path'];dest.write_text('{broken')
        entry.update(sha256=sha256(dest), error='JSONDecodeError');entry.pop('data')
    if damage == 'call': receipt['call_id'] = 'other'
    receipt['receipt_sha256'] = hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    write_json(p, receipt)
    result = classify_attempt(attempt, task, reward=0)
    assert result['classification'] == ('ValidOutcome' if damage == 'none' else 'VerifierError')


@pytest.mark.parametrize('case', ['agentops-correct', 'agentops-wrong',
    'agentops-missing-dependency', 'agentops-missing-interpreter'])
def test_control_submission_is_valid_python_after_transport_encoding(tmp_path, monkeypatch, case):
    import asyncio
    import shlex
    from types import SimpleNamespace
    from runtime.agents.verifier_controls import prepare_agentops
    from runtime.agents import verifier_evidence
    monkeypatch.setattr(verifier_evidence, 'frozen_contract',
        lambda task: {'capabilities': ['agentops_matrix_v1']})
    class Environment:
        async def exec(self, command, **kwargs):
            source = shlex.split(command)[-1]
            compile(source, '<remote-control>', 'exec')
            assert '\\n[testenv:py38]' in source
            return SimpleNamespace(return_code=0, stdout='syntax valid', stderr='')
    task = {'instance': {'package_sha256': 'package'}}
    asyncio.run(prepare_agentops(Environment(), task, {}, tmp_path, case))


@pytest.mark.parametrize('damage', ['python_shape', 'setup_shape', 'test_shape', 'result_shape', 'ci_shape', 'build_shape'])
def test_structurally_corrupted_reports_are_environment_failures(damage):
    snap = snapshot(report(True));receipt = snap['documents']['diagnostics/verifier-stages.json']
    artifacts = receipt['artifacts'];env = artifacts['agentops-tox-result.json']['data']['testenvs']['py38']
    if damage == 'python_shape': env['python'] = []
    elif damage == 'setup_shape': env['setup'] = [None]
    elif damage == 'test_shape': env['test'] = [None]
    elif damage == 'result_shape': env['result'] = None
    elif damage == 'ci_shape': artifacts['agentops-ci-stages.jsonl']['data'] = None
    elif damage == 'build_shape': artifacts['agentops-build.json']['data'] = []
    receipt['receipt_sha256'] = hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    assert not agentops_matrix_evidence(snap)['complete']
    assert agentops_dependency_failures(snap)


def test_official_unfixed_events_fail_without_becoming_environment_error():
    data = report(True)
    for env in data['testenvs'].values():
        env['test'][0]['output'] = ('=== test session starts ===\n'
            'collected 10 items\n\n'
            'FAILED tests/test_events.py::TestEvents::test_record_timestamp - AssertionError\n'
            '=== 1 failed, 9 passed in 2.75s ===\n')
    result = validate_matrix(data)
    assert result['complete'] and not result['passed']


def test_tox_255_and_ci_wrapper_1_preserve_normal_test_failure():
    snap = snapshot(report(True))
    receipt = snap['documents']['diagnostics/verifier-stages.json']
    receipt['artifacts']['agentops-ci-stages.jsonl']['data'][-1]['return_code'] = 255
    receipt['receipt_sha256'] = hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    result = agentops_matrix_evidence(snap)
    assert result['complete'] and not result['passed']
    assert agentops_dependency_failures(snap) == []
