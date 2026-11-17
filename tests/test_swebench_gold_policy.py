import hashlib
import json

import pytest

from benchmark_evaluation.swebench_verifier import classify, evidence_digest
from benchmark_evaluation.swebench_audit import collect
from core.io import write_json
from test_swebench_verifier import base, failed_document, bind_finding
from test_swebench_audit import fixture


def gold(d):
    d['gold'] = True
    d['gold_provenance'] = {'verified': True, 'input_matches_reference': True}
    return d


def test_verified_gold_failure_is_ineligible_without_inventing_environment_cause():
    result = classify(gold(failed_document()))
    assert result['classification'] == 'VerifierError'
    assert result['reason_code'] == 'gold_control_failed_unattributed'
    assert result['effective_resolved'] is None
    assert result['cause_confirmed'] is False and result['attribution'] == 'unknown'
    assert result['diagnostic_status'] == 'needs_investigation'


@pytest.mark.parametrize('kind', ['ordinary', 'gold_flag_only', 'wrong_identity', 'read_error'])
def test_gold_policy_does_not_override_model_or_untrusted_evidence(kind):
    d = failed_document()
    if kind == 'gold_flag_only':
        d['gold'] = True
    elif kind == 'wrong_identity':
        gold(d)['identity']['instance_id'] = 'other'
    elif kind == 'read_error':
        gold(d)['collection_errors'] = ['source changed']
    result = classify(d)
    assert result['classification'] == ('ValidOutcome' if kind == 'ordinary' else 'Review')
    assert result['effective_resolved'] is (False if kind == 'ordinary' else None)


@pytest.mark.parametrize('error,reason', [
    ('RecursionError: maximum recursion depth exceeded', 'gold_runtime_recursion_failure'),
    ('DeprecationWarning: collections.Iterable', 'gold_runtime_warning_failure'),
    ("AttributeError: vtk object has no attribute PointData", 'gold_dependency_api_failure'),
])
def test_verified_gold_observation_is_not_a_confirmed_runtime_cause(error, reason):
    result = classify(gold(failed_document(error=error)))
    assert result['classification'] == 'VerifierError' and result['reason_code'] == reason
    assert result['cause_confirmed'] is False


def test_gold_patch_defect_requires_explicit_bound_review_and_clean_assertion():
    d = gold(failed_document())
    bind_finding(d)
    f = d['findings'][0]
    f.update(reason_code='gold_patch_defect', attribution='gold_patch',
             supporting_sources=[{'path': 'pinned/upstream.patch', 'sha256': 'a' * 64}])
    result = classify(d)
    assert result['classification'] == 'ValidOutcome' and result['effective_resolved'] is False
    assert result['gold_patch_issue_confirmed'] is True
    # Reusing that decision with a different failure/evidence cannot whitelist gold.
    d['test_output'] += 'changed'
    result = classify(d)
    assert result['classification'] == 'VerifierError' and not result['gold_patch_issue_confirmed']


def test_unsupported_gold_defect_label_cannot_allow_valid_zero():
    d = gold(failed_document())
    bind_finding(d)
    d['findings'][0].update(reason_code='gold_patch_defect', attribution='gold_patch')
    assert classify(d)['classification'] == 'VerifierError'


def test_test_receipt_can_follow_end_marker_when_exact_host_output_is_bound():
    d = base()
    command = 'a' * 64
    line = '__DYCON_TEST_RESULT__ ' + json.dumps({'command_sha256': command,
        'returncode': 0, 'runner': 'unknown', 'scope': 'command'}) + '\n'
    d['test_output'] += line
    d['execution'] = {'returncode': 0, 'test_command_sha256': command,
        'output_sha256': hashlib.sha256(d['test_output'].encode()).hexdigest()}
    result = classify(d)
    assert result['classification'] == 'ValidOutcome'
    assert result['execution']['receipt_location'] == 'host_bound_merged_output'
    d['execution']['output_sha256'] = 'wrong'
    assert classify(d)['classification'] == 'Review'


def test_duplicate_receipt_outside_region_is_rejected_too():
    d = base()
    command = 'a' * 64
    line = '__DYCON_TEST_RESULT__ ' + json.dumps({'command_sha256': command,
        'returncode': 0, 'runner': 'pytest', 'scope': 'command'}) + '\n'
    d['test_output'] += line * 2
    d['execution'] = {'returncode': 0, 'test_command_sha256': command,
        'output_sha256': hashlib.sha256(d['test_output'].encode()).hexdigest()}
    assert classify(d)['classification'] == 'Review'


@pytest.mark.parametrize('token', ['[100%]', '[', '[100%]------------------------------'])
def test_pytest_nested_progress_tokens_are_reported_as_artifacts_not_test_conflicts(token):
    d = base()
    d['task']['repo'] = 'pytest-dev/pytest'
    d['task']['PASS_TO_PASS'].append(token)
    d['report']['tests_status']['PASS_TO_PASS']['success'] = list(d['task']['PASS_TO_PASS'])
    d['statuses'][token] = 'PASSED'
    d['test_output'] = d['test_output'].replace('1 passed', f'FAILED {token}\nPASSED {token}\n1 passed')
    result = classify(d)
    assert result['classification'] == 'ValidOutcome'
    assert result['coverage']['parser_artifact_ids'] == [token]
    d['task']['repo'] = 'other/repo'
    assert classify(d)['classification'] == 'Review'


def test_real_required_identity_conflict_is_not_suppressed_in_pytest():
    d = base()
    d['task']['repo'] = 'pytest-dev/pytest'
    test = d['task']['FAIL_TO_PASS'][0]
    d['test_output'] = d['test_output'].replace('1 passed', f'FAILED {test}\nPASSED {test}\n1 passed')
    assert classify(d)['classification'] == 'Review'


def test_mandatory_tool_skip_is_verifier_error_even_when_official_gold_reward_is_one():
    d = gold(base())
    test = 'test_extract (i18n.BasicTests)'
    d['task'].update(repo='django/django', FAIL_TO_PASS=[test], PASS_TO_PASS=[])
    d['statuses'] = {test: 'SKIPPED'}
    d['report']['tests_status'] = {g: {'success': [], 'failure': []} for g in ['FAIL_TO_PASS', 'PASS_TO_PASS']}
    d['test_output'] = d['test_output'].replace('1 passed', f"{test} ... skipped 'xgettext is mandatory for extraction tests'")
    result = classify(d)
    assert result['classification'] == 'VerifierError' and result['reason_code'] == 'missing_verifier_tool'
    assert result['official_result']['resolved'] is True and result['effective_resolved'] is None
    assert result['cause_confirmed'] is True
    # An unrelated mention of the tool cannot attribute the required test skip.
    d['test_output'] = d['test_output'].replace(test + ' ...', 'unrelated ...')
    assert classify(d)['classification'] == 'Review'


def test_pydicom_absolute_parameter_id_mismatch_is_diagnosed_without_reward_flip():
    d = gold(failed_document())
    frozen = 'tests/test_x.py::test_x[/old/root/file.dcm-data0]'
    d['task'].update(repo='pydicom/pydicom', FAIL_TO_PASS=[frozen], PASS_TO_PASS=[])
    d['statuses'] = {'tests/test_x.py::test_x[/file.dcm-data0]': 'PASSED'}
    d['report']['tests_status'] = {'FAIL_TO_PASS': {'success': [], 'failure': [frozen]},
                                 'PASS_TO_PASS': {'success': [], 'failure': []}}
    result = classify(d)
    assert result['classification'] == 'VerifierError'
    assert result['reason_code'] == 'required_test_parameter_identity_mismatch'
    assert result['cause_confirmed'] is True and result['effective_resolved'] is None


@pytest.mark.parametrize('failure_type,detail,expected', [
    ('network_error', 'failed to extract layer: failed to Lchown for UID 197609 (Hint: /etc/subuid)', 'container_image_ownership_unmapped'),
    ('timeout', 'evaluation timeout', 'gold_evaluation_timeout'),
])
def test_gold_infrastructure_failures_are_not_model_zeroes(failure_type, detail, expected):
    d = gold(base())
    d['official_result'] = {'completed': False, 'resolved': None}
    d['report'] = {}
    d['attempt_result'] = {'failure_type': failure_type, 'failure_detail': detail}
    result = classify(d)
    assert result['classification'] == 'VerifierError' and result['reason_code'] == expected
    assert result['effective_resolved'] is None


def test_collector_checks_actual_patch_artifact_not_just_gold_flag(tmp_path, monkeypatch):
    output, logs, env, task, outcome, rid, eid, _ = fixture(tmp_path, monkeypatch)
    (logs / 'patch.diff').write_text(task['patch'])
    assert collect(output, eid, rid, task, outcome, gold=True)['gold_provenance']['verified']
    (logs / 'patch.diff').write_text('model patch differs')
    d = collect(output, eid, rid, task, outcome, gold=True)
    assert not d['gold_provenance']['verified'] and d['collection_errors']
    assert classify(d)['classification'] == 'Review'


def test_collector_checks_host_output_digest(tmp_path, monkeypatch):
    output, logs, env, task, outcome, rid, eid, _ = fixture(tmp_path, monkeypatch)
    write_json(env / 'eval-execution.json', {'instance_id': task['instance_id'], 'evaluation_task_id': eid,
        'returncode': 0, 'output_sha256': 'wrong'})
    d = collect(output, eid, rid, task, outcome, gold=True)
    assert 'Execution output digest mismatch' in d['collection_errors']
    assert classify(d)['classification'] == 'Review'


def test_one_inside_and_one_outside_receipt_are_not_accepted_as_unique():
    d = base()
    command = 'a' * 64
    line = '__DYCON_TEST_RESULT__ ' + json.dumps({'command_sha256': command,
        'returncode': 0, 'runner': 'pytest', 'scope': 'command'}) + '\n'
    d['test_output'] = d['test_output'].replace('1 passed', '1 passed\n' + line) + line
    d['execution'] = {'returncode': 0, 'test_command_sha256': command,
        'output_sha256': hashlib.sha256(d['test_output'].encode()).hexdigest()}
    assert classify(d)['classification'] == 'Review'


def test_parser_artifacts_alone_cannot_substitute_for_real_required_tests():
    d = base()
    d['task'].update(repo='pytest-dev/pytest', FAIL_TO_PASS=['[100%]'], PASS_TO_PASS=[])
    d['statuses'] = {'[100%]': 'PASSED'}
    d['report']['tests_status'] = {'FAIL_TO_PASS': {'success': ['[100%]'], 'failure': []},
                                 'PASS_TO_PASS': {'success': [], 'failure': []}}
    assert classify(d)['classification'] == 'Review'
