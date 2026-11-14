import json

import pytest

from core.io import hash_json, sha256
from runtime.agents.verifier_audit import (
    POLICY,
    audit_zero_evaluation,
    classify_attempt,
    classify_zero_reward_trajectory,
    frozen_verifier_contract,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value,
                    encoding='utf-8')


def classify(root, **kwargs):
    return classify_zero_reward_trajectory(root, disposition='accepted', reward=0, **kwargs)[0]


@pytest.mark.parametrize(('log', 'expected'), [
    ('WARNING: Retrying: Failed to fetch https://example.org/a\n', 'Review'),
    ('Failed to download `pytest`\n', 'Review'),
    ('Could not fetch URL https://example.org/simple\n', 'Review'),
    ('error: Failed to download opencv-python==5.0.0.93\n'
     '  cause: Request failed after 4 retries\n'
     '  cause: Failed to fetch https://mirror/simple/opencv-python/\n', 'VerifierError'),
    ('Initialized project root\n'
     'error: Request failed after 3 retries\n'
     '  Caused by: Failed to fetch: https://mirror/simple/openpyxl/\n', 'VerifierError'),
    ('Looking in indexes: https://mirror/simple\n'
     'Collecting scipy\n'
     'ERROR: Exception:\nReadTimeoutError: HTTPSConnectionPool\n', 'VerifierError'),
    ('Initialized project root\n'
     'error: Request failed after 3 retries\n'
     '  Caused by: Failed to fetch: https://mirror/simple/openpyxl/\n'
     'Resolved 11 packages\nInstalled 1 package\n'
     'ModuleNotFoundError: No module named \'openpyxl\'\n', 'VerifierError'),
    ('ERROR: No matching distribution found for pytest\n', 'VerifierError'),
    ('ERROR: Could not find a version that satisfies the requirement pytest\n', 'VerifierError'),
    ('ERROR: Could not install packages due to an OSError\n', 'VerifierError'),
    ('ERROR: THESE PACKAGES DO NOT MATCH THE HASHES\n', 'VerifierError'),
    ('No solution found when resolving tool dependencies\n', 'VerifierError'),
    ('/verifier/test.sh: line 8: pytest: command not found\n', 'VerifierError'),
    ('/verifier/test.sh: line 8: python3: command not found\n', 'VerifierError'),
    ("/usr/bin/python: No module named pytest\n", 'VerifierError'),
    ('× Failed to download `pytest`\nCaused by: Request failed after 3 retries\n', 'VerifierError'),
    ('× Failed to download `pytest`\n', 'Review'),
    ('ERROR: No matching distribution found for pytest\nSuccessfully installed pytest\n', 'Review'),
    ('Successfully installed unrelated\nERROR: No matching distribution found for pytest\n', 'VerifierError'),
    ('Failed to fetch https://example.org\nSuccessfully installed pytest\n=== 2 failed in 1.2s ===\n', 'Review'),
    ('=== 2 passed, 1 failed, 3 warnings in 1.2s ===\n', 'Review'),
    ('=== 2 errors in 0.5s ===\n', 'Review'),
    ('=== 3 skipped in 0.5s ===\n', 'Review'),
    ('=== 0 passed in 0.5s ===\n', 'Review'),
    ('AssertionError: output differs\n', 'Review'),
    ("ModuleNotFoundError: No module named 'agent_solution'\n", 'Review'),
])
def test_final_failure_recovery_and_execution(tmp_path, log, expected):
    put(tmp_path / 'benchflow/a/session/verifier/test-stdout.txt', log)
    # A log-only legacy call lacks a frozen identity and cannot authorize a v2 verdict.
    assert classify(tmp_path) == 'Review'


def test_trainer_install_failures_are_never_verifier_evidence(tmp_path):
    put(tmp_path / 'benchflow/a/session/trainer/verifiers.jsonl',
        'ERROR: No matching distribution found for pytest\n')
    put(tmp_path / 'trainer/verifier/test-stdout.txt', 'pytest: command not found')
    assert classify(tmp_path) == 'Review'


def test_cross_stream_recovery_has_no_reliable_chronology(tmp_path):
    put(tmp_path / 'benchflow/a/verifier/test-stderr.txt', 'ERROR: No matching distribution found for pytest\n')
    put(tmp_path / 'benchflow/a/verifier/test-stdout.txt', 'Successfully installed pytest\n')
    assert classify(tmp_path) == 'Review'


def test_legacy_log_only_api_is_explicitly_conservative(tmp_path):
    put(tmp_path / 'benchflow/a/verifier/ctrf.json', {'results': {'summary': {'tests':3,'failed':3}}})
    put(tmp_path / 'benchflow/a/verifier/test-stderr.txt', 'ERROR: No matching distribution found for pytest')
    assert classify(tmp_path) == 'Review'


@pytest.mark.parametrize('report', [
    {'results': {'summary': {'tests': 0}, 'tests': []}},
    {'results': {'summary': {'tests': 3, 'skipped': 3}, 'tests': [{'status': 'skipped'}]}},
    'invalid json',
])
def test_empty_skipped_or_broken_ctrf_does_not_prove_execution(tmp_path, report):
    put(tmp_path / 'benchflow/a/verifier/ctrf.json', report)
    assert classify(tmp_path, ctrf_expected=True) == 'Review'


def test_ctrf_tests_without_summary(tmp_path):
    put(tmp_path / 'benchflow/a/verifier/ctrf.json', {'results': {'tests': [{'status': 'failed'}]}})
    assert classify(tmp_path) == 'Review'


@pytest.mark.parametrize('expected', [True, False, None])
def test_missing_ctrf_is_never_automatic_failure(tmp_path, expected):
    assert classify(tmp_path, ctrf_expected=expected) == 'Review'


def test_non_ctrf_requires_task_completion_contract(tmp_path):
    put(tmp_path / 'benchflow/a/verifier/test-stdout.txt', 'SCORECARD COMPLETE: 0/10\n')
    assert classify(tmp_path, ctrf_expected=False) == 'Review'
    assert classify(tmp_path, ctrf_expected=False,
                    task_completion_patterns=(r'^SCORECARD COMPLETE: \d+/\d+\s*$',)) == 'Review'


@pytest.mark.parametrize(('raw', 'report'), [
    ({'completed': False, 'error': {'type': 'TimeoutError'}}, {}),
    ({'status': 'completed_unknown'}, {}),
    ({'evaluation_status': 'verifier_incomplete'}, {}),
    ({}, {'evaluation_status': 'completed_unknown'}),
    ({}, {'evaluation_status': 'verifier_incomplete'}),
    ({}, {'success_policy': 'verifier_incomplete'}),
])
def test_derived_evaluation_report_is_not_execution_failure_evidence(tmp_path, raw, report):
    put(tmp_path / 'benchflow/a/verifier/test-stdout.txt', '=== 2 failed in 1s ===\n')
    put(tmp_path / 'evaluation-raw.json', raw)
    result = classify_attempt(tmp_path, {}, reward=0, evaluation_raw=raw)
    assert result['classification'] == 'Review'
    assert result['policy'] == POLICY


def test_bare_completed_false_and_malformed_structured_files_need_review(tmp_path):
    put(tmp_path / 'evaluation-raw.json', {'completed': False})
    put(tmp_path / 'evaluation-report.json', ['completed_unknown'])
    assert classify(tmp_path) == 'Review'
    put(tmp_path / 'evaluation-raw.json', '{bad')
    assert classify(tmp_path) == 'Review'


@pytest.mark.parametrize(('disposition', 'reward'), [('failed', 0), ('accepted', 1),
    ('accepted', None), ('accepted', False), ('accepted', '0')])
def test_candidates_are_strict(tmp_path, disposition, reward):
    assert classify_zero_reward_trajectory(tmp_path, disposition=disposition,
        reward=reward, evaluation_raw={'completed': False, 'error': 'failed'})[0] == 'NotCandidate'


def task_package(tmp_path, script):
    path = tmp_path / 'package/tests/test.sh'
    put(path, script)
    files = [{'path': 'tests/test.sh', 'sha256': sha256(path), 'mode': 0o644}]
    return {'instance': {'task_path': str(path.parent.parent), 'files': files,
                         'package_sha256': hash_json(files)}}


def test_contract_uses_frozen_hash_and_rejects_current_checkout_drift(tmp_path):
    task = task_package(tmp_path, 'pytest --ctrf /logs/verifier/ctrf.json\n')
    assert frozen_verifier_contract(task)['ctrf_expected'] is True
    put(tmp_path / 'package/tests/test.sh', 'echo new version\n')
    contract = frozen_verifier_contract(task)
    assert contract['ctrf_expected'] is None
    assert contract['verified'] is False


def test_non_ctrf_and_comments_are_not_assumed_completion_contracts(tmp_path):
    task = task_package(tmp_path, '# pytest --ctrf ctrf.json\npython scorer.py\n')
    assert frozen_verifier_contract(task)['ctrf_expected'] is None


def test_frozen_verifier_directory_contract_is_hash_guarded(tmp_path):
    path = tmp_path / 'package/verifier/test.sh'
    put(path, 'uvx pytest --ctrf /logs/verifier/ctrf.json test_outputs.py\n')
    files = [{'path': 'verifier/test.sh', 'sha256': sha256(path), 'mode': 0o644}]
    task = {'instance': {'task_path': str(path.parent.parent), 'files': files,
                         'package_sha256': hash_json(files)}}
    contract = frozen_verifier_contract(task)
    assert contract['verified'] is True
    assert contract['ctrf_expected'] is True


def test_settlement_unknown_identity_retains_original_reward(tmp_path):
    from rollout.retry import disposition

    raw = {'completed': True, 'rewards': {'reward': 0}}
    put(tmp_path / 'benchflow/a/verifier/test-stdout.txt', 'pytest: command not found\n')
    status, resolved, policy, receipt = audit_zero_evaluation(tmp_path, {}, 'AgentCompleted', False, 'normalized', raw)
    assert (status, resolved, policy) == ('AgentCompleted', False, 'normalized')
    assert disposition({}, status) == 'accepted'
    assert receipt['classification'] == 'Review'
    assert raw['rewards'] == {'reward': 0}
    assert audit_zero_evaluation(tmp_path, {}, 'LimitsExceeded', False, 'normalized', raw)[3] is None


def test_python_runtime_download_failure_is_not_a_package_install_failure(tmp_path):
    from runtime.agents.accepted_reward_audit import _audit_candidate

    attempt = tmp_path / 'runs/r/rollout/tasks/simpo/repetitions/rep-000/attempts/rep-000-attempt-000'
    put(attempt / 'benchflow/x/verifier/test-stdout.txt',
        'Downloading cpython-3.10.22-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz\n'
        'error: Request failed after 3 retries\nCaused by: tls handshake eof\n')
    row = {'instance_id': 'simpo', 'repetition_id': 'rep-000', 'attempt_id': 'rep-000-attempt-000',
           'sample_key': 's1', 'disposition': 'accepted', 'raw_reward': {'reward': 0}}
    result = _audit_candidate(tmp_path, 'r', row, {})
    assert result['classification'] == 'Review'
    assert result['reason_code'] == 'evidence_read_failed'


@pytest.mark.parametrize('evidence', ['pytest', 'earlier_summary', 'ctrf', 'recovered_stderr', 'nonzero'])
def test_existing_bootstrap_guard_respects_execution_and_recovery(tmp_path, evidence):
    import asyncio
    from types import SimpleNamespace

    from runtime.agents.skillsbench import _run_verifier_with_evidence

    log = tmp_path / 'benchflow/a/verifier/test-stdout.txt'
    stderr = log.with_name('test-stderr.txt')
    text = ('/verifier/test.sh: line 11: /root/.local/bin/env: No such file or directory\n'
            '/verifier/test.sh: line 17: uvx: command not found\n')
    if evidence == 'pytest':
        text += '=== 1 failed in 1s ===\n'
    elif evidence == 'earlier_summary':
        text = '=== 1 failed in 1s ===\n' + 'noise\n' * 2000 + text
    elif evidence == 'ctrf':
        put(log.with_name('ctrf.json'), {'results': {'summary': {'tests': 1, 'failed': 1}}})
    elif evidence == 'recovered_stderr':
        put(stderr, 'Successfully installed uv pytest\n')
    put(log, text)

    class Verifier:
        _rollout_paths = SimpleNamespace(test_stdout_path=log, test_stderr_path=stderr)

        async def verify(self):
            return SimpleNamespace(rewards={'reward': 1 if evidence == 'nonzero' else 0})

    owner = SimpleNamespace(output=tmp_path, ownership={'attempt_id': 'a'},
        sample=SimpleNamespace(task_key='task',instance_id='task'),
        cfg={'selection':{'tasks':{'task':{'instance':{'package_sha256':'a'*64}}}},
             'execution_context':{'profile':{}}})
    result = asyncio.run(_run_verifier_with_evidence(Verifier(), owner))
    assert result.rewards['reward'] == (1 if evidence == 'nonzero' else 0)
