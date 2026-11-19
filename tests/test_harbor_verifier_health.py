import copy
import json
from pathlib import Path

import pytest

from benchmark_evaluation.harbor import trial_outcome


def outcome(tmp_path, log, *, operation="execute", exit_code=None, execution=None, reward=0):
    trial = tmp_path / "trials/a"
    (trial / "verifier").mkdir(parents=True)
    if log is not None:
        (trial / "verifier/test-stdout.txt").write_text(log)
    if exit_code is not None:
        (trial / "agent").mkdir()
        (trial / "agent/exit-code.txt").write_text(str(exit_code))
    if execution is not None:
        (tmp_path / "verifier-execution.json").write_text(json.dumps(execution))
    raw = {"verifier_result": {"rewards": {"reward": reward}}, "reward_valid": True}
    before = copy.deepcopy(raw)
    row = trial_outcome(raw, task={"benchmark": "terminalbench_2_1", "instance_id": "test"},
                        attempt_id="a", evidence_dir=tmp_path, operation=operation)
    assert raw == before
    assert row["raw_reward"] == {"reward": reward}
    return row


@pytest.mark.parametrize("log", [
    'E: Unable to locate package curl\n/tests/test.sh: line 19: uvx: command not found\n',
    'error: Failed to download python\n  Caused by: operation timed out\n',
    '==================== 1 failed, 1 passed, 2 errors in 9.76s ====================\n',
    'ERROR collecting /tests/test_outputs.py\n==================== 1 error in 0.2s ====================\n',
])
def test_broken_verifier_zero_is_error(tmp_path, log):
    row = outcome(tmp_path, log)
    assert row["status"] == "infra_error"
    assert row["exit_status"] == "VerifierError"
    assert row["resolved"] is None
    assert row["verifier_audit"]["classification"] == "VerifierError"


def test_assertion_failures_are_valid_zero(tmp_path):
    row = outcome(tmp_path, '=================== 2 failed, 1 passed in 0.10s ===================\n',
                  execution={"schema_version": 1, "attempt_id": "a", "return_code": 1})
    assert row["status"] == "completed"
    assert row["resolved"] is False


def test_verifier_reads_bound_trial_name_instead_of_attempt_directory(tmp_path):
    attempt_id = "rep-000-attempt-000"
    trial_name = "dycon-hb-unique"
    for name, log in ((trial_name, "================ 1 failed in 0.10s ================\n"),
                      (attempt_id, "error: Failed to download python: operation timed out\n")):
        logs = tmp_path / "trials" / name / "verifier"
        logs.mkdir(parents=True)
        (logs / "test-stdout.txt").write_text(log)
    (tmp_path / "verifier-execution.json").write_text(json.dumps(
        {"schema_version": 1, "attempt_id": attempt_id, "return_code": 1}))
    row = trial_outcome({"trial_name": trial_name, "verifier_result": {"rewards": {"reward": 0}},
                         "reward_valid": True},
                        task={"benchmark": "terminalbench_2_1", "instance_id": "test"},
                        attempt_id=attempt_id, evidence_dir=tmp_path, operation="execute")
    assert row["status"] == "completed" and row["resolved"] is False
    assert row["verifier_audit"]["sources"][0]["path"] == f"trials/{trial_name}/verifier/test-stdout.txt"


def test_recovered_download_warning_does_not_invalidate_tests(tmp_path):
    row = outcome(tmp_path, 'curl: (60) SSL certificate problem\nretry succeeded\n'
                  '=================== 1 failed in 0.10s ===================\n')
    assert row["resolved"] is False


def test_oracle_failure_does_not_become_valid_reference_zero(tmp_path):
    row = outcome(tmp_path, '=================== 10 failed, 1 passed in 75.53s ===================\n',
                  operation="oracle", exit_code=1)
    assert row["resolved"] is None
    assert row["verifier_audit"]["reason_code"] == "oracle_reference_failed"



@pytest.mark.parametrize("reference_exit", [None, 0])
@pytest.mark.parametrize("reward", [0, 0.5])
def test_oracle_rejected_despite_completed_assertions(tmp_path, reference_exit, reward):
    row = outcome(tmp_path,
                  'E       AssertionError: Time pctg 86.92% (need 60.00%)\n'
                  '========================= 1 failed, 3 passed in 14.04s =========================\n',
                  operation="oracle", exit_code=reference_exit, reward=reward,
                  execution={"schema_version": 1, "attempt_id": "a", "return_code": 1})
    assert row["status"] == "infra_error"
    assert row["resolved"] is None
    assert row["exit_status"] == "VerifierError"
    audit = row["verifier_audit"]
    assert audit["policy"] == "harbor-verifier-health-v3"
    assert audit["reason_code"] == "oracle_reference_rejected"
    assert audit["evidence"]["verifier_health"]["reason_code"] == "tests_completed"
    assert audit["sources"]


@pytest.mark.parametrize("operation", ["execute", "nop", None])
def test_completed_assertion_zero_outside_oracle_is_valid(tmp_path, operation):
    row = outcome(tmp_path, '=================== 1 failed in 0.10s ===================\n',
                  operation=operation)
    assert row["status"] == "completed"
    assert row["resolved"] is False


def test_oracle_zero_missing_health_evidence_is_rejected(tmp_path):
    row = outcome(tmp_path, None, operation="oracle")
    assert row["exit_status"] == "VerifierError"
    assert row["resolved"] is None
    assert row["verifier_audit"]["evidence"]["verifier_health"]["classification"] == "Review"


def test_oracle_zero_without_evidence_directory_is_rejected():
    row = trial_outcome({"verifier_result": {"rewards": {"reward": 0}}},
                        task={"benchmark": "terminalbench_2_1", "instance_id": "test"},
                        attempt_id="a", operation="oracle")
    assert row["resolved"] is None
    assert row["verifier_audit"]["reason_code"] == "oracle_reference_rejected"
    assert row["verifier_audit"]["sources"] == []


def test_oracle_success_is_still_success(tmp_path):
    row = outcome(tmp_path, None, operation="oracle", reward=1)
    assert row["resolved"] is True
    assert row["status"] == "completed"


def test_oracle_unknown_reward_does_not_invent_a_rejection():
    row = trial_outcome({"verifier_result": {"rewards": None}},
                        task={"benchmark": "terminalbench_2_1", "instance_id": "test"},
                        attempt_id="a", operation="oracle")
    assert row["status"] == "unknown"
    assert row["verifier_audit"] is None


def test_oracle_rule_does_not_change_other_benchmarks():
    row = trial_outcome({"verifier_result": {"rewards": {"reward": 0}}},
                        task={"benchmark": "swebench_pro_v2", "instance_id": "test"},
                        attempt_id="a", operation="oracle")
    assert row["resolved"] is False
    assert row["verifier_audit"] is None


def test_model_failure_is_not_oracle_failure(tmp_path):
    row = outcome(tmp_path, '=================== 1 failed in 0.10s ===================\n', exit_code=1)
    assert row["resolved"] is False


def test_unproven_zero_remains_unknown(tmp_path):
    row = outcome(tmp_path, None)
    assert row["status"] == "unknown"
    assert row["resolved"] is None


def test_positive_reward_preserves_upstream_success(tmp_path):
    row = outcome(tmp_path, None, reward=1)
    assert row["resolved"] is True


def test_verifier_exit_two_without_test_summary_is_error(tmp_path):
    row = outcome(tmp_path, 'failed before tests',
                  execution={"schema_version": 1, "attempt_id": "a", "return_code": 2})
    assert row["verifier_audit"]["reason_code"] == "verifier_script_failed"


def test_receipt_from_another_attempt_is_not_trusted(tmp_path):
    row = outcome(tmp_path, 'failed before tests',
                  execution={"schema_version": 1, "attempt_id": "other", "return_code": 2})
    assert row["verifier_audit"]["classification"] == "Review"


def test_symlink_evidence_is_not_followed(tmp_path):
    trial = tmp_path / 'trials/a/verifier'
    trial.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.write_text('=================== 1 failed in 0.10s ===================\n')
    (trial/'test-stdout.txt').symlink_to(outside)
    from benchmark_evaluation.harbor_verifier import audit_verifier
    assert audit_verifier(tmp_path, 'a')['classification'] == 'Review'


@pytest.mark.parametrize("verifier_log,reason", [
    ("/tests/test.sh: line 19: uvx: command not found\n", "verifier_tool_missing"),
    ("=================== 1 failed, 3 passed in 14.04s ===================\n",
     "oracle_reference_rejected"),
])
def test_historical_audit_preserves_sources_and_rejects_reuse(
        tmp_path, monkeypatch, verifier_log, reason):
    from core.io import hash_json, read_json, sha256, write_json
    from runtime.backends.harbor import operations

    run = tmp_path / 'run'
    source = run / 'validation/harbor-controls/control'
    output = source / 'key'
    task = {'benchmark': 'terminalbench_2_1', 'instance_id': 'canary'}
    raw = {'verifier_result': {'rewards': {'reward': 0}}, 'reward_valid': True}
    previous = trial_outcome(raw, task=task, attempt_id='control')
    write_json(output/'worker-response.json', {'result': raw, 'binding': {
        'run_id': 'run', 'task_key': 'key', 'attempt_id': 'control'}})
    log = output/'trials/control/verifier/test-stdout.txt'
    log.parent.mkdir(parents=True)
    log.write_text(verifier_log)
    rows = {'key': previous}
    write_json(source/'summary.json', {'agent': 'oracle', 'tasks': rows,
                                     'control_sha256': hash_json(rows)})
    monkeypatch.setattr(operations, 'load_plan', lambda *args: {'run': run, 'config': {
        'benchmark': {'backend': 'terminalbench_2_1'}, 'selection': {'tasks': {'key': task}}}})
    original = {p: sha256(p) for p in source.rglob('*') if p.is_file()}
    result = operations.audit_controls(tmp_path, 'run', 'control', 'audit')
    assert result['counts']['VerifierError'] == 1
    assert result['reasons'][reason] == 1
    assert result['counts']['valid_zero_or_partial'] == 0
    assert all(sha256(p) == digest for p, digest in original.items())
    receipt = read_json(result['receipt'])
    assert receipt['tasks']['key']['raw_reward'] == {'reward': 0}
    assert receipt['tasks']['key']['resolved'] is None
    with pytest.raises(FileExistsError):
        operations.audit_controls(tmp_path, 'run', 'control', 'audit')
    # Binding a changed response to a historic summary must be rejected.
    response = read_json(output/'worker-response.json')
    response['result']['verifier_result']['rewards']['reward'] = 1
    write_json(output/'worker-response.json', response)
    with pytest.raises(ValueError, match='result hash'):
        operations.audit_controls(tmp_path, 'run', 'control', 'tampered')


def test_pinned_harbor_verifier_exit_receipt(tmp_path):
    import os
    import subprocess

    python = os.environ.get('DYCON_HARBOR_TEST_PYTHON')
    if not python:
        pytest.skip('Set DYCON_HARBOR_TEST_PYTHON to pinned Harbor interpreter')
    source = r'''
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import TaskOS
from harbor.utils.scripts import build_execution_command
from runtime.backends.harbor.environment import DyConDockerEnvironment

async def main():
    root = Path(sys.argv[1])
    env = DyConDockerEnvironment.__new__(DyConDockerEnvironment)
    env._affinity_cpus = None
    env.dycon = {'benchmark': 'terminalbench_2_1', 'attempt_id': 'a', 'output': str(root)}
    official = build_execution_command('/tests/test.sh',
        stdout_path='/logs/verifier/test-stdout.txt', task_os=TaskOS.LINUX)
    receipt = root/'verifier-execution.json'
    with patch.object(DockerEnvironment, 'exec', new_callable=AsyncMock) as parent:
        parent.return_value = SimpleNamespace(return_code=2)
        await env.exec(official)
        assert json.loads(receipt.read_text())['return_code'] == 2
        assert sum(call.args == (official,) for call in parent.await_args_list) == 1
        receipt.unlink()
        await env.exec('echo model command')
        assert not receipt.exists()
        parent.side_effect = TimeoutError('test timeout')
        try:
            await env.exec(official)
        except TimeoutError:
            pass
        else:
            raise AssertionError('timeout swallowed')
        assert json.loads(receipt.read_text())['exception_type'] == 'TimeoutError'
        receipt.unlink()
        env.dycon['benchmark'] = 'swebench_pro_v2'
        parent.side_effect = None
        await env.exec(official)
        assert not receipt.exists()
    print('verifier_execution_receipt_verified')

asyncio.run(main())
'''
    result = subprocess.run([python, '-c', source, str(tmp_path)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'verifier_execution_receipt_verified' in result.stdout


@pytest.mark.parametrize("reference_exit", [1, 2, 134])
def test_full_reward_does_not_hide_failed_oracle_reference(tmp_path, reference_exit):
    row = outcome(tmp_path, '=================== 4 passed in 0.10s ===================\n',
                  operation="oracle", exit_code=reference_exit, reward=1)
    assert row["status"] == "infra_error"
    assert row["resolved"] is None
    assert row["raw_reward"] == {"reward": 1}
    assert row["verifier_audit"]["reason_code"] == "oracle_reference_failed"
    assert row["verifier_audit"]["evidence"]["return_code"] == reference_exit


def test_full_reward_oracle_unreadable_exit_is_review(tmp_path):
    row = outcome(tmp_path, None, operation="oracle", exit_code="invalid", reward=1)
    assert row["status"] == "unknown"
    assert row["resolved"] is None
    assert row["verifier_audit"]["reason_code"] == "invalid_oracle_exit_receipt"


@pytest.mark.parametrize("operation", ["execute", "nop"])
def test_full_reward_model_or_nop_does_not_use_oracle_exit(tmp_path, operation):
    row = outcome(tmp_path, None, operation=operation, exit_code=1, reward=1)
    assert row["resolved"] is True


def test_explicit_zero_reference_exit_keeps_full_reward(tmp_path):
    row = outcome(tmp_path, None, operation="oracle", exit_code=0, reward=1)
    assert row["resolved"] is True
