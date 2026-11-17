import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.deadline import Deadline
from core.io import read_json
from runtime.containers.dependencies import (
    INSTALL_FAILURE_MARKER, classify_dependency_failure, failure_status, setup_error,
)
from runtime.containers.factory import EnvironmentSetupFailed, ManagedContainer, TaskContext
from runtime.containers.policy import EnvironmentPolicy


@pytest.mark.parametrize('text,reason', [
    ('ERROR: No matching distribution found for missing==9.9', 'missing_distribution'),
    ('ERROR: ResolutionImpossible: conflicting dependencies', 'version_conflict'),
    ('Temporary failure in name resolution\nERROR: No matching distribution found for demo', 'dependency_download_network_error'),
    ('ERROR: Package requires a different Python: Requires-Python >=3.10', 'python_or_platform_incompatible'),
    ('Offline wheelhouse is empty for configured requirements: /missing', 'wheelhouse_unavailable'),
])
def test_classifies_real_installation_failure(text, reason):
    result = classify_dependency_failure(text, stage='bootstrap', returncode=1)
    assert result['reason'] == reason
    assert result['failure_type'] == 'dependency_error'


def test_business_test_text_and_xtrace_trap_definition_are_not_dependency_failures():
    assert classify_dependency_failure('FAILED test_foo: No matching distribution found for demo') is None
    assert classify_dependency_failure("+ trap 'printf __DYCON_DEPENDENCY_INSTALL_FAILED__ rc=1' ERR") is None


def test_identical_missing_requirement_and_constraint_are_not_reported_as_version_conflict(tmp_path):
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    result = classify_dependency_failure('ERROR: Cannot install missing==1 because these package versions have conflicting dependencies.\nERROR: ResolutionImpossible',
        stage='bootstrap', requirements=['missing==1'], wheelhouse=str(wheels), offline=True)
    assert result['reason'] == 'missing_distribution'
    assert result['failed_requirements'] == ['missing==1']
    assert result['candidate_check']['unavailable_wheel_requirements'] == ['missing==1']


def test_rollout_preflight_conflict_is_structured_before_container_creation(cfg, tmp_path):
    from runtime.containers.rollout_adapter import RolloutEnvironment
    def spec(instance, **kwargs):
        raise ValueError('Conflicting offline requirement pins for pytest: 4.6.11 vs 8.0.0')
    factory = SimpleNamespace(cfg=cfg, spec=spec)
    context = TaskContext('preflight-run', 't-1', 'attempt-000', 'rollout', tmp_path / 'preflight')
    with pytest.raises(EnvironmentSetupFailed) as error:
        RolloutEnvironment(factory, {'instance_id': 't-1', 'repo': 'org/repo'}, context)
    assert error.value.dependency_failure['reason'] == 'version_conflict'
    assert failure_status(cfg['project_root'], 'preflight-run')['count'] == 1


def test_probe_version_mismatch_keeps_actual_version(cfg, tmp_path):
    from runtime.containers.factory import ContainerFactory
    context = TaskContext('probe-run', 't-1', 'attempt-000', 'harness', tmp_path / 'probe')
    context.output_dir.mkdir()
    factory = ContainerFactory.__new__(ContainerFactory)
    factory.cfg = cfg
    checks = [{'spec': 'demo==1.0', 'actual': '2.0', 'satisfied': False}]
    factory.execute = lambda *a: {'returncode': 3, 'output': json.dumps({'constraints': checks})}
    container = SimpleNamespace(context=context, policy=EnvironmentPolicy(cfg), instance={'repo': 'org/repo'})
    with pytest.raises(EnvironmentSetupFailed) as error:
        factory.probe(container, 'environment.json')
    assert error.value.dependency_failure['reason'] == 'installed_version_mismatch'
    assert error.value.dependency_failure['version_checks'][0]['actual'] == '2.0'


def test_harness_guard_stops_failed_install_without_starting_tests(cfg, tmp_path):
    python = tmp_path / 'python'
    python.write_text('#!/bin/bash\nprintf "ERROR: No matching distribution found for missing==9.9\\n"\nexit 1\n')
    python.chmod(0o755)
    cfg['environment']['variables']['PATH'] = str(tmp_path) + os.pathsep + os.environ['PATH']
    script = '''#!/bin/bash
set -uo pipefail
python -m pip install -e .
printf 'TESTS_STARTED\n'
'''
    result = subprocess.run(['bash'], input=EnvironmentPolicy(cfg).wrap_eval(script),
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert 'TESTS_STARTED' not in result.stdout
    failure = classify_dependency_failure(result.stdout + result.stderr)
    assert failure['stage'] == 'harness_install'
    assert failure['failed_requirements'] == ['missing==9.9']


def test_harness_guard_keeps_end_marker_after_ordinary_failed_test(cfg):
    script = '#!/bin/bash\nset -uo pipefail\nfalse\nprintf "END_TESTS\\n"\n'
    result = subprocess.run(['bash'], input=EnvironmentPolicy(cfg).wrap_eval(script),
                            text=True, capture_output=True)
    assert 'END_TESTS' in result.stdout
    assert INSTALL_FAILURE_MARKER not in result.stdout + result.stderr


def test_bootstrap_records_dependency_failure_before_close(cfg, tmp_path):
    policy = EnvironmentPolicy(cfg)
    out = tmp_path / 'container'
    out.mkdir()
    context = TaskContext('negative-run', 'task', 'attempt-000', 'rollout', out)
    closed = []
    factory = SimpleNamespace(cfg=cfg, policy_for=lambda row: policy,
        execute=lambda *a: {'returncode': 1, 'output': 'ERROR: No matching distribution found for missing==9.9'},
        close=lambda container: closed.append(container.id), event=lambda *a, **k: None)
    raw = SimpleNamespace(id='container-id', start=lambda: None)
    container = ManagedContainer(factory, raw, context, {'instance_id': 'task', 'repo': 'org/repo'}, {'policy_hash': 'fixture'})
    with pytest.raises(EnvironmentSetupFailed) as error:
        container.start()
    failure = error.value.dependency_failure
    assert failure['stage'] == 'bootstrap'
    assert failure['failed_requirements'] == ['missing==9.9']
    assert Path(failure['log_path']).read_text().startswith('ERROR:')
    assert read_json(out / 'bootstrap-receipt.json')['dependency_failure']['reason'] == 'missing_distribution'
    assert failure_status(cfg['project_root'], 'negative-run')['count'] == 1
    assert closed == ['container-id']


def test_harness_keeps_dependency_failure_unknown_even_if_official_parser_says_unresolved(cfg, monkeypatch):
    from conftest import import_fixture
    from application.execution.evidence import RunEvidence
    from benchmark_evaluation import harness

    rid = import_fixture(cfg)
    cfg['selection'] = RunEvidence(cfg['project_root'], 'source').select()
    official = SimpleNamespace(make_test_spec=lambda row, **kw: row,
        run_instance=lambda *a: {'completed': True, 'resolved': False}, dycon_restore=lambda: None)
    monkeypatch.setattr(harness, 'install_harness_adapter', lambda *a: official)
    monkeypatch.setattr(harness, '_evaluation_evidence', lambda *a:
        'ERROR: No matching distribution found for missing==9.9\n'
        '__DYCON_DEPENDENCY_INSTALL_FAILED__ rc=1 command=python -m pip install -e .\n')
    factory = SimpleNamespace(client=None, deadline=Deadline(), lock_images=lambda x: None, cleanup_owned=lambda x: None)
    summary = harness.evaluate(cfg, Path(cfg['project_root']) / 'runs/negative-harness', 'negative-eval', factory=factory)
    result = summary['outcomes'][rid]
    assert result['status'] in {'VerifierError', 'Review'} and result['resolved'] is None
    assert result['failure_type'] == 'dependency_error'
    assert summary['dependency_failure_count'] == 1
    assert summary['dependency_reason_counts'] == {'missing_distribution': 1}
    import hashlib
    owner = 'harness-' + hashlib.sha256(b'negative-harness:negative-eval').hexdigest()[:24]
    assert failure_status(cfg['project_root'], owner)['count'] == 1
    assert read_json(Path(cfg['project_root']) / 'runs/negative-harness/harness/status.json')['stage'] == 'incomplete'


def test_harness_recovers_bootstrap_diagnostic_when_official_wrapper_discards_exception(cfg, monkeypatch):
    from conftest import import_fixture
    from application.execution.evidence import RunEvidence
    from benchmark_evaluation import harness

    rid = import_fixture(cfg)
    cfg['selection'] = RunEvidence(cfg['project_root'], 'source').select()
    run = Path(cfg['project_root']) / 'runs/negative-bootstrap'
    def run_instance(spec, pred, force, nocache, client, evaluation_task_id, timeout):
        context = TaskContext('negative-bootstrap', 't-1', evaluation_task_id, 'harness',
                              run / 'harness/environment' / evaluation_task_id / 't-1')
        setup_error(cfg, context, 'ERROR: No matching distribution found for missing==9.9', stage='bootstrap')
        return {'completed': False, 'resolved': False}
    official = SimpleNamespace(make_test_spec=lambda row, **kw: row, run_instance=run_instance, dycon_restore=lambda: None)
    monkeypatch.setattr(harness, 'install_harness_adapter', lambda *a: official)
    factory = SimpleNamespace(client=None, deadline=Deadline(), lock_images=lambda x: None, cleanup_owned=lambda x: None)
    result = harness.evaluate(cfg, run, 'eval', factory=factory)['outcomes'][rid]
    assert result['failure_type'] == 'dependency_error'
    assert result['dependency_failure']['stage'] == 'bootstrap'
    assert result['resolved'] is None


def test_rollout_reports_dependency_failure_after_infrastructure_budget_exhausted(cfg, monkeypatch):
    from conftest import task
    from catalog.store import Catalog
    from rollout.runner import run_manifest
    from runtime.serving import manager
    from runtime.containers import rollout_adapter
    import core.model_identity
    import runtime.serving.startup
    from application.execution.status import run_status
    from application.execution.report import report

    from core.config import resolve
    cfg = resolve(cfg['config_source'], ['execution.infrastructure_retry_limit=0'])
    Catalog(cfg['project_root']).register_tasks([task()])
    monkeypatch.setattr(core.model_identity, 'verify_model', lambda *a: None)
    monkeypatch.setattr(runtime.serving.startup, 'check_tool_chain', lambda *a: None)
    class Service:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(manager, 'ServingManager', Service)
    def env(factory, instance, context):
        raise setup_error(cfg, context, 'ERROR: No matching distribution found for missing==9.9', stage='bootstrap')
    monkeypatch.setattr(rollout_adapter, 'RolloutEnvironment', env)
    factory = SimpleNamespace(cfg=cfg, deadline=Deadline(), lock_images=lambda x: None, cleanup_owned=lambda x: None)
    outcomes = run_manifest(cfg, 'dependency-rollout', factory=factory)
    row = next(iter(outcomes.values()))
    assert row['exit_status'] == 'infrastructure_exhausted'
    assert row['dependency_failure']['failed_requirements'] == ['missing==9.9']
    assert row['failure_type'] == 'dependency_error'
    status = run_status(cfg['project_root'], 'dependency-rollout')
    assert status['environment_failures']['count'] == 1
    assert next(iter(status['progress']['samples'].values()))['dependency_failure']['stage'] == 'bootstrap'
    assert status['progress']['environment_dependency_failure_count'] == 1
    assert report(cfg['project_root'], 'dependency-rollout')['environment_failures']['count'] == 1
