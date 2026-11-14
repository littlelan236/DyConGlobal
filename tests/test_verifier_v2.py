import json
from pathlib import Path

import pytest

from core.io import hash_json, sha256
from runtime.agents.verifier_audit import (
    POLICY,
    audit_zero_evaluation,
    classify_attempt,
    finite_reward,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding='utf-8')


@pytest.fixture
def bound_task(tmp_path, monkeypatch):
    from runtime.agents import verifier_evidence

    def build(script='uvx --with pytest pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py',
              *, caps=(), mode='pytest_exit_binary', extra=None):
        root = tmp_path/'package'
        put(root/'verifier/test.sh', script)
        put(root/'verifier/test_outputs.py', '# explicitly reviewed test fixture')
        for name, value in (extra or {}).items():
            put(root/name, value)
        files = [{'path': p.relative_to(root).as_posix(), 'sha256': sha256(p)}
                 for p in sorted(root.rglob('*')) if p.is_file()]
        package_hash = hash_json(files)
        task = {'instance_id': 'fixture', 'instance': {'instance_id':'fixture', 'files': files,
                'task_path': str(root), 'package_sha256': package_hash}}
        contract = {'instance_id':'fixture', 'files':{f['path']:f['sha256'] for f in files},
                    'score_mode':mode, 'capabilities':list(caps), 'package_sha256':package_hash}
        monkeypatch.setattr(verifier_evidence, 'registry', lambda: {package_hash:contract})
        return task
    return build


def report(tmp_path, *, trace=None, name='test_one', status='failed', suite='ctrf'):
    test = {'name':name, 'status':status}
    if trace:
        test['trace'] = trace
    put(tmp_path/f'benchflow/a/verifier/{suite}.json', {'results': {
        'tests':[test], 'summary':{'tests':1,'passed':int(status=='passed'),'failed':int(status=='failed')}}})


@pytest.mark.parametrize('prefix', ['cause:', 'Caused by:', '  CAUSE:'])
@pytest.mark.parametrize('package', ['pluggy', 'pytest-json-ctrf'])
def test_uv_retry_exhaustion_uses_bound_pytest_dependency(tmp_path, bound_task, prefix, package):
    task = bound_task()
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',
        f'error: Request failed after 3 retries\n{prefix} Failed to fetch: https://mirror/simple/{package}/\n')
    result = classify_attempt(tmp_path, task, reward=0, evaluation_raw={'completed':True})
    assert (result['classification'],result['primary_rule']) == ('VerifierError','R2')
    assert result['dependency_failures'][0]['dependency'] == package


@pytest.mark.parametrize('version', ['3.10.22','3.15.1','3.13.9'])
def test_cpython_download_versions_and_last_failure(tmp_path, bound_task, version):
    task = bound_task()
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',
        f'=== 1 failed in 1s ===\nDownloading cpython-{version}-x86_64.tar.gz\n'
        'error: Request failed after 3 retries\nCaused by: TLS handshake EOF\n')
    result = classify_attempt(tmp_path, task, reward=0, evaluation_raw={'completed':True})
    assert result['reason_code'] == 'python_runtime_download_failed'


@pytest.mark.parametrize('text', [
    'WARNING: Retrying Failed to fetch https://mirror/simple/pluggy/',
    'error: Failed to download pytest',
    'collected 12 items',
    '=== test session starts ===',
    '=== 4 skipped in 1s ===',
])
def test_progress_and_warnings_are_not_completed_or_terminal(tmp_path, bound_task, text):
    task = bound_task()
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',text)
    assert classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})['classification']=='Review'


def test_unknown_package_has_unknown_responsibility(tmp_path, bound_task):
    task = bound_task()
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',
        'error: Request failed after 3 retries\ncause: Failed to fetch https://mirror/simple/agent-library/')
    result = classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['classification']=='Review'
    assert result['reason_code']=='dependency_ownership_unknown'


def test_ctrf_success_does_not_cover_later_installer_abort(tmp_path, bound_task):
    task = bound_task()
    report(tmp_path,status='passed')
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',
        'error: Request failed after 3 retries\ncause: Failed to fetch https://mirror/simple/pluggy/')
    result=classify_attempt(tmp_path,task,reward=.66,evaluation_raw={'completed':True})
    assert result['classification']=='VerifierError'


@pytest.mark.parametrize('reward', [0,.25,.66,.999])
def test_bound_partial_reward_settlement_preserves_raw(tmp_path,bound_task,reward):
    task=bound_task()
    raw={'completed':True,'rewards':{'reward':reward}}
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt','pytest: command not found\n')
    status,resolved,policy,receipt=audit_zero_evaluation(tmp_path,task,'AgentCompleted',False,'normalized',raw)
    assert (status,resolved,policy)==('VerifierError',None,'verifier_incomplete')
    assert receipt==classify_attempt(tmp_path,task,reward=reward,evaluation_raw=raw)
    assert raw['rewards']['reward']==reward


def test_install_success_requires_same_package_and_test_completion(tmp_path,bound_task):
    task=bound_task()
    log=tmp_path/'benchflow/a/verifier/test-stdout.txt'
    abort='error: Request failed after 3 retries\ncause: Failed to fetch https://mirror/simple/pluggy/\n'
    put(log,abort+'Successfully installed unrelated\n')
    assert classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})['classification']=='VerifierError'
    put(log,abort+'Successfully installed pluggy\n')
    assert classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})['classification']=='Review'
    put(log,abort+'Successfully installed pluggy\n=== test session starts ===\n=== 1 failed in 1s ===\n')
    report(tmp_path)
    result = classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['classification'] == 'Review'
    assert result['reason_code'] == 'recovery_environment_unknown'


def test_cross_stream_success_cannot_supply_chronology(tmp_path,bound_task):
    task=bound_task()
    put(tmp_path/'benchflow/a/verifier/test-stderr.txt',
        'error: Request failed after 3 retries\ncause: Failed to fetch https://mirror/simple/pluggy/')
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt','Successfully installed pluggy\n')
    assert classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})['classification']=='Review'


@pytest.mark.parametrize('source_only',[False,True])
def test_mathjax_required_fixture_vs_source_literal(tmp_path,bound_task,source_only):
    task=bound_task(caps=['mathjax_fixture'],extra={'verifier/renderer.py':'MathJax.startup.promise'})
    trace='mathjax_renderer\n> renderer = MathJaxRenderer()\n/verifier/test_outputs.py:214: in __init__\n'
    trace+=('> raise TimeoutError("Page.wait_for_function: Timeout 20000ms exceeded")' if source_only
            else 'E playwright._impl._errors.TimeoutError: Page.wait_for_function: Timeout 20000ms exceeded.')
    report(tmp_path,trace=trace,name='test_render')
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt','=== 6 passed, 1 error in 21s ===\n')
    result=classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['classification']==('ValidOutcome' if source_only else 'VerifierError')
    if not source_only:
        assert result['primary_rule']=='R1'


def test_go_nested_assertion_retains_toolchain_abort(tmp_path,bound_task):
    task=bound_task(caps=['required_go_build'])
    report(tmp_path,trace='E AssertionError: make descriptions failed:\nE go: downloading go1.28\nE https://toolchain/ EOF',name='test_build')
    result=classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['reason_code']=='required_toolchain_download_failed'


def test_fuzzing_all_suites_and_partial_reward(tmp_path,bound_task):
    task=bound_task(caps=['fuzz_driver'],mode='suite_success_fraction_truncate2')
    for suite in ['ctrf','ctrf_arrow','ctrf_ujson','ctrf_black','ctrf_ipython','ctrf_minisgl']:
        report(tmp_path,status='passed',suite=suite)
    report(tmp_path,suite='ctrf_ipython',name='test_fuzz_driver_runs',trace=
        'E AssertionError: Running fuzz.py for ipython failed with code 2: error: Request failed after 3 retries\n'
        'E cause: Failed to fetch https://mirror/simple/pytokens/')
    result=classify_attempt(tmp_path,task,reward=.66,evaluation_raw={'completed':True})
    assert result['reason_code']=='required_subsuite_dependency_download_failed'
    assert len(result['ctrf_reports'])==6


def test_changed_secondary_scorer_disables_adapter(tmp_path,bound_task):
    task=bound_task()
    put(Path(task['instance']['task_path'])/'verifier/test_outputs.py','# changed')
    report(tmp_path,status='passed')
    assert classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})['classification']=='Review'


def test_formal_worker_seam_archives_decision_before_final_report(tmp_path,bound_task):
    from runtime.agents.skillsbench import _audit_before_settlement

    task=bound_task()
    raw={'completed':True,'rewards':{'reward':.66}}
    put(tmp_path/'evaluation-raw.json',raw)
    put(tmp_path/'subject/subject.json',{'binding':{},'outputs':[]})
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt','pytest: command not found')
    result=_audit_before_settlement(tmp_path,task,'AgentCompleted',False,'normalized',raw)
    assert result[:3]==('VerifierError',None,'verifier_incomplete')
    receipt=json.loads((tmp_path/'diagnostics/verifier-zero-audit.json').read_text())
    assert receipt==result[3] and receipt['policy']==POLICY
    assert not (tmp_path/'evaluation-report.json').exists()
    assert raw['rewards']['reward']==.66


def test_settled_worker_recovery_consumes_receipt_without_reclassification(tmp_path,monkeypatch):
    from runtime.agents.skillsbench import _audit_before_settlement

    previous={'exit_status':'AgentCompleted','resolved':False,'evaluation_policy':'legacy',
              'verifier_zero_audit':{'policy':'old','classification':'ValidZero'}}
    put(tmp_path/'attempt.json',previous)
    put(tmp_path/'finalization/05-settled.json',{'stage':'settled'})
    before=(tmp_path/'attempt.json').read_bytes()
    monkeypatch.setattr('runtime.agents.verifier_audit.classify_attempt',
                        lambda *a,**kw: pytest.fail('settled attempt was reclassified'))
    assert _audit_before_settlement(tmp_path,{},'VerifierError',None,'new',{}) == (
        'AgentCompleted',False,'legacy',previous['verifier_zero_audit'])
    assert (tmp_path/'attempt.json').read_bytes()==before


def test_half_written_evaluation_defers_to_existing_recovery(tmp_path):
    from runtime.agents.skillsbench import _audit_before_settlement

    put(tmp_path/'evaluation-report.json',{'schema_version':1,'success_policy':'legacy'})
    before=(tmp_path/'evaluation-report.json').read_bytes()
    with pytest.raises(ValueError,match='recovery'):
        _audit_before_settlement(tmp_path,{},'AgentCompleted',False,'new',{})
    assert (tmp_path/'evaluation-report.json').read_bytes()==before


@pytest.mark.parametrize('kind',['escape','symlink','hardlink','oversize'])
def test_subject_archive_rejects_unsafe_requested_members(tmp_path,kind):
    import io
    import tarfile

    from application.execution.subject import read_archive_members

    archive=tmp_path/'subject.tar'
    with tarfile.open(archive,'w') as stream:
        member=tarfile.TarInfo('../pom.xml' if kind=='escape' else 'pom.xml')
        if kind in {'symlink','hardlink'}:
            member.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE
            member.linkname='/outside'
            stream.addfile(member)
        else:
            payload=b'x'*32
            member.size=len(payload)
            stream.addfile(member,io.BytesIO(payload))
    with pytest.raises(ValueError):
        read_archive_members(archive,sha256(archive),lambda name:True,member_limit=16)


@pytest.mark.parametrize('path,expected',[('/verifier/reference.txt','VerifierError'),
                                          ('/verifier/unknown.txt','Review'),
                                          ('/root/report.json','ValidOutcome')])
def test_reference_and_submission_ownership_are_distinct(tmp_path,bound_task,path,expected):
    task=bound_task(extra={'verifier/reference.txt':'independent official reference',
                          'verifier/test_outputs.py':'OUTPUT_FILE = "/root/report.json"',
                          'task.md':'Write the submitted output to /root/report.json'})
    report(tmp_path,trace=f"E FileNotFoundError: [Errno 2] No such file or directory: '{path}'")
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt','=== 1 error in 1s ===')
    result=classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['classification']==expected
    if path=='/verifier/reference.txt':
        assert result['primary_rule']=='R4'


def test_nested_missing_pytest_dependency_has_evaluator_ownership(tmp_path,bound_task):
    task=bound_task()
    report(tmp_path,trace="E ModuleNotFoundError: No module named 'pytest_json_ctrf'")
    result=classify_attempt(tmp_path,task,reward=0,evaluation_raw={'completed':True})
    assert result['primary_rule']=='R3'


@pytest.mark.parametrize('reward', [0, .25, .66, .999, -1])
def test_partial_rewards_enter_shared_policy(tmp_path, reward):
    raw = {'completed': True, 'rewards': {'reward': reward}}
    result = audit_zero_evaluation(tmp_path, {}, 'AgentCompleted', False, 'normalized', raw)
    assert result[3]['policy'] == POLICY
    assert result[3]['reward'] == reward
    assert raw['rewards']['reward'] == reward
    if reward < 0:
        assert result[3]['reason_code'] == 'invalid_reward_range'


@pytest.mark.parametrize('value', [False, True, None, '0', float('nan'), float('inf')])
def test_non_numeric_rewards_never_coerce(value):
    assert not finite_reward(value)


def test_derived_status_cannot_prove_verifier_failure(tmp_path):
    put(tmp_path/'evaluation-report.json', {'success_policy': 'verifier_incomplete'})
    result = classify_attempt(tmp_path, {}, reward=0, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review'


def test_all_ctrf_reports_are_consumed_and_corruption_is_not_hidden(tmp_path):
    put(tmp_path/'benchflow/a/verifier/ctrf.json', {'results': {'tests': [{'status': 'passed'}]}})
    put(tmp_path/'benchflow/a/verifier/ctrf_ipython.json', '{broken')
    result = classify_attempt(tmp_path, {}, reward=.66, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review'
    assert result['processed'] is False
    assert len(result['ctrf_reports']) == 2


def test_online_and_offline_use_identical_raw_snapshot(tmp_path):
    raw = {'completed': True, 'rewards': {'reward': .33}}
    put(tmp_path/'evaluation-raw.json', raw)
    direct = classify_attempt(tmp_path, {}, reward=.33)
    online = audit_zero_evaluation(tmp_path, {}, 'AgentCompleted', False, 'normalized', raw)[3]
    assert direct == online


def test_source_constant_is_not_runtime_timeout(tmp_path):
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt',
        '> raise TimeoutError("MathJax Page.wait_for_function Timeout")\n')
    result = classify_attempt(tmp_path, {}, reward=0, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review'


def test_same_raw_input_has_one_policy_for_both_production_adapters(tmp_path):
    raw = {'completed': True, 'rewards': {'reward': .66}}
    put(tmp_path/'evaluation-raw.json', raw)
    direct = classify_attempt(tmp_path, {}, reward=.66, evaluation_raw=raw)
    online = audit_zero_evaluation(tmp_path, {}, 'AgentCompleted', False, 'normalized', raw)[3]
    assert direct == online


def test_capture_timeout_is_not_verifier_timeout(tmp_path):
    raw = {'completed': True, 'rewards': {'reward': .25}, 'capture_error': 'TimeoutError'}
    put(tmp_path/'evaluation-raw.json', raw)
    result = classify_attempt(tmp_path, {}, reward=.25, evaluation_raw=raw)
    assert result['classification'] == 'Review'
    assert result['reason_code'] == 'frozen_verifier_contract_unknown'


def test_zero_command_count_is_not_completion_evidence():
    from runtime.agents.verifier_scores import score_check
    snap = {'contract': {'score_mode': 'command_success_fraction', 'verified': True,
                         'status': 'supported', 'command_labels': []}, 'reward': 0,
            'logs': [{'path': 'test-stdout.txt', 'text': '=== Success count: 0/0 commands passed ===\n'
                      'Successful: []\nFailed: []\n'}], 'ctrf_reports': [], 'artifacts': {}}
    assert not score_check(snap)['complete']


def test_malformed_scenario_artifact_returns_incomplete():
    from runtime.agents.verifier_scores import score_check
    snap = {'contract': {'score_mode': 'scenario_score_or_zero', 'verified': True,
                         'status': 'supported'}, 'reward': 0,
            'logs': [{'path': 'test-stdout.txt', 'text': 'scenario_3: 0.000\nFinal score: 0.000\n'}],
            'ctrf_reports': [], 'artifacts': {'verifier/scores/scenario_3.txt': 'invalid'}}
    assert score_check(snap)['issue'] == 'invalid_scenario_score_artifact'


@pytest.mark.parametrize('exception', ['RuntimeError: unexpected scorer crash',
                                      "ModuleNotFoundError: No module named 'unreviewed'",
                                      "FileNotFoundError: [Errno 2] No such file or directory: '/root/unknown.json'"])
def test_unknown_exception_is_not_a_valid_binary_zero(tmp_path, bound_task, exception):
    task = bound_task()
    report(tmp_path, trace='E '+exception)
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt', '=== 1 error in 1s ===\n')
    result = classify_attempt(tmp_path, task, reward=0, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review'
    assert result['reason_code'] == 'exception_ownership_unknown'


def test_worker_negative_reward_records_review_and_preserves_range_rejection(tmp_path, bound_task):
    from runtime.agents.skillsbench import _audit_before_settlement, _map_evaluation
    raw = {'completed': True, 'rewards': {'reward': -.25}}
    status, resolved, policy = _map_evaluation({}, raw, 'complete')
    result = _audit_before_settlement(tmp_path, bound_task(), status, resolved, policy, raw)
    assert result[:3] == ('UnsupportedRewardPolicy', None, 'unsupported_reward_range')
    assert result[3]['reason_code'] == 'invalid_reward_range'
    assert raw['rewards']['reward'] == -.25


@pytest.mark.parametrize('status', ['LimitsExceeded', 'SampleTokenBudgetExceeded', 'OpenHandsError',
                                   'ContextWindowExceededError', 'CleanupError'])
def test_non_candidate_status_is_preserved(tmp_path, bound_task, status):
    raw = {'completed': True, 'rewards': {'reward': .66}}
    put(tmp_path/'benchflow/a/verifier/test-stdout.txt', 'pytest: command not found')
    assert audit_zero_evaluation(tmp_path, bound_task(), status, None, 'original', raw) == (
        status, None, 'original', None)


@pytest.mark.parametrize('mutation', ['status', 'negative_count', 'bool_count', 'unnamed'])
def test_invalid_ctrf_schema_does_not_prove_completion(tmp_path, bound_task, mutation):
    task = bound_task()
    report(tmp_path)
    path = tmp_path/'benchflow/a/verifier/ctrf.json'
    value = json.loads(path.read_text())
    if mutation == 'status':
        value['results']['tests'][0]['status'] = 'unexpected'
    elif mutation == 'unnamed':
        del value['results']['tests'][0]['name']
    else:
        value['results']['summary']['tests'] = -1 if mutation == 'negative_count' else True
    put(path, value)
    result = classify_attempt(tmp_path, task, reward=0, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review' and not result['processed']


def test_duplicate_test_with_required_failure_has_ambiguous_retry(tmp_path, bound_task):
    task = bound_task(caps=['mathjax_fixture'], extra={'verifier/renderer.py': 'MathJax.startup.promise'})
    trace = ('mathjax_renderer\n> renderer = MathJaxRenderer()\n'
             'E TimeoutError: Page.wait_for_function: Timeout 20000ms exceeded.')
    report(tmp_path, trace=trace, name='test_render')
    path = tmp_path/'benchflow/a/verifier/ctrf.json'
    value = json.loads(path.read_text())
    value['results']['tests'].append({'name': 'test_render', 'status': 'passed'})
    put(path, value)
    result = classify_attempt(tmp_path, task, reward=0, evaluation_raw={'completed': True})
    assert result['classification'] == 'Review'
    assert result['reason_code'] == 'duplicate_test_identity'
