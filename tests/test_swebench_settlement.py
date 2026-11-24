"""Exercise the real SWE collector -> classifier -> settlement -> evidence chain."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.execution.evidence import RunEvidence
from benchmark_evaluation import harness
from benchmark_evaluation.swebench_audit import collect, audit_run
from benchmark_evaluation.swebench_verifier import classify
from benchmark_evaluation.swebench_settlement import project, label, training_label, validate, POLICY
from core.deadline import Deadline
from core.io import hash_json, read_json, write_json, sha256
from difficulty.datasets.eligibility import eligible_evaluation
from conftest import import_fixture
from test_swebench_verifier import base, failed_document


@pytest.mark.parametrize('classification,resolved,status,reward', [
    ('ValidOutcome', True, 'completed', 1), ('ValidOutcome', False, 'completed', 0),
    ('VerifierError', None, 'VerifierError', None), ('Review', None, 'Review', None),
])
def test_score_projection_preserves_raw_and_never_coerces_unknown(classification, resolved, status, reward):
    raw = {'status': 'completed', 'resolved': True}
    audit = {'classification': classification, 'effective_resolved': resolved,
             'official_result': {'completed': True, 'resolved': resolved if type(resolved) is bool else True}}
    result = project(raw, audit, gold=classification == 'Review')
    validate(result)
    assert result['status'] == status and result['reward'] == reward
    assert result['resolved'] is resolved and label(result) is resolved
    assert result['raw_outcome'] == raw and raw == {'status': 'completed', 'resolved': True}


def test_invalid_validoutcome_payload_does_not_create_zero_or_one():
    raw = {'status': 'completed', 'resolved': False}
    result = project(raw, {'classification': 'ValidOutcome', 'effective_resolved': True,
                          'official_result': {'completed': True, 'resolved': False}})
    validate(result)
    assert result['status'] == 'completed_unknown' and result['reward'] is None


@pytest.mark.parametrize('kind', ['VerifierError', 'Review', 'missing_effective', 'mismatch'])
def test_legacy_sidecar_is_not_ignored_by_training(kind):
    outcome = {'status': 'completed', 'resolved': True, 'verifier_audit': {
        'classification': kind if kind in {'Review','VerifierError'} else 'ValidOutcome',
        'effective_resolved': None if kind == 'missing_effective' else False if kind == 'mismatch' else None}}
    assert label(outcome) is None


@pytest.mark.parametrize('tamper', ['resolved', 'reward', 'status', 'scoring_eligible'])
def test_evidence_validator_rejects_invalid_score_payload(tamper):
    result = project({'status': 'completed', 'resolved': False}, {
        'classification': 'VerifierError', 'effective_resolved': None, 'official_result': {}})
    result[tamper] = {'resolved': False, 'reward': 0, 'status': 'completed', 'scoring_eligible': True}[tamper]
    with pytest.raises(ValueError): validate(result)


def install_fake_runner(cfg, monkeypatch, *, resolved=True, broken=False, dependency=False, raise_first=False):
    rid = import_fixture(cfg)
    selected = RunEvidence(cfg['project_root'], 'source').select()
    task = selected['tasks'][selected['rollouts'][rid]['task_key']]['instance']
    task.update(repo='pytest-dev/pytest', version='7.0', FAIL_TO_PASS=['tests/test_x.py::test_x'],
                PASS_TO_PASS=[], patch='reference patch\n', test_patch='test patch\n')
    selected['tasks'][selected['rollouts'][rid]['task_key']]['repo'] = task['repo']
    cfg['selection'] = selected
    run = Path(cfg['project_root'])/'runs/settled'
    calls = []
    def run_instance(spec, pred, force, nocache, client, eid, timeout):
        calls.append(eid)
        if raise_first and len(calls)==1: raise ConnectionError('reset')
        logs = run/'harness/harness'/eid/rid/task['instance_id']
        logs.mkdir(parents=True)
        (logs/'patch.diff').write_text(pred['model_patch'])
        (logs/'eval.sh').write_text("#!/bin/bash\n: '>>>>> Start Test Output'\npytest -rA tests/test_x.py\n: '>>>>> End Test Output'\n")
        text = "+ : '>>>>> Start Test Output'\n"
        if broken:
            text += 'ImportError loading conftest\nE   AttributeError: `np.Inf` was removed in the NumPy 2.0 release\n'
        else:
            if not resolved:text += '_____ test_x _____\nE   AssertionError: expected\n'
            text += ('PASSED' if resolved else 'FAILED')+' tests/test_x.py::test_x\n'
        text += "+ : '>>>>> End Test Output'\n"
        if dependency:
            text = '__DYCON_DEPENDENCY_INSTALL_FAILED__ rc=1 command=python -m pip install -e .\n'+text
            env=run/'harness/environment'/eid/task['instance_id']
            write_json(env/'dependency-failure.json', {'stage':'harness_install','reason':'missing_distribution'})
        (logs/'test_output.txt').write_text(text)
        (logs/'run_instance.log').write_text('runner fixture\n')
        success = [task['FAIL_TO_PASS'][0]] if resolved else []
        write_json(logs/'report.json', {task['instance_id']:{'resolved':resolved, 'patch_successfully_applied':True,
            'tests_status':{'FAIL_TO_PASS':{'success':success,'failure':[] if resolved else task['FAIL_TO_PASS']},
                            'PASS_TO_PASS':{'success':[],'failure':[]}}}})
        return {'completed':True,'resolved':resolved}
    official=SimpleNamespace(make_test_spec=lambda row,**kw:row,run_instance=run_instance,dycon_restore=lambda:None)
    monkeypatch.setattr(harness,'install_harness_adapter',lambda *args:official)
    monkeypatch.setattr(harness,'NETWORK_RETRY_DELAY_SECONDS',0)
    factory=SimpleNamespace(lock_images=lambda x:None,client=None,deadline=Deadline(),cleanup_owned=lambda x:None)
    return rid,run,factory,calls


@pytest.mark.parametrize('gold,broken,resolved,expected', [
    (False,False,True,'completed'),(False,False,False,'completed'),
    (False,True,False,'completed'),(False,True,True,'completed'),
    (True,True,False,'VerifierError'),(True,True,True,'Review'),
])
def test_actual_harness_settlement_and_training_eligibility(cfg,monkeypatch,gold,broken,resolved,expected):
    rid,run,factory,calls=install_fake_runner(cfg,monkeypatch,resolved=resolved,broken=broken)
    summary=harness.evaluate(cfg,run,'eval',factory=factory,gold=gold)
    out=summary['outcomes'][rid]
    assert out['status']==expected
    assert out['reward']==(int(resolved) if expected=='completed' else None)
    assert out['official_result']=={'completed':True,'resolved':resolved}
    if broken and not gold:
        assert out['verifier_audit']['classification'] == 'OfficialOutcome'
        assert out['verifier_audit']['cause_confirmed'] is False
    assert out['raw_outcome']['resolved'] is resolved
    assert summary['verifier_audit_mode']=='settlement' and summary['scoring_policy']==POLICY
    if not gold:
        ev=RunEvidence(cfg['project_root'],'settled').read();row=ev['rollouts'][rid]
        assert row['evaluation_status']==expected and row['reward']==out['reward']
        assert training_label(row,ev,rid) is (resolved if expected=='completed' else None)
        assert eligible_evaluation(row,ev,rid,'resolved_only') is (expected=='completed' and resolved)
    # Offline replay must consume original scoring, including projected null.
    write_json(run/'harness/selection.json',cfg['selection'])
    result=audit_run(cfg['project_root'],'settled',Path(cfg['project_root'])/'replay')
    audit=read_json(Path(cfg['project_root'])/'replay/tasks'/f'{rid}.json')
    assert audit['official_result']==out['official_result']
    assert audit['classification']==out['verifier_audit']['classification']


@pytest.mark.parametrize('completed,resolved,execution_status,expected_class', [
    (True, True, 'completed', 'OfficialOutcome'),
    (True, False, 'completed', 'OfficialOutcome'),
    (False, False, 'ConnectionError', 'Unscored'),
])
def test_carry_forward_uses_persisted_official_grade_without_editing_old_review(
        completed, resolved, execution_status, expected_class):
    official = {'completed': completed, 'resolved': resolved}
    legacy = project({'status': execution_status, 'resolved': resolved}, {
        'classification': 'Review', 'reason_code': 'execution_evidence_incomplete',
        'official_result': official}, gold=True)
    legacy['verifier_audit'].update(path='old-audit.json', sha256='hash')
    previous = copy.deepcopy(legacy)
    result = harness._carry_forward_outcome(legacy, gold=False)
    validate(result)
    assert result['verifier_audit']['classification'] == expected_class
    assert result['verifier_audit']['source_audit'] == {'path': 'old-audit.json', 'sha256': 'hash'}
    assert result['status'] == execution_status
    assert result['resolved'] is (resolved if completed else None)
    assert result['reward'] == (int(resolved) if completed else None)
    assert result['official_result'] == official
    assert legacy == previous


def test_retry_success_discards_previous_unscored_settlement(cfg,monkeypatch):
    rid,run,factory,calls=install_fake_runner(cfg,monkeypatch,raise_first=True)
    out=harness.evaluate(cfg,run,'retry',factory=factory)['outcomes'][rid]
    assert len(calls)==2 and out['status']=='completed' and out['reward']==1
    assert out['attempts'][0]['status']=='ConnectionError' and out['attempts'][0]['reward'] is None
    assert out['verifier_audit']['classification']=='ValidOutcome' and 'error' not in out


def test_dependency_error_overrides_official_false_before_evidence_write(cfg,monkeypatch):
    rid,run,factory,_=install_fake_runner(cfg,monkeypatch,resolved=False,dependency=True)
    out=harness.evaluate(cfg,run,'dependency',factory=factory)['outcomes'][rid]
    assert out['status']=='VerifierError' and out['reward'] is None
    assert out['official_result']['resolved'] is False
    assert out['failure_type']=='dependency_error'
    assert RunEvidence(cfg['project_root'],'settled').read()['rollouts'][rid]['resolved'] is None


def test_missing_submission_is_supported_zero_but_missing_gold_is_not(cfg,monkeypatch):
    rid,run,factory,_=install_fake_runner(cfg,monkeypatch)
    row=cfg['selection']['rollouts'][rid]
    patch=Path(cfg['project_root'])/row['patch_path'];patch.write_text('')
    row['patch_sha256']=sha256(patch)
    out=harness.evaluate(cfg,run,'empty',factory=factory)['outcomes'][rid]
    assert out['status']=='completed' and out['reward']==0 and out['attempts']==[]
    write_json(run/'harness/selection.json',cfg['selection'])
    target=Path(cfg['project_root'])/'empty-replay'
    audit_run(cfg['project_root'],'settled',target)
    assert read_json(target/'tasks'/f'{rid}.json')['classification']=='ValidOutcome'


def test_success_label_cannot_bypass_legacy_sidecar_when_audit_is_off(cfg):
    rid=import_fixture(cfg,resolved=True)
    ev=RunEvidence(cfg['project_root'],'source').read();row=ev['rollouts'][rid]
    out=ev['evaluations'][row['selected_evaluation_id']]['outcomes'][rid]
    out['verifier_audit']={'classification':'Review','effective_resolved':None}
    assert training_label(row,ev,rid) is None
    assert not eligible_evaluation(row,ev,rid,'resolved_only')


def test_true_legacy_swebench_without_audit_cannot_create_new_success_labels(cfg):
    rid=import_fixture(cfg,resolved=True)
    ev=RunEvidence(cfg['project_root'],'source').read();row=ev['rollouts'][rid]
    ev['tasks'][row['task_key']]['benchmark']='swebench'
    assert training_label(row,ev,rid) is None


def test_timeout_suffix_does_not_misclassify_valid_host_digest(tmp_path,monkeypatch):
    from test_swebench_audit import fixture
    output,logs,env,task,outcome,rid,eid,_=fixture(tmp_path,monkeypatch)
    text=(logs/'test_output.txt').read_text()
    (logs/'test_output.txt').write_text(text+'\n\nTimeout error: 300 seconds exceeded.')
    write_json(env/'eval-execution.json',{'instance_id':task['instance_id'],'evaluation_task_id':eid,
        'returncode':124,'timed_out':True,'output_sha256':hashlib.sha256(text.encode()).hexdigest()})
    doc=collect(output,eid,rid,task,outcome,gold=True)
    assert 'Execution output digest mismatch' not in doc['collection_errors']


def test_malformed_bound_submission_is_valid_zero_but_hunk_failure_is_unscored(cfg,monkeypatch):
    from benchmark_evaluation import swebench_audit
    rid,run,factory,_=install_fake_runner(cfg,monkeypatch)
    selected=cfg['selection'];task=selected['tasks'][selected['rollouts'][rid]['task_key']]['instance']
    eid='task-malformed';logs=run/'harness/harness'/eid/rid/task['instance_id'];logs.mkdir(parents=True)
    (logs/'patch.diff').write_text('garbage')
    (logs/'run_instance.log').write_text('>>>>> Patch Apply Failed:\nerror: No valid patches in input (allow with "--allow-empty")\n')
    outcome={'status':'completed','resolved':False,'patch_sha256':hashlib.sha256(b'garbage').hexdigest(),
             'official_result':{'completed':False,'resolved':False}}
    doc=collect(run/'harness',eid,rid,task,outcome)
    result=classify(doc)
    assert result['classification']=='ValidOutcome' and result['reason_code']=='invalid_patch'
    out=project(outcome,result);validate(out)
    assert out['reward']==0
    (logs/'run_instance.log').write_text('>>>>> Patch Apply Failed:\nerror: patch failed: module.py:20\n')
    assert classify(collect(run/'harness',eid,rid,task,outcome))['classification']=='Unscored'
    (logs/'patch.diff').write_text('other bytes')
    assert classify(collect(run/'harness',eid,rid,task,outcome))['reason_code']=='evidence_integrity_error'


@pytest.mark.parametrize('audit_enabled',[False,True])
@pytest.mark.parametrize('classification',['VerifierError','Review'])
def test_real_grpo_builder_excludes_whole_trajectory_even_with_general_audit_off(tmp_path,audit_enabled,classification):
    import yaml
    from test_rl_dataset import token_source
    from RL.dataset.build import build_dataset,iter_rows
    from RL.dataset.loader import iter_groups
    from RL.grpo.groups import GroupBatch
    f,config_file,config=token_source(tmp_path)
    config['audit']['enabled']=audit_enabled;config_file.write_text(yaml.safe_dump(config))
    ep=tmp_path/'runs'/f['run']/'evidence.json';ev=read_json(ep);row=ev['rollouts'][f['rid']]
    row.update(selected_evaluation_id='legacy',evaluation_status='completed',resolved=True)
    outcome={'status':'completed','resolved':True,'patch_sha256':row['patch_sha256']}
    report=tmp_path/'runs'/f['run']/'harness/legacy.json';write_json(report,{f['rid']:outcome})
    ev['evaluations']['legacy']={'report_path':str(report),'report_sha256':sha256(report),
                               'outcomes':{f['rid']:outcome},'provenance':{}}
    # Model a legacy completed score with a later observational invalid sidecar.
    # Keeping the old raw true must not allow this trajectory into new training.
    outcome['verifier_audit']={'classification':classification,'effective_resolved':None}
    write_json(ep,ev)
    destination=Path(build_dataset(f['cfg'],config_file)['path'])
    groups=list(iter_rows(destination/'groups.jsonl'));branches=list(iter_rows(destination/'branches.jsonl'))
    assert len(groups)==1 and len(branches)==6
    assert not groups[0]['data_ready'] and all(not b['data_ready'] for b in branches)
    assert all(b['historical_score'] is not None for b in branches)
    assert not list(iter_groups(destination))
    sources=list(iter_rows(destination/'rollouts.jsonl'))
    assert sources[0]['terminal_reward'] is None and sources[0]['resolved'] is None
    assert groups[0]['terminal_scoring_exclusion']
    with pytest.raises(ValueError,match='terminal_evaluation_ineligible'):
        GroupBatch.from_dataset_group({'group':groups[0],'members':branches},as_tensors=False,collate=False)
