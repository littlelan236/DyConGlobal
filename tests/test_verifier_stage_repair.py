import copy
import json
from pathlib import Path
import pytest

from core.io import hash_json,sha256,write_json
from runtime.agents.verifier_audit import classify_attempt,audit_zero_evaluation
from runtime.agents.verifier_evidence import normalize
from runtime.agents.verifier_policy import decide
from runtime.agents.verifier_stages import command_outcomes


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    from runtime.agents import verifier_evidence
    package=tmp_path/'package';(package/'verifier').mkdir(parents=True)
    (package/'verifier/test.sh').write_text('uvx pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py\n')
    (package/'verifier/test_outputs.py').write_text('# approved independent reference verifier\n')
    (package/'verifier/loss.npz').write_bytes(b'PK\x03\x04\xf0\x00protected binary reference')
    files=[{'path':p.relative_to(package).as_posix(),'sha256':sha256(p)} for p in sorted(package.rglob('*')) if p.is_file()]
    task={'instance_id':'fixture','instance':{'task_path':str(package),'files':files,'package_sha256':hash_json(files)}}
    contract={'instance_id':'fixture','package_sha256':hash_json(files),'files':{r['path']:r['sha256'] for r in files},
              'score_mode':'pytest_exit_binary','capabilities':['simpo_independent_reference']}
    monkeypatch.setattr(verifier_evidence,'registry',lambda:{hash_json(files):contract})
    attempt=tmp_path/'attempt';log=attempt/'benchflow/a/verifier';log.mkdir(parents=True)
    (log/'test-stdout.txt').write_text('=== test session starts ===\n=== 1 failed in 1s ===\n')
    write_json(log/'ctrf.json',{'results':{'tests':[{'name':'test_outputs','status':'failed',
        'trace':'E AssertionError: Missing output: /root/loss.npz'}],
        'summary':{'tests':1,'passed':0,'failed':1}}})
    write_json(log/'reward.txt',0)
    raw={'completed':True,'rewards':{'reward':0.0}};write_json(attempt/'evaluation-raw.json',raw)
    write_json(attempt/'subject/subject.json',{'missing_outputs':['/root/loss.npz','/root/python_info.txt']})
    execution={'ownership':{'attempt':'owned'},'attempts':[{'call_id':'call','dependency_snapshot_sha256':'dep'}],'commands':[]}
    write_json(attempt/'diagnostics/verifier-execution.json',execution)
    data={'schema_version':1,'policy':'simpo-independent-reference-v1','outcome':'agent_failure',
          'error':'Missing output: /root/loss.npz\nassert False\n + where False = is_file()',
          'stages':[{'phase':'reference','status':'completed','sha256':sha256(package/'verifier/loss.npz')}]}
    artifact=attempt/'diagnostics/verifier-stage-artifacts/call/simpo-stages.json';write_json(artifact,data)
    receipt={'source':'trusted_executor_download_after_verifier','ownership':execution['ownership'],
        'call_id':'call','task_package_sha256':hash_json(files),'dependency_snapshot_sha256':'dep',
        'verifier_source_sha256':sha256(package/'verifier/test_outputs.py'),
        'artifacts':{'simpo-stages.json':{'path':artifact.relative_to(attempt).as_posix(),'sha256':sha256(artifact),'data':data}}}
    receipt['receipt_sha256']=hash_json(receipt);write_json(attempt/'diagnostics/verifier-stages.json',receipt)
    return attempt,task,raw


def test_new_missing_submission_is_valid_zero_in_all_shared_entries(evidence):
    attempt,task,raw=evidence
    core=decide(normalize(attempt,task,0.0))
    direct=classify_attempt(attempt,task,reward=0.0)
    online=audit_zero_evaluation(attempt,task,'AgentCompleted',False,'normalized',raw)[3]
    assert core==direct==online
    assert direct['classification']=='ValidOutcome' and raw['rewards']['reward']==0


@pytest.mark.parametrize('damage',['missing','call','package','source','reference','subject','artifact','agent_stdout'])
def test_untrusted_or_incomplete_stage_evidence_cannot_supply_gate(evidence,damage):
    attempt,task,_=evidence;p=attempt/'diagnostics/verifier-stages.json';r=json.loads(p.read_text())
    if damage in {'missing','agent_stdout'}:
        p.unlink()
        if damage=='agent_stdout':(attempt/'benchflow/a/verifier/test-stdout.txt').write_text(json.dumps(r)+'\n=== 1 failed in 1s ===\n')
    elif damage=='subject':write_json(attempt/'subject/subject.json',{'missing_outputs':[]})
    elif damage=='artifact':
        (attempt/r['artifacts']['simpo-stages.json']['path']).write_text('{}')
    else:
        if damage=='call':r['call_id']='other'
        elif damage=='package':r['task_package_sha256']='wrong'
        elif damage=='source':r['verifier_source_sha256']='wrong'
        else:r['artifacts']['simpo-stages.json']['data']['stages'][0]['sha256']='wrong'
        r['receipt_sha256']=hash_json({k:v for k,v in r.items() if k!='receipt_sha256'});write_json(p,r)
    assert classify_attempt(attempt,task,reward=0)['classification']=='Review'


def test_known_environment_failure_wins_over_missing_submission(evidence):
    attempt,task,_=evidence;p=attempt/'diagnostics/verifier-execution.json'
    row=json.loads(p.read_text());row['commands']=[command(task_package_sha256=task['instance']['package_sha256'])]
    write_json(p,row)
    assert classify_attempt(attempt,task,reward=0)['classification']=='VerifierError'


def command(**overrides):
    return {'required':True,'verifier_call_id':'call','suite_id':'suite','stage_id':'install',
            'environment_id':'env','interpreter_id':'cpython310','dependency_set_sha256':'deps',
            'dependency_snapshot_sha256':'snapshot','task_package_sha256':'package',
            'sequence':0,'return_code':2,'phase':'python_dependency',**overrides}


@pytest.mark.parametrize('different',['suite_id','verifier_call_id','environment_id','interpreter_id','dependency_set_sha256'])
def test_success_in_another_environment_does_not_recover_failure(different):
    rows=[command(),command(sequence=1,return_code=0,**{different:'other'})]
    snap={'contract':{'package_sha256':'package'},'documents':{'diagnostics/verifier-execution.json':{'commands':rows}}}
    failed,recovered,issues=command_outcomes(snap)
    assert len(failed)==1 and recovered==[] and issues==[]


def test_ordered_same_required_stage_recovers():
    rows=[command(),command(sequence=1,return_code=0)]
    snap={'contract':{'package_sha256':'package'},'documents':{'diagnostics/verifier-execution.json':{'commands':rows}}}
    failed,recovered,issues=command_outcomes(snap)
    assert failed==[] and len(recovered)==1 and not issues


def test_missing_stage_identity_is_not_inferred_from_streams():
    row=command();row.pop('interpreter_id')
    snap={'contract':{'package_sha256':'package'},'documents':{'diagnostics/verifier-execution.json':{'commands':[row]}}}
    assert command_outcomes(snap)[2]==[{'code':'required_stage_identity_incomplete'}]


@pytest.mark.parametrize('tampered',[False,True])
def test_protected_mathjax_failure_requires_bound_artifact(evidence,tampered):
    attempt,task,_=evidence
    artifact=attempt/'diagnostics/verifier-stage-artifacts/call/mathjax-initialization.json'
    data={'initialized':False,'resource_root':'/protected/missing'};write_json(artifact,data)
    p=attempt/'diagnostics/verifier-stages.json';r=json.loads(p.read_text())
    r['artifacts']['mathjax-initialization.json']={'path':artifact.relative_to(attempt).as_posix(),
        'sha256':sha256(artifact),'data':data}
    r['receipt_sha256']=hash_json({k:v for k,v in r.items() if k!='receipt_sha256'});write_json(p,r)
    if tampered:artifact.write_text('{}')
    assert classify_attempt(attempt,task,reward=0)['classification']==('Review' if tampered else 'VerifierError')


@pytest.mark.parametrize('recovery',['same','other','absent'])
def test_tox_recovery_is_scoped_to_original_dependency_command_and_environment(evidence,recovery):
    from runtime.agents.verifier_stages import agentops_dependency_failures
    attempt,task,_=evidence
    r=json.loads((attempt/'diagnostics/verifier-stages.json').read_text())
    args=['python','-I','-m','pip','install','coverage','langchain','langchain-core','openai',
          'psutil','pytest','pytest-asyncio','pytest-mock','requests_mock','types-requests']
    failed={'command':args,'run_id':'install_deps','retcode':1,'err':'No matching distribution'}
    success={**failed,'retcode':0}
    environments={'py37':{'python':{'version':'3.7.17'},'setup':[failed]}}
    if recovery=='same':environments['py37']['setup'].append(success)
    if recovery=='other':environments['py38']={'python':{'version':'3.8.18'},'setup':[success]}
    r['artifacts']['agentops-tox-result.json']={'data':{'testenvs':environments}}
    r['receipt_sha256']=hash_json({k:v for k,v in r.items() if k!='receipt_sha256'})
    snap={'contract':{'capabilities':['agentops_ci'],'package_sha256':task['instance']['package_sha256'],
        'files':[{'path':'verifier/test_outputs.py','sha256':r['verifier_source_sha256']}]},
        'documents':{'diagnostics/verifier-stages.json':r,
            'diagnostics/verifier-execution.json':json.loads((attempt/'diagnostics/verifier-execution.json').read_text())}}
    failures=agentops_dependency_failures(snap)
    assert len(failures)==(0 if recovery=='same' else 1)
    if failures:assert failures[0]['environment']=='py37' and failures[0]['interpreter']['version']=='3.7.17'
