import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark_evaluation.swebench_receipts import instrument, original_script, get_pytest_binding
from benchmark_evaluation.swebench_verifier import classify, digest, evidence_digest, failure_blocks, _inner_pytest, test_region as region
from test_swebench_verifier import base, failed_document


def session(d, *, runner='pytest'):
    binding={'token':'a'*32,'command_sha256':'c'*64,'plugin_sha256':'b'*64}
    tests=[];reports=[]
    for test,state in d['statuses'].items():
        tests.append(test)
        for phase in ['setup','call','teardown']:
            row=dict(nodeid=test,when=phase,outcome='failed' if phase=='call' and state=='FAILED' else 'passed')
            if row['outcome']=='failed':row['crash']={'message':'ValueError: bad value'}
            reports.append(row)
    value=dict(schema_version=1,binding=binding,pid=17,ppid=16,ancestors=[{'pid':17,'ppid':16}],
               executable=sys.executable,argv=['pytest'],cwd='/testbed',exitstatus=1,
               collected=tests,collection_errors=[],reports=reports)
    command=dict(command_sha256='c'*64,runner=runner,returncode=1,scope='command')
    d['test_output']+='\n__DYCON_TEST_RESULT__ '+json.dumps(command)+'\n__DYCON_PYTEST_SESSION__ '+json.dumps(value)+'\n'
    d['execution']=dict(returncode=0,test_command_sha256='c'*64,pytest_binding=binding,
                        output_sha256=hashlib.sha256(d['test_output'].encode()).hexdigest())
    return value


def finding(d):
    test=d['task']['FAIL_TO_PASS'][0]
    d['sources']=[{'path':'test_output.txt','sha256':hashlib.sha256(d['test_output'].encode()).hexdigest()}]
    block=failure_blocks(region(d['test_output'])[0])[0]
    d['findings']=[dict(reason_code='subject_call_exception',attribution='subject',stage='test',
        evidence_strength='confirmed',recovered=False,required_tests=[test],basis='Exact tested contract and buggy implementation reviewed',
        supporting_sources=d['sources'],evidence=d['sources'],evidence_digest=evidence_digest(d),
        exceptions={test:'ValueError'},failure_block_sha256={test:digest(block)})]


def test_did_not_warn_is_a_precise_pytest_expectation_failure():
    assert classify(failed_document(error='Failed: DID NOT WARN. No warnings of type UserWarning were emitted.'))['effective_resolved'] is False
    assert classify(failed_document(error='Failed: unknown failure'))['effective_resolved'] is None


def test_reviewed_subject_exception_needs_matching_live_call_and_sources():
    d=failed_document(error='ValueError: bad value');session(d);finding(d)
    assert classify(d)['classification']=='ValidOutcome'
    assert classify(d)['effective_resolved'] is False


@pytest.mark.parametrize('mutation',['output','digest','source','exception','block','phase','setup','duplicate','nested','rc2','timeout','gold','candidate'])
def test_subject_attribution_cannot_bypass_safeguards(mutation):
    d=failed_document(error='ValueError: bad value');session(d);finding(d)
    if mutation=='output':d['test_output']+='changed'
    elif mutation=='digest':d['findings'][0]['evidence_digest']='wrong'
    elif mutation=='source':d['findings'][0]['evidence'][0]['sha256']='wrong'
    elif mutation=='exception':d['findings'][0]['exceptions'][d['task']['FAIL_TO_PASS'][0]]='TypeError'
    elif mutation=='block':d['findings'][0]['failure_block_sha256']={}
    elif mutation=='gold':d['gold']=True
    elif mutation=='timeout':d['execution']['timed_out']=True;finding(d)
    elif mutation=='candidate':
        d['test_output']=d['test_output'].replace('ValueError: bad value','ValueError: Network is unreachable')
        d['execution']['output_sha256']=hashlib.sha256(d['test_output'].encode()).hexdigest();finding(d)
    else:
        prefix,raw=d['test_output'].split('__DYCON_PYTEST_SESSION__ ')
        row=json.loads(raw)
        if mutation=='phase':row['reports'][1]['when']='setup'
        elif mutation=='setup':row['reports'][0]['outcome']='failed'
        elif mutation=='duplicate':row['reports'].append(row['reports'][1])
        elif mutation=='rc2':row['exitstatus']=2
        d['test_output']=prefix+'__DYCON_PYTEST_SESSION__ '+json.dumps(row)+'\n'
        if mutation=='nested':d['test_output']+='__DYCON_PYTEST_SESSION__ '+json.dumps(row)+'\n'
        d['execution']['output_sha256']=hashlib.sha256(d['test_output'].encode()).hexdigest();finding(d)
    assert classify(d)['effective_resolved'] is None


def test_tox_receipt_requires_unique_complete_inner_session():
    d=failed_document();session(d,runner='tox')
    assert classify(d)['effective_resolved'] is False
    d['execution']['pytest_binding']['token']='wrong'
    assert classify(d)['effective_resolved'] is None


def test_real_pytest_plugin_and_unchanged_command(tmp_path):
    (tmp_path/'test_contract.py').write_text('def test_pass():\n    pass\ndef test_fail():\n    raise ValueError("bad value")\n')
    raw=f"#!/bin/bash\nset -uxo pipefail\n: '>>>>> Start Test Output'\n{sys.executable} -m pytest -rA -v {tmp_path}/test_contract.py\n: '>>>>> End Test Output'\ntrue\n"
    # Runner detection uses the same literal commands as the official harness.
    raw=raw.replace(sys.executable+' -m pytest','python -m pytest')
    wrapped,details=instrument(raw)
    assert original_script(wrapped)==raw
    result=subprocess.run(['bash'],input=wrapped,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          cwd=tmp_path,timeout=30,env={**os.environ,'PATH':str(Path(sys.executable).parent)+':'+os.environ['PATH']})
    assert result.returncode==0,result.stdout[-2000:]
    binding=get_pytest_binding(wrapped)
    folder=Path('/tmp/dycon-pytest-'+binding['token'])
    bodies=[json.loads(p.read_text()) for p in folder.glob('*.json') if p.stem.isdigit()]
    assert '__DYCON_PYTEST_SESSION__' not in result.stdout
    execution=dict(pytest_binding=binding,test_command_sha256=details['command_sha256'],output_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(), receipt_protocol=2, output_verified=True, pytest_sessions=bodies)
    inner=_inner_pytest(result.stdout,execution,{})
    assert inner is not None,result.stdout[-4000:]
    assert inner['exitstatus']==1 and len(inner['tests'])==2
    failure=next(v['call'] for v in inner['tests'].values() if v['call']['outcome']=='failed')
    assert failure['crash']['message']=='ValueError: bad value'


def test_resources_reject_hash_mismatch_and_target_escape(tmp_path):
    from runtime.containers.resources import assets, preparation
    source=tmp_path/'runtime/resources/v1/data.txt';source.parent.mkdir(parents=True);source.write_bytes(b'upstream\x00data')
    row=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),destination='/testbed/pvlib/data/data.txt')
    cfg={'project_root':str(tmp_path),'environment':{'variables':{'DYCON_SWEBENCH_RESOURCES':json.dumps([row])}}}
    assert len(assets(cfg))==1
    assert 'python -c' in preparation(cfg)
    row['destination']='/testbed/../etc/data.txt';cfg['environment']['variables']['DYCON_SWEBENCH_RESOURCES']=json.dumps([row])
    with pytest.raises(ValueError):assets(cfg)
    row['destination']='/testbed/pvlib/data/data.txt'
    row['sha256']='bad';cfg['environment']['variables']['DYCON_SWEBENCH_RESOURCES']=json.dumps([row])
    with pytest.raises(ValueError):assets(cfg)


def test_fixture_error_collection_failure_and_missing_data_cannot_become_subject_zero():
    for error in ['FileNotFoundError: /testbed/missing.txt','ImportError: missing dependency',
                  'TypeError: unexpected fixture error','ValueError: bad value']:
        d=failed_document(error=error,state='ERROR');session(d)
        assert classify(d)['effective_resolved'] is None


def test_verified_gold_nonrequired_failure_keeps_real_command_exit_guard():
    d=base();d['gold']=True;d['gold_provenance']={'verified':True}
    command=dict(command_sha256='c'*64,runner='pytest',returncode=1,scope='command')
    d['test_output']+='\n__DYCON_TEST_RESULT__ '+json.dumps(command)+'\n'
    d['execution']={'returncode':0,'test_command_sha256':'c'*64,'output_sha256':hashlib.sha256(d['test_output'].encode()).hexdigest()}
    result=classify(d)
    assert result['classification']=='Review'
    assert 'required_pass_nonrequired_failure' in result['blocker_codes']
