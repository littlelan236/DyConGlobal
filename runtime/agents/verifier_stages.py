"""Protected verifier artifacts bound to the trusted executor's call identity."""
from __future__ import annotations

import json
from pathlib import Path
import shlex

from core.io import hash_json,now,sha256,write_json

ARTIFACTS={
    'simpo-code-reproduction':['simpo-stages.json'],
    'latex-formula-extraction':['mathjax-initialization.json'],
    'fix-build-agentops':['agentops-build.json','agentops-ci-stages.jsonl','agentops-tox-result.json'],
}


async def capture(owner,call_id):
    task=owner.cfg['selection']['tasks'][owner.sample.task_key]
    if task['instance'].get('verifier_adaptation',{}).get('policy')!='skillsbench-verifier-repair-v1':
        return
    names=ARTIFACTS.get(owner.sample.instance_id,[])
    if not names:return
    root=owner.output/'diagnostics/verifier-stage-artifacts'/call_id
    root.mkdir(parents=True,exist_ok=True)
    row={'schema_version':1,'source':'trusted_executor_download_after_verifier',
         'ownership':owner.ownership,'call_id':call_id,'task_package_sha256':task['instance']['package_sha256'],
         'dependency_snapshot_sha256':owner.cfg['execution_context']['profile'].get('dependency_snapshot_sha256'),
         'verifier_source_sha256':sha256(Path(task['instance']['task_path'])/'verifier/test_outputs.py'),
         'artifacts':{},'at':now()}
    for name in names:
        source='/logs/verifier/'+name
        probe=await owner.env.exec('test -f '+shlex.quote(source)+' && test ! -L '+shlex.quote(source)+
            ' && stat -c %s '+shlex.quote(source),user='root',timeout_sec=15)
        if probe.return_code:continue
        if not (probe.stdout or '').strip().isdigit() or int(probe.stdout)>2*1024*1024:
            row['artifacts'][name]={'error':'artifact_size_exceeds_bound'};continue
        dest=root/name
        await owner.env.download_file(source,dest)
        entry={'path':dest.relative_to(owner.output).as_posix(),'sha256':sha256(dest),'size':dest.stat().st_size}
        try:
            if name.endswith('.jsonl'):
                data=[json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
            else:data=json.loads(dest.read_text())
            entry['data']=data
            if name=='agentops-tox-result.json':entry['environments']=sorted(data.get('testenvs',{}))
        except (UnicodeError,ValueError) as exc:entry['error']=type(exc).__name__
        row['artifacts'][name]=entry
    row['receipt_sha256']=hash_json(row)
    write_json(owner.output/'diagnostics/verifier-stages.json',row)


def simpo_evidence(snap):
    """A new package's failure gate needs the protected reference and executor proof."""
    if 'simpo_independent_reference' not in snap['contract']['capabilities']:return None
    docs=snap['documents'];receipt=docs.get('diagnostics/verifier-stages.json',{})
    execution=docs.get('diagnostics/verifier-execution.json',{})
    calls=execution.get('attempts',[]);call=calls[-1] if calls else {}
    package=snap['contract']['package_sha256']
    source=next((f['sha256'] for f in snap['contract']['files'] if f['path']=='verifier/test_outputs.py'),None)
    if (not receipt or receipt.get('source')!='trusted_executor_download_after_verifier'
            or receipt.get('task_package_sha256')!=package or receipt.get('verifier_source_sha256')!=source
            or not receipt.get('call_id') or receipt['call_id']!=call.get('call_id')
            or receipt.get('ownership')!=execution.get('ownership')
            or receipt.get('dependency_snapshot_sha256')!=call.get('dependency_snapshot_sha256')
            or hash_json({k:v for k,v in receipt.items() if k!='receipt_sha256'})!=receipt.get('receipt_sha256')):
        return {'verified':False,'reason':'protected_stage_evidence_missing_or_unbound'}
    entry=receipt.get('artifacts',{}).get('simpo-stages.json',{})
    data=entry.get('data',{})
    if (entry.get('error') or data.get('policy')!='simpo-independent-reference-v1'
            or data.get('outcome') not in {'passed','agent_failure','verifier_error'}):
        return {'verified':False,'reason':'protected_stage_artifact_invalid'}
    reference=next((r for r in data.get('stages',[]) if r.get('phase')=='reference'),{})
    expected=next((f['sha256'] for f in snap['contract']['files'] if f['path']=='verifier/loss.npz'),None)
    if data['outcome']!='verifier_error' and (reference.get('status')!='completed' or reference.get('sha256')!=expected):
        return {'verified':False,'reason':'independent_reference_not_attested'}
    return {'verified':True,**data,'call_id':receipt['call_id']}


def command_outcomes(snap):
    """Recovery requires the same call, suite, phase, interpreter and dependency identity."""
    document=snap['documents'].get('diagnostics/verifier-execution.json',{})
    groups={};issues=[]
    keys=('verifier_call_id','suite_id','stage_id','environment_id','interpreter_id',
          'dependency_set_sha256','dependency_snapshot_sha256','task_package_sha256')
    for row in document.get('commands',[]):
        if row.get('required') is not True:continue
        if (any(not row.get(k) for k in keys) or type(row.get('sequence')) is not int
                or type(row.get('return_code')) is not int
                or row.get('task_package_sha256')!=snap['contract']['package_sha256']):
            issues.append({'code':'required_stage_identity_incomplete'});continue
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    unresolved=[];recoveries=[]
    for identity,rows in groups.items():
        sequences=[r['sequence'] for r in rows]
        if len(set(sequences))!=len(sequences):
            issues.append({'code':'required_stage_order_unknown'});continue
        rows=sorted(rows,key=lambda r:r['sequence'])
        if rows[-1]['return_code']:
            unresolved.append(rows[-1])
        elif any(r['return_code'] for r in rows[:-1]):
            recoveries.append({'stage':rows[-1].get('phase'),'call_id':identity[0],
                'suite_id':identity[1],'environment_identity':identity[3],
                'interpreter_id':identity[4],'dependency_set_sha256':identity[5],
                'evaluation_recovered':rows[-1].get('phase')=='task_test',
                'verification':'ordered_same_required_command_and_environment',
                'sequences':sequences})
    return unresolved,recoveries,issues


def mathjax_failure(snap):
    receipt=bound_receipt(snap)
    if receipt is None:return None
    entry=receipt.get('artifacts',{}).get('mathjax-initialization.json',{})
    data=entry.get('data',{})
    if not entry.get('error') and data.get('initialized') is False:
        return {'call_id':receipt['call_id'],'resource_root':data.get('resource_root'),
                'reason':'trusted_mathjax_initialization_failed'}
    return None


def bound_receipt(snap):
    receipt=snap['documents'].get('diagnostics/verifier-stages.json',{})
    execution=snap['documents'].get('diagnostics/verifier-execution.json',{})
    calls=execution.get('attempts',[]);call=calls[-1] if calls else {}
    source=next((f['sha256'] for f in snap['contract']['files'] if f['path']=='verifier/test_outputs.py'),None)
    if (receipt.get('source')!='trusted_executor_download_after_verifier'
            or receipt.get('call_id')!=call.get('call_id') or not receipt.get('call_id')
            or receipt.get('ownership')!=execution.get('ownership')
            or receipt.get('task_package_sha256')!=snap['contract']['package_sha256']
            or receipt.get('verifier_source_sha256')!=source
            or receipt.get('dependency_snapshot_sha256')!=call.get('dependency_snapshot_sha256')
            or hash_json({k:v for k,v in receipt.items() if k!='receipt_sha256'})!=receipt.get('receipt_sha256')):
        return None
    return receipt


def agentops_dependency_failures(snap):
    if 'agentops_ci' not in snap['contract']['capabilities']:return []
    if 'agentops_matrix_v1' in snap['contract']['capabilities']:
        evidence=agentops_matrix_evidence(snap)
        if evidence['complete']:return []
        receipt=bound_receipt(snap) or {}
        return [{'call_id':receipt.get('call_id'),'environment':'required_matrix','sequence':0,
                 'stderr':'; '.join(evidence['issues']),'stage':'required_matrix'}]
    receipt=bound_receipt(snap)
    if receipt is None:return []
    entry=receipt.get('artifacts',{}).get('agentops-tox-result.json',{})
    if entry.get('error'):return []
    expected={'coverage','langchain','langchain-core','openai','psutil','pytest',
              'pytest-asyncio','pytest-mock','requests-mock','types-requests'}
    failures=[]
    for name,environment in entry.get('data',{}).get('testenvs',{}).items():
        if name not in {'py37','py38','py39','py310','py311','py312'}:continue
        python=environment.get('python',{})
        latest={}
        for sequence,command in enumerate(environment.get('setup',[])):
            args=command.get('command',[])
            if ('install' not in args or command.get('run_id')!='install_deps'
                    or {a.replace('_','-') for a in args[args.index('install')+1:]}!=expected
                    or type(command.get('retcode')) is not int):continue
            latest[hash_json(args)]=(sequence,command)
        for sequence,command in latest.values():
            if not command['retcode']:continue
            failures.append({'call_id':receipt['call_id'],'environment':name,'sequence':sequence,
                'interpreter_id':hash_json(python),'interpreter':python,
                'dependency_set_sha256':hash_json(sorted(expected)),
                'dependency_snapshot_sha256':receipt['dependency_snapshot_sha256'],
                'task_package_sha256':receipt['task_package_sha256'],
                'return_code':command['retcode'],'stdout':command.get('output'),
                'stderr':command.get('err'),'stage':'install_deps'})
    return failures


def agentops_matrix_evidence(snap):
    """Require protected, call-bound stages for the revised contract only."""
    if 'agentops_matrix_v1' not in snap['contract']['capabilities']:return None
    from runtime.agents.agentops_matrix import validate_matrix
    receipt=bound_receipt(snap)
    if receipt is None or not receipt.get('dependency_snapshot_sha256'):
        return {'complete':False,'passed':False,'issues':['protected_matrix_receipt_missing_or_unbound']}
    artifacts=receipt.get('artifacts',{})
    names=['agentops-tox-result.json','agentops-build.json','agentops-ci-stages.jsonl']
    if any(n not in artifacts or artifacts[n].get('error') for n in names):
        return {'complete':False,'passed':False,'issues':['protected_matrix_artifact_missing_or_invalid']}
    try:
        result=validate_matrix(artifacts[names[0]].get('data'),snap['contract']['required_environments'])
    except (AttributeError,TypeError,KeyError,IndexError,ValueError):
        return {'complete':False,'passed':False,'issues':['protected_matrix_report_schema_invalid']}
    ci=artifacts[names[2]].get('data',[])
    if not isinstance(ci,list) or not isinstance(artifacts[names[1]].get('data'),dict):
        return {'complete':False,'passed':False,'issues':['protected_ci_report_schema_invalid']}
    stages={r.get('phase'):r.get('return_code') for r in ci if isinstance(r,dict)}
    build=artifacts[names[1]].get('data',{})
    if (stages.get('setup-python')!=0 or stages.get('pip-install-tox')!=0
            or type(stages.get('tox-matrix')) is not int
            or bool(stages.get('tox-matrix'))!=bool(build.get('return_code'))
            or type(build.get('return_code')) is not int
            or (result['complete'] and bool(build['return_code'])==result['passed'])):
        result['issues'].append('required_ci_stage_missing_or_inconsistent')
        result.update(complete=False,passed=False)
    result['call_id']=receipt['call_id']
    return result
