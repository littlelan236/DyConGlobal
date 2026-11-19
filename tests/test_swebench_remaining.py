import copy
import json
from types import SimpleNamespace

import pytest

from core.io import read_json, sha256, write_json, hash_json
from benchmark_evaluation import swebench_repair_suite as suite
from benchmark_evaluation.swebench_test_identity import KEY, parser_for, validate_collection


@pytest.mark.parametrize('broken', [None, 'digest', 'audit', 'cleanup', 'selection', 'source'])
def test_failed_only_is_bound_and_excludes_previously_passed(tmp_path, broken):
    baseline = {'run_id':'source', 'execution_finished':True, 'audit_complete':True,
                'cleanup':{'cleanup_complete':True},
                'records':{'a':{'gold_valid':True}, 'b':{'gold_valid':False}}}
    if broken == 'audit': baseline['audit_complete'] = False
    if broken == 'cleanup': baseline['cleanup']['cleanup_complete'] = False
    if broken == 'source': baseline['run_id'] = 'other'
    p = tmp_path/'temp/ledger.json'; write_json(p, baseline)
    plan = {'source_run':'source', 'task_ids':['b'],
            'failed_only_baseline':{'path':str(p), 'sha256':sha256(p)}}
    if broken == 'digest': write_json(p, {})
    if broken == 'selection': plan['task_ids'] = ['a','b']
    if broken:
        with pytest.raises(ValueError): suite.selected_task_ids(tmp_path,plan,{'a':{},'b':{}})
    else:
        assert suite.selected_task_ids(tmp_path,plan,{'a':{},'b':{}}) == ['b']


def identity_task():
    frozen = 'tests.py::Test::test_pair[/n/fs/p-swe-bench/temp/tmp123/pydicom__pydicom__1.3/pydicom/data/test_files/a.dcm-/n/fs/p-swe-bench/temp/tmp123/pydicom__pydicom__1.3/pydicom/data/test_files/b.dcm]'
    raw = 'tests.py::Test::test_pair[/testbed/pydicom/data/test_files/a.dcm-/testbed/pydicom/data/test_files/b.dcm]'
    task = {'repo':'pydicom/pydicom','FAIL_TO_PASS':[frozen],'PASS_TO_PASS':[],
            'environment_overrides':{'variables':{KEY:json.dumps([{'required':frozen,'raw':raw}])}}}
    return task, frozen, raw


def test_path_contract_preserves_both_parameters_and_terminal_failure():
    task, frozen, raw = identity_task()
    parse = parser_for(task, lambda *_:{'truncated[/b.dcm]':'PASSED'})
    assert parse('FAILED '+raw, None)[frozen] == 'FAILED'
    assert frozen not in parse('PASSED '+raw.replace('/a.dcm-', '/wrong.dcm-'), None)
    with pytest.raises(ValueError,match='Ambiguous physical'):
        parse('PASSED '+raw+'\nFAILED '+raw,None)


@pytest.mark.parametrize('broken',['method','parameter','collision','unknown_required'])
def test_path_contract_rejects_unsafe_mapping(broken):
    task, frozen, raw = identity_task()
    rows=json.loads(task['environment_overrides']['variables'][KEY])
    if broken == 'method': rows[0]['raw']=raw.replace('test_pair','test_other')
    if broken == 'parameter': rows[0]['raw']=raw.replace('/a.dcm-','/wrong.dcm-')
    if broken == 'collision': rows.append(copy.deepcopy(rows[0]))
    if broken == 'unknown_required': task['FAIL_TO_PASS']=[]
    task['environment_overrides']['variables'][KEY]=json.dumps(rows)
    with pytest.raises(ValueError): parser_for(task,lambda *_:{})


def test_path_contract_requires_current_bound_collection_and_all_phases():
    task, frozen, raw = identity_task()
    binding={'token':'test'}
    session={'binding':binding,'collected':[raw], 'reports':[
        {'nodeid':raw,'when':phase} for phase in ['setup','call','teardown']]}
    execution={'pytest_sessions':[session],'pytest_binding':binding}
    validate_collection(task,execution)
    session['reports'].pop()
    with pytest.raises(ValueError,match='complete physical phases'):validate_collection(task,execution)


def test_resource_transform_is_limited_to_official_california_checksum(tmp_path, monkeypatch):
    from runtime.containers import resources
    p=tmp_path/'runtime/resources/california.tgz';p.parent.mkdir(parents=True);p.write_bytes(b'tampered')
    row={'source':str(p),'destination':'/testbed/.cache/sklearn/cal_housing_py3.pkz',
         'sha256':sha256(p),'transform':'sklearn-california-v1','loader_sha256':'a'*64}
    cfg={'project_root':str(tmp_path),'environment':{'variables':{'DYCON_SWEBENCH_RESOURCES':json.dumps([row])}}}
    with pytest.raises(ValueError,match='California housing source contract'):resources.assets(cfg)
    row['transform']='arbitrary-code'
    cfg['environment']['variables']['DYCON_SWEBENCH_RESOURCES']=json.dumps([row])
    with pytest.raises(ValueError,match='Unsupported resource transformation'):resources.assets(cfg)


def test_disabled_negative_never_loads_or_executes_negative_and_only_runs_failed(tmp_path,monkeypatch):
    source=tmp_path/'runs/source'; source.mkdir(parents=True)
    tasks={i:{'instance_id':i,'instance':{'instance_id':i,'repo':'test/repo'}} for i in ['a','b']}
    cfg={'selection':{'tasks':tasks},'environment':{'wheelhouse':'old','variables':{}}}
    baseline=tmp_path/'temp/baseline.json';write_json(baseline,{'run_id':'source',
        'execution_finished':True,'audit_complete':True,'cleanup':{'cleanup_complete':True},
        'records':{'a':{'gold_valid':True},'b':{'gold_valid':False}}})
    write_json(source/'harness/selection.json',{'source_run_id':'submission-source',
        'tasks':tasks,'rollouts':{'r-b':{'task_key':'b'}}})
    plan={'schema_version':1,'suite_id':'remaining','source_run':'source',
          'task_count':1,'task_ids':['b'],'task_ids_sha256':hash_json(['b']),
          'failed_only_baseline':{'path':str(baseline),'sha256':sha256(baseline)},
          'run_negative':False,'final_batch':'remaining-gold', 'stage_timeout_seconds':100,
          'wheelhouse_id':'fresh','wheel_source':'wheels','task_settings':{},
          'collection_smoke':[],'timeout_diagnostics':[],'experiments':[],'image_repairs':[]}
    file=tmp_path/'input.json';write_json(file,plan)
    loaded=[]
    def load(root,rid):
        loaded.append(rid)
        assert rid=='source'
        return {'config':cfg,'run':source}
    monkeypatch.setattr(suite,'load_plan',load)
    monkeypatch.setattr(suite,'publish_task_settings',lambda *_:None)
    calls=[]
    class Process:
        pid=123
        def __init__(self,args,**kwargs):
            calls.append(args)
            if 'prepare' in args:
                write_json(tmp_path/'runs/remaining-remaining-gold/harness/selection.json',{'tasks':{'b':tasks['b']}})
            if 'acceptance' in args:
                write_json(args[args.index('--output')+1]+'/summary.json',{'run_id':'remaining-remaining-gold',
                    'gold_valid_count':1,'records':{'b':{'gold_valid':True}}})
        def wait(self,**kwargs):return 0
    monkeypatch.setattr(suite.subprocess,'Popen',Process)
    result=suite.run(tmp_path,file)
    evaluations=[args for args in calls if args[1:3]==['harness','run']]
    assert len(evaluations)==1 and '--gold' in evaluations[0]
    creation=next(args for args in calls if 'create' in args)
    assert creation[creation.index('--task-id')+1:] == ['b']
    assert result['task_count']==1 and result['gold_valid_count']==1
    assert read_json(tmp_path/'temp/remaining/pair-results.json')['negative_enabled'] is False
