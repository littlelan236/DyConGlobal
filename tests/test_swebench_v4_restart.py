import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io import read_json, sha256, write_json


def test_release_preflight_checks_all_candidates_before_first_artifact(tmp_path, monkeypatch):
    from runtime import wheelhouse
    requests = []
    def fetch(url, **kwargs):
        requests.append(url)
        files = [] if '/missing/' in url else [{'filename':'demo-1.0-py3-none-any.whl',
            'packagetype':'bdist_wheel','url':'https://files.pythonhosted.org/demo.whl',
            'size':1,'digests':{'sha256':'0'*64}}]
        return io.BytesIO(json.dumps({'urls':files}).encode())
    monkeypatch.setattr(wheelhouse.urllib.request, 'urlopen', fetch)
    with pytest.raises(ValueError, match='No published wheel for missing==1.0'):
        wheelhouse.download_releases(tmp_path, [('demo','1.0'),('missing','1.0')], ['cp39'])
    assert len(requests) == 2 and all(u.endswith('/json') for u in requests)
    assert not list(tmp_path.iterdir())


def test_unavailable_release_does_not_create_publication_staging(tmp_path, monkeypatch):
    from runtime import wheelhouse
    plan=tmp_path/'download.json'
    write_json(plan, {'schema_version':1,'releases':[{'name':'missing','version':'1.0'}],'python_tags':['cp39']})
    monkeypatch.setattr(wheelhouse.urllib.request, 'urlopen', lambda *a,**k:io.BytesIO(b'{"urls":[]}'))
    with pytest.raises(ValueError, match='No published wheel'):
        wheelhouse.build({'project_root':str(tmp_path)},'missing-wheel', download_plan=plan)
    assert not (tmp_path/'temp/wheelhouse/missing-wheel').exists()
    assert not (tmp_path/'runtime/wheelhouse/missing-wheel').exists()


@pytest.mark.parametrize('broken', [None, 'digest', 'audit', 'cleanup', 'tasks', 'source'])
def test_reused_baseline_requires_bound_complete_evidence(tmp_path, monkeypatch, broken):
    from benchmark_evaluation import swebench_repair_suite as suite
    task={'instance_id':'a','test_patch':'unchanged','environment_overrides':{}}
    payload={'run_id':'baseline','records':{'a':{'gold_valid':True}}, 'execution_finished':True,
        'audit_complete':True,'cleanup':{'checked':True,'cleanup_complete':True},'gold_valid_count':1}
    if broken=='audit':payload['audit_complete']=False
    if broken=='cleanup':payload['cleanup']['cleanup_complete']=False
    if broken=='tasks':payload['records']={}
    path=tmp_path/'temp/baseline/summary.json';write_json(path,payload)
    plan={'reuse_baseline_batches':{'collection-smoke':{'path':str(path),'sha256':sha256(path)}}}
    if broken=='digest':path.write_text('{}')
    old=copy.deepcopy(task)
    if broken=='source':old['test_patch']='changed'
    monkeypatch.setattr(suite,'load_plan',lambda *_:{'config':{'selection':{'tasks':{'key':{'instance_id':'a','instance':old}}}}})
    if broken:
        with pytest.raises(ValueError):suite.reuse_batch(tmp_path,plan,'collection-smoke',['a'],{'a':task})
    else:
        result=suite.reuse_batch(tmp_path,plan,'collection-smoke',['a'],{'a':task})
        assert result['reused'] is True and result['gold_valid_count']==1


def test_dependency_gate_runs_before_evaluation_and_preserves_failed_receipt(tmp_path, monkeypatch):
    from benchmark_evaluation import swebench_repair_suite as suite
    from core.io import hash_json
    plan=tmp_path/'input.json'
    write_json(plan, {'schema_version':1,'suite_id':'dependency-gate','source_run':'source',
        'task_count':1,'task_ids_sha256':hash_json(['a']),'wheelhouse_id':'fresh',
        'wheel_source':str(tmp_path/'wheels')})
    monkeypatch.setattr(suite,'load_plan',lambda *_:{'config':{'selection':{'tasks':{'key':{
        'instance_id':'a','instance':{'instance_id':'a'}}}},'environment':{'wheelhouse':'old'}},'run':tmp_path/'runs/source'})
    calls=[]
    class Process:
        pid=123
        def __init__(self,args,**kwargs):calls.append(args)
        def wait(self,**kwargs):return 1
    monkeypatch.setattr(suite.subprocess,'Popen',Process)
    with pytest.raises(RuntimeError,match='wheelhouse-build failed'):
        suite.run(tmp_path,plan)
    assert len(calls)==1 and calls[0][1:3]==['wheelhouse','build']
    receipt=read_json(tmp_path/'temp/dependency-gate/completion.json')
    assert receipt['state']=='failed' and receipt['batches']==[]
