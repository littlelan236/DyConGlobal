import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.io import hash_json, sha256, write_json
from hidden.artifacts import read_arrays, read_repaired_session
from hidden.repair_points import _tensor, recover_points


def fixture(tmp_path, monkeypatch, partial=False):
    import hidden.repair_points as points
    run='run'; sid='cap-s-a-attempt'; sp=tmp_path/'spool'/run
    sp.joinpath('raw').mkdir(parents=True)
    data=sp/'raw/sample.npz'; np.savez(data,layer_0=np.ones(4,dtype=np.float32))
    md={'session_id':sid,'run_id':run,'task_id':'task','query_id':'q','api_attempt_id':'api'}
    req={**md,'request_id':'native','anchors':['think_close'],'prompt_token_ids':[10],
         'output_token_ids':[11,20],'request_sha256':hash_json([10]),'output_sha256':hash_json([11,20]),
         'close_token_ids':[11],'open_token_ids':[12]}
    write_json(sp/'requests/native.json',req)
    event={**md,'request_id':'native','sample_id':'sample','anchor':'think_close','absolute_position':1,
           'token_id':11,'request_sha256':req['request_sha256'],'model_fingerprint':'model',
           'tensor':'raw/sample.npz','sha256':sha256(data),'status':'persisted',
           'storage_shapes':{'layer_0':[4]},'storage_dtypes':{'layer_0':'float32'}}
    write_json(sp/'index/sample.json',event)
    messages=[{'role':'assistant','extra':{'request':{'dycon_capture':md},'response':{'id':'native'}}}]
    if partial:
        second={**md,'query_id':'q2','api_attempt_id':'api2'}
        messages.append({'role':'assistant','extra':{'request':{'dycon_capture':second},'response':{'id':'missing'}}})
        event2={**event,**second,'request_id':'missing-worker','sample_id':'sample2'}
        write_json(sp/'index/sample2.json',event2)
    trajectory=tmp_path/'runs/run/trajectory.json'; write_json(trajectory,{'messages':messages})
    row={'run_id':run,'sample_key':'s-a','attempt_id':'attempt','task_key':'task-key',
         'trajectory_path':'runs/run/trajectory.json','trajectory_sha256':sha256(trajectory),'model_fingerprint':'model'}
    evidence={'rollouts':{'rollout':row},'tasks':{'task-key':{'instance_id':'task'}}}
    monkeypatch.setattr(points,'RunEvidence',lambda *a: SimpleNamespace(read=lambda:evidence))
    monkeypatch.setattr(points,'reasoning_targets',lambda *a: ([(1,'think_close')],{}))
    cfg={'project_root':str(tmp_path),'parallel':{'enabled':False},
         'capture':{'anchors':['think_close'],'layers':[0],
                    'export':{'spool_root':'spool','root':'online','upload_max_elapsed_seconds':30}},
         'paths':{'nas_root':str(tmp_path/'nas')}}
    write_json(tmp_path/'runs/run/resolved-config.json',cfg)
    return cfg,sid,sp,event


def test_complete_point_published_and_reader_checks_dependencies(tmp_path,monkeypatch):
    cfg,sid,sp,event=fixture(tmp_path,monkeypatch)
    original=sha256(sp/'raw/sample.npz')
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='repair')
    assert report['summary']['states']=={'complete':1}
    assert report['summary']['published_points']==1
    nas=tmp_path/'nas/online/run'
    session=read_repaired_session(nas,'repair',sid)
    assert session['points'][0]['recovery']['binding']=='native_exact'
    assert not session['coverage']['eligible_for_training']
    assert np.array_equal(read_arrays(nas,session['points'][0])['layer_0'],np.ones(4))
    assert sha256(sp/'raw/sample.npz')==original
    again=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='repair')
    assert again['summary']==report['summary']
    (nas/'raw/sample.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_repaired_session(nas,'repair',sid)


def test_missing_native_ledger_preserves_point_without_claiming_exactness(tmp_path,monkeypatch):
    cfg,sid,sp,event=fixture(tmp_path,monkeypatch,partial=True)
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='repair')
    assert report['summary']['states']=={'partial':1}
    assert report['summary']['binding_counts']=={'native_exact':1,'canonical_identity_only':1}
    assert report['summary']['published_points']==2
    c=report['sessions'][0]['coverage']
    assert c['expected_known']==1 and not c['expected_count_complete']
    assert len(c['unbound_responses'])==1 and c['missing_known']==[]


def test_corrupt_point_is_reported_without_losing_other_point(tmp_path,monkeypatch):
    cfg,sid,sp,event=fixture(tmp_path,monkeypatch)
    bad={**event,'sample_id':'bad','tensor':'raw/bad.npz'}
    (sp/'raw/bad.npz').write_bytes(b'bad tensor')
    write_json(sp/'index/bad.json',bad)
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='repair')
    assert report['summary']['published_points']==1
    assert report['summary']['rejected_points']==1


def test_point_identity_conflict_never_silently_deduplicates(tmp_path,monkeypatch):
    cfg,sid,sp,event=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Contradictory'):
        _tensor([(sp,event),(sp,{**event,'query_id':'conflict'})],tmp_path/'nas',cfg,{})
