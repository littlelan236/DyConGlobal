"""Bounded synthetic parallel cohort I/O and shard-open measurements."""
import builtins
import copy
import io
import os
from pathlib import Path
import time

from core.io import hash_json,read_json,sha256,write_json
from application.execution.capture_publication import finalize_capture_publication
from test_capture_publication_scale import cohort,tree_bytes,CountedFile


def test_parallel_75_tasks_5_repetitions_2_branches(tmp_path,monkeypatch):
    import hidden.repair_points as repair
    sessions=375
    cfg,run_id,items=cohort(tmp_path,monkeypatch,sessions,[3]*sessions,[3]*sessions,75)
    run=tmp_path/'runs'/run_id
    nas=tmp_path/'nas/online'/run_id
    cfg['parallel']={'enabled':True,'branch_count':2}
    write_json(run/'resolved-config.json',cfg)
    evidence=read_json(run/'evidence.json')
    for rid,row in evidence['rollouts'].items():
        sid='cap-'+row['sample_key']+'-'+row['attempt_id']
        trajectory_path=tmp_path/row['trajectory_path']
        trajectory=read_json(trajectory_path)
        first=trajectory['messages'][0]
        md=first['extra']['request']['dycon_capture']
        md.update(decision_id='d-'+rid,branch_id=1,parent_prefix_sha256='prefix-'+rid)
        native=first['extra']['response']['id']
        first['extra']['response']['dycon_native_request_id']=native
        request_path=nas/'requests'/(native+'.json')
        request=read_json(request_path); request.update(md); write_json(request_path,request)
        loser={**request,'request_id':'loser-'+rid,'branch_id':0,'api_attempt_id':'api-loser-'+rid}
        write_json(nas/'requests'/('loser-'+rid+'.json'),loser)
        decision=dict(session_id=sid,decision_id=md['decision_id'],query_id=md['query_id'],
            parent_prefix_sha256=md['parent_prefix_sha256'],branch_count=2,selected_branch_id=1,step_id=0,
            candidates=[dict(branch_id=0,native_request_id=loser['request_id'],api_attempt_id=loser['api_attempt_id'],score=1),
                        dict(branch_id=1,native_request_id=native,api_attempt_id=md['api_attempt_id'],score=2)])
        first['extra']['dycon_parallel']=copy.deepcopy(decision)
        write_json(trajectory_path,trajectory)
        row['trajectory_sha256']=sha256(trajectory_path)
        index_path=nas/'index'/('shard-'+sid+'.json')
        events=read_json(index_path); events[0].update(md)
        loser_event={**events[0],'sample_id':'loser-point-'+rid,'request_id':loser['request_id'],
                     'branch_id':0,'api_attempt_id':loser['api_attempt_id']}
        events.append(loser_event);write_json(index_path,events)
        base=nas/'sessions'/sid/'parallel'
        write_json(base/'decisions'/(decision['decision_id']+'.json'),decision)
        rows=[dict(decision_id=decision['decision_id'],branch_id=c['branch_id'],selected=c['branch_id']==1,
            native_request_id=c['native_request_id'],score=c['score'],rollout_id=rid,trajectory_sha256=row['trajectory_sha256'])
              for c in decision['candidates']]
        write_json(base/'branch-index.json',rows);write_json(base/'branch-coverage.json',{'candidate_count':2})
        write_json(base/'branch-receipt.json',dict(rollout_id=rid,trajectory_sha256=row['trajectory_sha256'],
            files={str(p.relative_to(nas)):sha256(p) for p in base.rglob('*.json')}))
    write_json(run/'evidence.json',evidence)
    counts={'read_bytes':0,'write_bytes':0};opens=0
    original_tar=repair.tarfile.open
    def counted_tar(*args,**kwargs):
        nonlocal opens
        opens+=1
        return original_tar(*args,**kwargs)
    def opener(original):
        def counted(path,*args,**kwargs):
            stream=original(path,*args,**kwargs)
            return CountedFile(stream,counts) if isinstance(path,(str,Path)) and Path(path).is_relative_to(nas) else stream
        return counted
    before=tree_bytes(tmp_path); nas_before=tree_bytes(nas); stamp=time.monotonic()
    with monkeypatch.context() as audit:
        audit.setattr(repair.tarfile,'open',counted_tar)
        audit.setattr(builtins,'open',opener(builtins.open))
        audit.setattr(io,'open',opener(io.open))
        summary=finalize_capture_publication(cfg,run_id,items=items)
    elapsed=time.monotonic()-stamp
    assert summary['status']=='complete',summary
    assert summary['published_points']==1125 and summary['branch_points']==750
    assert opens==375
    report=dict(task_count=75,repetitions=5,branches=2,sessions=sessions,canonical_points=1125,
        candidate_points=750,seconds=elapsed,tar_opens=opens,logical_nas_io=counts,
        additional_storage_bytes=tree_bytes(tmp_path)-before,additional_nas_bytes=tree_bytes(nas)-nas_before,
        limitation='Synthetic 4-value float32 tensors and local filesystem paths; not physical NAS/GPU throughput')
    if os.environ.get('DYCON_CAPTURE_TEST_REPORT_ROOT'):
        write_json(Path(os.environ['DYCON_CAPTURE_TEST_REPORT_ROOT'])/'parallel-scale.json',report)
    print(report)
