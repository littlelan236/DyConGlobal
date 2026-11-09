"""Parallel repair binds selected and candidate evidence independently."""
import copy
import os
from pathlib import Path

import pytest

from core.io import hash_json, read_json, sha256, write_json
from hidden.artifacts import read_repaired_session
from hidden.repair_points import recover_points
from test_partial_publication import sequence


def parallel_fixture(tmp_path, monkeypatch, *, loser_hidden=False):
    cfg, evidence = sequence(tmp_path, monkeypatch, None)
    cfg['parallel'] = {'enabled': True, 'branch_count': 2}
    write_json(tmp_path / 'runs/run/resolved-config.json', cfg)
    spool = tmp_path / 'spool/run'
    sid = 'cap-s-a-attempt'
    trajectory_path = tmp_path / 'runs/run/trajectory.json'
    trajectory = read_json(trajectory_path)
    md = trajectory['messages'][0]['extra']['request']['dycon_capture']
    md.update(decision_id='d', branch_id=1, parent_prefix_sha256='prefix')
    response = trajectory['messages'][0]['extra']['response']
    response['dycon_native_request_id'] = 'native-0'
    winner = read_json(spool / 'requests/native-0.json')
    winner.update(decision_id='d', branch_id=1, parent_prefix_sha256='prefix')
    write_json(spool / 'requests/native-0.json', winner)
    write_json(tmp_path / 'nas/online/run/requests/native-0.json', winner)
    winner_event = read_json(spool / 'index/0.json')
    winner_event.update(decision_id='d', branch_id=1)
    write_json(spool / 'index/0.json', winner_event)
    write_json(tmp_path / 'nas/online/run/index/0.json', winner_event)
    loser = {**winner, 'request_id': 'loser', 'branch_id': 0, 'api_attempt_id': 'other'}
    write_json(spool / 'requests/loser.json', loser)
    if loser_hidden:
        event = {**winner_event, 'sample_id': 'loser-point', 'request_id': 'loser',
                 'branch_id': 0, 'api_attempt_id': 'other'}
        write_json(spool / 'index/loser.json', event)
    candidates = [{'branch_id': 0, 'native_request_id': 'loser', 'api_attempt_id': 'other', 'score': 1,'status':'eligible'},
                  {'branch_id': 1, 'native_request_id': 'native-0', 'api_attempt_id': 'api0', 'score': 2,'status':'eligible'}]
    decision = {'session_id': sid, 'decision_id': 'd', 'query_id': 'q0',
                'parent_prefix_sha256': 'prefix', 'branch_count': 2,'step_id':0,
                'selected_branch_id': 1, 'candidates': candidates,'target':'work','endpoint':'terminal'}
    trajectory['messages'][0]['extra']['dycon_parallel'] = copy.deepcopy(decision)
    write_json(trajectory_path, trajectory)
    evidence['rollouts']['rollout']['trajectory_sha256'] = sha256(trajectory_path)
    base=spool/'sessions'/sid/'parallel'
    write_json(base/'decisions/d.json', decision)
    rows=[]
    for candidate in candidates:
        rows.append({'decision_id':'d','branch_id':candidate['branch_id'],
            'selected':candidate['branch_id']==1,'native_request_id':candidate['native_request_id'],
            'score':candidate['score'],'rollout_id':'rollout','trajectory_sha256':sha256(trajectory_path)})
    write_json(base/'branch-index.json',rows)
    write_json(base/'branch-coverage.json',{'candidate_count':2})
    files={str(p.relative_to(spool)):sha256(p) for p in base.rglob('*.json')}
    write_json(base/'branch-receipt.json',{'rollout_id':'rollout','trajectory_sha256':sha256(trajectory_path),'files':files})
    return cfg, spool, sid


@pytest.mark.parametrize('loser_hidden', [False, True])
def test_parallel_selected_is_canonical_and_candidate_is_independent(tmp_path, monkeypatch, loser_hidden):
    cfg, spool, sid = parallel_fixture(tmp_path, monkeypatch, loser_hidden=loser_hidden)
    report = recover_points(cfg, 'run', session_ids=[sid], apply=True, repair_id='parallel',
                            visualization_parallel=True)
    assert not report['input_issues']
    coverage = report['sessions'][0]['coverage']
    assert coverage['canonical_coverage']['status'] == 'complete'
    assert coverage['all_candidate_coverage']['status'] == ('complete' if loser_hidden else 'partial')
    points = read_repaired_session(tmp_path / 'nas/online/run', 'parallel', sid)['points']
    winner = next(point for point in points if point['sample_id'] == 'sample-0')
    assert winner['recovery']['binding'] == 'native_exact'
    assert winner['recovery']['branch_id'] == 1 and winner['recovery']['actual_path']
    if loser_hidden:
        loser = next(point for point in points if point['sample_id'] == 'loser-point')
        assert loser['recovery']['binding'] == 'native_exact_candidate'
        assert loser['recovery']['actual_path'] is False


def test_parallel_conflicting_winner_cannot_promote_canonical_points(tmp_path, monkeypatch):
    cfg, spool, sid = parallel_fixture(tmp_path, monkeypatch)
    path = spool / 'sessions' / sid / 'parallel/decisions/d.json'
    decision = read_json(path)
    decision['selected_branch_id'] = 0
    write_json(path, decision)
    result = recover_points(cfg, 'run', session_ids=[sid], visualization_parallel=True)
    assert 'checksum mismatch' in result['sessions'][0]['coverage']['canonical_error']
    assert result['sessions'][0]['coverage']['observed_native_exact'] == 0


def test_canonical_ledger_hash_conflict_never_promotes_winner(tmp_path,monkeypatch):
    from hidden.repair_points import _bindings
    sid='session'
    md={'session_id':sid,'run_id':'run','task_id':'task','query_id':'q','api_attempt_id':'a'}
    request={**md,'request_id':'native','anchors':['think_close'],'prompt_token_ids':[10],
             'output_token_ids':[11,20],'close_token_ids':[11],'open_token_ids':[10],
             'ledger_sha256':'correct'}
    response={'dycon_native_request_id':'native','ledger_sha256':'conflicting'}
    message={'extra':{'request':{'dycon_capture':md}}}
    bound,_,unbound,_=_bindings(sid,'task','run',[(0,message,response)],[request])
    assert bound=={} and unbound[0]['reason']=='Canonical/native ledger checksum disagreement'


def test_missing_winner_never_uses_loser_tensor(tmp_path,monkeypatch):
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    for base in (spool,tmp_path/'nas/online/run'):
        path=base/'index/0.json'
        if path.exists():path.unlink()
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='missing-winner',visualization_parallel=True)
    coverage=report['sessions'][0]['coverage']
    assert coverage['canonical_coverage']['status']=='partial'
    assert coverage['all_candidate_coverage']['status']=='partial'
    points=read_repaired_session(tmp_path/'nas/online/run','missing-winner',sid)['points']
    assert not any(p['sample_id']=='sample-0' for p in points)
    assert next(p for p in points if p['sample_id']=='loser-point')['recovery']['actual_path'] is False


def test_corrupt_loser_does_not_discard_proven_winner(tmp_path,monkeypatch):
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    event=read_json(spool/'index/loser.json')
    event['tensor']='raw/corrupt.npz'
    (spool/'raw/corrupt.npz').write_bytes(b'corrupt')
    write_json(spool/'index/loser.json',event)
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='bad-loser',visualization_parallel=True)
    coverage=report['sessions'][0]['coverage']
    assert coverage['canonical_coverage']['status']=='complete'
    assert coverage['all_candidate_coverage']['status']=='partial'
    assert report['summary']['rejected_points']==1
    points=read_repaired_session(tmp_path/'nas/online/run','bad-loser',sid)['points']
    assert next(p for p in points if p['sample_id']=='sample-0')['recovery']['binding']=='native_exact'


def test_unsealed_candidate_inventory_is_not_complete(tmp_path,monkeypatch):
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    (spool/'sessions'/sid/'parallel/branch-receipt.json').unlink()
    report=recover_points(cfg,'run',session_ids=[sid],visualization_parallel=True)
    scope=report['sessions'][0]['coverage']['all_candidate_coverage']
    assert scope['expected_count_complete'] is False
    assert scope['status']=='partial'


def test_parallel_manifest_ack_interruption_resumes(tmp_path,monkeypatch):
    import hidden.repair_points as repair
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    original=repair.publish_files
    def interrupted(source,target,files,timeout):
        if 'capture-repairs/interrupted/receipt.json' in files:
            raise TimeoutError('Manifest ACK interrupted')
        return original(source,target,files,timeout)
    monkeypatch.setattr(repair,'publish_files',interrupted)
    with pytest.raises(TimeoutError):
        repair.recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='interrupted',visualization_parallel=True)
    report_path=tmp_path/'nas/online/run/capture-repairs/interrupted/report.json'
    before=sha256(report_path)
    monkeypatch.setattr(repair,'publish_files',original)
    result=repair.recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='interrupted',visualization_parallel=True)
    assert result['summary']['published_sessions']==1
    assert sha256(report_path)==before


def test_duplicate_candidate_coordinates_rejected(tmp_path,monkeypatch):
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    event=read_json(spool/'index/loser.json')
    event['sample_id']='duplicate-loser'
    write_json(spool/'index/duplicate.json',event)
    result=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='duplicates',visualization_parallel=True)
    assert result['sessions'][0]['coverage']['canonical_coverage']['status']=='complete'
    assert result['sessions'][0]['coverage']['all_candidate_coverage']['status']=='partial'
    points=read_repaired_session(tmp_path/'nas/online/run','duplicates',sid)['points']
    assert not any(point['request_id']=='loser' for point in points)


def test_candidate_ledger_conflict_cannot_be_native_exact(tmp_path,monkeypatch):
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    base=spool/'sessions'/sid/'parallel'
    path=base/'decisions/d.json'
    decision=read_json(path);decision['candidates'][0]['ledger_sha256']='incorrect'
    write_json(path,decision)
    receipt=read_json(base/'branch-receipt.json')
    receipt['files'][str(path.relative_to(spool))]=sha256(path)
    write_json(base/'branch-receipt.json',receipt)
    result=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='bad-ledger',visualization_parallel=True)
    scope=result['sessions'][0]['coverage']
    assert scope['canonical_coverage']['status']=='complete'
    assert scope['all_candidate_coverage']['issues'][0]['reason']=='candidate_ledger_checksum_conflict'
    points=read_repaired_session(tmp_path/'nas/online/run','bad-ledger',sid)['points']
    loser=next(point for point in points if point['request_id']=='loser')
    assert loser['recovery']['binding']!='native_exact_candidate'


@pytest.mark.parametrize('gap', [None,'hidden','native'])
def test_visualization_publication_preserves_independent_points(tmp_path,monkeypatch,gap):
    from types import SimpleNamespace
    import hidden.publish_visualization as publisher
    from hidden.publish_visualization import publish_visualization,read_visualization_publication
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=True)
    if gap:
        name='index/1.json' if gap=='hidden' else 'requests/native-1.json'
        for base in (spool,tmp_path/'nas/online/run'):
            path=base/name
            if path.exists(): path.unlink()
    row={'run_id':'run','sample_key':'s-a','attempt_id':'attempt','task_key':'task-key',
         'trajectory_path':'runs/run/trajectory.json','trajectory_sha256':sha256(tmp_path/'runs/run/trajectory.json'),
         'model_fingerprint':'model'}
    evidence={'rollouts':{'rollout':row},'tasks':{'task-key':{'instance_id':'task'}}}
    import hidden.publish_visualization as publisher_for_actions
    monkeypatch.setattr(publisher_for_actions,'read_actions',lambda *args:[{'message_index':0},{'message_index':2}])
    monkeypatch.setattr(publisher,'RunEvidence',lambda *a:SimpleNamespace(read=lambda:evidence))
    repair='parallel'
    audit=tmp_path/'runs/run/hidden/capture-repairs'/repair
    selection={'rollouts':[{'rollout_id':'rollout','session_id':sid}],'excluded':[]}
    digest=hash_json(selection)
    path=audit/'publication-selection.json'
    write_json(path,{'selection':selection,'selection_sha256':digest})
    write_json(audit/'publication-selection-receipt.json',{'selection_sha256':digest,'files':{path.name:sha256(path)}})
    report=recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id=repair,
                          selection_sha256=digest,visualization_parallel=True)
    assert report['summary']['published_sessions']==1
    pub=publish_visualization(cfg,'run',repair_id=repair,publication_id='visual',selection=selection,
                              selection_sha256=digest)
    assert pub['eligible_for_training'] is False
    view=read_visualization_publication(cfg,'run','visual')
    points=view['canonical-point-index']
    assert [p['sample_id'] for p in points]==(['sample-0','sample-2'] if gap else ['sample-0','sample-1','sample-2'])
    assert all(p['actual_path'] for p in points)
    assert any(not p['actual_path'] for p in view['branch-point-index'])
    truths=view['canonical-truth']
    assert [t['message_index'] for t in truths]==[0,2]
    assert [t['remaining_reasoning_tokens'] for t in truths]==([None,0] if gap=='native' else [4,0])
    assert publish_visualization(cfg,'run',repair_id=repair,publication_id='visual',selection=selection,
                                 selection_sha256=digest)==pub
    with pytest.raises(ValueError,match='repair ID'):
        read_visualization_publication(cfg,'run','visual',repair_id='other')
    from enrichment.online_trace import verify_publication
    with pytest.raises(ValueError,match='not eligible'):
        verify_publication(view['directory'],tmp_path/'nas/online/run')


@pytest.mark.parametrize('missing_winner',[False,True])
def test_publication_plot_readers_keep_prediction_and_truth_separate(tmp_path,monkeypatch,missing_winner):
    from types import SimpleNamespace
    import numpy as np
    import hidden.publish_visualization as publisher
    import difficulty.evaluation.parallel as parallel
    import difficulty.evaluation.trajectory_sources as trajectory_sources
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=False)
    for base in (spool,tmp_path/'nas/online/run'):
        path=base/('requests/native-0.json' if missing_winner else 'requests/native-1.json')
        if path.exists(): path.unlink()
    row={'run_id':'run','sample_key':'s-a','attempt_id':'attempt','task_key':'task-key',
         'trajectory_path':'runs/run/trajectory.json','trajectory_sha256':sha256(tmp_path/'runs/run/trajectory.json'),
         'model_fingerprint':'model','instance_id':'task','termination_reason':'Submitted'}
    evidence={'rollouts':{'rollout':row},'tasks':{'task-key':{'instance_id':'task'}}}
    monkeypatch.setattr(publisher,'RunEvidence',lambda *a:SimpleNamespace(read=lambda:evidence))
    selection={'rollouts':[{'rollout_id':'rollout','session_id':sid}],'excluded':[]}
    audit=tmp_path/'runs/run/hidden/capture-repairs/parallel'
    digest=hash_json(selection)
    path=audit/'publication-selection.json'
    write_json(path,{'selection':selection,'selection_sha256':digest})
    write_json(audit/'publication-selection-receipt.json',{'selection_sha256':digest,'files':{path.name:sha256(path)}})
    recover_points(cfg,'run',session_ids=[sid],apply=True,repair_id='parallel',
                   selection_sha256=digest,visualization_parallel=True)
    publisher.publish_visualization(cfg,'run',repair_id='parallel',publication_id='visual',
        selection=selection,selection_sha256=digest)
    view=publisher.read_visualization_publication(cfg,'run','visual')
    actions=[{'message_index':i} for i in range(3)]
    monkeypatch.setattr(trajectory_sources,'read_actions',lambda *args:actions)
    monkeypatch.setattr(trajectory_sources,'predict_model',lambda model,x:np.full(len(x),2.0))
    model_path=tmp_path/'model';model_path.write_bytes(b'model')
    predictions,_,audit,_=trajectory_sources.predict_visualization(
        cfg,'run',view,evidence,object(),0,model_path=model_path,target='target')
    assert set(predictions)==({('rollout',1),('rollout',2)} if missing_winner else {('rollout',0),('rollout',2)})
    cache=tmp_path/'difficulty/evaluations/cache'
    write_json(cache/'coverage-audit.json',audit)
    monkeypatch.setattr(trajectory_sources,'verify_receipt',lambda *args:None)
    with pytest.raises(ValueError,match='cache publication/selection/source/model mismatch'):
        trajectory_sources.predict_visualization(cfg,'run',view,{**evidence,'rollouts':{}},object(),0,
            cache_id='cache',model_path=model_path,target='target')
    assert [v['remaining_reasoning_tokens'] for v in view['canonical-truth']]==([None,1,0] if missing_winner else [None,None,0])
    monkeypatch.setattr(parallel,'load_snapshot',lambda *args:cfg)
    monkeypatch.setattr(parallel,'read_actions',lambda *args:actions)
    monkeypatch.setattr(parallel,'select_trajectories',lambda *args:{'sources':[row]})
    data=parallel.collect_parallel_scores(cfg,'run','rollout',publication_id='visual')
    assert data['steps'][0]['selected_branch_id']==1
    assert [branch['score'] for branch in data['branches']]==[1,2]
    assert [branch['hidden_available'] for branch in data['branches']]==[False,not missing_winner]
    call=getattr(parallel.visualize_parallel,'__wrapped__',parallel.visualize_parallel)
    rendered=call(cfg,'run','parallel-plot',rollout_id='rollout',publication_id='visual')
    assert Path(rendered['plot']).read_bytes().startswith(b'\x89PNG')


@pytest.mark.parametrize('loser_hidden,status',[(False,'partial'),(True,'complete')])
def test_run_end_parallel_automatic_publication(tmp_path,monkeypatch,loser_hidden,status):
    from types import SimpleNamespace
    import hidden.repair_points as points
    import hidden.publish_visualization as publisher
    import application.execution.capture_publication as service
    cfg,spool,sid=parallel_fixture(tmp_path,monkeypatch,loser_hidden=loser_hidden)
    evidence=points.RunEvidence().read()
    evidence['rollouts']['rollout'].update(repetition_id='rep-000',termination_reason='Submitted')
    cfg['capture'].update(enabled=True,publication_seconds=60)
    cfg['capture']['export']['enabled']=True
    write_json(tmp_path/'runs/run/resolved-config.json',cfg)
    item=dict(sample_key='s-a',instance_id='task',task_key='task-key',repetition_id='rep-000')
    result={**item,'attempt_id':'attempt','rollout_id':'rollout','exit_status':'Submitted',
            'disposition':'accepted','capture_status':'partial'}
    result_path=tmp_path/'runs/run/rollout/tasks/task/repetitions/rep-000/result.json'
    write_json(result_path,result)
    monkeypatch.setattr(service,'RunEvidence',lambda *a:SimpleNamespace(read=lambda:evidence))
    monkeypatch.setattr(publisher,'RunEvidence',lambda *a:SimpleNamespace(read=lambda:evidence))
    monkeypatch.setattr(service,'assert_inactive',lambda *a:None)
    before=sha256(result_path)
    summary=service.finalize_capture_publication(cfg,'run',items=[item])
    assert summary['status']==status,summary
    assert summary['canonical_coverage']['status']=='complete'
    assert summary['all_candidate_coverage']['status']==status
    assert summary['published_points']==3
    assert summary['purpose']=='visualization' and not summary['eligible_for_training']
    assert '--publication-id' in summary['visualization_commands']['parallel']
    assert sha256(result_path)==before
    again=service.finalize_capture_publication(cfg,'run',items=[item])
    assert again['publication_receipt_sha256']==summary['publication_receipt_sha256'],again
    if os.environ.get('DYCON_CAPTURE_TEST_REPORT_ROOT'):
        output=tmp_path/'runs/run/enrichment/online-publications'/summary['publication_id']
        example=Path(os.environ['DYCON_CAPTURE_TEST_REPORT_ROOT'])/'parallel-examples'/status
        write_json(example/'capture-publication.json',summary)
        for name in ('publication','canonical-point-index','canonical-truth','branch-point-index','decisions','coverage','receipt','point-exclusions'):
            write_json(example/(name+'.json'),read_json(output/(name+'.json')))
