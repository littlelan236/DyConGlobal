"""Behavioral contracts for CPU queries, immutable sources and managed projection."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from core.io import hash_json, read_json, write_json
from RL.artifacts import commit_receipt, verify_receipt
from RL.observability import MetricReader, matches, process, snapshot, source
from RL.grpo.observability import index_worker, query

ROOT = Path(__file__).parents[1]


def attempt(tmp_path, name='fixture', *, status='training', initial=0):
    storage = tmp_path/'storage'
    folder = storage/'RL/checkpoints'/name
    folder.mkdir(parents=True)
    cfg={'project_root':str(ROOT)}
    proof={'eligible_group_ids':['g'], 'group_count':1, 'planned_updates':3,
           'advantage_policy':'consensus_top_half_zero_sum_v1', 'advantage_config_sha256':'policy',
           'dataset_receipt_sha256':'dataset', 'exclusions':[]}
    contract={'input':proof}
    write_json(folder/'attempt.json',{'attempt_id':name,'status':status,'contract':contract,
        'initial_state':{'global_step':initial,'group_cursor':0}})
    write_json(folder/'grpo-contract.json',contract)
    write_json(folder/'resolved-config.json',{'training':{'max_updates':3},'validation':{'stop_after_update':2}})
    (folder/'metrics.jsonl').touch()
    return cfg,storage,folder


def metric(folder, step=1, **extra):
    row={'global_step':step,'group_cursor':step,'consumed_branches':step*3,
         'consumed_tokens':step*30,'loss':1/step,'policy_loss':.2,'k3_penalty_mean':.3,'seconds':2.,**extra}
    with (folder/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    return row


def test_policy_ratio_is_split_by_advantage_sign_and_published_to_tensorboard():
    from RL.observability_tensorboard import scalar_values

    row = {'ratio_mean': 1.25, 'positive_ratio_mean': .5,
           'negative_ratio_mean': 2.0, 'k3_penalty_mean': 0.,
           'clip_fraction': 0., 'policy_loss': 0., 'branch_count': 2,
           'token_count': 4, 'positive_branches': 1, 'negative_branches': 1}
    assert row['positive_ratio_mean'] == pytest.approx(.5)
    assert row['negative_ratio_mean'] == pytest.approx(2.0)
    values = scalar_values(row, kl_beta=.02)
    assert values['train/policy_ratio_positive_advantage'] == pytest.approx(.5)
    assert values['train/policy_ratio_negative_advantage'] == pytest.approx(2.0)
    assert values['train/k3_loss'] == pytest.approx(.0)
    assert 'train/k3_loss_fraction' not in values
    weighted = scalar_values({**row, 'loss': .5, 'k3_penalty_mean': .25}, kl_beta=.02)
    assert weighted['train/k3_loss'] == pytest.approx(.005)
    assert weighted['train/k3_loss_fraction'] == pytest.approx(.01)
    assert 'train/k3_loss_fraction' not in scalar_values(
        {**row, 'loss': 0., 'k3_penalty_mean': 1.}, kl_beta=.02)
    zero_beta = scalar_values({**row, 'loss': -.5, 'k3_penalty_mean': 1.0}, include_k3=False, kl_beta=0.)
    assert zero_beta['train/k3_loss'] == zero_beta['train/k3_loss_fraction'] == 0.
    assert 'train/k3_penalty_mean' not in zero_beta
    assert zero_beta['drift/k3/objective_mean_after_guard'] == 1.
    negative_loss = scalar_values({**row, 'loss': -.5, 'k3_penalty_mean': .25}, kl_beta=.02)
    assert negative_loss['train/k3_loss_fraction'] == pytest.approx(-.01)


def test_large_metrics_partial_append_dedup_and_unknown_arrays(tmp_path):
    cfg,root,folder=attempt(tmp_path)
    huge={'global_step':1,'loss':2.,'attention_masks':['x'*5000000]}
    raw=json.dumps(huge).encode()+b'\n'
    path=folder/'metrics.jsonl'
    path.write_bytes(raw+b'{"global_step":2')
    reader=MetricReader(path)
    rows=list(reader.read())
    assert rows[0][0]=={'global_step':1,'loss':2.}
    assert reader.offset==len(raw)
    with path.open('ab') as f: f.write(b',"loss":1}\n')
    assert list(reader.read())[0][0]['global_step']==2
    assert list(reader.read())==[]
    with path.open('ab') as f: f.write(b'{"global_step":2,"loss":1}\n')
    assert list(reader.read())==[]
    metric(folder,2,loss=9)
    with pytest.raises(ValueError,match='Conflicting'): list(reader.read())


@pytest.mark.parametrize('fault',['bad','nan','truncated','replaced','oversized'])
def test_metric_failure_keeps_cursor_and_is_visible(tmp_path,monkeypatch,fault):
    _,_,folder=attempt(tmp_path)
    metric(folder)
    reader=MetricReader(folder/'metrics.jsonl'); list(reader.read()); offset=reader.offset
    path=folder/'metrics.jsonl'
    if fault=='truncated': path.write_bytes(b'')
    elif fault=='replaced':
        other=path.with_suffix('.other');other.write_bytes(path.read_bytes());os.replace(other,path)
    else:
        if fault=='oversized':
            monkeypatch.setattr('RL.observability.MAX_METRIC_BYTES',400)
            metric(folder,2,noise='x'*500)
        else:
            with path.open('ab') as f: f.write(b'{bad}\n' if fault=='bad' else b'{"global_step":2,"loss":NaN}\n')
    with pytest.raises(ValueError): list(reader.read())
    assert reader.offset==offset


def test_tail_expands_beyond_four_megabytes_and_watch_preserves_state(tmp_path):
    cfg,root,folder=attempt(tmp_path)
    for step in range(1,14): metric(folder,step,noise='x'*600000)
    reader=MetricReader(folder/'metrics.jsonl');reader.seek_tail(10)
    a=snapshot(cfg,'fixture',root,reader=reader)
    b=snapshot(cfg,'fixture',root,reader=reader)
    assert a['global_step']==b['global_step']==13
    assert len(a['recent_updates'])==10
    assert a['stage']=='unknown' and a['observed_liveness']=='not_running_with_stale_state'


@pytest.mark.parametrize('state',['training','paused','completed','failed','preparing_reference'])
def test_status_keeps_raw_state_and_checkpoint_is_independent(tmp_path,state):
    cfg,root,folder=attempt(tmp_path,status=state)
    metric(folder,2)
    cp=folder/'checkpoints/step-000001'
    cp.mkdir(parents=True)
    write_json(cp/'state.json',{'state':{'global_step':1,'group_cursor':1},'contract':read_json(folder/'grpo-contract.json')})
    commit_receipt(cp,'lora-checkpoint-v1')
    newer=folder/'checkpoints/step-000002';newer.mkdir()
    write_json(newer/'state.json',{'state':{'global_step':2}})
    result=snapshot(cfg,'fixture',root)
    assert result['status']==state
    assert result['last_completed_update']==2 and result['recoverable_checkpoint_step']==1
    assert result['execution_stop_target']==2


def test_source_root_identity_and_path_ownership(tmp_path):
    cfg,root,folder=attempt(tmp_path/'a')
    _,root2,_=attempt(tmp_path/'b')
    first=source(cfg,'fixture',root)
    second=source(cfg,'fixture',root2)
    assert first[-1]!=second[-1]
    assert first[2]['storage_root']==str(root)
    with pytest.raises(ValueError): source(cfg,'../fixture',root)


def test_online_resume_reports_campaign_checkpoint_and_separate_barrier(tmp_path):
    cfg, root, folder = attempt(tmp_path, name='resumed')
    metric(folder, 2)
    checkpoint_root = folder.parent / 'original/checkpoints'
    cp = checkpoint_root / 'step-000002'
    cp.mkdir(parents=True)
    contract = read_json(folder / 'grpo-contract.json')
    write_json(cp / 'state.json', {'state': {'global_step': 2, 'group_cursor': 2}, 'contract': contract})
    commit_receipt(cp, 'lora-checkpoint-v1')
    state = tmp_path / 'pipeline.json'
    write_json(state, {'campaign_id': 'campaign', 'committed_step': 1,
        'policy_version': 'S1', 'estimator_version': 'E1', 'prefetched_step': 2})
    record = read_json(folder / 'attempt.json')
    record['online_artifacts'] = {'campaign_id': 'campaign', 'checkpoint_root': str(checkpoint_root),
                                  'state_path': str(state)}
    write_json(folder / 'attempt.json', record)
    result = snapshot(cfg, 'resumed', root)
    assert result['last_completed_update'] == result['recoverable_checkpoint_step'] == 2
    assert result['checkpoint']['path'] == str(cp)
    assert result['online_pipeline']['committed_step'] == 1
    write_json(state, {'campaign_id': 'another'})
    assert any('campaign differs' in warning for warning in snapshot(cfg, 'resumed', root)['warnings'])


def test_active_discovers_registered_training_outside_configured_scan_roots(tmp_path):
    from RL.observability import active
    cfg,root,_=attempt(tmp_path/'one',name='configured')
    _,other,_=attempt(tmp_path/'two',name='registered-other')
    cfg={'project_root':str(tmp_path),'paths':{'artifact_scan_roots':[str(root)]}}
    _,_,identity,cache=source(cfg,'registered-other',other)
    write_json(cache/'source.json',identity)
    write_json(cache/'live.json',{'process':process(os.getpid()),'stage':'group_forward'})
    result=active(cfg)
    match=next(a for a in result['attempts'] if a['attempt_id']=='registered-other')
    assert match['source']['storage_root']==str(other)
    assert match['observed_liveness']=='running'


def legacy(tmp_path, *, bad=None):
    cfg,root,folder=attempt(tmp_path)
    rows=[{'branch_sample_id':f'b{i}','historical_score':float(3-i), 'retained':i<2,
           'prompt_token_ids':[1,2,3], 'completion_token_ids':[4,5+i]} for i in range(3)]
    value={'group':{'group_id':'g','expected_branch_count':3,'branch_ids':['b0','b1','b2']},
           'members':rows,'frozen_advantages':[.5,.5,-1.], 'advantage_diagnostics':{}}
    order={'group_id':'g','branches':copy.deepcopy(rows),'advantages':[.5,.5,-1.]}
    contract=read_json(folder/'grpo-contract.json')
    contract['input']['group_order_sha256']=hash_json([order])
    write_json(folder/'attempt.json',{'attempt_id':'fixture','status':'training','contract':contract})
    write_json(folder/'grpo-contract.json',contract)
    if bad=='score': rows[0]['historical_score']=5.
    if bad=='retained': rows[0]['retained']=False
    if bad=='order': rows.reverse()
    if bad=='advantage': value['frozen_advantages'][0]=7.
    if bad=='token': rows[0]['completion_token_ids'][0]=9
    write_json(folder/'frozen-groups.json',[value])
    _,_,identity,cache=source(cfg,'fixture',root)
    write_json(cache/'source.json',identity)
    return cfg,root,folder,cache


def test_legacy_index_hashes_contract_streams_tokens_and_queries_native_order(tmp_path):
    cfg,root,folder,cache=legacy(tmp_path)
    index_worker(cache/'source.json')
    result=query(cfg,'fixture',storage_root=root,group_ordinal=0)
    branches=result['groups'][0]['branches']
    assert [b['advantage'] for b in branches]==[.5,.5,-1.]
    assert [b['native_ordinal'] for b in branches]==[0,1,2]
    assert result['source_integrity']=='locally_fingerprinted_unsealed'
    assert query(cfg,'fixture',storage_root=root,branch_id='b1')['groups'][0]['branches'][0]['branch_id']=='b1'
    assert query(cfg,'fixture',storage_root=root,group_id='g',limit=1)['groups'][0]['truncated']
    import sqlite3
    with sqlite3.connect(cache/'legacy-advantages/index.sqlite3') as con:
        payload=con.execute('SELECT payload FROM groups').fetchone()[0]
    assert 'token_ids' not in payload


def test_rank_uses_frozen_lower_is_better_diagnostics_and_native_ties():
    from RL.grpo.observability import group_view
    value={'group':{'group_id':'tied'},'members':[
        {'branch_sample_id':str(i),'historical_score':score} for i,score in enumerate([1.,1.,3.])],
        'frozen_advantages':[1.,-.5,-.5],
        'advantage_diagnostics':{'ranks':[0,0,2],'order':[0,1,2]}}
    result=group_view(value,0,{}, {'committed_consumed_groups':0})
    assert [b['rank'] for b in result['branches']]==[0,0,2]
    assert [b['tie_break_order'] for b in result['branches']]==[0,1,2]
    assert all(b['rank_provenance']=='frozen_diagnostics' for b in result['branches'])


@pytest.mark.parametrize('fault',['score','retained','order','advantage','token'])
def test_legacy_mismatch_never_publishes_ready_index(tmp_path,fault):
    _,_,_,cache=legacy(tmp_path,bad=fault)
    with pytest.raises(ValueError): index_worker(cache/'source.json')
    assert read_json(cache/'legacy-advantages/status.json')['state']=='failed'
    assert not (cache/'legacy-advantages/index.sqlite3').exists()


def test_prepared_metadata_never_opens_tokens_and_matches_training(tmp_path,monkeypatch):
    from RL.config import load
    from RL.grpo.data import prepare
    from RL.grpo.prepared import materialize
    cfg,_=load(ROOT/'RL/experiments/grpo-offline-k3-smoke.yaml',ROOT)
    cfg.runtime.output_root=str(tmp_path/'output')
    raw,proof=prepare(cfg)
    reader=materialize(cfg,ROOT,'fixture')
    expected={g['group']['group_id']:g for g in raw}
    reader.verify_metadata_indexes()
    opened=[]
    original=Path.open
    def guard(path,*args,**kwargs):
        opened.append(path.name)
        assert path.name not in ('prompt-tokens.bin','completion-tokens.bin')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    for gid in reader.ids:
        value=reader.metadata(gid)
        assert value['frozen_advantages']==expected[gid]['frozen_advantages']
        assert [r.get('retained') for r in value['members']]==[r.get('retained') for r in expected[gid]['members']]
    assert not any('tokens.bin' in name for name in opened)


def test_cli_formats_filters_and_no_torch_import(tmp_path):
    cfg,root,folder,cache=legacy(tmp_path)
    index_worker(cache/'source.json')
    for fmt in ('json','csv','table'):
        command=[sys.executable,'-m','core.cli','lora-advantages','fixture','--storage-root',str(root),
                 '--group-ordinal','0','--format',fmt]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert 'b0' in result.stdout
    command=[sys.executable,'-c',
        'import sys; from RL.observability import snapshot; from RL.grpo.observability import query; assert "torch" not in sys.modules; print("no torch")']
    assert subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10).returncode==0


def test_tensorboard_managed_replay_stop_rebuild_and_sealed_bytes(tmp_path):
    pytest.importorskip('tensorboard')
    from urllib.request import ProxyHandler, build_opener
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from RL.observability_tensorboard import manage
    cfg,root,folder=attempt(tmp_path,status='completed')
    attempt_file = folder / 'attempt.json'
    metadata = read_json(attempt_file)
    metadata['contract']['schedule'] = {'grpo': {'kl_beta': .02}}
    write_json(attempt_file, metadata)
    write_json(folder / 'grpo-contract.json', metadata['contract'])
    for step in (1, 2):
        metric(folder, step, gradient_norm=.4, positive_ratio_mean=1.1, negative_ratio_mean=.9)
    commit_receipt(folder,'lora-attempt-v1')
    receipt=verify_receipt(folder)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    try:
        started=manage(cfg,'fixture',action='start',storage_root=root,port=port)
        assert started['running']
        same=manage(cfg,'fixture',action='start',storage_root=root,port=port)
        assert same['sync']==started['sync']
        accumulator=EventAccumulator(str(Path(started['logdir'])/'fixture'));accumulator.Reload()
        assert [p.step for p in accumulator.Scalars('train/loss')]==[1,2]
        assert [p.value for p in accumulator.Scalars('train/k3_loss')] == pytest.approx([.006, .006])
        assert [p.value for p in accumulator.Scalars('drift/k3/objective_mean_after_guard')] == pytest.approx([.3, .3])
        assert [p.value for p in accumulator.Scalars('train/k3_loss_fraction')] == pytest.approx([.006, .012])
        opener = build_opener(ProxyHandler({}))

        def read_api(path):
            with opener.open(started['url'] + path, timeout=5) as response:
                return json.load(response)

        plugins = read_api('/data/plugins_listing')
        assert plugins['custom_scalars']['tab_name'] == 'Pinned'
        assert plugins['custom_scalars']['enabled'] and plugins['scalars']['enabled']
        categories = read_api('/data/plugin/custom_scalars/layout')['category']
        assert [category['title'] for category in categories] == ['Pinned']
        assert [chart['multiline']['tag'] for chart in categories[0]['chart']] == [
            ['^train/gradient_norm$'], ['^train/loss$'], ['^drift/k3/objective_mean_after_guard$'],
            ['^train/k3_loss_fraction$'], ['^train/policy_ratio_positive_advantage$'],
            ['^train/policy_ratio_negative_advantage$'], ['^timing/step_seconds$'],
        ]
        assert verify_receipt(folder)==receipt
        before=time.monotonic();metric(folder,3)
        for _ in range(50):
            accumulator.Reload()
            if len(accumulator.Scalars('train/loss'))==3: break
            time.sleep(.2)
        assert time.monotonic()-before<10
        assert [p.step for p in accumulator.Scalars('train/loss')]==[1,2,3]
        # Recover a legacy receipt captured before the worker's exec completed.
        _, _, _, cache = source(cfg, 'fixture', root)
        service_path = cache / 'tensorboard/service.json'
        legacy_record = read_json(service_path)
        legacy_record['sync']['command'] = []
        write_json(service_path, legacy_record)
        stopped=manage(cfg,'fixture',action='stop',storage_root=root,port=port)
        assert not stopped['sync_alive'] and not stopped['tensorboard_alive']
        again=manage(cfg,'fixture',action='start',storage_root=root,port=port)
        assert again['sync_status']['generation']!=started['sync_status']['generation']
        accumulator=EventAccumulator(str(Path(again['logdir'])/'fixture'));accumulator.Reload()
        assert [p.step for p in accumulator.Scalars('train/loss')]==[1,2,3]
    finally:
        manage(cfg,'fixture',action='stop',storage_root=root,port=port)


def test_nonrunning_tensorboard_expires_query_lease(tmp_path):
    pytest.importorskip('tensorboard')
    from RL.observability_tensorboard import manage
    cfg,root,folder=attempt(tmp_path,status='completed')
    metric(folder,1)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    started=manage(cfg,'fixture',action='start',storage_root=root,port=port,lease_seconds=5)
    assert started['lifecycle']=='query_lease'
    assert started['lease_seconds']==5
    time.sleep(6)
    for _ in range(20):
        state=manage(cfg,'fixture',storage_root=root)
        if not state['running']:
            break
        time.sleep(.2)
    assert not state['running']
    assert state['auto_stop_reason']=='query_lease_expired'


def test_pid_reuse_and_port_conflict_never_signal_unowned_process(tmp_path):
    from RL.observability_tensorboard import manage
    cfg,root,folder=attempt(tmp_path)
    assert not matches({**process(os.getpid()),'starttime':'wrong'})
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.listen()
        with pytest.raises(OSError): manage(cfg,'fixture',action='start',storage_root=root,port=port)


@pytest.mark.parametrize('failed_process',['sync','tensorboard'])
def test_independent_process_crash_replays_new_generation_without_duplicate_steps(tmp_path,failed_process):
    pytest.importorskip('tensorboard')
    import signal
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from RL.observability_tensorboard import manage
    cfg,root,folder=attempt(tmp_path)
    metric(folder,1);metric(folder,2)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    try:
        service=manage(cfg,'fixture',action='start',storage_root=root,port=port)
        os.kill(service[failed_process]['pid'],signal.SIGKILL)
        for _ in range(30):
            if not matches(service[failed_process]): break
            time.sleep(.05)
        if failed_process=='sync':
            # Model an event already published while the durable cursor lagged.
            _,_,_,cache=source(cfg,'fixture',root)
            cp=cache/'tensorboard/sync-cursor.json'
            cursor=read_json(cp);cursor['offset']=0;write_json(cp,cursor)
        observed=manage(cfg,'fixture',storage_root=root)
        assert not observed[failed_process+'_alive']
        assert observed[('tensorboard' if failed_process=='sync' else 'sync')+'_alive']
        recovered=manage(cfg,'fixture',action='start',storage_root=root,port=port)
        assert recovered['sync_status']['generation']!=service['sync_status']['generation']
        accumulator=EventAccumulator(str(Path(recovered['logdir'])/'fixture'));accumulator.Reload()
        assert [p.step for p in accumulator.Scalars('train/loss')]==[1,2]
    finally:
        manage(cfg,'fixture',action='stop',storage_root=root,port=port)


def test_truncated_tensorboard_source_requires_explicit_rebuild(tmp_path):
    pytest.importorskip('tensorboard')
    from RL.observability_tensorboard import manage
    cfg,root,folder=attempt(tmp_path)
    metric(folder,1);metric(folder,2)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    try:
        manage(cfg,'fixture',action='start',storage_root=root,port=port)
        (folder/'metrics.jsonl').write_bytes(b'')
        for _ in range(40):
            state=manage(cfg,'fixture',storage_root=root)
            if not state['sync_alive']: break
            time.sleep(.1)
        assert state['sync_status']['state']=='failed'
        assert snapshot(cfg,'fixture',root)['status']=='training'
        with pytest.raises(RuntimeError,match='Synchronizer failed'):
            manage(cfg,'fixture',action='start',storage_root=root,port=port)
        rebuilt=manage(cfg,'fixture',action='rebuild',storage_root=root,port=port)
        assert rebuilt['running'] and rebuilt['sync_status']['last_step']==0
    finally:
        manage(cfg,'fixture',action='stop',storage_root=root,port=port)


def test_forged_service_metadata_cannot_signal_unrelated_process(tmp_path):
    from RL.observability_tensorboard import manage
    cfg,root,_=attempt(tmp_path)
    _,_,_,cache=source(cfg,'fixture',root)
    write_json(cache/'tensorboard/service.json',{'state':'running','sync':process(os.getpid())})
    with pytest.raises(ValueError,match='refusing to signal'):
        manage(cfg,'fixture',action='stop',storage_root=root)
    assert process(os.getpid()) is not None
