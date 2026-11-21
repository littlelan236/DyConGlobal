from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import copy
import pytest

from RL.online_failures import baseline_failure_reward, verifier_error
from RL.online_runtime import (OnlineRow, OnlineRolloutBatch, validate_trajectory_coverage,
                               online_row_masses, effective_source_mass)
from RL.online_bootstrap import SourceMass


def policy():
    return {'baseline_failure_reward': -3., 'verifier_error_retries': 3}


def config():
    return {'operation':'rollout', 'execution': {'online_rl_failure_policy':policy(),
        'limits_retry': {'enabled':True, 'retry_statuses':['LimitsExceeded','reasoning_loop'],
                         'seed_policy':'repetition_round_offset_v1'}}}


@pytest.mark.parametrize('status', ['LimitsExceeded','reasoning_loop'])
def test_only_online_baseline_accepts_limit_failure(status):
    from rollout.retry import sample_accepted
    cfg = config()
    result = {'exit_status':status, 'online_source':'baseline'}
    assert baseline_failure_reward(cfg, 'baseline', result) == -3
    assert sample_accepted(cfg, result)
    assert not sample_accepted(cfg, {**result, 'online_source':'dycon'})
    del cfg['execution']['online_rl_failure_policy']
    assert not sample_accepted(cfg, result)


@pytest.mark.parametrize('status', ['TimeExceeded','ContextWindowExceededError','APIError',
    'ContainerCorrupted','ParallelSelectionError','RepeatedFormatError','Timeout','Submitted'])
def test_other_terminations_never_receive_minus_three(status):
    assert baseline_failure_reward(config(), 'baseline', {'exit_status':status}) is None


def row(source='baseline', tid='b'):
    return OnlineRow('row-'+tid,'task',source,'g',(1,2,3),(3,),(-2.,),None,
                     -1.,tid,1,0,False)


def request():
    return {'step':0,'tasks':['task'],'policy_version':'S0','estimator_version':'E0',
        'request_sha256':'sha','deployment':{'failure_policy':policy()},
        'work_items':[{'trajectory_id':'b','task_id':'task','source':'baseline'},
                      {'trajectory_id':'d','task_id':'task','source':'dycon'}]}


def discard(source, tid):
    return {'trajectory_id':tid,'task_id':'task','source':source,
            'reason':'verifier_error_retries_exhausted','evaluation_id':'eval','rollout_id':'roll',
            'retry_limit':3,'attempt_count':4}


def test_discard_coverage_and_source_renormalization():
    batch = OnlineRolloutBatch(0,('task',),'S0','E0',(row(),),request_sha256='sha',
                              discarded_trajectories=(discard('dycon','d'),))
    validate_trajectory_coverage(batch,request())
    mass=effective_source_mass(batch,SourceMass(.3,.7))
    assert online_row_masses(batch.rows,mass)==pytest.approx([1.])
    restored=OnlineRolloutBatch.from_dict(batch.as_dict())
    assert restored==batch
    with pytest.raises(ValueError):
        validate_trajectory_coverage(replace(batch,discarded_trajectories=()),request())
    with pytest.raises(ValueError):
        validate_trajectory_coverage(replace(batch,discarded_trajectories=(discard('baseline','b'),)),request())
    bad=discard('dycon','d');bad['attempt_count']=3
    with pytest.raises(ValueError):
        validate_trajectory_coverage(replace(batch,discarded_trajectories=(bad,)),request())


def test_limit_reward_requires_reason_and_bound_value():
    b=replace(row(),advantage=-3.,reward_reason='LimitsExceeded')
    b=OnlineRow.from_dict(b.__dict__)
    batch=OnlineRolloutBatch(0,('task',),'S0','E0',(b,row('dycon','d')),request_sha256='sha')
    validate_trajectory_coverage(batch,request())
    with pytest.raises(ValueError):
        OnlineRow.from_dict({**b.__dict__,'reward_reason':None})
    with pytest.raises(ValueError):
        validate_trajectory_coverage(replace(batch,rows=(replace(b,advantage=-4.),row('dycon','d'))),request())


def test_empty_batch_skips_optimizer_and_scheduler(monkeypatch):
    torch=pytest.importorskip('torch')
    from RL.online_runtime import TorchOnlineUpdater
    model=torch.nn.Linear(2,2)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    monkeypatch.setattr('RL.training.optimizer_and_scheduler',lambda *args:(opt,sched))
    before=copy.deepcopy(model.state_dict()); epoch=sched.last_epoch
    batch=OnlineRolloutBatch(0,('task',),'S0','E0',(),request_sha256='sha',
        discarded_trajectories=(discard('baseline','b'),discard('dycon','d')))
    validate_trajectory_coverage(batch,request())
    batch=OnlineRolloutBatch.from_dict(batch.as_dict())
    result=TorchOnlineUpdater(model,SimpleNamespace(grad_clip=1.),source_mass=SourceMass())(batch,0)
    assert result['skipped_update'] and sched.last_epoch==epoch
    assert not opt.state
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


@pytest.mark.parametrize('retries,succeed_after',[(0,None),(3,None),(2,2),(3,0)])
def test_actual_harness_verifier_retry_and_discard(cfg,monkeypatch,retries,succeed_after):
    pytest.importorskip('docker')
    from conftest import import_fixture
    from application.execution.evidence import RunEvidence
    from benchmark_evaluation import harness
    from core.deadline import Deadline
    rid=import_fixture(cfg)
    cfg['selection']=RunEvidence(cfg['project_root'],'source').select()
    cfg['execution']['online_rl_failure_policy']={**policy(),'verifier_error_retries':retries}
    calls=[]
    def run_instance(*args):
        calls.append(1)
        if succeed_after is None or len(calls)<=succeed_after:raise RuntimeError('verifier execution failed')
        return {'completed':True,'resolved':False}
    official=SimpleNamespace(make_test_spec=lambda row,**kw:row,run_instance=run_instance,dycon_restore=lambda:None)
    monkeypatch.setattr(harness,'install_harness_adapter',lambda *args:official)
    monkeypatch.setattr(harness,'NETWORK_RETRY_DELAY_SECONDS',0)
    monkeypatch.setattr('benchmark_evaluation.swebench_audit.collect',
        lambda output,eid,rid,task,outcome,**kw:outcome)
    def classify(value):
        valid=value['status']=='completed'
        return {'classification':'ValidOutcome' if valid else 'VerifierError',
                'effective_resolved':False if valid else None,
                'official_result':{'completed':valid,'resolved':False if valid else None},
                'reason_code':'test','evidence_digest':'test'}
    monkeypatch.setattr('benchmark_evaluation.swebench_verifier.classify',classify)
    factory=SimpleNamespace(lock_images=lambda x:None,client=None,deadline=Deadline(),cleanup_owned=lambda x:None)
    result=harness.evaluate(cfg,Path(cfg['project_root'])/'runs/run','online-error-test',factory=factory)
    outcome=result['outcomes'][rid]
    expected=retries+1 if succeed_after is None else min(retries+1,succeed_after+1)
    assert len(calls)==expected and len(outcome['attempts'])==expected
    if succeed_after is None:
        assert verifier_error(outcome)
        assert outcome['online_rl_discard']['attempt_count']==retries+1
        # Registration must preserve the marker for NativeRunAssembler.
        saved=RunEvidence(cfg['project_root'],'run').read()['evaluations']
        assert any(x['outcomes'][rid].get('online_rl_discard') for x in saved.values())
    else:
        assert outcome['resolved'] is False and 'online_rl_discard' not in outcome


def test_admission_only_disables_limit_resampling_for_online_baseline(tmp_path,monkeypatch):
    pytest.importorskip('docker')
    from rollout import runner
    cfg=config();cfg['project_root']=str(tmp_path);cfg['parallel']={'enabled':True,'branch_count':2}
    monkeypatch.setattr(runner,'_boundary',lambda *args:None)
    monkeypatch.setattr(runner,'run_task',lambda received,*args,**kwargs:received)
    baseline=runner.admitted_task(cfg,'run','key',{},None,{'online_source':'baseline'})
    assert baseline['execution']['limits_retry']['enabled'] is False
    assert baseline['execution']['limits_retry']['retry_statuses']==['LimitsExceeded','reasoning_loop']
    dycon=runner.admitted_task(cfg,'run','key',{},None,{'online_source':'dycon'})
    assert dycon['execution']['limits_retry']['enabled'] is True
    cfg['execution'].pop('online_rl_failure_policy')
    ordinary=runner.admitted_task(cfg,'run','key',{},None,{'online_source':'baseline'})
    assert ordinary['execution']['limits_retry']['enabled'] is True


def test_ensure_evaluated_skips_limit_baseline_and_accepts_exhausted_verifier(tmp_path,monkeypatch):
    import application.execution.harness  # Bind real load_plan before patching its provider.
    from RL.online_evidence import ensure_evaluated
    data={'tasks':{'t':{'instance_id':'task'}},'rollouts':{
        'limit':{'task_key':'t','repetition_id':'rep-001','termination_reason':'LimitsExceeded'},
        'bad':{'task_key':'t','repetition_id':'rep-000','selected_evaluation_id':'eval'}},
        'evaluations':{'eval':{'outcomes':{'bad':{'status':'VerifierError',
           'online_rl_discard':{'reason':'verifier_error_retries_exhausted'}}}}}}
    class Evidence:
        def __init__(self,*args):pass
        def verify(self):pass
        def read(self):return data
        def select(self):return {'rollouts':data['rollouts']}
    monkeypatch.setattr('application.execution.evidence.RunEvidence',Evidence)
    monkeypatch.setattr('application.execution.run_plan.load_plan',lambda *args:{'config':config(),
        'run':tmp_path,'work_items':{'items':[{'instance_id':'task','repetition_id':'rep-001','online_source':'baseline'},
                                            {'instance_id':'task','repetition_id':'rep-000','online_source':'dycon'}]}})
    monkeypatch.setattr('application.execution.harness.run',lambda *args:pytest.fail('must reuse exhausted result'))
    assert ensure_evaluated(tmp_path,'run')['state']=='reused'


def test_new_defaults_do_not_migrate_legacy_campaign(tmp_path):
    import json
    from RL.config import online_training_defaults,OnlineBootstrap
    raw={'algorithm':{'name':'online_bootstrap_grpo_v1'},'run':{'start':{'mode':'fresh'}},
         'online_bootstrap':{'state_path':'state.json'}}
    fresh=online_training_defaults(raw,tmp_path)['online_bootstrap']
    assert fresh['baseline_failure_reward']==-3 and fresh['verifier_error_retries']==3
    (tmp_path/'state.json.campaign.json').write_text(json.dumps({'contract':{'options':{}}}))
    legacy=online_training_defaults(raw,tmp_path)['online_bootstrap']
    assert 'baseline_failure_reward' not in legacy and 'verifier_error_retries' not in legacy
    for value in [0,1,float('nan')]:
        with pytest.raises(ValueError):OnlineBootstrap(tasks=['task'],initial_estimator='manifest',baseline_failure_reward=value)


def test_native_assembly_keeps_minus_three_and_drops_verifier_failure(tmp_path,monkeypatch):
    from RL.online_evidence import NativeRunAssembler
    from core.io import write_json,sha256
    cfg=config();cfg.update(project_root=str(tmp_path),paths={'nas_root':str(tmp_path)},capture={'export':{'root':'native'}})
    run=tmp_path/'runs/run';run.mkdir(parents=True)
    trajectory=tmp_path/'trajectory.json';write_json(trajectory,{'messages':[]})
    data={'rollouts':{
        'b':{'trajectory_path':'trajectory.json','trajectory_sha256':sha256(trajectory),'termination_reason':'LimitsExceeded'},
        'd':{'selected_evaluation_id':'eval'}},
        'evaluations':{'eval':{'outcomes':{'d':{'status':'VerifierError',
            'online_rl_discard':{'reason':'verifier_error_retries_exhausted','retry_limit':3,'attempt_count':4}}}}}}
    evidence_path=tmp_path/'evidence.json';write_json(evidence_path,data)
    class Evidence:
        path=evidence_path
        def __init__(self,*args):pass
        def verify(self):pass
        def read(self):return data
    items=[{'instance_id':'task','online_source':src,'online_repetition':0,'repetition_id':tid}
           for src,tid in [('baseline','b'),('dycon','d')]]
    for item in items:write_json(run/'rollout/tasks/task/repetitions'/item['repetition_id']/'result.json',{})
    monkeypatch.setattr('application.execution.evidence.RunEvidence',Evidence)
    monkeypatch.setattr('application.execution.run_plan.load_plan',lambda *args:{'config':cfg,'run':run,
        'work_items':{'items':items},'plan':{'plan_sha256':'plan'}})
    monkeypatch.setattr('rollout.retry.verify_final',lambda root,rid,item,*args:item['repetition_id'])
    monkeypatch.setattr('hidden.artifacts.response_records',lambda *args:[(0,{}, {})])
    monkeypatch.setattr('RL.online_evidence.native_completion',lambda *args:({
        'request_id':'native','prompt_token_ids':[1,2],'output_token_ids':[3]},(-.1,)))
    result=NativeRunAssembler(cfg,feature={},advantage_settings={})(request(),{'exit_code':0},run_id='run')
    batch=OnlineRolloutBatch.from_dict(result)
    assert len(batch.rows)==1 and batch.rows[0].advantage==-3
    assert batch.rows[0].reward_reason=='LimitsExceeded'
    assert not batch.resolved_points
    assert batch.discarded_trajectories[0]['trajectory_id']=='d'
    validate_trajectory_coverage(batch,request())

@pytest.mark.parametrize('status',['LimitsExceeded','reasoning_loop'])
def test_real_baseline_attempt_is_accepted_and_reused_without_resampling(cfg,monkeypatch,status):
    pytest.importorskip('docker')
    from tests.test_limits_retry import configure,fake_runtime
    from tests.conftest import task
    from catalog.store import Catalog
    from application.execution.run_plan import create_run,load_plan
    from application.execution.evidence import RunEvidence
    from RL.online_runtime import make_online_run_work_items
    from rollout.runner import admitted_task
    from rollout.sampling import sample_from_item
    from core.io import hash_json,write_json
    cfg=configure(cfg)
    cfg['execution']['online_rl_failure_policy']=policy()
    cfg['config_hash']=hash_json({k:v for k,v in cfg.items() if k!='config_hash'})
    Catalog(cfg['project_root']).register_tasks([task('task')])
    selection=Catalog(cfg['project_root']).select_tasks(task_ids=['task'],benchmark=cfg['benchmark']['backend'],revision=None)
    manifest=make_online_run_work_items(selection,'online-attempt',m=1,base_seed=42,branch_count=1)
    run=create_run(cfg,'online-attempt',operation='rollout',task_ids=['task'],repeat_count=2,online_work_items=manifest)
    loaded=load_plan(cfg['project_root'],run.name)
    factory,calls,_,_=fake_runtime(monkeypatch,[status,'Submitted'])
    items=sorted(loaded['work_items']['items'],key=lambda x:x['online_source'])
    results=[]
    for item in items:
        sample=sample_from_item(item);definition=loaded['config']['selection']['tasks'][item['task_key']]
        runtime={'work_items':loaded['work_items'],'online_source':item['online_source'],
                 'serving_cfg':{**loaded['config'],'parallel':{'enabled':False,'branch_count':1}}}
        result=admitted_task(loaded['config'],run.name,item['task_key'],definition,factory,runtime,sample)
        if item['online_source']=='baseline':
            assert result['exit_status']==status and result['physical_attempts']==1
            assert result['resample_retry_count']==0 and result['online_source']=='baseline'
            before=list(calls)
            again=admitted_task(loaded['config'],run.name,item['task_key'],definition,factory,runtime,sample)
            assert again==result and calls==before
        results.append(result)
    write_json(run/'status.json',{'stage':'complete'})
    selected=RunEvidence(cfg['project_root'],run.name).select()
    assert len(selected['rollouts'])==2 and len(calls)==2


def test_agent_limit_fact_does_not_accept_capture_or_other_failures():
    from rollout.retry import sample_accepted
    for status in ['CaptureError','APIError','ContainerCorrupted','Cancelled']:
        assert not sample_accepted(config(), {'online_source':'baseline',
            'exit_status':status,'agent_exit_status':'LimitsExceeded','disposition':'blocked'})
