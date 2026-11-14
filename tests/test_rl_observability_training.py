"""Stage callbacks must preserve parameters, optimizer state and cursor semantics."""
import copy
import pytest
import torch
from RL.grpo.objective import ObjectiveResult
from RL.training import train_grpo
from tests.test_grpo_training import group


def exercise(callback=None, device='cpu'):
    model=torch.nn.Linear(1,1,bias=False).to(device)
    model.weight.data.fill_(.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    groups=[group(i,float(i+1)) for i in range(5)]
    if device!='cpu':
        for g in groups: g.loss_mask=g.loss_mask.to(device)
    def objective(m,g):
        loss=m.weight.sum()*g.slope
        return ObjectiveResult(loss,loss,loss*0,{})
    result=train_grpo(model,groups,objective,optimizer,scheduler,group_batch_size=2,
        grad_clip=100.,on_stage=callback)
    return model.weight.detach().cpu(),result,copy.deepcopy(optimizer.state_dict())


@pytest.mark.parametrize('failure',[False,True])
def test_stage_callback_does_not_change_training_or_optimizer(failure):
    events=[]
    def stage(name,**fields):
        events.append((name,fields))
        if failure: raise OSError('injected diagnostic failure')
    expected=exercise()
    if failure:
        with pytest.warns(UserWarning,match='callback failed'): actual=exercise(stage)
    else: actual=exercise(stage)
    torch.testing.assert_close(expected[0],actual[0],rtol=0,atol=0)
    assert expected[1]==actual[1]
    for k in expected[2]['state'][0]:
        torch.testing.assert_close(expected[2]['state'][0][k],actual[2]['state'][0][k],rtol=0,atol=0)
    windows=[fields for name,fields in events if name=='window']
    assert [w['inflight_update'] for w in windows]==[1,2,3]
    assert [w['window_end_group_ordinal'] for w in windows]==[2,4,5]
    assert [fields['group_ordinal'] for name,fields in events if name=='group_forward' and 'group_ordinal' in fields]==list(range(5))


def test_append_observation_uses_segment_schedule_and_keeps_inherited_prefix(tmp_path):
    from pathlib import Path
    from core.io import write_json
    from RL.observability import snapshot
    from RL.grpo.observability import query
    from tests.test_grpo_prepared import dataset,config,train,extension
    dataset(tmp_path,'a',range(5));cfg=config(tmp_path)
    parent,_,_,_=train(cfg,'obs-parent',stop=1)
    checkpoint=parent/'checkpoints/step-000001'
    dataset(tmp_path,'b',range(8));childcfg=config(tmp_path,'b',updates=5)
    prepared=extension(tmp_path,childcfg,checkpoint)
    child,_,result,_=train(childcfg,'obs-appended',mode='append_resume',parent=checkpoint,stop=4,
                          extension=prepared['extension_manifest'])
    # The helper bypasses only public config snapshot publication; persist the
    # same resolved fixture config for the read-only query contract.
    resolved=childcfg.model_dump(mode='json');resolved['validation']['stop_after_update']=4
    write_json(child/'resolved-config.json',resolved)
    rootcfg={'project_root':str(Path(__file__).parents[1])}
    progress=snapshot(rootcfg,'obs-appended',childcfg.runtime.output_root)
    assert progress['global_step']==4 and progress['max_updates']==5
    assert progress['execution_stop_target']==4 and progress['committed_consumed_groups']==7
    group=query(rootcfg,'obs-appended',storage_root=childcfg.runtime.output_root,group_ordinal=5)['groups'][0]
    assert group['segment_id']=='segment-000001' and group['segment_local_ordinal']==0
    assert group['consumption']=='consumed'
    pending=query(rootcfg,'obs-appended',storage_root=childcfg.runtime.output_root,group_ordinal=7)['groups'][0]
    assert pending['consumption']=='pending'
    assert (child/'metrics.jsonl').read_bytes().splitlines()[0]==(parent/'metrics.jsonl').read_bytes().splitlines()[0]
