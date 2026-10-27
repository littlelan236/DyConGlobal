"""Audit regression tests: denominator, diagnostic specificity, and canonical parsing."""
import hashlib
import json

import pytest

from action.analyzers.submissions import failure_kind, inventory, summarize
from action.datasets.trajectories import read_actions, read_messages


@pytest.mark.parametrize('outcome,empty,expected', [
    ({'status':'infra_error','failure_type':'other_error','reason':'MissingPatch'},True,'empty_patch'),
    ({'status':'completed','resolved':False,'failure_type':'invalid_patch','reason':'MissingPatch'},True,'empty_patch'),
    ({'status':'infra_error','failure_type':'invalid_patch'},False,'invalid_nonempty_patch'),
    ({'status':'invalidpatch'},False,'invalid_nonempty_patch'),
    ({'status':'completed','failure_type':'invalid_patch'},False,'invalid_nonempty_patch'),
    ({'status':'infra_error','failure_detail':'Failed to apply patch to container: git apply'},False,'invalid_nonempty_patch'),
    ({'status':'infra_error','failure_type':'network_error','error':'Connection reset'},True,None),
    ({'status':'infra_error','failure_type':'dependency_error'},False,None),
    ({'status':'ConnectionError','failure_type':'network_error'},False,None),
    ({'status':'completed','resolved':False},False,None),
    ({'status':'completed','resolved':True},True,None),
    ({'status':'infra_error'},True,'empty_patch'),
    ({'status':'infra_error'},False,None),
])
def test_specific_patch_cause(outcome,empty,expected):
    assert failure_kind(outcome,empty)==expected


def test_missing_evaluation_stays_in_submitted_denominator():
    unknown={'latest_evaluation':None,'patch_empty':True,'latest_patch_failure':None,
             'ever_patch_failure':False,'selected_patch_failure':None}
    bad={**unknown,'latest_evaluation':{'outcome':{'status':'completed'}},
         'latest_patch_failure':'empty_patch','ever_patch_failure':True}
    result=summarize([unknown,bad])
    assert result['submitted']==2 and result['evaluated']==1 and result['unevaluated']==1
    assert result['latest_failure_over_submitted']==0.5
    assert result['latest_failure_over_evaluated']==1
    assert summarize([])['latest_failure_over_evaluated'] is None


def test_shared_reader_keeps_action_units_and_checks_hash(tmp_path):
    messages=[{'role':'system','content':'s'},
              {'role':'assistant','content':'review','reasoning_content':'inspect fix',
               'tool_calls':[{'function':{'name':'bash','arguments':'{}'}}]},
              {'role':'tool','content':'observation'},
              {'role':'assistant','content':'submit'}]
    raw=json.dumps({'messages':messages}).encode()
    (tmp_path/'raw.json').write_bytes(raw)
    source={'rollout_id':'r-test','trajectory_path':'raw.json',
            'trajectory_sha256':hashlib.sha256(raw).hexdigest()}
    assert read_messages(tmp_path,source)==messages
    assert read_actions(tmp_path,source)==[
        {'message_index':1,'tool_calls':messages[1]['tool_calls']},
        {'message_index':3,'tool_calls':[]}]
    (tmp_path/'raw.json').write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='checksum mismatch'):
        read_messages(tmp_path,source)


def test_scope_is_same_80_tasks_strictly_above_30(tmp_path,monkeypatch):
    tasks=[{'instance_id':f'task-{i}'} for i in range(80)]
    for name,n,operation in [('reference',31,'rollout'),('exactly30',30,'rollout'),('reevaluation',60,'harness')]:
        p=tmp_path/'runs'/name;p.mkdir(parents=True)
        (p/'plan.json').write_text(json.dumps({'operation':operation}))
        (p/'tasks.json').write_text(json.dumps({'tasks':tasks}))
    def fake_read(self):
        n=30 if self.run_id=='exactly30' else 31
        return {'tasks':{'t':{'partition':'train'}},'rollouts':{
            f'r-{i}':{'task_key':'t','submission_status':'submitted'} for i in range(n)}}
    monkeypatch.setattr('action.analyzers.submissions.RunEvidence.read',fake_read)
    candidates=inventory(tmp_path,80,30,'reference')
    assert {r['source_run_id']:r['include'] for r in candidates}=={'exactly30':False,'reference':True}
