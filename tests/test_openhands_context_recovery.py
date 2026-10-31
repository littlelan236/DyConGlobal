import requests
from core.deadline import Deadline
from rollout.adapters.openhands import request_role, CONDENSATION_PREFIX, CONDENSATION_SUFFIX
from rollout.adapters.openhands_bridge import OpenHandsBridge, UpstreamModelError


def test_bridge_preserves_context_error_for_sdk_condenser(cfg,tmp_path,monkeypatch):
    profile={'bridge_listen':'127.0.0.1','bridge_advertise':'127.0.0.1','request_roles':'unqualified'}
    bridge=OpenHandsBridge(cfg,{'attempt_id':'a'},profile,tmp_path,Deadline(5))
    payload={'error':{'message':"This model's maximum context length is 32768 tokens.",
                      'type':'BadRequestError','code':400}}
    def fail(request):raise UpstreamModelError(400,payload)
    monkeypatch.setattr(bridge,'complete',fail)
    with bridge:
        r=requests.post(bridge.base_url+'/chat/completions',json={'messages':[]},
                        headers={'Authorization':'Bearer '+bridge.secret},timeout=3)
        assert r.status_code==400 and r.json()==payload


def test_pinned_sdk_condenser_is_not_a_primary_agent_step():
    body={'messages':[{'role':'user','content':CONDENSATION_PREFIX+
         '\n<EVENT>prior tool results</EVENT>\n'+CONDENSATION_SUFFIX}]}
    qualified={'request_roles':'openhands-1.22.1'}
    assert request_role(body,qualified)=='condensation'
    body['messages'][0]['content']='Please summarize this task'
    assert request_role(body,qualified)=='unqualified'


def test_preserve_thinking_restores_only_bound_winner_history():
    from rollout.adapters.openhands import preserve_reasoning_history
    winner={'role':'assistant','reasoning_content':'native winner thought',
            'tool_calls':[{'id':'selected','function':{'name':'terminal','arguments':'{}'}}]}
    records=[{'role':'agent_decision','selected_response':{'choices':[{'message':winner}]}}]
    request={'messages':[{'role':'assistant','content':'action','tool_calls':winner['tool_calls']},
                         {'role':'assistant','tool_calls':[{'id':'loser'}]}]}
    assert preserve_reasoning_history(request,records)==1
    assert request['messages'][0]['reasoning_content']=='native winner thought'
    assert 'reasoning_content' not in request['messages'][1]
