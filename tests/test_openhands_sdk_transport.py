import json
from types import SimpleNamespace
import httpx
import pytest
import requests
from runtime.agents import openhands_transport as transport
from runtime.agents.skillsbench import _acp_details
from rollout.adapters.openhands_bridge import OpenHandsBridge, upstream_error_evidence
from core.deadline import Deadline

@pytest.mark.parametrize('kind,phase', [(httpx.ConnectTimeout,'connect'),(httpx.ReadTimeout,'read'),(httpx.WriteTimeout,'write'),(httpx.PoolTimeout,'pool')])
def test_actual_exception_chain_phase(kind, phase):
    try:
        try: raise kind('DO-NOT-EXPORT secret')
        except kind as e: raise RuntimeError('provider wrapper') from e
    except RuntimeError as e:
        evidence = transport.exception_evidence(e)
    assert evidence['phase'] == phase
    assert 'DO-NOT-EXPORT' not in json.dumps(evidence)
    assert _acp_details({'exception_type': kind.__name__})['phase'] == phase

def test_unknown_and_ambiguous_phase_remain_unknown():
    assert _acp_details({'exception_type': 'APITimeoutError'})['phase'] == 'unknown'
    assert _acp_details({'error': 'ReadTimeout ConnectTimeout'})['phase'] == 'unknown'
    assert transport.exception_evidence(ValueError('timeout'))['phase'] == 'unknown'

def test_native_retry_policy_and_effective_http_timeout(monkeypatch):
    events=[]
    monkeypatch.setenv('DYCON_OPENHANDS_MODEL_SECONDS','2')
    monkeypatch.setattr(transport,'_emit', lambda state,event,**fields: events.append(transport.safe_event({**state['identity'],'event':event,**fields})))
    llm=SimpleNamespace(timeout=2,num_retries=0,litellm_extra_body={'metadata':{'llm_type':'condenser'}})
    sends=[]
    def handle(request):
        sends.append(request)
        raise httpx.ReadTimeout('secret=do-not-export',request=request)
    def native(**kwargs):
        assert kwargs['max_retries']==0
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            return client.post(kwargs['api_base']+'/chat/completions',timeout=kwargs['timeout'])
    with pytest.raises(httpx.ReadTimeout):
        transport.completion(native,llm,api_base='http://127.0.0.1:18190/v1',api_key='supersecret',timeout=2)
    assert len(sends)==1
    assert events[0]['sdk_timeout']==2 and events[0]['sdk_retries']==0
    failed=next(e for e in events if e['event']=='sdk_http_failed')
    assert failed['phase']=='read' and failed['http_timeout']==dict.fromkeys(('read','connect','write','pool'),2)
    assert sends[0].headers['X-DyCon-SDK-Request-ID']==events[0]['sdk_request_id']
    assert events[-1]['event']=='sdk_call_failed' and events[-1]['role']=='condenser'
    assert 'supersecret' not in json.dumps(events) and 'do-not-export' not in json.dumps(events)

def test_ordinary_sdk_unchanged_when_driver_not_enabled(monkeypatch):
    monkeypatch.delenv('DYCON_OPENHANDS_MODEL_SECONDS',raising=False)
    seen=[]
    assert transport.completion(lambda **kw:seen.append(kw) or 1,SimpleNamespace(),timeout=3)==1
    assert seen==[{'timeout':3}]

def test_sdk_event_whitelist_discards_credentials():
    row=transport.safe_event({'event':'sdk_call_failed','sdk_request_id':'sdk-'+'a'*32,
       'phase':'read','Authorization':'Bearer secret','url':'http://secret','messages':['prompt'],
       'exception_types':['httpx.ReadTimeout','sk-secret'],'http_timeout':{'read':3,'secret':1}})
    assert row['http_timeout']=={'read':3}
    assert row['exception_types']==['httpx.ReadTimeout']
    assert 'secret' not in json.dumps(row) and 'prompt' not in json.dumps(row)

def test_upstream_error_response_has_bounded_safe_summary():
    response=httpx.Response(500,json={'error':{
        'type':'InternalServerError','code':'engine_error','param':None,
        'message':'backend failure token=secret https://private.invalid/detail'}},
        request=httpx.Request('POST','http://127.0.0.1/v1/chat/completions'))
    evidence=transport.response_error_evidence(response)
    row=transport.safe_event({'event':'sdk_http_finished','sdk_request_id':'sdk-'+'a'*32,
        'http_status':500,**evidence})
    assert row['http_error_type']=='InternalServerError'
    assert row['http_error_code']=='engine_error'
    assert row['http_error_body_sha256']
    assert row['http_error_message']=='backend failure token=[redacted] [url]'
    assert 'secret' not in json.dumps(row) and 'private.invalid' not in json.dumps(row)

def test_sdk_http_outcome_is_joined_to_lifecycle_receipt(cfg,tmp_path):
    bridge=OpenHandsBridge(cfg,{'run_id':'run','attempt_id':'a'},
        {'bridge_listen':'127.0.0.1','bridge_advertise':'127.0.0.1','cleanup_seconds':2},
        tmp_path,Deadline(5))
    sdk_id='sdk-'+'c'*32
    state={'lock':__import__('threading').Lock(),'events':[],'sdk_request_id':sdk_id,
        'request_id':'req-'+'d'*32}
    bridge.handlers[object()]=state
    bridge.record_sdk_event({'event':'sdk_http_finished','sdk_request_id':sdk_id,
        'http_status':500,'http_error_type':'InternalServerError','http_error_code':'engine_error'})
    bridge.record_sdk_event({'event':'sdk_call_failed','sdk_request_id':sdk_id,
        'exception_types':['httpx.HTTPStatusError'],'phase':'unknown'})
    bridge._flush_events(state)
    saved=json.loads(next((tmp_path/'diagnostics').glob('request-lifecycle-*.json')).read_text())
    assert saved['sdk_http_status']==500
    assert saved['http_error_code']=='engine_error'
    assert saved['sdk_exception_types']==['httpx.HTTPStatusError']
    assert [event['sdk_event'] for event in saved['events']]==['sdk_http_finished','sdk_call_failed']

def test_upstream_model_error_evidence_is_safe_and_typed():
    evidence=upstream_error_evidence({'error':{
        'type':'InternalServerError','code':'engine_error',
        'message':'failed token=secret https://private.invalid/trace'}})
    assert evidence['upstream_error_type']=='InternalServerError'
    assert evidence['upstream_error_code']=='engine_error'
    assert evidence['upstream_error_message']=='failed token=[redacted] [url]'
    assert evidence['upstream_error_payload_sha256']
    assert 'secret' not in json.dumps(evidence) and 'private.invalid' not in json.dumps(evidence)

def test_authenticated_sdk_events_are_attempt_bound_and_do_not_cancel(cfg,tmp_path):
    profile={'bridge_listen':'127.0.0.1','bridge_advertise':'127.0.0.1','cleanup_seconds':2}
    bridge=OpenHandsBridge(cfg,{'run_id':'run','attempt_id':'a'},profile,tmp_path,Deadline(5))
    event={'event':'sdk_call_started','sdk_request_id':'sdk-'+'a'*32,'sdk_timeout':28800,'sdk_retries':0,'native_max_retries':0}
    with bridge:
        endpoint=bridge.base_url+'/dycon/sdk-events'
        assert requests.post(endpoint,json=event,timeout=2).status_code==401
        auth={'Authorization':'Bearer '+bridge.secret}
        assert requests.post(endpoint,json={**event,'sdk_request_id':'../../escape'},headers=auth,timeout=2).status_code==400
        assert requests.post(endpoint,json=event,headers=auth,timeout=2).status_code==204
        assert not bridge.closed.is_set() and bridge.records==[]
    paths=list((tmp_path/'diagnostics').glob('sdk-transport-*'))
    assert len(paths)==1
    row=json.loads(paths[0].read_text())
    assert row['binding']=={'run_id':'run','attempt_id':'a'}
    assert row['events'][0]['sdk_timeout']==28800


def test_diagnostic_network_failure_keeps_local_typed_evidence(monkeypatch,tmp_path,capsys):
    log=tmp_path/'local.jsonl'
    monkeypatch.setenv('DYCON_OPENHANDS_TRANSPORT_LOG',str(log))
    class Unavailable:
        def open(self,*a,**k): raise OSError('DO-NOT-EXPORT credential')
    monkeypatch.setattr(transport,'build_opener',lambda *a:Unavailable())
    state={'identity':{'sdk_request_id':'sdk-'+'a'*32,'sdk_timeout':28800,'sdk_retries':0},
           'endpoint':'http://private-token.invalid','secret':'supersecret'}
    transport._emit(state,'sdk_call_failed',phase='connect',exception_types=['httpx.ConnectTimeout'])
    event=json.loads(log.read_text())
    assert event['phase']=='connect' and event['sdk_timeout']==28800
    assert 'supersecret' not in log.read_text() and 'DO-NOT-EXPORT' not in capsys.readouterr().err

def test_local_sdk_evidence_collected_before_cleanup(cfg,tmp_path):
    import asyncio
    from runtime.agents.skillsbench import OwnedPlanes
    row={'event':'sdk_call_failed','sdk_request_id':'sdk-'+'b'*32,'phase':'read',
         'exception_types':['httpx.ReadTimeout'],'sdk_timeout':28800,'secret':'discard'}
    class Env:
        async def exec(self,command,**kwargs):
            assert command.startswith('test -f /tmp/dycon-sdk-transport.jsonl')
            return SimpleNamespace(return_code=0,stdout=json.dumps(row)+'\ninvalid line\n')
    owner=OwnedPlanes.__new__(OwnedPlanes)
    owner.cfg={'execution_context':{'profile':{'sandbox_user':'agent','cleanup_seconds':2}}}
    owner.output=tmp_path;owner.ownership={'run':'run','attempt':'a'}
    asyncio.run(owner._collect_sdk_transport(Env()))
    saved=json.loads((tmp_path/'diagnostics/sdk-local-transport.json').read_text())
    assert saved['available'] and saved['events'][0]['phase']=='read' and saved['invalid_lines']==1
    assert 'secret' not in json.dumps(saved)


@pytest.mark.parametrize('payload', [
    {'error': {'type':'limit','retryable':False,'correlation_id':'req-'+'a'*32}},
    {'error': {'type':'upstream_error','message':'format_error','retryable':False}},
    {'error': {'type':'FormatError','code':'format_error','retryable':True,'correlation_id':'wrong-owner'}},
])
def test_only_owner_bound_typed_format_errors_are_recoverable(monkeypatch,payload):
    monkeypatch.setenv('DYCON_OPENHANDS_MODEL_SECONDS','2')
    monkeypatch.setattr(transport,'_emit',lambda *args,**kw:None)
    llm=SimpleNamespace(timeout=2,num_retries=0,litellm_extra_body={})
    def handle(request): return httpx.Response(400,json=payload,request=request)
    def native(**kwargs):
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            response=client.post(kwargs['api_base']+'/chat/completions',timeout=2)
            response.raise_for_status()
    with pytest.raises(httpx.HTTPStatusError):
        transport.completion(native,llm,api_base='http://127.0.0.1:19000/v1',api_key='injection-only')
