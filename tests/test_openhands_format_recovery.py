"""Regression: malformed tools recover; only unanimous native loops yield loop outcomes."""
from types import SimpleNamespace
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from core.deadline import Deadline
from rollout.adapters.openhands import OpenHandsPolicy, OpenHandsProtocolError, OpenHandsLimit
from rollout.adapters.openhands_bridge import OpenHandsBridge


@pytest.mark.parametrize('malformed_count', [1, 32])
def test_format_candidates_remain_recoverable_even_with_loop_siblings(malformed_count):
    candidates = [SimpleNamespace(branch_id=i, error=OpenHandsProtocolError('undeclared tool'))
                  for i in range(malformed_count)]
    candidates += [SimpleNamespace(branch_id=i, error=OpenHandsLimit({
        'extra': {'exit_status': 'reasoning_loop'}, 'content': 'native loop'}))
        for i in range(malformed_count, 32)]
    selected, terminal, reason = OpenHandsPolicy().failure(candidates, [], False, RuntimeError)
    assert selected is candidates[0] and terminal is None
    assert isinstance(selected.error, OpenHandsProtocolError) and selected.error.retryable
    assert reason == 'all_branches_protocol_error'


def test_all_32_proven_loops_keep_terminal_loop_outcome():
    error = OpenHandsLimit({'content': 'native loop', 'extra': {'exit_status': 'reasoning_loop'}})
    candidates = [SimpleNamespace(branch_id=i, error=error) for i in range(32)]
    selected, terminal, reason = OpenHandsPolicy().failure(candidates, [], False, RuntimeError)
    assert selected is None and terminal.messages[0]['extra']['exit_status'] == 'reasoning_loop'


def test_real_http_format_error_keeps_attempt_open_and_next_request_can_succeed(cfg, tmp_path):
    count = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            count.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            tool = 'str_replace' if len(count) == 1 else 'terminal'
            raw = {'id': 'r-' + str(len(count)), 'usage': {'prompt_tokens': 7, 'completion_tokens': 23},
                'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                'content': None, 'tool_calls': [{'id': 't-' + str(len(count)), 'type': 'function',
                'function': {'name': tool, 'arguments': '{}'}}]}}]}
            body = json.dumps(raw).encode(); self.send_response(200)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    config = copy.deepcopy(cfg); config['serving'].update(management='external',
        base_url='http://127.0.0.1:' + str(server.server_port) + '/v1')
    config['capture']['enabled'] = False
    profile = {'bridge_listen':'127.0.0.1','bridge_advertise':'127.0.0.1','request_roles':'unqualified'}
    bridge = OpenHandsBridge(config, {'attempt_id':'a'}, profile, tmp_path, Deadline(15))
    request = {'messages':[{'role':'user','content':'test'}],
        'tools':[{'type':'function','function':{'name':'terminal','parameters':{}}}]}
    try:
        with bridge:
            auth = {'Authorization': 'Bearer ' + bridge.secret}
            first = requests.post(bridge.base_url + '/chat/completions',json=request,headers=auth,timeout=5)
            assert first.status_code == 400
            error = first.json()['error']
            assert error['type'] == 'FormatError' and error['code'] == 'format_error' and error['retryable'] is True
            assert not bridge.closed.is_set() and bridge.limit_reason is None
            corrected = copy.deepcopy(request)
            corrected['messages'].append({'role':'user','content':'FormatError: use declared tools'})
            second = requests.post(bridge.base_url + '/chat/completions',json=corrected,headers=auth,timeout=5)
            assert second.status_code == 200 and not bridge.closed.is_set()
        receipts = [json.loads(p.read_text()) for p in (tmp_path/'terminal-receipts').glob('*.json')]
        assert len(receipts) == 2
        assert sum(r['model_call_receipts'][0]['usage']['completion_tokens'] for r in receipts) == 46
        assert len(count) == 2
    finally:
        server.shutdown(); server.server_close();thread.join(timeout=2)


@pytest.mark.parametrize('format_kind', ['undeclared', 'invalid_json'])
@pytest.mark.parametrize('loop_count', [0, 31, 32])
@pytest.mark.parametrize('mode', ['parallel', 'parallel_baseline'])
def test_full_coordinator_32_branches_distinguishes_formats_from_proven_loops(loop_count, mode, format_kind):
    from rollout.decision.coordinator import DecisionCoordinator
    from rollout.adapters.openhands import Response, parse_response
    scores, decisions, released = [], [], []
    class Client:
        def call(self, action, payload):
            if action == 'results':
                return {'results': scores, 'estimator_sha256':'sha','target':'t','endpoint':'e'}
            if action == 'release': released.append(payload)
            return {}
    def record(evidence, timeout):
        decisions.append(evidence);return evidence
    cfg={'parallel':{'branch_count':32,'score_timeout_ms':1,'fallback_policy':'stop','mode':mode},
         'generation':{'seed':42},'capture':{'control_seconds':1}}
    coordinator=DecisionCoordinator(cfg,SimpleNamespace(record_decision=record),Client(),error_policy=OpenHandsPolicy())
    def generate(messages, md, seed):
        branch=md['branch_id']
        if branch < loop_count:
            native='native-'+str(branch)
            loop={k:md[k] for k in ['session_id','decision_id','branch_id','api_attempt_id']}
            loop.update(native_request_id=native,abort_completed=True)
            scores.append({'branch_id':branch,'api_attempt_id':md['api_attempt_id'],'native_request_id':native,
                'reason':'reasoning_loop','status':'failed','reasoning_loop':loop})
            return Response({'id':native,'choices':[{'finish_reason':'abort','message':{'role':'assistant'}}]})
        scores.append({'branch_id':branch,'status':'scored','score':0.1})
        return Response({'id':'format-'+str(branch),'choices':[{'finish_reason':'tool_calls',
            'message':{'role':'assistant','content':None,'tool_calls':[{'id':'t-'+str(branch),'type':'function',
            'function':{'name':'not_offered' if format_kind=='undeclared' else 'terminal',
                'arguments':'{}' if format_kind=='undeclared' else '[broken JSON'}}]}}]})
    request={'messages':[],'tools':[{'type':'function','function':{'name':'terminal'}}]}
    args=([],{'session_id':'s','task_id':'task','step_id':0,'query_id':'q'},generate,
          lambda response:parse_response(response,request),lambda response:{'cost':0})
    if loop_count==32:
        with pytest.raises(OpenHandsLimit) as raised:coordinator.run(*args)
        assert raised.value.messages[0]['extra']['limit_reason']=='all_parallel_branches_reasoning_loop'
    else:
        with pytest.raises(OpenHandsProtocolError) as raised:coordinator.run(*args)
        assert raised.value.messages[0]['role']=='user'
        assert 'FormatError' in raised.value.messages[0]['content']
    assert len(decisions)==1 and len(decisions[0]['candidates'])==32 and len(released)==1
    assert sum(r['reason']=='reasoning_loop' for r in decisions[0]['candidates'])==loop_count
