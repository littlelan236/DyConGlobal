"""Exercise the published SDK, not a model-free startup probe (no GPU)."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='true'
os.environ['OPENHANDS_SUPPRESS_BANNER']='1'
os.environ['DYCON_OPENHANDS_MODEL_SECONDS']='1'
from openhands.sdk import LLM
from openhands.sdk.llm import Message, TextContent

class SDKTransportLiveTest(unittest.TestCase):
    def test_timeout_is_one_native_request_for_agent_and_condenser(self):
        events, sends = [], []
        gate = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_POST(self):
                body=self.rfile.read(int(self.headers['Content-Length']))
                if self.path.endswith('/dycon/sdk-events'):
                    events.append(json.loads(body)); self.send_response(204)
                    self.send_header('Content-Length','0'); self.end_headers(); return
                sends.append(self.headers.get('X-DyCon-SDK-Request-ID'))
                gate.wait(8)
                self.close_connection=True
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for role in ['agent','condenser']:
                llm=LLM(model='openai/qwen3.6-27b',api_key='injection-only',
                    base_url='http://127.0.0.1:'+str(server.server_port)+'/v1',
                    timeout=1,num_retries=0,usage_id=role)
                before=len(sends)
                with self.assertRaises(Exception):
                    llm.completion([Message(role='user',content=[TextContent(text='injection')])])
                self.assertEqual(len(sends)-before,1)
                failed=[e for e in events if e['event']=='sdk_call_failed' and e['role']==role]
                self.assertEqual(len(failed),1)
                self.assertEqual(failed[0]['phase'],'read')
                http_events=[e for e in events if e['event']=='sdk_http_failed' and e['role']==role]
                self.assertEqual(len(http_events),1)
                self.assertEqual(http_events[0]['openai_retry_count'],0)
                self.assertEqual(http_events[0]['http_timeout'],dict.fromkeys(['connect','read','write','pool'],1))
                self.assertIn('httpx.ReadTimeout',http_events[0]['exception_types'])
                self.assertEqual(sends[-1],failed[0]['sdk_request_id'])
            print('PUBLISHED_SDK_TIMEOUT_EVIDENCE',json.dumps(events))
        finally:
            gate.set();server.shutdown();server.server_close();thread.join(timeout=2)


class SDKFormatRecoveryLiveTest(unittest.TestCase):
    def test_native_agent_appends_correction_and_finishes_next_turn(self):
        import tempfile
        from openhands.sdk import Agent, Conversation
        from openhands.sdk.event import MessageEvent
        from openhands.sdk.conversation.state import ConversationExecutionStatus
        requests, events = [], []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path.endswith('/dycon/sdk-events'):
                    self.send_response(204); self.send_header('Content-Length','0');self.end_headers();return
                requests.append(body)
                if len(requests) == 1:
                    status = 400
                    payload = {'error': {'message':'Response used an undeclared tool',
                        'type':'FormatError','code':'format_error','retryable':True,
                        'correlation_id':'req-'+'a'*32}}
                else:
                    status = 200
                    payload = {'id':'format-corrected','object':'chat.completion','created':1,'model':'gpt-4o',
                        'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant',
                        'content':None,'tool_calls':[{'id':'finish-corrected','type':'function',
                        'function':{'name':'finish','arguments':json.dumps({'message':'done'})}}]}}],
                        'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
                data=json.dumps(payload).encode();self.send_response(status)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            llm=LLM(model='openai/gpt-4o',api_key='injection-only',
                base_url='http://127.0.0.1:'+str(server.server_port)+'/v1',timeout=2,num_retries=0)
            with tempfile.TemporaryDirectory(prefix='dycon-format-agent-') as work:
                agent=Agent(llm=llm,tools=[])
                conversation=Conversation(agent=agent,workspace=work,callbacks=[events.append],max_iteration_per_run=3)
                conversation.send_message('Complete this controlled format-recovery test.')
                conversation.run()
                self.assertEqual(conversation.state.execution_status,ConversationExecutionStatus.FINISHED)
                corrections=[e for e in events if isinstance(e,MessageEvent) and e.source=='user'
                    and 'FormatError' in str(e.llm_message)]
                self.assertEqual(len(corrections),1)
                self.assertEqual(len(requests),2)
                self.assertIn('FormatError',json.dumps(requests[1]['messages']))
                print('NATIVE_AGENT_FORMAT_RECOVERY',json.dumps({'http_requests':len(requests),
                    'correction_messages':len(corrections),'execution_status':str(conversation.state.execution_status)}))
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)

if __name__=='__main__': unittest.main()

