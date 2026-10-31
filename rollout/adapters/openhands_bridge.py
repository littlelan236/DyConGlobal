"""Attempt-bound OpenAI bridge: buffer candidates, return exactly one winner."""
from __future__ import annotations

import copy
import json
import os
import secrets
import threading
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import requests

from core.cancellation import Cancelled
from core.io import hash_json, now, sha256, write_json
from hidden.backends.vllm.online import CaptureClient
from rollout.adapters.openhands import (OpenHandsPolicy, Response, parse_response, request_role,
                                        preserve_reasoning_history, OpenHandsLimit, OpenHandsProtocolError)
from rollout.decision.coordinator import DecisionCoordinator, active
from rollout.loop_escape import LoopEscapeHistory
from rollout.step_loop import StepLoopDetector
from core.action_identity import canonical_action, ActionIdentityUnavailable


class UpstreamModelError(RuntimeError):
    def __init__(self,status,payload):
        self.status,self.payload=status,payload
        super().__init__(f'Model HTTP {status}: '+str(payload)[:1000])


def buffered_sse(raw):
    """Serialize only a completed winner; no losing candidate delta escapes."""
    choice = raw["choices"][0]
    message = copy.deepcopy(choice["message"])
    for index, call in enumerate(message.get("tool_calls") or []):
        call["index"] = index
    base = {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"choices", "usage"}}
    base["object"] = "chat.completion.chunk"
    chunks = [{**base, "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
              {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
               "usage": raw.get("usage", {})}]
    return ("".join("data: " + json.dumps(c, ensure_ascii=False) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n").encode("utf-8")


class OpenHandsBridge:
    def __init__(self, cfg, binding, profile, output, deadline, runtime=None, capture=None):
        self.cfg, self.binding, self.profile = cfg, copy.deepcopy(binding), profile
        self.output, self.deadline = Path(output), deadline
        self.runtime, self.capture = runtime or {}, capture
        self.owner_execution = self.runtime.get('execution_attempt_id', os.environ.get('DYCON_EXECUTION_ATTEMPT'))
        self.control_token = self.runtime.get('cancel_token', os.environ.get('DYCON_CANCEL_TOKEN', ''))
        self.secret = secrets.token_urlsafe(32)
        self.records = []
        self.query_ids = {}
        self.decisions = 0
        self.lock = threading.Lock()
        self.history = LoopEscapeHistory()
        self.server = self.thread = None
        self.limit_reason = None
        self.loop_hit = None
        self.step_loop = StepLoopDetector(cfg['agent'].get('loop_repeat_limit'), cfg['agent'].get('loop_max_period', 7))
        self.closed = threading.Event()
        self.cancel_lock = threading.Lock()

    def abort(self, reason):
        """Stop only this attempt's session, then wait for terminal native output."""
        self.closed.set()
        if self.cfg['serving']['management'] != 'managed' or not self.cancel_lock.acquire(blocking=False):
            return
        try:
            endpoint = self.cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
            response = requests.post(endpoint + '/dycon/capture/engine/cancel',
                json={'run_id': self.binding.get('run_id'),
                      'session_id': self.capture['session_id'] if self.capture else self.binding.get('session_id'),
                      'execution_attempt_id': self.owner_execution, 'wait_seconds': 15},
                headers={'X-DyCon-Cancel-Token': self.control_token}, timeout=20)
            response.raise_for_status()
            value = {**response.json(), 'reason': reason, 'at': now()}
            write_json(self.output / 'request-cancellation.json', value)
            if not value.get('abort_completed'):
                raise RuntimeError('Attempt native cancellation has unconfirmed requests')
        except Exception as exc:
            write_json(self.output / 'request-cancellation-error.json', {'reason': reason, 'error': repr(exc), 'at': now()})
        finally:
            self.cancel_lock.release()

    def _check(self):
        if self.closed.is_set():
            raise Cancelled('OpenHands attempt disconnected or stopped')
        cancellation = self.runtime.get("cancellation")
        if cancellation:
            cancellation.check()
        self.deadline.check()

    def _persist(self, record):
        write_json(self.output / "requests" / (record["request_id"] + ".json"), record)

    def _terminal(self, record, status, reason=None):
        receipt = {"schema_version": 1, "request_id": record["request_id"],
                   "binding": record["binding"], "api_attempt_index": record["api_attempt_index"],
                   "status": status, "reason": reason, "started_at": record["started_at"],
                   "finished_at": record.get("finished_at", now()),
                   "model_call_receipts": [{k: call.get(k) for k in ('status', 'response_id', 'usage', 'error_type', 'finished_at', 'usage_known')}
                                           for call in record.get('model_calls', [])],
                   "model_calls": len(record.get("model_calls", []))}
        path = self.output / "terminal-receipts" / (record["request_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError("Conflicting API terminal receipt")
        if not path.exists():
            write_json(path, receipt)

    def complete(self, incoming):
        self._check()
        if not self.lock.acquire(blocking=False):
            raise ValueError("Concurrent requests on one OpenHands session require explicit role binding")
        record = None
        try:
            original = copy.deepcopy(incoming)
            if not isinstance(original.get("messages"), list) or incoming.get("n", 1) != 1:
                raise ValueError("Expected OpenAI chat messages and n=1")
            role = request_role(original, self.profile)
            strict_roles=self.cfg['capture']['enabled'] or self.cfg.get('parallel',{}).get('enabled')
            # Baseline requests can be inspected before profile qualification.
            if role == 'unqualified' and not strict_roles:
                role=request_role(original,{**self.profile,'request_roles':'openhands-1.22.1'})
            record = {"schema_version": 1, "request_id": "req-" + uuid4().hex,
                      "binding": self.binding, "role": role, "request": original,
                      "incoming_request_sha256": hash_json(original), "started_at": now(),
                      "step_id": self.decisions if role == "agent_decision" else None}
            record["api_attempt_index"] = sum(
                previous.get("step_id") == record["step_id"] and previous.get("role") == role
                for previous in self.records)
            self.records.append(record)
            self._persist(record)
            if strict_roles and (role=='unqualified' or self.profile.get('request_roles')!='openhands-1.22.1'):
                raise ValueError("Unqualified OpenHands request role cannot enter capture or parallel")
            if role == "agent_decision" and self.decisions >= self.cfg["agent"]["step_limit"]:
                self.limit_reason = "step_limit"
                raise ValueError("Frozen agent step limit reached")
            # A later OpenHands request contains actual tool results from the
            # preceding winner. Materialize completion only from that evidence,
            # never when merely returning a side-effect-capable response.
            observed={m.get('tool_call_id') for m in original['messages'] if m.get('role')=='tool'}
            for previous in self.records[:-1]:
                summary=(previous.get('extra') or {}).get('dycon_parallel')
                response=previous.get('selected_response') or {}
                choices=response.get('choices') or []
                calls=((choices[0].get('message') or {}).get('tool_calls') or []) if choices else []
                ids={c['id'] for c in calls}
                if summary and ids and ids<=observed and not previous.get('execution_observed'):
                    details={'source':'OpenHands_tool_results_in_followup_request',
                             'tool_call_ids':sorted(ids),'followup_request_id':record['request_id']}
                    self.runtime['capture_controller'].record_parallel_action(summary,'completed',details)
                    previous['execution_observed']=details
                    self._persist(previous)
            self._persist(record)
            request = copy.deepcopy(original)
            if self.cfg['generation'].get('preserve_thinking'):
                record['restored_reasoning_messages']=preserve_reasoning_history(request,self.records[:-1])
            # OpenHands/LiteLLM request metadata is not part of vLLM's chat
            # completion schema. Keep it in the bound incoming request, but do
            # not forward provider-specific fields that make vLLM reject the
            # otherwise valid tool request with HTTP 400.
            for key in ("api_key", "authorization", "password", "dycon_capture", "capture_select",
                        "stream_options", "prompt_cache_key", "reasoning_effort"):
                request.pop(key, None)
            request.pop("steering", None)
            request.update(model=self.cfg["model"]["served_name"], n=1, stream=False)
            # Reuse the existing single generation/capture configuration mapping.
            from rollout.model import model_config
            options = model_config(self.cfg, self.capture if role == "agent_decision" else False)["model_kwargs"]
            for key in ("temperature", "top_p", "max_tokens", "seed", "presence_penalty"):
                request[key] = options[key]
            request.update(options["extra_body"])
            if self.cfg['serving']['management'] == 'managed':
                request['dycon_request_owner'] = {
                    'run_id': self.binding.get('run_id'), 'execution_attempt_id': self.owner_execution,
                    'session_id': self.capture['session_id'] if self.capture else self.binding.get('session_id'),
                    'required_max_tokens': self.cfg['generation']['max_tokens'],
                    'max_model_len': self.cfg['serving']['max_model_len'], 'request_role': role}
            metadata = copy.deepcopy(self.capture) if role == "agent_decision" and self.capture else None
            if metadata:
                query = self.query_ids.setdefault((self.decisions, record["incoming_request_sha256"]), uuid4().hex)
                metadata.update(step_id=self.decisions, query_id=query, api_attempt_id=uuid4().hex)
                request["dycon_capture"] = metadata
                record["capture"] = metadata
            record["effective_request_sha256"] = hash_json(request)
            record["effective_request"] = copy.deepcopy(request)
            record["model_calls"] = []
            self._persist(record)

            def generate(messages, capture, seed):
                self._check()
                body = copy.deepcopy(request)
                body["messages"], body["seed"] = messages, seed
                if capture:
                    body["dycon_capture"] = capture
                api_id = (capture or {}).get('api_attempt_id') or uuid4().hex
                if body.get('dycon_request_owner'):
                    body['dycon_request_owner']['api_attempt_id'] = api_id
                call = {"request_sha256": hash_json(body), "seed": seed,
                        "capture": copy.deepcopy(capture), "started_at": now(), 'status': 'running', 'api_attempt_id': api_id}
                record["model_calls"].append(call)
                self._persist(record)
                try:
                    response = requests.post(self.cfg["serving"]["base_url"].rstrip("/") + "/chat/completions",
                    json=body, headers={"Authorization": "Bearer " + os.environ.get(
                        self.cfg["serving"]["api_key_env"], "EMPTY")},
                    timeout=self.deadline.remaining(self.cfg["timeouts"]["model_seconds"]))
                except BaseException as exc:
                    call.update(status='cancelled' if isinstance(exc, Cancelled) else 'failed',
                                error_type=type(exc).__name__, finished_at=now(), usage_known=False)
                    self._persist(record)
                    self.abort('upstream_' + type(exc).__name__)
                    raise
                if not response.ok:
                    call.update(status="failed", http_status=response.status_code, finished_at=now())
                    try:payload=response.json()
                    except ValueError:payload={'error':{'message':'Upstream model request failed','type':'upstream_error'}}
                    raise UpstreamModelError(response.status_code,payload)
                raw_response = response.json()
                call.update(status="completed", response_id=raw_response.get("id"), response=raw_response,
                            usage=raw_response.get("usage", {}), usage_known=bool(raw_response.get('usage')), finished_at=now())
                if self.runtime.get('request_control_enabled'):
                    owner = body['dycon_request_owner']
                    endpoint = self.cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
                    terminal = requests.post(endpoint + '/dycon/capture/engine/request-terminal', json=owner,
                        headers={'X-DyCon-Cancel-Token': self.control_token},
                        timeout=min(20, self.deadline.remaining(self.cfg['timeouts']['model_seconds'])))
                    terminal.raise_for_status()
                    native = terminal.json()
                    if native.get('owner') != owner:
                        raise ValueError('Native terminal owner differs from the physical API attempt')
                    path = self.output / 'native-terminals' / (api_id + '.json')
                    write_json(path, native)
                    call['native_terminal'] = {'path': str(path), 'sha256': sha256(path)}
                    raw_response['dycon_native_finish_reason'] = native['finish_reason']
                    raw_response['dycon_reasoning_loop'] = native.get('reasoning_loop')
                self._persist(record)
                result = Response(raw_response)
                self._check()
                return result

            if role == "agent_decision" and active(self.cfg, self.decisions):
                coordinator = DecisionCoordinator(self.cfg, self.runtime.get("capture_controller"),
                    cancellation=self.runtime.get("cancellation"), loop_escape_history=self.history,
                    error_policy=OpenHandsPolicy())
                message = coordinator.run(request["messages"], metadata, generate,
                    lambda result: parse_response(result, original), lambda result: {"cost": 0.0},
                    request_envelope={k: v for k, v in request.items() if k not in {"dycon_capture", "seed"}})
                raw = message["extra"]["response"]
                record["extra"] = {"dycon_parallel": message["extra"]["dycon_parallel"]}
                record["capture"] = metadata
            else:
                raw = generate(request["messages"], metadata, self.cfg["generation"]["seed"]).model_dump()
                if role == "agent_decision":
                    parse_response(raw, original)
                    if metadata:
                        client = CaptureClient(self.runtime["capture_control_url"],
                                               timeout=self.cfg["capture"]["control_seconds"])
                        client.wait_output(metadata["session_id"], metadata["step_id"], raw["id"])
            if role == "agent_decision":
                selected = parse_response(raw, original)
                try:
                    action, _ = canonical_action(selected)
                except ActionIdentityUnavailable:
                    action = None
                fingerprint = hash_json({'reasoning': selected.get('reasoning_content') or '',
                    'content': selected.get('content') or '', 'action': action})
                hit = self.step_loop.feed(fingerprint, match_policy='exact_reasoning_content_and_canonical_actions_v1')
                if hit is not None:
                    self.loop_hit = hit
                    self.limit_reason = 'repeated_reasoning_action_loop'
                    record.update(rejected_response=copy.deepcopy(raw), loop_detection=hit, action_executed=False)
                    write_json(self.output / 'loop-detection.json', {**hit, 'request_id': record['request_id'],
                        'response_sha256': hash_json(raw), 'binding': self.binding, 'at': now()})
                    raise OpenHandsLimit({'content': 'Repeated reasoning-action loop detected',
                        'extra': {'exit_status': 'LimitsExceeded', 'limit_reason': self.limit_reason, 'loop_detection': hit}})
                self.decisions += 1
            record.update(selected_response=raw, response_sha256=hash_json(raw), finished_at=now(), status="completed")
            self._persist(record)  # Persist before returning a side-effect-capable tool bundle.
            self._terminal(record, "completed")
            return raw
        except BaseException as exc:
            if record is not None:
                status = "cancelled" if isinstance(exc, Cancelled) else "failed"
                reason = ("cancelled" if isinstance(exc, Cancelled) else
                          "limit" if isinstance(exc, OpenHandsLimit) else
                          "protocol" if isinstance(exc, OpenHandsProtocolError) else type(exc).__name__)
                if isinstance(exc, OpenHandsLimit):
                    payload = exc.messages[0].get('extra', {}) if exc.messages else {}
                    self.limit_reason = self.limit_reason or ('reasoning_loop' if payload.get('exit_status') == 'reasoning_loop' else 'generation_limit')
                if isinstance(exc, OpenHandsProtocolError):
                    self.step_loop.clear()
                setattr(exc, 'dycon_request_id', record['request_id'])
                record.update(error=repr(exc), status=status, finished_at=now())
                self._persist(record)
                self._terminal(record, status, reason)
            raise
        finally:
            self.lock.release()

    def __enter__(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Default HTTP logging would expose the capability URL.

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                if self.headers.get("Authorization") != "Bearer " + bridge.secret:
                    self.send_error(401)
                    return
                done = threading.Event()
                monitor = None
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    request = json.loads(self.rfile.read(length))
                    def watch_peer():
                        while not done.wait(0.25):
                            try:
                                if select.select([self.connection], [], [], 0)[0]:
                                    if not self.connection.recv(1, socket.MSG_PEEK):
                                        bridge.abort('SDK peer disconnected')
                                        return
                            except OSError:
                                bridge.abort('SDK peer socket closed')
                                return
                    monitor = threading.Thread(target=watch_peer, name='dycon-openhands-peer', daemon=True)
                    monitor.start()
                    response = bridge.complete(request)
                    stream = request.get("stream", False)
                    data = buffered_sse(response) if stream else json.dumps(response).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except UpstreamModelError as exc:
                    # The SDK needs the real 400 ContextWindowExceeded error
                    # to activate its own condenser rather than retrying a 502.
                    data=json.dumps(exc.payload).encode()
                    self.send_response(exc.status)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Cancelled:
                    self.send_error(499, "Run cancelled")
                except (OpenHandsLimit, OpenHandsProtocolError) as exc:
                    error_type = "limit" if isinstance(exc, OpenHandsLimit) else "protocol"
                    data=json.dumps({"error":{"message":str(exc),"type":error_type,
                        'correlation_id':getattr(exc,'dycon_request_id',None),'retryable':False}}).encode()
                    self.send_response(400); self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                except Exception as exc:
                    # Detailed evidence is kept locally; never echo credentials or raw prompts.
                    kind = 'transport' if isinstance(exc, requests.RequestException) else 'unknown'
                    data=json.dumps({'error':{'message':'DyCon request failed; inspect the owning attempt receipt',
                        'type':kind,'correlation_id':getattr(exc,'dycon_request_id',None),'retryable':False}}).encode()
                    self.send_response(502); self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                finally:
                    done.set()

        bind_error = None
        for _ in range(32):
            try:
                self.server = ThreadingHTTPServer((self.profile["bridge_listen"], 0), Handler)
                break
            except OSError as exc:
                bind_error = exc
                if getattr(exc, "errno", None) not in {48, 98, 10048}:
                    raise
        else:
            raise bind_error or OSError("Could not bind the attempt-local OpenHands bridge")
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="dycon-openhands-bridge", daemon=True)
        self.thread.start()
        host = self.profile["bridge_advertise"]
        self.base_url = f"http://{host}:{self.server.server_port}/v1"
        return self

    def __exit__(self, *args):
        self.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.lock.locked():
            self.abort('agent attempt terminal')
            if not self.lock.acquire(timeout=self.profile.get('cleanup_seconds', 60)):
                raise TimeoutError('OpenHands bridge still has an in-flight API attempt')
            self.lock.release()
