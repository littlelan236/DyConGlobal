"""Attempt-bound OpenAI bridge: buffer candidates, return exactly one winner."""
from __future__ import annotations

import copy
import json
import os
import secrets
import threading
import select
import socket
import time
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
        self.lifecycle_lock = threading.Lock()
        self._request_state = threading.local()
        self.cancel_origin = None
        self.cancel_reason = None
        self._cancel_started = False
        self._cancel_finished = threading.Event()
        self.handlers = {}
        self.sdk_events_lock = threading.Lock()
        self.sdk_events = {}

    def record_sdk_event(self, payload):
        from runtime.agents.openhands_transport import safe_event
        event = safe_event(payload); sdk_id = event['sdk_request_id']
        with self.sdk_events_lock:
            if sdk_id not in self.sdk_events and len(self.sdk_events) >= 4096:
                raise ValueError('SDK diagnostic call bound exceeded')
            row = self.sdk_events.setdefault(sdk_id, {'schema_version': 1,
                'binding': self.binding, 'execution_attempt_id': self.owner_execution,
                'sdk_request_id': sdk_id, 'source': 'actual_sdk_http_transport', 'events': []})
            if len(row['events']) >= 32: raise ValueError('SDK diagnostic event bound exceeded')
            row['events'].append({**event, 'observed_at': now()})
            write_json(self.output / 'diagnostics' / ('sdk-transport-' + sdk_id + '.json'), row)


    def _state(self):
        return getattr(self._request_state, "value", None)

    def _event(self, name, state=None, **details):
        state = state if state is not None else self._state()
        if state is None:
            return
        with state["lock"]:
            event = {"event": name, "at": now()}
            event.update({key: value[:256] if isinstance(value, str) else value
                          for key, value in details.items()
                          if isinstance(value, (str, int, float, bool)) or value is None})
            if len(state["events"]) < 256:
                state["events"].append(event)
            else:
                state["events_dropped"] = state.get("events_dropped", 0) + 1
            state["phase"] = name
            if name == "upstream_started":
                state["upstream_active"] += 1
            elif name == "upstream_terminal":
                state["upstream_active"] = max(0, state["upstream_active"] - 1)
                state["upstream_terminal"] = True
            elif name == "result_ready":
                state["result_ready"] = True
            elif name == "response_write_started":
                state["response_write_started"] = True
            elif name == "response_write_finished":
                state["response_write_finished"] = True
                state["delivery_status"] = "write_complete"
            elif name == "handler_terminal":
                state["handler_terminal"] = True

    def _flush_events(self, state):
        with state["lock"]:
            row = {"schema_version": 1, "request_id": state["request_id"],
                   "binding": self.binding, "sdk_request_id": state.get("sdk_request_id"),
                   **{key: state.get(key) for key in (
                       "phase", "delivery_status", "upstream_terminal", "upstream_active",
                       "peer_closed", "handler_terminal", "events_dropped")},
                   "events": list(state["events"])}
        path = self.output / "diagnostics" / ("request-lifecycle-" + state["request_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)

    def _peer_lost(self, state, event, error_type=None):
        # Claim and commit under the same request lock as upstream completion.
        # Native cancellation is session-scoped, so an in-flight peer loss
        # explicitly terminates THIS physical attempt; a terminal response never does.
        with state["lock"]:
            state["peer_closed"] = True
            cancel = (state["upstream_active"] > 0 and not state["result_ready"]
                      and not state["handler_terminal"] and not state["peer_cancel_claimed"])
            if cancel:
                state["peer_cancel_claimed"] = True
                self.closed.set()
        self._event(event, state=state, error_type=error_type,
                    cancel_claimed=cancel, policy="terminate_attempt_only_while_upstream_active")
        if cancel:
            self.abort("SDK peer disconnected during active upstream request",
                       origin="peer_disconnect", state=state)

    def _watch_peer(self, connection, state):
        while not state["done"].wait(0.25):
            with state["lock"]:
                if state["handler_terminal"] or state["response_write_finished"]:
                    return
            try:
                if select.select([connection], [], [], 0)[0]:
                    if not connection.recv(1, socket.MSG_PEEK):
                        self._peer_lost(state, "peer_eof")
                        return
            except (OSError, ValueError) as exc:
                self._peer_lost(state, "peer_socket_error", type(exc).__name__)
                return

    def abort(self, reason, *, origin="attempt_lifecycle", state=None):
        """Idempotently terminate this bridge's session, preserving first cause."""
        self.closed.set()
        with self.lifecycle_lock:
            if self._cancel_started:
                return
            self._cancel_started = True
            self.cancel_origin, self.cancel_reason = origin, reason
        self._event("cancel_requested", state=state, origin=origin, reason=reason)
        try:
            if self.cfg['serving']['management'] != 'managed':
                write_json(self.output / 'request-cancellation.json', {
                    'reason': reason, 'origin': origin, 'scope': 'attempt_session',
                    'abort_completed': None, 'native_control_available': False, 'at': now()})
                return
            session = self.capture['session_id'] if self.capture else self.binding.get('session_id')
            if not session or not self.binding.get('run_id') or not self.owner_execution:
                raise ValueError('Native cancellation requires run, execution and session owner')
            endpoint = self.cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
            response = requests.post(endpoint + '/dycon/capture/engine/cancel',
                json={'run_id': self.binding['run_id'], 'session_id': session,
                      'execution_attempt_id': self.owner_execution, 'wait_seconds': 15},
                headers={'X-DyCon-Cancel-Token': self.control_token},
                timeout=min(20, self.profile.get('cleanup_seconds', 60)))
            response.raise_for_status()
            value = {**response.json(), 'reason': reason, 'origin': origin,
                     'scope': 'attempt_session', 'at': now()}
            write_json(self.output / 'request-cancellation.json', value)
            self._event("cancel_confirmed", state=state, origin=origin,
                        abort_completed=bool(value.get('abort_completed')))
            if not value.get('abort_completed'):
                raise RuntimeError('Attempt native cancellation has unconfirmed requests')
        except Exception as exc:
            write_json(self.output / 'request-cancellation-error.json', {
                'reason': reason, 'origin': origin, 'error_type': type(exc).__name__, 'at': now()})
        finally:
            self._cancel_finished.set()

    def _check(self):
        if self.closed.is_set():
            self._event("cancel_observed", origin=self.cancel_origin)
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
        self._event("admission_check")
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
            record = {"schema_version": 1, "request_id": (self._state() or {}).get("request_id") or "req-" + uuid4().hex,
                      "binding": self.binding, "role": role, "request": original,
                      "incoming_request_sha256": hash_json(original), "started_at": now(),
                      "step_id": self.decisions if role == "agent_decision" else None}
            state = self._state()
            if state is not None:
                state["request_id"] = record["request_id"]
            record["api_attempt_index"] = sum(
                previous.get("step_id") == record["step_id"] and previous.get("role") == role
                for previous in self.records)
            record["sdk_request_id"] = (self._state() or {}).get("sdk_request_id")
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

            http_state = self._state()

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
                self._event("upstream_started", state=http_state, api_attempt_id=api_id)
                try:
                    from rollout.progress import generation_phase
                    with generation_phase(self.runtime):
                        response = requests.post(self.cfg["serving"]["base_url"].rstrip("/") + "/chat/completions",
                        json=body, headers={"Authorization": "Bearer " + os.environ.get(
                            self.cfg["serving"]["api_key_env"], "EMPTY")},
                        timeout=self.deadline.remaining(self.cfg["timeouts"]["model_seconds"]))
                except BaseException as exc:
                    call.update(status='cancelled' if isinstance(exc, Cancelled) else 'failed',
                                error_type=type(exc).__name__, finished_at=now(), usage_known=False)
                    self._persist(record)
                    self._event("upstream_terminal", state=http_state, status="failed",
                                error_type=type(exc).__name__)
                    self.abort('upstream_' + type(exc).__name__, origin='upstream_error', state=http_state)
                    raise
                self._event("upstream_terminal", state=http_state,
                            status="http_response", http_status=response.status_code)
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
            self._event("result_ready")
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

            def do_GET(self):
                if self.path != "/v1/dycon-health":
                    self.send_error(404)
                    return
                if self.headers.get("Authorization") != "Bearer " + bridge.secret:
                    self.send_error(401)
                    return
                body = json.dumps({"run_id": bridge.binding.get("run_id"),
                                   "attempt_id": bridge.binding.get("attempt_id"), "model_requests": 0}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write(self, state, status, data, content_type="application/json"):
                bridge._event("response_write_started", status=status)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                except (OSError, ValueError) as exc:
                    bridge._event("response_write_failed", error_type=type(exc).__name__)
                else:
                    bridge._event("response_write_finished", bytes=len(data), status=status)

            def do_POST(self):
                if self.path == "/v1/dycon/sdk-events":
                    if self.headers.get("Authorization") != "Bearer " + bridge.secret:
                        self.send_error(401); return
                    try:
                        self.connection.settimeout(2)
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 32768: raise ValueError('Invalid SDK diagnostic size')
                        body = self.rfile.read(length)
                        if len(body) != length: raise ValueError('Incomplete SDK diagnostic')
                        bridge.record_sdk_event(json.loads(body))
                        self.send_response(204); self.send_header('Content-Length', '0'); self.end_headers()
                    except (ValueError, TypeError, AttributeError):
                        self.send_error(400)
                    return
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                if self.headers.get("Authorization") != "Bearer " + bridge.secret:
                    self.send_error(401)
                    return
                state = {"lock": threading.Lock(), "done": threading.Event(), "events": [],
                         "request_id": "req-" + uuid4().hex, "phase": "handler_started",
                         "upstream_terminal": False, "upstream_active": 0, "result_ready": False,
                         "response_write_started": False, "response_write_finished": False,
                         "handler_terminal": False, "peer_closed": False,
                         "peer_cancel_claimed": False, "delivery_status": "delivery_unconfirmed"}
                import re
                sdk_id = self.headers.get('X-DyCon-SDK-Request-ID', '')
                state['sdk_request_id'] = sdk_id if re.fullmatch(r'sdk-[a-f0-9]{32}', sdk_id) else None
                bridge._request_state.value = state
                current = threading.current_thread()
                with bridge.lifecycle_lock:
                    bridge.handlers[current] = state
                monitor = None
                try:
                    bridge._event("request_received")
                    self.connection.settimeout(bridge.profile.get('cleanup_seconds', 60))
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Incomplete request body")
                    request = json.loads(body)
                    bridge._event("body_read", bytes=length)
                    monitor = threading.Thread(target=bridge._watch_peer,
                        args=(self.connection, state), name='dycon-openhands-peer', daemon=True)
                    monitor.start()
                    response = bridge.complete(request)
                    bridge._event("result_ready")
                    stream = request.get("stream", False)
                    data = buffered_sse(response) if stream else json.dumps(response).encode()
                    self._write(state, 200, data, "text/event-stream" if stream else "application/json")
                except UpstreamModelError as exc:
                    self._write(state, exc.status, json.dumps(exc.payload).encode())
                except Cancelled:
                    bridge._event("request_cancelled", origin=bridge.cancel_origin or "run_or_deadline")
                    data = {"error": {"message": "Owning attempt stopped", "type": "cancelled",
                        "correlation_id": state['request_id'], "retryable": False}}
                    self._write(state, 499, json.dumps(data).encode())
                except OpenHandsProtocolError as exc:
                    bridge._event("format_error", error_type="FormatError", retryable=True)
                    self._write(state, 400, json.dumps({"error": {"message": str(exc),
                        "type": "FormatError", "code": "format_error",
                        "correlation_id": state['request_id'], "retryable": True}}).encode())
                except OpenHandsLimit as exc:
                    self._write(state, 400, json.dumps({"error": {"message": str(exc), "type": "limit",
                        "correlation_id": state['request_id'], "retryable": False}}).encode())
                except Exception as exc:
                    bridge._event("handler_error", error_type=type(exc).__name__)
                    kind = 'transport' if isinstance(exc, requests.RequestException) else 'unknown'
                    self._write(state, 502, json.dumps({'error': {
                        'message': 'DyCon request failed; inspect the owning attempt receipt',
                        'type': kind, 'correlation_id': state['request_id'], 'retryable': False}}).encode())
                finally:
                    bridge._event("handler_terminal")
                    state['done'].set()
                    if monitor is not None:
                        monitor.join(timeout=bridge.profile.get('cleanup_seconds', 60))
                        if monitor.is_alive():
                            bridge._event("monitor_cleanup_timeout")
                    bridge._flush_events(state)
                    with bridge.lifecycle_lock:
                        bridge.handlers.pop(current, None)
                    bridge._request_state.value = None

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
        until = time.monotonic() + self.profile.get('cleanup_seconds', 60)
        self.closed.set()
        self.server.shutdown()
        with self.lifecycle_lock:
            handlers = list(self.handlers.items())
        for _, state in handlers:
            state['done'].set()
        if self.lock.locked():
            self.abort('agent attempt terminal', origin='attempt_exit')
        for thread, _ in handlers:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0, until - time.monotonic()))
        self.server.server_close()
        self.thread.join(timeout=max(0, until - time.monotonic()))
        if any(thread.is_alive() for thread, _ in handlers):
            raise TimeoutError('OpenHands bridge handler exceeded attempt cleanup budget')
        if self.lock.locked():
            if not self.lock.acquire(timeout=max(0, until - time.monotonic())):
                raise TimeoutError('OpenHands bridge still has an in-flight API attempt')
            self.lock.release()
