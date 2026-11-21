"""Attempt-bound OpenAI bridge: buffer candidates, return exactly one winner."""
from __future__ import annotations

import copy
import errno
import json
import os
import re
import secrets
import selectors
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import requests

from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.cancellation import Cancelled
from core.io import hash_json, now, sha256, write_json
from hidden.backends.vllm.online import CaptureClient
from rollout.adapters.openhands import (
    OpenHandsLimit,
    OpenHandsPolicy,
    OpenHandsProtocolError,
    Response,
    parse_response,
    preserve_reasoning_history,
    request_role,
)
from rollout.decision.coordinator import DecisionCoordinator, active
from rollout.loop_escape import LoopEscapeHistory
from rollout.sample_token_budget import BudgetStopped, SampleTokenBudget
from rollout.step_loop import StepLoopDetector


class UpstreamModelError(RuntimeError):
    def __init__(self,status,payload):
        self.status,self.payload=status,payload
        super().__init__(f'Model HTTP {status}: '+str(payload)[:1000])


def upstream_error_evidence(payload):
    evidence = {'upstream_error_payload_sha256': hash_json(payload)}
    error = payload.get('error') if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return evidence
    for key in ('type', 'code', 'param'):
        value = error.get(key)
        if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value):
            evidence['upstream_error_' + key] = value
    message = error.get('message')
    if isinstance(message, str):
        from runtime.agents.openhands_transport import safe_error_message
        message = safe_error_message(message)
        if message:
            evidence['upstream_error_message'] = message
    return evidence


def is_context_window_error(status, payload):
    """Recognize explicit provider context rejection, not arbitrary HTTP 400s."""
    error = payload.get('error') if isinstance(payload, dict) else None
    if status != 400 or not isinstance(error, dict):
        return False
    if error.get('code') in ('context_length_exceeded', 'context_window_exceeded'):
        return True
    message = error.get('message')
    return (error.get('param') == 'input_tokens' and isinstance(message, str)
            and 'maximum context length' in message.lower()
            and 'tokens' in message.lower()
            and bool(re.search(r'\b(?:requested|exceed(?:s|ed)?)\b', message.lower())))


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
        self.token_budget = SampleTokenBudget(cfg, binding)
        self.query_ids = {}
        self.decisions = 0
        self.lock = threading.Lock()
        self.history = LoopEscapeHistory()
        self.server = self.thread = None
        self.limit_reason = None
        self.context_window_error = None
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
        with self.lifecycle_lock:
            active = [state for state in self.handlers.values()
                      if state.get('sdk_request_id') == sdk_id]
        for state in active:
            if event['event'] == 'sdk_http_finished':
                state['sdk_http_status'] = event.get('http_status')
                for key in ('http_error_type', 'http_error_code', 'http_error_param',
                            'http_error_message', 'http_error_body_sha256',
                            'http_error_body_omitted', 'http_error_body_unavailable'):
                    if key in event:
                        state[key] = event[key]
            elif event['event'] == 'sdk_call_failed':
                state['sdk_exception_types'] = event.get('exception_types', [])
                state['sdk_failure_phase'] = event.get('phase', 'unknown')
                if event.get('http_status') is not None:
                    state['sdk_http_status'] = event.get('http_status')
                for key in ('http_error_type', 'http_error_code', 'http_error_param',
                            'http_error_message', 'http_error_body_sha256',
                            'http_error_body_omitted', 'http_error_body_unavailable'):
                    if key in event:
                        state[key] = event[key]
            self._event('sdk_transport_event', state=state,
                        sdk_event=event['event'], http_status=event.get('http_status'),
                        exception_types=','.join(event.get('exception_types', []))[:256],
                        phase=event.get('phase'), http_error_type=event.get('http_error_type'),
                        http_error_code=event.get('http_error_code'),
                        http_error_message=event.get('http_error_message'))


    def _state(self):
        return getattr(self._request_state, "value", None)

    def _budget_stop(self, reason):
        self.limit_reason = reason
        receipt = {'binding': self.binding, 'reason': reason, 'at': now(),
                   'budget': self.token_budget.snapshot()}
        write_json(self.output / 'sample-token-budget.json', receipt)
        status = ('SampleTokenBudgetExceeded' if reason == 'sample_completion_token_budget'
                  else 'SampleTokenBudgetUsageUnknown')
        return OpenHandsLimit({'content': reason, 'extra': {'exit_status': status,
                              'limit_reason': reason, 'budget': receipt['budget']}})

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
                       "peer_closed", "handler_terminal", "events_dropped", "sdk_http_status",
                       "sdk_exception_types", "sdk_failure_phase", "http_error_type",
                       "http_error_code", "http_error_param", "http_error_message",
                       "http_error_body_sha256", "upstream_error_payload_sha256",
                       "upstream_error_http_status",
                       "upstream_error_type", "upstream_error_code",
                       "upstream_error_param", "upstream_error_message")},
                   "events": list(state["events"])}
        path = self.output / "diagnostics" / ("request-lifecycle-" + state["request_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)

    def _peer_lost(self, state, event, error_type=None, error_errno=None, error_message=None):
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
        if error_message:
            from runtime.agents.openhands_transport import safe_error_message
            error_message = safe_error_message(str(error_message))
        self._event(event, state=state, error_type=error_type,
                    error_errno=error_errno,
                    error_message=error_message,
                    cancel_claimed=cancel, policy="terminate_attempt_only_while_upstream_active")
        if cancel:
            self.abort("SDK peer disconnected during active upstream request",
                       origin="peer_disconnect", state=state)

    def _watch_peer(self, connection, state):
        try:
            # DefaultSelector uses epoll on Linux, without select()'s FD_SETSIZE bound.
            with selectors.DefaultSelector() as selector:
                registered = False
                while not state["done"].wait(0.25):
                    with state["lock"]:
                        if state["handler_terminal"] or state["response_write_finished"]:
                            return
                    if state["done"].is_set():
                        return
                    if connection.fileno() < 0:
                        self._peer_lost(state, "peer_socket_closed")
                        return
                    if not registered:
                        selector.register(connection, selectors.EVENT_READ)
                        registered = True
                    if selector.select(0):
                        try:
                            data = connection.recv(1, socket.MSG_PEEK)
                        except (BlockingIOError, InterruptedError):
                            continue
                        except OSError as exc:
                            if exc.errno not in {errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN}:
                                raise
                            self._peer_lost(state, "peer_socket_error", type(exc).__name__,
                                            exc.errno, str(exc))
                            return
                        if not data:
                            self._peer_lost(state, "peer_eof")
                            return
        except (OSError, ValueError) as exc:
            if state["done"].is_set():
                return
            if connection.fileno() < 0:
                self._peer_lost(state, "peer_socket_closed")
                return
            # A failed monitor is not evidence that the SDK disconnected.
            from runtime.agents.openhands_transport import safe_error_message
            self._event("peer_monitor_error", state=state, error_type=type(exc).__name__,
                        error_errno=getattr(exc, 'errno', None),
                        error_message=safe_error_message(str(exc)), cancel_claimed=False)

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
                try:
                    while True:
                        self._check()
                        allowance = self.token_budget.reserve(api_id, role, body['max_tokens'], record['request_id'])
                        if allowance is not None:
                            break
                        time.sleep(.05)
                except BudgetStopped as exc:
                    raise self._budget_stop(exc.reason) from exc
                body['max_tokens'] = allowance
                trajectory_capped = (self.token_budget.limit is not None
                    and self.token_budget.selected_trajectory and role == 'agent_decision'
                    and allowance == self.token_budget.remaining())
                if body.get('thinking_token_budget') is not None:
                    body['thinking_token_budget'] = min(body['thinking_token_budget'], max(0, allowance - 1))
                if body.get('dycon_request_owner'):
                    body['dycon_request_owner']['api_attempt_id'] = api_id
                    body['dycon_request_owner']['required_max_tokens'] = allowance
                call = {"request_sha256": hash_json(body), "seed": seed,
                        "capture": copy.deepcopy(capture), "started_at": now(), 'status': 'running', 'api_attempt_id': api_id}
                record["model_calls"].append(call)
                self._persist(record)
                self._event("upstream_started", state=http_state, api_attempt_id=api_id)
                try:
                    from rollout.progress import generation_phase
                    from rollout.online_admission import online_request_admission
                    with generation_phase(self.runtime), online_request_admission(self.cfg, self.runtime):
                        response = requests.post(self.cfg["serving"]["base_url"].rstrip("/") + "/chat/completions",
                        json=body, headers={"Authorization": "Bearer " + os.environ.get(
                            self.cfg["serving"]["api_key_env"], "EMPTY")},
                        timeout=self.deadline.remaining(self.cfg["timeouts"]["model_seconds"]))
                except BaseException as exc:
                    self.token_budget.record_unknown(api_id, role)
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
                    self.token_budget.record_unknown(api_id, role)
                    call.update(status="failed", http_status=response.status_code,
                                finished_at=now(), usage_known=False)
                    try:payload=response.json()
                    except ValueError:payload={'error':{'message':'Upstream model request failed','type':'upstream_error'}}
                    if is_context_window_error(response.status_code, payload):
                        call['context_window_error'] = {
                            'api_attempt_id': api_id, 'http_status': response.status_code,
                            **upstream_error_evidence(payload)}
                    raise UpstreamModelError(response.status_code,payload)
                try:
                    raw_response = response.json()
                except (ValueError, TypeError):
                    self.token_budget.record_unknown(api_id, role)
                    raise self._budget_stop('sample_token_usage_unknown')
                usage = raw_response.get("usage", {})
                budget_error = None
                budget_exceeded = False
                if self.token_budget.limit is not None:
                    try:
                        budget_row = self.token_budget.record(api_id, role, usage)
                        budget_exceeded = budget_row.get('exhausted', False)
                    except BudgetStopped as exc:
                        budget_error = self._budget_stop(exc.reason)
                    if budget_exceeded:
                        self.limit_reason = "sample_completion_token_budget"
                call.update(status="completed", response_id=raw_response.get("id"), response=raw_response,
                            usage=usage, usage_known=(isinstance(usage, dict)
                                and type(usage.get('completion_tokens')) is int
                                and usage['completion_tokens'] >= 0 and budget_error is None),
                            finished_at=now())
                if self.token_budget.limit is not None:
                    budget_summary = self.token_budget.snapshot()
                    call['sample_budget'] = {key: budget_summary.get(key) for key in
                                            ('schema_version', 'scope', 'accounting', 'binding', 'limit',
                                             'completion_tokens', 'by_role', 'exhausted', 'usage_unknown')}
                    call['sample_budget']['ledger_path'] = str(self.token_budget.path)
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
                if budget_error is not None:
                    raise budget_error
                if budget_exceeded:
                    raise self._budget_stop('sample_completion_token_budget')
                choices = raw_response.get('choices') or []
                if (trajectory_capped and choices
                        and (choices[0].get('finish_reason') == 'length'
                             or raw_response.get('dycon_native_finish_reason') == 'length')
                        and isinstance(usage, dict) and usage.get('completion_tokens') == allowance):
                    # A capped losing candidate must not stop a shorter winner.
                    # If no candidate survives, the normal policy propagates this.
                    raise OpenHandsLimit({'content': 'sample_completion_token_budget',
                        'extra': {'exit_status': 'SampleTokenBudgetExceeded',
                                  'limit_reason': 'sample_completion_token_budget'}})
                result = Response(raw_response)
                self._check()
                return result

            if role == "agent_decision" and active(self.cfg, self.decisions) and self.runtime.get("online_source") != "baseline":
                # Online bootstrap may share one managed vLLM service while
                # selecting the rollout profile per work item.  Baseline rows
                # use one ordinary trajectory; DyCon rows explicitly select
                # estimator-backed parallel consensus.  The immutable run
                # config remains untouched and only this physical attempt gets
                # the profile overlay.
                coordinator_cfg = self.cfg
                online_source = self.runtime.get("online_source")
                if online_source in {"baseline", "dycon"}:
                    coordinator_cfg = copy.deepcopy(self.cfg)
                    parallel = dict(coordinator_cfg.get("parallel") or {})
                    if online_source == "baseline":
                        parallel.update(enabled=False, branch_count=1, mode="parallel_baseline")
                    else:
                        parallel.update(enabled=True, mode="parallel_consensus")
                    coordinator_cfg["parallel"] = parallel
                coordinator = DecisionCoordinator(coordinator_cfg, self.runtime.get("capture_controller"),
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
            if (role == 'agent_decision' and self.token_budget.limit is not None
                    and self.token_budget.selected_trajectory):
                selected_calls = [call for call in record['model_calls']
                                  if call.get('response_id') == raw.get('id')]
                if len(selected_calls) != 1 or not raw.get('id'):
                    raise ValueError('Selected response has no unique physical API receipt')
                try:
                    budget = self.token_budget.select(record['request_id'], selected_calls[0]['api_attempt_id'])
                except BudgetStopped as exc:
                    raise self._budget_stop(exc.reason) from exc
                record['sample_budget'] = {key: budget.get(key) for key in
                    ('schema_version', 'scope', 'accounting', 'binding', 'limit',
                     'completion_tokens', 'by_role', 'exhausted', 'usage_unknown')}
                self._persist(record)
                if budget['exhausted']:
                    raise self._budget_stop('sample_completion_token_budget')
            if self.limit_reason in {'sample_completion_token_budget', 'sample_token_usage_unknown'}:
                raise self._budget_stop(self.limit_reason)
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
            self.context_window_error = None
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
                    self.limit_reason = self.limit_reason or payload.get('limit_reason') or (
                        'reasoning_loop' if payload.get('exit_status') == 'reasoning_loop' else 'generation_limit')
                    if self.limit_reason in {'sample_completion_token_budget', 'sample_token_usage_unknown'}:
                        self._budget_stop(self.limit_reason)
                if isinstance(exc, OpenHandsProtocolError):
                    self.step_loop.clear()
                calls = record.get('model_calls', [])
                # Classify the failed decision only when every candidate was
                # rejected for context; a recovered or losing branch is not terminal.
                if (calls and all(call.get('context_window_error') for call in calls)
                        and not self.closed.is_set()):
                    self.context_window_error = {
                        'schema_version': 1, 'binding': self.binding,
                        'exit_status': 'ContextWindowExceededError',
                        'cause_source': 'upstream_model_response',
                        'request_id': record['request_id'], 'role': record['role'],
                        'model_calls': [call['context_window_error'] for call in calls],
                        'observed_at': now()}
                    write_json(self.output / 'diagnostics/context-window-error.json',
                               self.context_window_error)
                elif not (isinstance(exc, OpenHandsLimit)
                          and self.limit_reason == 'sample_token_usage_unknown'):
                    self.context_window_error = None
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
                    evidence = upstream_error_evidence(exc.payload)
                    with state['lock']:
                        state['upstream_error_http_status'] = exc.status
                        state.update(evidence)
                    bridge._event('upstream_model_error', state=state,
                                  http_status=exc.status, **evidence)
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
