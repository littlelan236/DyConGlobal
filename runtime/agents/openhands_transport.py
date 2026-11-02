"""Pinned SDK transport observation, published into the existing SDK wheel.

Observe the existing LiteLLM/OpenAI/httpx path; never create another model client.
Only the bounded DyCon driver is affected. No prompt, URL, credential or exception
message is exported. Native SDK retries follow the driver's explicit zero policy.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

import httpx

_PHASES = {'connect', 'read', 'write', 'pool', 'sdk_consume', 'unknown'}
_ERRORS = {'LLMTimeoutError', 'Timeout', 'OpenAIError', 'APITimeoutError',
           'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
           'TimeoutError', 'APIConnectionError', 'ConnectError', 'ReadError',
           'WriteError', 'RemoteProtocolError', 'APIError', 'HTTPStatusError', 'FunctionCallValidationError'}
_EVENTS = {'sdk_call_started', 'sdk_http_finished', 'sdk_http_failed',
           'sdk_call_finished', 'sdk_call_failed'}
SDK_TRANSPORT_LOG = '/tmp/dycon-sdk-transport.jsonl'


def exception_evidence(exc):
    types, seen, phase = [], set(), 'unknown'
    while exc is not None and id(exc) not in seen and len(seen) < 16:
        seen.add(id(exc))
        name = type(exc).__name__
        if name in _ERRORS:
            types.append(type(exc).__module__ + '.' + name)
        if name in {'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout'}:
            phase = name.removesuffix('Timeout').lower()
        exc = exc.__cause__ or exc.__context__
    return {'exception_types': types, 'phase': phase}


def safe_event(value):
    """Bound and whitelist authenticated SDK telemetry before disk persistence."""
    if not isinstance(value, dict) or value.get('event') not in _EVENTS:
        raise ValueError('Invalid SDK event')
    sdk_id = value.get('sdk_request_id', '')
    if not isinstance(sdk_id, str) or re.fullmatch(r'sdk-[a-f0-9]{32}', sdk_id) is None:
        raise ValueError('Invalid SDK request identity')
    row = {'event': value['event'], 'sdk_request_id': sdk_id,
           'phase': value.get('phase') if value.get('phase') in _PHASES else 'unknown'}
    for k in ('sdk_timeout', 'sdk_retries', 'native_max_retries', 'http_attempt',
              'openai_retry_count', 'elapsed_seconds', 'http_status'):
        v = value.get(k)
        if type(v) in (int, float) and math.isfinite(v) and v >= 0:
            row[k] = v
    row['role'] = value.get('role') if value.get('role') in {'agent', 'condenser', 'critic'} else 'unknown'
    row['http_timeout'] = {k: v for k, v in (value.get('http_timeout') or {}).items()
        if k in {'connect', 'read', 'write', 'pool'} and
        (v is None or (type(v) in (int, float) and math.isfinite(v) and v >= 0))}
    row['exception_types'] = [v for v in value.get('exception_types', [])[:16]
        if isinstance(v, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]{0,128}', v)
        and v.rsplit('.', 1)[-1] in _ERRORS]
    row['versions'] = {k: v for k, v in (value.get('versions') or {}).items()
        if k in {'openhands-sdk', 'litellm', 'openai', 'httpx', 'httpcore'}
        and isinstance(v, str) and re.fullmatch(r'[0-9][0-9a-zA-Z.+-]{0,40}', v)}
    row['trace'] = [{k: e[k] for k in ('phase', 'state') if k in e}
        for e in value.get('trace', [])[:64] if isinstance(e, dict)
        and e.get('phase') in _PHASES and e.get('state') in {'started', 'complete', 'failed'}]
    return row


@lru_cache(maxsize=1)
def _versions():
    result = {}
    for name in ('openhands-sdk', 'litellm', 'openai', 'httpx', 'httpcore'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return result


def _emit(state, event, **fields):
    row = safe_event({**state['identity'], 'event': event, **fields})
    data = json.dumps(row, allow_nan=False).encode()
    # This survives a bridge-connect failure; OwnedPlanes collects it BEFORE cleanup.
    path = os.environ.get('DYCON_OPENHANDS_TRANSPORT_LOG', SDK_TRANSPORT_LOG)
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            if os.fstat(fd).st_size + len(data) < 8 * 1024 * 1024:
                os.write(fd, data + b'\n')
            else:
                print('DYCON_SDK_LOCAL_LOG_LIMIT', file=sys.stderr)
        finally:
            os.close(fd)
    except OSError as exc:
        print('DYCON_SDK_LOCAL_LOG_UNAVAILABLE ' + type(exc).__name__, file=sys.stderr)
    request = Request(state['endpoint'], data=data, method='POST',
                      headers={'Authorization': 'Bearer ' + state['secret'],
                               'Content-Type': 'application/json'})
    try:
        # Diagnostic delivery cannot inherit a public proxy or recursively use httpx.
        with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
            response.read(64)
    except Exception as exc:
        # Export absence explicitly without exposing the exception message or endpoint.
        print('DYCON_SDK_DIAGNOSTIC_UNAVAILABLE ' + type(exc).__name__, file=sys.stderr)


def _phase(name):
    if 'connect_tcp' in name or 'start_tls' in name: return 'connect'
    if 'send_request' in name: return 'write'
    if 'receive_response' in name: return 'read'
    if 'wait_for_connection' in name: return 'pool'
    return 'unknown'


# One transparent, context-local observer for all imported copies of this module.
_active = getattr(httpx.Client, '_dycon_transport_context', None)
if _active is None:
    _active = ContextVar('dycon_sdk_transport', default=None)
    httpx.Client._dycon_transport_context = _active
    _send = httpx.Client.send

    def _observed_send(client, request, **kwargs):
        state = _active.get()
        if (state is None or (request.url.host, request.url.port) != state['address']
                or not request.url.path.endswith('/chat/completions')):
            return _send(client, request, **kwargs)
        state['http_attempts'] += 1
        request.headers['X-DyCon-SDK-Request-ID'] = state['identity']['sdk_request_id']
        previous = request.extensions.get('trace')
        trace = []
        def observed_trace(name, info):
            phase, terminal = _phase(name), name.rsplit('.', 1)[-1]
            if len(trace) < 64 and terminal in {'started', 'complete', 'failed'}:
                trace.append({'phase': phase, 'state': terminal})
            if previous is not None: previous(name, info)
        request.extensions['trace'] = observed_trace
        retry = request.headers.get('x-stainless-retry-count', '0')
        facts = {'http_attempt': state['http_attempts'],
                 'http_timeout': request.extensions.get('timeout') or {},
                 'openai_retry_count': int(retry) if retry.isdigit() else None}
        try:
            response = _send(client, request, **kwargs)
        except Exception as exc:
            state['last_error'] = {**facts, **exception_evidence(exc), 'trace': trace}
            _emit(state, 'sdk_http_failed', **state['last_error'])
            raise
        _emit(state, 'sdk_http_finished', **facts, http_status=response.status_code, trace=trace)
        if response.status_code == 400:
            try:
                error = response.json().get('error', {})
            except (ValueError, AttributeError):
                error = {}
            if (isinstance(error, dict) and error.get('type') == 'FormatError'
                    and error.get('code') == 'format_error' and error.get('retryable') is True
                    and re.fullmatch(r'req-[a-f0-9]{32}', str(error.get('correlation_id', '')))):
                state['format_error'] = True

        return response

    httpx.Client.send = _observed_send


def completion(function, llm, **kwargs):
    """Keep the native call and frozen budget; close its nested retry leak."""
    if os.environ.get('DYCON_OPENHANDS_MODEL_SECONDS') is None:
        return function(**kwargs)
    # num_retries controls OpenHands/Tenacity only. LiteLLM separately defaults
    # OpenAI.max_retries to two, which replayed calls despite the driver's zero.
    kwargs['max_retries'] = 0
    parsed = urlsplit(kwargs.get('api_base') or '')
    if not parsed.hostname or not kwargs.get('api_key'):
        return function(**kwargs)
    metadata = (getattr(llm, 'litellm_extra_body', None) or {}).get('metadata', {})
    identity = {'sdk_request_id': 'sdk-' + uuid4().hex,
                'sdk_timeout': getattr(llm, 'timeout', None),
                'sdk_retries': getattr(llm, 'num_retries', None), 'native_max_retries': 0,
                'role': metadata.get('llm_type', getattr(llm, 'usage_id', 'unknown')),
                'versions': _versions()}
    state = {'identity': identity, 'address': (parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)),
             'endpoint': kwargs['api_base'].rstrip('/') + '/dycon/sdk-events',
             'secret': kwargs['api_key'], 'http_attempts': 0, 'last_error': {}}
    token = _active.set(state)
    started = time.monotonic()
    try:
        _emit(state, 'sdk_call_started')
        result = function(**kwargs)
    except Exception as exc:
        evidence = exception_evidence(exc)
        if evidence['phase'] == 'unknown' and state['last_error']:
            evidence['phase'] = state['last_error']['phase']
        _emit(state, 'sdk_call_failed', **evidence, elapsed_seconds=time.monotonic()-started)
        if state.get('format_error'):
            # The pinned Agent.step appends this native validation error as a
            # correction message and continues. This is not an SDK retry.
            from openhands.sdk.llm.exceptions import FunctionCallValidationError
            tools = [tool.get('function', {}).get('name') for tool in kwargs.get('tools', [])]
            names = ', '.join(name for name in tools if isinstance(name, str))
            raise FunctionCallValidationError(
                'FormatError: Use a declared top-level tool with valid JSON arguments. '
                'Do not call tool subcommands as tool names. Available tools: ' + names) from exc
        raise
    else:
        _emit(state, 'sdk_call_finished', phase='sdk_consume', elapsed_seconds=time.monotonic()-started)
        return result
    finally:
        _active.reset(token)
