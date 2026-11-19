"""Client-side capture drain shared by rollout backends."""

import time
from pathlib import Path

from core.io import now, write_json
from runtime.serving.diagnostics import safe_response_summary, response_error_type, sanitize_error_text


def drain_capture(cfg, output, capture_client, session_id, diagnostic_errors=None, *, serving_endpoint=None):
    """One producer barrier and native/hidden drain for every terminal path."""
    import requests
    from uuid import uuid4

    endpoint = (serving_endpoint or cfg['serving']['base_url']).rstrip('/').removesuffix('/v1')
    started_at, started = now(), time.monotonic()
    receipt = {'schema_version': 1, 'session_id': session_id, 'started_at': started_at,
               'phase': 'native_flush'}
    response = None
    correlation_id = uuid4().hex
    try:
        flush_started = time.monotonic()
        response = requests.post(endpoint + '/dycon/capture/engine/flush', json={'session_id': session_id},
            timeout=cfg['capture'].get('flush_timeout_seconds') or cfg['capture']['finalize_seconds'],
            headers={'X-DyCon-Correlation-ID': correlation_id})
        response.raise_for_status()
        value = response.json()
        try:
            if value.get('stage_diagnostics'):
                capture_client.report_diagnostics(session_id, value['stage_diagnostics'])
        except Exception as diagnostic_exc:
            if diagnostic_errors is not None and getattr(diagnostic_exc, 'capture_diagnostic', None):
                diagnostic_errors.append(diagnostic_exc.capture_diagnostic)
        receipt['native_flush_seconds'] = time.monotonic() - flush_started
        if value.get('failures'):
            raise RuntimeError('Native capture drain failed: ' + str(value['failures']))
        receipt['native_flush'] = value
        receipt['phase'] = 'session_stop'
        stop_started = time.monotonic()
        # Stop can wait for native output and export queues sequentially. The
        # normal control-request timeout must not expire before these bounded
        # drains can finish and persist their diagnostic. No producers remain.
        stop_timeout = max(cfg['capture'].get('control_seconds', 10),
                           cfg['capture'].get('finalize_request_seconds') or 0,
                           2 * cfg['capture']['finalize_seconds'] + 30)
        receipt['session_stop_timeout_seconds'] = stop_timeout
        previous_timeout = getattr(capture_client, 'timeout', None)
        try:
            capture_client.timeout = stop_timeout
            stopped = capture_client.stop(session_id)
        finally:
            capture_client.timeout = previous_timeout
        receipt.update(session_stop=stopped, session_stop_seconds=time.monotonic() - stop_started,
            order=['bridge_closed', 'inflight_completed_or_cancelled', 'native_and_hidden_drained', 'session_stopped'],
            elapsed_seconds=time.monotonic() - started, phase='complete', finished_at=now())
        write_json(Path(output) / 'capture-drain.json', receipt)
    except Exception as exc:
        diagnostic = getattr(exc, 'capture_diagnostic', None)
        if diagnostic is None:
            summary = safe_response_summary(response.content[:512], 512) if response is not None else ''
            diagnostic = {
                'schema_version': 1, 'stage': receipt['phase'], 'error_type': type(exc).__name__,
                'method': 'POST', 'path': '/dycon/capture/engine/flush',
                'status_code': getattr(response, 'status_code', None),
                'correlation_id': (response.headers.get('X-DyCon-Correlation-ID', correlation_id)
                                   if response is not None else correlation_id),
                'duration_seconds': time.monotonic() - started,
                'server_error_type': response_error_type(summary), 'response_summary': summary,
            }
        if diagnostic_errors is not None:
            diagnostic_errors.append(diagnostic)
        receipt['capture_diagnostic'] = diagnostic
        receipt.update(failed=True, error_type=type(exc).__name__, error=sanitize_error_text(exc),
            elapsed_seconds=time.monotonic() - started, finished_at=now())
        write_json(Path(output) / 'capture-drain.json', receipt)
        raise
