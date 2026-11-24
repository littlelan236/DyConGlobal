"""Manifest-driven, fair scheduling of durable physical rollout attempts."""
from __future__ import annotations

import concurrent.futures
import copy
import os
import re
import time
import traceback
from collections import deque
from pathlib import Path

from application.execution.evidence import RunEvidence
from application.execution.run_plan import create_run, load_plan
from application.execution.status import update_run_status
from core.cancellation import CURRENT, Cancelled, CompletionWake, cancellable_run, policy, request_for
from core.io import identifier, now, read_json, sha256, write_json
from core.locking import exclusive_run, file_lock
from rollout.artifacts import import_rollout
from rollout.progress import Progress
from rollout.retry import INFRASTRUCTURE as INFRASTRUCTURE
from rollout.retry import (
    ONLINE_LIMIT_RESAMPLE_RETRIES,
    RETRY_STATUSES,
    SEED_POLICY,
    TERMINAL,
    counts_recorded,
    disposition,
    enabled,
    resampling_enabled,
    initialize_counts,
    online_sampling_discard,
    publish_once,
    rebuild_counts,
    record_agent_outcome,
    sample_accepted,
    sample_settled,
    sample_work,
    verify_final,
)
from rollout.sampling import AttemptContext, SampleContext, effective_seed, sample_from_item


def _boundary(run, factory):
    if request_for(run, os.environ.get('DYCON_EXECUTION_ATTEMPT')) is not None:
        return 'Cancelled'
    try:
        factory.deadline.check()
    except TimeoutError:
        return 'DeadlineExceeded'
    return None


def _phase(runtime, sample, phase):
    tracker = runtime.get('progress_tracker')
    if tracker:
        try:
            tracker.phase(sample.sample_key, phase)
        except Exception as exc:
            runtime.setdefault('progress_errors', []).append(repr(exc))


def _recover_capture_status(cfg, run, session_id, rid, digest):
    path = run / 'hidden/publication.json'
    if path.exists():
        state = read_json(path).get('sessions', {}).get(session_id)
        if state == 'complete':
            options = cfg['capture']['export']
            candidates = [Path(cfg['project_root']) / options['spool_root'] / run.name,
                          Path(cfg['paths']['nas_root']) / options['root'] / run.name]
            proofs = [base / 'sessions' / session_id / 'finalization.json' for base in candidates]
            proof = next((p for p in proofs if p.exists()), None)
            if proof is None:
                raise ValueError('Recovered capture publication lacks its finalization identity')
            binding = read_json(proof)
            if binding.get('rollout_id') != rid or binding.get('trajectory_sha256') != digest:
                raise ValueError('Recovered capture publication binds another trajectory')
        return state
    return None


def _finish_attempt(cfg, run_id, task, sample, context, out, runtime, state,
                    capture_client=None, capture_session_id=None, recovering=False):
    """The sole canonical import/finalize path, also used after a crash."""
    from runtime.backends.registry import adapter_for
    adapter = adapter_for(cfg, {"attempt": task})
    root = Path(cfg['project_root'])
    run = root / 'runs' / run_id
    path = out / 'raw.traj.json'
    cancellation = runtime.get('cancellation')
    if cancellation is not None and cancellation.event.is_set():
        result = {**sample.as_dict(), **context, 'instance_id': sample.instance_id,
                  'exit_status': 'Cancelled', 'agent_exit_status': state.get('agent_exit_status'),
                  'disposition': 'interrupted', 'capture_status': 'partial' if capture_session_id else 'disabled',
                  'capture_session_id': capture_session_id, 'usage': None,
                  'started_at': state['started_at'], 'finished_at': now(), 'timings': state.get('timings', {})}
        write_json(out / 'attempt.json', result)
        return result
    _phase(runtime, sample, 'finalizing')
    stamp = time.monotonic()
    evidence = RunEvidence(root, run_id).read()
    candidates = [(rid, row) for rid, row in evidence['rollouts'].items()
                  if row.get('sample_key') == sample.sample_key and row.get('attempt_id') == context['attempt_id']]
    if len(candidates) > 1:
        raise ValueError('Multiple canonical trajectories for one physical attempt')
    if candidates:
        rid, row = candidates[0]
        canonical = root / row['trajectory_path']
        from application.execution.subject import artifact_kinds
        for kind in artifact_kinds(row):
            if sha256(root / row[kind + '_path']) != row[kind + '_sha256']:
                raise ValueError('Recovered canonical bytes changed')
        adapter.verify_subject(root, row)
        if path.exists() and sha256(path) != row['trajectory_sha256']:
            raise ValueError('Recovered raw and canonical bytes disagree')
    else:
        subject_spec = adapter.subject_spec(task, out, context, state)
        rid = import_rollout(root, {
            'source': str(path), 'task': task, 'run_id': run_id,
            'attempt_id': context['attempt_id'], 'purposes': ['baseline'],
            'model_fingerprint': cfg['model']['fingerprint'], 'sample': sample.as_dict(),
            'attempt_context': context, 'run_local': True, **subject_spec,
        })
        row = RunEvidence(root, run_id).read()['rollouts'][rid]
        canonical = root / row['trajectory_path']
    raw = read_json(canonical)
    usage = {'cost': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0}
    seen_decisions = set()
    for message in raw.get('messages', []):
        extra = message.get('extra') or {}
        parallel = extra.get('dycon_parallel') or {}
        decision_id = parallel.get('decision_id') or extra.get('decision_id')
        if decision_id and decision_id in seen_decisions:
            continue
        if decision_id:
            seen_decisions.add(decision_id)
        usage['cost'] += extra.get('cost', 0.0) or 0.0
        tokens = extra.get('total_usage') or parallel.get('total_usage') or (extra.get('response') or {}).get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens'):
            usage[key] += tokens.get(key, 0) or 0
    state['rollout_id'] = rid
    write_json(out / 'execution-state.json', state)
    errors = list(state.get('capture_errors', []))
    capture_status = state.get('capture_status', 'disabled')
    attempt_capture_diagnostics = {
        'schema_version': 1, 'run_id': run_id,
        'execution_attempt_id': os.environ.get('DYCON_EXECUTION_ATTEMPT'),
        'sample_key': sample.sample_key, 'attempt_id': context.get('attempt_id'),
        'session_id': capture_session_id, 'stage': 'finalize',
        'errors': list(state.get('capture_diagnostic_errors', []))[:32],
    }
    if capture_session_id is not None:
        try:
            recovered_status = _recover_capture_status(cfg, run, capture_session_id, rid, sha256(canonical)) if recovering else None
            if recovered_status == 'complete':
                capture_status = 'complete'
                errors = []
            elif capture_client is not None:
                capture_client.timeout = cfg['capture'].get('finalize_request_seconds') or (cfg['capture']['export']['upload_max_elapsed_seconds'] + cfg['capture']['finalize_seconds'])
                finalized = capture_client.finalize(
                    capture_session_id, rollout_id=rid, trajectory_path=str(canonical),
                    trajectory_sha256=sha256(canonical), model_fingerprint=cfg['model']['fingerprint'])
                capture_status = finalized['capture_status']
                server_diagnostics = finalized.get('capture_diagnostics') or {}
                server_diagnostics = server_diagnostics.get('stages') or {}
                for key in ('schema_version', 'stage', 'run_id', 'execution_attempt_id',
                            'sample_key', 'attempt_id', 'session_id', 'stages', 'by_anchor',
                            'by_request', 'missing', 'missing_count', 'missing_by_anchor',
                            'truncated_count', 'position_truncated_count', 'queue'):
                    if key in server_diagnostics:
                        attempt_capture_diagnostics[key] = server_diagnostics[key]
                if capture_status not in {'complete', 'not_exported'}:
                    raise RuntimeError('Capture is ' + capture_status)
            else:
                raise RuntimeError('Capture finalization requires recovery of session ' + capture_session_id + '; use dycon capture recover')
        except Exception as exc:
            capture_status = 'partial'
            diagnostic = getattr(exc, 'capture_diagnostic', None)
            if diagnostic:
                attempt_capture_diagnostics.setdefault('errors', []).append(diagnostic)
            errors.append(repr(exc))
            write_json(out / 'capture-errors.json', errors)
    # Delete raw bytes only after import. Unacknowledged capture spool is untouched.
    path.unlink(missing_ok=True)
    status = state['agent_exit_status']
    if errors and cfg['capture']['failure_policy'] == 'block_rollout':
        status = 'CaptureError'
    if state.get('cleanup_error'):
        status = 'CleanupError'
    if state.get('stats_error'):
        status = 'StatisticsError'
    if adapter.subject_failed(root, row, state):
        status = 'WebTraceError'
    timings = state.setdefault('timings', {})
    timings['capture_finalize_seconds'] = time.monotonic() - stamp
    result = {
        'instance_id': sample.instance_id, **sample.as_dict(), **context,
        'rollout_id': rid, 'exit_status': status, 'agent_exit_status': state['agent_exit_status'],
        'started_at': state['started_at'], 'finished_at': now(), 'error': state.get('error'),
        'capture_status': capture_status, 'capture_errors': errors,
        'timings': timings, 'capture_session_id': capture_session_id, 'usage': usage,
        'progress_errors': runtime.get('progress_errors', []),
    }
    if 'capture_stop_retry' in state:
        result['capture_stop_retry'] = state['capture_stop_retry']
    dependency_failure = (state.get('error') or {}).get('dependency_failure')
    if dependency_failure:
        result['dependency_failure'] = dependency_failure
        result['failure_type'] = 'dependency_error'
    result['disposition'] = 'blocked' if status in {'CaptureError', 'StatisticsError', 'CleanupError', 'WebTraceError'} else disposition(cfg, status)
    if capture_session_id is not None or attempt_capture_diagnostics.get('errors'):
        try:
            # Derived diagnostics are best-effort and cannot affect rollout status.
            write_json(out / 'capture-diagnostics.json', attempt_capture_diagnostics)
        except Exception:
            pass
    write_json(out / 'attempt.json', result)
    return result


def _execute_attempt(cfg, run_id, task, factory, runtime, sample, context, out):
    """One clean environment, agent, model and capture session per attempt."""
    from runtime.backends.contracts import AttemptBinding
    from runtime.backends.registry import adapter_for
    adapter = adapter_for(cfg, {"attempt": task})
    run_context = adapter.open_run(cfg, run_id, factory=factory)
    runtime = {**runtime, 'sample_key': sample.sample_key}
    tracker = runtime.get('progress_tracker')
    if tracker is not None:
        try:
            tracker.started(context, runtime.get('retry_kind', 'initial'))
        except Exception as exc:
            runtime.setdefault('progress_errors', []).append(repr(exc))
    _phase(runtime, sample, 'bootstrap')
    return adapter.execute_attempt(run_context, AttemptBinding(task, sample, context, out, runtime))


def _execute_mini_attempt(cfg, run_id, task, factory, runtime, sample, context, out, adapter):
    """The single mini loop, parameterized by explicit benchmark hooks."""
    from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
    from rollout.model import make_model
    from runtime.containers.factory import TaskContext
    RolloutEnvironment = adapter.environment_class()
    DurableAgent = adapter.agent_class()

    run = Path(cfg['project_root']) / 'runs' / run_id
    path = out / 'raw.traj.json'
    attempt_id, iid = context['attempt_id'], sample.instance_id
    local_cfg = copy.deepcopy(cfg)
    local_cfg['generation']['seed'] = context['effective_seed']
    runtime = {**runtime, 'sample_key': sample.sample_key}
    task_context = TaskContext(run_id, iid, attempt_id, 'rollout', out)
    env = agent = capture_client = capture_session_id = None
    capture_errors = []
    capture_status = 'disabled'
    status, patch, error = 'Interrupted', '', None
    state = {'started_at': now(), 'timings': {
        'queue_wait_seconds': runtime.get('queue_wait_seconds', 0.0),
        'admission_wait_seconds': runtime.get('admission_wait_seconds', 0.0),
        'container_bootstrap_seconds': 0.0, 'agent_seconds': 0.0,
        'capture_finalize_seconds': 0.0, 'stats_update_seconds': 0.0}}
    publish_once(out / 'attempt-started.json', {'attempt_id': attempt_id, 'started_at': state['started_at'], 'retry_kind': runtime.get('retry_kind', 'initial'),
                  'execution_attempt_id': os.environ.get('DYCON_EXECUTION_ATTEMPT')})
    write_json(out / 'execution-state.json', state)
    _phase(runtime, sample, 'bootstrap')
    stage_start = time.monotonic()

    def capture_failure(exc, abort=True):
        diagnostic = getattr(exc, 'capture_diagnostic', None)
        if diagnostic:
            state.setdefault('capture_diagnostic_errors', []).append(diagnostic)
        capture_errors.append(repr(exc))
        write_json(out / 'capture-errors.json', capture_errors)
        if abort and cfg['capture']['failure_policy'] == 'block_rollout':
            from runtime.serving.errors import CaptureError
            raise CaptureError(str(exc)) from exc

    try:
        env = RolloutEnvironment(factory, task['instance'], task_context)
        state['timings']['container_bootstrap_seconds'] = time.monotonic() - stage_start
        capture_context = None
        if cfg.get('capture', {}).get('enabled'):
            capture_context = False
            capture_status = 'collecting'
            try:
                inventory = runtime['capture_token_inventory']
                capture_context = capture_request_metadata(
                    'cap-' + sample.sample_key + '-' + attempt_id, iid, run_id,
                    f'pending-{iid}-{attempt_id}', 0, cfg['capture']['anchors'],
                    newline_token_ids=inventory['newline_token_ids'], close_token_ids=inventory['close_token_ids'],
                    tokenizer_fingerprint=inventory['tokenizer_fingerprint'],
                    newline_policy=cfg['capture']['newline_policy'], capture_select=inventory['capture_select'])
                capture_client = CaptureClient(runtime['capture_control_url'], timeout=cfg['capture']['control_seconds'])
                capture_session_id = capture_context['session_id']
                write_json(out / 'capture-session.json', {'session_id': capture_session_id, 'metadata': capture_context})
                capture_client.start(capture_context, graph_mode='FULL', prefix_cache_policy='safe_boundary',
                    diagnostic_identity={'run_id': run_id, 'execution_attempt_id': os.environ.get('DYCON_EXECUTION_ATTEMPT'),
                                         'sample_key': sample.sample_key, 'attempt_id': attempt_id})
            except Exception as exc:
                capture_context = False
                capture_status = 'failed'
                capture_failure(exc)
        agent_cfg = adapter.agent_config(cfg)
        local_cfg['agent'] = agent_cfg
        agent = DurableAgent(
            make_model(local_cfg, factory.deadline, capture_context, runtime), env,
            system_template=agent_cfg['system_template'], instance_template=agent_cfg['instance_template'],
            step_limit=cfg['agent']['step_limit'], cost_limit=cfg['agent']['cost_limit'],
            max_consecutive_format_errors=cfg['agent']['max_consecutive_format_errors'],
            loop_repeat_limit=cfg['agent'].get('loop_repeat_limit'), loop_max_period=cfg['agent'].get('loop_max_period', 7),
            output_path=path)
        _phase(runtime, sample, 'agent')
        stage_start = time.monotonic()
        info = agent.run(adapter.prompt(task))
        status, patch = info.get('exit_status', 'Unknown'), info.get('submission', '')
    except Exception as exc:
        if agent is not None and getattr(exc, 'response_message', None) is not None:
            agent.messages.append(exc.response_message)
        status = type(exc).__name__
        error = {'error': str(exc), 'traceback': traceback.format_exc()}
        from runtime.containers.dependencies import from_exception
        failure = from_exception(exc)
        if failure:
            error['dependency_failure'] = failure
    finally:
        state['timings']['agent_seconds'] = time.monotonic() - stage_start if agent is not None else 0.0
        state.update(agent_exit_status=status, error=error, capture_status=capture_status, capture_errors=capture_errors)
        try:
            extra = {'instance_id': iid, 'info': {
                'exit_status': status, 'submission': patch, 'attempt_id': attempt_id,
                'dycon': {'run_id': run_id, 'config_hash': cfg['config_hash'], 'attempt_context': context}}}
            if agent is not None:
                agent.save(path, extra)
            elif not path.exists():
                write_json(path, {**extra, 'messages': []})
            try:
                subject_path, subject_errors = adapter.finalize_subject(task, out, context, env, patch, status)
                if subject_errors:
                    state['web_subject_errors'] = subject_errors
                if subject_path is not None:
                    state['subject_path'] = str(subject_path)
            except Exception as exc:
                state['web_subject_error'] = repr(exc)
            stamp = time.monotonic()
            if counts_recorded(run):
                try:
                    record_agent_outcome(run, out, context, status)
                except Exception as exc:
                    # The immutable agent fact survives. Cleanup must still run.
                    state['stats_error'] = repr(exc)
                    write_json(out / 'statistics-error.json', {'error': repr(exc)})
            state['timings']['stats_update_seconds'] = time.monotonic() - stamp
            write_json(out / 'execution-state.json', state)
        finally:
            _phase(runtime, sample, 'finalizing')
            stamp = time.monotonic()
            try:
                if capture_client is not None and capture_session_id is not None and not (runtime.get('cancellation') and runtime['cancellation'].event.is_set()):
                    try:
                        import requests
                        endpoint = cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
                        from hidden.backends.vllm.online import CaptureClient
                        flush_result = CaptureClient(endpoint, timeout=cfg['capture'].get('flush_timeout_seconds') or cfg['capture']['finalize_seconds'])._request(
                            'POST', '/dycon/capture/engine/flush', {'session_id': capture_session_id})
                        try:
                            plugin_diagnostics = flush_result.get('stage_diagnostics')
                            if plugin_diagnostics:
                                capture_client.report_diagnostics(capture_session_id, plugin_diagnostics)
                        except Exception as diagnostic_exc:
                            diagnostic = getattr(diagnostic_exc, 'capture_diagnostic', None)
                            if diagnostic:
                                state.setdefault('capture_diagnostic_errors', []).append(diagnostic)
                        if flush_result.get('failures'):
                            raise RuntimeError('Capture drain failed: ' + str(flush_result['failures']))
                    except Exception as exc:
                        capture_failure(exc, abort=False)
                        capture_status = 'partial'
                    capture_cfg = cfg['capture']
                    capture_client.timeout = max(capture_cfg.get('finalize_request_seconds') or 0,
                        2 * capture_cfg['finalize_seconds'] + capture_cfg['control_seconds'])
                    online = cfg['execution'].get('online_rl_failure_policy') is not None
                    retry_limit = 1 if online else 0
                    stop_timeouts = []
                    for attempt in range(retry_limit + 1):
                        try:
                            capture_client.stop(capture_session_id)
                            break
                        except Exception as exc:
                            diagnostic = getattr(exc, 'capture_diagnostic', None) or {}
                            drain_timeout = diagnostic.get('server_error_type') == 'ExportQueueDrainTimeout'
                            if drain_timeout:
                                stop_timeouts.append(diagnostic.get('correlation_id'))
                                if attempt < retry_limit:
                                    continue
                            if online and len(stop_timeouts) == retry_limit + 1:
                                state['capture_stop_retry'] = {
                                    'reason': 'capture_stop_retries_exhausted', 'retry_limit': retry_limit,
                                    'attempt_count': len(stop_timeouts), 'correlation_ids': stop_timeouts}
                            capture_failure(exc, abort=False)
                            capture_status = 'partial'
                            break
                    try:
                        capture_client.prepare(capture_session_id, path)
                        write_json(out / 'capture-prepared.json', {'session_id': capture_session_id, 'trajectory_sha256': sha256(path)})
                    except Exception as exc:
                        capture_failure(exc, abort=False)
                        capture_status = 'partial'
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        state['cleanup_error'] = repr(exc)
                state.update(capture_status=capture_status, capture_errors=capture_errors)
                state['timings']['capture_drain_seconds'] = time.monotonic() - stamp
                write_json(out / 'execution-state.json', state)
    return _finish_attempt(cfg, run_id, task, sample, context, out, runtime, state,
                           capture_client, capture_session_id)


def _recover_attempt(cfg, run_id, task, sample, context, out, runtime):
    """Recover confirmed saved agent output without another model call."""
    from runtime.backends.registry import adapter_for
    adapter = adapter_for(cfg, {"attempt": task})
    if adapter.binding.backend_family == "harbor":
        return adapter.recover_attempt(cfg, run_id, task, sample, context, out, runtime)
    run = Path(cfg['project_root']) / 'runs' / run_id
    path = out / 'raw.traj.json'
    if not path.exists():
        matches = [row for row in RunEvidence(cfg['project_root'], run_id).read()['rollouts'].values()
                   if row.get('sample_key') == sample.sample_key and row.get('attempt_id') == context['attempt_id']]
        if len(matches) > 1:
            raise ValueError('Conflicting recovered canonical attempt')
        if matches:
            row = matches[0]
            path = Path(cfg['project_root']) / row['trajectory_path']
            if sha256(path) != row['trajectory_sha256']:
                raise ValueError('Recovered canonical SHA changed')
    fact_path = out / 'agent-outcome.json'
    if not path.exists():
        if fact_path.exists():
            raise ValueError('Confirmed agent outcome has no recoverable trajectory')
        return None
    raw = read_json(path)
    info = raw.get('info', {})
    binding = info.get('dycon', {}).get('attempt_context')
    status = info.get('exit_status')
    recoverable = status in TERMINAL or disposition(cfg, status) == 'selection_retry'
    if not fact_path.exists() and (not recoverable or binding is None):
        return None
    if binding != context or info.get('attempt_id') != context['attempt_id'] or raw.get('instance_id') != sample.instance_id:
        raise ValueError('Saved trajectory does not match reserved attempt context')
    if not recoverable and not fact_path.exists():
        return None
    if counts_recorded(run):
        record_agent_outcome(run, out, context, status)
    state_path = out / 'execution-state.json'
    state = read_json(state_path) if state_path.exists() else {'started_at': now(), 'timings': {}}
    state.update(agent_exit_status=status, stats_error=None)
    session_path = out / 'capture-session.json'
    sid = read_json(session_path)['session_id'] if session_path.exists() else None
    if sid and path == out / 'raw.traj.json' and not (out / 'capture-prepared.json').exists():
        imported = any(row.get('sample_key') == sample.sample_key and row.get('attempt_id') == context['attempt_id']
                       for row in RunEvidence(cfg['project_root'], run_id).read()['rollouts'].values())
        if not imported:
            # Native receipt joining can change raw bytes. Never freeze an unprepared
            # capture trajectory merely to advance its sample to a fresh seed.
            return {**sample.as_dict(), **context, 'exit_status': 'CaptureError',
                    'agent_exit_status': status, 'disposition': 'blocked',
                    'capture_status': 'pending_recovery', 'capture_session_id': sid,
                    'capture_errors': ['Agent output is saved, but capture prepare did not finish; raw bytes retained for recovery'],
                    'started_at': state['started_at'], 'timings': state.get('timings', {})}
    result = _finish_attempt(cfg, run_id, task, sample, context, out, runtime, state,
                            capture_session_id=sid, recovering=True)
    result['recovered_without_generation'] = True
    write_json(out / 'attempt.json', result)
    return result


def _retry_cursor(cfg, history):
    round_id = 0
    retry_kind = 'initial'
    for record in history:
        round_id = record.get('seed_round', round_id)
        kind = record.get('disposition') or disposition(cfg, record.get('exit_status'))
        if kind in {'retry_pending', 'selection_retry'}:
            round_id += 1
            retry_kind = 'resample'
        elif kind == 'infrastructure_retry':
            retry_kind = 'infrastructure'
        elif kind == 'operator_retry':
            retry_kind = 'operator'
    return round_id, retry_kind


def _decorate_result(cfg, sample, work, out, result, history):
    result = {**sample.as_dict(), **result}
    if cfg.get("_online_rl_source") == "baseline":
        result["online_source"] = "baseline"
    assignment = out / 'serving-assignment.json'
    if assignment.exists():
        result['serving_assignment'] = read_json(assignment)
    result.setdefault('seed_round', 0)
    result.setdefault('effective_seed', sample.sample_seed)
    result.setdefault('disposition', disposition(cfg, result.get('exit_status')))
    if (cfg.get('_online_rl_source') in {'baseline', 'dycon'}
            and (result.get('exit_status') in {'CaptureError', 'CleanupError', 'OpenHandsError', 'TimeoutError'}
                 or (result.get('failure_type') == 'dependency_error'
                     and result['disposition'] == 'failed'))):
        result['disposition'] = 'infrastructure_retry'
    records = [*history, result]
    started = [p for p in (work / 'attempts').iterdir() if (p / 'attempt-started.json').exists()]
    contexts = [read_json(p / 'attempt-context.json') for p in started if (p / 'attempt-context.json').exists()]
    legacy = [p for p in (work / 'attempts').iterdir() if p.is_dir() and not (p / 'attempt-context.json').exists()]
    result.update(
        physical_attempts=len(started) + len(legacy),
        resample_retry_count=len({c['seed_round'] for c in contexts if c['seed_round'] > 0}),
        infrastructure_retry_count=max(0, len(legacy) - 1) + sum(read_json(p / 'attempt-started.json').get('retry_kind') == 'infrastructure' for p in started))
    selection_failures = sum(r.get('exit_status') == 'ParallelSelectionError' for r in records)
    if cfg.get('execution', {}).get('selection_retry', {}).get('enabled', False):
        result['selection_retry_count'] = selection_failures
    if result.get('disposition') == 'selection_retry':
        result['selection_retry_count'] = selection_failures - 1
        if selection_failures > cfg['execution']['selection_retry']['max_retries']:
            result.update(exit_status='selection_retry_exhausted', last_exit_status='ParallelSelectionError', disposition='failed')
    if cfg.get('_online_rl_source') == 'dycon' and result['disposition'] == 'retry_pending':
        if sum(r.get('exit_status') in RETRY_STATUSES for r in records) > ONLINE_LIMIT_RESAMPLE_RETRIES:
            result.update(exit_status='limits_retry_exhausted', last_exit_status=result['exit_status'],
                          disposition='failed')
    if result.get('disposition') == 'infrastructure_retry' and result['infrastructure_retry_count'] >= cfg['execution']['infrastructure_retry_limit']:
        result.update(exit_status='infrastructure_exhausted', last_exit_status=result['exit_status'], disposition='failed')
    write_json(out / 'attempt.json', result)
    next_round, retry_kind = _retry_cursor(cfg, records[:-1] + [result])
    write_json(work / 'retry-state.json', {
        'sample_key': sample.sample_key, 'seed_round': next_round, 'retry_kind': retry_kind,
        'last_attempt_id': result['attempt_id'], 'last_status': result['exit_status'],
        'resample_retry_count': result['resample_retry_count'],
        'selection_retry_count': result.get('selection_retry_count', 0),
        'infrastructure_retry_count': result['infrastructure_retry_count']})
    if result['disposition'] not in {'retry_pending', 'selection_retry', 'infrastructure_retry'}:
        write_json(work / 'result.json', result)
    else:
        (work / 'result.json').unlink(missing_ok=True)
    return result


def run_task(cfg, run_id, task_key, task, factory, runtime=None, sample=None):
    """Run at most one physical attempt; retry scheduling belongs to run_manifest."""
    runtime = runtime or {}
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    iid = identifier(task['instance_id'])
    sample = sample or SampleContext(run_id, task_key, iid, 'rep-000', 0,
        'legacy-sample-' + task_key[:24], cfg['generation']['seed'])
    run = Path(cfg['project_root']) / 'runs' / run_id
    work = sample_work(run, sample)
    with file_lock(work / 'execution.lock', nonblocking=True):
        boundary = _boundary(run, factory)
        if boundary:
            return {**sample.as_dict(), 'exit_status': boundary, 'disposition': 'pending'}
        from application.execution.failed_retry import (
            retry_index, superseded, supersede_incomplete_attempt, worker_failure_authorization,
        )
        operator_retries = runtime.get('operator_retries')
        if operator_retries is None:
            operator_retries = retry_index(run)
        result_path = work / 'result.json'
        worker_authorization = worker_failure_authorization(operator_retries, sample.sample_key)
        operator_predecessor = None
        if result_path.exists():
            old = read_json(result_path)
            if old.get('sample_key') != sample.sample_key:
                raise ValueError('Completed sample identity changed')
            old_superseded = superseded(operator_retries, sample.sample_key, old)
            if sample_settled(cfg, old) and not old_superseded:
                if sample_accepted(cfg, old):
                    verify_final(cfg['project_root'], run_id, sample.as_dict(), old, RunEvidence(cfg['project_root'], run_id).read())
                return old
            if old_superseded and old.get('attempt_id') is None:
                operator_predecessor = old
        elif worker_authorization is not None:
            operator_predecessor = worker_authorization['outcome']
        manifest = runtime.get('work_items')
        if manifest is None:
            manifest = {'run_id': run_id, 'repeat_count': 1, 'items': [sample.as_dict()],
                        'seed_policy': {'branch_stride': max(1, cfg.get('parallel', {}).get('branch_count', 1))}}
        if 'limits_retry' in cfg['execution'] and not (run / 'plan.json').exists():
            initialize_counts(run, manifest)
        attempts = work / 'attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(re.escape(sample.attempt_prefix) + r'-(\d+)$')
        existing = sorted((int(m.group(1)), p) for p in attempts.iterdir() if (m := pattern.fullmatch(p.name)))
        history = ([{**operator_predecessor, 'disposition': 'operator_retry'}]
                   if operator_predecessor is not None else [])
        reserved = None
        for _, out in existing:
            context_path = out / 'attempt-context.json'
            context = read_json(context_path) if context_path.exists() else None
            if context:
                for key in ('run_id', 'task_key', 'sample_key', 'repetition_id', 'sample_seed'):
                    if context.get(key) != sample.as_dict()[key]:
                        raise ValueError('Reserved attempt identity changed')
                expected_policy = SEED_POLICY if resampling_enabled(cfg) else sample.seed_policy
                if context.get('seed_policy') != expected_policy:
                    raise ValueError('Reserved attempt seed policy changed')
                if context['attempt_id'] != out.name:
                    raise ValueError('Reserved attempt directory identity changed')
                actual = cfg.get('parallel', {}).get('branch_count', 1) if cfg.get('parallel', {}).get('enabled') else 1
                if context['effective_seed'] != effective_seed(sample, manifest, context['seed_round'], actual):
                    raise ValueError('Reserved attempt seed changed')
            outcome_path = out / 'attempt.json'
            result = read_json(outcome_path) if outcome_path.exists() else None
            if result is not None and superseded(operator_retries, sample.sample_key, result):
                history.append({**result, 'disposition': 'operator_retry'})
                continue
            if (result is None and context
                    and supersede_incomplete_attempt(run, sample.sample_key, worker_authorization, out, context)):
                history.append({**context, 'exit_status': 'WorkerFailure', 'disposition': 'operator_retry'})
                continue
            needs_recovery = result is None or result.get('disposition') == 'blocked'
            if context and needs_recovery:
                recovered = _recover_attempt(cfg, run_id, task, sample, context, out, runtime)
                if recovered:
                    return _decorate_result(cfg, sample, work, out, recovered, history)
                if not (out / 'attempt-started.json').exists():
                    reserved = (out, context)
                    break
            if result is None:
                started = out / 'attempt-started.json'
                generation = read_json(started).get('execution_attempt_id') if started.exists() else None
                termination = run / 'execution/attempts' / generation / 'termination.json' if generation else None
                user_cancelled = termination is not None and termination.exists() and read_json(termination).get('request_id') is not None
                result = {**sample.as_dict(), **(context or {}), 'attempt_id': out.name,
                          'exit_status': 'Cancelled' if user_cancelled else 'Interrupted',
                          'disposition': 'interrupted' if user_cancelled else 'infrastructure_retry'}
                write_json(outcome_path, result)
            if sample_accepted(cfg, result):
                verify_final(cfg['project_root'], run_id, sample.as_dict(), result, RunEvidence(cfg['project_root'], run_id).read())
                return _decorate_result(cfg, sample, work, out, result, history)
            if result.get('disposition') in {'failed', 'blocked'}:
                return _decorate_result(cfg, sample, work, out, result, history)
            history.append(result)
        round_id, retry_kind = _retry_cursor(cfg, history)
        infra_used = sum(read_json(p / 'attempt-started.json').get('retry_kind') == 'infrastructure'
                         for _, p in existing if (p / 'attempt-started.json').exists())
        infra_used += max(0, sum(not (p / 'attempt-context.json').exists() for _, p in existing) - 1)
        if retry_kind == 'infrastructure' and infra_used >= cfg['execution']['infrastructure_retry_limit']:
            last = {**history[-1], 'exit_status': 'infrastructure_exhausted', 'disposition': 'failed'}
            return _decorate_result(cfg, sample, work, existing[-1][1], last, history[:-1])
        if _boundary(run, factory):
            return {**sample.as_dict(), 'exit_status': _boundary(run, factory), 'disposition': 'pending'}
        actual = cfg.get('parallel', {}).get('branch_count', 1) if cfg.get('parallel', {}).get('enabled') else 1
        try:
            seed = effective_seed(sample, manifest, round_id, actual)
        except ValueError as exc:
            if cfg.get('_online_rl_source') and not str(exc).startswith('seed_space_exhausted:'):
                raise
            value = {**sample.as_dict(), 'exit_status': 'SeedSpaceExhausted', 'error': str(exc), 'disposition': 'failed',
                     'physical_attempts': sum((p / 'attempt-started.json').exists() for _, p in existing)}
            write_json(result_path, value)
            return value
        if reserved:
            out, context = reserved
        else:
            floors_path = run / 'rollout/replan-attempt-floors.json'
            floors = read_json(floors_path) if floors_path.exists() else {}
            floor = floors.get(sample.instance_id + '/' + sample.repetition_id, -1)
            number = max([floor, *(n for n, _ in existing)]) + 1
            attempt_id = f'{sample.attempt_prefix}-{number:03d}'
            out = attempts / attempt_id
            out.mkdir(exist_ok=True)
            context = AttemptContext(run_id, task_key, sample.sample_key, sample.repetition_id,
                                     attempt_id, round_id, seed, sample.sample_seed,
                                     SEED_POLICY if resampling_enabled(cfg) else sample.seed_policy).as_dict()
            publish_once(out / 'attempt-context.json', context)
        boundary = _boundary(run, factory)
        if boundary:
            return {**sample.as_dict(), 'exit_status': boundary, 'disposition': 'pending'}
        # Persist retry accounting at the actual physical start, not reservation.
        runtime = {**runtime, 'retry_kind': retry_kind}
        if 'serving_worker_id' in runtime:
            write_json(out / 'serving-assignment.json', {
                'run_id': run_id, 'sample_key': sample.sample_key, 'attempt_id': context['attempt_id'],
                'execution_attempt_id': os.environ.get('DYCON_EXECUTION_ATTEMPT'),
                'worker_id': runtime['serving_worker_id'], 'gpu': runtime['serving_worker_gpu'],
                'base_url': runtime['serving_worker_endpoint'], 'tensor_parallel_size': 1,
                'config_hash': cfg['config_hash'], 'assigned_at': now(),
            })
        try:
            result = _execute_attempt(cfg, run_id, task, factory, runtime, sample, context, out)
        except Exception as exc:
            if (cfg.get('_online_rl_source') not in {'baseline', 'dycon'}
                    or not (isinstance(exc, (TimeoutError, ConnectionError))
                            or type(exc).__name__ in INFRASTRUCTURE)):
                raise
            result = {**context, 'exit_status': type(exc).__name__, 'disposition': 'infrastructure_retry',
                      'error': repr(exc)}
        return _decorate_result(cfg, sample, work, out, result, history)


@exclusive_run
@cancellable_run
def run_manifest(cfg, run_id, resume=False, factory=None, *, task_ids=(), check_only=False):
    from core.model_identity import model_identity_verification_enabled, verify_model
    from runtime.backends.registry import adapter_for
    from runtime.serving.manager import ServingManager

    root = Path(cfg["project_root"])
    run_id = identifier(run_id)
    if not resume:
        if (root / "runs" / run_id).exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        create_run(
            cfg, run_id, operation="rollout", task_ids=task_ids,
            repeat_count=1,
        )
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    selection = cfg["selection"]
    work_items = loaded["work_items"]["items"]
    from core.config import assert_frozen

    assert_frozen(cfg)
    # Select only after the immutable run config has replaced the caller's YAML.
    adapter = adapter_for(cfg, selection["tasks"])
    if adapter.binding.agent == "minisweagent":
        os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(root / "state/miniswe-config")
        os.environ["MSWEA_SILENT_STARTUP"] = "1"
        if (Path(os.environ["MSWEA_GLOBAL_CONFIG_DIR"]) / ".env").exists():
            raise ValueError("mini-swe-agent .env overrides are disabled; use config.yaml and API-key environment")
    run = root / 'runs' / run_id
    CURRENT.get().start()
    if counts_recorded(run) or enabled(cfg):
        rebuild_counts(run, loaded['work_items'])
    from application.execution.failed_retry import retry_index, superseded
    operator_retries = retry_index(run)
    settled = {}
    evidence = RunEvidence(root, run_id).read()
    if resume and not check_only:
        for item in work_items:
            result_path = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
            if not result_path.exists():
                continue
            result = read_json(result_path)
            if result.get('sample_key') != item['sample_key']:
                raise ValueError('Completed sample identity changed')
            if sample_settled(cfg, result) and not superseded(operator_retries, item['sample_key'], result):
                if result.get('disposition') == 'accepted':
                    verify_final(root, run_id, item, result, evidence)
                settled[item['sample_key']] = result
        if len(settled) == len(work_items):
            tracker = Progress(run, cfg, work_items, settled)
            tracker._write()
            _settle_run_status(run, cfg, work_items, settled)
            _finalize_capture_publication(cfg, run_id, work_items)
            return settled
    if factory is not None:
        boundary = _boundary(run, factory)
        if boundary:
            update_run_status(run, 'cancelled' if boundary == 'Cancelled' else 'failed', reason=boundary)
            return settled
    elif request_for(run, os.environ.get('DYCON_EXECUTION_ATTEMPT')) is not None:
        update_run_status(run, 'cancelled', reason='Cancelled')
        return settled
    from hidden.recover_online import BackgroundCaptureRecovery
    with BackgroundCaptureRecovery(cfg, run_id, CURRENT.get(), enabled=resume and not check_only) as recovery:
        return _run_manifest_active(cfg, run_id, loaded, adapter, settled, factory,
                                    operator_retries, check_only, recovery)


def _run_manifest_active(cfg, run_id, loaded, adapter, settled, factory, operator_retries, check_only, recovery):
    from core.model_identity import model_identity_verification_enabled, verify_model
    from runtime.serving.manager import ServingManager

    root = Path(cfg['project_root'])
    run = root / 'runs' / run_id
    selection = cfg['selection']
    work_items = loaded['work_items']['items']
    from core.preflight import long_checks_enabled, timed_check
    model_checks = {}
    try:
        if model_identity_verification_enabled(cfg):
            deep = long_checks_enabled(cfg)
            with timed_check(model_checks, "model_sha", enabled=deep):
                verify_model(cfg) if deep else verify_model(cfg, deep=False)
    finally:
        write_json(run / "validation/startup-model-identity.json", model_checks)
    run = root / "runs" / run_id
    write_json(run / "process.json", {"pid": os.getpid(), "supervisor_pid": os.getppid()})
    backend_context = adapter.open_run(cfg, run_id, factory=factory)
    factory = backend_context.factory
    outcomes = dict(settled)
    capture_producers = []
    update_run_status(run, "rollout")
    try:
        checked = adapter.preflight(backend_context, selection["tasks"])
        if not checked.passed:
            raise ValueError(adapter.binding.adapter + ' preflight rejected run: ' + '; '.join(checked.blockers))
        sharded = cfg['serving'].get('trajectory_sharding', 'none') == 'per_gpu_tp1'
        if sharded:
            from runtime.serving.pool import ServingPool
            context = ServingPool(cfg, run / 'serving', run_id)
        else:
            context = ServingManager(cfg, run / 'serving')
        with context as serving:
            runtime = {} if sharded else getattr(serving, 'runtime', {})
            capture_producers = ([worker.manager.runtime.get('capture_controller') for worker in serving.workers]
                                 if sharded else [runtime.get('capture_controller')])
            worker_limit = serving.capacity if sharded else cfg['execution']['rollout_workers']
            if loaded['plan'].get('online_contract'):
                worker_limit = min(worker_limit, cfg['execution']['rollout_workers'])
            cancellation = CURRENT.get()
            runtime['cancellation'] = cancellation
            wake = CompletionWake(cancellation)

            def cancel_runtime():
                # Never put cancellation control onto the worker/branch executors.
                import threading
                def control():
                    if sharded:
                        errors = serving.cancel(run_id, cancellation.attempt_id)
                        if errors:
                            write_json(run / 'execution/attempts' / (cancellation.attempt_id or 'direct') / 'cancel-control-error.json',
                                       {'workers': errors, 'at': now()})
                        return
                    if runtime.get('capture_controller') is not None:
                        runtime['capture_controller'].cancel_run(run_id)
                    if cfg['serving']['management'] == 'managed' and cfg['capture']['enabled']:
                        import requests
                        endpoint = cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
                        try:
                            requests.post(endpoint + '/dycon/capture/engine/cancel',
                                json={'run_id': run_id, 'execution_attempt_id': cancellation.attempt_id},
                                headers={'X-DyCon-Cancel-Token': os.environ.get('DYCON_CANCEL_TOKEN', '')},
                                timeout=policy(cfg)['control_request_seconds']).raise_for_status()
                        except Exception as exc:
                            write_json(run / 'execution/attempts' / (cancellation.attempt_id or 'direct') / 'cancel-control-error.json',
                                       {'error': repr(exc), 'at': now()})
                threading.Thread(target=control, name='dycon-cancel-control', daemon=True).start()
            cancellation.add_callback(cancel_runtime)
            runtime["validation_mixed"] = check_only
            startup_task = next(iter(selection['tasks'].values()))
            if sharded:
                for worker in serving.workers:
                    adapter.startup(backend_context, worker.cfg, startup_task, {**runtime, **worker.runtime()})
            else:
                adapter.startup(backend_context, cfg, startup_task, runtime)
            # Resolve lazy native driver imports before worker admission. This
            # keeps the scheduler's first concurrent attempts from serializing
            # on Python module initialization.
            if hasattr(adapter, "agent_class"):
                adapter.agent_class()
                adapter.environment_class()
            if check_only:
                update_run_status(run, "validated")
                return {"startup": "passed"}
            # Explicit failed-task reruns go first, ahead of untouched pending tasks.
            pending = deque(sorted((item for item in work_items if item['sample_key'] not in settled),
                                   key=lambda item: item['sample_key'] not in operator_retries))
            queued_at = {item['sample_key']: time.monotonic() for item in pending}
            running = {}
            tracker = Progress(run, cfg, work_items, settled)
            runtime['progress_tracker'] = tracker
            runtime['work_items'] = loaded['work_items']
            runtime['operator_retries'] = operator_retries
            for item in pending:
                tracker.phases[item['sample_key']] = 'ready'
            tracker._write()
            stopped = None
            from rollout.startup_failure_gate import StartupFailureGate
            startup_gate = StartupFailureGate(cfg['execution'].get('startup_failure_limit'))
            reservations = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as pool:
                def fill():
                    nonlocal stopped
                    if stopped:
                        return
                    while pending and len(running) < worker_limit:
                        stopped = _boundary(run, factory)
                        if stopped:
                            return
                        worker = serving.acquire() if sharded else None
                        if sharded and worker is None:
                            return
                        item = pending.popleft()
                        task = selection['tasks'][item['task_key']]
                        task_runtime = {**runtime, 'queue_wait_seconds': time.monotonic() - queued_at[item['sample_key']]}
                        # Online bootstrap work manifests carry a source profile
                        # per logical trajectory; preserve it through the
                        # existing attempt runtime so OpenHands can select
                        # baseline versus estimator-consensus semantics while
                        # reusing the same managed serving process.
                        if item.get('online_source') in {'baseline', 'dycon'}:
                            task_runtime['online_source'] = item['online_source']
                        if item.get('candidate_index') is not None:
                            task_runtime['online_candidate_index'] = item['candidate_index']
                        if worker is not None:
                            task_runtime.update(worker.runtime())
                        tracker.phase(item['sample_key'], 'admission')
                        try:
                            future = pool.submit(admitted_task, cfg, run_id, item['task_key'], task, factory,
                                                 task_runtime, sample_from_item({**item, 'run_id': run_id}))
                        except BaseException:
                            if worker is not None:
                                serving.release(worker)
                            raise
                        running[future] = item
                        if worker is not None:
                            reservations[future] = worker
                        future.add_done_callback(wake.notify)

                fill()
                while pending or running:
                    if not running:
                        fill()
                        if not running:
                            break
                    finished = wake.wait(running, cancellation)
                    if cancellation.event.is_set():
                        stopped = 'Cancelled'
                        pending.clear()
                        for future in running:
                            future.cancel()
                        # Cooperative control makes running futures finish; supervisor
                        # bounds this executor join if their synchronous I/O is stuck.
                        break
                    for future in finished:
                        item = running.pop(future)
                        if sharded:
                            serving.release(reservations.pop(future))
                        try:
                            value = future.result()
                        except Exception as exc:
                            value = {**item, 'exit_status': 'WorkerFailure', 'disposition': 'failed',
                                     'error': repr(exc), 'traceback': traceback.format_exc()}
                        kind = value.get('disposition') or disposition(cfg, value.get('exit_status'))
                        tracker.result(item['sample_key'], value)
                        if startup_gate.observe(value):
                            stopped = 'StartupFailureThreshold'
                        if startup_gate.limit is not None:
                            write_json(run / 'rollout/startup-failure-gate.json', startup_gate.receipt(run_id))
                        boundary = _boundary(run, factory)
                        if boundary:
                            stopped = boundary
                        if kind in {'retry_pending', 'selection_retry', 'infrastructure_retry', 'admission_waiting'} and not stopped:
                            pending.append(item)
                            queued_at[item['sample_key']] = time.monotonic()
                            tracker.phase(item['sample_key'], 'ready')
                        else:
                            outcomes[item['sample_key']] = value
                    # Requeue before refilling; a temporarily empty running set is not completion.
                    fill()
            if stopped:
                update_run_status(run, 'cancelled' if stopped == 'Cancelled' else 'failed',
                              reason=stopped, pending_samples=len(work_items) - sum(sample_accepted(cfg, v) for v in outcomes.values()))
                if counts_recorded(run):
                    rebuild_counts(run, loaded['work_items'])
                return outcomes
        if counts_recorded(run):
            rebuild_counts(run, loaded['work_items'])
        tracker._write()
        _settle_run_status(run, cfg, work_items, outcomes)
        return outcomes
    except Cancelled:
        update_run_status(run, 'cancelling', reason='Cancelled')
        return outcomes
    except BaseException as exc:
        update_run_status(run, "failed", error=repr(exc))
        raise
    finally:
        try:
            recovery.close()
        finally:
            try:
                adapter.close_run(backend_context, CURRENT.get())
            finally:
                if not check_only:
                    _finalize_capture_publication(cfg, run_id, work_items, producers=capture_producers)


def _settle_run_status(run, cfg, work_items, outcomes):
    failed, discarded = [], []
    for item in work_items:
        key = item['sample_key']
        result = outcomes.get(key, {})
        if sample_accepted(cfg, result):
            continue
        if item.get('online_source') in {'baseline', 'dycon'} and online_sampling_discard(result):
            discarded.append(key)
        else:
            failed.append(key)
    update_run_status(run, 'failed' if failed else 'complete', failed=failed,
                      **({'discarded_samples': discarded} if discarded else {}))


def _finalize_capture_publication(cfg, run_id, work_items, *, producers=()):
    # run_manifest owns the run lock. All worker/serving/backend contexts have
    # exited before this call, including cancellation and exception paths.
    try:
        from application.execution.capture_publication import finalize_capture_publication
        return finalize_capture_publication(cfg, run_id, items=work_items, producers=producers)
    except Exception:
        # Even an import or summary-storage failure cannot change rollout facts.
        return None


def admitted_task(cfg, run_id, key, task, factory, runtime, sample=None):
    # Per-attempt config is a derived serving route; the run's immutable snapshot
    # and config_hash remain authoritative for all other execution semantics.
    if 'serving_cfg' in runtime:
        cfg = copy.deepcopy(runtime['serving_cfg'])
    if runtime.get('online_source') in {'baseline', 'dycon'}:
        cfg = copy.deepcopy(cfg)
        cfg['_online_rl_source'] = runtime['online_source']
    if runtime.get('online_source') == 'baseline':
        # The shared server keeps the estimator broker available for DyCon.
        # Only this trajectory's agent bypasses branch generation/scoring.
        if (cfg.get('execution', {}).get('online_rl_failure_policy') or {}).get('baseline_failure_reward') is not None:
            cfg['execution']['limits_retry'] = {**cfg['execution'].get('limits_retry', {}), 'enabled': False}
        cfg.setdefault('parallel', {})['enabled'] = False
        cfg['parallel']['branch_count'] = 1
    run = Path(cfg['project_root']) / 'runs' / run_id
    boundary = _boundary(run, factory)
    if boundary:
        return {**sample.as_dict(), 'exit_status': boundary, 'disposition': 'pending'}
    controller = runtime.get('capture_controller')
    stamp = time.monotonic()
    if controller and controller.spool:
        options = cfg['capture']['export']
        try:
            controller.spool.budget.admit(options['spool_high_watermark'], options['spool_low_watermark'],
                factory.deadline.remaining(cfg['capture'].get('admission_seconds') or options['upload_max_elapsed_seconds']))
        except TimeoutError as exc:
            # No physical attempt has started. Keep capacity waits in the queue.
            return {**sample.as_dict(), 'exit_status': 'AdmissionWaiting',
                    'disposition': 'admission_waiting', 'error': str(exc)}
    runtime = {**runtime, 'admission_wait_seconds': time.monotonic() - stamp}
    return run_task(cfg, run_id, key, task, factory, runtime, sample=sample)


def run_online_manifest(cfg, run_id, *, task_ids, m, branch_count=None, resume=False, request=None):
    """Execute one mixed online batch through the standard run lifecycle.

    The run planner freezes one DyCon item plus ``m`` baseline repetitions per
    task.  Existing supervisor, serving, retry, cancellation and evidence code
    then owns execution; this helper only supplies the online profile manifest.
    """
    from application.execution.run_plan import _selection_from_tasks, create_run
    from catalog.store import Catalog
    from RL.online_runtime import make_online_run_work_items

    root = Path(cfg["project_root"])
    run_id = identifier(run_id)
    from core.io import run_path as locate_run
    run_directory = locate_run(root, run_id)
    if not run_directory.exists():
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("online task_ids must be non-empty and unique")
        selection = _selection_from_tasks(Catalog(root), task_ids=task_ids, cfg=cfg)
        branches = int(branch_count or cfg.get("parallel", {}).get("branch_count", 1))
        manifest = make_online_run_work_items(selection, run_id, m=int(m),
                                              base_seed=int(cfg["generation"]["seed"]),
                                              branch_count=branches)
        if request is not None:
            ordered = []
            available = {(item["instance_id"], item["online_source"], item["online_repetition"]): item
                         for item in manifest["items"]}
            repetitions = {}
            for work in request["work_items"]:
                pair = (work["task_id"], work["source"])
                repetition = repetitions.get(pair, 0)
                repetitions[pair] = repetition + 1
                ordered.append(available.pop((*pair, repetition)))
            if available:
                raise ValueError("online work order does not cover all frozen samples")
            manifest["items"] = ordered
        create_run(cfg, run_id, operation="rollout", task_ids=list(task_ids),
                   repeat_count=int(m) + 1, online_work_items=manifest)
        resume = False
    from application.execution.run_plan import load_plan
    from application.execution.supervisor import launch, wait
    from core.io import read_json
    loaded = load_plan(root, run_id)
    deployment = (request or {}).get("deployment")
    if deployment:
        frozen = loaded["config"]
        if (frozen.get("execution", {}).get("online_rl_failure_policy") or {}) != (deployment.get("failure_policy") or {}):
            raise ValueError("existing online run differs from frozen failure policy")
        if ((frozen["serving"].get("lora") or {}).get("identity_sha256") != deployment["adapter_identity_sha256"]
                or frozen["parallel"].get("estimator_sha256") != deployment["estimator_manifest_sha256"]
                or frozen["parallel"]["branch_count"] != int(deployment["branch_count"])
                or frozen["model"]["fingerprint"] != deployment["base_fingerprint"]
                or frozen["serving"]["gpus"] != request["worker_ids"]):
            raise ValueError("existing online run differs from immutable generating deployment")
    if (loaded["plan"].get("online_contract", {}).get("baseline_repetitions_per_task") != int(m)
            or set(loaded["config"]["selection"]["tasks"][key]["instance_id"]
                   for key in loaded["config"]["selection"]["task_keys"]) != set(task_ids)):
        raise ValueError("existing online run differs from requested task/profile contract")
    latest = run_directory / "execution/latest.json"
    if latest.is_file():
        attempt = run_directory / "execution/attempts" / read_json(latest)["attempt"]
        receipt_path = attempt / "exit.json"
        if not receipt_path.is_file():
            return wait(root, run_id)
        receipt = read_json(receipt_path)
        if receipt.get("exit_code") == 0:
            return wait(root, run_id)
        resume = True
    return launch(root, run_id, resume=resume, detach=False)
