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

from core.cancellation import CURRENT, Cancelled, CompletionWake, cancellable_run, request_for, policy
from application.execution.evidence import RunEvidence
from application.execution.run_plan import create_run, load_plan
from core.io import identifier, now, read_json, sha256, update_status, write_json
from core.locking import exclusive_run, file_lock
from rollout.artifacts import import_rollout
from rollout.progress import Progress
from rollout.retry import INFRASTRUCTURE as INFRASTRUCTURE
from rollout.retry import (
    SEED_POLICY,
    TERMINAL,
    counts_recorded,
    disposition,
    enabled,
    initialize_counts,
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
        if task.get('benchmark') == 'webarena_verified':
            from application.execution.subject import verify_subject
            verify_subject(root, row)
        if path.exists() and sha256(path) != row['trajectory_sha256']:
            raise ValueError('Recovered raw and canonical bytes disagree')
    else:
        subject_spec = {}
        if task.get('benchmark') == 'webarena_verified':
            subject_path = out / 'web-subject' / 'subject.json'
            if not subject_path.exists():
                from runtime.agents.webarena import finalize_subject
                subject_path, _ = finalize_subject(task, out, context, None, '',
                                                   state.get('agent_exit_status', 'Unknown'))
            subject_spec = {'subject': str(subject_path), 'subject_sha256': sha256(subject_path)}
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
                if capture_status not in {'complete', 'not_exported'}:
                    raise RuntimeError('Capture is ' + capture_status)
            else:
                raise RuntimeError('Capture finalization requires recovery of session ' + capture_session_id + '; use dycon capture recover')
        except Exception as exc:
            capture_status = 'partial'
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
    if task.get('benchmark') == 'webarena_verified' and (
            state.get('web_subject_error') or state.get('web_subject_errors') or
            (state['agent_exit_status'] == 'Submitted' and not read_json(root / row['subject_path'])['complete'])):
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
    dependency_failure = (state.get('error') or {}).get('dependency_failure')
    if dependency_failure:
        result['dependency_failure'] = dependency_failure
        result['failure_type'] = 'dependency_error'
    result['disposition'] = 'blocked' if status in {'CaptureError', 'StatisticsError', 'CleanupError', 'WebTraceError'} else disposition(cfg, status)
    write_json(out / 'attempt.json', result)
    return result


def _execute_attempt(cfg, run_id, task, factory, runtime, sample, context, out):
    """One clean environment, agent, model and capture session per attempt."""
    if task.get("benchmark") == "skillsbench":
        from runtime.agents.skillsbench import execute_attempt

        return execute_attempt(cfg, run_id, task, runtime, sample, context, out, factory)
    webarena = task.get('benchmark') == 'webarena_verified'
    from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
    from rollout.agent import DurableAgent
    from rollout.model import make_model
    from runtime.containers.factory import TaskContext
    if webarena:
        from runtime.containers.webarena_adapter import WebArenaEnvironment
        RolloutEnvironment = WebArenaEnvironment
    else:
        from runtime.containers.rollout_adapter import RolloutEnvironment

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
    tracker = runtime.get('progress_tracker')
    if tracker:
        try:
            tracker.started(context, runtime.get('retry_kind', 'initial'))
        except Exception as exc:
            runtime.setdefault('progress_errors', []).append(repr(exc))
    _phase(runtime, sample, 'bootstrap')
    stage_start = time.monotonic()

    def capture_failure(exc, abort=True):
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
                capture_client.start(capture_context, graph_mode='FULL', prefix_cache_policy='safe_boundary')
            except Exception as exc:
                capture_context = False
                capture_status = 'failed'
                capture_failure(exc)
        agent_cfg = cfg['agent']
        if webarena:
            from runtime.agents.webarena import WebArenaAgent
            DurableAgent = WebArenaAgent
            profile = cfg['execution_context']['profile']
            agent_cfg = {**agent_cfg, **{key: profile[key] for key in (
                'system_template', 'instance_template', 'observation_template', 'format_error_template')}}
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
        info = agent.run(task['instance']['intent'] if webarena else task['instance']['problem_statement'])
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
            if webarena:
                try:
                    from runtime.agents.webarena import finalize_subject
                    subject_path, subject_errors = finalize_subject(
                        task, out, context, env, patch, status)
                    if subject_errors:
                        state['web_subject_errors'] = subject_errors
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
                        flushed = requests.post(endpoint + '/dycon/capture/engine/flush',
                            json={'session_id': capture_session_id},
                            timeout=cfg['capture'].get('flush_timeout_seconds') or cfg['capture']['finalize_seconds'])
                        flushed.raise_for_status()
                        if flushed.json().get('failures'):
                            raise RuntimeError('Capture drain failed: ' + str(flushed.json()['failures']))
                    except Exception as exc:
                        capture_failure(exc, abort=False)
                        capture_status = 'partial'
                    capture_client.timeout = cfg['capture']['finalize_seconds'] + cfg['capture']['control_seconds']
                    try:
                        capture_client.stop(capture_session_id)
                    except Exception as exc:
                        capture_failure(exc, abort=False)
                        capture_status = 'partial'
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
    if not fact_path.exists() and (status not in TERMINAL or binding is None):
        return None
    if binding != context or info.get('attempt_id') != context['attempt_id'] or raw.get('instance_id') != sample.instance_id:
        raise ValueError('Saved trajectory does not match reserved attempt context')
    if status not in TERMINAL and not fact_path.exists():
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
        if kind == 'retry_pending':
            round_id += 1
            retry_kind = 'resample'
        elif kind == 'infrastructure_retry':
            retry_kind = 'infrastructure'
        elif kind == 'operator_retry':
            retry_kind = 'operator'
    return round_id, retry_kind


def _decorate_result(cfg, sample, work, out, result, history):
    result = {**sample.as_dict(), **result}
    assignment = out / 'serving-assignment.json'
    if assignment.exists():
        result['serving_assignment'] = read_json(assignment)
    result.setdefault('seed_round', 0)
    result.setdefault('effective_seed', sample.sample_seed)
    result.setdefault('disposition', disposition(cfg, result.get('exit_status')))
    records = [*history, result]
    started = [p for p in (work / 'attempts').iterdir() if (p / 'attempt-started.json').exists()]
    contexts = [read_json(p / 'attempt-context.json') for p in started if (p / 'attempt-context.json').exists()]
    legacy = [p for p in (work / 'attempts').iterdir() if p.is_dir() and not (p / 'attempt-context.json').exists()]
    result.update(
        physical_attempts=len(started) + len(legacy),
        resample_retry_count=len({c['seed_round'] for c in contexts if c['seed_round'] > 0}),
        infrastructure_retry_count=max(0, len(legacy) - 1) + sum(read_json(p / 'attempt-started.json').get('retry_kind') == 'infrastructure' for p in started))
    if result.get('disposition') == 'infrastructure_retry' and result['infrastructure_retry_count'] >= cfg['execution']['infrastructure_retry_limit']:
        result.update(exit_status='infrastructure_exhausted', last_exit_status=result['exit_status'], disposition='failed')
    write_json(out / 'attempt.json', result)
    next_round, retry_kind = _retry_cursor(cfg, records[:-1] + [result])
    write_json(work / 'retry-state.json', {
        'sample_key': sample.sample_key, 'seed_round': next_round, 'retry_kind': retry_kind,
        'last_attempt_id': result['attempt_id'], 'last_status': result['exit_status'],
        'resample_retry_count': result['resample_retry_count'],
        'infrastructure_retry_count': result['infrastructure_retry_count']})
    if result['disposition'] not in {'retry_pending', 'infrastructure_retry'}:
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
        from application.execution.failed_retry import retry_index, superseded
        operator_retries = runtime.get('operator_retries')
        if operator_retries is None:
            operator_retries = retry_index(run)
        result_path = work / 'result.json'
        if result_path.exists():
            old = read_json(result_path)
            if old.get('sample_key') != sample.sample_key:
                raise ValueError('Completed sample identity changed')
            if sample_settled(cfg, old) and not superseded(operator_retries, sample.sample_key, old):
                if sample_accepted(cfg, old):
                    verify_final(cfg['project_root'], run_id, sample.as_dict(), old, RunEvidence(cfg['project_root'], run_id).read())
                return old
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
        history = []
        reserved = None
        for _, out in existing:
            context_path = out / 'attempt-context.json'
            context = read_json(context_path) if context_path.exists() else None
            if context:
                for key in ('run_id', 'task_key', 'sample_key', 'repetition_id', 'sample_seed'):
                    if context.get(key) != sample.as_dict()[key]:
                        raise ValueError('Reserved attempt identity changed')
                expected_policy = SEED_POLICY if enabled(cfg) else sample.seed_policy
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
            value = {**sample.as_dict(), 'exit_status': 'SeedSpaceExhausted', 'error': str(exc), 'disposition': 'failed'}
            write_json(result_path, value)
            return value
        if reserved:
            out, context = reserved
        else:
            number = max((n for n, _ in existing), default=-1) + 1
            attempt_id = f'{sample.attempt_prefix}-{number:03d}'
            out = attempts / attempt_id
            out.mkdir()
            context = AttemptContext(run_id, task_key, sample.sample_key, sample.repetition_id,
                                     attempt_id, round_id, seed, sample.sample_seed,
                                     SEED_POLICY if enabled(cfg) else sample.seed_policy).as_dict()
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
        result = _execute_attempt(cfg, run_id, task, factory, runtime, sample, context, out)
        return _decorate_result(cfg, sample, work, out, result, history)


@exclusive_run
@cancellable_run
def run_manifest(cfg, run_id, resume=False, factory=None, *, task_ids=(), check_only=False):
    from core.backends import identity
    skillsbench = identity(cfg) == ("skillsbench", "openhands")
    webarena = identity(cfg) == ("webarena_verified", "minisweagent")
    if not skillsbench:
        os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(cfg["project_root"]) / "state/miniswe-config")
        os.environ["MSWEA_SILENT_STARTUP"] = "1"
        if (Path(os.environ["MSWEA_GLOBAL_CONFIG_DIR"]) / ".env").exists():
            raise ValueError(
                "mini-swe-agent .env overrides are disabled; use config.yaml and API-key environment"
            )
    from core.model_identity import verify_model
    if skillsbench:
        from runtime.agents.skillsbench import SkillsBenchFactory
        ContainerFactory = SkillsBenchFactory
    elif webarena:
        from runtime.containers.webarena_adapter import WebArenaFactory
        ContainerFactory = WebArenaFactory
    else:
        from runtime.containers.factory import ContainerFactory
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
                if sample_accepted(cfg, result):
                    verify_final(root, run_id, item, result, evidence)
                settled[item['sample_key']] = result
        if len(settled) == len(work_items):
            tracker = Progress(run, cfg, work_items, settled)
            tracker._write()
            failed = [k for k, v in settled.items() if not sample_accepted(cfg, v)]
            update_status(run / 'status.json', 'failed' if failed else 'complete', failed=failed)
            return settled
    if factory is not None:
        boundary = _boundary(run, factory)
        if boundary:
            update_status(run / 'status.json', 'cancelled' if boundary == 'Cancelled' else 'failed', reason=boundary)
            return settled
    elif request_for(run, os.environ.get('DYCON_EXECUTION_ATTEMPT')) is not None:
        update_status(run / 'status.json', 'cancelled', reason='Cancelled')
        return settled
    # Recover capture publication using the established no-model recovery path.
    if resume and cfg.get('capture', {}).get('enabled'):
        needs_capture_recovery = any(p.exists() for p in (run / 'rollout/tasks').glob('*/repetitions/*/attempts/*/capture-session.json'))
        if needs_capture_recovery:
            from hidden.recover_online import recover_online
            try:
                recover_online.__wrapped__(cfg, run_id)
            except Exception as exc:
                write_json(run / 'capture-recovery-error.json', {'error': repr(exc)})

    verify_model(cfg)
    run = root / "runs" / run_id
    write_json(run / "process.json", {"pid": os.getpid(), "supervisor_pid": os.getppid()})
    factory = factory or (ContainerFactory(cfg, run_id) if skillsbench else
                          ContainerFactory(cfg, run_id) if webarena else ContainerFactory(cfg))
    outcomes = dict(settled)
    update_status(run / "status.json", "rollout")
    try:
        factory.lock_images([t["instance"] for t in selection["tasks"].values()])
        if skillsbench:
            from runtime.agents.skillsbench import preflight
            checked = preflight(root, run_id, workers=min(8, cfg['execution']['rollout_workers']))
            write_json(run / 'validation/startup-preflight.json', checked)
            if checked['blockers']:
                raise ValueError('SkillsBench preflight rejected run: ' + '; '.join(checked['blockers']))
        elif webarena:
            from runtime.containers.webarena_adapter import preflight
            checked = preflight(root, run_id, factory=factory, tasks=list(selection['tasks'].values()))
            write_json(run / 'validation/webarena-preflight.json', checked)
            if not checked.get('passed'):
                raise ValueError('WebArena preflight rejected run: ' + '; '.join(map(str, checked.get('blockers', []))))
        sharded = cfg['serving'].get('trajectory_sharding', 'none') == 'per_gpu_tp1'
        if sharded:
            from runtime.serving.pool import ServingPool
            context = ServingPool(cfg, run / 'serving', run_id)
        else:
            context = ServingManager(cfg, run / 'serving')
        with context as serving:
            runtime = {} if sharded else getattr(serving, 'runtime', {})
            worker_limit = serving.capacity if sharded else cfg['execution']['rollout_workers']
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
            if not skillsbench:
                from runtime.serving.startup import check_tool_chain
                startup_task = next(iter(selection['tasks'].values()))
                if sharded:
                    for worker in serving.workers:
                        check_tool_chain(worker.cfg, run_id, startup_task, factory, {**runtime, **worker.runtime()})
                else:
                    check_tool_chain(cfg, run_id, startup_task, factory, runtime)
            if check_only:
                update_status(run / "status.json", "validated")
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
                        if kind in {'retry_pending', 'infrastructure_retry', 'admission_waiting'} and not stopped:
                            pending.append(item)
                            queued_at[item['sample_key']] = time.monotonic()
                            tracker.phase(item['sample_key'], 'ready')
                        else:
                            outcomes[item['sample_key']] = value
                    # Requeue before refilling; a temporarily empty running set is not completion.
                    fill()
            if stopped:
                update_status(run / 'status.json', 'cancelled' if stopped == 'Cancelled' else 'failed',
                              reason=stopped, pending_samples=len(work_items) - sum(sample_accepted(cfg, v) for v in outcomes.values()))
                if counts_recorded(run):
                    rebuild_counts(run, loaded['work_items'])
                return outcomes
        if cfg["capture"]["enabled"] and cfg["capture"]["export"]["enabled"]:
            from enrichment.online_trace import finalize_token_events

            source = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
            target = root / "runs" / run_id / "enrichment/online"
            if not target.exists() and (source / "requests").exists():
                try:
                    inventory = read_json(source / "inventory.json")
                    finalize_token_events(
                        source,
                        target,
                        open_ids=inventory.get("open_token_ids", [248068]),
                        close_ids=inventory["close_token_ids"],
                    )
                except Exception as exc:
                    write_json(run / "capture-finalization-error.json", {"error": repr(exc)})
                    if cfg["capture"]["failure_policy"] == "block_rollout":
                        from runtime.serving.errors import CaptureError

                        raise CaptureError(str(exc)) from exc
        if counts_recorded(run):
            rebuild_counts(run, loaded['work_items'])
        tracker._write()
        failed = [item['sample_key'] for item in work_items
                  if not sample_accepted(cfg, outcomes.get(item['sample_key'], {}))]
        update_status(run / "status.json", "failed" if failed else "complete", failed=failed)
        return outcomes
    except Cancelled:
        update_status(run / 'status.json', 'cancelling', reason='Cancelled')
        return outcomes
    except BaseException as exc:
        update_status(run / "status.json", "failed", error=repr(exc))
        raise
    finally:
        cancellation = CURRENT.get()
        if cancellation is not None and cancellation.event.is_set() and hasattr(factory, 'cleanup_cancelled'):
            factory.cleanup_cancelled(run_id)
        else:
            factory.cleanup_owned(run_id)


def admitted_task(cfg, run_id, key, task, factory, runtime, sample=None):
    # Per-attempt config is a derived serving route; the run's immutable snapshot
    # and config_hash remain authoritative for all other execution semantics.
    if 'serving_cfg' in runtime:
        cfg = copy.deepcopy(runtime['serving_cfg'])
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
