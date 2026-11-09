from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from core.deadline import Deadline
from rollout.adapters.openhands import OpenHandsLimit
from rollout.adapters.openhands_bridge import OpenHandsBridge
from rollout.retry import disposition
from rollout.sample_token_budget import BudgetStopped, SampleTokenBudget


def binding(attempt):
    return {'run_id': 'run-a', 'task_key': 'task-a', 'sample_key': 'sample-a',
            'repetition_id': 'rep-000', 'attempt_id': attempt}


def cfg(tmp_path, limit=10):
    return {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': limit}}


def test_new_physical_attempt_gets_full_budget_and_keeps_old_evidence(tmp_path):
    first = SampleTokenBudget(cfg(tmp_path), binding('rep-000-attempt-001'))
    assert first.reserve('api1', 'agent_decision', 10) == 10
    first.record('api1', 'agent_decision', {'completion_tokens': 4})
    second = SampleTokenBudget(cfg(tmp_path), binding('rep-000-attempt-002'))
    assert second.reserve('api2', 'condensation', 10) == 10
    assert second.path == Path(tmp_path) / 'runs/run-a/rollout/sample-token-budgets/sample-a/rep-000-attempt-002.json'
    assert first.lock_path != second.lock_path
    assert first.snapshot()['completion_tokens'] == 4
    assert second.snapshot()['completion_tokens'] == 0


def test_unknown_and_pending_are_isolated_per_attempt(tmp_path):
    first = SampleTokenBudget(cfg(tmp_path, 20), binding('a1'))
    first.reserve('api1', 'condensation', 10)
    first.record_unknown('api1', 'condensation')
    with pytest.raises(BudgetStopped, match='sample_token_usage_unknown'):
        first.reserve('api2', 'agent_decision', 1)
    second = SampleTokenBudget(cfg(tmp_path, 20), binding('a2'))
    assert second.reserve('api2', 'agent_decision', 20) == 20
    pending = SampleTokenBudget(cfg(tmp_path, 20), binding('a3'))
    assert pending.reserve('api3', 'agent_decision', 15) == 15
    assert pending.reserve('api4', 'agent_decision', 6) == 5
    fresh = SampleTokenBudget(cfg(tmp_path, 20), binding('a4'))
    assert fresh.reserve('api4', 'agent_decision', 20) == 20


def test_parallel_reservations_cannot_overcommit_and_release_unused(tmp_path):
    ledger = SampleTokenBudget(cfg(tmp_path, 100), binding('a1'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(pool.map(lambda i: ledger.reserve(f'api{i}', 'agent_decision', 30), range(8)))
    assert sum(x for x in allowed if x is not None) == 100
    for api, row in ledger.snapshot()['requests'].items():
        ledger.record(api, row['role'], {'completion_tokens': 1})
    assert ledger.remaining() == 96


def test_receipts_are_idempotent_and_binding_is_strict(tmp_path):
    ledger = SampleTokenBudget(cfg(tmp_path), binding('a1'))
    ledger.reserve('api1', 'agent_decision', 10)
    ledger.record('api1', 'agent_decision', {'completion_tokens': 4})
    ledger.record('api1', 'agent_decision', {'completion_tokens': 4})
    with pytest.raises(ValueError, match='cannot be admitted twice'):
        ledger.reserve('api1', 'agent_decision', 1)
    with pytest.raises(ValueError, match='no reservation'):
        ledger.record('api2', 'agent_decision', {'completion_tokens': 1})
    with pytest.raises(ValueError, match='no reservation'):
        ledger.record_unknown('api2', 'agent_decision')


def test_corrupt_or_legacy_ledgers_fail_closed_and_old_path_is_untouched(tmp_path):
    root = Path(tmp_path) / 'runs/run-a/rollout/sample-token-budgets'
    root.mkdir(parents=True)
    legacy = root / 'sample-a.json'
    legacy.write_text('{"schema_version": 1, "completion_tokens": 99}\n')
    ledger = SampleTokenBudget(cfg(tmp_path, 10), binding('a1'))
    assert ledger.reserve('api1', 'agent_decision', 10) == 10
    assert legacy.read_text() == '{"schema_version": 1, "completion_tokens": 99}\n'
    valid_attempt_bytes = ledger.path.read_text()
    ledger.path.write_text('{"schema_version": 1}\n')
    with pytest.raises(ValueError, match='schema or scope'):
        ledger.snapshot()
    other = SampleTokenBudget(cfg(tmp_path, 10), binding('a2'))
    other.path.parent.mkdir(parents=True, exist_ok=True)
    other.path.write_text(valid_attempt_bytes)
    with pytest.raises(ValueError, match='binding or frozen limit'):
        other.snapshot()
    assert legacy.read_text() == '{"schema_version": 1, "completion_tokens": 99}\n'


def test_disabled_budget_keeps_legacy_behavior():
    ledger = SampleTokenBudget({'execution': {}}, {})
    assert ledger.reserve('api', 'unqualified', 55) == 55
    assert ledger.snapshot() is None


@pytest.mark.parametrize('missing_usage', [False, True])
def test_bridge_budget_receipt_is_attempt_bound(tmp_path, monkeypatch, missing_usage):
    config = cfg(tmp_path, 3)
    config.update({'model': {'served_name': 'fixture'},
                   'capture': {'enabled': False}, 'parallel': {'enabled': False},
                   'serving': {'management': 'external', 'base_url': 'http://fixture/v1',
                               'api_key_env': 'FIXTURE_KEY'}, 'timeouts': {'model_seconds': 3},
                   'agent': {'step_limit': 10, 'loop_repeat_limit': 3, 'loop_max_period': 2},
                   'generation': {'seed': 1, 'preserve_thinking': False}})
    bridge = OpenHandsBridge(config, binding('a1'), {'request_roles': 'unqualified'},
                             tmp_path / 'out', Deadline(10))
    def post(url, **kwargs):
        raw = {'id': 'response1', 'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                  'finish_reason': 'stop'}]}
        if not missing_usage:
            raw['usage'] = {'prompt_tokens': 5, 'completion_tokens': 3}
        return SimpleNamespace(ok=True, status_code=200, json=lambda: raw)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', post)
    monkeypatch.setattr('rollout.model.model_config', lambda *args, **kwargs: {'model_kwargs': {
        'temperature': 0, 'top_p': 1, 'max_tokens': 3, 'seed': 1,
        'presence_penalty': 0, 'extra_body': {}}})
    with pytest.raises(OpenHandsLimit):
        bridge.complete({'messages': []})
    receipt = json.loads((tmp_path / 'out/sample-token-budget.json').read_text())
    assert receipt['binding']['attempt_id'] == 'a1'
    assert receipt['budget']['scope'] == 'physical_attempt'
    status = 'SampleTokenBudgetUsageUnknown' if missing_usage else 'SampleTokenBudgetExceeded'
    assert disposition(config, status) == 'failed'
    assert receipt['reason'] == ('sample_token_usage_unknown' if missing_usage
                                 else 'sample_completion_token_budget')


CONTEXT_MESSAGE = ("This model's maximum context length is 262144 tokens. However, you requested "
                   "98304 output tokens and your prompt contains at least 163841 input tokens, "
                   "for a total of at least 262145 tokens.")


@pytest.mark.parametrize('payload', [
    {'error': {'code': 'context_length_exceeded'}},
    {'error': {'code': 'context_window_exceeded'}},
    *[{'error': {'type': 'BadRequestError', 'param': 'input_tokens', 'message':
                CONTEXT_MESSAGE.replace('98304', str(output)).replace('163841', str(prompt))}}
      for output, prompt in [(98304, 163841), (97720, 164425), (68553, 193592)]],
])
def test_context_error_recognizes_structured_codes_and_observed_payloads(payload):
    from rollout.adapters.openhands_bridge import is_context_window_error
    assert is_context_window_error(400, payload)
    assert not is_context_window_error(500, payload)


@pytest.mark.parametrize('payload', [
    None, [], {'error': 'context_length_exceeded'},
    {'error': {'type': 'BadRequestError', 'param': 'input_tokens', 'message': 'invalid input'}},
    {'error': {'param': 'temperature', 'message': CONTEXT_MESSAGE}},
    {'error': {'message': 'context is unavailable', 'code': ['context_length_exceeded']}},
])
def test_context_error_does_not_guess_from_generic_failures(payload):
    from rollout.adapters.openhands_bridge import is_context_window_error
    assert not is_context_window_error(400, payload)


def context_bridge(tmp_path, monkeypatch, *, limit=262144):
    config = cfg(tmp_path, limit)
    config.update({'model': {'served_name': 'fixture'},
                   'capture': {'enabled': False}, 'parallel': {'enabled': False},
                   'serving': {'management': 'external', 'base_url': 'http://fixture/v1',
                               'api_key_env': 'FIXTURE_KEY'}, 'timeouts': {'model_seconds': 3},
                   'agent': {'step_limit': 10, 'loop_repeat_limit': 3, 'loop_max_period': 2},
                   'generation': {'seed': 1, 'preserve_thinking': False}})
    monkeypatch.setattr('rollout.model.model_config', lambda *args, **kwargs: {'model_kwargs': {
        'temperature': 0, 'top_p': 1, 'max_tokens': 98304, 'seed': 1,
        'presence_penalty': 0, 'extra_body': {}}})
    return OpenHandsBridge(config, binding('a1'), {'request_roles': 'unqualified'},
                           tmp_path / 'out', Deadline(10))


def test_context_rejection_then_budget_stop_keeps_primary_reason_and_usage(tmp_path, monkeypatch):
    from rollout.adapters.openhands_bridge import UpstreamModelError
    from rollout.retry import sample_accepted
    from runtime.agents.skillsbench import (_bridge_agent_status, _map_evaluation,
                                           _terminal_evidence, backend_failure_result)
    bridge = context_bridge(tmp_path, monkeypatch)
    payload = {'error': {'type': 'BadRequestError', 'param': 'input_tokens', 'message': CONTEXT_MESSAGE}}
    posts = []
    def post(*args, **kwargs):
        posts.append(kwargs['json'])
        return SimpleNamespace(ok=False, status_code=400, json=lambda: payload)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', post)
    with pytest.raises(UpstreamModelError):
        bridge.complete({'messages': []})
    receipt = json.loads((bridge.output / 'diagnostics/context-window-error.json').read_text())
    assert receipt['binding'] == binding('a1')
    assert receipt['request_id'] == bridge.records[0]['request_id']
    assert receipt['model_calls'][0]['api_attempt_id'] == bridge.records[0]['model_calls'][0]['api_attempt_id']
    with pytest.raises(OpenHandsLimit):
        bridge.complete({'messages': [{'role': 'user', 'content': 'compress history'}]})
    assert len(posts) == 1
    assert bridge.limit_reason == 'sample_token_usage_unknown'
    status = _bridge_agent_status(bridge, 'OpenHandsError')
    assert status == 'ContextWindowExceededError'
    assert _map_evaluation({'error': 'ACP internal error'}, {'completed': False}, 'disabled',
                           agent_status=status) == (status, None, 'agent_failed_before_verification')
    assert disposition(bridge.cfg, status) == 'accepted'
    assert sample_accepted(bridge.cfg, {'exit_status': status})
    # The SkillsBench normal-result settlement writes accepted explicitly;
    # context termination must agree with the shared disposition contract.
    assert status in {'Submitted', 'AgentCompleted', 'AgentTimeout', 'ContextWindowExceededError'}
    assert bridge.token_budget.snapshot()['usage_unknown'] is True
    assert bridge.token_budget.snapshot()['completion_tokens'] == 0
    evidence = _terminal_evidence(bridge.output, agent_status=status,
                                  capture_status='disabled', evaluation_status='not_run')
    assert evidence['primary_cause'] == status
    assert evidence['cause_source'] == 'upstream_model_response'
    assert 'diagnostics/context-window-error.json' in [f['path'] for f in evidence['evidence_files']]
    # Finalization failures retain the agent reason while preserving their
    # independent failure disposition and partial capture evidence.
    result = backend_failure_result(SimpleNamespace(as_dict=lambda: {'sample_key': 'sample-a'}),
        {'attempt_id': 'a1'}, bridge.output, 'started', RuntimeError('capture failed'),
        bridge=bridge, capture_session='cap', agent_status='OpenHandsError')
    assert result['exit_status'] == result['agent_exit_status'] == status
    assert result['disposition'] == 'failed' and result['capture_status'] == 'partial'
    assert result['usage']['complete'] is False and result['resolved'] is None
    assert result['sample_token_budget']['usage_unknown'] is True


def test_successful_recovery_does_not_leave_a_context_terminal(tmp_path, monkeypatch):
    from rollout.adapters.openhands_bridge import UpstreamModelError
    from runtime.agents.skillsbench import _bridge_agent_status, _map_evaluation
    bridge = context_bridge(tmp_path, monkeypatch, limit=None)
    payload = {'error': {'code': 'context_length_exceeded'}}
    responses = iter([
        SimpleNamespace(ok=False, status_code=400, json=lambda: payload),
        SimpleNamespace(ok=True, status_code=200, json=lambda: {
            'id': 'recovered', 'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                          'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 1, 'prompt_tokens': 1}}),
        SimpleNamespace(ok=False, status_code=500, json=lambda: {'error': {'type': 'InternalServerError'}}),
    ])
    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', lambda *a, **kw: next(responses))
    with pytest.raises(UpstreamModelError):
        bridge.complete({'messages': []})
    assert bridge.context_window_error
    assert bridge.complete({'messages': []})['id'] == 'recovered'
    assert bridge.context_window_error is None
    with pytest.raises(UpstreamModelError):
        bridge.complete({'messages': []})
    assert _bridge_agent_status(bridge, 'OpenHandsError') == 'OpenHandsError'
    assert _map_evaluation({}, {'completed': True, 'rewards': {'reward': 1}}, 'disabled',
                           agent_status='ContextWindowExceededError')[0] == 'Submitted'


@pytest.mark.parametrize('second_status,expected', [
    (400, 'ContextWindowExceededError'), (500, 'OpenHandsError')])
def test_failed_parallel_decision_requires_all_candidates_context_rejected(
        tmp_path, monkeypatch, second_status, expected):
    from runtime.agents.skillsbench import _bridge_agent_status
    from rollout.adapters.openhands_bridge import UpstreamModelError
    bridge = context_bridge(tmp_path, monkeypatch, limit=None)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.active', lambda *a: True)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.request_role', lambda *a: 'agent_decision')
    statuses = iter([400, second_status])
    def post(*a, **kw):
        status = next(statuses)
        payload = {'error': {'code': 'context_length_exceeded'}} if status == 400 else {'error': {}}
        return SimpleNamespace(ok=False, status_code=status, json=lambda: payload)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', post)
    class Coordinator:
        def __init__(self, *a, **kw): pass
        def run(self, messages, metadata, generate, *a, **kw):
            for _ in range(2):
                try:
                    generate(messages, metadata, 1)
                except UpstreamModelError:
                    pass
            raise RuntimeError('All candidates failed')
    monkeypatch.setattr('rollout.adapters.openhands_bridge.DecisionCoordinator', Coordinator)
    with pytest.raises(RuntimeError, match='All candidates failed'):
        bridge.complete({'messages': []})
    assert len(bridge.records[0]['model_calls']) == 2
    assert _bridge_agent_status(bridge, 'OpenHandsError') == expected


@pytest.mark.parametrize('status', ['Cancelled', 'AgentTimeout', 'AgentCompleted', 'VerifierError'])
def test_context_evidence_does_not_override_independent_terminal(status):
    from runtime.agents.skillsbench import _bridge_agent_status
    bridge = SimpleNamespace(context_window_error={'exit_status': 'ContextWindowExceededError'},
                             limit_reason=None)
    assert _bridge_agent_status(bridge, status) == status
