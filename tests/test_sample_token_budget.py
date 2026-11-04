from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.deadline import Deadline
from rollout.adapters.openhands import OpenHandsLimit
from rollout.adapters.openhands_bridge import OpenHandsBridge
from rollout.retry import disposition
from rollout.sample_token_budget import BudgetStopped, SampleTokenBudget


def binding(attempt):
    return {'run_id': 'run-a', 'task_key': 'task-a', 'sample_key': 'sample-a',
            'repetition_id': 'rep-000', 'attempt_id': attempt}


def test_roles_and_seed_attempts_share_durable_idempotent_total(tmp_path):
    cfg = {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': 10}}
    first = SampleTokenBudget(cfg, binding('a1'))
    assert first.reserve('api1', 'agent_decision', 10) == 10
    first.record('api1', 'agent_decision', {'completion_tokens': 4})
    first.record('api1', 'agent_decision', {'completion_tokens': 4})
    retry = SampleTokenBudget(cfg, binding('a2'))
    assert retry.reserve('api2', 'condensation', 10) == 6
    retry.record('api2', 'condensation', {'completion_tokens': 3})
    assert retry.remaining() == 3
    assert retry.reserve('api3', 'agent_decision', 10) == 3
    retry.record('api3', 'agent_decision', {'completion_tokens': 3})
    with pytest.raises(BudgetStopped, match='sample_completion_token_budget'):
        retry.reserve('api4', 'agent_decision', 10)
    assert retry.snapshot()['by_role'] == {'agent_decision': 7, 'condensation': 3}


def test_parallel_reservations_cannot_overcommit(tmp_path):
    cfg = {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': 100}}
    ledger = SampleTokenBudget(cfg, binding('a1'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(pool.map(lambda i: ledger.reserve(f'api{i}', 'agent_decision', 30), range(8)))
    assert sum(x for x in allowed if x is not None) == 100
    pending = ledger.snapshot()['requests']
    for api, row in pending.items():
        ledger.record(api, row['role'], {'completion_tokens': 1})
    assert ledger.remaining() == 96


@pytest.mark.parametrize('unknown', [False, True])
def test_unfinished_or_unknown_usage_blocks_new_attempt(tmp_path, unknown):
    cfg = {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': 20}}
    first = SampleTokenBudget(cfg, binding('a1'))
    first.reserve('api1', 'condensation', 10)
    if unknown:
        first.record_unknown('api1', 'condensation')
    retry = SampleTokenBudget(cfg, binding('a2'))
    with pytest.raises(BudgetStopped, match='sample_token_usage_unknown'):
        retry.reserve('api2', 'agent_decision', 10)


def test_disabled_budget_keeps_legacy_behavior():
    ledger = SampleTokenBudget({'execution': {}}, {})
    assert ledger.reserve('api', 'unqualified', 55) == 55
    assert ledger.snapshot() is None


def test_budget_rejects_nonpositive_limits_and_changed_frozen_limit(tmp_path):
    from pydantic import ValidationError

    from core.config import Execution
    for limit in [0, -1, True]:
        with pytest.raises(ValidationError):
            Execution.model_validate({'rollout_workers': 1, 'harness_workers': 1,
                'infrastructure_retry_limit': 0, 'pipeline_wall_seconds': 0,
                'sample_completion_token_budget': limit})
    cfg = {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': 10}}
    first = SampleTokenBudget(cfg, binding('a1'))
    first.reserve('api', 'condensation', 5)
    cfg['execution']['sample_completion_token_budget'] = 20
    with pytest.raises(ValueError, match='frozen limit'):
        SampleTokenBudget(cfg, binding('a2')).snapshot()


@pytest.mark.parametrize('missing', [False, True])
def test_bridge_caps_request_stops_without_seed_retry_and_keeps_failure_cause(cfg, tmp_path, monkeypatch, missing):
    cfg['execution']['sample_completion_token_budget'] = 3
    cfg['capture']['enabled'] = False
    cfg['parallel'] = {'enabled': False}
    cfg['serving']['management'] = 'external'
    profile = {'request_roles': 'unqualified', 'cleanup_seconds': 1}
    bridge = OpenHandsBridge(cfg, binding('a1'), profile, tmp_path / 'out', Deadline(10))
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs['json'])
        raw = {'id': 'response1', 'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                  'finish_reason': 'stop'}]}
        if not missing:
            raw['usage'] = {'prompt_tokens': 5, 'completion_tokens': 3}
        return SimpleNamespace(ok=True, status_code=200, json=lambda: raw)

    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', post)
    with pytest.raises(OpenHandsLimit):
        bridge.complete({'messages': []})
    assert len(calls) == 1 and calls[0]['max_tokens'] == 3
    with pytest.raises(OpenHandsLimit):
        bridge.complete({'messages': []})
    assert len(calls) == 1
    reason = 'sample_token_usage_unknown' if missing else 'sample_completion_token_budget'
    assert bridge.limit_reason == reason
    status = 'SampleTokenBudgetUsageUnknown' if missing else 'SampleTokenBudgetExceeded'
    cfg['execution']['limits_retry'] = {'enabled': True, 'seed_policy': 'repetition_round_offset_v1',
                                      'retry_statuses': ['LimitsExceeded', 'reasoning_loop']}
    assert disposition(cfg, status) == 'failed'
    from runtime.agents.skillsbench import backend_failure_result
    sample = SimpleNamespace(as_dict=lambda: {'sample_key': 'sample-a'})
    result = backend_failure_result(sample, binding('a1'), tmp_path / 'out', 'start',
        TimeoutError('capture stop timeout'), bridge=bridge, capture_session='session', persist=False)
    assert result['exit_status'] == status
    assert result['usage']['completion_tokens'] == (0 if missing else 3)
