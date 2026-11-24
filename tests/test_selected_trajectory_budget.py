"""Selected-trajectory budgets keep baseline condensation and physical usage."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json

import pytest

from core.deadline import Deadline
from rollout.adapters.openhands import OpenHandsLimit, parse_response
from rollout.adapters.openhands_bridge import OpenHandsBridge
from rollout.sample_token_budget import BudgetStopped, SampleTokenBudget


def binding(attempt='a1'):
    return dict(run_id='run-a', task_key='task-a', sample_key='sample-a',
                repetition_id='rep-000', attempt_id=attempt)


def config(tmp_path, limit=10, parallel=True):
    return {'project_root': str(tmp_path), 'execution': {'sample_completion_token_budget': limit},
            'model': {'served_name': 'fixture'}, 'capture': {'enabled': False},
            'parallel': {'enabled': parallel},
            'serving': {'management': 'external', 'base_url': 'http://fixture/v1', 'api_key_env': 'FIXTURE_KEY'},
            'timeouts': {'model_seconds': 3}, 'agent': {'step_limit': 100, 'loop_repeat_limit': 8},
            'generation': {'seed': 1, 'preserve_thinking': False}}


def test_32_candidates_charge_one_winner_and_preserve_all_usage(tmp_path):
    ledger = SampleTokenBudget(config(tmp_path), binding())
    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(pool.map(lambda i: ledger.reserve(f'api{i}', 'agent_decision', 9, 'req1'), range(32)))
    assert allowed == [9] * 32
    for i in range(32):
        ledger.record(f'api{i}', 'agent_decision', {'completion_tokens': 6})
    assert ledger.remaining() == 10
    assert ledger.select('req1', 'api13')['completion_tokens'] == 6
    assert ledger.select('req1', 'api13')['completion_tokens'] == 6
    with pytest.raises(ValueError, match='Conflicting selected'):
        ledger.select('req1', 'api0')
    rebuilt = SampleTokenBudget(config(tmp_path), binding())
    row = rebuilt.snapshot()
    assert row['schema_version'] == 3
    assert sum(r['completion_tokens'] for r in row['requests'].values()) == 192
    assert rebuilt.remaining() == 4
    assert rebuilt.reserve('next', 'agent_decision', 9, 'req2') == 4
    assert SampleTokenBudget(config(tmp_path), binding('a2')).remaining() == 10


def test_unknown_loser_is_not_selected_unknown(tmp_path):
    ledger = SampleTokenBudget(config(tmp_path), binding())
    for api in ['unknown', 'winner']:
        ledger.reserve(api, 'agent_decision', 10, 'req1')
    ledger.record('unknown', 'agent_decision', {})
    ledger.record('winner', 'agent_decision', {'completion_tokens': 2})
    ledger.select('req1', 'winner')
    assert not ledger.snapshot().get('usage_unknown')
    ledger.reserve('selected-unknown', 'agent_decision', 8, 'req2')
    ledger.record_unknown('selected-unknown', 'agent_decision')
    with pytest.raises(BudgetStopped, match='sample_token_usage_unknown'):
        ledger.select('req2', 'selected-unknown')
    with pytest.raises(BudgetStopped, match='sample_token_usage_unknown'):
        ledger.reserve('blocked', 'condensation', 1, 'req3')


def test_condensation_matches_baseline_and_legacy_attempt_is_not_reinterpreted(tmp_path):
    for parallel in [False, True]:
        ledger = SampleTokenBudget(config(tmp_path / str(parallel), parallel=parallel), binding())
        ledger.reserve('api', 'condensation', 10, 'condense')
        ledger.record('api', 'condensation', {'completion_tokens': 3})
        ledger.reserve('agent', 'agent_decision', 10, 'req')
        ledger.record('agent', 'agent_decision', {'completion_tokens': 4})
        if parallel:
            ledger.select('req', 'agent')
        assert ledger.remaining() == 3
        assert ledger.snapshot()['by_role'] == {'agent_decision': 4, 'condensation': 3}
    old = SampleTokenBudget(config(tmp_path, parallel=False), binding())
    old.reserve('old', 'agent_decision', 10, 'oldreq')
    old.record('old', 'agent_decision', {'completion_tokens': 9})
    saved = old.path.read_bytes()
    with pytest.raises(ValueError, match='schema or scope'):
        SampleTokenBudget(config(tmp_path), binding()).snapshot()
    assert old.path.read_bytes() == saved


def bridge_fixture(tmp_path, monkeypatch, responses, *, parallel=True, limit=10):
    cfg = config(tmp_path, limit=limit, parallel=parallel)
    bridge = OpenHandsBridge(cfg, binding(), {'request_roles': 'openhands-1.22.1'},
                             tmp_path / 'out', Deadline(20))
    sent = []
    response_iter = iter(responses)
    def post(*args, **kwargs):
        sent.append(kwargs['json'])
        tokens, finish = next(response_iter)
        raw = {'id': 'response' + str(len(sent)), 'choices': [{'message': {
            'role': 'assistant', 'content': 'ok' + str(len(sent))}, 'finish_reason': finish}]}
        if finish == 'native_length':
            raw['choices'][0]['finish_reason'] = 'stop'
            raw['dycon_native_finish_reason'] = 'length'
        if tokens is not None:
            raw['usage'] = {'completion_tokens': tokens, 'prompt_tokens': 10000}
        return SimpleNamespace(ok=True, status_code=200, json=lambda: raw)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.requests.post', post)
    monkeypatch.setattr('rollout.adapters.openhands_bridge.request_role', lambda req, *a: req.get('fixture_role', 'agent_decision'))
    monkeypatch.setattr('rollout.adapters.openhands_bridge.active', lambda *a: parallel)
    monkeypatch.setattr('rollout.model.model_config', lambda *a, **kw: {'model_kwargs': {
        'temperature': 0, 'top_p': 1, 'max_tokens': 10, 'seed': 1, 'presence_penalty': 0, 'extra_body': {}}})
    return bridge, sent


def coordinator(monkeypatch, count, selected=0):
    class Coordinator:
        def __init__(self, *a, **kw): pass
        def run(self, messages, metadata, generate, parse, *a, **kw):
            candidates, errors = [], []
            for i in range(count):
                try:
                    candidates.append(generate(messages, metadata, i).model_dump())
                except OpenHandsLimit as exc:
                    errors.append(exc)
            if not candidates:
                raise errors[0]
            chosen = candidates[selected]
            message = parse(chosen)
            message['extra']['dycon_parallel'] = {}
            return message
    monkeypatch.setattr('rollout.adapters.openhands_bridge.DecisionCoordinator', Coordinator)


def test_bridge_selection_not_all_candidate_usage_and_condensation(tmp_path, monkeypatch):
    bridge, sent = bridge_fixture(tmp_path, monkeypatch, [(6, 'stop')] * 32 + [(2, 'stop'), (2, 'stop')])
    coordinator(monkeypatch, 32)
    bridge.complete({'messages': []})
    assert bridge.token_budget.snapshot()['completion_tokens'] == 6
    assert len(bridge.records[0]['model_calls']) == 32
    assert sum(c['usage']['completion_tokens'] for c in bridge.records[0]['model_calls']) == 192
    bridge.complete({'messages': [], 'fixture_role': 'condensation'})
    assert bridge.token_budget.remaining() == 2
    coordinator(monkeypatch, 1)
    with pytest.raises(OpenHandsLimit, match='sample_completion_token_budget'):
        bridge.complete({'messages': []})
    assert sent[-1]['max_tokens'] == 2
    receipt = json.loads((bridge.output / 'sample-token-budget.json').read_text())
    assert receipt['budget']['completion_tokens'] == 10
    assert receipt['budget']['by_role'] == {'agent_decision': 8, 'condensation': 2}
    assert not bridge.records[-1].get('selected_response')


def test_bridge_unknown_loser_and_unknown_winner(tmp_path, monkeypatch):
    bridge, _ = bridge_fixture(tmp_path, monkeypatch, [(None, 'stop'), (2, 'stop')])
    coordinator(monkeypatch, 2, selected=1)
    assert bridge.complete({'messages': []})['usage']['completion_tokens'] == 2
    assert not bridge.limit_reason
    other, _ = bridge_fixture(tmp_path / 'other', monkeypatch, [(None, 'stop'), (2, 'stop')])
    coordinator(monkeypatch, 2, selected=0)
    with pytest.raises(OpenHandsLimit, match='sample_token_usage_unknown'):
        other.complete({'messages': []})


@pytest.mark.parametrize('finish', ['length', 'native_length'])
def test_capped_losing_candidate_does_not_stop_shorter_winner(tmp_path, monkeypatch, finish):
    bridge, _ = bridge_fixture(tmp_path, monkeypatch, [(5, finish), (2, 'stop')], limit=5)
    coordinator(monkeypatch, 2)
    assert bridge.complete({'messages': []})['usage']['completion_tokens'] == 2
    assert bridge.token_budget.remaining() == 3
    assert bridge.limit_reason is None


def test_all_candidates_hit_remaining_budget_produces_budget_receipt(tmp_path, monkeypatch):
    bridge, _ = bridge_fixture(tmp_path, monkeypatch, [(5, 'length')] * 3, limit=5)
    coordinator(monkeypatch, 3)
    with pytest.raises(OpenHandsLimit, match='sample_completion_token_budget'):
        bridge.complete({'messages': []})
    assert bridge.limit_reason == 'sample_completion_token_budget'
    assert (bridge.output / 'sample-token-budget.json').exists()


def test_disabled_parallel_budget_does_not_select_or_fail(tmp_path, monkeypatch):
    bridge, _ = bridge_fixture(tmp_path, monkeypatch, [(6, 'stop')] * 3, limit=None)
    coordinator(monkeypatch, 3)
    assert bridge.complete({'messages': []})['usage']['completion_tokens'] == 6
    assert bridge.token_budget.snapshot() is None
