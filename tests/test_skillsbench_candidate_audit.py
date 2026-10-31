from types import SimpleNamespace

import numpy as np
import pytest

from core.io import hash_json, sha256, write_json
from runtime.agents.validation import audit_candidate


def test_terminal_api_failures_are_counted_separately_from_missing_receipts(tmp_path):
    from runtime.agents.validation import audit_api_terminals
    request = {'request_id': 'req', 'binding': {'attempt': 'a'}, 'status': 'failed',
               'finished_at': 'finished', 'role': 'agent_decision', 'step_id': 2,
               'error': 'reasoning_loop', 'model_calls': []}
    write_json(tmp_path / 'requests/req-req.json', request)
    missing = audit_api_terminals(tmp_path)
    assert missing['missing'] == ['req'] and not missing['verified']
    write_json(tmp_path / 'terminal-receipts/req.json', {
        'request_id': 'req', 'binding': request['binding'], 'status': 'failed'})
    valid = audit_api_terminals(tmp_path)
    assert valid['verified'] and valid['failed_or_cancelled_api_attempts'][0]['error'] == 'reasoning_loop'
    write_json(tmp_path / 'terminal-receipts/req.json', {
        'request_id': 'req', 'binding': {'attempt': 'other'}, 'status': 'failed'})
    with pytest.raises(ValueError, match='binding'):
        audit_api_terminals(tmp_path)


@pytest.mark.parametrize('tamper', [None, 'cost', 'score', 'tensor', 'identity'])
def test_all_candidate_audit_rechecks_native_and_published_prediction(tmp_path, monkeypatch, tamper):
    expected = {'run_id': 'run', 'session_id': 's', 'decision_id': 'd', 'branch_id': 1, 'api_attempt_id': 'a'}
    ledger = {**expected, 'request_id': 'native', 'prompt_token_ids': [10],
              'output_token_ids': [20, 11, 30], 'request_sha256': hash_json([10])}
    write_json(tmp_path / 'ledger.json', ledger)
    candidate = {'ledger_ref': 'ledger.json', 'ledger_sha256': sha256(tmp_path / 'ledger.json'),
        'native_request_id': 'native', 'output_sha256': hash_json([20, 11, 30]),
        'completion_tokens': 3, 'close_sample_id': 'point', 'score': 3., 'status': 'eligible'}
    branch = {'close_events': [{'sample_id': 'point', 'request_id': 'native'}]}
    arrays = {'layer_0': np.array([1., 2.])}
    if tamper == 'cost':candidate['completion_tokens'] += 1
    if tamper == 'score':candidate['score'] += 1
    if tamper == 'tensor':arrays['layer_0'][0] = np.nan
    if tamper == 'identity':expected['branch_id'] = 0
    monkeypatch.setattr('hidden.artifacts.read_arrays', lambda *args: arrays)
    predictor = SimpleNamespace(predict=lambda batch: batch.sum(axis=(1, 2)))
    arguments = (tmp_path, candidate, branch, expected, {'sites': [0], 'hidden_size': 2}, predictor, 2)
    if tamper is None:
        assert audit_candidate(*arguments, capture_layers=[0]) == 0.
    else:
        with pytest.raises(ValueError):
            audit_candidate(*arguments, capture_layers=[0])
