"""Exact admission identity and terminal receipt lifecycle contracts."""
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from hidden.backends.vllm.online import capture_request_metadata
from runtime.serving.capture import CaptureControlServer, OnlineCaptureController


@pytest.fixture
def capture(tmp_path):
    controller = OnlineCaptureController(tmp_path, {"export": {"root": "online"}})
    metadata = capture_request_metadata("session", "task", "run", "pending", 0, ["think_close"])
    controller.start(metadata)
    yield controller, metadata
    controller.close()


def branch(metadata, number, count=4, decision="dec-test"):
    return {**metadata, "decision_id": decision, "branch_id": number, "branch_count": count,
            "api_attempt_id": f"attempt-{number}", "query_id": "query", "prompt_token_ids": [10, 20],
            "decision_deadline_unix": time.time() + 30}


def output(native, metadata):
    return {"request_id": native, "metadata": metadata, "prompt_token_ids": [10, 20],
            "output_token_ids": [30, 40], "finish_reason": "stop"}


def decision(metadata, branches):
    return {"session_id": metadata["session_id"], "step_id": metadata["step_id"],
            "decision_id": branches[0]["decision_id"], "branch_count": len(branches),
            "decision_deadline_unix": branches[0]["decision_deadline_unix"], "selected_branch_id": 0,
            "candidates": [{"branch_id": row["branch_id"], "api_attempt_id": row["api_attempt_id"],
                            "response": {"id": f"external-{row['branch_id']}"}} for row in branches]}


def branches(metadata, count=4, name="dec-test"):
    rows = [branch(metadata, i, count, name) for i in range(count)]
    deadline = rows[0]["decision_deadline_unix"]
    for row in rows:
        row["decision_deadline_unix"] = deadline
    return rows


@pytest.mark.parametrize("count", [4, 32])
def test_out_of_order_concurrent_http_receipts_are_exact_and_idempotent(capture, count):
    controller, metadata = capture
    rows = branches(metadata, count)
    server = CaptureControlServer(controller).start()
    barrier = threading.Barrier(count)

    def one(row):
        barrier.wait(timeout=10)
        native = f"opaque-native-{row['branch_id']}"
        response = requests.post(server.base_url + "/dycon/capture/request/bind",
                                 json={"request_id": native, "metadata": row}, timeout=15)
        response.raise_for_status()
        for _ in range(2):
            response = requests.post(server.base_url + "/dycon/capture/request/output",
                                     json=output(native, row), timeout=15)
            response.raise_for_status()
        assert response.json()["idempotent"]

    try:
        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(one, reversed(rows)))
        saved = controller.record_decision(decision(metadata, rows), timeout=1)
        assert len(controller.request_outputs) == count
        assert len({r["native_request_id"] for r in saved["candidates"]}) == count
        assert all(r["ledger_ref"] and r["ledger_sha256"] and r["output_sha256"] for r in saved["candidates"])
        assert len(list((controller.export_root / 'sessions/session/requests').glob('*.json'))) == count
    finally:
        server.close()


@pytest.mark.parametrize("bound,expected", [(False, "no_receipt"), (True, "receipt_delayed")])
def test_missing_and_delayed_receipts_keep_raw_response_and_diagnostics(capture, bound, expected):
    controller, metadata = capture
    rows = branches(metadata, 1)
    if bound:
        controller.bind("native", rows[0])
    with pytest.raises(ValueError, match=expected):
        controller.record_decision(decision(metadata, rows), timeout=0)
    artifact = json.loads((controller.export_root / 'sessions/session/parallel/failed-dec-test.json').read_text())
    assert artifact['candidates'][0]['response'] == {'id': 'external-0'}
    diag = artifact['receipt_diagnostics']
    assert diag['branches'][0]['receipt_count'] == 0
    assert diag['branches'][0]['status'] == expected
    assert diag['reason'] == expected


def test_short_external_id_never_establishes_native_binding(capture):
    controller, metadata = capture
    row = branches(metadata, 1)[0]
    controller.bind('short-randomized', row)
    with pytest.raises(KeyError, match='exact native'):
        controller.record_output(output('short', row))


@pytest.mark.parametrize('conflict', ['session_id', 'step_id', 'decision_id', 'branch_id', 'api_attempt_id'])
def test_output_cannot_override_bound_identity(capture, conflict):
    controller, metadata = capture
    row = branches(metadata, 1)[0]
    controller.bind('native', row)
    value = output('native', row)
    value['metadata'] = {**row, conflict: 'wrong'}
    with pytest.raises(ValueError, match='identity differs'):
        controller.record_output(value)
    assert not controller.request_outputs
    assert controller.decision_failures[('session', 'dec-test')]['reason'] == 'output_identity_conflict'


def test_conflicting_retry_does_not_overwrite_the_first_ledger(capture):
    controller, metadata = capture
    rows = branches(metadata, 1)
    controller.bind('native', rows[0])
    controller.record_output(output('native', rows[0]))
    original = copy.deepcopy(controller.request_outputs['native'])
    value = {**output('native', rows[0]), 'output_token_ids': [999]}
    with pytest.raises(ValueError, match='Conflicting native output receipt'):
        controller.record_output(value)
    assert controller.request_outputs['native'] == original
    with pytest.raises(ValueError, match='conflicting_output_receipt'):
        controller.record_decision(decision(metadata, rows), timeout=0)


def test_two_branches_cannot_share_a_native_request(capture):
    controller, metadata = capture
    rows = branches(metadata, 2)
    controller.bind('native', rows[0])
    with pytest.raises(ValueError, match='conflicting capture bindings'):
        controller.bind('native', rows[1])
    assert controller.decision_failures[('session', 'dec-test')]['reason'] == 'native_request_binding_conflict'


def test_one_attempt_cannot_bind_two_native_requests(capture):
    controller, metadata = capture
    rows = branches(metadata, 1)
    controller.bind('native', rows[0])
    with pytest.raises(ValueError, match='multiple native request'):
        controller.bind('other-native', rows[0])
    assert controller.decision_failures[('session', 'dec-test')]['reason'] == 'duplicate_attempt_binding'


def test_duplicate_receipt_is_diagnosed_even_for_corrupted_input(capture):
    controller, metadata = capture
    rows = branches(metadata, 1)
    controller.bind('native', rows[0])
    controller.record_output(output('native', rows[0]))
    controller.request_outputs['corrupt-duplicate'] = {**controller.request_outputs['native'], 'request_id': 'other'}
    controller.attempt_outputs[controller._attempt_key(controller.request_bindings['native'][1])].add('corrupt-duplicate')
    with pytest.raises(ValueError, match='duplicate_receipt'):
        controller.record_decision(decision(metadata, rows), timeout=0)


def test_reset_late_output_cannot_contaminate_the_next_decision(capture):
    controller, metadata = capture
    old = branches(metadata, 1, 'dec-old')
    controller.bind('old-native', old[0])
    controller.failure('session', 'control connection reset', decision_id='dec-old')
    with pytest.raises(ValueError, match='capture_control_unavailable'):
        controller.record_decision(decision(metadata, old), timeout=0)
    new = branches(metadata, 1, 'dec-new')
    controller.bind('new-native', new[0])
    with pytest.raises(ValueError, match='terminal'):
        controller.record_output(output('old-native', old[0]))
    controller.record_output(output('new-native', new[0]))
    saved = controller.record_decision(decision(metadata, new), timeout=0)
    assert saved['candidates'][0]['native_request_id'] == 'new-native'


def test_expired_decision_rejects_bind_and_keeps_failure(capture):
    controller, metadata = capture
    row = {**branches(metadata, 1)[0], 'decision_deadline_unix': time.time() - 1}
    with pytest.raises(ValueError, match='deadline exceeded'):
        controller.bind('native', row)
    assert controller.decision_failures[('session', 'dec-test')]['reason'] == 'decision_deadline_exceeded'


def test_event_wait_resumes_when_delayed_output_arrives(capture):
    controller, metadata = capture
    rows = branches(metadata, 1)
    controller.bind('native', rows[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.record_decision, decision(metadata, rows), 2)
        controller.record_output(output('native', rows[0]))
        assert future.result(timeout=3)['candidates'][0]['native_request_id'] == 'native'


def test_scoring_identity_must_match_the_receipt_attempt(capture):
    controller, metadata = capture
    rows = branches(metadata, 1)
    controller.bind('native', rows[0])
    controller.record_output(output('native', rows[0]))
    value = decision(metadata, rows)
    value['candidates'][0]['score_identity'] = {'api_attempt_id': 'another-attempt'}
    with pytest.raises(ValueError, match='score_identity_conflict'):
        controller.record_decision(value, timeout=0)


def test_expired_broker_does_not_admit_a_branch_or_return_a_score(tmp_path, monkeypatch):
    from test_parallel_rollout import provider

    from runtime.serving.parallel import ScoringBroker

    broker = ScoringBroker(provider(tmp_path, monkeypatch), 'model', 2)
    try:
        with pytest.raises(ValueError, match='deadline exceeded'):
            broker.reserve('expired', [0], time.time() - 1)
        broker.reserve('decision', [0], time.time() + 5)
        broker.groups['decision']['expires'] = time.monotonic() - 1
        with pytest.raises(RuntimeError, match='deadline exceeded'):
            broker.acquire_branch('decision', 0)
        with pytest.raises(ValueError, match='deadline exceeded'):
            broker.results('decision', 0)
    finally:
        broker.close()


def test_spool_reservation_failure_keeps_raw_decision_in_run_diagnostics(capture, monkeypatch):
    controller, metadata = capture
    rows = branches(metadata, 1)

    def exhausted(*args, **kwargs):
        raise OSError('capture disk reservation exhausted')

    monkeypatch.setattr(controller, '_write', exhausted)
    with pytest.raises(ValueError, match='no_receipt'):
        controller.record_decision(decision(metadata, rows), timeout=0)
    path = controller.root / 'runs/standalone/capture-diagnostics/sessions/session/parallel/failed-dec-test.json'
    value = json.loads(path.read_text())
    assert 'reservation exhausted' in value['capture_artifact_persistence_error']
    assert value['evidence']['candidates'][0]['response'] == {'id': 'external-0'}
    assert value['evidence']['receipt_diagnostics']['reason'] == 'no_receipt'
    assert not controller.request_outputs


def test_output_conflict_diagnostic_survives_spool_write_failure(capture, monkeypatch):
    controller, metadata = capture
    row = branches(metadata, 1)[0]
    controller.bind('native', row)
    controller.record_output(output('native', row))

    def exhausted(*args, **kwargs):
        raise OSError('capture spool full')

    monkeypatch.setattr(controller, '_write', exhausted)
    raw = {**output('native', row), 'output_token_ids': [999]}
    with pytest.raises(ValueError, match='Conflicting native output receipt'):
        controller.record_output(raw)
    paths = list((controller.root / 'runs/standalone/capture-diagnostics').rglob('*.json'))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())['evidence']['raw_output'] == raw
    assert controller.request_outputs['native']['output_token_ids'] == [30, 40]


@pytest.mark.parametrize('failed_decision', ['dec-test', 'dec-previous'])
def test_persisted_control_failure_blocks_only_its_own_decision(capture, failed_decision):
    controller, metadata = capture
    rows = branches(metadata, 2)
    controller.bind('native-0', rows[0])
    controller.record_output(output('native-0', rows[0]))
    value = decision(metadata, rows)
    value['candidates'][1]['response'] = None
    path = controller.export_root / 'failures/control-reset.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'session_id': 'session', 'decision_id': failed_decision,
                                'reason': 'control connection reset before failure callback could be acknowledged'}))
    if failed_decision == 'dec-test':
        with pytest.raises(ValueError, match='capture_control_unavailable'):
            controller.record_decision(value, timeout=0)
    else:
        saved = controller.record_decision(value, timeout=0)
        assert saved['candidates'][0]['native_request_id'] == 'native-0'
