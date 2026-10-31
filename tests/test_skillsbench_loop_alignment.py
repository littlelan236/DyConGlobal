import asyncio
import copy
from types import SimpleNamespace

import pytest

from core.deadline import Deadline
from core.io import read_json
from rollout.adapters.openhands import OpenHandsLimit, parse_response
from rollout.adapters.openhands_bridge import OpenHandsBridge
from rollout.step_loop import StepLoopDetector, repeated_suffix


@pytest.mark.parametrize('period', range(1, 8))
def test_shared_cross_step_detector_proves_exact_period_and_stops_before_action(period):
    detector = StepLoopDetector(8, 7)
    sequence = list(range(period)) * 8
    for i, fingerprint in enumerate(sequence):
        hit = detector.feed(fingerprint, match_policy='test')
        if i < len(sequence) - 1:
            assert hit is None
    assert hit['period'] == period and hit['repeats'] == 8
    assert hit['action_executed'] is False
    assert repeated_suffix(sequence, 8, 7, len(sequence)-1, match_policy='test') == hit


def test_unknown_tool_ids_do_not_change_cross_step_identity_or_execute_hit(cfg, tmp_path, monkeypatch):
    from rollout.adapters import openhands_bridge
    cfg['agent']['loop_repeat_limit'] = 3
    cfg['generation']['require_reasoning_close'] = False
    profile = {'request_roles': 'unqualified'}
    bridge = OpenHandsBridge(cfg, {'attempt_id': 'a'}, profile, tmp_path, Deadline(10))
    request = {'messages': [{'role': 'user', 'content': 'task'}], 'tools': [
        {'type': 'function', 'function': {'name': 'terminal', 'parameters': {}}}]}
    responses = []
    def generate(*args, **kwargs):
        raw = {'id': 'response-' + str(len(responses)), 'usage': {'completion_tokens': 5}, 'choices': [
            {'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None,
                'reasoning_content': 'same reasoning', 'tool_calls': [
                    {'id': 'fresh-transport-id-' + str(len(responses)), 'type': 'function',
                     'function': {'name': 'terminal', 'arguments': '{"command":"pwd"}'}}]}}]}
        responses.append(copy.deepcopy(raw))
        return SimpleNamespace(ok=True, json=lambda: raw)
    monkeypatch.setattr(openhands_bridge.requests, 'post', generate)
    assert bridge.complete(request) == responses[-1]
    assert bridge.complete(request) == responses[-1]
    with pytest.raises(OpenHandsLimit, match='Repeated reasoning-action'):
        bridge.complete(request)
    assert bridge.decisions == 2
    assert bridge.records[-1].get('selected_response') is None
    assert bridge.records[-1]['rejected_response'] == responses[-1]
    assert bridge.records[-1]['model_calls'][0]['response'] == responses[-1]
    assert read_json(tmp_path / 'loop-detection.json')['action_executed'] is False


@pytest.mark.parametrize('reason,status', [('length', 'LimitsExceeded'), ('abort', 'LimitsExceeded')])
def test_native_terminal_cannot_be_hidden_by_tool_parser_finish(reason, status):
    raw = {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant'}}],
           'dycon_native_finish_reason': reason}
    with pytest.raises(OpenHandsLimit) as value:
        parse_response(raw, {})
    assert value.value.messages[0]['extra']['exit_status'] == status


def test_condensation_owner_never_inherits_primary_capture_selector():
    from runtime.serving.vllm_plugin import DyConCapturePlugin, OWNER_CONTEXT, REQUEST_CONTEXT, STREAM
    plugin = DyConCapturePlugin()
    plugin.control_only = False
    def process():
        return SimpleNamespace(prompt_token_ids=[1, 2], sampling_params=None, capture_select=None)
    plugin.engine = SimpleNamespace(input_processor=SimpleNamespace(process_inputs=process),
                                    _add_request=lambda: None, generate=lambda: None)
    plugin._attach_admission()
    owner = OWNER_CONTEXT.set({'session_id': 'primary', 'request_role': 'condensation'})
    metadata = REQUEST_CONTEXT.set(None)
    try:
        request = plugin.engine.input_processor.process_inputs()
        assert request.capture_select[STREAM]['exclude_generation_window'] == [0, None]
        assert request.capture_select[STREAM]['exclude_prompt_window'] == [0, None]
    finally:
        OWNER_CONTEXT.reset(owner)
        REQUEST_CONTEXT.reset(metadata)
        for lane in (plugin.wait_lane, plugin.receipt_lane, plugin.publish_lane, plugin.control_lane, plugin.result_lane, plugin.failure_lane):
            lane.close()


@pytest.mark.parametrize('capture', [False, True])
def test_native_loop_guard_covers_openhands_baseline_with_or_without_capture(monkeypatch, tmp_path, capture):
    from runtime.serving.vllm_plugin import OWNER_CONTEXT, REQUEST_CONTEXT
    from tests.test_reasoning_loop_plugin import Params, plugin_fixture
    plugin, posts, _ = plugin_fixture(monkeypatch)
    plugin.scoring = None
    plugin.request_receipt_root = tmp_path / 'terminals'
    plugin.reasoning_loop_options = plugin.parallel_options['reasoning_loop']
    plugin.parallel_options = {}
    async def run():
        owner = {'run_id': 'r', 'session_id': 's', 'api_attempt_id': 'api', 'request_role': 'agent_decision'}
        token = OWNER_CONTEXT.set(owner)
        metadata = {'run_id': 'r', 'session_id': 's', 'step_id': 0, 'anchors': ['think_close']} if capture else None
        md = REQUEST_CONTEXT.set(metadata)
        try:
            return [output async for output in plugin.engine.generate({}, Params(), 'baseline')]
        finally:
            OWNER_CONTEXT.reset(token)
            REQUEST_CONTEXT.reset(md)
    try:
        outputs = asyncio.run(run())
        assert outputs[-1].outputs[0].finish_reason == 'abort'
        assert plugin.engine.aborts == ['baseline-native']
        terminal = read_json(next((tmp_path / 'terminals').glob('*.json')))
        assert terminal['reasoning_loop']['abort_completed'] is True
        assert terminal['native_request_id'] == 'baseline-native'
        assert len(terminal['output_token_ids']) < 256
    finally:
        for lane in (plugin.wait_lane, plugin.receipt_lane, plugin.publish_lane, plugin.control_lane, plugin.result_lane, plugin.failure_lane):
            lane.close()
