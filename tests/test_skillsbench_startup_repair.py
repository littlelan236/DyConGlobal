"""Regression at real shell/HTTP/native receipt boundaries (no GPU required)."""
import asyncio
import copy
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from application.execution.subject import export_workspace
from core.config import Generation, validate_generation_contract
from core.deadline import Deadline
from core.io import read_json
from hidden.backends.vllm.online import capture_request_metadata
from rollout.adapters.openhands_bridge import OpenHandsBridge
from runtime.serving.capture import OnlineCaptureController


class ShellEnvironment:
    async def exec(self, command, **kwargs):
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def download_file(self, source, target):
        Path(target).write_bytes(Path(source).read_bytes())


@pytest.mark.parametrize('kind', ['file', 'directory', 'missing'])
def test_real_output_probe_preserves_kind_mode_separator_and_shell_paths(tmp_path, kind):
    path = tmp_path / "output with 'quotes' ; literal $(touch unwanted)"
    if kind == 'file':
        path.write_text('payload')
    elif kind == 'directory':
        path.mkdir()
        (path / 'child.txt').write_text('payload')
    result = asyncio.run(export_workspace(ShellEnvironment(), [str(path)], tmp_path / 'subject', {}, 'a' * 64, {}))
    value = read_json(result)
    assert value['complete'] == (kind != 'missing')
    if kind != 'missing':
        assert value['outputs'][0]['kind'] == ('directory_archive' if kind == 'directory' else 'file')
        probe = read_json(result.parent / 'probes/000.json')
        assert probe['stdout'].splitlines()[0] == kind
        assert probe['stdout'].splitlines()[1].isdigit()
    assert not (tmp_path / 'unwanted').exists()


@pytest.mark.parametrize('budget', [98304, 100000, 81920])
def test_generation_rejects_equal_or_excess_reasoning_budget(cfg, budget):
    generation = {**cfg['generation'], 'max_tokens': 98304, 'thinking_token_budget': budget}
    if budget >= 98304:
        with pytest.raises(ValueError, match='smaller than max_tokens'):
            Generation.model_validate(generation)
    else:
        assert Generation.model_validate(generation).thinking_token_budget == 81920


def test_frozen_context_cannot_mix_old_and_new_generation(cfg):
    cfg['execution_context'] = {'generation': {**cfg['generation'], 'max_tokens': 1}}
    with pytest.raises(ValueError, match='execution_context.generation'):
        validate_generation_contract(cfg)


def test_stop_drains_already_bound_output_without_reopening(tmp_path):
    controller = OnlineCaptureController(tmp_path, {'export': {'root': 'online'}, 'finalize_seconds': 2})
    md = capture_request_metadata('s', 'task', 'run', 'pending', 0, ['think_close'])
    controller.start(md)
    controller.bind('native', {**md, 'prompt_token_ids': [10]})
    ready = threading.Event()
    original_wait = controller._output_condition.wait_for
    def waiting(*args, **kwargs):
        ready.set()
        return original_wait(*args, **kwargs)
    controller._output_condition.wait_for = waiting
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(controller.stop, 's')
            assert ready.wait(1)
            assert controller.sessions['s'].status == 'draining'
            with pytest.raises(KeyError):
                controller.bind('unexpected', {**md, 'prompt_token_ids': [10]})
            controller.record_output({'request_id': 'native', 'prompt_token_ids': [10], 'output_token_ids': [20]})
            assert future.result(timeout=2)['status'] == 'stopped'
        assert controller.stop('s')['status'] == 'stopped'
        assert controller.sessions['s'].status == 'stopped'
    finally:
        controller.close()


@pytest.mark.parametrize('error,expected', [('length', 'limit'), ('undeclared', 'protocol')])
def test_bridge_returns_nonretryable_structured_error_and_preserves_cost(cfg, tmp_path, monkeypatch, error, expected):
    from rollout.adapters import openhands_bridge
    cfg['serving']['management'] = 'external'
    cfg['capture']['enabled'] = False
    raw = {'id': 'native-output', 'usage': {'prompt_tokens': 7, 'completion_tokens': 23},
           'choices': [{'finish_reason': 'length' if error == 'length' else 'tool_calls',
                        'message': {'role': 'assistant', 'content': None, 'tool_calls': []}}]}
    if error == 'undeclared':
        raw['choices'][0]['message']['tool_calls'] = [{'id': 'one', 'type': 'function',
            'function': {'name': 'not_offered', 'arguments': '{}'}}]
    calls = []
    actual_post = requests.post
    def post(url, **kwargs):
        if url.endswith('/chat/completions') and not url.startswith(bridge.base_url):
            calls.append(url)
            return SimpleNamespace(ok=True, json=lambda: copy.deepcopy(raw))
        return actual_post(url, **kwargs)
    profile = {'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1', 'request_roles': 'unqualified'}
    bridge = OpenHandsBridge(cfg, {'attempt_id': 'a'}, profile, tmp_path, Deadline(10))
    monkeypatch.setattr(openhands_bridge.requests, 'post', post)
    with bridge:
        response = post(bridge.base_url + '/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'task'}],
            'tools': [{'type': 'function', 'function': {'name': 'terminal', 'parameters': {}}}]},
            headers={'Authorization': 'Bearer ' + bridge.secret}, timeout=3)
        assert response.status_code == 400
        assert response.json()['error']['type'] == expected
    assert len(calls) == 1
    terminal = read_json(next((tmp_path / 'terminal-receipts').glob('*.json')))
    assert terminal['reason'] == expected
    assert terminal['model_call_receipts'][0]['usage']['completion_tokens'] == 23


def test_session_cancellation_keeps_other_session_and_scoring_alive(monkeypatch):
    from runtime.serving.vllm_plugin import DyConCapturePlugin
    monkeypatch.setenv('DYCON_CAPTURE_EXPORT_OPTIONS', '{}')
    plugin = DyConCapturePlugin()
    aborted, decisions = [], []
    async def abort(key, **kwargs):
        aborted.append(key)
        plugin.native_requests.pop(key)
        plugin.native_changed.set()
    plugin.engine = SimpleNamespace(abort=abort)
    plugin.scoring = SimpleNamespace(cancel=lambda key: decisions.append(key))
    plugin.native_requests = {
        'owned': {'metadata': {'session_id': 'a', 'decision_id': 'd-a'}, 'phase': 'running'},
        'other': {'metadata': {'session_id': 'b', 'decision_id': 'd-b'}, 'phase': 'running'}}
    try:
        result = asyncio.run(plugin.cancel_native(session_id='a', wait_seconds=1))
        assert result['abort_completed']
        assert aborted == ['owned'] and decisions == ['d-a']
        assert 'other' in plugin.native_requests and not plugin.cancelled_run
        plugin._check_cancelled({'session_id': 'b', 'decision_id': 'd-b'})
    finally:
        for lane in (plugin.wait_lane, plugin.receipt_lane, plugin.publish_lane, plugin.control_lane, plugin.result_lane, plugin.failure_lane):
            lane.close()


def test_step_coverage_checks_all_steps_even_when_totals_match():
    from runtime.agents.validation import audit_step_coverage
    requests = [{'request_id': str(i), 'prompt_token_ids': [10],
                 'output_token_ids': [20, 99, 30], 'close_token_ids': [99]} for i in range(3)]
    def point(rid, sample):
        return {'request_id': rid, 'sample_id': sample, 'token_id': 99,
                'absolute_position': 2, 'graph_execution': 'full'}
    points = [point('0', 'a'), point('0', 'b'), point('2', 'c')]
    result = audit_step_coverage(requests, points)
    assert not result['verified'] and result['expected_steps'] == len(points)
    assert result['missing'] == ['0', '1'] and result['duplicate'] == ['0']
    result = audit_step_coverage(requests, [point(str(i), str(i)) for i in range(3)])
    assert result['verified'] and result['verified_steps'] == 3
    requests[-1]['output_token_ids'] = [20, 99]
    result = audit_step_coverage(requests, [point(str(i), str(i)) for i in range(3)])
    assert not result['verified'] and result['unfinished'] == ['2']


def test_capture_failure_keeps_official_reward_and_all_model_costs(tmp_path):
    from runtime.agents.skillsbench import backend_failure_result
    from core.io import write_json
    write_json(tmp_path / 'evaluation-raw.json', {'completed': True, 'rewards': {'reward': 1}})
    bridge = SimpleNamespace(records=[{'model_calls': [
        {'usage_known': True, 'usage': {'prompt_tokens': 50, 'completion_tokens': 11}},
        {'usage_known': False, 'status': 'failed'}]}])
    sample = SimpleNamespace(as_dict=lambda: {'sample_key': 'sample'})
    result = backend_failure_result(sample, {'attempt_id': 'a'}, tmp_path, 'started',
        RuntimeError('native drain failed'), bridge=bridge, capture_session='cap', agent_status='AgentCompleted')
    assert result['exit_status'] == 'CaptureError' and result['agent_exit_status'] == 'AgentCompleted'
    assert result['resolved'] is True and result['capture_status'] == 'partial'
    assert result['usage']['completion_tokens'] == 11 and not result['usage']['complete']
    assert result['disposition'] == 'failed'


def test_cancelled_attempt_can_retain_native_usage_without_publishing_a_failure(tmp_path):
    from runtime.agents.skillsbench import backend_failure_result
    bridge = SimpleNamespace(records=[{'model_calls': [{'usage_known': True,
        'usage': {'prompt_tokens': 100, 'completion_tokens': 8}}]}])
    result = backend_failure_result(SimpleNamespace(as_dict=lambda: {}), {}, tmp_path, 'start',
        RuntimeError('cancelled'), bridge=bridge, capture_session='s', persist=False)
    assert result['usage']['completion_tokens'] == 8 and result['usage']['complete']
    assert not (tmp_path / 'attempt.json').exists()
