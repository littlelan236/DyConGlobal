import socket
import threading
import time

import requests

from core.deadline import Deadline
from rollout.adapters import openhands_bridge as bridge_module
from rollout.adapters.openhands_bridge import OpenHandsBridge


class _Response:
    ok = True
    status_code = 200

    def json(self):
        return {"id": "upstream-1", "choices": [{"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_completed_response_eof_does_not_cancel_attempt(cfg, tmp_path, monkeypatch):
    cfg["serving"]["base_url"] = "http://model.invalid/v1"
    profile = {"bridge_listen": "127.0.0.1", "bridge_advertise": "127.0.0.1",
               "request_roles": "unqualified", "cleanup_seconds": 2}
    bridge = OpenHandsBridge(cfg, {"attempt_id": "a"}, profile, tmp_path, Deadline(5))
    original_post = requests.post

    def upstream_or_bridge(url, *args, **kwargs):
        if url.startswith(bridge.base_url):
            return original_post(url, *args, **kwargs)
        return _Response()

    monkeypatch.setattr(bridge_module.requests, "post", upstream_or_bridge)
    headers = {"Authorization": "Bearer " + bridge.secret}
    with bridge:
        for _ in range(2):
            response = original_post(bridge.base_url + "/chat/completions", json={"messages": []},
                                     headers=headers, timeout=3)
            assert response.status_code == 200
            time.sleep(0.35)
            assert not bridge.closed.is_set()
    diagnostics = list((tmp_path / "diagnostics").glob("request-lifecycle-*.json"))
    assert len(diagnostics) == 2
    assert all(any(event["event"] == "response_write_finished" for event in
                   __import__("json").loads(path.read_text())["events"]) for path in diagnostics)


def test_peer_watcher_stops_before_reading_closed_socket(cfg, tmp_path, monkeypatch):
    cfg["serving"]["base_url"] = "http://model.invalid/v1"
    profile = {"bridge_listen": "127.0.0.1", "bridge_advertise": "127.0.0.1",
               "request_roles": "unqualified", "cleanup_seconds": 2}
    bridge = OpenHandsBridge(cfg, {"attempt_id": "a"}, profile, tmp_path, Deadline(5))
    left, right = socket.socketpair()
    state = {"done": threading.Event(), "lock": threading.Lock(),
             "handler_terminal": False, "response_write_finished": False,
             "upstream_active": 1, "result_ready": False, "peer_cancel_claimed": False}
    left.close()
    peer_events = []
    monkeypatch.setattr(bridge, "_peer_lost", lambda state, event, error_type=None:
                        peer_events.append((event, error_type)))
    monkeypatch.setattr(bridge_module.select, "select",
                        lambda *args: (_ for _ in ()).throw(AssertionError("closed socket selected")))

    bridge._watch_peer(left, state)

    right.close()
    assert peer_events == [("peer_socket_closed", None)]
    assert not bridge.closed.is_set()


def test_concurrent_peer_disconnect_and_socket_close_leave_no_watcher_tracebacks(cfg, tmp_path, monkeypatch):
    profile = {'request_roles': 'unqualified', 'cleanup_seconds': 1}
    bridge = OpenHandsBridge(cfg, {'attempt_id': 'a'}, profile, tmp_path, Deadline(20))
    errors = []
    events = []
    monkeypatch.setattr(bridge, '_peer_lost', lambda state, event, error_type=None: events.append(event))
    def watch(sock, state):
        try:
            bridge._watch_peer(sock, state)
        except BaseException as exc:
            errors.append(exc)
    for i in range(64):
        left, right = socket.socketpair()
        state = {'done': threading.Event(), 'lock': threading.Lock(),
                 'handler_terminal': False, 'response_write_finished': False}
        worker = threading.Thread(target=watch, args=(left, state))
        worker.start()
        if i % 2:
            left.close()
        else:
            right.close()
        worker.join(timeout=.6)
        state['done'].set()
        worker.join(timeout=1)
        left.close()
        right.close()
        assert not worker.is_alive()
    assert not errors
    assert len(events) == 64
