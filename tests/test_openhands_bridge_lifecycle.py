import errno
import socket
import threading
import time

import pytest
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
    class ClosedSocketSelector(bridge_module.selectors.DefaultSelector):
        def register(self, *args, **kwargs):
            raise AssertionError("closed socket selected")

    monkeypatch.setattr(bridge_module.selectors, "DefaultSelector", ClosedSocketSelector)

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


def _active_peer_state():
    return {"done": threading.Event(), "lock": threading.Lock(), "events": [],
            "handler_terminal": False, "response_write_finished": False,
            "upstream_active": 1, "result_ready": False, "peer_closed": False,
            "peer_cancel_claimed": False}


def test_high_fd_peer_remains_connected_then_eof_cancels_attempt(cfg, tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    resource = pytest.importorskip("resource")
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    if limits[1] != resource.RLIM_INFINITY and limits[1] <= 2048:
        pytest.skip("Hard descriptor limit does not permit a socket above FD_SETSIZE")
    bridge = OpenHandsBridge(cfg, {"attempt_id": "a"}, {}, tmp_path, Deadline(5))
    state = _active_peer_state()
    cancellations = []
    monkeypatch.setattr(bridge, "abort", lambda *args, **kwargs: cancellations.append(kwargs))
    left, right = socket.socketpair()
    high = None
    worker = None
    try:
        if limits[0] != resource.RLIM_INFINITY and limits[0] <= 2048:
            resource.setrlimit(resource.RLIMIT_NOFILE, (2049, limits[1]))
        high = socket.socket(fileno=fcntl.fcntl(left.fileno(), fcntl.F_DUPFD, 2048))
        left.close()
        assert high.fileno() >= 2048
        worker = threading.Thread(target=bridge._watch_peer, args=(high, state))
        worker.start()
        time.sleep(0.4)
        assert worker.is_alive()
        assert not cancellations
        assert not bridge.closed.is_set()
        right.close()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(cancellations) == 1
        assert cancellations[0]["origin"] == "peer_disconnect"
        assert state["peer_closed"]
        assert state["peer_cancel_claimed"]
        assert any(event["event"] == "peer_eof" for event in state["events"])
    finally:
        state["done"].set()
        if worker is not None:
            worker.join(timeout=2)
        if high is not None:
            high.close()
        left.close()
        right.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)


@pytest.mark.parametrize("phase", ["register", "select"])
@pytest.mark.parametrize("error", [ValueError("monitor failure"), OSError(errno.EBADF, "bad descriptor")])
def test_monitor_failure_does_not_claim_disconnect(cfg, tmp_path, monkeypatch, phase, error):
    bridge = OpenHandsBridge(cfg, {"attempt_id": "a"}, {}, tmp_path, Deadline(5))
    state = _active_peer_state()
    cancellations = []
    monkeypatch.setattr(bridge, "abort", lambda *args, **kwargs: cancellations.append(kwargs))

    class BrokenSelector(bridge_module.selectors.DefaultSelector):
        def register(self, *args, **kwargs):
            if phase == "register":
                raise error
            return super().register(*args, **kwargs)

        def select(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(bridge_module.selectors, "DefaultSelector", BrokenSelector)
    left, right = socket.socketpair()
    try:
        bridge._watch_peer(left, state)
        assert not cancellations
        assert not bridge.closed.is_set()
        assert not state["peer_closed"]
        assert not state["peer_cancel_claimed"]
        assert len(state["events"]) == 1
        assert state["events"][0]["event"] == "peer_monitor_error"
        assert state["events"][0]["cancel_claimed"] is False
    finally:
        left.close()
        right.close()


def test_connection_reset_still_cancels_attempt(cfg, tmp_path, monkeypatch):
    bridge = OpenHandsBridge(cfg, {"attempt_id": "a"}, {}, tmp_path, Deadline(5))
    state = _active_peer_state()
    cancellations = []
    monkeypatch.setattr(bridge, "abort", lambda *args, **kwargs: cancellations.append(kwargs))
    left, right = socket.socketpair()

    class ResetSocket:
        def fileno(self):
            return left.fileno()

        def recv(self, *args):
            raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")

    try:
        right.sendall(b"ready")
        bridge._watch_peer(ResetSocket(), state)
        assert len(cancellations) == 1
        assert state["peer_closed"]
        assert state["peer_cancel_claimed"]
        assert state["events"][0]["event"] == "peer_socket_error"
    finally:
        left.close()
        right.close()
