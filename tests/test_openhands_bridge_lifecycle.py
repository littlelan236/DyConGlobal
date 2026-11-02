import time

import requests

from rollout.adapters import openhands_bridge as bridge_module
from rollout.adapters.openhands_bridge import OpenHandsBridge
from core.deadline import Deadline


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
