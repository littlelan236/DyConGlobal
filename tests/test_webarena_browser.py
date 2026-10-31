import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.containers.webarena_adapter import (
    SOURCE_REVISION,
    WebArenaEnvironment,
    parse_command,
    render_urls,
    validate_answer,
    validate_pool_manifest,
)


@pytest.mark.parametrize("command", [
    "ls /", "browser click [1]; browser click [2]", "browser click [1] && echo hi",
    "browser click [1]\nbrowser click [2]", "browser new_tab click [2]",
    "browser goto [file:///verifier/answers.json]", "browser stop [plain answer]",
    "browser stop [{\"task_type\":\"MUTATE\",\"status\":\"SUCCESS\"}]",
])
def test_unsafe_or_multi_action_never_reaches_worker(command):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_command(command)


def test_stop_preserves_actual_structured_answer():
    answer = {"task_type": "RETRIEVE", "status": "NOT_FOUND_ERROR", "retrieved_data": [],
              "error_details": "No matching item was visible"}
    kind, result = parse_command("browser stop [" + json.dumps(answer) + "]")
    assert kind == "stop" and result == answer
    with pytest.raises(ValueError):
        validate_answer({**answer, "retrieved_data": [float("nan")]})


def test_selected_urls_cannot_access_unleased_site():
    sites = {"reddit": {"base_url": "http://reddit"}}
    assert render_urls({"start_urls": ["__REDDIT__/f/news"]}, sites) == ["http://reddit/f/news"]
    with pytest.raises(ValueError):
        render_urls({"start_urls": ["__GITLAB__/projects"]}, sites)
    with pytest.raises(ValueError):
        render_urls({"start_urls": ["http://reddit.evil/x"]}, sites)


def test_runtime_pool_rejects_mutable_or_external_image_identity():
    ref = {"store": "wav", "manifest": "images/reddit.json", "manifest_sha256": "a" * 64,
           "image_id": "sha256:" + "b" * 64}
    value = {"schema_version": 1, "benchmark": "webarena_verified", "revision": SOURCE_REVISION,
             "sites": {"reddit": {"image_ref": ref, "port": 80, "base_url": "http://reddit"}}}
    assert validate_pool_manifest(value) == value
    value["sites"]["reddit"]["base_url"] = "http://127.0.0.1"
    with pytest.raises(ValueError):
        validate_pool_manifest(value)
    value["sites"]["reddit"]["base_url"] = "http://reddit"
    value["sites"]["reddit"]["image_ref"]["image_id"] = "latest"
    with pytest.raises(ValueError):
        validate_pool_manifest(value)


def test_worker_rejects_stale_request_without_executing():
    from runtime.containers.webarena_worker import BrowserSession
    worker = BrowserSession.__new__(BrowserSession)
    worker.env = SimpleNamespace(step=lambda _: pytest.fail("stale request executed"))
    worker.finished = False
    worker.step_index = 1
    worker.observation_sha256 = "current"
    with pytest.raises(ValueError, match="stale"):
        worker.action({"step": 0, "observation_sha256": "old", "action": "click [1]"})


def test_finish_requires_real_har_flush_and_is_idempotent(tmp_path):
    env = WebArenaEnvironment.__new__(WebArenaEnvironment)
    env._finished = None
    env.browser = object()
    env.subject_dir = tmp_path
    env.actions_path = tmp_path / "actions.jsonl"
    env.network = SimpleNamespace(id="network-1")
    env._containers = []
    env.name = "lease-1"
    env.step_index = 2
    env.sites = {"reddit": {"base_url": "http://reddit"}}
    env.factory = SimpleNamespace(_images={})
    calls = []
    def rpc(*args, **kwargs):
        calls.append(args)
        return {"har_flushed": True}
    env._rpc = rpc
    with pytest.raises(ValueError, match="flushed"):
        env.finish()
    (tmp_path / "network.har").write_text('{"log":{"entries":[]}}')
    result = env.finish()
    assert result["runtime_identity"]["environments"] == {"reddit": {"urls": ["http://reddit"]}}
    assert result["har_flushed"] is True
    assert env.finish() is result and len(calls) == 2


def test_close_flushes_before_deleting_mutable_state(tmp_path):
    env = WebArenaEnvironment.__new__(WebArenaEnvironment)
    events = []
    env._closed = False
    env._acquired = False
    env.browser = object()
    env.subject_dir = tmp_path
    env._containers = [SimpleNamespace(id="container-1", remove=lambda **_: events.append("container"))]
    env._volumes = [SimpleNamespace(name="volume-1", remove=lambda: events.append("volume"))]
    env.network = SimpleNamespace(remove=lambda: events.append("network"))
    env.factory = SimpleNamespace(_active={env})
    env.finish = lambda: events.append("flush")
    env.close()
    env.close()
    assert events == ["flush", "container", "volume", "network"]
    assert json.loads((Path(tmp_path) / "cleanup.json").read_text())["cleanup_complete"]
