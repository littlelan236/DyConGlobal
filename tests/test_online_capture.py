import numpy as np
import pytest

from baseline.rollout.model import model_config
from hidden.backends.vllm.online import (
    capture_request_metadata,
    classify_capture_anchor,
    publish_capture_result,
    safe_cache_hit_length,
)
from runtime.serving.capture import (
    AnchorError,
    CaptureControlServer,
    OnlineCaptureController,
    validate_anchor,
)


def test_qwen_open_anchor_and_close_requires_followup():
    assert validate_anchor([1, 248068, 198], "think_open")["position"] == 1
    with pytest.raises(AnchorError, match="think_open_unavailable"):
        validate_anchor([1, 248068], "think_open")
    with pytest.raises(AnchorError, match="no_followup_forward"):
        validate_anchor([1, 2], "think_close", token_ids={"think_close": [9]})


def test_safe_boundary_never_reuses_target():
    assert safe_cache_hit_length(100, 99) == 99
    assert safe_cache_hit_length(100, 200) == 99
    assert safe_cache_hit_length(100, 80) == 80
    with pytest.raises(ValueError):
        safe_cache_hit_length(4, -1)


def test_async_export_and_controller(tmp_path):
    cfg = {"enabled": True, "graph_mode": "FULL", "export": {"root": str(tmp_path / "online"), "queue_size": 2}}
    controller = OnlineCaptureController(tmp_path, cfg)
    metadata = capture_request_metadata("s", "task", "run", "rollout", 0, ["think_open"])
    controller.start(metadata)
    controller.bind("request", metadata)
    result = controller.record(
        "request",
        {"0": np.ones((1, 4), dtype=np.float32)},
        prompt_token_ids=[1, 248068, 198],
        anchor="think_open",
        graph_replays=1,
        cache_hit_tokens=2,
    )
    assert result["status"] == "queued"
    controller.queue.queue.join()
    assert list((tmp_path / "online" / "tensors").glob("*.npz"))
    assert controller.status("s")["metrics"]["graph_replays"] == 1
    controller.close()


def test_control_server_lifecycle(tmp_path):
    import requests

    controller = OnlineCaptureController(tmp_path, {"export": {"root": "hidden/online"}})
    server = CaptureControlServer(controller).start()
    try:
        response = requests.get(server.base_url + "/dycon/capture/status", timeout=2)
        assert response.status_code == 200
        assert response.json()["sessions"] == []
    finally:
        server.close()


def test_capture_request_uses_union_selector(cfg):
    cfg["capture"]["enabled"] = True
    cfg["capture"]["anchors"] = ["think_open", "reasoning_newline", "think_close"]
    cfg["model"]["served_name"] = "fixture"
    body = model_config(
        cfg,
        {
            "session_id": "s",
            "task_id": "t",
            "run_id": "r",
            "rollout_id": "ro",
            "step_id": 0,
            "newline_token_ids": [198, 271, 1358],
            "close_token_ids": [248069],
        },
    )["model_kwargs"]["extra_body"]
    assert body["capture_select"]["tact_states"]["prompt_positions"] == [-2]
    assert set(body["capture_select"]["tact_states"]["generation_tokens"]) == {198, 271, 1358, 248069}


def test_capture_metadata_carries_selector_and_tokenizer_identity():
    metadata = capture_request_metadata(
        "s",
        "task",
        "run",
        "rollout",
        2,
        ["think_open", "reasoning_newline", "think_close"],
        newline_token_ids=[198, 271],
        close_token_ids=[248069],
        tokenizer_fingerprint="tok-v1",
        newline_policy="all",
    )
    assert metadata["tokenizer_fingerprint"] == "tok-v1"
    assert metadata["capture_select_sha256"]
    assert metadata["capture_select"]["tact_states"]["prompt_positions"] == [-2]
    assert set(metadata["capture_select"]["tact_states"]["generation_tokens"]) == {198, 271, 248069}


def test_async_export_writes_raw_sidecar_and_session_receipt(tmp_path):
    cfg = {"enabled": True, "graph_mode": "FULL", "export": {"root": str(tmp_path / "online"), "queue_size": 2}}
    controller = OnlineCaptureController(tmp_path, cfg)
    metadata = capture_request_metadata("raw-session", "task", "run", "rollout", 0, ["think_open"])
    controller.start(metadata)
    controller.bind("request", metadata)
    result = controller.record(
        "request",
        {"hidden": np.ones((1, 4), dtype=np.float32)},
        prompt_token_ids=[1, 248068, 198],
        anchor="think_open",
        token_text="<think>",
        phase="prompt",
        absolute_position=1,
        token_ref=["rollout", 0, "prompt", 1, 248068],
    )
    assert result["status"] == "queued"
    controller.queue.queue.join()
    controller.stop("raw-session")
    assert list((tmp_path / "online" / "raw").glob("*.npz"))
    assert list((tmp_path / "online" / "raw").glob("*.json"))
    assert (tmp_path / "online" / "sessions" / "raw-session" / "receipt.json").exists()
    controller.close()


def test_native_capture_rows_keep_vllm_coordinates(tmp_path):
    from types import SimpleNamespace

    class Result:
        outputs = [SimpleNamespace(request_id="internal-1", prompt_token_ids=[1, 248068, 198])]
        layers = [0]

        def sample_positions(self, _):
            return [1, 3, 4]

        def sample_token_ids(self, _):
            return [248068, 271, 248069]

        def sample_rows(self, _, __):
            return np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    class Client:
        def __init__(self):
            self.calls = []

        def submit_staged(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return kwargs

    assert classify_capture_anchor(1, 248068, 3, [271], [248069]) == "think_open"
    client = Client()
    published = publish_capture_result(
        Result(),
        [{"prefix_token_ids": [1, 248068, 198]}],
        client,
        tmp_path,
        newline_token_ids=[271],
        close_token_ids=[248069],
    )
    assert len(published) == 3
    assert [call[1]["anchor"] for call in client.calls] == ["think_open", "reasoning_newline", "think_close"]
    assert [call[1]["absolute_position"] for call in client.calls] == [1, 3, 4]


def test_http_staging_binds_explicit_native_request(tmp_path):
    from core.io import read_json
    from hidden.backends.vllm.online import CaptureClient

    controller = OnlineCaptureController(tmp_path, {"export": {"root": "hidden/online"}})
    server = CaptureControlServer(controller).start()
    client = CaptureClient(server.base_url)
    try:
        for step in [0, 1]:
            metadata = capture_request_metadata(f"session-{step}", "task", "run", f"rollout-{step}", step,
                                                ["reasoning_newline"], newline_token_ids=[271])
            metadata.update(prompt_token_ids=[1, 248068, 198], message_index=step + 2)
            client.start(metadata)
            client.bind(f"native-{step}", metadata)
            client.submit_staged(controller.export_root / "staging", f"native-{step}",
                                 {"hidden": np.ones((2, 4))}, anchor="reasoning_newline",
                                 token_id=271, position=4, absolute_position=4)
        controller.queue.queue.join()
        rows = [read_json(p) for p in (controller.export_root / "raw").glob("*.json")]
        assert len(rows) == 2
        for row in rows:
            assert row["token_ref"] == [f"rollout-{row['step_id']}", row["step_id"], "generation_input", 4, 271]
            assert row["generation_index"] == 1
        assert not list((controller.export_root / "staging").glob("*.npz"))
    finally:
        server.close()
