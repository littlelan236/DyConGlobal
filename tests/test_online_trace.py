from enrichment.online_trace import reasoning_blocks
from core.io import hash_json, sha256, write_json
from runtime.serving.capture_worker import execution_at
from runtime.serving.capture import OnlineCaptureController

import numpy as np
import pytest


def test_reasoning_blocks_separate_answer_newlines_and_repeated_close():
    blocks = reasoning_blocks([1, 10, 198], [20, 271, 21, 11, 198, 11, 271], [10], [11])
    assert blocks == [{"start": 0, "end": 3, "close": 3, "complete": True}]


def test_multiple_reasoning_blocks_and_incomplete_tail():
    blocks = reasoning_blocks([10, 198], [20, 11, 21, 10, 22, 271], [10], [11])
    assert blocks == [{"start": 0, "end": 1, "close": 1, "complete": True},
                      {"start": 4, "end": 6, "close": None, "complete": False}]


def test_execution_evidence_is_per_position_not_lifetime_counter():
    rows = {"r": [{"start": 10, "end": 20, "graph_execution": "full", "batch_kind": "decode"},
                  {"start": 20, "end": 21, "graph_execution": "eager", "batch_kind": "mixed"}]}
    assert execution_at(rows, "r", 19)["graph_execution"] == "full"
    assert execution_at(rows, "r", 20)["graph_execution"] == "eager"
    with pytest.raises(ValueError, match="Missing"):
        execution_at(rows, "r", 21)


def test_finalization_binds_exact_tokens_and_rollout_hash(tmp_path):
    trajectory = tmp_path / "rollout.json"
    prompt, output = [1, 10, 198], [20, 271, 11, 30]
    write_json(trajectory, {"messages": [{"role": "assistant", "extra": {"response": {
        "prompt_token_ids": prompt, "output_token_ids": output,
        "dycon_native_request_id": "native-1"}}}]})
    controller = OnlineCaptureController(tmp_path, {"export": {"root": "online"}})
    try:
        metadata = {"session_id": "session", "task_id": "task", "run_id": "run",
                    "rollout_id": "pending", "step_id": 0, "anchors": ["reasoning_newline"]}
        controller.start(metadata)
        request = controller.bind("native-1", metadata)
        request.update(prompt_token_ids=prompt, request_sha256=hash_json(prompt))
        controller.record_output({"request_id": "native-1", "prompt_token_ids": prompt,
                                  "output_token_ids": output})
        result = controller.finalize("session", rollout_id="rollout", trajectory_path=str(trajectory),
                                     trajectory_sha256=sha256(trajectory), model_fingerprint="model")
        assert result["accepted_native_request_ids"] == ["native-1"]
        trajectory.write_text("changed")
        with pytest.raises(ValueError, match="checksum"):
            controller.finalize("session", rollout_id="rollout", trajectory_path=str(trajectory),
                                trajectory_sha256=result["trajectory_sha256"], model_fingerprint="model")
    finally:
        controller.close()
