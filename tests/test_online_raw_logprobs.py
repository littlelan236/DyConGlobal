"""Native sidecar integrity, publication and dataset stream acceptance."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.io import hash_json, read_json, sha256, write_json
from core.config import CaptureLogprobs, Capture
from runtime.serving.raw_logprobs import RawLogprobWriter, read_values, STREAM


def native_record():
    return {"request_id": "native-0", "run_id": "run-0", "session_id": "session-0",
            "decision_id": "decision-0", "branch_id": 0, "step_id": 0,
            "api_attempt_id": "api-0", "tokenizer_fingerprint": "tokenizer-0",
            "prompt_token_ids": [1, 2], "output_token_ids": [3, 4, 5]}


def test_strict_raw_config():
    assert CaptureLogprobs().enabled is False
    for options in ({"probability_semantics": "processed_sampling_v1"},
                    {"temperature": .6}, {"mode": "processed"}):
        with pytest.raises(ValueError):
            CaptureLogprobs(**options)
    with pytest.raises(ValueError, match="persisted native capture"):
        Capture(logprobs={"enabled": True})


def test_sidecar_fp32_hash_alignment_and_nonfinite(tmp_path):
    record = native_record()
    writer = RawLogprobWriter(tmp_path, tmp_path / "export", {})
    try:
        descriptor = writer.write(record, [-.1, -.2, -np.inf])
        assert descriptor["bytes"] == 12 and descriptor["negative_infinity_count"] == 1
        values = read_values(tmp_path / "export", descriptor, output_ids=record["output_token_ids"], record=record)
        assert values.dtype == np.dtype("float32")
        assert np.isneginf(values[-1])
        assert writer.write(record, [-.1, -.2, -np.inf]) == descriptor
        with pytest.raises(ValueError, match="Immutable"):
            writer.write(record, [-.1, -.3, -np.inf])
        for invalid in ([-.1], [-.1, np.nan, -.2], [-.1, np.inf, -.2], [-.1, .1, -.2]):
            with pytest.raises(ValueError):
                writer.write(record, invalid)
        with pytest.raises(ValueError, match="alignment"):
            read_values(tmp_path / "export", descriptor, output_ids=[3, 5, 4])
        wrong = copy.deepcopy(descriptor)
        wrong["probability_semantics"] = "processed_sampling_v1"
        with pytest.raises(ValueError, match="semantics"):
            read_values(tmp_path / "export", wrong)
        wrong = copy.deepcopy(descriptor)
        wrong["identity"]["branch_id"] = 1
        with pytest.raises(ValueError, match="identity"):
            read_values(tmp_path / "export", wrong, record=record)
        (tmp_path / "export" / descriptor["locator"]).write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            read_values(tmp_path / "export", descriptor)
    finally:
        writer.close()


def test_request_and_queue_budgets(tmp_path):
    writer = RawLogprobWriter(tmp_path, tmp_path / "export", {"max_request_tokens": 2, "queue_size": 1})
    try:
        with pytest.raises(ValueError, match="token budget"):
            writer.write(native_record(), [-1., -1., -1.])
        writer.capacity.acquire()
        with pytest.raises(RuntimeError, match="queue is full"):
            writer.write(native_record(), [-1., -1., -1.])
        writer.capacity.release()
    finally:
        writer.close()


def online_dataset(tmp_path):
    from test_rl_dataset import token_source, build
    fixture = token_source(tmp_path)
    f, file, dc = fixture
    # Probability checks must remain strict with the native audit switched off.
    dc["audit"]["enabled"] = False
    file.write_text(yaml.safe_dump(dc), encoding="utf-8")
    config = read_json(tmp_path / "runs" / f["run"] / "resolved-config.json")
    writer = RawLogprobWriter(tmp_path, f["nas"], {}, source_identity={
        "model_fingerprint": config["model"]["fingerprint"],
        "checkpoint_inventory_sha256": config["model"]["fingerprint"],
        "serving_dtype": config["serving"].get("dtype"), "adapter": None})
    try:
        for path in sorted((f["nas"] / "requests").glob("*.json")):
            record = read_json(path)
            record["raw_t1_probability"] = writer.write(record, [-.1, -.2, -.3])
            write_json(path, record)
    finally:
        writer.close()
    return build(fixture), f


def test_dataset_neutral_stream_and_integrity_even_with_native_audit_disabled(tmp_path):
    from RL.dataset.build import check_dataset
    from RL.dataset.loader import iter_groups
    from RL.dataset.online_probabilities import OnlineProbabilityReader
    ds, f = online_dataset(tmp_path)
    check_dataset(ds)
    reader = OnlineProbabilityReader(ds)
    group = next(iter_groups(ds))
    from RL.dataset.probability import ProbabilityStream
    stream_binding = read_json(ds / "dataset.json")["probability_streams"][STREAM]
    shared = ProbabilityStream(ds, kind="rl-dataset", manifest_name=stream_binding["manifest"],
        index_name=stream_binding["index"], dataset_receipt_sha256=stream_binding["dataset_binding"], stream=STREAM)
    shared.validate_dataset_rows(group["members"])
    for row in group["members"]:
        np.testing.assert_allclose(shared.values(row), [-.1, -.2, -.3])
    shared.verify_unchanged()
    assert group["group"]["raw_probability_complete"] is True
    for row in group["members"]:
        result = reader.record(row)
        np.testing.assert_allclose(result["values"], [-.1, -.2, -.3])
        assert result["count"] == row["completion_count"]
        assert result["mask_spans"] == row["mask_spans"]
        assert "reference" not in result and "role" not in result
    row = group["members"][0]
    desc = row["probability_streams"][STREAM]["descriptor"]
    (f["nas"] / desc["locator"]).write_bytes(np.array([-.1, -.2, -.9], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="checksum"):
        reader.values(row)
    with pytest.raises(ValueError, match="checksum"):
        check_dataset(ds)


def test_historical_dataset_has_explicitly_unavailable_stream(tmp_path):
    from test_rl_dataset import token_source, build
    from RL.dataset.loader import iter_groups
    from RL.dataset.online_probabilities import OnlineProbabilityReader
    ds = build(token_source(tmp_path))
    group = next(iter_groups(ds))
    assert group["group"]["raw_probability_complete"] is False
    with pytest.raises(ValueError, match="no raw T=1"):
        OnlineProbabilityReader(ds).values(group["members"][0])


def test_native_controller_probability_failure_keeps_output_and_retry_integrity(tmp_path):
    from runtime.serving.capture import OnlineCaptureController
    from hidden.backends.vllm.online import capture_request_metadata
    options = {"export": {"root": "online"}, "logprobs": {"enabled": True}}
    controller = OnlineCaptureController(tmp_path, options)
    metadata = capture_request_metadata("session", "task", "run", "pending", 0, ["think_close"])
    controller.start(metadata)
    try:
        controller.bind("native-0", metadata)
        value = {"request_id": "native-0", "metadata": metadata,
                 "prompt_token_ids": [10, 20], "output_token_ids": [30, 40],
                 "finish_reason": "stop", "raw_t1_logprobs": [-.2, -.4]}
        controller.record_output(value)
        saved = controller.request_outputs["native-0"]
        assert saved["raw_t1_probability"]["status"] == "sealed"
        assert "raw_t1_logprobs" not in saved
        assert controller.record_output(value)["idempotent"]
        with pytest.raises(ValueError, match="Conflicting native output"):
            controller.record_output({**value, "raw_t1_logprobs": [-.3, -.4]})
        assert controller.request_outputs["native-0"] == saved
        # An aborted/partial probability vector still preserves actual tokens.
        controller.bind("native-1", metadata)
        controller.record_output({**value, "request_id": "native-1", "finish_reason": "abort",
                                  "raw_t1_logprobs": [-.2]})
        failed = controller.request_outputs["native-1"]
        assert failed["output_token_ids"] == [30, 40] and failed["finish_reason"] == "abort"
        assert failed["raw_t1_probability"]["status"] == "failed"
        assert controller.sessions["session"].status == "active"
    finally:
        controller.close()


def test_spool_upload_acks_sidecar_and_recovery_rejects_missing_bytes(tmp_path):
    from runtime.serving.spool import CaptureSpool
    options = {"spool_root": "spool", "root": "online", "spool_max_bytes": 2**24,
               "upload_workers": 1, "upload_max_elapsed_seconds": 20, "shard_max_bytes": 2**20}
    spool = CaptureSpool(tmp_path, options, "run-0", tmp_path / "nas")
    writer = RawLogprobWriter(tmp_path, spool.root, {}, spool=spool)
    record = native_record()
    try:
        record["raw_t1_probability"] = writer.write(record, [-.1, -.2, -.3])
        relative = "requests/" + hash_json(record["request_id"]) + ".json"
        spool.write_json(relative, record, immutable=True)
        # Deliberately use no live session hints: recovery must find the vector.
        assert spool.publish_session("session-0", 20)
        descriptor = record["raw_t1_probability"]
        np.testing.assert_allclose(read_values(spool.nas, descriptor, record=record), [-.1, -.2, -.3])
        spool.bind_session_metadata("session-0", [record["request_id"]], [])
        assert descriptor["locator"] in spool._session_metadata_hints["session-0"][0]
        # No success publication can be emitted if its sealed vector disappears.
        (spool.root / descriptor["locator"]).unlink()
        with pytest.raises(FileNotFoundError):
            spool.bind_session_metadata("session-0", [record["request_id"]], [])
    finally:
        writer.close()
        spool.close(20)
