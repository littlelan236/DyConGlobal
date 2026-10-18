import numpy as np
import pytest

from core.io import write_json
from hidden.evaluation.compare_vllm_replay import compare_replay


def _artifact(root, value):
    root.mkdir()
    np.savez_compressed(root / "x.npz", hidden=np.asarray(value, dtype=np.float16))
    write_json(root / "index.json", [{"sample_id": "e", "path": "x.npz", "token_id": 271, "absolute_position": 10}])


def test_fp16_vllm_replay_metrics_and_dtype_gate(tmp_path):
    online, replay = tmp_path / "online", tmp_path / "replay"
    _artifact(online, [[1, 2, 3]])
    _artifact(replay, [[1, 2, 3.001]])
    result = compare_replay(online, replay, dtype="float16")
    assert result["reference"] == "vllm_native_replay"
    assert result["passed"]
    with pytest.raises(ValueError, match="FP16-only"):
        compare_replay(online, replay, dtype="bfloat16")


def test_async_sidecar_index_is_supported(tmp_path):
    online, replay = tmp_path / "online", tmp_path / "replay"
    for root in (online, replay):
        (root / "index").mkdir(parents=True)
        np.savez_compressed(root / "event.npz", hidden=np.ones((2, 4), dtype=np.float16))
        write_json(
            root / "index" / "event.json",
            {
                "sample_id": "event",
                "path": "event.npz",
                "token_id": 271,
                "absolute_position": 12,
                "token_ref": ["r", 0, "generation_input", 12, 271],
            },
        )
    result = compare_replay(online, replay, dtype="float16")
    assert result["passed"] and result["samples"] == 1


def test_comparison_rejects_corruption_in_any_layer(tmp_path):
    online, replay = tmp_path / "online", tmp_path / "replay"
    for root in [online, replay]:
        root.mkdir()
        write_json(root / "index.json", [{"sample_id": "event", "path": "event.npz",
                                          "token_ref": ["r", 0, "generation_input", 12, 271]}])
    np.savez(online / "event.npz", layer_0=np.ones(4), layer_32=np.ones(4))
    np.savez(replay / "event.npz", layer_0=np.ones(4), layer_32=-np.ones(4))
    result = compare_replay(online, replay)
    assert not result["passed"]
    assert result["layers"]["layer_0"]["min_cosine"] == 1
    assert result["layers"]["layer_32"]["min_cosine"] == -1
