from types import SimpleNamespace

import numpy as np
import pytest

from hidden.evaluation.assess_rollout_dtype import generated_references, summarize_errors


class Tensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def numpy(self):
        return self.value


class Captured:
    outputs = [SimpleNamespace(prompt_token_ids=[10, 11], outputs=[SimpleNamespace(token_ids=[20, 99, 30])])]

    def sample_positions(self, i):
        return [1, 2, 3]

    def sample_token_ids(self, i):
        return [11, 20, 99]

    def sample_rows(self, i, layer):
        return [Tensor(np.array([layer, j], np.float32)) for j in range(3)]


def test_online_references_use_consumed_token_without_future_leakage():
    rows, values = generated_references(
        Captured(),
        [{"prefix_token_ids": [10, 11], "sample_id": "s", "task_id": "t", "rollout_id": "r"}],
        [0, 2],
        99,
    )
    assert [r["kind"] for r in rows] == ["think_open", "decode_token", "think_close"]
    assert [r["prefix_token_ids"] for r in rows] == [[10, 11], [10, 11, 20], [10, 11, 20, 99]]
    assert values.shape == (3, 2, 2)


def test_reference_rejects_wrong_token_labels():
    capture = Captured()
    capture.sample_token_ids = lambda i: [11, 20, 98]
    with pytest.raises(ValueError, match="exact emitted"):
        generated_references(
            capture,
            [{"prefix_token_ids": [10, 11], "sample_id": "s", "task_id": "t", "rollout_id": "r"}],
            [0, 2],
            99,
        )


def test_precision_summary_preserves_marker_groups():
    reference = np.ones((2, 3, 4))
    actual = reference.copy()
    actual[1] *= 1.1
    result = summarize_errors(actual, reference, [{"kind": "think_open"}, {"kind": "think_close"}])
    assert result["think_open"]["relative_l2_max"] == 0
    assert result["think_close"]["relative_l2_p50"] == pytest.approx(0.1)
    assert result["all"]["site_vectors"] == 6
