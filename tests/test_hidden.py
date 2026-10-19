import numpy as np
import pytest
from conftest import ToyTokenizer, import_fixture

from action.datasets.index import create_action_index
from core.io import read_json
from difficulty.datasets.index import build_index
from enrichment.trajectory import enrich
from hidden.extract import extract_hidden


class Adapter:
    fingerprint = "fixture-model"
    hidden_size = 4

    batch_size = 4
    metadata = {"name": "vllm_eager", "runtime_source_sha256": "fixture"}

    def capture_batch(self, rows, layers, chunk_size):
        return {
            row["sample_id"]: np.ones((len(layers), self.hidden_size), dtype=np.float32)
            for row in rows
        }

    def sites(self):
        return [{"kind": "final_norm", "block_index": None}]


def test_capture_index_and_action_share_tensors(cfg):
    import_fixture(cfg, resolved=True)
    enrich(cfg, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    out = extract_hidden(cfg, "dataset", "capture", Adapter())
    assert len(read_json(out / "index.json")) == 4
    assert extract_hidden(cfg, "dataset", "capture", Adapter()) == out
    index = build_index(cfg, "dataset", "capture", "joined")
    assert len(read_json(index / "index.json")["samples"]) == 4
    action = create_action_index(cfg, "joined", "actions")
    assert read_json(action / "index.json")["taxonomy_version"] is None


def test_prediction_evaluation_uses_frozen_labels(cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from core.io import write_json
    from difficulty.evaluation.run import evaluate_predictions

    rid = import_fixture(cfg, resolved=True)
    dataset = enrich(cfg, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    extract_hidden(cfg, "dataset", "capture", Adapter())
    build_index(cfg, "dataset", "capture", "joined")
    rows = read_json(dataset / f"{rid}.samples.json")
    path = tmp_path / "predictions.json"
    write_json(
        path,
        [
            {
                "sample_id": row["sample_id"],
                "target": "remaining_steps",
                "prediction": row["labels"]["remaining_steps"],
            }
            for row in rows
        ],
    )
    out = evaluate_predictions(cfg, "joined", path, "score")
    assert all(x["mae"] == 0 for x in read_json(out / "metrics.json")["metrics"].values())
