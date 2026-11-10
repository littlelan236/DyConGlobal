from pathlib import Path

import yaml

from RL.config import load

import pytest
from RL.config import Config


def test_grpo_smoke_template_is_explicit_and_strict():
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    assert config.algorithm.name == "offline_grpo_k3_v1"
    assert config.algorithm.dataset_id == "synthetic-grpo-smoke-v1"
    assert config.grpo.max_epochs == 1
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw["grpo"]["probability_mode"] = "processed_replay"
    bad = root / "RL/experiments/.grpo-invalid-test.yaml"
    bad.write_text(yaml.safe_dump(raw))
    try:
        try:
            load(bad, root)
        except ValueError:
            pass
        else:
            raise AssertionError("processed probability mode was accepted")
    finally:
        bad.unlink()


@pytest.mark.parametrize("field,value", [("max_epochs",2), ("clip_epsilon",.2), ("kl_beta",0.),
    ("probability_mode","processed_sampling_v1"), ("reference","latest_policy"),
    ("top_k",0), ("unknown",1)])
def test_grpo_schema_rejects_unsupported_objective(field,value):
    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw["grpo"][field] = value
    with pytest.raises(ValueError):
        Config.model_validate(raw)


@pytest.mark.parametrize("missing", ["name","dataset_root","dataset_id"])
def test_missing_dataset_algorithm_binding_rejected(missing):
    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    del raw["algorithm"][missing]
    with pytest.raises(ValueError):
        Config.model_validate(raw)
