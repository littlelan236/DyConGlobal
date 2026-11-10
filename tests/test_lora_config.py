import copy
from pathlib import Path

import pytest
import yaml

from RL.config import Config, attempt_directory, load, schedule_contract


@pytest.fixture
def values():
    return yaml.safe_load((Path(__file__).parents[1] / "RL/experiments/lora-smoke.yaml").read_text())


@pytest.mark.parametrize("section,key,value", [
    ("lora", "rank", 0), ("lora", "rank", True), ("lora", "dropout", float("nan")),
    ("runtime", "gpu", -1), ("runtime", "gpu", "0"), ("model", "dtype", "float32"),
    ("training", "learning_rate", float("inf")), ("lora", "target_scope", "mlp_only"),
    ("validation", "max_sequence_length", 262144), ("run", "unexpected", 1),
    ("model", "gradient_checkpointing", False),
])
def test_strict_rejections(values, section, key, value):
    values[section][key] = value
    with pytest.raises(ValueError):
        Config.model_validate(values)


@pytest.mark.parametrize("mode,checkpoint", [("fresh", "/x"), ("resume", None), ("warm_start", None)])
def test_start_modes(values, mode, checkpoint):
    values["run"]["start"] = dict(mode=mode, checkpoint=checkpoint)
    with pytest.raises(ValueError):
        Config.model_validate(values)


def test_duplicate_key_and_relative_paths(values, tmp_path):
    source = tmp_path / "train.yaml"
    source.write_text(yaml.safe_dump(values) + "schema_version: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load(source, tmp_path)
    values["model"]["path"] = "models/Qwen3.5-9B"
    values["runtime"]["output_root"] = "outputs"
    source.write_text(yaml.safe_dump(values))
    config, _ = load(source, tmp_path)
    assert config.model.path == str(tmp_path / "models/Qwen3.5-9B")
    assert attempt_directory(config) == tmp_path / "outputs/RL/checkpoints" / config.run.attempt_id
    values["runtime"]["output_root"] = "../escape"
    source.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError):
        load(source, tmp_path)


def test_output_symlink_and_existing_start(values, tmp_path):
    values["runtime"]["output_root"] = str(tmp_path)
    (tmp_path / "RL").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        attempt_directory(Config.model_validate(values))
    values["run"]["start"] = dict(mode="resume", checkpoint="missing")
    source = tmp_path / "train.yaml"
    source.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="does not exist"):
        load(source, tmp_path)


def test_resume_whitelist(values):
    first = Config.model_validate(values)
    other = copy.deepcopy(values)
    other["run"].update(attempt_id="resumed", start=dict(mode="resume", checkpoint="/checkpoint"))
    other["runtime"].update(gpu=1, output_root="/other")
    other["validation"]["stop_after_update"] = 2
    second = Config.model_validate(other)
    assert schedule_contract(first) == schedule_contract(second)
    second.training.learning_rate *= 2
    assert schedule_contract(first) != schedule_contract(second)
