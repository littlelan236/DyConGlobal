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
    ("runtime", "gpu", -1), ("runtime", "gpu", "0"), ("model", "dtype", "float64"),
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


def test_grpo_defaults_to_frozen_prefix_length_pool_shuffle(values):
    from RL.config import load
    root=Path(__file__).parents[1]
    config,_=load(root/"RL/experiments/grpo-offline-k3-smoke.yaml",root)
    assert config.grpo.shuffle_algorithm=="prefix_length_pool128_v1"
    assert "shuffle_algorithm" not in schedule_contract(config)["grpo"]
    config.grpo.shuffle_algorithm="stable_random_v1"
    assert schedule_contract(config)["grpo"].get("shuffle_algorithm") is None


def test_shared_objective_preserves_offline_schedule():
    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw.setdefault("grpo", {})["kl_beta"] = 0.0
    legacy = Config.model_validate(raw)
    del raw["grpo"]["kl_beta"]
    raw["objective"] = {"kl": {"beta": 0.0}}
    migrated = Config.model_validate(raw)
    assert schedule_contract(legacy) == schedule_contract(migrated)
    assert migrated.objective.kl.beta == 0.0
    assert Config.model_validate(migrated.model_dump()) == migrated

    raw["offline_baseline"] = {"dataset": "/unused", "kl_beta": 0.02}
    with pytest.raises(ValueError, match="Conflicting kl_beta"):
        Config.model_validate(raw)
    del raw["offline_baseline"]["kl_beta"]
    mixed = Config.model_validate(raw)
    assert mixed.grpo.kl_beta == mixed.offline_baseline.kl_beta == 0.0


def test_baseline_epoch_schedule_without_explicit_updates(values, tmp_path):
    (tmp_path / "manifest.json").write_text('{"entries":[{"task":"a"},{"task":"b"},{"task":"c"}]}')
    values["algorithm"] = {"name": "offline_baseline_grpo_v1"}
    values["offline_baseline"] = {"dataset": str(tmp_path), "epochs": 3, "tasks_per_update": 2}
    values["objective"] = {"kl": {"beta": 0.0}, "ppo": {"clip_epsilon": 0.2}}
    values["training"].pop("max_updates")
    cfg = Config.model_validate(values)
    assert cfg.training.max_updates == 6
    assert cfg.offline_baseline.dycon_weight == 0
    assert cfg.offline_baseline.clip_epsilon == 0.2
