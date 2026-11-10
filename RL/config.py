"""Strict, independent LoRA configuration. No Torch imports in configuration/CLI."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.io import identifier

PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Start(Strict):
    mode: Literal["fresh", "resume", "warm_start"]
    checkpoint: str | None

    @model_validator(mode="after")
    def check_start(self):
        if (self.mode == "fresh") != (self.checkpoint is None):
            raise ValueError("fresh requires null checkpoint; resume/warm_start require a concrete directory")
        return self


class Run(Strict):
    attempt_id: str
    mode: Literal["dry-run", "smoke"]
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    start: Start

    @model_validator(mode="after")
    def check_id(self):
        identifier(self.attempt_id)
        return self


class Model(Strict):
    path: str
    dtype: Literal["bfloat16"]
    attention_backend: Literal["sdpa"]
    gradient_checkpointing: Literal[True]


class Lora(Strict):
    rank: PositiveInt
    alpha: PositiveInt
    dropout: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]
    target_scope: Literal["text_backbone_all_linear"]


class Runtime(Strict):
    gpu: NonnegativeInt
    output_root: str


class Training(Strict):
    max_updates: PositiveInt
    gradient_accumulation_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonnegativeFloat
    grad_clip: PositiveFloat
    scheduler: Literal["constant", "linear"]
    checkpoint_every_updates: PositiveInt


class Validation(Strict):
    fixture_id: Literal["causal-text-smoke-v1"]
    max_sequence_length: Annotated[int, Field(ge=8, le=2048)]
    stop_after_update: PositiveInt | None


class Config(Strict):
    schema_version: Literal[1]
    run: Run
    model: Model
    lora: Lora
    runtime: Runtime
    training: Training
    validation: Validation

    @model_validator(mode="after")
    def check_stop(self):
        stop = self.validation.stop_after_update
        if stop is not None and stop >= self.training.max_updates:
            raise ValueError("stop_after_update must precede the complete schedule target")
        return self


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def absolute(root, value):
    path = Path(value).expanduser()
    if not value.strip() or ".." in path.parts:
        raise ValueError("Paths must be nonempty and must not contain '..'")
    return (path if path.is_absolute() else Path(root) / path).resolve()


def attempt_directory(config):
    root = Path(config.runtime.output_root).resolve()
    if root == Path(root.anchor):
        raise ValueError("Output root must not be the filesystem root")
    directory = root / "RL/checkpoints" / identifier(config.run.attempt_id)
    if directory.resolve() != directory or not directory.is_relative_to(root):
        raise ValueError("LoRA output path crosses a symlink or escapes its storage root")
    return directory


def load(path, project_root):
    source = absolute(project_root, str(path))
    config = Config.model_validate(yaml.load(source.read_text(encoding="utf-8"), Loader=UniqueLoader))
    config.model.path = str(absolute(project_root, config.model.path))
    config.runtime.output_root = str(absolute(project_root, config.runtime.output_root))
    if config.run.start.checkpoint is not None:
        checkpoint = absolute(project_root, config.run.start.checkpoint)
        if not checkpoint.is_dir():
            raise ValueError("Start checkpoint/adapter directory does not exist")
        config.run.start.checkpoint = str(checkpoint)
    attempt_directory(config)
    return config, source


def schedule_contract(config):
    """Only attempt metadata, device/storage, start and pause controls may differ on resume."""
    return {
        "schema_version": config.schema_version,
        "seed": config.run.seed,
        "model": config.model.model_dump(exclude={"path"}),
        "lora": config.lora.model_dump(),
        "training": config.training.model_dump(),
        "validation": config.validation.model_dump(exclude={"stop_after_update"}),
    }
