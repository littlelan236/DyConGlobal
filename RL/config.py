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


class Memory(Strict):
    loss_chunk_tokens: NonnegativeInt = 0
    mlp_chunk_tokens: NonnegativeInt = 0
    exact_length: bool = False
    attention_policy: Literal["auto", "flash_only"] = "auto"
    activation_offload: Literal["none", "checkpoint_boundary"] = "none"
    host_budget_gib: PositiveFloat = 64.0
    profile_first_update: bool = False


class Supervision(Strict):
    mode: Literal["labels", "explicit_mask", "uniform"] = "labels"
    max_targets_per_sample: PositiveInt | None = None

    @model_validator(mode="after")
    def check_selection(self):
        if (self.mode == "uniform") != (self.max_targets_per_sample is not None):
            raise ValueError("Only uniform supervision requires max_targets_per_sample; other modes require null")
        return self


class Training(Strict):
    max_updates: PositiveInt
    gradient_accumulation_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonnegativeFloat
    grad_clip: PositiveFloat
    scheduler: Literal["constant", "linear"]
    checkpoint_every_updates: PositiveInt
    memory: Memory = Field(default_factory=Memory)
    supervision: Supervision = Field(default_factory=Supervision)
    checkpoint_policy: Literal["layer", "selective_mlp", "layer_group"] = "layer"
    checkpoint_group_layers: PositiveInt = 4


class Validation(Strict):
    fixture_id: Literal["causal-text-smoke-v1"]
    max_sequence_length: Annotated[int, Field(ge=8, le=262144)]
    stop_after_update: PositiveInt | None


class Algorithm(Strict):
    name: Literal["offline_grpo_k3_v1"]
    dataset_root: str
    dataset_id: str
    require_data_ready: Literal[True] = True

    @model_validator(mode="after")
    def check_binding(self):
        identifier(self.dataset_id)
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must be nonempty")
        return self


class Grpo(Strict):
    group_batch_size: PositiveInt = 4
    max_epochs: Literal[1] = 1
    shuffle_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42
    top_k: PositiveInt | None = 4
    top_k_fraction: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    score_scale_floor: PositiveFloat = 0.1
    boundary_tie_sigma: NonnegativeFloat = 0.05
    clip_epsilon: Literal[0.1] = 0.1
    kl_beta: Literal[0.02] = 0.02
    reference: Literal["initial_policy"] = "initial_policy"
    probability_mode: Literal["raw_t1"] = "raw_t1"

    @model_validator(mode="after")
    def check_top_k(self):
        if (self.top_k is None) == (self.top_k_fraction is None):
            raise ValueError("Exactly one of grpo.top_k and grpo.top_k_fraction is required")
        return self


class Config(Strict):
    schema_version: Literal[1]
    run: Run
    model: Model
    lora: Lora
    runtime: Runtime
    training: Training
    validation: Validation
    algorithm: Algorithm | None = None
    grpo: Grpo | None = None

    @model_validator(mode="after")
    def check_stop(self):
        stop = self.validation.stop_after_update
        if stop is not None and stop >= self.training.max_updates:
            raise ValueError("stop_after_update must precede the complete schedule target")
        memory = self.training.memory
        if self.training.supervision.mode != "labels" and not memory.loss_chunk_tokens:
            raise ValueError("Sparse supervision requires loss_chunk_tokens > 0 before vocabulary projection")
        if self.lora.dropout != 0:
            raise ValueError("Only dropout=0 is supported; chunk recomputation needs reproducible RNG")
        if self.training.checkpoint_policy == "selective_mlp" and not memory.mlp_chunk_tokens:
            raise ValueError("selective_mlp requires mlp_chunk_tokens > 0")
        if memory.activation_offload != "none" and self.training.checkpoint_policy not in {"layer", "layer_group"}:
            raise ValueError("checkpoint_boundary offload requires layer checkpointing")
        if self.algorithm is not None and self.grpo is None:
            raise ValueError("GRPO algorithm requires a grpo configuration")
        if self.grpo is not None and self.algorithm is None:
            raise ValueError("grpo configuration requires algorithm.name=offline_grpo_k3_v1")
        if self.grpo and (self.training.supervision.mode != "labels"
                          or self.training.gradient_accumulation_steps != 1):
            raise ValueError("GRPO uses all native targets and group_batch_size; require labels and accumulation=1")
        if self.validation.max_sequence_length > 2048:
            if not (memory.loss_chunk_tokens and memory.exact_length and memory.attention_policy == "flash_only"):
                raise ValueError("Long sequences require chunked CE, exact-length batches and flash_only attention")
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
    if config.algorithm is not None:
        config.algorithm.dataset_root = str(absolute(project_root, config.algorithm.dataset_root))
    if config.run.start.checkpoint is not None:
        checkpoint = absolute(project_root, config.run.start.checkpoint)
        if not checkpoint.is_dir():
            raise ValueError("Start checkpoint/adapter directory does not exist")
        config.run.start.checkpoint = str(checkpoint)
    attempt_directory(config)
    return config, source


def schedule_contract(config):
    """Only attempt metadata, device/storage, start and pause controls may differ on resume."""
    contract = {
        "schema_version": config.schema_version,
        "seed": config.run.seed,
        "model": config.model.model_dump(exclude={"path"}),
        "lora": config.lora.model_dump(),
        "training": config.training.model_dump(),
        "validation": config.validation.model_dump(exclude={"stop_after_update"}),
    }
    if config.algorithm:
        contract.update(algorithm=config.algorithm.model_dump(), grpo=config.grpo.model_dump())
    return contract
