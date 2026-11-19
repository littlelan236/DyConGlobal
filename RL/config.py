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


class ConfigMigration(Strict):
    source_schedule_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    target_schedule_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reason: Annotated[str, Field(min_length=1)]


class Start(Strict):
    mode: Literal["fresh", "resume", "warm_start", "append_resume"]
    checkpoint: str | None
    extension_manifest: str | None = None
    config_migration: ConfigMigration | None = None
    optimizer_backend_migration: ConfigMigration | None = None

    @model_validator(mode="after")
    def check_start(self):
        if (self.mode == "fresh") != (self.checkpoint is None):
            raise ValueError("fresh requires null checkpoint; resume/warm_start require a concrete directory")
        if (self.mode == "append_resume") != (self.extension_manifest is not None):
            raise ValueError("append_resume requires an explicit extension_manifest; other modes forbid it")
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
    dtype: Literal["bfloat16", "float32"]
    attention_backend: Literal["sdpa"]
    gradient_checkpointing: Literal[True]
    # GDN is selected explicitly so installing an optional package cannot
    # change the numerical path of an existing training configuration.
    gdn_backend: Literal["reference_pytorch", "fla_chunk"] = "reference_pytorch"


class Lora(Strict):
    rank: PositiveInt
    alpha: PositiveInt
    dropout: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]
    target_scope: Literal["text_backbone_all_linear"]


class Runtime(Strict):
    gpu: NonnegativeInt | Annotated[str, Field(pattern=r"^MIG-[0-9a-fA-F-]{36}$")] | None = None
    gpus: list[NonnegativeInt] | None = None
    distributed_timeout_seconds: Annotated[int, Field(ge=30, le=86400)] = 3600
    data_preparation_async: bool = False
    topology_policy: Literal["exact", "preserve_updates"] = "exact"
    bf16_allow_reduced_precision_reduction: bool = False
    bf16_allow_split_k: bool = True
    cuda_allocator: "CudaAllocator" = Field(default_factory=lambda: CudaAllocator())
    output_root: str

    @model_validator(mode="after")
    def check_devices(self):
        if self.bf16_allow_reduced_precision_reduction and not self.bf16_allow_split_k:
            raise ValueError("BF16 reduced-precision reduction requires bf16_allow_split_k=true")
        if (self.gpu is None) == (self.gpus is None):
            raise ValueError("Specify exactly one of runtime.gpu or runtime.gpus")
        if self.gpus is not None and (len(self.gpus) not in {1, 2, 4, 8} or len(set(self.gpus)) != len(self.gpus)):
            raise ValueError("runtime.gpus requires 1, 2, 4 or 8 distinct physical GPU indexes")
        return self


class CudaAllocator(Strict):
    """Allocator options applied before the first CUDA context is created."""
    expandable_segments: bool = True
    backend: Literal["native", "cudaMallocAsync"] | None = None


class HostOffload(Strict):
    """Opt-in scoped reclamation; includes other process-wide pinned users."""
    enabled: bool = False
    reclaim_policy: Literal["after_use", "bounded_reuse"] = "after_use"
    allocator_budget_gib: PositiveFloat = 2.0
    cache_limit_gib: NonnegativeFloat = 0.25
    reserve_gib: PositiveFloat = 8.0
    reserve_fraction: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.1

    @model_validator(mode="after")
    def check_cache(self):
        if self.cache_limit_gib >= self.allocator_budget_gib:
            raise ValueError("host_offload.cache_limit_gib must be below allocator_budget_gib")
        return self


class MemoryPressure(Strict):
    device_headroom_gib: PositiveFloat = 8.0
    retry_headroom_increment_gib: PositiveFloat = 4.0
    max_retries: Annotated[int, Field(ge=0, le=3)] = 2


class AdamMomentOffload(Strict):
    mode: Literal["disabled", "memory_pressure"] = "disabled"
    backend: Literal["gpu_streamed_fp32_v1"] = "gpu_streamed_fp32_v1"
    host_state_budget_gib: PositiveFloat = 16.0
    node_state_budget_gib: PositiveFloat = 64.0
    tile_pair_mib: Annotated[int, Field(ge=1, le=256)] = 64
    gpu_scratch_budget_mib: PositiveInt = 256
    pinned_staging_budget_mib: PositiveInt = 512
    low_watermark_gib: PositiveFloat = 12.0
    high_watermark_gib: PositiveFloat = 20.0
    restore_stable_windows: PositiveInt = 3
    min_residency_windows: PositiveInt = 3
    max_restore_gib_per_window: PositiveFloat = 2.0

    @model_validator(mode="after")
    def check_budgets(self):
        if self.high_watermark_gib <= self.low_watermark_gib:
            raise ValueError("Adam high watermark must exceed low watermark")
        if self.gpu_scratch_budget_mib < self.tile_pair_mib * 4:
            raise ValueError("Adam scratch must cover input pair and functional temporaries (4 x tile pair)")
        if self.pinned_staging_budget_mib < self.tile_pair_mib + 1:
            raise ValueError("Adam staging must cover one charged tile pair including allocator rounding")
        if self.node_state_budget_gib < self.host_state_budget_gib:
            raise ValueError("Node Adam budget must cover rank budget")
        return self


class Memory(Strict):
    loss_chunk_tokens: NonnegativeInt = 0
    mlp_chunk_tokens: NonnegativeInt = 0
    exact_length: bool = False
    attention_policy: Literal["auto", "flash_only"] = "auto"
    activation_offload: Literal["none", "checkpoint_boundary", "auto", "memory_pressure"] = "none"
    host_budget_gib: PositiveFloat = 64.0
    profile_first_update: bool = False
    auto_offload: "AutoOffload" = Field(default_factory=lambda: AutoOffload())
    oom_retry: "OOMRetry" = Field(default_factory=lambda: OOMRetry())
    memory_telemetry: "MemoryTelemetry" = Field(default_factory=lambda: MemoryTelemetry())
    host_offload: HostOffload = Field(default_factory=HostOffload)
    memory_pressure: MemoryPressure = Field(default_factory=MemoryPressure)
    adam_moment_offload: AdamMomentOffload = Field(default_factory=AdamMomentOffload)

    @model_validator(mode="after")
    def check_pressure(self):
        adam = self.adam_moment_offload
        if adam.mode != "disabled":
            if not self.host_offload.enabled:
                raise ValueError("Adam offload requires managed host_offload.enabled")
            if adam.pinned_staging_budget_mib / 1024 >= self.host_offload.allocator_budget_gib:
                raise ValueError("Adam pinned staging is a subset of the process allocator budget")
        if self.activation_offload == "memory_pressure" and not self.host_offload.enabled:
            raise ValueError("memory_pressure requires host_offload.enabled with an actual allocation budget")
        return self


class Supervision(Strict):
    mode: Literal["labels", "explicit_mask", "uniform"] = "labels"
    max_targets_per_sample: PositiveInt | None = None

    @model_validator(mode="after")
    def check_selection(self):
        if (self.mode == "uniform") != (self.max_targets_per_sample is not None):
            raise ValueError("Only uniform supervision requires max_targets_per_sample; other modes require null")
        return self


class AutoOffload(Strict):
    prompt_tokens_threshold: NonnegativeInt = 49152
    sequence_tokens_threshold: NonnegativeInt = 49152
    device_headroom_gib: PositiveFloat = 8.0


class OOMRetry(Strict):
    enabled: bool = False
    max_retries: Annotated[int, Field(ge=0, le=1)] = 1


class MemoryTelemetry(Strict):
    enabled: bool = True
    granularity: Literal["step", "phase"] = "phase"


class Training(Strict):
    max_updates: PositiveInt
    gradient_accumulation_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonnegativeFloat
    grad_clip: PositiveFloat
    scheduler: Literal["constant", "linear"]
    warmup_updates: NonnegativeInt = 0
    checkpoint_every_updates: PositiveInt
    memory: Memory = Field(default_factory=Memory)
    supervision: Supervision = Field(default_factory=Supervision)
    checkpoint_policy: Literal["layer", "selective_mlp", "layer_group"] = "layer"
    checkpoint_group_layers: PositiveInt = 4

    @model_validator(mode="after")
    def check_warmup(self):
        if self.warmup_updates > self.max_updates:
            raise ValueError("training.warmup_updates cannot exceed max_updates")
        return self


class Validation(Strict):
    fixture_id: Literal["causal-text-smoke-v1"]
    max_sequence_length: Annotated[int, Field(ge=8, le=262144)]
    stop_after_update: PositiveInt | None
    skip_checkpoint_export: bool = False
    capacity_microsteps: PositiveInt | None = None


class Algorithm(Strict):
    name: Literal["offline_grpo_k3_v1"]
    dataset_root: str
    dataset_id: str
    require_data_ready: Literal[True] = True
    group_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_binding(self):
        identifier(self.dataset_id)
        if len(self.group_ids) != len(set(self.group_ids)):
            raise ValueError("Duplicate selected full-policy parent ID")
        for group_id in self.group_ids:
            identifier(group_id)
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must be nonempty")
        return self


class ProbabilityArtifact(Strict):
    scope: Literal["rl-replay", "rl-probability"]
    root: str
    id: str

    @model_validator(mode="after")
    def check_identity(self):
        identifier(self.id)
        path = Path(self.root)
        if not path.is_absolute() or ".." in path.parts or path == Path(path.anchor):
            raise ValueError("Probability artifact root must be an explicit absolute directory")
        return self


class ModelForwardCache(Strict):
    source: Literal["model_forward"] = "model_forward"


class ArtifactReferenceCache(Strict):
    source: Literal["probability_artifact"]
    artifact: ProbabilityArtifact
    dtype_compatibility: Literal["same_dtype", "fp16_source_bf16_training"]
    initial_diagnostic_branches: NonnegativeInt = 0


class DatasetReferenceCache(Strict):
    source: Literal["dataset_probability"]
    stream: Annotated[str, Field(min_length=1)]
    dtype_compatibility: Literal["same_dtype", "fp16_source_bf16_training"]
    initial_diagnostic_branches: NonnegativeInt = 0


ReferenceCache = Annotated[ModelForwardCache | ArtifactReferenceCache | DatasetReferenceCache,
                           Field(discriminator="source")]


class PrefixReuse(Strict):
    mode: Literal["disabled", "boundary_grad"] = "disabled"


class RatioGuard(Strict):
    enabled: bool = False
    threshold: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 1e-3


class TokenDiagnostics(Strict):
    enabled: bool = False
    compare_full_branch: bool = False
    compare_base: bool = False
    k3_threshold: PositiveFloat = 10.0
    max_tokens_per_branch: Annotated[int, Field(ge=1, le=32)] = 8
    context_tokens: Annotated[int, Field(ge=0, le=32)] = 12


class Grpo(Strict):
    token_diagnostics: TokenDiagnostics = Field(default_factory=TokenDiagnostics)
    ratio_guard: RatioGuard = Field(default_factory=RatioGuard)
    prefix_reuse: PrefixReuse = Field(default_factory=PrefixReuse)
    advantage_policy: Literal["historical_topk_zero_sum_v1", "consensus_top_half_zero_sum_v1"] = "historical_topk_zero_sum_v1"
    group_batch_size: PositiveInt = 4
    parallel_branch_chunk_size: Literal[16] | None = None
    parallel_branch_schedule: Literal["distinct_parent_two_pass_v1"] | None = None
    branch_partition_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42
    max_epochs: Literal[1] = 1
    shuffle_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42
    shuffle_algorithm: Literal["stable_random_v1", "prefix_length_pool128_v1"] = "prefix_length_pool128_v1"
    top_k: PositiveInt | None = 4
    top_k_fraction: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    score_scale_floor: PositiveFloat = 0.1
    boundary_tie_sigma: NonnegativeFloat = 0.05
    clip_epsilon: Literal[0.1] = 0.1
    kl_beta: NonnegativeFloat = 0.02
    k3_cap: Annotated[float, Field(gt=0, le=3.4028234663852886e38, allow_inf_nan=False)] | None = None
    reference: Literal["initial_policy"] = "initial_policy"
    probability_mode: Literal["raw_t1"] = "raw_t1"
    reference_cache: ReferenceCache = Field(default_factory=ModelForwardCache)

    @model_validator(mode="after")
    def check_top_k(self):
        if self.shuffle_algorithm == "prefix_length_pool128_v1" and self.group_batch_size > 128:
            raise ValueError("prefix_length_pool128_v1 requires group_batch_size <=128")
        if (self.top_k is None) == (self.top_k_fraction is None):
            raise ValueError("Exactly one of grpo.top_k and grpo.top_k_fraction is required")
        if self.advantage_policy == "consensus_top_half_zero_sum_v1" and (
                self.top_k is not None or self.top_k_fraction != 0.5 or self.boundary_tie_sigma != 0):
            raise ValueError("Consensus top-half requires top_k=null, top_k_fraction=0.5 and boundary_tie_sigma=0")
        return self


class Preflight(Strict):
    verify_identity: bool = False
    audit_native_tokens: bool = False
    validate_dataset: bool = False
    verify_probabilities: bool = False
    verify_trajectory_integrity: bool = False
    verify_checkpoint: bool = False
    check_resources: bool = False
    verify_model_state: bool = False
    run_diagnostics: bool = False


class PreparedBinding(Strict):
    id: str
    root: str
    manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    receipt_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def check_id(self):
        identifier(self.id)
        return self


class Config(Strict):
    schema_version: Literal[1, 2]
    run: Run
    model: Model
    lora: Lora | None = None
    tuning: "Tuning | None" = None
    execution: "Execution | None" = None
    optimizer: "Optimizer | None" = None
    runtime: Runtime
    training: Training
    validation: Validation
    algorithm: Algorithm | None = None
    grpo: Grpo | None = None
    preflight: Preflight = Field(default_factory=Preflight)
    prepared_data: PreparedBinding | None = None

    @model_validator(mode="after")
    def check_stop(self):
        full = self.tuning is not None and self.tuning.mode == "full"
        if self.run.start.optimizer_backend_migration is not None and (
                not full or self.run.start.mode != "resume" or
                self.training.memory.adam_moment_offload.mode != "memory_pressure"):
            raise ValueError("Optimizer backend migration requires full/FSDP2 resume into streamed AdamW")
        if self.training.memory.adam_moment_offload.mode != "disabled" and not full:
            raise ValueError("Adam moment offload is currently integrated only with full/FSDP2")
        if self.validation.skip_checkpoint_export and (not full or self.validation.stop_after_update is None):
            raise ValueError("No-export validation requires full/FSDP2 and an explicit stop_after_update")
        if self.validation.capacity_microsteps is not None and (
                not full or not self.validation.skip_checkpoint_export
                or self.training.memory.adam_moment_offload.mode == "disabled"):
            raise ValueError("Capacity microsteps require full/FSDP2 streamed Adam and no-export validation")
        if self.run.start.config_migration is not None and (
                full or self.run.start.mode != "resume" or self.algorithm is None):
            raise ValueError("Config migration requires complete-checkpoint LoRA GRPO resume")
        if self.algorithm and self.algorithm.group_ids and not full:
            raise ValueError("Explicit algorithm.group_ids currently requires full-policy prepared training")
        if self.schema_version == 1:
            if self.lora is None or any(v is not None for v in (self.tuning, self.execution, self.optimizer)):
                raise ValueError("schema 1 requires LoRA and forbids schema 2 fields")
        elif self.tuning is None:
            raise ValueError("schema 2 requires an explicit tuning mode")
        elif full:
            if self.lora is not None or self.execution is None or self.optimizer is None:
                raise ValueError("full requires execution/optimizer and forbids LoRA fields")
            if self.algorithm is None or self.grpo is None:
                raise ValueError("full requires offline GRPO")
            if self.runtime.bf16_allow_reduced_precision_reduction or not self.runtime.bf16_allow_split_k:
                raise ValueError("FSDP2 milestone fixes BF16 numerics: reduced_precision_reduction=false, split_k=true")
            if (self.runtime.topology_policy != "exact" or self.runtime.data_preparation_async
                    or self.run.start.mode == "append_resume"):
                raise ValueError("FSDP2 requires exact topology, prepared data and no append_resume")
            split = self.grpo.parallel_branch_chunk_size is not None
            if not split and (self.grpo.prefix_reuse.mode != "disabled"
                    or self.grpo.parallel_branch_schedule is not None):
                raise ValueError("FSDP2 requires complete branches and disabled prefix reuse")
            if split and (self.grpo.prefix_reuse.mode != "disabled"
                    or len(self.runtime.gpus or []) not in {2, 4, 8}
                    or self.grpo.group_batch_size != 8):
                raise ValueError("Full FSDP2 8x16 requires 2/4/8 GPUs, 8 parents per update and disabled prefix reuse")
            memory = self.training.memory
            if (memory.oom_retry.enabled or memory.mlp_chunk_tokens
                    or self.training.checkpoint_policy != "layer"):
                raise ValueError("FSDP2 requires layer checkpointing, no retry/MLP chunking")
            if memory.activation_offload != "none" and not memory.host_offload.enabled:
                raise ValueError("Full activation offload requires managed host_offload.enabled")
            if self.validation.max_sequence_length > 2048 and not memory.host_offload.enabled:
                raise ValueError("Full length >2048 requires managed host offload; capacity remains unverified")
            if self.grpo.token_diagnostics.compare_base or self.grpo.token_diagnostics.compare_full_branch:
                raise ValueError("FSDP2 token diagnostics cannot run adapter/base comparison forwards")
            if self.run.start.mode == "warm_start" and self.grpo.reference_cache.source != "model_forward":
                raise ValueError("full warm_start builds a new model-forward anchor")
        elif self.lora is None or self.execution is not None or self.optimizer is not None:
            raise ValueError("schema 2 LoRA requires lora and uses the existing execution/optimizer")
        if self.runtime.gpus and len(self.runtime.gpus) > 1 and self.algorithm is None:
            raise ValueError("Multi-GPU training currently requires offline GRPO")
        if self.run.start.mode == "append_resume" and self.algorithm is None:
            raise ValueError("append_resume requires offline GRPO")
        if self.prepared_data is not None and self.algorithm is None:
            raise ValueError("prepared_data requires offline GRPO")
        stop = self.validation.stop_after_update
        if stop is not None and stop >= self.training.max_updates:
            raise ValueError("stop_after_update must precede the complete schedule target")
        memory = self.training.memory
        if (memory.activation_offload in {"auto", "memory_pressure"} or memory.oom_retry.enabled) and self.algorithm is None:
            raise ValueError("Adaptive offload and OOM retry currently require offline GRPO")
        if memory.oom_retry.enabled and self.training.checkpoint_policy not in {"layer", "layer_group"}:
            raise ValueError("OOM retry requires checkpoint boundary offload capability")
        if self.training.supervision.mode != "labels" and not memory.loss_chunk_tokens:
            raise ValueError("Sparse supervision requires loss_chunk_tokens > 0 before vocabulary projection")
        if self.lora is not None and self.lora.dropout != 0:
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
        if self.model.dtype == "float32":
            if self.validation.max_sequence_length > 2048 or memory.attention_policy == "flash_only":
                raise ValueError("FP32 currently supports length <=2048 and attention_policy=auto; Flash requires BF16")
            if self.grpo and self.grpo.reference_cache.source != "model_forward":
                raise ValueError("FP32 requires a model_forward anchor; external probability precision is not validated")
        if self.model.gdn_backend == "fla_chunk" and self.model.dtype != "bfloat16":
            raise ValueError("fla_chunk requires a BF16 model")
        if self.grpo:
            split = self.grpo.parallel_branch_chunk_size is not None
            protocol = self.grpo.parallel_branch_schedule is not None
            count = len(self.runtime.gpus or [self.runtime.gpu])
            if split != protocol:
                raise ValueError("8x16 requires both parallel_branch_chunk_size=16 and distinct_parent_two_pass_v1")
            portable = self.runtime.topology_policy == "preserve_updates"
            if portable and (count not in {2, 4, 8} or self.run.start.mode not in {"fresh", "resume"}):
                raise ValueError("preserve_updates requires 2/4/8 GPUs and fresh or resume")
            if count == 8 and not split and not portable and not full:
                raise ValueError("8 GPUs require the explicit distinct-parent 8x16 execution protocol")
            required_prefix = "disabled" if full else "boundary_grad"
            if split and ((count not in {2, 4, 8} if portable or full else count != 8) or self.runtime.gpus is None
                          or self.grpo.group_batch_size != 8
                          or self.grpo.prefix_reuse.mode != required_prefix):
                expected = "disabled prefix reuse" if full else "boundary_grad"
                raise ValueError(f"8x16 requires the supported topology, group_batch_size=8 and {expected}")
            if split and self.run.start.mode == "append_resume":
                raise ValueError("8x16 supports exact resume; append_resume execution schedules are not supported")
        elif self.runtime.topology_policy != "exact":
            raise ValueError("preserve_updates requires offline GRPO")
        if self.grpo and self.grpo.prefix_reuse.mode == "boundary_grad":
            if self.training.checkpoint_policy != "layer":
                raise ValueError("boundary_grad requires layer checkpointing")
        return self


class Tuning(Strict):
    mode: Literal["lora", "full"]
    scope: Literal["language_model"] = "language_model"
    freeze_vision: Literal[True] = True


class Execution(Strict):
    backend: Literal["fsdp2"]
    sharding: Literal["full_shard"]
    topology_policy: Literal["exact"]
    checkpoint_format: Literal["rl-full-checkpoint-v1"] = "rl-full-checkpoint-v1"
    parameter_dtype: Literal["float32"] = "float32"
    reduce_dtype: Literal["float32"] = "float32"


class Optimizer(Strict):
    name: Literal["adamw"]
    master_weights_dtype: Literal["float32"]
    states_dtype: Literal["float32"]
    weight_decay_policy: Literal["exclude_norm_bias_embedding_v1"] = "exclude_norm_bias_embedding_v1"


Config.model_rebuild()


def is_full(config):
    return config.tuning is not None and config.tuning.mode == "full"


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
    if config.run.start.extension_manifest is not None:
        config.run.start.extension_manifest = str(absolute(project_root, config.run.start.extension_manifest))
    if config.prepared_data is not None:
        config.prepared_data.root = str(absolute(project_root, config.prepared_data.root))
    attempt_directory(config)
    return config, source


def schedule_contract(config):
    """Only attempt metadata, device/storage, start and pause controls may differ on resume."""
    contract = {
        "schema_version": config.schema_version,
        "preflight": config.preflight.model_dump(),
        "seed": config.run.seed,
        "model": config.model.model_dump(exclude={"path"}),
        **({"lora": config.lora.model_dump()} if config.lora is not None else {}),
        "training": {**config.training.model_dump(), "memory": memory_contract(config.training.memory)},
        "validation": config.validation.model_dump(exclude={"stop_after_update", "skip_checkpoint_export", "capacity_microsteps"}),
    }
    if config.training.warmup_updates == 0:
        contract["training"].pop("warmup_updates", None)
    if config.schema_version == 2:
        contract["tuning"] = config.tuning.model_dump()
        if is_full(config):
            contract.update(execution=config.execution.model_dump(), optimizer=config.optimizer.model_dump())
    if config.algorithm:
        contract.update(algorithm=config.algorithm.model_dump(), grpo=config.grpo.model_dump(exclude={"reference_cache", "token_diagnostics"}))
        if config.grpo.parallel_branch_chunk_size is None:
            for key in ("parallel_branch_chunk_size", "parallel_branch_schedule", "branch_partition_seed"):
                contract["grpo"].pop(key, None)
        # The immutable prepared segment binds the shuffle algorithm and order.
        # Preserve compatibility with previously sealed schedules.
        contract["grpo"].pop("shuffle_algorithm", None)
        if not config.grpo.ratio_guard.enabled:
            contract["grpo"].pop("ratio_guard")
        if config.grpo.k3_cap is None:
            contract["grpo"].pop("k3_cap")
        if config.grpo.prefix_reuse.mode == "disabled":
            contract["grpo"].pop("prefix_reuse")
        if config.grpo.advantage_policy == "historical_topk_zero_sum_v1":
            contract["grpo"].pop("advantage_policy")
        # Preserve the legacy schedule byte-for-byte for the default model-forward path.
        if config.grpo.reference_cache.source != "model_forward":
            contract["grpo"]["reference_cache"] = config.grpo.reference_cache.model_dump()
    # Historical reference schedules predate this explicit selector. New
    # execution identity is frozen separately; retain read-only artifact hashes.
    if config.model.gdn_backend == "reference_pytorch":
        contract["model"].pop("gdn_backend", None)
    if config.algorithm is not None and not config.algorithm.group_ids:
        contract["algorithm"].pop("group_ids", None)
    return contract


def memory_contract(memory):
    """Keep default operational fields compatible with pre-existing checkpoints."""
    result = memory.model_dump()
    defaults = Memory()
    for name in ("auto_offload", "oom_retry", "memory_telemetry", "host_offload", "memory_pressure", "adam_moment_offload"):
        if getattr(memory, name) == getattr(defaults, name):
            result.pop(name, None)
    if memory.host_offload.enabled or memory.activation_offload == "memory_pressure":
        from RL.allocator import HOST_POLICY_VERSION, HOST_NATIVE_CACHE_MB
        result["offload_policy_version"] = HOST_POLICY_VERSION
        result["host_allocator_backend"] = "torch_size_separated"
        result["host_native_cache_max_block_bytes"] = HOST_NATIVE_CACHE_MB * 2**20
    return result
