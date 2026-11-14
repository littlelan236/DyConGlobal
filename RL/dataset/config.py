from pathlib import Path
from typing import Annotated, Literal
import yaml
from pydantic import Field, model_validator
from RL.config import Strict, UniqueLoader
from core.io import identifier

PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]

class Sources(Strict):
    runs: list[str]
    binding_file: str | None = None
    rollout_ids: list[str] = Field(default_factory=list)
    all_submitted: bool = False
    @model_validator(mode="after")
    def validate_selection(self):
        if not self.runs or len(self.runs) != len(set(self.runs)): raise ValueError("Declare distinct nonempty source runs")
        for x in self.runs: identifier(x)
        if sum((self.binding_file is not None, bool(self.rollout_ids), self.all_submitted)) != 1: raise ValueError("Choose one explicit selector")
        if self.rollout_ids and len(self.runs) != 1: raise ValueError("Multiple runs require binding_file")
        for x in self.rollout_ids: identifier(x)
        return self
class Filters(Strict):
    benchmark_family: Literal["swebench"] = "swebench"
    rollout_modes: list[Literal["parallel_consensus"]] = Field(default_factory=lambda: ["parallel_consensus"])
    outcome: Literal["any"] = "any"
class Quality(Strict):
    token_evidence: Literal["native_exact"] = "native_exact"
    group_policy: Literal["complete_native_group_v1"] = "complete_native_group_v1"
    min_branches: Annotated[int, Field(ge=2)] = 2
    alias_policy: Literal["canonical_origin_once_v1"] = "canonical_origin_once_v1"
class Audit(Strict):
    enabled: bool = False
class Supervision(Strict):
    policy: Literal["native_completion_reasoning_action_v1"] = "native_completion_reasoning_action_v1"
class Storage(Strict):
    dataset_root: str
    cache_root: str
    max_shard_bytes: Annotated[int, Field(ge=1024)] = 268435456
    memory_budget_gib: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4.0
    ledger_read_workers: Annotated[int, Field(ge=1, le=32)] = 8
class DatasetConfig(Strict):
    schema_version: Literal[1]
    dataset_id: str
    sources: Sources
    filters: Filters = Field(default_factory=Filters)
    quality: Quality = Field(default_factory=Quality)
    audit: Audit = Field(default_factory=Audit)
    supervision: Supervision = Field(default_factory=Supervision)
    storage: Storage
class DatasetBinding(Strict):
    id: str
    root: str
class Runtime(Strict):
    backend: Literal["native_vllm_trace_v1"] = "native_vllm_trace_v1"
    python: str = ".venvs/latent-vllm/bin/python"
    gpus: list[NonnegativeInt]
    tensor_parallel_size: PositiveInt = 1
    # Independent workers, one physical GPU each.  This is deliberately
    # restricted to TP=1: workers own disjoint complete groups and never share
    # a vLLM engine or a CUDA context.
    parallel_workers: PositiveInt = 1
    max_num_seqs: PositiveInt = 2
    max_num_batched_tokens: PositiveInt = 4096
    max_model_len: PositiveInt = 32768
    kv_cache_memory_bytes: PositiveInt = 8589934592
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    cudagraph_mode: Literal["NONE", "FULL", "FULL_AND_PIECEWISE"] = "FULL"
    cudagraph_capture_sizes: list[PositiveInt] | None = None
    mamba_cache_mode: Literal["none", "align", "all"] = "align"
    subprocess_seconds: PositiveInt = 1209600
    @model_validator(mode="after")
    def validate_resources(self):
        if not self.gpus or len(set(self.gpus)) != len(self.gpus):
            raise ValueError("Declare distinct physical GPUs")
        if self.parallel_workers > 1 and self.tensor_parallel_size != 1:
            raise ValueError("Independent parallel replay workers require tensor_parallel_size=1")
        if len(self.gpus) != self.tensor_parallel_size * self.parallel_workers:
            raise ValueError("Declare one GPU per TP rank and independent worker")
        if self.enforce_eager != (self.cudagraph_mode == "NONE"):
            raise ValueError("enforce_eager must agree with cudagraph_mode NONE versus graph execution")
        if not self.enable_prefix_caching and self.mamba_cache_mode != "none":
            raise ValueError("Prefix caching disabled requires mamba_cache_mode none")
        return self
class ReplayStorage(Strict):
    output_root: str
    max_shard_bytes: Annotated[int, Field(ge=1024)] = 268435456
    commit_every_branches: PositiveInt = 32
    commit_interval_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5.0
class ReplayScheduling(Strict):
    mode: Literal["continuous", "batch"] = "continuous"
    order: Literal["group_locality"] = "group_locality"
    max_active_groups: PositiveInt = 2
    max_inflight_requests: PositiveInt = 32
    max_inflight_tokens: PositiveInt = 2621440
    prefix_strategy: Literal["concurrent", "leader_finished"] = "concurrent"
class ReplayIO(Strict):
    ledger_read_workers: Annotated[int, Field(ge=1, le=32)] = 8
    prefetch_groups: PositiveInt = 2
    prefetch_memory_budget_gib: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 2.0
class CoveragePlanBinding(DatasetBinding):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class ReplaySelection(Strict):
    group_ids: list[str] = Field(default_factory=list)
    coverage_plan: CoveragePlanBinding | None = None
    @model_validator(mode="after")
    def distinct(self):
        if self.coverage_plan and self.group_ids: raise ValueError("coverage_plan and group_ids are mutually exclusive")
        if len(self.group_ids) != len(set(self.group_ids)): raise ValueError("Duplicate selected group ID")
        return self
class ReplayPreflight(Strict):
    enabled: bool = False

class ReplayConfig(Strict):
    schema_version: Literal[2]
    probability_semantics: Literal["raw_t1_output_tokens_v1"] = "raw_t1_output_tokens_v1"
    token_cache: DatasetBinding | None = None
    scheduling: ReplayScheduling = Field(default_factory=ReplayScheduling)
    io: ReplayIO = Field(default_factory=ReplayIO)
    selection: ReplaySelection = Field(default_factory=ReplaySelection)
    # A new immutable replay ID may reuse the committed prefix of an explicitly
    # stopped raw replay. Exact recovery of this ID still freezes all settings.
    resume_from: DatasetBinding | None = None
    @model_validator(mode="before")
    @classmethod
    def migrate(cls, value):
        if isinstance(value, dict) and value.get("schema_version") != 2:
            raise ValueError("Legacy processed replay YAML is read-only: use schema_version 2, raw semantics and a NEW replay_id; never recover a processed ID as raw")
        return value
    replay_id: str
    preflight: ReplayPreflight = Field(default_factory=ReplayPreflight)
    dataset: DatasetBinding
    runtime: Runtime
    storage: ReplayStorage
    @property
    def sampling_semantics(self): return self.probability_semantics
def path(root, value):
    p=Path(value).expanduser()
    if not str(value).strip() or ".." in p.parts: raise ValueError("Invalid path")
    return (p if p.is_absolute() else Path(root)/p).resolve()
def load(file, project_root, replay=False):
    source=path(project_root,file); value=yaml.load(source.read_text(encoding="utf-8"),Loader=UniqueLoader)
    model=ReplayConfig if replay else DatasetConfig; cfg=model.model_validate(value)
    identifier(cfg.replay_id if replay else cfg.dataset_id)
    if replay:
        identifier(cfg.dataset.id)
        if cfg.resume_from:
            identifier(cfg.resume_from.id)
            cfg.resume_from.root = str(path(project_root, cfg.resume_from.root))
            if cfg.resume_from.id == cfg.replay_id and cfg.resume_from.root == str(path(project_root, cfg.storage.output_root)):
                raise ValueError("resume_from requires a new replay ID")
        if cfg.token_cache:
            identifier(cfg.token_cache.id)
            cfg.token_cache.root = str(path(project_root, cfg.token_cache.root))
        if cfg.token_cache:
            identifier(cfg.token_cache.id)
            cfg.token_cache.root = str(path(project_root, cfg.token_cache.root))
        if cfg.selection.coverage_plan:
            identifier(cfg.selection.coverage_plan.id)
            cfg.selection.coverage_plan.root = str(path(project_root, cfg.selection.coverage_plan.root))
        cfg.dataset.root=str(path(project_root,cfg.dataset.root)); cfg.storage.output_root=str(path(project_root,cfg.storage.output_root))
        # Resolving a venv Python symlink loses the interpreter's environment.
        raw_python = Path(cfg.runtime.python).expanduser()
        if '..' in raw_python.parts: raise ValueError('Invalid interpreter path')
        cfg.runtime.python = str(raw_python.absolute() if raw_python.is_absolute() else Path(project_root).absolute()/raw_python)
    else:
        if cfg.sources.binding_file: cfg.sources.binding_file=str(path(project_root,cfg.sources.binding_file))
        cfg.storage.dataset_root=str(path(project_root,cfg.storage.dataset_root)); cfg.storage.cache_root=str(path(project_root,cfg.storage.cache_root))
    return cfg,source


class ProbabilitySource(Strict):
    kind: Literal["dataset_stream", "replay", "probability"]
    id: str
    root: str
    stream: str | None = None
    role: Literal["existing", "backfill"] = "existing"
    @model_validator(mode="after")
    def validate_source(self):
        identifier(self.id)
        if (self.kind == "dataset_stream") != (self.stream is not None):
            raise ValueError("Only dataset_stream requires an explicit stream")
        if self.role == "backfill" and self.kind != "replay":
            raise ValueError("Backfill must be an explicitly planned replay")
        return self

class ProbabilityOutput(DatasetBinding):
    max_shard_bytes: Annotated[int, Field(ge=1024)] = 268435456
    memory_budget_gib: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4.0

class CohortPolicy(Strict):
    allowed_execution_cohorts: list[str] = Field(default_factory=list)
    @model_validator(mode="after")
    def distinct(self):
        import re
        if (len(set(self.allowed_execution_cohorts)) != len(self.allowed_execution_cohorts)
                or any(not re.fullmatch(r"[0-9a-f]{64}", s) for s in self.allowed_execution_cohorts)):
            raise ValueError("Declare distinct exact execution cohort hashes")
        return self

class ProbabilityConfig(Strict):
    schema_version: Literal[1]
    dataset: DatasetBinding
    output: ProbabilityOutput
    plan: DatasetBinding
    coverage_plan: CoveragePlanBinding | None = None
    group_ids: list[str] = Field(default_factory=list)
    sources: list[ProbabilitySource] = Field(default_factory=list)
    allowed_missing_reasons: list[Literal["absent", "recorded_failed"]] = Field(default_factory=lambda: ["absent"])
    cohort_policy: CohortPolicy = Field(default_factory=CohortPolicy)
    expected_scoring_identity: dict | None = None
    deduplicate_identical: bool = False
    allow_partial: bool = False
    merge_policy: Literal["existing_valid_then_backfill"] = "existing_valid_then_backfill"
    @model_validator(mode="after")
    def distinct(self):
        for binding in (self.dataset, self.output, self.plan): identifier(binding.id)
        if len(set(self.group_ids)) != len(self.group_ids): raise ValueError("Duplicate target group ID")
        keys = [(s.kind, s.root, s.id, s.stream) for s in self.sources]
        if len(set(keys)) != len(keys): raise ValueError("Duplicate probability source")
        if sum(s.role == "backfill" for s in self.sources) > 1: raise ValueError("Declare at most one backfill replay")
        return self

def load_probability(file, project_root):
    source = path(project_root, file)
    config = ProbabilityConfig.model_validate(yaml.load(source.read_text(encoding="utf-8"), Loader=UniqueLoader))
    for binding in [config.dataset, config.output, config.plan, *config.sources,
                    *([config.coverage_plan] if config.coverage_plan else [])]:
        binding.root = str(path(project_root, binding.root))
    return config, source
