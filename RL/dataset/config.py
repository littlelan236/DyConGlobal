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
    max_num_seqs: PositiveInt = 2
    max_num_batched_tokens: PositiveInt = 4096
    max_model_len: PositiveInt = 32768
    kv_cache_memory_bytes: PositiveInt = 8589934592
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    subprocess_seconds: PositiveInt = 86400
    @model_validator(mode="after")
    def validate_resources(self):
        if not self.gpus or len(set(self.gpus)) != len(self.gpus) or len(self.gpus) != self.tensor_parallel_size: raise ValueError("Declare one GPU per TP rank")
        return self
class ReplayStorage(Strict):
    output_root: str
    max_shard_bytes: Annotated[int, Field(ge=1024)] = 268435456
class ReplayConfig(Strict):
    schema_version: Literal[1]
    replay_id: str
    dataset: DatasetBinding
    runtime: Runtime
    storage: ReplayStorage
    @property
    def sampling_semantics(self): return "processed_sampling_v1"
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
        cfg.dataset.root=str(path(project_root,cfg.dataset.root)); cfg.storage.output_root=str(path(project_root,cfg.storage.output_root))
        # Resolving a venv Python symlink loses the interpreter's environment.
        raw_python = Path(cfg.runtime.python).expanduser()
        if '..' in raw_python.parts: raise ValueError('Invalid interpreter path')
        cfg.runtime.python = str(raw_python.absolute() if raw_python.is_absolute() else Path(project_root).absolute()/raw_python)
    else:
        if cfg.sources.binding_file: cfg.sources.binding_file=str(path(project_root,cfg.sources.binding_file))
        cfg.storage.dataset_root=str(path(project_root,cfg.storage.dataset_root)); cfg.storage.cache_root=str(path(project_root,cfg.storage.cache_root))
    return cfg,source
