"""One editable config; no implicit overlay or second split definition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.io import artifact_roots, hash_json, identifier, owned_path, read_json, run_path, sha256, write_json
from difficulty.probe_config import RidgeProbe


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Environment(Strict):
    docker_host: str
    docker_executable: str
    docker_data_root: str | None = None

    @model_validator(mode="after")
    def validate_docker_data_root(self):
        if self.docker_data_root is not None:
            path = Path(self.docker_data_root)
            if not path.is_absolute() or path == Path("/") or ".." in path.parts:
                raise ValueError("environment.docker_data_root must be an absolute non-root path")
        return self
    wheelhouse: str | None
    cwd: str
    user: str
    platform: str
    locale: str
    constraints: list[str]
    repo_pins: dict[str, list[str]]
    variables: dict[str, str]
    network_fallback: bool
    network_mode: str = "none"
    requirements_cache: str


class Model(Strict):
    path: str | None
    fingerprint: str | None
    served_name: str | None
    trust_remote_code: bool


class Serving(Strict):
    base_url: str
    api_key_env: str
    management: Literal["external", "managed"]
    python: str
    gpus: list[int | str]
    tensor_parallel_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_devices(self):
        from runtime.serving.resources import normalize_devices

        devices = normalize_devices(self.gpus)
        if len(devices) != self.tensor_parallel_size:
            raise ValueError(
                "serving.gpus count must equal serving.tensor_parallel_size"
            )
        return self
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(gt=0)
    startup_seconds: int = Field(gt=0)
    shutdown_seconds: int = Field(gt=0)
    tool_parser: str | None
    reasoning_parser: str | None
    language_model_only: bool
    dtype: Literal["float16", "bfloat16"] = "float16"
    kv_cache_memory_bytes: int | None = Field(default=None, gt=0)
    max_num_seqs: int = Field(default=16, gt=0)
    max_num_batched_tokens: int = Field(default=1024, gt=0)


class Generation(Strict):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1.0, gt=0)
    max_tokens: int = Field(gt=0)
    seed: int
    enable_thinking: bool
    preserve_thinking: bool
    thinking_token_budget: int | None = Field(default=None, ge=0)


class Agent(Strict):
    step_limit: int = Field(gt=0)
    cost_limit: float = Field(ge=0)
    max_consecutive_format_errors: int = Field(gt=0)
    loop_repeat_limit: int | None = Field(default=None, ge=2)
    loop_max_period: int = Field(default=7, ge=1, le=7)
    system_template: str
    instance_template: str
    observation_template: str
    format_error_template: str


class Images(Strict):
    load_workers: int = Field(gt=0, le=8)
    reserve_bytes: int = Field(ge=0)
    expansion_factor: float = Field(ge=1)
    minimum_load_bytes: int = Field(ge=0)
    max_reserved_bytes: int = Field(gt=0)
    load_seconds: int = Field(gt=0)
    admission_seconds: int = Field(gt=0)
    cleanup: bool


class LimitsRetry(Strict):
    enabled: bool = False
    retry_statuses: list[Literal["LimitsExceeded", "reasoning_loop"]] = Field(
        default_factory=lambda: ["LimitsExceeded", "reasoning_loop"])
    seed_policy: Literal["repetition_round_offset_v1"] = "repetition_round_offset_v1"

    @model_validator(mode="after")
    def exact_statuses(self):
        if len(self.retry_statuses) != 2 or set(self.retry_statuses) != {"LimitsExceeded", "reasoning_loop"}:
            raise ValueError("limits_retry.retry_statuses must contain exactly LimitsExceeded and reasoning_loop")
        return self


class Cancellation(Strict):
    control_request_seconds: float = Field(default=3.0, gt=0, le=30)
    cooperative_grace_seconds: float = Field(default=10.0, gt=0, le=300)
    term_grace_seconds: float = Field(default=30.0, gt=0, le=300)
    kill_verify_seconds: float = Field(default=5.0, gt=0, le=60)
    cleanup_seconds: float = Field(default=15.0, gt=0, le=300)


class Execution(Strict):
    cancellation: Cancellation = Field(default_factory=Cancellation)
    limits_retry: LimitsRetry = Field(default_factory=LimitsRetry)
    rollout_workers: int = Field(gt=0)
    harness_workers: int = Field(gt=0)
    infrastructure_retry_limit: int = Field(ge=0)
    pipeline_wall_seconds: int = Field(ge=0)


class Timeouts(Strict):
    container_start_seconds: int = Field(gt=0)
    container_lifetime_seconds: int = Field(gt=0)
    bootstrap_seconds: int = Field(gt=0)
    tool_command_seconds: int = Field(gt=0)
    model_seconds: int = Field(gt=0)
    evaluation_seconds: int = Field(gt=0)
    metadata_request_seconds: int = Field(gt=0)


class Enrichment(Strict):
    anchors: list[Literal["think_open", "think_close"]]
    close_occurrence: Literal["first", "all"]
    token_evidence: Literal["exact", "reconstructed"]


class VLLMHidden(Strict):
    python: str = ".venvs/latent-vllm/bin/python"
    gpus: list[int] = []
    tensor_parallel_size: int = Field(default=1, gt=0)
    max_num_seqs: int = Field(default=16, gt=0)
    max_num_batched_tokens: int = Field(default=4096, gt=0)
    max_model_len: int = Field(default=32768, gt=1)
    max_inflight_requests: int = Field(default=32, gt=0)
    gpu_memory_utilization: float = Field(default=0.8, gt=0, le=1)
    kv_cache_memory_bytes: int | None = Field(default=None, gt=0)
    allow_shared_gpus: bool = False
    min_free_gpu_bytes: int = Field(default=0, ge=0)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    moe_backend: Literal["auto", "triton"] = "auto"
    capture_budget_bytes: int = Field(default=268435456, gt=0)
    fetch_bytes: int = Field(default=16777216, gt=0)

    @model_validator(mode="after")
    def unique_devices(self):
        if len(set(self.gpus)) != len(self.gpus) or any(g < 0 for g in self.gpus):
            raise ValueError("vLLM GPUs must be unique nonnegative physical indices")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("vLLM token budget must accommodate max_num_seqs")
        if self.allow_shared_gpus and (not self.min_free_gpu_bytes or not self.kv_cache_memory_bytes):
            raise ValueError("Shared replay GPUs require an explicit free-memory floor and KV byte budget")
        return self


class Hidden(Strict):
    python: str
    dtype: Literal["float32", "float16", "bfloat16"]
    device: str
    chunk_size: int = Field(gt=0)
    layers: list[int]
    adapter: Literal["vllm_eager"]
    vllm: VLLMHidden = Field(default_factory=VLLMHidden)


class Benchmark(Strict):
    partition: Literal["train", "val", "test", "excluded"] | None


class CaptureExport(Strict):
    enabled: bool = True
    root: str = "hidden/online"
    format: Literal["npz"] = "npz"
    queue_size: int = Field(default=256, gt=0)
    flush_interval_ms: int = Field(default=100, ge=0)
    spool_root: str = "hidden/spool"
    spool_max_bytes: int = Field(default=53687091200, gt=0)
    spool_high_watermark: float = Field(default=0.8, gt=0, lt=1)
    spool_low_watermark: float = Field(default=0.6, gt=0, lt=1)
    shard_max_bytes: int = Field(default=134217728, gt=0)
    upload_workers: int = Field(default=2, gt=0, le=4)
    upload_max_elapsed_seconds: int = Field(default=600, gt=0)

    @model_validator(mode="after")
    def storage_limits(self):
        for name in ("root", "spool_root"):
            path = Path(getattr(self, name))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"capture.export.{name} must be project relative")
        if self.spool_low_watermark > self.spool_high_watermark:
            raise ValueError("spool watermarks require low <= high")
        if 2 * self.shard_max_bytes >= self.spool_max_bytes:
            raise ValueError("spool must accommodate raw and sealed shard simultaneously")
        return self


class Capture(Strict):
    enabled: bool = False
    backend: Literal["easysteer"] = "easysteer"
    mode: Literal["online"] = "online"
    graph_mode: Literal["FULL"] = "FULL"
    # Zero delegates prompt chunking to the native scheduler.
    prefill_token_chunk_size: int = Field(default=0, ge=0)
    allow_unsupported_gdn: bool = False
    anchors: list[Literal["think_open", "reasoning_newline", "think_close"]] = [
        "think_open",
        "reasoning_newline",
        "think_close",
    ]
    newline_policy: Literal["all", "double_only"] = "all"
    newline_token_source: Literal["tokenizer"] = "tokenizer"
    layers: list[int] = []
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    prefix_cache_policy: Literal["safe_boundary"] = "safe_boundary"
    capture_stage: Literal["post_steering"] = "post_steering"
    failure_policy: Literal["record_and_continue", "block_rollout"] = "record_and_continue"
    control_host: str = "127.0.0.1"
    control_port: int = Field(default=0, ge=0, le=65535)
    control_seconds: int = Field(default=30, gt=0)
    finalize_seconds: int = Field(default=120, gt=0)
    export: CaptureExport = Field(default_factory=CaptureExport)


class Steering(Strict):
    backend: Literal["easysteer"]
    enabled: bool
    payload: str | None


class Paths(Strict):
    nas_root: str
    artifact_roots: dict[str, str] = Field(default_factory=dict)
    artifact_scan_roots: list[str] = Field(default_factory=list)


class Difficulty(Strict):
    estimator: None
    ridge_probe: RidgeProbe | None = None


class Action(Strict):
    taxonomy: None


class ParallelReasoningLoop(Strict):
    enabled: bool = False
    include_post_close: bool = False
    min_generated_tokens: int = Field(default=2048, ge=1, le=81920)
    repeat_count: int = Field(default=8, ge=2, le=32)
    max_period_tokens: int = Field(default=2048, ge=1, le=8192)
    min_repeated_tokens: int = Field(default=1024, ge=1, le=65536)
    check_interval_tokens: int = Field(default=64, ge=1, le=1024)


class ParallelLoopEscape(Strict):
    enabled: bool = False


class ParallelConsensus(Strict):
    retention_ratio: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    min_retained_branches: int = Field(default=2, ge=2)
    rule: Literal["exact_action"] = "exact_action"
    tie_policy: Literal["estimator_score"] = "estimator_score"


class Parallel(Strict):
    """Optional branch selection; the predictor is supplied by a later adapter."""

    enabled: bool = False
    mode: Literal["parallel", "parallel_baseline", "parallel_consensus"] = "parallel"
    branch_count: int = Field(default=2, gt=0, le=32)
    policy: Literal["first_step", "every_k_steps", "every_step"] = "first_step"
    every_k_steps: int | None = Field(default=None, gt=0)
    estimator_ref: str | None = None
    max_inflight_branches: int = Field(default=16, gt=0)
    dynamic_branch_admission: bool = False
    admission_timeout_seconds: int = Field(default=1800, gt=0, le=28800)
    score_timeout_ms: int = Field(default=200, ge=1, le=30000)
    decision_ttl_seconds: int = Field(default=28860, gt=0)
    persist_all_branch_close: bool = True
    save_sites: list[int] = []
    fallback_policy: Literal["first_valid", "stop"] = "first_valid"
    validation_repetitions: int = Field(default=0, ge=0, le=10)
    consensus: ParallelConsensus | None = None
    reasoning_loop: ParallelReasoningLoop = Field(default_factory=ParallelReasoningLoop)
    loop_escape: ParallelLoopEscape = Field(default_factory=ParallelLoopEscape)

    @model_validator(mode="after")
    def validate_policy(self):
        if self.mode == "parallel_consensus" and self.enabled:
            if self.branch_count <= 1:
                raise ValueError("parallel consensus requires branch_count > 1")
            if not self.estimator_ref:
                raise ValueError("parallel consensus requires estimator_ref")
            if self.consensus is None:
                self.consensus = ParallelConsensus()
            if self.consensus.min_retained_branches > self.branch_count:
                raise ValueError("parallel consensus min_retained_branches is invalid")
            if self.fallback_policy != "stop":
                raise ValueError("parallel consensus requires fallback_policy=stop")
        if self.mode == "parallel_baseline":
            if not self.enabled:
                raise ValueError("parallel baseline requires enabled=true")
            if self.branch_count <= 1:
                raise ValueError("parallel baseline requires branch_count > 1")
            if not self.reasoning_loop.enabled:
                raise ValueError("parallel baseline requires reasoning_loop.enabled=true")
            if not self.estimator_ref:
                raise ValueError("parallel baseline requires estimator_ref")
        if self.policy == "every_k_steps" and self.every_k_steps is None:
            raise ValueError("parallel.every_k_steps is required for every_k_steps")
        if self.policy != "every_k_steps" and self.every_k_steps is not None:
            raise ValueError("parallel.every_k_steps is only valid for every_k_steps")
        if self.enabled and self.branch_count > 1 and not self.persist_all_branch_close:
            raise ValueError("parallel rollout requires all branch close hidden to be persisted")
        if len(set(self.save_sites)) != len(self.save_sites) or any(x < 0 for x in self.save_sites):
            raise ValueError("parallel.save_sites must contain unique nonnegative sites")
        return self


class SkillsBenchImages(Strict):
    source_root: str
    revision: str
    store: str
    workers: int = Field(gt=0, le=16)
    build_seconds: int = Field(gt=0)
    reserve_bytes: int = Field(ge=0)
    host_proxy: str | None
    build_proxy: str
    manage_build_proxy: bool = False
    task_compatibility: dict[str, dict] = Field(default_factory=dict)
    verified_wheels: dict[str, list[dict]] = Field(default_factory=dict)
    source_overrides: dict[str, dict] = Field(default_factory=dict)
    python_read_seconds: int = Field(default=60, gt=0)
    python_index: str
    ubuntu_mirror: str
    debian_mirror: str
    mirror_no_proxy: str
    docker_context: str
    builder_name: str


class Config(Strict):
    schema_version: Literal[1]
    paths: Paths
    benchmark: Benchmark
    model: Model
    serving: Serving
    generation: Generation
    agent: Agent
    environment: Environment
    images: Images
    skillsbench_images: SkillsBenchImages | None = None
    execution: Execution
    timeouts: Timeouts
    enrichment: Enrichment
    hidden: Hidden
    difficulty: Difficulty
    action: Action
    steering: Steering
    capture: Capture = Field(default_factory=Capture)
    parallel: Parallel = Field(default_factory=Parallel)

    @model_validator(mode="after")
    def cross_fields(self):
        if self.steering.enabled != bool(self.steering.payload):
            raise ValueError("Enabled steering requires a payload; disabled steering must not have one")
        if self.timeouts.bootstrap_seconds >= self.timeouts.container_lifetime_seconds:
            raise ValueError("Bootstrap must fit container lifetime")
        if len(set(self.hidden.layers)) != len(self.hidden.layers) or any(x < 0 for x in self.hidden.layers):
            raise ValueError("Hidden layers must be unique nonnegative indices")
        if len(set(self.capture.layers)) != len(self.capture.layers) or any(
            x < 0 for x in self.capture.layers
        ):
            raise ValueError("Capture layers must be unique nonnegative indices")
        if self.capture.enabled and self.capture.prefix_cache_policy != "safe_boundary":
            raise ValueError("Online capture requires the safe_boundary cache policy")
        if self.capture.enabled and self.serving.management != "managed":
            raise ValueError("Online capture requires managed serving")
        if not self.capture.anchors or len(set(self.capture.anchors)) != len(self.capture.anchors):
            raise ValueError("Capture anchors must be nonempty and unique")
        if self.capture.enabled and self.capture.prefill_token_chunk_size != 0:
            raise ValueError("Production online capture requires native prefill_token_chunk_size=0")
        if self.parallel.enabled and self.parallel.branch_count > 1 and self.parallel.max_inflight_branches < self.parallel.branch_count:
            raise ValueError("parallel capacity must accommodate one decision")
        if self.parallel.enabled and self.parallel.branch_count > 1 and not self.capture.enabled:
            raise ValueError("parallel rollout requires online capture for branch hidden")
        if self.parallel.enabled and self.parallel.branch_count > 1:
            if not self.capture.export.enabled or "think_close" not in self.capture.anchors:
                raise ValueError("parallel requires persistent think_close capture")
            if not self.generation.enable_thinking or not self.parallel.estimator_ref:
                raise ValueError("parallel requires thinking and external estimator_ref")
            if not self.parallel.save_sites or not set(self.parallel.save_sites) <= set(self.capture.layers):
                raise ValueError("parallel.save_sites must be a nonempty subset of capture.layers")
            if self.parallel.max_inflight_branches > self.serving.max_num_seqs:
                raise ValueError("parallel capacity must not exceed serving.max_num_seqs")
            if self.parallel.decision_ttl_seconds <= self.timeouts.model_seconds:
                raise ValueError("parallel reservation TTL must exceed model request timeout")
        return self


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(v, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def resolve(path, overrides=()):
    path = Path(path).resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    # Optional defaults preserve old config files while allowing dotted overrides.
    # Existing keys still undergo strict validation; unknown keys are not discarded.
    hidden = raw.get("hidden", {})
    options = VLLMHidden().model_dump()
    options.update(hidden.get("vllm", {}))
    hidden["vllm"] = options
    for item in overrides:
        dotted, value = item.split("=", 1)
        target = raw
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown config field: {dotted}")
        target[keys[-1]] = yaml.safe_load(value)
    cfg = Config.model_validate(raw).model_dump()
    if "skillsbench_images" not in raw:
        cfg.pop("skillsbench_images", None)
    if "cancellation" not in raw["execution"]:
        cfg["execution"].pop("cancellation", None)
    if "limits_retry" not in raw["execution"]:
        cfg["execution"].pop("limits_retry", None)
    # Adding host storage configuration must not alter legacy config identities.
    if "docker_data_root" not in raw["environment"]:
        cfg["environment"].pop("docker_data_root", None)
    # Adding optional loop protection must not change legacy config identities.
    for field in ("loop_repeat_limit", "loop_max_period"):
        if field not in raw["agent"]:
            cfg["agent"].pop(field)
    # Preserve old snapshot/config identities when the new optional pilot is absent.
    if "ridge_probe" not in raw.get("difficulty", {}):
        cfg["difficulty"].pop("ridge_probe")
    if "parallel" not in raw:
        cfg.pop("parallel")
    elif "reasoning_loop" not in raw["parallel"]:
        cfg["parallel"].pop("reasoning_loop")
    if "include_post_close" not in raw.get("parallel", {}).get("reasoning_loop", {}):
        cfg.get("parallel", {}).get("reasoning_loop", {}).pop("include_post_close", None)
    if "loop_escape" not in raw.get("parallel", {}):
        cfg.get("parallel", {}).pop("loop_escape", None)
    if "mode" not in raw.get("parallel", {}):
        cfg.get("parallel", {}).pop("mode", None)
    if cfg.get("parallel", {}).get("consensus") is None:
        cfg.get("parallel", {}).pop("consensus", None)
    root = path.parent
    if cfg.get("parallel", {}).get("enabled") and cfg["parallel"]["branch_count"] > 1:
        from runtime.serving.parallel import artifact_contract

        options = cfg["parallel"]
        options["estimator_ref"] = str((root / options["estimator_ref"]).absolute())
        contract = artifact_contract(options["estimator_ref"], model_fingerprint=cfg["model"]["fingerprint"])
        if not set(contract["sites"]) <= set(options["save_sites"]):
            raise ValueError("Predictor sites must be included in parallel.save_sites")
        options["estimator_sha256"] = sha256(options["estimator_ref"])
    cfg["hidden"]["vllm"]["python"] = str((root / cfg["hidden"]["vllm"]["python"]).absolute())
    for section, field in [
        ("model", "path"),
        ("environment", "wheelhouse"),
        ("environment", "requirements_cache"),
        ("serving", "python"),
        ("hidden", "python"),
        ("steering", "payload"),
    ]:
        value = cfg[section][field]
        if value is not None:
            cfg[section][field] = str((root / value).absolute())
    cfg.update(project_root=str(root), config_source=str(path), overrides=list(overrides))
    if cfg["steering"]["enabled"]:
        cfg["steering"]["payload_sha256"] = sha256(cfg["steering"]["payload"])
    cfg["config_source_sha256"] = sha256(path)
    cfg["config_hash"] = hash_json(cfg)
    return cfg


def snapshot(cfg, run_id, selection):
    assert_frozen(cfg)
    run_id = identifier(run_id)
    roots = artifact_roots(cfg, "run")
    target_root = Path(cfg.get("paths", {}).get("artifact_roots", {}).get("run", cfg["project_root"])).resolve()
    run = target_root / "runs" / run_id
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=False)
    # Keep legacy project-root/runs/<id> callers working while the physical run is elsewhere.
    alias = Path(cfg["project_root"]) / "runs" / run_id
    if alias != run and not alias.exists():
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(run, target_is_directory=True)
    frozen = copy.deepcopy(cfg)
    if target_root != Path(cfg["project_root"]).resolve():
        frozen["run_storage_root"] = str(target_root)
        frozen["config_hash"] = hash_json({k: v for k, v in frozen.items() if k != "config_hash"})
    frozen["selection"] = selection
    frozen["selection_sha256"] = hash_json(selection)
    write_json(run / "resolved-config.json", frozen)
    write_json(run / "selection.json", selection)
    return run


def load_snapshot(root, run_id):
    run_id = identifier(run_id)
    candidates = [Path(root).resolve() / "runs" / run_id]
    config_path = Path(root).resolve() / "config.yaml"
    if config_path.exists():
        try:
            raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=UniqueLoader) or {}
        except yaml.YAMLError:
            raw = {}
        for value in raw.get("paths", {}).get("artifact_scan_roots", []):
            candidate = Path(value).expanduser().resolve() / "runs" / run_id
            if candidate not in candidates:
                candidates.append(candidate)
        preferred = raw.get("paths", {}).get("artifact_roots", {}).get("run")
        if preferred:
            candidate = Path(preferred).expanduser().resolve() / "runs" / run_id
            if candidate not in candidates:
                candidates.insert(0, candidate)
    run_dir = next((candidate for candidate in candidates if (candidate / "resolved-config.json").exists()), None)
    if run_dir is None:
        raise FileNotFoundError(f"Run snapshot not found: {run_id}")
    cfg = read_json(run_dir / "resolved-config.json")
    if Path(cfg["project_root"]).resolve() != Path(root).resolve():
        raise ValueError("Run snapshot belongs to another project")
    expected = cfg["config_hash"]
    base = {k: v for k, v in cfg.items() if k not in {"config_hash", "selection", "selection_sha256"}}
    if hash_json(base) != expected:
        raise ValueError("Run configuration snapshot was modified")
    if hash_json(cfg["selection"]) != cfg["selection_sha256"]:
        raise ValueError("Frozen run selection was modified")
    return cfg


def require_model(cfg):
    if not all(cfg["model"][k] for k in ["path", "fingerprint", "served_name"]):
        raise ValueError(
            "Model path, fingerprint and served_name are intentionally unset; configure them first"
        )
    if not Path(cfg["model"]["path"]).is_dir():
        raise FileNotFoundError(cfg["model"]["path"])


def assert_frozen(cfg):
    base = {k: v for k, v in cfg.items() if k not in {"config_hash", "selection", "selection_sha256"}}
    if hash_json(base) != cfg["config_hash"]:
        raise ValueError("Resolved configuration was modified after hashing")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    print(json.dumps(resolve(a.config, a.set), indent=2))
