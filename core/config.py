"""One editable config; no implicit overlay or second split definition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.io import hash_json, identifier, owned_path, read_json, write_json


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Environment(Strict):
    docker_host: str
    docker_executable: str
    wheelhouse: str | None
    cwd: str
    user: str
    platform: str
    locale: str
    constraints: list[str]
    repo_pins: dict[str, list[str]]
    variables: dict[str, str]
    network_fallback: bool
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
    gpus: list[int]
    tensor_parallel_size: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(gt=0)
    startup_seconds: int = Field(gt=0)
    shutdown_seconds: int = Field(gt=0)
    tool_parser: str | None
    reasoning_parser: str | None
    language_model_only: bool


class Generation(Strict):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    seed: int
    enable_thinking: bool
    preserve_thinking: bool


class Agent(Strict):
    step_limit: int = Field(gt=0)
    cost_limit: float = Field(ge=0)
    max_consecutive_format_errors: int = Field(gt=0)
    system_template: str
    instance_template: str
    observation_template: str
    format_error_template: str


class Images(Strict):
    load_workers: int = Field(gt=0, le=4)
    reserve_bytes: int = Field(ge=0)
    expansion_factor: float = Field(ge=1)
    minimum_load_bytes: int = Field(ge=0)
    max_reserved_bytes: int = Field(gt=0)
    load_seconds: int = Field(gt=0)
    admission_seconds: int = Field(gt=0)
    cleanup: bool


class Execution(Strict):
    rollout_workers: int = Field(gt=0)
    harness_workers: int = Field(gt=0, le=4)
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


class Hidden(Strict):
    python: str
    dtype: Literal["float32", "float16", "bfloat16"]
    device: str
    chunk_size: int = Field(gt=0)
    layers: list[int]
    adapter: Literal["hf_causal"]


class Benchmark(Strict):
    partition: Literal["train", "val", "test", "excluded"] | None
    purposes: list[Literal["training", "baseline", "action_analysis"]]
    profile: str | None


class Steering(Strict):
    backend: Literal["easysteer"]
    enabled: bool
    payload: str | None


class Paths(Strict):
    nas_root: str


class Difficulty(Strict):
    estimator: None


class Action(Strict):
    taxonomy: None


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
    execution: Execution
    timeouts: Timeouts
    enrichment: Enrichment
    hidden: Hidden
    difficulty: Difficulty
    action: Action
    steering: Steering

    @model_validator(mode="after")
    def cross_fields(self):
        if self.steering.enabled != bool(self.steering.payload):
            raise ValueError("Enabled steering requires a payload; disabled steering must not have one")
        if self.timeouts.bootstrap_seconds >= self.timeouts.container_lifetime_seconds:
            raise ValueError("Bootstrap must fit container lifetime")
        if len(set(self.hidden.layers)) != len(self.hidden.layers) or any(x < 0 for x in self.hidden.layers):
            raise ValueError("Hidden layers must be unique nonnegative indices")
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
    root = path.parent
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
            cfg[section][field] = str((root / value).resolve())
    cfg.update(project_root=str(root), config_source=str(path), overrides=list(overrides))
    cfg["config_hash"] = hash_json(cfg)
    return cfg


def snapshot(cfg, run_id, selection):
    run = owned_path(cfg["project_root"], "runs/" + identifier(run_id))
    run.mkdir(parents=True, exist_ok=False)
    frozen = copy.deepcopy(cfg)
    frozen["selection"] = selection
    frozen["selection_sha256"] = hash_json(selection)
    write_json(run / "resolved-config.json", frozen)
    write_json(run / "selection.json", selection)
    return run


def load_snapshot(root, run_id):
    cfg = read_json(owned_path(root, f"runs/{identifier(run_id)}/resolved-config.json"))
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


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    print(json.dumps(resolve(a.config, a.set), indent=2))
