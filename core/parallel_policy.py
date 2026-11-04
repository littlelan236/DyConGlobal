"""Strict per-run parallel policy inputs, never accepted by the root YAML schema."""

import copy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from core.config import Config, Parallel, ParallelConsensus, Strict, UniqueLoader
from core.io import hash_json

RepresentativePolicy = Literal["estimator_score", "min_step_tokens"]
RetentionRule = Literal["effective_count_v1", "shannon_exp_keep_v1", "baseline_floor_v1",
                        "effective_count_full_above_v1", "effective_count_plus_n_v1"]


def effective_policy(parallel):
    """Selection identity excludes independently revisable admission/resource limits."""
    fields = ("enabled", "mode", "fallback_policy", "branch_count", "consensus", "entropy",
              "estimator_ref", "estimator_sha256")
    return copy.deepcopy({key: parallel[key] for key in fields if key in parallel})


class RunConsensus(ParallelConsensus):
    representative_policy: RepresentativePolicy = "estimator_score"


class RunEntropy(Strict):
    retention_rule: RetentionRule
    full_retention_threshold: int | None = Field(default=None, ge=1, strict=True)
    n: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def threshold_rule(self):
        if self.retention_rule != "effective_count_full_above_v1" and self.full_retention_threshold is not None:
            raise ValueError("full_retention_threshold requires effective_count_full_above_v1")
        if self.retention_rule == "effective_count_plus_n_v1":
            if self.n is None:
                raise ValueError("effective_count_plus_n_v1 requires an explicit n")
        elif self.n is not None:
            raise ValueError("n requires effective_count_plus_n_v1")
        return self


class RunParallel(Parallel):
    mode: Literal["parallel", "parallel_baseline", "parallel_consensus", "parallel_enthophy"] = "parallel"
    consensus: RunConsensus | None = None
    entropy: RunEntropy | None = None

    @model_validator(mode="after")
    def run_policy(self):
        if self.mode in {"parallel_consensus", "parallel_enthophy"} and not self.enabled:
            raise ValueError("Run consensus policy requires enabled=true")
        if self.mode == "parallel_enthophy":
            if not self.enabled or self.branch_count < 2 or not self.estimator_ref:
                raise ValueError("parallel_enthophy requires enabled, multiple branches and estimator_ref")
            if self.fallback_policy != "stop":
                raise ValueError("parallel_enthophy requires fallback_policy=stop")
            if self.consensus is None:
                self.consensus = RunConsensus()
            if self.consensus.min_retained_branches > self.branch_count:
                raise ValueError("parallel consensus min_retained_branches is invalid")
            if self.entropy is None:
                raise ValueError("parallel_enthophy requires an explicit entropy.retention_rule")
        elif self.entropy is not None:
            raise ValueError("entropy.retention_rule is only valid for parallel_enthophy")
        return self


class RunConfig(Config):
    parallel: RunParallel = Field(default_factory=RunParallel)


class ConsensusInput(Strict):
    representative_policy: RepresentativePolicy | None = None
    retention_ratio: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    min_retained_branches: int | None = Field(default=None, ge=2)


class RunParallelPolicy(Strict):
    schema_version: Literal[1]
    enabled: bool | None = None
    mode: Literal["parallel_consensus", "parallel_enthophy"] | None = None
    fallback_policy: Literal["first_valid", "stop"] | None = None
    consensus: ConsensusInput | None = None
    entropy: RunEntropy | None = None


def validate_effective_parallel(cfg):
    """Validate runtime settings without inserting defaults into frozen dictionaries."""
    raw = copy.deepcopy({key: cfg[key] for key in Config.model_fields if key in cfg})
    raw.get("parallel", {}).pop("estimator_sha256", None)
    # Unused benchmark image-builder metadata may use an older frozen schema.
    # Validate only the active backend; never rewrite its original snapshot bytes.
    if raw.get("benchmark", {}).get("backend", "swebench") != "skillsbench":
        raw.pop("skillsbench_images", None)
    RunConfig.model_validate(raw, context={"frozen_snapshot": True})


def apply_run_parallel(cfg, path=None, *, from_run=None, overrides=()):
    """Return a copied config and immutable input bytes, or leave legacy runs untouched."""
    from core.config import with_derived_parallel_capacity

    cfg = with_derived_parallel_capacity(cfg)
    if path is None and "parallel_policy" not in cfg:
        return cfg, None
    result = copy.deepcopy(cfg)
    raw_bytes = None
    if path is not None:
        path = Path(path).resolve()
        raw_bytes = path.read_bytes()
        raw = yaml.load(raw_bytes.decode("utf-8"), Loader=UniqueLoader)
        policy = RunParallelPolicy.model_validate(raw)
        updates = policy.model_dump(exclude_unset=True)
        if any(value is None for value in updates.values()):
            raise ValueError("Run parallel policy fields cannot be null")
        for section in ("consensus", "entropy"):
            if any(value is None for value in updates.get(section, {}).values()):
                raise ValueError("Run parallel policy fields cannot be null")
        overlap = set()
        for key, value in updates.items():
            if key == "schema_version":
                continue
            overlap.update({"parallel." + key + "." + child for child in value} if isinstance(value, dict)
                           else {"parallel." + key})
        if any(any(item.split("=", 1)[0] == key or item.split("=", 1)[0].startswith(key + ".") or
                   key.startswith(item.split("=", 1)[0] + ".")
                   for key in overlap) for item in overrides):
            raise ValueError("Conflicting --set and --parallel-config sources")
        parallel = result.setdefault("parallel", {})
        for key, value in updates.items():
            if key == "schema_version":
                continue
            if isinstance(value, dict):
                target = parallel.get(key) or {}
                if (key == "entropy" and "retention_rule" in value and
                        value["retention_rule"] != "effective_count_full_above_v1" and
                        "full_retention_threshold" not in value):
                    target = {k: v for k, v in target.items() if k != "full_retention_threshold"}
                if (key == "entropy" and "retention_rule" in value and
                        value["retention_rule"] != "effective_count_plus_n_v1" and "n" not in value):
                    target = {k: v for k, v in target.items() if k != "n"}
                parallel[key] = {**target, **value}
            else:
                parallel[key] = value
        if parallel.get("mode") != "parallel_enthophy":
            if "entropy" in updates:
                raise ValueError("entropy.retention_rule is only valid for parallel_enthophy")
            parallel.pop("entropy", None)
        import hashlib

        result["parallel_policy"] = {
            "schema_version": 1, "input_path": "parallel-config.yaml",
            "input_sha256": hashlib.sha256(raw_bytes).hexdigest(), "source": str(path),
        }
    elif from_run is not None:
        from core.io import run_path

        raw_bytes = (run_path(result["project_root"], from_run) / "parallel-config.yaml").read_bytes()
        result["parallel_policy"]["inherited_from_run"] = from_run
    parallel = result.get("parallel", {})
    if (parallel.get("entropy") or {}).get("retention_rule") == "effective_count_full_above_v1":
        parallel["entropy"].setdefault("full_retention_threshold", 10)
    if parallel.get("mode") not in {"parallel_consensus", "parallel_enthophy"}:
        raise ValueError("Run parallel policies require a consensus rollout mode")
    parallel["consensus"] = {**ParallelConsensus().model_dump(), **(parallel.get("consensus") or {})}
    parallel["consensus"].setdefault("representative_policy", "estimator_score")
    validate_effective_parallel(result)
    result["parallel_policy"]["effective"] = effective_policy(parallel)
    result["parallel_policy"]["effective_sha256"] = hash_json(effective_policy(parallel))
    result["config_hash"] = hash_json({k: v for k, v in result.items()
                                     if k not in {"config_hash", "selection", "selection_sha256"}})
    return result, raw_bytes


def validate_policy_snapshot(cfg, run):
    """Check the run-local input binding; do not read its mutable source file."""
    proof = cfg.get("parallel_policy")
    if proof is None:
        return
    from core.io import sha256

    if proof.get("input_path") != "parallel-config.yaml":
        raise ValueError("Invalid run parallel policy input path")
    if sha256(Path(run) / "parallel-config.yaml") != proof["input_sha256"]:
        raise ValueError("Frozen run parallel policy input was modified")
    effective = effective_policy(cfg["parallel"])
    if proof.get("effective") != effective or proof.get("effective_sha256") != hash_json(effective):
        raise ValueError("Frozen run parallel policy differs from effective settings")
    validate_effective_parallel(cfg)
