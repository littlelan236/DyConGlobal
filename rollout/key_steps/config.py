"""Strict, run-local inputs for source collection and pure-replay continuations."""

import copy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.io import hash_json, read_json


class KeyStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    mode: Literal["source", "scan", "continuation"]
    branch_count: Literal[8] = 8
    backend: Literal["initial_replay_v1"] = "initial_replay_v1"
    state_roots: list[str] = Field(default_factory=lambda: ["/testbed", "/tmp"])
    small_file_bytes: int = Field(default=1048576, gt=0, le=1048576)
    max_state_paths: int = Field(default=30000, gt=0)
    inventory: dict | None = None
    schedule_seed: int = 20261124
    service_interval_seconds: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    retry_backoff_seconds: float = Field(default=30.0, ge=0, allow_inf_nan=False)
    # A service applies to the entire task, including an unpredictable suffix.
    task_services: dict[str, list[str]] = Field(default_factory=dict)
    # Unknown external users share one conservative bucket by default.
    network_default: Literal["unknown-external", "local-only"] = "unknown-external"

    @model_validator(mode="after")
    def valid(self):
        if self.state_roots != ["/testbed", "/tmp"]:
            raise ValueError("v1 state coverage is explicitly /testbed and /tmp")
        if (self.mode in {"scan", "continuation"}) != (self.inventory is not None):
            raise ValueError("Only continuation mode requires an inventory")
        return self


def prepare(cfg, path, *, from_run=None):
    """Called by run create BEFORE snapshot/plan hashes are published."""
    from core.config import UniqueLoader

    raw = yaml.load(Path(path).read_text(encoding="utf8"), Loader=UniqueLoader)
    raw = dict(raw)
    inventory_path = raw.get("inventory")
    if isinstance(inventory_path, str):
        inventory_path = (Path(path).resolve().parent / inventory_path).resolve()
        raw["inventory"] = read_json(inventory_path)
    options = KeyStepConfig.model_validate(raw).model_dump()
    result = copy.deepcopy(cfg)
    if (
        result["benchmark"].get("backend", "swebench") != "swebench"
        or result["agent"]["backend"] != "minisweagent"
    ):
        raise ValueError("Key-step v1 requires SWE-bench / mini-swe-agent")
    if result["serving"]["management"] != "managed" or result["serving"]["tensor_parallel_size"] != 1:
        raise ValueError("Generation entropy v1 requires managed TP=1")
    if result["steering"]["enabled"]:
        raise ValueError("Key-step v1 currently requires unsteered generation")
    if type(result["generation"].get("seed")) is not int:
        raise ValueError("Key-step entropy requires an explicit sampler seed")
    result["capture"]["enabled"] = True
    result["capture"].setdefault("logprobs", {})["enabled"] = True
    if not result["capture"].get("export", {}).get("enabled"):
        raise ValueError("Key-step collection requires persisted native token publication")
    result["execution"].setdefault("limits_retry", {})["enabled"] = False
    if options["mode"] == "source":
        if not result["parallel"].get("enabled") or not result["parallel"].get("estimator_ref"):
            raise ValueError("Source collection requires an existing estimator configuration")
        result["parallel"].update(
            branch_count=8, mode="parallel_consensus", policy="every_step", fallback_policy="stop"
        )
        result["parallel"].pop("entropy", None)
        result["parallel"]["consensus"] = result["parallel"].get("consensus") or {
            "rule": "exact_action",
            "tie_policy": "estimator_score",
            "retention_ratio": 0.5,
            "min_retained_branches": 2,
        }
    else:
        inv = options["inventory"]
        expected_schema = "key-step-scan-v1" if options["mode"] == "scan" else "key-step-inventory-v1"
        if inv.get("schema") != expected_schema or inv.get("branch_count") != 8:
            raise ValueError("Expected a sealed B=8 inventory")
        digest = inv.get("sha256")
        if digest != hash_json({k: v for k, v in inv.items() if k != "sha256"}):
            raise ValueError("Inventory hash mismatch")
        if not from_run or inv["source_run"] != from_run:
            raise ValueError("Continuation run requires --from-run matching its inventory")
        if inv["source_config_hash"] != cfg.get("config_hash"):
            # Clone overrides rehash cfg. The immutable source is authoritative.
            from core.config import load_snapshot

            if inv["source_config_hash"] != load_snapshot(cfg["project_root"], from_run)["config_hash"]:
                raise ValueError("Inventory source configuration changed")
        result["parallel"].update(enabled=False, branch_count=1, mode="parallel")
        if options["mode"] == "scan":
            result["capture"]["enabled"] = False
            result["capture"]["logprobs"]["enabled"] = False
        result["parallel"].pop("entropy", None)
    # The original clone receipt still records its source policy. This new mode
    # has a distinct input surface, validated below and bound by config_hash.
    result.pop("parallel_policy", None)
    result["key_steps"] = options
    result["config_hash"] = hash_json(
        {k: v for k, v in result.items() if k not in {"config_hash", "selection", "selection_sha256"}}
    )
    from core.parallel_policy import validate_effective_parallel

    validate_effective_parallel(result)
    return result


def work_items(cfg, selection, run_id, repeat_count, *, extension=None):
    from rollout.sampling import make_work_items

    settings = cfg.get("key_steps") or {}
    if settings.get("mode") not in {"scan", "continuation"}:
        return make_work_items(
            selection,
            run_id,
            repeat_count,
            cfg["generation"]["seed"],
            cfg.get("parallel", {}).get("branch_count", 1),
            extension=extension,
        )
    if extension:
        raise ValueError("Continuation inventory fixes all logical members; repetitions cannot be appended")
    inv = settings["inventory"]
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "repeat_count": repeat_count,
        "selection_sha256": hash_json(selection),
        "seed_policy": {"name": "continuation_identity_v1", "base_seed": 0, "branch_stride": 1},
        "items": [],
    }
    for index, child in enumerate(inv["children"]):
        task = selection["tasks"].get(child["task_key"])
        if task is None or task["instance_id"] != child["instance_id"]:
            raise ValueError("Continuation task is outside frozen selection")
        result["items"].append(
            {
                "run_id": run_id,
                "task_key": child["task_key"],
                "instance_id": child["instance_id"],
                "repetition_id": f"rep-{index:06d}",
                "repetition_index": index,
                "sample_key": "s-" + hash_json([run_id, child["child_id"]])[:32],
                "sample_seed": child["seed"],
                "seed_policy": "continuation_identity_v1",
                "continuation": child,
            }
        )
    if not result["items"]:
        raise ValueError("Inventory has no selected children")
    return result
