"""The sole writable task/split/rollout catalog. All updates are locked transactions."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from core.config import Strict
from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json
from core.locking import file_lock


class Task(Strict):
    benchmark: str
    revision: str
    instance_id: str
    repo: str
    base_commit: str
    partition: Literal["train", "val", "test", "excluded"]
    exclusion_reason: str | None = None
    instance: dict

    @model_validator(mode="after")
    def consistent_instance(self):
        for key in ["instance_id", "repo", "base_commit"]:
            if self.instance.get(key) != getattr(self, key):
                raise ValueError(f"Task {key} disagrees with benchmark instance")
        identifier(self.instance_id)
        return self


class Rollout(Strict):
    task_key: str
    run_id: str
    attempt_id: str
    purposes: list[Literal["training", "baseline", "action_analysis"]]
    trajectory_path: str
    trajectory_sha256: str
    patch_path: str
    patch_sha256: str
    submission_status: Literal["submitted", "unsubmitted"]
    termination_reason: str
    evaluation_status: Literal["not_run", "running", "completed", "infra_error"]
    resolved: bool | None
    selected_evaluation_id: str | None
    model_fingerprint: str
    provenance: dict

    @model_validator(mode="after")
    def valid_outcome(self):
        if (self.evaluation_status == "completed") != (self.resolved is not None):
            raise ValueError("Only completed evaluation has a boolean resolved outcome")
        if self.evaluation_status == "completed" and not self.selected_evaluation_id:
            raise ValueError("Completed outcome needs evaluation evidence")
        if self.resolved and self.submission_status != "submitted":
            raise ValueError("Unsubmitted trajectory cannot be a resolved rollout")
        return self


class Manifest(Strict):
    schema_version: Literal[1]
    catalog_revision: int = Field(ge=0)
    tasks: dict[str, Task]
    rollouts: dict[str, Rollout]
    evaluations: dict[str, dict]
    baseline_selections: dict[str, dict[str, str]]

    @model_validator(mode="after")
    def references(self):
        groups = {}
        for task in self.tasks.values():
            identity = (task.repo, task.base_commit, task.instance_id)
            if identity in groups and groups[identity] != task.partition:
                raise ValueError("Same task cannot occur in multiple partitions")
            groups[identity] = task.partition
        for rid, r in self.rollouts.items():
            identifier(rid)
            if r.task_key not in self.tasks:
                raise ValueError(f"Unknown task for {rid}")
            if r.selected_evaluation_id:
                e = self.evaluations.get(r.selected_evaluation_id)
                if not e or rid not in e["outcomes"] or e["outcomes"][rid]["patch_sha256"] != r.patch_sha256:
                    raise ValueError("Evaluation must bind the exact rollout patch")
                outcome = e["outcomes"][rid]
                if outcome["status"] != r.evaluation_status or outcome["resolved"] != r.resolved:
                    raise ValueError("Rollout outcome disagrees with selected evaluation")
            for p in [r.trajectory_path, r.patch_path]:
                if Path(p).is_absolute() or ".." in Path(p).parts:
                    raise ValueError("Catalog artifacts must be project-relative")
        for profile, tasks in self.baseline_selections.items():
            for task, rid in tasks.items():
                if rid not in self.rollouts or self.rollouts[rid].task_key != task:
                    raise ValueError(f"Invalid baseline selection: {profile}/{task}")
        return self


def empty_manifest():
    return {
        "schema_version": 1,
        "catalog_revision": 0,
        "tasks": {},
        "rollouts": {},
        "evaluations": {},
        "baseline_selections": {},
    }


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "baseline/manifest.json"

    def read(self):
        return Manifest.model_validate(read_json(self.path)).model_dump()

    @contextmanager
    def transaction(self):
        with file_lock(self.root / "state/catalog.lock"):
            original = self.read()
            data = copy.deepcopy(original)
            yield data
            validated = Manifest.model_validate(data).model_dump()
            if validated != original:
                validated["catalog_revision"] = original["catalog_revision"] + 1
                write_json(self.path, validated)

    def select(self, *, partition=None, purposes=(), profile=None):
        data = self.read()
        if profile is not None and profile not in data["baseline_selections"]:
            raise ValueError("Unknown baseline selection profile")
        ids = [
            rid
            for rid, r in data["rollouts"].items()
            if (
                data["tasks"][r["task_key"]]["partition"] != "excluded"
                if partition is None
                else data["tasks"][r["task_key"]]["partition"] == partition
            )
            and (not purposes or set(purposes).intersection(r["purposes"]))
            and (profile is None or data["baseline_selections"][profile].get(r["task_key"]) == rid)
        ]
        return {
            "catalog_revision": data["catalog_revision"],
            "catalog_sha256": hash_json(data),
            "rollout_ids": sorted(ids),
            "tasks": copy.deepcopy(data["tasks"]),
            "rollouts": {rid: copy.deepcopy(data["rollouts"][rid]) for rid in ids},
        }

    def select_tasks(self, *, partition=None):
        data = self.read()
        tasks = {
            key: copy.deepcopy(task)
            for key, task in data["tasks"].items()
            if (task["partition"] != "excluded" if partition is None else task["partition"] == partition)
        }
        return {
            "catalog_revision": data["catalog_revision"],
            "catalog_sha256": hash_json(data),
            "task_keys": sorted(tasks),
            "tasks": tasks,
        }

    def verify(self):
        data = self.read()
        for rid, row in data["rollouts"].items():
            for field in ["trajectory", "patch"]:
                p = owned_path(self.root, row[field + "_path"])
                if sha256(p) != row[field + "_sha256"]:
                    raise ValueError(f"Changed {field}: {rid}")
        for e in data["evaluations"].values():
            if sha256(owned_path(self.root, e["report_path"])) != e["report_sha256"]:
                raise ValueError("Changed evaluation report")
        return {
            "revision": data["catalog_revision"],
            "tasks": len(data["tasks"]),
            "rollouts": len(data["rollouts"]),
            "verified": True,
        }

    def record_evaluation(self, eval_id, report_path, outcomes, provenance):
        identifier(eval_id)
        report = owned_path(self.root, report_path)
        with self.transaction() as data:
            if eval_id in data["evaluations"]:
                raise FileExistsError(eval_id)
            for rid, result in outcomes.items():
                row = data["rollouts"][rid]
                if result["patch_sha256"] != row["patch_sha256"]:
                    raise ValueError("Patch mismatch")
                row.update(
                    evaluation_status=result["status"],
                    resolved=result["resolved"],
                    selected_evaluation_id=eval_id,
                )
            data["evaluations"][eval_id] = {
                "report_path": str(report.relative_to(self.root)),
                "report_sha256": sha256(report),
                "outcomes": outcomes,
                "provenance": provenance,
            }

    def register_rollout(self, rollout_id, row):
        identifier(rollout_id)
        with self.transaction() as data:
            if row["task_key"] not in data["tasks"]:
                raise ValueError("Rollout task is not registered in the manifest")
            if rollout_id in data["rollouts"]:
                raise FileExistsError(rollout_id)
            data["rollouts"][rollout_id] = row

    def register_tasks(self, rows):
        keys = []
        with self.transaction() as data:
            for raw in rows:
                task = Task.model_validate(raw).model_dump()
                key = hash_json({k: task[k] for k in ["benchmark", "revision", "instance_id"]})
                if key in data["tasks"] and data["tasks"][key] != task:
                    raise ValueError("Task is already registered with conflicting metadata")
                data["tasks"][key] = task
                keys.append(key)
        return keys

    def select_baseline(self, profile, task_key, rollout_id):
        identifier(profile)
        with self.transaction() as data:
            chosen = data["baseline_selections"].setdefault(profile, {})
            if task_key in chosen and chosen[task_key] != rollout_id:
                raise ValueError("Baseline already selected; create a new profile to change it")
            chosen[task_key] = rollout_id
