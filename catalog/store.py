"""Benchmark task catalog.

The catalog is deliberately limited to benchmark definitions. Rollouts and
evaluations are run-scoped evidence and live under ``runs/<run>/evidence.json``.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from core.config import Strict
from core.io import hash_json, identifier, read_json, write_json
from core.locking import file_lock


class Task(Strict):
    benchmark: str
    revision: str
    instance_id: str
    repo: str | None = None
    base_commit: str | None = None
    partition: Literal["train", "val", "test", "excluded"]
    exclusion_reason: str | None = None
    instance: dict

    @model_validator(mode="after")
    def consistent_instance(self):
        if self.benchmark == "skillsbench":
            if self.repo is not None or self.base_commit is not None:
                raise ValueError("SkillsBench tasks must not contain SWE repository placeholders")
            from catalog.skillsbench import validate_payload
            validate_payload(self.instance, self.instance_id, self.revision)
            identifier(self.instance_id)
            return self
        if not self.repo or not self.base_commit:
            raise ValueError("SWE tasks require repo and base_commit")
        for key in ("instance_id", "repo", "base_commit"):
            if self.instance.get(key) != getattr(self, key):
                raise ValueError(f"Task {key} disagrees with benchmark instance")
        identifier(self.instance_id)
        return self

    @model_serializer(mode="wrap")
    def preserve_swe_shape(self, handler):
        value = handler(self)
        if self.benchmark == "skillsbench":
            value.pop("repo", None)
            value.pop("base_commit", None)
        return value


class Manifest(Strict):
    schema_version: Literal[2]
    catalog_revision: int = Field(ge=0)
    tasks: dict[str, Task]

    @model_validator(mode="after")
    def unique_partitions(self):
        groups = {}
        for task in self.tasks.values():
            identity = (task.benchmark, task.revision, task.repo, task.base_commit, task.instance_id)
            if identity in groups and groups[identity] != task.partition:
                raise ValueError("Same task cannot occur in multiple partitions")
            groups[identity] = task.partition
        return self


def empty_manifest():
    return {"schema_version": 2, "catalog_revision": 0, "tasks": {}}


class Catalog:
    """The global catalog contains benchmark tasks only."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "catalog/manifest.json"

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

    def select_tasks(self, *, partition=None, task_ids=(), benchmark=None, revision=None):
        data = self.read()
        if benchmark is not None:
            data["tasks"] = {k: t for k, t in data["tasks"].items()
                             if (t["benchmark"] == "skillsbench") == (benchmark == "skillsbench")
                             and (revision is None or t["revision"] == revision)}
        requested = set(task_ids)
        matches = [t["instance_id"] for t in data["tasks"].values() if t["instance_id"] in requested]
        if len(matches) != len(set(matches)):
            raise ValueError("Ambiguous task ID across benchmark revisions; select a frozen snapshot")
        unknown = requested - {t["instance_id"] for t in data["tasks"].values()}
        if unknown:
            raise ValueError(f"Unknown requested task IDs: {sorted(unknown)}")
        tasks = {
            key: copy.deepcopy(task)
            for key, task in data["tasks"].items()
            if (task["partition"] != "excluded" if partition is None else task["partition"] == partition)
            and (not requested or task["instance_id"] in requested)
        }
        return {
            "catalog_revision": data["catalog_revision"],
            "catalog_sha256": hash_json(data),
            "task_keys": sorted(tasks),
            "tasks": tasks,
        }

    def register_tasks(self, rows):
        keys = []
        with self.transaction() as data:
            for raw in rows:
                task = Task.model_validate(raw).model_dump()
                key = hash_json({k: task[k] for k in ("benchmark", "revision", "instance_id")})
                if key in data["tasks"] and data["tasks"][key] != task:
                    raise ValueError("Task is already registered with conflicting metadata")
                data["tasks"][key] = task
                keys.append(key)
        return keys

    def verify(self):
        data = self.read()
        return {"revision": data["catalog_revision"], "tasks": len(data["tasks"]), "verified": True}
