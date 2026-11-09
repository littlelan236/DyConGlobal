"""Import immutable official task packages, without manufacturing SWE fields."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from catalog.store import Catalog
from core.io import hash_json, identifier, read_json, sha256, write_json


def package_manifest(path):
    path = Path(path).resolve()
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_symlink():
            raise ValueError("Task package symlinks require explicit capability review: " + str(p))
        if p.is_file():
            rows.append({"path": p.relative_to(path).as_posix(), "sha256": sha256(p),
                         "mode": p.stat().st_mode & 0o777})
    return rows


def validate_payload(payload, instance_id, revision):
    required = {"instance_id", "task_path", "package_sha256", "files", "metadata", "images"}
    if set(payload) - {"verifier_index_adaptation"} != required or payload["instance_id"] != instance_id:
        raise ValueError("Invalid SkillsBench task payload")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("SkillsBench requires a full immutable source revision")
    if hash_json(payload["files"]) != payload["package_sha256"]:
        raise ValueError("Task package manifest hash changed")
    for row in payload["files"]:
        p = Path(row["path"])
        if p.is_absolute() or ".." in p.parts or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Unsafe task package manifest")


def inspect_capabilities(task):
    payload = task["instance"]
    metadata = payload["metadata"]
    sandbox = metadata.get("sandbox", metadata.get("environment", {}))
    user = metadata.get("user")
    reasons = []
    if len(payload["images"]) != 1 or payload["images"][0]["service"] != "main":
        reasons.append("multi-service tasks are not qualified")
    if sandbox.get("gpus", 0):
        reasons.append("task GPU requirements are not qualified")
    if user:
        reasons.append("interactive document users require separate qualification")
    interfaces = metadata.get("metadata", {}).get("interface", [])
    if any(x in {"browser", "gui", "desktop"} for x in interfaces):
        reasons.append("GUI/browser tasks are not qualified")
    return {"supported": not reasons, "reasons": reasons, "services": len(payload["images"]),
            "sandbox": sandbox, "re_evaluation": False, "action_analysis": False}


def verify_package(task, *, deep=True):
    payload = task["instance"]
    validate_payload(payload, task["instance_id"], task["revision"])
    path = Path(payload["task_path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("Invalid SkillsBench task package path")
    if deep:
        if package_manifest(path) != payload["files"]:
            raise ValueError("Pinned SkillsBench task package changed")
    else:
        for row in payload["files"]:
            member = path / row["path"]
            if (not member.resolve().is_relative_to(path.resolve()) or member.is_symlink()
                    or not member.is_file() or member.stat().st_mode & 0o777 != row["mode"]):
                raise ValueError("Pinned SkillsBench package path changed: " + row["path"])
    if payload.get('verifier_index_adaptation'):
        from catalog.skillsbench_index import verify_adaptation
        verify_adaptation(task)
    return inspect_capabilities(task)


def import_tasks(cfg, snapshot_id, task_ids, partition="train", *, oracle_only=False):
    from runtime.images.skillsbench import inventory
    from catalog.store import Task
    snapshot_id = identifier(snapshot_id)
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Select a nonempty explicit list of distinct task IDs")
    plan = inventory(cfg)
    by_id = {t["task_id"]: t for t in plan["tasks"]}
    unknown = set(task_ids) - set(by_id)
    if unknown:
        raise ValueError("Unknown official task IDs: " + str(sorted(unknown)))
    rows = []
    from runtime.agents.skillsbench import attach_image_references
    for iid in task_ids:
        path = Path(plan["source_root"]) / "tasks" / iid
        text = (path / "task.md").read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError("Missing native task.md frontmatter")
        metadata = yaml.safe_load(text.split("---", 2)[1])
        files = package_manifest(path)
        units = [u for u in plan["units"] if u["task_id"] == iid]
        task = Task.model_validate({"benchmark": "skillsbench", "revision": plan["revision"],
                                   "instance_id": iid, "partition": partition,
                                   "instance": {"instance_id": iid, "task_path": str(path),
                                                "package_sha256": hash_json(files), "files": files,
                                                "metadata": metadata, "images": units}}).model_dump()
        task = attach_image_references(cfg, task)
        capability = inspect_capabilities(task)
        if not capability["supported"] and not oracle_only:
            raise ValueError(iid + ": " + "; ".join(capability["reasons"]))
        rows.append(task)
    target = Path(cfg["project_root"]) / "catalog/snapshots" / snapshot_id / "manifest.json"
    tasks = {hash_json({k: t[k] for k in ("benchmark", "revision", "instance_id")}): t for t in rows}
    result = {"schema_version": 1, "benchmark": "skillsbench", "revision": plan["revision"],
              "tasks": tasks, "capabilities": {k: inspect_capabilities(t) for k, t in tasks.items()}}
    if oracle_only:
        result["validation_only"] = True
    if target.exists() and read_json(target) != result:
        raise FileExistsError("Snapshot ID already binds different task content")
    Catalog(cfg["project_root"]).register_tasks(rows)
    if not target.exists():
        write_json(target, result)
    return {"snapshot_ref": str(target.relative_to(cfg["project_root"])),
            "snapshot_sha256": sha256(target), "task_count": len(tasks)}
