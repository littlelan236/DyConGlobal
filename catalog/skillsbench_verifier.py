"""Receipt-bound correction of one exact SkillsBench verifier revision."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from core.io import hash_json, identifier, read_json, sha256, write_json

POLICY = "pddl-airport-plan-v1"
TASK_ID = "pddl-airport-planning"
MEMBER = "verifier/test_outputs.py"
UPSTREAM_SHA256 = "1f8c6cd4be1ed7fdf0925494643dd2d78848c9ec4c855a7a7837612675ac87e4"
PATCH = Path(__file__).parent / "verifier_patches/pddl_airport_test_outputs.py"


def verify_adaptation(task):
    if task['instance']['verifier_adaptation'].get('policy') == 'skillsbench-verifier-repair-v1':
        from catalog.verifier_repairs import verify_adaptation as verify_repair
        return verify_repair(task)
    from catalog.skillsbench import verify_package

    payload = task["instance"]
    adaptation = payload["verifier_adaptation"]
    path = Path(adaptation["receipt_path"])
    if sha256(path) != adaptation["receipt_sha256"]:
        raise ValueError("Verifier correction receipt changed")
    receipt = read_json(path)
    original = receipt["upstream_task"]
    if original["instance"].get("verifier_adaptation"):
        raise ValueError("Nested verifier corrections are not supported")
    verify_package(original)
    if (adaptation["policy"] != POLICY or receipt["policy"] != POLICY
            or task["instance_id"] != TASK_ID or original["instance_id"] != TASK_ID
            or any(task[key] != original[key] for key in task if key != "instance")
            or receipt["source_verifier_sha256"] != UPSTREAM_SHA256
            or sha256(Path(original["instance"]["task_path"]) / MEMBER) != UPSTREAM_SHA256
            or receipt["effective_verifier_sha256"] != sha256(PATCH)
            or sha256(Path(payload["task_path"]) / MEMBER) != sha256(PATCH)
            or receipt["effective_task_path"] != payload["task_path"]):
        raise ValueError("Verifier correction identity mismatch")
    expected = copy.deepcopy(original["instance"])
    next(row for row in expected["files"] if row["path"] == MEMBER)["sha256"] = sha256(PATCH)
    expected.update(task_path=payload["task_path"], package_sha256=hash_json(expected["files"]),
                    verifier_adaptation=adaptation)
    if payload != expected:
        raise ValueError("Verifier correction modified unapproved task content")


def publish_variant(cfg, snapshot_id, source_ref):
    from catalog.skillsbench import package_manifest, verify_package

    root = Path(cfg["project_root"]).resolve()
    source_path = (root / source_ref).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError("Source snapshot must be project-owned")
    source_hash = sha256(source_path)
    source = read_json(source_path)
    if source.get("benchmark") != "skillsbench":
        raise ValueError("Expected a SkillsBench snapshot")
    matching = [(key, task) for key, task in source["tasks"].items()
                if task["instance_id"] == TASK_ID]
    if len(matching) != 1:
        raise ValueError("Source must contain exactly one airport task")
    for task in source["tasks"].values():
        verify_package(task)
    key, original = matching[0]
    if (original["instance"].get("verifier_adaptation")
            or sha256(Path(original["instance"]["task_path"]) / MEMBER) != UPSTREAM_SHA256):
        raise ValueError("Verifier correction source hash mismatch or already corrected")
    snapshot_id = identifier(snapshot_id)
    publication = root / "catalog/task-variants" / snapshot_id
    target = root / "catalog/snapshots" / snapshot_id / "manifest.json"
    if publication.exists() or target.parent.exists():
        raise FileExistsError("Task variants are immutable; choose a new ID")
    # Exclusive mkdir claims the ID; failed publications remain inspectable.
    publication.mkdir(parents=True, exist_ok=False)
    effective = publication / "tasks" / TASK_ID
    shutil.copytree(original["instance"]["task_path"], effective)
    (effective / MEMBER).write_bytes(PATCH.read_bytes())
    files = package_manifest(effective)
    result = copy.deepcopy(source)
    task = result["tasks"][key]
    receipt_path = publication / "receipts" / (TASK_ID + ".json")
    write_json(receipt_path, {
        "schema_version": 1, "policy": POLICY, "upstream_task": original,
        "source_snapshot": {"path": str(source_path), "sha256": source_hash},
        "source_verifier_sha256": UPSTREAM_SHA256,
        "effective_verifier_sha256": sha256(PATCH), "effective_task_path": str(effective),
        "reason": "Accept documented function syntax; require explicit VALID result; fix invalid diagnostic",
    })
    task["instance"].update(task_path=str(effective), files=files, package_sha256=hash_json(files),
        verifier_adaptation={"policy": POLICY, "receipt_path": str(receipt_path),
                             "receipt_sha256": sha256(receipt_path)})
    verify_package(task)
    if sha256(source_path) != source_hash:
        raise ValueError("Source snapshot changed during publication")
    result["verifier_correction"] = {"policy": POLICY, "source_snapshot": str(source_path),
                                       "source_snapshot_sha256": source_hash}
    write_json(target, result)
    return {"snapshot_ref": str(target.relative_to(root)), "snapshot_sha256": sha256(target),
            "policy": POLICY, "adapted_tasks": 1, "tasks": len(result["tasks"]),
            "receipt_path": str(receipt_path)}
