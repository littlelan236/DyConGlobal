"""One-time, receipt-backed migration of legacy run directories.

The production run reader intentionally understands only schema 2. This module
is the isolated source-format reader used before the cutover marker is written.
It never starts workload processes or makes model requests.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock
from rollout.sampling import make_work_items, write_work_items


def _tree_hash(path: Path) -> str:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".migration-" not in item.name:
            rows.append({"path": str(item.relative_to(path)), "sha256": sha256(item)})
    return hash_json(rows)


def _live(run: Path) -> bool:
    status = read_json(run / "status.json") if (run / "status.json").exists() else {}
    if status.get("stage") in {"complete", "failed", "validated", "planned", "cancelled"}:
        return False
    latest = run / "execution/latest.json"
    if not latest.exists():
        return False
    try:
        attempt = read_json(run / "execution/attempts" / read_json(latest)["attempt"] / "supervisor.json")
        os.kill(int(attempt["pid"]), 0)
        return True
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False


def _entry(run: Path, root: Path) -> dict:
    entry = {"source_run_id": run.name, "source_path": str(run.relative_to(root)), "outcome": None}
    entry["source_tree_sha256"] = _tree_hash(run)
    plan_path = run / "plan.json"
    if not plan_path.exists():
        entry.update(outcome="migrated_artifact_only", reason="no_plan")
        return entry
    plan = read_json(plan_path)
    entry.update(operation=plan.get("operation"), source_schema=plan.get("schema_version"))
    if plan.get("schema_version") == 2 and plan.get("migration_state") == "active":
        entry.update(outcome="already_active", active_path=entry["source_path"])
    elif _live(run):
        entry.update(outcome="blocked_live_state", reason="live_supervisor")
    else:
        entry.update(outcome="planned_migration", target_schema=2)
    return entry


def inventory(root, manifest_path):
    root = Path(root).resolve()
    runs = root / "runs"
    entries = [_entry(run, root) for run in sorted(runs.iterdir()) if run.is_dir() and not run.name.startswith(".migration-")]
    manifest = {
        "schema_version": 1,
        "kind": "runs_to_repetition",
        "created_at": now(),
        "host": socket.gethostname(),
        "project_root": str(root),
        "entries": entries,
    }
    manifest["inventory_sha256"] = hash_json(manifest)
    write_json(manifest_path, manifest)
    return manifest


def _migrate_copy(source: Path, staged: Path, *, root: Path, archive_rel: str):
    shutil.copytree(source, staged)
    plan = read_json(staged / "plan.json")
    source_schema = plan.get("schema_version", 1)
    cfg = read_json(staged / "resolved-config.json")
    selection = read_json(staged / "selection.json")
    tasks = read_json(staged / "tasks.json")
    operation = plan["operation"]
    plan.update(schema_version=2, migration_state="active", repeat_count=1)
    tasks["schema_version"] = 2
    write_json(staged / "tasks.json", tasks)
    for name in ("progress.json", "artifacts.json", "status.json", "code-identity.json"):
        path = staged / name
        if path.exists():
            value = read_json(path)
            value["schema_version"] = 2
            if name == "status.json":
                value["migration_state"] = "active"
            write_json(path, value)
    if operation == "rollout":
        branch_count = cfg.get("parallel", {}).get("branch_count", 1)
        work = make_work_items(selection, source.name, 1, cfg["generation"]["seed"], branch_count)
        write_work_items(staged, work)
        evidence_path = staged / "evidence.json"
        if evidence_path.exists():
            evidence = read_json(evidence_path)
            items_by_task = {item["task_key"]: item for item in work["items"]}
            for row in evidence.get("rollouts", {}).values():
                item = items_by_task.get(row.get("task_key"), work["items"][0])
                row.setdefault("sample_key", item["sample_key"])
                row.setdefault("repetition_id", "rep-000")
                row.setdefault("repetition_index", 0)
                row.setdefault("seed_provenance", "legacy_unrecorded")
            write_json(evidence_path, evidence)
    else:
        work = make_work_items(selection, source.name, 1, cfg["generation"]["seed"], 1)
        write_work_items(staged, work)
        write_json(staged / "sample-scope.json", {
            "schema_version": 1,
            "source_run_id": source.name,
            "sample_keys": [item["sample_key"] for item in work["items"]],
            "source_evidence_sha256": sha256(staged / "evidence.json") if (staged / "evidence.json").exists() else None,
        })
    migration = {
        "schema_version": 1,
        "source_run_id": source.name,
        "source_path": str(source.relative_to(root)),
        "archive_path": archive_rel,
        "source_tree_sha256": _tree_hash(source),
        "source_schema": source_schema,
        "target_schema": 2,
        "migrated_at": now(),
    }
    write_json(staged / "migration/source.json", migration)
    plan["plan_sha256"] = hash_json({k: v for k, v in plan.items() if k != "plan_sha256"})
    write_json(staged / "plan.json", plan)
    return migration


def apply(root, manifest_path, *, archive_root=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("inventory_sha256") != hash_json({k: v for k, v in manifest.items() if k != "inventory_sha256"}):
        raise ValueError("Migration inventory was modified")
    archive_root = (root / (archive_root or "archive/runs-schema1")).resolve()
    if not archive_root.is_relative_to(root):
        raise ValueError("Archive root escapes project")
    with file_lock(root / "state/migration.lock", nonblocking=True):
        for entry in manifest["entries"]:
            if entry["outcome"] in {"already_active", "migrated_active", "migrated_artifact_only"}:
                continue
            if entry["outcome"] == "blocked_live_state":
                raise RuntimeError(f"Cannot migrate live run: {entry['source_run_id']}")
            source = root / entry["source_path"]
            archive = archive_root / entry["source_run_id"]
            if archive.exists() or not source.exists():
                raise FileExistsError(f"Migration source/archive conflict: {entry['source_run_id']}")
            archive.parent.mkdir(parents=True, exist_ok=True)
            if entry.get("outcome") == "migrated_artifact_only":
                source.rename(archive)
                entry.update(outcome="migrated_artifact_only", archive_path=str(archive.relative_to(root)))
                continue
            stage = root / "runs" / (".migration-" + entry["source_run_id"])
            if stage.exists():
                shutil.rmtree(stage)
            migration = _migrate_copy(source, stage, root=root, archive_rel=str(archive.relative_to(root)))
            source.rename(archive)
            stage.rename(source)
            entry.update(outcome="migrated_active", archive_path=str(archive.relative_to(root)),
                         active_path=str(source.relative_to(root)), target_tree_sha256=_tree_hash(source),
                         migration=migration)
        manifest["applied_at"] = now()
        manifest["manifest_sha256"] = hash_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        write_json(manifest_path, manifest)
        if any(e["outcome"] in {"blocked_live_state", "planned_migration"} for e in manifest["entries"]):
            raise RuntimeError("Migration did not reach a final outcome")
        cutover = {
            "schema_version": 1,
            "kind": "runs_to_repetition_cutover",
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "active_run_ids": [e["source_run_id"] for e in manifest["entries"] if e["outcome"] in {"migrated_active", "already_active"}],
            "archive_root": str(archive_root.relative_to(root)),
            "cutover_at": now(),
        }
        write_json(root / "migration/CUTOVER.json", cutover)
    return manifest


def verify(root, manifest_path):
    root = Path(root).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("manifest_sha256"):
        expected = manifest["manifest_sha256"]
        actual = hash_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        if expected != actual:
            raise ValueError("Migration manifest was modified")
    checked = []
    for entry in manifest["entries"]:
        if entry["outcome"] == "migrated_active":
            active = root / entry["active_path"]
            archive = root / entry["archive_path"]
            if not active.is_dir() or not archive.is_dir():
                raise ValueError(f"Missing migration copy: {entry['source_run_id']}")
            if _tree_hash(archive) != entry["source_tree_sha256"]:
                raise ValueError(f"Archive bytes changed: {entry['source_run_id']}")
            if read_json(active / "plan.json").get("schema_version") != 2:
                raise ValueError(f"Active run is not schema 2: {entry['source_run_id']}")
            checked.append(entry["source_run_id"])
    cutover = root / "migration/CUTOVER.json"
    if not cutover.exists():
        raise ValueError("Migration CUTOVER is missing")
    return {"verified": True, "active_runs": checked, "entries": len(manifest["entries"])}
