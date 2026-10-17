"""Remove audited interrupted checkpoints, preserving all limit terminations.

Audit is read-only. Apply requires the frozen audit plan and unchanged catalog.
Only project-owned trajectory/patch payloads are removed; legacy data is untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from baseline.catalog import Catalog
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock

LIMIT_STATUSES = {"LimitsExceeded", "ContextWindowExceededError", "TimeoutError", "WallTimeExceeded"}
NORMAL_STATUSES = LIMIT_STATUSES | {"Submitted", "RepeatedFormatError"}


def classify(termination, attempt_statuses, run_stages, *, orphaned=False):
    """Explicit terminal/limit evidence always wins over a stale sidecar."""
    if termination in NORMAL_STATUSES:
        return None
    if termination:
        # ContainerCorrupted alone cannot distinguish TTL from external failure.
        return None
    if any(s in NORMAL_STATUSES for s in attempt_statuses):
        return None
    if "Interrupted" in attempt_statuses:
        return "explicit_interrupted_attempt"
    if any(s in {"stopped_by_user", "paused", "interrupted", "killed"} for s in run_stages):
        return "stopped_run_unfinalized_checkpoint"
    if orphaned and all(s is None for s in attempt_statuses):
        return "orphaned_unfinalized_checkpoint"
    return None


def audit(root, source, cleanup_id):
    root, source = Path(root).resolve(), Path(source).resolve()
    out = owned_path(root, f"baseline/provenance/{identifier(cleanup_id)}")
    if out.exists():
        raise FileExistsError(out)
    catalog = Catalog(root)
    catalog.verify()
    data = catalog.read()
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()
    active = [line for line in processes if str(source) in line and "clean_interrupted" not in line]
    rows, retained = [], []
    for rid, row in data["rollouts"].items():
        if row["termination_reason"] in NORMAL_STATUSES:
            continue
        raw_path = owned_path(root, row["trajectory_path"])
        raw = read_json(raw_path)
        messages = raw.get("messages", [])
        steps = sum(m.get("role") == "assistant" for m in messages)
        limit = raw.get("info", {}).get("config", {}).get("agent", {}).get("step_limit")
        evidence = []
        for alias in row.get("provenance", {}).get("source_occurrences", []):
            trajectory = owned_path(source, alias["path"])
            attempt_file = trajectory.parent / "attempt.json"
            run_file = source / "runs" / identifier(alias["run_id"]) / "status.json"
            entry = {"source_occurrence": alias}
            for label, f in [("attempt", attempt_file), ("run_status", run_file)]:
                entry[label] = (
                    {"path": str(f), "record": read_json(f), "sha256": sha256(f)} if f.exists() else None
                )
            entry["source_mtime"] = trajectory.stat().st_mtime if trajectory.exists() else None
            evidence.append(entry)
        attempt_statuses = [
            e["attempt"]["record"].get("exit_status") if e["attempt"] else None for e in evidence
        ]
        stages = [e["run_status"]["record"].get("stage") if e["run_status"] else None for e in evidence]
        orphaned = (
            bool(evidence)
            and not active
            and all(
                e["source_mtime"] is not None and time.time() - e["source_mtime"] > 86400 for e in evidence
            )
        )
        reason = classify(row["termination_reason"], attempt_statuses, stages, orphaned=orphaned)
        # A stopped checkpoint at the configured step boundary is conservative: keep.
        if limit and steps >= limit:
            reason = None
        entry = {
            "rollout_id": rid,
            "task_id": data["tasks"][row["task_key"]]["instance_id"],
            "run_id": row["run_id"],
            "attempt_id": row["attempt_id"],
            "termination_reason": row["termination_reason"],
            "assistant_steps": steps,
            "step_limit": limit,
            "reason": reason,
            "evidence": evidence,
            "trajectory_sha256": row["trajectory_sha256"],
            "patch_sha256": row["patch_sha256"],
        }
        if reason:
            if row["resolved"] is not None or row["submission_status"] == "submitted":
                raise ValueError("Refuse to remove a graded or submitted trajectory")
            rows.append(entry)
        else:
            entry["retention_reason"] = "explicit_terminal_status_or_possible_container_lifetime_limit"
            retained.append(entry)
    out.mkdir(parents=True)
    write_json(out / "manifest-before.json", data)
    plan = {
        "schema_version": 1,
        "cleanup_id": cleanup_id,
        "created_at": now(),
        "catalog_hash": hash_json(data),
        "source_root": str(source),
        "active_source_processes": active,
        "remove": rows,
        "retained_candidates": retained,
        "reason_counts": dict(Counter(r["reason"] for r in rows)),
    }
    write_json(out / "plan.json", plan)
    return out / "plan.json"


def export_catalog(root, out):
    data = Catalog(root).read()
    fields = [
        "rollout_id",
        "task_id",
        "run_id",
        "attempt_id",
        "partition",
        "purpose",
        "status",
        "submission_status",
        "evaluation_status",
        "resolved",
        "selected_evaluation_id",
        "trajectory_path",
        "trajectory_sha256",
        "model_name",
    ]
    rows, aliases = [], []
    for rid, row in data["rollouts"].items():
        task = data["tasks"][row["task_key"]]
        entry = {k: row[k] for k in fields if k in row}
        entry.update(
            rollout_id=rid,
            task_id=task["instance_id"],
            partition=task["partition"],
            purpose=",".join(row["purposes"]),
            status="resolved" if row["resolved"] is True else row["submission_status"],
            model_name=row["provenance"].get("legacy_model_identity", {}).get("model_name", ""),
        )
        rows.append(entry)
        for alias in row["provenance"].get("source_occurrences", []):
            aliases.append({"rollout_id": rid, "task_id": task["instance_id"], **alias})
    for name, columns, records in [
        ("trajectory-index.csv", fields, rows),
        (
            "source-occurrences.csv",
            ["rollout_id", "task_id", "path", "run_id", "attempt_id", "selected"],
            aliases,
        ),
    ]:
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    return rows, aliases


def apply(root, plan_file):
    root, plan_file = Path(root).resolve(), Path(plan_file).resolve()
    if not plan_file.is_relative_to(root / "baseline/provenance"):
        raise ValueError("Cleanup plan must be project-owned provenance")
    plan = read_json(plan_file)
    out = plan_file.parent
    with file_lock(root / "state/interrupted-cleanup.lock"):
        catalog = Catalog(root)
        if (out / "receipt.json").exists():
            receipt = read_json(out / "receipt.json")
            if receipt["plan_sha256"] != sha256(plan_file):
                raise ValueError("Cleanup plan changed")
            catalog.verify()
            return receipt
        data = catalog.read()
        if hash_json(data) != plan["catalog_hash"]:
            raise ValueError("Catalog changed after audit; prepare another plan")
        remove_ids = {r["rollout_id"] for r in plan["remove"]}
        payloads = []
        for rid in remove_ids:
            row = data["rollouts"][rid]
            if (
                row["termination_reason"]
                or row["resolved"] is not None
                or row["submission_status"] == "submitted"
            ):
                raise ValueError("Only unfinalized, unsubmitted, ungraded checkpoints are eligible")
            for kind, parent in [("trajectory", "trajectories"), ("patch", "patches")]:
                path = owned_path(root, row[kind + "_path"])
                if path.parent != (root / "baseline" / parent / rid).resolve():
                    raise ValueError("Unexpected cleanup target")
                if sha256(path) != row[kind + "_sha256"]:
                    raise ValueError("Payload changed after audit")
                payloads.append(path)
        if any(remove_ids.intersection(e["outcomes"]) for e in data["evaluations"].values()):
            raise ValueError("Removed rows are referenced by evaluation evidence")
        if any(remove_ids.intersection(s.values()) for s in data["baseline_selections"].values()):
            raise ValueError("Removed rows are selected as baseline")
        # First retire catalog rows atomically. A crash can leave unreferenced files,
        # but cannot leave training records pointing at deleted payloads.
        with catalog.transaction() as updated:
            if hash_json(updated) != plan["catalog_hash"]:
                raise ValueError("Concurrent catalog modification")
            for rid in remove_ids:
                del updated["rollouts"][rid]
        for path in payloads:
            path.unlink()
            if not any(path.parent.iterdir()):
                path.parent.rmdir()
        rows, aliases = export_catalog(root, out)
        result = {
            "cleanup_id": plan["cleanup_id"],
            "completed_at": now(),
            "plan_sha256": sha256(plan_file),
            "removed": len(remove_ids),
            "removed_by_reason": plan["reason_counts"],
            "remaining": len(rows),
            "remaining_status_counts": dict(Counter(r["status"] for r in rows)),
            "remaining_termination_counts": dict(
                Counter(r["termination_reason"] for r in catalog.read()["rollouts"].values())
            ),
            "remaining_source_occurrences": len(aliases),
            "verified": catalog.verify(),
            "exports": {n: sha256(out / n) for n in ["trajectory-index.csv", "source-occurrences.csv"]},
        }
        write_json(out / "receipt.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cleanup-id", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        print(audit(args.root, args.source, args.cleanup_id))
    else:
        print(json.dumps(apply(args.root, args.plan), indent=2))


if __name__ == "__main__":
    main()
