"""Wake after image derivation, cold-audit repaired images, publish passing ones."""

from __future__ import annotations

import argparse
import os
import select
import traceback
from pathlib import Path

from core.config import resolve
from core.io import now, read_json, write_json
from core.locking import file_lock
from runtime.scripts.consolidate_gold_audits import consolidate
from runtime.scripts.gold_image_audit import audit
from runtime.scripts.repair_image_ownership import publish


def wait_for_exit(pid: int) -> None:
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def run(root: Path, pid: int, store_id: str, audit_id: str, workers: int) -> dict:
    store = Path(resolve(root / "config.yaml")["paths"]["nas_root"]) / store_id
    out = root / "state/gold-image-audit" / (audit_id + "-coordinator")
    out.mkdir(parents=True, exist_ok=False)
    status = {"stage": "waiting-for-image-derivation", "started_at": now(), "pid": pid}
    write_json(out / "status.json", status)
    wait_for_exit(pid)
    try:
        completion = read_json(store / "dataset/complete.json")
        instances_path = store / "dataset/instances.json"
        rows = read_json(instances_path)
        ids = read_json(store / "dataset/task-ids.json")
        if len(rows) != completion["tasks"] or set(ids) != {row["instance_id"] for row in rows}:
            raise ValueError("Derived image cohort is incomplete or inconsistent")
        status.update(stage="cold-gold", tasks=len(rows), updated_at=now())
        write_json(out / "status.json", status)
        result = audit(
            root,
            instances_path,
            audit_id + "-audit",
            workers,
            cold_start=True,
            nas_root="/VM/wangtianyi/DyConGlobal",
        )
        summary = root / "state/gold-image-audit" / (audit_id + "-audit") / "summary.json"
        cfg = resolve(root / "config.yaml")
        receipt = publish(cfg, rows, summary)
        initial_path = root / "state/gold-image-audit/nonverified600-repair-20261020/pre-repair-results.json"
        initial = read_json(initial_path)
        initial_ids = set(initial["failed"])
        retry_ids = {row["instance_id"] for row in rows}
        if not retry_ids <= initial_ids:
            raise ValueError("Derived image cohort contains tasks outside original failed selection")
        # Convert the pre-repair merged result back to a summary input, then bind
        # only tested derived-image outcomes into the final merged evidence.
        baseline_path = out / "baseline-summary.json"
        write_json(baseline_path, {
            "results": [row for row in initial["results"] if row["instance_id"] not in retry_ids]
        })
        merged_path = out / "consolidated-final.json"
        consolidate([baseline_path, summary], merged_path)
        final = read_json(merged_path)
        status.update(
            stage="complete" if final["tasks"] == len(initial["results"]) else "needs-investigation",
            audit_summary=str(summary), tasks=result["tasks"], resolved=result["resolved"],
            infra_error=result["infra_error"], published=receipt["tasks"],
            failed=sorted(set(ids) - set(receipt["tasks"])),
            merged_tasks=final["tasks"], merged_resolved=final["resolved"],
            merged_cold_lifecycle_verified=final["cold_lifecycle_verified"],
            final_summary=str(merged_path), finished_at=now(),
        )
    except Exception:
        status.update(stage="error", error=traceback.format_exc(), finished_at=now())
    write_json(out / "status.json", status)
    write_json(out / "summary.json", status)
    print(status, flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    with file_lock(args.root / "state/gold-image-audit" / (args.audit_id + ".coordinator.lock")):
        raise SystemExit(
            0
            if run(args.root.resolve(), args.pid, args.store, args.audit_id, args.workers).get("stage")
            == "passed"
            else 1
        )
