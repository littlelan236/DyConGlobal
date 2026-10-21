"""Wake after a 600-task gold audit and cold-retry unsuccessful tasks."""

from __future__ import annotations

import argparse
import copy
import os
import select
import traceback
from pathlib import Path

from core.io import now, read_json, write_json
from core.locking import file_lock
from runtime.containers.repairs import environment_overrides
from runtime.scripts.gold_image_audit import audit
from runtime.scripts.nas_cold_gold import purge


def wait_for_exit(pid: int) -> None:
    """Wait for one process-exit event without polling."""
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def failed_ids(summary: dict, all_ids: set[str]) -> set[str]:
    results = {row["instance_id"]: row for row in summary.get("results", [])}
    return {
        iid
        for iid in all_ids
        if (row := results.get(iid)) is None
        or row.get("resolved") is not True
        or row.get("status") != "completed"
        or row.get("cold_lifecycle_verified") is not True
    }


def load_initial_state(root: Path, audit_id: str, rows: list[dict]) -> tuple[dict, Path]:
    folder = root / "state/gold-image-audit" / audit_id
    summary_path = folder / "summary.json"
    if summary_path.exists():
        return read_json(summary_path), folder
    progress = folder / "progress.json"
    error = {
        "audit_id": audit_id,
        "tasks": len(rows),
        "results": read_json(progress).get("results", []) if progress.exists() else [],
        "error": "gold audit exited without summary.json",
    }
    write_json(folder / "wake-error.json", error)
    return error, folder


def retry_failed(
    root: Path,
    rows: list[dict],
    initial: dict,
    initial_folder: Path,
    run_id: str,
    workers: int,
    max_attempts: int,
) -> dict:
    all_ids = {row["instance_id"] for row in rows}
    latest = {row["instance_id"]: row for row in initial.get("results", [])}
    pending = [row for row in rows if row["instance_id"] in failed_ids(initial, all_ids)]
    out = root / "state/gold-image-audit" / run_id
    out.mkdir(parents=False, exist_ok=False)
    cfg_path = initial_folder / "audit-config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing frozen audit config: {cfg_path}")
    cfg = copy.deepcopy(read_json(cfg_path))
    cfg.update(project_root=str(root), require_cold_images=True)
    cfg.setdefault("execution", {})["harness_workers"] = workers
    cfg.setdefault("images", {}).update(cleanup=True, load_workers=4)
    write_json(out / "initial-summary.json", initial)
    write_json(out / "instances.json", rows)
    write_json(
        out / "repair-plan.json",
        {
            "repair_boundary": "runtime.containers.repairs.environment_overrides",
            "tasks": [
                {
                    "instance_id": row["instance_id"],
                    "environment_overrides": environment_overrides(cfg, row),
                }
                for row in pending
            ],
        },
    )
    status = {
        "stage": "started",
        "started_at": now(),
        "initial_audit": str(initial_folder / "summary.json"),
        "initial_failed": len(pending),
        "attempts": [],
    }
    write_json(out / "status.json", status)

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        attempt_dir = out / f"attempt-{attempt}-preflight"
        attempt_dir.mkdir()
        purge(cfg, pending, attempt_dir, set())
        selected = out / f"attempt-{attempt}-ids.json"
        write_json(selected, [row["instance_id"] for row in pending])
        audit_id = f"{run_id}-attempt{attempt}"
        try:
            result = audit(root, out / "instances.json", audit_id, workers, selected, cold_start=True, cfg=cfg)
            summary_path = root / "state/gold-image-audit" / audit_id / "summary.json"
            status["attempts"].append(str(summary_path))
            current = {row["instance_id"]: row for row in result.get("results", [])}
            latest.update(current)
            pending = [row for row in pending if row["instance_id"] in failed_ids(result, all_ids)]
        except Exception:
            status["attempts"].append({"audit_id": audit_id, "error": traceback.format_exc()})
            break
        status.update(stage="retrying" if pending else "passed", pending=len(pending), updated_at=now())
        write_json(out / "status.json", status)

    failed = sorted(row["instance_id"] for row in pending)
    status.update(
        stage="passed" if not failed else "needs-investigation",
        failed=failed,
        resolved=sum(row.get("resolved") is True for row in latest.values()),
        cold_lifecycle_verified=sum(row.get("cold_lifecycle_verified") is True for row in latest.values()),
        results=sorted(latest.values(), key=lambda row: row["instance_id"]),
        finished_at=now(),
    )
    write_json(out / "summary.json", status)
    write_json(out / "status.json", status)
    return status


def run(root: Path, pid: int, instances_path: Path, audit_id: str, workers: int, max_attempts: int) -> int:
    rows = read_json(instances_path)
    wait_for_exit(pid)
    initial, folder = load_initial_state(root, audit_id, rows)
    run_id = audit_id + "-wakeup-repair"
    try:
        result = retry_failed(root, rows, initial, folder, run_id, workers, max_attempts)
    except Exception:
        result = {"stage": "error", "error": traceback.format_exc(), "finished_at": now()}
        out = root / "state/gold-image-audit" / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "summary.json", result)
        write_json(out / "status.json", result)
    print(result, flush=True)
    return 0 if result.get("stage") == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--max-attempts", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    with file_lock(args.root / "state/gold-image-audit" / (args.audit_id + ".wakeup.lock")):
        raise SystemExit(
            run(
                args.root.resolve(),
                args.pid,
                args.instances.resolve(),
                args.audit_id,
                args.workers,
                args.max_attempts,
            )
        )
