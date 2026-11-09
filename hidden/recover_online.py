"""Recover published artifacts from local bytes without any model requests."""

from pathlib import Path

from core.config import load_snapshot
from core.io import now, read_json, sha256, write_json
from core.locking import exclusive_run, file_lock
from hidden.artifacts import read_branch_index
from runtime.serving.spool import CaptureSpool


@exclusive_run
def recover_online(cfg, run_id):
    frozen = load_snapshot(cfg["project_root"], run_id)
    spool = CaptureSpool(
        frozen["project_root"],
        frozen["capture"]["export"],
        run_id,
        frozen["paths"]["nas_root"],
    )
    sessions = {}
    try:
        jobs = spool.recover()
        for path in sorted((spool.root / "sessions").glob("*/finalization.json")):
            sid = path.parent.name
            finalization = read_json(path)
            from application.execution.evidence import verified_trajectory
            trajectory = verified_trajectory(frozen["project_root"], {**finalization, "producer_run_id": run_id})
            if sha256(trajectory) != finalization["trajectory_sha256"]:
                raise ValueError("Recovery trajectory hash mismatch")
            coverage = read_json(path.parent / "coverage.json")
            branch_path = path.parent / "parallel/branch-coverage.json"
            branch_complete = not branch_path.exists() or not read_branch_index(spool.root, sid)["coverage"]["missing"]
            published = spool.publish_session(sid, frozen["capture"]["export"]["upload_max_elapsed_seconds"])
            state = (
                "complete"
                if published and not coverage["missing"] and branch_complete
                else "pending_upload"
                if not published
                else "partial"
            )
            sessions[sid] = state
            with file_lock(Path(cfg["project_root"]) / "state/product-manifests.lock"):
                for owner in ("hidden", "enrichment"):
                    manifest_path = Path(cfg["project_root"]) / owner / "manifest.json"
                    manifest = read_json(manifest_path)
                    entry = manifest.get("online_captures", {}).get(sid)
                    if entry:
                        entry.update(status=state, published=state == "complete",
                                     eligible_for_training=state == "complete" and entry.get("dataset_role") != "diagnostic")
                        write_json(manifest_path, manifest)
            if state == "complete":
                spool.release_session_metadata(sid)
        run_hidden = Path(cfg["project_root"]) / "runs" / run_id / "hidden"
        run_hidden.mkdir(parents=True, exist_ok=True)
        write_json(run_hidden / "publication.json", {
            "schema_version": 1, "run_id": run_id, "sessions": sessions,
            "nas_root": frozen["paths"]["nas_root"], "updated_at": now(),
        })
        return {"retried_jobs": jobs, "sessions": sessions}
    finally:
        spool.close(frozen["capture"]["export"]["upload_max_elapsed_seconds"])
