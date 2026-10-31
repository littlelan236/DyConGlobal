"""Physical, immutable import of explicitly supplied old rollout records."""

from __future__ import annotations

import shutil
from pathlib import Path

from application.execution.evidence import RunEvidence
from catalog.store import Task
from core.io import artifact_reference, hash_json, identifier, owned_path, read_json, run_path, sha256
from core.locking import file_lock


def import_rollout(root, spec):
    root = Path(root).resolve()
    source = Path(spec["source"]).resolve()
    original = read_json(source)
    task = Task.model_validate(spec["task"]).model_dump()
    iid = original.get("instance_id", task["instance_id"])
    if iid != task["instance_id"]:
        raise ValueError("Trajectory belongs to another task")
    if spec.get("expected_sha256") and sha256(source) != spec["expected_sha256"]:
        raise ValueError("Source trajectory hash mismatch")
    info = original.get("info", {})
    run_id, attempt_id = identifier(spec["run_id"]), identifier(spec["attempt_id"])
    digest = sha256(source)
    task_key = hash_json({k: task[k] for k in ["benchmark", "revision", "instance_id"]})
    # Same exact stored trajectory is not a new independent sample, regardless of destination run.
    rid = "r-" + hash_json({"task_key": task_key, "trajectory_sha256": digest})[:32]
    termination = info.get("exit_status", "Unknown")
    workspace_subject = spec.get("subject")
    submission = info.get("submission") or "" if not workspace_subject else ""
    with file_lock(root / f"state/import-{rid}.lock"):
        run_catalog = RunEvidence(root, run_id)
        run_data = run_catalog.read()
        if rid in run_data["rollouts"]:
            existing = run_data["rollouts"][rid]
            if run_data["tasks"].get(task_key) != task or existing["model_fingerprint"] != spec["model_fingerprint"]:
                raise ValueError("Conflicting task metadata or model for reused trajectory")
            with run_catalog.transaction() as data:
                data["rollouts"][rid]["purposes"] = sorted(set(existing["purposes"] + spec.get("purposes", [])))
            run_catalog.verify()
            return rid
        artifact_root = root / "runs" / run_id / "rollout"
        trajectory = owned_path(artifact_root, f"trajectories/{rid}/raw.traj.json")
        patch = owned_path(artifact_root, f"patches/{rid}/patch.diff")
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        if not workspace_subject:
            patch.parent.mkdir(parents=True, exist_ok=True)
        # Incomplete imports are recoverable; existing bytes must agree, never overwrite.
        if trajectory.exists() and sha256(trajectory) != digest:
            raise ValueError("Import destination contains different data")
        if not trajectory.exists():
            shutil.copyfile(source, trajectory)
        if not workspace_subject:
            patch.write_text(submission, encoding="utf-8")
        if sha256(trajectory) != digest:
            raise ValueError("Source changed during import")
        row = {
            "task_key": task_key,
            "benchmark": task["benchmark"],
            "revision": task["revision"],
            "agent": spec.get("agent", "unknown"),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "purposes": spec.get("purposes", []),
            "trajectory_path": artifact_reference(root, trajectory),
            "trajectory_sha256": digest,
            "submission_status": "not_applicable" if workspace_subject else (
                "submitted" if termination == "Submitted" else "unsubmitted"),
            "termination_reason": termination,
            "evaluation_status": "not_run",
            "resolved": None,
            "selected_evaluation_id": None,
            "model_fingerprint": spec["model_fingerprint"],
            "provenance": {
                "source": str(source),
                "source_sha256": digest,
                "source_run_id": run_id,
                "source_attempt_id": attempt_id,
                "reused_from": info.get("tact", {}).get("reused_from"),
            },
        }
        if task["benchmark"] == "skillsbench":
            row["task_package_sha256"] = task["instance"]["package_sha256"]
        if workspace_subject:
            from application.execution.subject import verify_subject
            manifest_path = Path(workspace_subject).resolve()
            if sha256(manifest_path) != spec["subject_sha256"]:
                raise ValueError("Workspace subject changed before import")
            subject_root = artifact_root / "submissions" / rid
            subject_root.mkdir(parents=True, exist_ok=True)
            target = subject_root / "subject.json"
            if target.exists() and sha256(target) != spec["subject_sha256"]:
                raise ValueError("Conflicting immutable workspace subject")
            if not target.exists():
                shutil.copyfile(manifest_path, target)
            for output in read_json(manifest_path)["outputs"]:
                src = manifest_path.parent / output["path"]
                dst = subject_root / output["path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(src, dst)
            row.update({"evaluation_subject": {"kind": "workspace", "schema_version": 1,
                                                "re_evaluation": False},
                        "subject_path": artifact_reference(root, target),
                        "subject_sha256": sha256(target)})
            verify_subject(root, row)
        else:
            row.update({"patch_path": artifact_reference(root, patch), "patch_sha256": sha256(patch)})
        sample = spec.get("sample") or {}
        if sample:
            row.update({
                "sample_key": sample.get("sample_key"),
                "repetition_id": sample.get("repetition_id"),
                "repetition_index": sample.get("repetition_index"),
                "sample_seed": sample.get("sample_seed"),
                "seed_policy": sample.get("seed_policy"),
            })
            row["provenance"].update({"sample_key": sample.get("sample_key"),
                                      "repetition_id": sample.get("repetition_id"),
                                      "sample_seed": sample.get("sample_seed")})
        execution = spec.get("attempt_context")
        if execution:
            for key in ("effective_seed", "seed_round", "seed_policy"):
                row[key] = execution[key]
            row["provenance"]["attempt_context"] = execution
        run_catalog.register_rollout(rid, row, task)
        if workspace_subject and spec.get("evaluation") is not None:
            report = Path(spec["evaluation_report"]).resolve()
            if not report.is_relative_to(run_path(root, run_id).resolve()):
                raise ValueError("SkillsBench evaluation report must be run-local")
            outcome = {**spec["evaluation"], "subject_sha256": row["subject_sha256"]}
            run_catalog.record_evaluation(spec["evaluation_id"], report,
                {rid: outcome}, {"benchmark": "skillsbench", "policy": outcome["success_policy"],
                                 "live_verifier": True})
    return rid


def import_list(root, path):
    specs = read_json(path)
    if not isinstance(specs, list) or not specs:
        raise ValueError("Provide a nonempty explicit rollout import list")
    return [import_rollout(root, spec) for spec in specs]
