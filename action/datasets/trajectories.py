"""Freeze canonical trajectory bindings across explicit source runs."""

import copy
import hashlib
import json
from pathlib import Path

from application.execution.evidence import RunEvidence
from core.io import hash_json, identifier, owned_path, read_json, sha256
from difficulty.datasets.selection import select_run


def source_key(source):
    return (source.get("source_run_id", source.get("run_id", "")), source["rollout_id"])


def select_trajectories(cfg, run_id=None, rollout_ids=(), rollout_list=None):
    if rollout_ids and rollout_list is not None:
        raise ValueError("Use --rollout-id or --rollout-list, not both")
    if isinstance(rollout_ids, (str, bytes)):
        raise ValueError("Rollout IDs must be a sequence, not a bare string")
    run_ids = run_id or cfg.get("source_run_id")
    if not run_ids:
        raise ValueError("Trajectory selection requires an explicit source run before rollout IDs")
    run_ids = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("Source runs must be nonempty and contain no duplicate IDs")
    run_ids = sorted(identifier(r) for r in run_ids)
    if rollout_ids and len(run_ids) != 1:
        raise ValueError("--rollout-id requires exactly one source run; use explicit bindings for multiple runs")
    requests = {r: [] for r in run_ids}
    entries, selection_source = {}, None
    explicit = bool(rollout_ids) or rollout_list is not None
    if rollout_list is not None:
        raw = Path(rollout_list).read_bytes()
        values = json.loads(raw)
        if not isinstance(values, list) or not values:
            raise ValueError("Rollout list must be a nonempty JSON list of IDs or binding objects")
        for value in values:
            if isinstance(value, str) and len(run_ids) == 1:
                value = {"rollout_id": value, "source_run_id": run_ids[0]}
            if not isinstance(value, dict) or set(value) - {"source_run_id", "rollout_id", "trajectory_sha256", "cohort"}:
                raise ValueError("Invalid rollout binding; multiple source runs require source_run_id")
            source_run = value.get("source_run_id", run_ids[0] if len(run_ids) == 1 else None)
            if source_run not in requests or not isinstance(value.get("rollout_id"), str):
                raise ValueError("Rollout binding requires a declared source_run_id and rollout_id")
            rid = identifier(value["rollout_id"])
            key = (source_run, rid)
            if key in entries:
                raise ValueError("Duplicate rollout binding")
            if value.get("cohort") is not None and not isinstance(value["cohort"], str):
                raise ValueError("Cohort must be a string")
            if value.get("trajectory_sha256") is not None and not isinstance(value["trajectory_sha256"], str):
                raise ValueError("Trajectory SHA must be a string")
            entries[key] = value
            requests[source_run].append(rid)
        if any(not ids for ids in requests.values()):
            raise ValueError("Each declared source run needs explicit rollout bindings")
        selection_source = {"path": str(Path(rollout_list).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    elif rollout_ids:
        requests[run_ids[0]] = list(rollout_ids)
    root = Path(cfg["project_root"])
    sources, seen, aliases, snapshots, runs, requested = [], {}, [], {}, [], []
    for run in run_ids:
        selected = select_run(root, run, requests[run])
        evidence_path = root / "runs" / run / "evidence.json"
        # Freeze the selected evidence, including exact evaluation/task bindings.
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        if hash_json(evidence) != selected["evidence_sha256"]:
            raise ValueError("Source evidence changed during selection")
        ids = selected["rollout_ids"]
        snapshot = {"schema_version": 1, "run_id": run, "revision": selected["evidence_revision"],
                    "rollouts": copy.deepcopy(selected["rollouts"]), "tasks": selected["tasks"], "evaluations": {}}
        for rid in ids:
            eid = selected["rollouts"][rid].get("selected_evaluation_id")
            if eid and eid not in snapshot["evaluations"]:
                evaluation = copy.deepcopy(evidence["evaluations"][eid])
                evaluation["outcomes"] = {k: v for k, v in evaluation["outcomes"].items() if k in ids}
                snapshot["evaluations"][eid] = evaluation
        RunEvidence(root, run).validate(snapshot)
        snapshots[run] = snapshot
        config_path = root / "runs" / run / "resolved-config.json"
        classifier_cwd = cfg.get("environment", {}).get("cwd")
        config_source = None
        if config_path.is_file():
            config_bytes = config_path.read_bytes()
            source_config = json.loads(config_bytes)
            classifier_cwd = source_config.get("environment", {}).get("cwd", classifier_cwd)
            config_source = {"path": str(config_path.relative_to(root)), "sha256": hashlib.sha256(config_bytes).hexdigest()}
        runs.append({"run_id": run, "evidence_path": str(evidence_path.relative_to(root)),
                     "evidence_revision": selected["evidence_revision"], "evidence_sha256": selected["evidence_sha256"],
                     "evidence_file_sha256": hashlib.sha256(evidence_bytes).hexdigest(), "snapshot_path": f"source-evidence/{run}.json",
                     "classifier_cwd": classifier_cwd, "config_source": config_source})
        for rid in ids:
            row = selected["rollouts"][rid]
            task = selected["tasks"][row["task_key"]]
            binding = entries.get((run, rid), {})
            if binding.get("trajectory_sha256") is not None and binding["trajectory_sha256"] != row["trajectory_sha256"]:
                raise ValueError(f"Selection/manifest trajectory SHA mismatch: {run}/{rid}")
            path = owned_path(root, row["trajectory_path"])
            if not path.is_relative_to((root / "runs" / run).resolve()):
                raise ValueError("Canonical trajectory must belong to its source run")
            key = {"source_run_id": run, "rollout_id": rid}
            requested.append(key)
            source = {"source_run_id": run, "run_id": run, "rollout_id": rid,
                      "trajectory_path": row["trajectory_path"], "trajectory_sha256": row["trajectory_sha256"],
                      "task_key": row["task_key"], "instance_id": task["instance_id"], "repo": task["repo"],
                      "partition": task["partition"], "model_fingerprint": row["model_fingerprint"],
                      "resolved": row["resolved"], "submission_status": row["submission_status"],
                      "termination_reason": row["termination_reason"], "cohort": binding.get("cohort"),
                      "classifier_cwd": classifier_cwd}
            digest = source["trajectory_sha256"]
            if digest in seen:
                canonical = seen[digest]
                if canonical["task_key"] != source["task_key"]:
                    raise ValueError("Identical trajectory bound to different tasks")
                if any(canonical[k] != source[k] for k in ("model_fingerprint", "cohort", "classifier_cwd")):
                    raise ValueError("Identical trajectory has conflicting model/cohort/classifier context")
                # Verify aliases too: a duplicate SHA declaration is not proof of its bytes.
                read_actions(root, source)
                aliases.append({"alias": key, "canonical": {"source_run_id": canonical["source_run_id"], "rollout_id": canonical["rollout_id"]}})
                continue
            seen[digest] = source
            sources.append(source)
    return {"schema_version": 2, "selection_mode": "explicit" if explicit else "all_eligible",
            "source_runs": runs, "selection_source": selection_source, "requested_rollouts": requested,
            "duplicate_trajectory_aliases": aliases, "sources": sources, "source_snapshots": snapshots}


def read_actions(root, source):
    """Read one SHA-bound byte snapshot; avoid ledger hydration and future outputs."""
    raw = owned_path(root, source["trajectory_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["trajectory_sha256"]:
        raise ValueError(f"Trajectory checksum mismatch: {source['rollout_id']}")
    trajectory = json.loads(raw)
    messages = trajectory.get("messages") if isinstance(trajectory, dict) else None
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("Trajectory must contain a messages list")
    return [
        {"message_index": i, "tool_calls": m.get("tool_calls") or []}
        for i, m in enumerate(messages)
        if m.get("role") == "assistant"
    ]
