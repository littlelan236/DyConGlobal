"""Freeze explicit multi-run bindings without estimator or resolved-only filtering."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from application.execution.evidence import RunEvidence
from core.io import artifact_roots, hash_json, identifier, owned_path, read_json, sha256


def select_sources(cfg, source_runs, *, rollout_ids=(), rollout_list=None, all_submitted=False):
    runs = [source_runs] if isinstance(source_runs, str) else list(source_runs or [])
    if not runs or len(runs) != len(set(runs)):
        raise ValueError("Declare nonempty, distinct source runs")
    runs = sorted(identifier(run) for run in runs)
    if isinstance(rollout_ids, (str, bytes)):
        raise ValueError("rollout_ids must be a sequence")
    if sum((bool(rollout_ids), rollout_list is not None, bool(all_submitted))) != 1:
        raise ValueError(
            "Choose exactly one explicit selector: rollout IDs, binding list, or --all-submitted"
        )
    if rollout_ids and len(runs) != 1:
        raise ValueError("Multiple runs require source_run_id + rollout_id bindings")
    requests = {run: [] for run in runs}
    bindings, selection_source = {}, None
    if rollout_list is not None:
        raw = Path(rollout_list).read_bytes()
        entries = json.loads(raw)
        if not isinstance(entries, list) or not entries:
            raise ValueError("Binding list must be a nonempty JSON list")
        selection_source = {
            "path": str(Path(rollout_list).resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for entry in entries:
            if isinstance(entry, str) and len(runs) == 1:
                entry = {"source_run_id": runs[0], "rollout_id": entry}
            if not isinstance(entry, dict) or set(entry) - {
                "source_run_id",
                "rollout_id",
                "trajectory_sha256",
            }:
                raise ValueError("Invalid explicit rollout binding")
            run = entry.get("source_run_id", runs[0] if len(runs) == 1 else None)
            if run not in requests:
                raise ValueError("Binding must name a declared source_run_id")
            rid = identifier(entry.get("rollout_id"))
            key = (run, rid)
            if key in bindings:
                raise ValueError("Duplicate rollout binding")
            bindings[key] = entry
            requests[run].append(rid)
        if any(not values for values in requests.values()):
            raise ValueError("Every declared source run needs explicit bindings")
    elif rollout_ids:
        requests[runs[0]] = list(rollout_ids)
        if len(requests[runs[0]]) != len(set(requests[runs[0]])):
            raise ValueError("Duplicate rollout binding")
    sources, run_records, snapshots, configs, dependencies = [], [], {}, {}, {}
    for run in runs:
        run_matches = [
            root / "runs" / run for root in artifact_roots(cfg, "run") if (root / "runs" / run).is_dir()
        ]
        if not run_matches:
            project = Path(cfg["project_root"]).resolve() / "runs" / run
            run_matches = [project] if project.is_dir() else []
        if not run_matches:
            raise FileNotFoundError(f"Source run not found in registered run roots: {run}")
        if len({path.resolve() for path in run_matches}) != 1:
            raise ValueError(f"Conflicting source run locations across roots: {run}")
        run_dir = run_matches[0].resolve()
        config_path = run_dir / "resolved-config.json"
        if not config_path.is_file():
            raise ValueError(f"Source run has no frozen config: {run}")
        source_cfg = read_json(config_path)
        owner = Path(source_cfg.get("project_root", cfg["project_root"])).resolve()
        evidence = RunEvidence(owner, run)
        if not evidence.path.is_file() or evidence.path.resolve() != (run_dir / "evidence.json").resolve():
            raise ValueError("Run resolver and RunEvidence owner disagree")
        original = evidence.read()
        ids = requests[run]
        if all_submitted:
            ids = sorted(
                rid
                for rid, row in original["rollouts"].items()
                if row["submission_status"] == "submitted"
                and not row.get("provenance", {}).get("quarantined")
                and original["tasks"][row["task_key"]]["partition"] != "excluded"
            )
        if not ids:
            raise ValueError(f"No explicitly selected eligible rollouts: {run}")
        selected = evidence.select(ids)
        if selected["evidence_sha256"] != hash_json(original):
            raise ValueError("Run evidence changed during selection; retry with a stable snapshot")
        snapshot = {
            "schema_version": original["schema_version"],
            "run_id": run,
            "revision": original["revision"],
            "rollouts": {},
            "tasks": copy.deepcopy(selected["tasks"]),
            "evaluations": {},
        }
        for rid in selected["rollout_ids"]:
            row = selected["rollouts"][rid]
            expected = bindings.get((run, rid), {}).get("trajectory_sha256")
            if expected is not None and expected != row["trajectory_sha256"]:
                raise ValueError(f"Expected trajectory SHA mismatch: {run}/{rid}")
            trajectory = owned_path(owner, row["trajectory_path"])
            if not trajectory.is_relative_to(run_dir):
                raise ValueError("Canonical trajectory is outside its owning run")
            if sha256(trajectory) != row["trajectory_sha256"]:
                raise ValueError(f"Canonical trajectory checksum mismatch: {run}/{rid}")
            task = selected["tasks"][row["task_key"]]
            if row["model_fingerprint"] != source_cfg["model"]["fingerprint"]:
                raise ValueError("Rollout model differs from frozen run model")
            snapshot["rollouts"][rid] = copy.deepcopy(original["rollouts"][rid])
            eid = row.get("selected_evaluation_id")
            if eid:
                evaluation = copy.deepcopy(original["evaluations"][eid])
                evaluation["outcomes"] = {
                    k: v for k, v in evaluation["outcomes"].items() if k in selected["rollout_ids"]
                }
                snapshot["evaluations"][eid] = evaluation
            sources.append(
                {
                    "source_run_id": run,
                    "rollout_id": rid,
                    "source_project_root": str(owner),
                    "run_directory": str(run_dir),
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": row["trajectory_sha256"],
                    "task_key": row["task_key"],
                    "benchmark": task["benchmark"],
                    "revision": task["revision"],
                    "instance_id": task["instance_id"],
                    "model_fingerprint": row["model_fingerprint"],
                    "attempt_id": row.get("attempt_id"),
                    "repetition_id": row.get("repetition_id"),
                    "submission_status": row["submission_status"],
                    "resolved": row["resolved"],
                }
            )
            dependencies[str(trajectory)] = row["trajectory_sha256"]
        evidence.validate(snapshot)
        snapshots[run], configs[run] = snapshot, source_cfg
        dependencies[str(config_path)] = sha256(config_path)
        run_records.append(
            {
                "source_run_id": run,
                "run_directory": str(run_dir),
                "evidence_revision": original["revision"],
                "evidence_sha256": hash_json(original),
                "config_sha256": sha256(config_path),
                "snapshot": f"source-evidence/{run}/evidence.json",
            }
        )
    selection = {
        "schema_version": 1,
        "selection_mode": "all_submitted_frozen_bindings" if all_submitted else "explicit",
        "source_runs": run_records,
        "selection_source": selection_source,
        "bindings": [
            {k: s[k] for k in ("source_run_id", "rollout_id", "trajectory_sha256")} for s in sources
        ],
    }
    return {
        "selection": selection,
        "sources": sources,
        "snapshots": snapshots,
        "configs": configs,
        "dependencies": dependencies,
    }
