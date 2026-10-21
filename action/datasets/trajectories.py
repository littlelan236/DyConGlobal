"""Select canonical trajectories using the catalog, without enrichment or model IO."""

import hashlib
import json
from pathlib import Path

from catalog.store import Catalog
from core.io import hash_json, owned_path


def select_trajectories(cfg, rollout_ids=(), rollout_list=None):
    if rollout_ids and rollout_list is not None:
        raise ValueError("Use --rollout-id or --rollout-list, not both")
    catalog = Catalog(cfg["project_root"]).read()
    expected, cohorts = {}, {}
    selection_source = None
    if rollout_list is not None:
        raw = Path(rollout_list).read_bytes()
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("Rollout list must be a JSON list of IDs or binding objects")
        ids = []
        for entry in entries:
            if isinstance(entry, str):
                rid = entry
            elif isinstance(entry, dict) and isinstance(entry.get("rollout_id"), str):
                rid = entry["rollout_id"]
                digest = entry.get("trajectory_sha256")
                if digest is not None:
                    expected[rid] = digest
                if entry.get("cohort") is not None:
                    cohorts[rid] = str(entry["cohort"])
            else:
                raise ValueError("Invalid rollout list entry")
            ids.append(rid)
        selection_source = {
            "path": str(Path(rollout_list).resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    elif rollout_ids:
        ids = list(rollout_ids)
    else:
        benchmark = cfg["benchmark"]
        profile = benchmark["profile"]
        if profile is not None and profile not in catalog["baseline_selections"]:
            raise ValueError("Unknown baseline selection profile")
        ids = [
            rid
            for rid, row in catalog["rollouts"].items()
            if (
                benchmark["partition"] is None
                or catalog["tasks"][row["task_key"]]["partition"] == benchmark["partition"]
            )
            and (not benchmark["purposes"] or set(row["purposes"]).intersection(benchmark["purposes"]))
            and (profile is None or catalog["baseline_selections"][profile].get(row["task_key"]) == rid)
            and not row["provenance"].get("quarantined")
            and catalog["tasks"][row["task_key"]]["partition"] != "excluded"
        ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Rollout selection must be nonempty and contain no duplicate IDs")
    sources, seen, aliases = [], {}, {}
    for rid in sorted(ids):
        if rid not in catalog["rollouts"]:
            raise ValueError(f"Unknown rollout: {rid}")
        row = catalog["rollouts"][rid]
        task = catalog["tasks"][row["task_key"]]
        if row["provenance"].get("quarantined") or task["partition"] == "excluded":
            raise ValueError(f"Quarantined/excluded rollout cannot be analyzed: {rid}")
        if rid in expected and expected[rid] != row["trajectory_sha256"]:
            raise ValueError(f"Selection/manifest trajectory SHA mismatch: {rid}")
        digest = row["trajectory_sha256"]
        # Copied attempts do not constitute independent trajectories.
        if digest in seen:
            if seen[digest][1] != row["task_key"]:
                raise ValueError("Identical trajectory bound to different tasks")
            aliases[rid] = seen[digest][0]
            continue
        seen[digest] = (rid, row["task_key"])
        sources.append(
            {
                "rollout_id": rid,
                "trajectory_path": row["trajectory_path"],
                "trajectory_sha256": digest,
                "task_key": row["task_key"],
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "partition": task["partition"],
                "run_id": row["run_id"],
                "model_fingerprint": row["model_fingerprint"],
                "resolved": row["resolved"],
                "submission_status": row["submission_status"],
                "termination_reason": row["termination_reason"],
                "cohort": cohorts.get(rid),
            }
        )
    return {
        "catalog_revision": catalog["catalog_revision"],
        "catalog_sha256": hash_json(catalog),
        "selection_source": selection_source,
        "requested_rollout_ids": sorted(ids),
        "duplicate_trajectory_aliases": aliases,
        "sources": sources,
    }


def read_actions(root, source):
    """Read one SHA-bound byte snapshot; avoid ledger hydration and future outputs."""
    raw = owned_path(root, source["trajectory_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["trajectory_sha256"]:
        raise ValueError(f"Trajectory checksum mismatch: {source['rollout_id']}")
    messages = json.loads(raw).get("messages")
    if not isinstance(messages, list):
        raise ValueError("Trajectory must contain a messages list")
    return [
        {"message_index": i, "tool_calls": m.get("tool_calls") or []}
        for i, m in enumerate(messages)
        if m.get("role") == "assistant"
    ]
