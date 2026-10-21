"""Run first, trajectory second: shared difficulty dataset selection contract."""
import json
from pathlib import Path
from application.execution.evidence import RunEvidence
from core.io import identifier


def select_run(root, source_run, rollout_ids=(), rollout_list=None):
    if not source_run:
        raise ValueError("Specify source_run before selecting trajectories")
    source_run = identifier(source_run)
    if rollout_ids and rollout_list is not None:
        raise ValueError("Use rollout IDs or a rollout list, not both")
    if rollout_list is not None:
        entries = json.loads(Path(rollout_list).read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("Rollout list must be a nonempty JSON list")
        rollout_ids = [r if isinstance(r, str) else r["rollout_id"] for r in entries]
    if isinstance(rollout_ids, (str, bytes)):
        raise ValueError("Rollout IDs must be a sequence, not a bare string")
    result = RunEvidence(root, source_run).select(rollout_ids)
    if not result["rollout_ids"]:
        raise ValueError("No eligible trajectories in the specified source run")
    if rollout_list is not None:
        for entry in entries:
            if isinstance(entry, dict) and entry.get("trajectory_sha256") is not None:
                if result["rollouts"][entry["rollout_id"]]["trajectory_sha256"] != entry["trajectory_sha256"]:
                    raise ValueError("Selected trajectory SHA mismatch")
    return result
