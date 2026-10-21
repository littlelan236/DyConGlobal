"""Freeze postprocessing input evidence from one explicit source run."""
from application.execution.evidence import RunEvidence
from core.io import hash_json, sha256, owned_path, read_json, write_json


def rollout_selection(cfg):
    run_id = cfg.get("source_run_id") or cfg.get("run_id")
    if not run_id:
        raise ValueError("Analysis selection requires an explicit source run")
    root = owned_path(cfg["project_root"], f"runs/{cfg.get('run_id') or run_id}")
    path = root / "analysis-selection.json"
    if path.exists():
        selection = read_json(path)
        digest = selection.pop("selection_sha256")
        if hash_json(selection) != digest:
            raise ValueError("Analysis selection was modified")
        if selection.get("source_run_id", run_id) != run_id:
            raise ValueError("Frozen selection belongs to a different source run")
    else:
        selection = RunEvidence(cfg["project_root"], run_id).select()
        if not selection["rollout_ids"]:
            raise ValueError("Run has no canonical trajectories to process")
        write_json(path, {**selection, "selection_sha256": hash_json(selection)})
    for rid, row in selection["rollouts"].items():
        for field in ("trajectory", "patch"):
            if sha256(owned_path(cfg["project_root"], row[field+"_path"])) != row[field+"_sha256"]:
                raise ValueError(f"Changed {field}: {rid}")
    return selection
