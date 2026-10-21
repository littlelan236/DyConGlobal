"""Freeze postprocessing inputs by run identity, never by the current cohort."""
from catalog.store import Catalog
from core.io import hash_json, sha256, owned_path, read_json, write_json


def rollout_selection(cfg):
    data = Catalog(cfg["project_root"]).read()
    run_id = cfg.get("run_id")
    if not run_id:
        return Catalog(cfg["project_root"]).select(
            partition=cfg["benchmark"]["partition"], purposes=cfg["benchmark"]["purposes"],
            profile=cfg["benchmark"]["profile"])
    root = owned_path(cfg["project_root"], f"runs/{run_id}")
    path = root / "analysis-selection.json"
    if path.exists():
        selection = read_json(path)
        digest = selection.pop("selection_sha256")
        if hash_json(selection) != digest:
            raise ValueError("Analysis selection was modified")
    else:
        if cfg["operation"] == "rollout":
            rollouts = {rid: row for rid, row in data["rollouts"].items()
                        if row["run_id"] == run_id and not row["provenance"].get("quarantined")}
        else:
            harness = read_json(root / "harness/selection.json")
            rollouts = harness["rollouts"]
        if not rollouts:
            raise ValueError("Run has no canonical trajectories to process")
        selection = {"catalog_revision": data["catalog_revision"], "catalog_sha256": hash_json(data),
                     "rollout_ids": sorted(rollouts), "rollouts": rollouts,
                     "tasks": {r["task_key"]: data["tasks"][r["task_key"]] for r in rollouts.values()}}
        write_json(path, {**selection, "selection_sha256": hash_json(selection)})
    for rid, row in selection["rollouts"].items():
        for field in ("trajectory", "patch"):
            if sha256(owned_path(cfg["project_root"], row[field+"_path"])) != row[field+"_sha256"]:
                raise ValueError(f"Changed {field}: {rid}")
    return selection
