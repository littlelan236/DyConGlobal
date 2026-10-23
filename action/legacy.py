"""Read-only schema-1 action receipt verification; never writes or migrates."""
import json
from action.artifacts import analysis_directory
from action.datasets.trajectories import read_actions
from action.evaluation.summary import legacy_summarize
from core.io import hash_json, owned_path, read_json, sha256

def check_analysis(cfg, analysis_id):
    out = analysis_directory(cfg["project_root"], analysis_id)
    receipt = read_json(out / "receipt.json")
    required = {"selection.json", "resolved-config.json", "summary.json", "examples.json", "labels.jsonl"}
    if set(receipt["files"]) != required:
        raise ValueError("Incomplete action receipt")
    for name, digest in receipt["files"].items():
        if sha256(owned_path(out, name)) != digest:
            raise ValueError(f"Action artifact changed: {name}")
    selection = read_json(out / "selection.json")
    if hash_json(selection) != receipt["selection_sha256"]:
        raise ValueError("Action selection changed")
    for source in selection["sources"]:
        read_actions(cfg["project_root"], source)
    rows = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    expected = {
        (source["rollout_id"], s) for source in selection["sources"] for s in range(source["assistant_steps"])
    }
    if len(rows) != len(expected) or {(r["rollout_id"], r["step_id"]) for r in rows} != expected:
        raise ValueError("Action step coverage changed")
    summary = read_json(out / "summary.json")
    if any(summary[key] != value for key, value in legacy_summarize(rows, selection["sources"]).items()):
        raise ValueError("Action summary disagrees with labels")
    return {
        "analysis_id": analysis_id,
        "verified": True,
        "sources": len(selection["sources"]),
        "steps": len(rows),
        "taxonomy_version": receipt["taxonomy_version"],
    }
