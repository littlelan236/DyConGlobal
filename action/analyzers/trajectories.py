"""Official trajectory action analysis and receipt verification workflows."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from action.analyzers.taxonomy import VERSION, Classifier, classify_action_evidence
from action.datasets.trajectories import read_actions, select_trajectories
from action.evaluation.summary import summarize
from core.io import artifact_dir, hash_json, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact

CODE_FILES = [
    "action/analyzers/taxonomy.py",
    "action/analyzers/trajectories.py",
    "action/datasets/trajectories.py",
    "action/evaluation/summary.py",
]


@exclusive_artifact("action", "analysis_id")
def classify_trajectories(cfg, analysis_id, rollout_ids=(), rollout_list=None):
    root = Path(cfg["project_root"])
    out = artifact_dir(cfg, "action", analysis_id)
    if out.exists():
        raise FileExistsError(f"Use a new analysis ID: {out}")
    selection = select_trajectories(cfg, rollout_ids, rollout_list)
    sources, rows, examples = [], [], {}
    for source in selection["sources"]:
        steps = read_actions(root, source)
        sources.append({**source, "assistant_steps": len(steps)})
        classifier = Classifier(cwd=cfg["environment"]["cwd"])
        for s, step in enumerate(steps):
            row = {
                "action_id": f"{source['rollout_id']}:{s}",
                "rollout_id": source["rollout_id"],
                "step_id": s,
                "message_index": step["message_index"],
                "total_steps": len(steps),
                **classify_action_evidence(step, classifier),
                "phase_bin": min(4, 5 * s // max(1, len(steps) - 1)),
            }
            rows.append(row)
            bucket = examples.setdefault(row["label"], [])
            if len(bucket) < 12:
                bucket.append(row)
    selection["sources"] = sources
    metadata = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "taxonomy_version": VERSION,
        "created_at": now(),
        "config_hash": cfg["config_hash"],
        "selection_sha256": hash_json(selection),
        "code": {name: sha256(Path(__file__).resolve().parents[2] / name) for name in CODE_FILES},
        "packages": {name: version(name) for name in ("tree-sitter", "tree-sitter-bash")},
        "python_version": platform.python_version(),
        "unit": "one assistant message, one primary category; unsupported shapes are OTHER",
        "scope_note": "Static requested behavior, not execution success or verified psychological intent",
    }
    metadata["input_identity"] = hash_json({k: v for k, v in metadata.items() if k != "created_at"})
    summary = {**metadata, **summarize(rows, sources)}
    # Never expose a partially written analysis as a completed artifact.
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".action-", dir=out.parent) as staging:
        stage = Path(staging)
        write_json(stage / "selection.json", selection)
        write_json(stage / "resolved-config.json", cfg)
        write_json(stage / "summary.json", summary)
        write_json(stage / "examples.json", examples)
        with (stage / "labels.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(
            stage / "receipt.json",
            {
                **metadata,
                "files": {
                    name: sha256(stage / name)
                    for name in (
                        "selection.json",
                        "resolved-config.json",
                        "summary.json",
                        "examples.json",
                        "labels.jsonl",
                    )
                },
            },
        )
        stage.rename(out)
    return str(out)


def check_analysis(cfg, analysis_id):
    out = artifact_dir(cfg, "action", analysis_id)
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
    if any(summary[key] != value for key, value in summarize(rows, selection["sources"]).items()):
        raise ValueError("Action summary disagrees with labels")
    return {
        "analysis_id": analysis_id,
        "verified": True,
        "sources": len(selection["sources"]),
        "steps": len(rows),
        "taxonomy_version": receipt["taxonomy_version"],
    }
