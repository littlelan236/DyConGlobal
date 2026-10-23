"""Project-owned, immutable canonical action analyses across explicit source runs."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from action.analyzers.taxonomy import VERSION, Classifier, classify_action_evidence
from action.datasets.trajectories import read_actions, select_trajectories, source_key
from action.evaluation.distributions import PHASE_SCHEMA, TRANSITION_SCHEMA, phase_metadata, summarize
from core.io import artifact_dir, hash_json, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact

CODE_FILES = [
    "action/analyzers/taxonomy.py", "action/analyzers/trajectories.py", "action/datasets/trajectories.py",
    "action/evaluation/distributions.py", "action/evaluation/visualize.py", "action/artifacts.py",
    "core/io.py", "core/cli.py", "core/locking.py", "difficulty/datasets/selection.py",
    "application/execution/evidence.py", "difficulty/receipts.py",
]


def classify_sources(root, sources, analysis_id):
    enriched, rows, examples = [], [], {}
    for source in sources:
        steps = read_actions(root, source)
        enriched.append({**source, "assistant_steps": len(steps)})
        if not isinstance(source["classifier_cwd"], str):
            raise ValueError("Action classification requires a frozen classifier cwd")
        classifier = Classifier(cwd=source["classifier_cwd"])
        for step_id, step in enumerate(steps):
            row = {"action_id": f"{source['source_run_id']}:{source['rollout_id']}:{step_id}",
                   "analysis_id": analysis_id, "source_run_id": source["source_run_id"],
                   "rollout_id": source["rollout_id"], "step_id": step_id,
                   "message_index": step["message_index"], "total_steps": len(steps),
                   **classify_action_evidence(step, classifier), **phase_metadata(step_id, len(steps))}
            rows.append(row)
            bucket = examples.setdefault(row["label"], [])
            if len(bucket) < 12:
                bucket.append(row)
    return enriched, rows, examples


def input_identity(analysis):
    return hash_json({k: analysis[k] for k in ("analysis_id", "taxonomy_version", "config_hash", "selection_sha256",
                                             "code", "packages", "python_version", "phase_schema", "transition_schema")})


@exclusive_artifact("action", "analysis_id")
def classify_trajectories(cfg, analysis_id, rollout_ids=(), rollout_list=None, run_id=None):
    root = Path(cfg["project_root"])
    out = artifact_dir(cfg, "action", analysis_id)
    if out.exists():
        raise FileExistsError(f"Use a new analysis ID: {out}")
    implementation = Path(__file__).resolve().parents[2]
    before = {name: (implementation / name).read_bytes() for name in CODE_FILES}
    selection = select_trajectories(cfg, run_id, rollout_ids, rollout_list)
    snapshots = selection.pop("source_snapshots")
    sources, rows, examples = classify_sources(root, selection["sources"], analysis_id)
    selection["sources"] = sources
    summary = summarize(rows, sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".action-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for source in selection["source_runs"]:
            write_json(stage / source["snapshot_path"], snapshots[source["run_id"]])
            source["snapshot_sha256"] = sha256(stage / source["snapshot_path"])
        for name, content in before.items():
            path = stage / "code" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        analysis = {"schema_version": 2, "state": "complete", "analysis_id": analysis_id,
                    "taxonomy_version": VERSION, "created_at": now(), "config_hash": cfg["config_hash"],
                    "selection_sha256": hash_json(selection), "source_runs": selection["source_runs"],
                    "rollouts": sources, "selection_mode": selection["selection_mode"],
                    "requested_rollouts": selection["requested_rollouts"],
                    "duplicate_trajectory_aliases": selection["duplicate_trajectory_aliases"],
                    "code": {name: sha256(stage / "code" / name) for name in before},
                    "packages": {name: version(name) for name in ("tree-sitter", "tree-sitter-bash", "matplotlib", "numpy")},
                    "python_version": platform.python_version(), "phase_schema": PHASE_SCHEMA,
                    "transition_schema": TRANSITION_SCHEMA,
                    "plot_defaults": {"aggregation": "micro", "dpi": 200, "formats": ["png"]},
                    "unit": "one assistant message, one primary category; unsupported shapes are OTHER",
                    "scope_note": "Static requested behavior, not execution success or verified psychological intent"}
        analysis["input_identity"] = input_identity(analysis)
        write_json(stage / "analysis.json", analysis)
        write_json(stage / "selection.json", selection)
        write_json(stage / "resolved-config.json", cfg)
        write_json(stage / "summary.json", summary)
        write_json(stage / "transition.json", summary["transition"])
        write_json(stage / "examples.json", examples)
        with (stage / "labels.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        from action.evaluation.visualize import render_plots
        render_plots(stage / "plots", summary, analysis_id=analysis_id,
                     labels_sha256=sha256(stage / "labels.jsonl"), summary_sha256=sha256(stage / "summary.json"))
        if any((implementation / name).read_bytes() != content for name, content in before.items()):
            raise ValueError("Action implementation changed during analysis")
        # Inputs can change while figures are rendering; validate every requested source again.
        verify_frozen_sources(root, stage, analysis, selection)
        receipt = {"schema_version": 2, "state": "complete", "analysis_id": analysis_id,
                   "completed_at": now(), "taxonomy_version": VERSION, "input_identity": analysis["input_identity"],
                   "files": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
        write_json(stage / "receipt.json", receipt)
        stage.rename(out)
    return str(out)


def verify_frozen_sources(root, directory, analysis, selection):
    from application.execution.evidence import RunEvidence
    snapshots, proof = {}, {s["run_id"]: s for s in analysis["source_runs"]}
    if len(proof) != len(analysis["source_runs"]):
        raise ValueError("Duplicate source evidence")
    for run, source in proof.items():
        snapshot = read_json(owned_path(directory, source["snapshot_path"]))
        RunEvidence(root, run).validate(snapshot)
        if snapshot["revision"] != source["evidence_revision"]:
            raise ValueError("Source snapshot revision mismatch")
        snapshots[run] = snapshot
    requested = {(r["source_run_id"], r["rollout_id"]) for r in selection["requested_rollouts"]}
    frozen = {(run, rid) for run, data in snapshots.items() for rid in data["rollouts"]}
    if len(requested) != len(selection["requested_rollouts"]) or requested != frozen:
        raise ValueError("Frozen source coverage differs from requested rollouts")
    canonical = {source_key(s): s for s in selection["sources"]}
    aliases = selection["duplicate_trajectory_aliases"]
    alias_keys = {(a["alias"]["source_run_id"], a["alias"]["rollout_id"]) for a in aliases}
    if (len(canonical) != len(selection["sources"]) or len(alias_keys) != len(aliases)
            or set(canonical) & alias_keys or set(canonical) | alias_keys != requested):
        raise ValueError("Invalid canonical/alias trajectory coverage")
    for run, rid in sorted(requested):
        snapshot = snapshots[run]
        row = snapshot["rollouts"][rid]
        task = snapshot["tasks"][row["task_key"]]
        path = owned_path(root, row["trajectory_path"])
        if not path.is_relative_to((Path(root) / "runs" / run).resolve()):
            raise ValueError("Canonical trajectory escapes source run")
        if row.get("provenance", {}).get("quarantined") or task["partition"] == "excluded":
            raise ValueError("Frozen selection contains an excluded trajectory")
        read_actions(root, {"rollout_id": rid, **row})
        if (run, rid) in canonical:
            source = canonical[run, rid]
            expected = {"source_run_id": run, "run_id": run, "rollout_id": rid,
                        "trajectory_path": row["trajectory_path"], "trajectory_sha256": row["trajectory_sha256"],
                        "task_key": row["task_key"], "instance_id": task["instance_id"], "repo": task["repo"],
                        "partition": task["partition"], "model_fingerprint": row["model_fingerprint"],
                        "resolved": row["resolved"], "submission_status": row["submission_status"],
                        "termination_reason": row["termination_reason"], "classifier_cwd": proof[run]["classifier_cwd"]}
            if any(source.get(k) != value for k, value in expected.items()):
                raise ValueError("Action source binding disagrees with frozen evidence")
    for alias in aliases:
        target = canonical.get((alias["canonical"]["source_run_id"], alias["canonical"]["rollout_id"]))
        source = snapshots[alias["alias"]["source_run_id"]]["rollouts"][alias["alias"]["rollout_id"]]
        if target is None or any(source[k] != target[k] for k in ("task_key", "trajectory_sha256")):
            raise ValueError("Invalid duplicate trajectory alias")


def check_analysis(cfg, analysis_id):
    from action.artifacts import analysis_directory, load_local, verify_views
    out = analysis_directory(cfg["project_root"], analysis_id)
    receipt = read_json(out / "receipt.json")
    if receipt.get("schema_version") == 1:
        from action.legacy import check_analysis as check_legacy
        return {**check_legacy(cfg, analysis_id), "legacy": True}
    out, analysis, selection, rows, summary, receipt = load_local(cfg["project_root"], analysis_id)
    if input_identity(analysis) != analysis["input_identity"]:
        raise ValueError("Action input identity changed")
    verify_frozen_sources(cfg["project_root"], out, analysis, selection)
    enriched, expected, examples = classify_sources(cfg["project_root"], selection["sources"], out.name)
    if enriched != selection["sources"] or expected != rows:
        raise ValueError("Action labels or step coverage disagree with canonical inputs")
    if read_json(out / "examples.json") != examples:
        raise ValueError("Action examples disagree with labels")
    views = verify_views(cfg["project_root"], out, analysis, receipt)
    implementation = Path(__file__).resolve().parents[2]
    matches = all((implementation / name).is_file() and sha256(implementation / name) == digest
                  for name, digest in analysis["code"].items())
    return {"analysis_id": out.name, "verified": True, "legacy": False,
            "sources": len(selection["sources"]), "steps": len(rows), "source_run_ids": [s["run_id"] for s in analysis["source_runs"]],
            "taxonomy_version": analysis["taxonomy_version"], "plot_views": views,
            "current_implementation_matches_snapshot": matches}
