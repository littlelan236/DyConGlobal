"""Import explicit old harness evidence only when the evaluated patch matches."""

import shutil
from pathlib import Path

from baseline.catalog import Catalog
from core.io import identifier, read_json, sha256, write_json
from core.locking import file_lock


def import_evaluation(root, spec):
    root = Path(root).resolve()
    eval_id = identifier(spec["evaluation_id"])
    source = Path(spec["report"]).resolve()
    if sha256(source) != spec["report_sha256"]:
        raise ValueError("Source report checksum mismatch")
    original = read_json(source)
    catalog = Catalog(root)
    current = catalog.read()
    outcomes = {}
    for rid, binding in spec["bindings"].items():
        rollout = current["rollouts"][rid]
        if sha256(binding["evaluated_patch"]) != rollout["patch_sha256"]:
            raise ValueError("Old evaluation graded a different patch")
        result = original
        for key in binding["outcome_path"]:
            result = result[key]
        completed = result.get("completed", type(result.get("resolved")) is bool)
        if completed and type(result.get("resolved")) is not bool:
            raise ValueError("Completed source evaluation has no boolean success outcome")
        outcomes[rid] = {
            "status": "completed" if completed else "infra_error",
            "resolved": result["resolved"] if completed else None,
            "patch_sha256": rollout["patch_sha256"],
        }
    if not outcomes:
        raise ValueError("No explicit rollout bindings in evaluation import")
    with file_lock(root / f"state/evaluation-import-{eval_id}.lock"):
        out = root / "baseline/evaluations" / eval_id
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        shutil.copyfile(source, out / "source-report.json")
        if sha256(out / "source-report.json") != spec["report_sha256"]:
            raise ValueError("Evaluation source changed while importing")
        report = {
            "outcomes": outcomes,
            "source_report_sha256": spec["report_sha256"],
            "source_report": str(source),
        }
        write_json(out / "summary.json", report)
        catalog.record_evaluation(
            eval_id,
            out / "summary.json",
            outcomes,
            {"imported": True, "source_report_sha256": spec["report_sha256"]},
        )
    return eval_id
