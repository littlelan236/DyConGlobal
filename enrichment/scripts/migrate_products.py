"""Migrate completed legacy enrichment/hidden products by trajectory hash."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from application.execution.evidence import RunEvidence
from core.io import now, read_json, sha256, write_json


def migrate(root, source, migration_id, source_run):
    root, source = Path(root).resolve(), Path(source).resolve()
    out_e = root / f"enrichment/outputs/{migration_id}"
    out_h = root / f"hidden/outputs/{migration_id}"
    em = root / "enrichment/manifest.json"
    hm = root / "hidden/manifest.json"
    if em.exists() or hm.exists() or out_e.exists() or out_h.exists():
        raise FileExistsError("manifest/output already exists")
    catalog = RunEvidence(root, source_run)
    catalog.verify()
    data = catalog.read()
    current = {r["trajectory_sha256"]: (rid, r) for rid, r in data["rollouts"].items()}
    old_h = read_json(source / "baseline/hidden/manifest.json")
    completed = read_json(source / "baseline/enrichment/completed.json")
    tasks = []
    skipped = []
    for h in old_h["tasks"]:
        match = current.get(h["source_sha256"])
        if not match:
            skipped.append(
                {
                    "instance_id": h["instance_id"],
                    "source_sha256": h["source_sha256"],
                    "reason": "trajectory_not_in_current_catalog",
                }
            )
            continue
        rid, row = match
        iid = h["instance_id"]
        ann = source / "baseline/annotation" / f"{iid}.jsonl"
        npz = source / "baseline/hidden" / f"{iid}.npz"
        dec = source / "baseline/hidden" / f"{iid}.decisions.json"
        ext = source / "baseline/hidden" / f"{iid}.extraction.json"
        for f in [ann, npz, dec, ext]:
            if not f.exists():
                raise FileNotFoundError(f)
        tasks.append(
            {
                "instance_id": iid,
                "rollout_id": rid,
                "task_key": row["task_key"],
                "trajectory_path": row["trajectory_path"],
                "trajectory_sha256": h["source_sha256"],
                "source": {"old_annotation": str(ann), "old_hidden": str(npz)},
                "steps": h["steps"],
                "annotation_steps": h["annotation_steps"],
                "unavailable": h.get("unavailable", []),
                "enrichment": {},
                "hidden": {},
            }
        )
    out_e.mkdir(parents=True)
    out_h.mkdir(parents=True)
    for t in tasks:
        iid = t["instance_id"]
        pairs = [
            (source / "baseline/annotation" / f"{iid}.jsonl", out_e / f"{iid}.jsonl", "enrichment"),
            (source / "baseline/hidden" / f"{iid}.npz", out_h / f"{iid}.npz", "hidden"),
            (source / "baseline/hidden" / f"{iid}.decisions.json", out_h / f"{iid}.decisions.json", "hidden"),
            (
                source / "baseline/hidden" / f"{iid}.extraction.json",
                out_h / f"{iid}.extraction.json",
                "hidden",
            ),
        ]
        for src, dst, kind in pairs:
            shutil.copyfile(src, dst)
            item = {
                "path": str(dst.relative_to(root)),
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size,
                "anchor": "think_close",
                "anchor_text": "</think>",
                "trajectory_path": t["trajectory_path"],
                "trajectory_sha256": t["trajectory_sha256"],
            }
            t[kind][dst.name] = item
        # annotation rows are the enrichment state product; enforce source/task identity and record counts.
        lines = [json.loads(x) for x in (out_e / f"{iid}.jsonl").read_text().splitlines() if x.strip()]
        if any(
            x.get("instance_id") != iid or x.get("source_sha256") != t["trajectory_sha256"] for x in lines
        ):
            raise ValueError(f"annotation identity mismatch {iid}")
        t["enrichment"]["row_count"] = len(lines)
        t["enrichment"]["anchor"] = "think_close"
        t["enrichment"]["anchor_text"] = "</think>"
        meta = read_json(out_h / f"{iid}.extraction.json")
        if meta.get("token_boundary", {}).get("text") not in (None, "</think>"):
            raise ValueError(f"hidden anchor mismatch {iid}")
        t["hidden"]["anchor"] = "think_close"
        t["hidden"]["anchor_text"] = "</think>"
    base = {
        "schema_version": 1,
        "manifest_id": migration_id,
        "created_at": now(),
        "source_project": str(source),
        "scope": "completed legacy enrichment and hidden products whose trajectory hash exists in current catalog",
        "anchor": {
            "name": "think_close",
            "text": "</think>",
            "semantics": "latent prefix closes at the current reasoning close marker; no future action/observation is included",
        },
        "trajectory_binding": "exact SHA-256 of current baseline manifest rollout",
        "completed_summary": completed,
        "skipped": skipped,
        "task_count": len(tasks),
    }
    e = {
        **base,
        "product": "enrichment",
        "artifact_root": str(out_e.relative_to(root)),
        "tasks": tasks,
        "files": {
            k: v for t in tasks for k, v in t["enrichment"].items() if isinstance(v, dict) and "sha256" in v
        },
    }
    h = {
        **base,
        "product": "hidden",
        "artifact_root": str(out_h.relative_to(root)),
        "model": old_h["model"],
        "template_kwargs": old_h["template_kwargs"],
        "token_boundary": {
            "text": "</think>",
            "token_ids": old_h["token_boundary"].get("token_ids"),
            "site": "close_token_as_input",
        },
        "tasks": tasks,
        "files": {
            k: v for t in tasks for k, v in t["hidden"].items() if isinstance(v, dict) and "sha256" in v
        },
    }
    write_json(em, e)
    write_json(hm, h)
    receipt = {
        "migration_id": migration_id,
        "completed_at": now(),
        "tasks": len(tasks),
        "skipped": len(skipped),
        "enrichment_bytes": sum(
            x["bytes"]
            for t in tasks
            for x in t["enrichment"].values()
            if isinstance(x, dict) and "bytes" in x
        ),
        "hidden_bytes": sum(
            x["bytes"] for t in tasks for x in t["hidden"].values() if isinstance(x, dict) and "bytes" in x
        ),
        "enrichment_manifest_sha256": sha256(em),
        "hidden_manifest_sha256": sha256(hm),
        "verified_catalog": catalog.verify(),
    }
    write_json(root / f"enrichment/outputs/{migration_id}/receipt.json", receipt)
    write_json(root / f"hidden/outputs/{migration_id}/receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--migration-id", required=True)
    p.add_argument("--source-run", required=True)
    a = p.parse_args()
    migrate(a.root, a.source, a.migration_id, a.source_run)


if __name__ == "__main__":
    main()
