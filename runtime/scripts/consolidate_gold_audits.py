"""Consolidate repeated gold audits without rewriting the original run reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.io import now, read_json, sha256, write_json


def consolidate(paths: list[Path], output: Path):
    reports = [read_json(p) for p in paths]
    rows = {r["instance_id"]: r for r in reports[0]["results"]}
    provenance = {iid: str(paths[0]) for iid in rows}
    for path, report in zip(paths[1:], reports[1:]):
        for result in report["results"]:
            iid = result["instance_id"]
            previous = rows[iid]
            for key in ["image_id", "patch_sha256"]:
                if previous[key] != result[key]:
                    raise ValueError(f"Incompatible {key} for {iid}: {path}")
            rows[iid] = result
            provenance[iid] = str(path)
    failures = [
        iid
        for iid, r in rows.items()
        if r.get("resolved") is not True or not r.get("cold_lifecycle_verified")
    ]
    summary = {
        "tasks": len(rows),
        "resolved": sum(r.get("resolved") is True for r in rows.values()),
        "cold_lifecycle_verified": sum(r.get("cold_lifecycle_verified") is True for r in rows.values()),
        "passed": not failures,
        "failed": sorted(failures),
        "finished_at": now(),
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in paths],
        "result_sources": provenance,
        "results": sorted(rows.values(), key=lambda r: r["instance_id"]),
    }
    write_json(output, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = consolidate(args.summary, args.output)
    print({k: v for k, v in summary.items() if k not in ["sources", "result_sources", "results"]})
