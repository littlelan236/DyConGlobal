"""Produce rootless-compatible NAS images, then publish only gold-validated repairs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from catalog.store import Catalog
from core.config import resolve
from core.io import identifier, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.nas import NasImageStore
from runtime.images.ownership import derive_image, validate_repair


def publish(cfg, instances, summary_path):
    from runtime.scripts.gold_image_audit import sha256_text

    summary = read_json(summary_path)
    passed = {r["instance_id"]: r for r in summary["results"]
              if r.get("resolved") is True and r.get("cold_lifecycle_verified") is True}
    candidates = {r["instance_id"]: r for r in instances if r["instance_id"] in passed}
    updated = []
    with Catalog(cfg["project_root"]).transaction() as data:
        for task in data["tasks"].values():
            iid = task["instance_id"]
            if iid not in candidates:
                continue
            row = candidates[iid]
            if "source_nas_image" not in row:
                continue
            source = task["instance"]
            if {k: v for k, v in row.items() if k not in {"nas_image", "source_nas_image"}} != {
                k: v for k, v in source.items() if k not in {"nas_image", "source_nas_image"}
            }:
                raise ValueError("Image repair cannot change task/test definitions")
            if row["source_nas_image"] != source.get("source_nas_image", source["nas_image"]):
                raise ValueError("Image repair source disagrees with catalog")
            store = NasImageStore(cfg, row["nas_image"])
            # The cold-gold audit itself deep-hashes every imported OCI graph.
            store.verify(deep=False)
            if store.record["repair"]["source"] != row["source_nas_image"]:
                raise ValueError("Image repair source disagrees with NAS evidence")
            validate_repair(cfg, row["source_nas_image"], row["nas_image"])
            if passed[iid]["image_id"] != store.image_id or passed[iid]["patch_sha256"] != sha256_text(row["patch"]):
                raise ValueError("Gold evidence does not bind repaired image and original patch")
            task["instance"] = copy.deepcopy(row)
            updated.append(iid)
    receipt = {"tasks": updated, "summary": str(summary_path), "summary_sha256": sha256(summary_path),
               "catalog_revision": Catalog(cfg["project_root"]).read()["catalog_revision"]}
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--only-file", type=Path)
    p.add_argument("--max-id", type=int, default=65535)
    p.add_argument("--publish-summary", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    cfg = resolve(root / "config.yaml")
    destination = Path(cfg["paths"]["nas_root"]) / identifier(a.store)
    rows = read_json(a.instances)
    selected = set(read_json(a.only_file)) if a.only_file else {r["instance_id"] for r in rows}
    if not selected <= {r["instance_id"] for r in rows}:
        raise ValueError("Unknown image repair tasks")
    with file_lock(root / "state" / (identifier(a.store) + ".lock")):
        if a.publish_summary:
            receipt = publish(cfg, rows, a.publish_summary)
            write_json(destination / "publication.json", receipt)
            print(receipt, flush=True)
            return
        output = []
        for row in rows:
            if row["instance_id"] in selected:
                source = NasImageStore(cfg, row["nas_image"])
                repaired = derive_image(source, destination, row["instance_id"], a.max_id)
                if repaired != row["nas_image"]:
                    row = {**row, "source_nas_image": row["nas_image"], "nas_image": repaired}
                print("DERIVED " + row["instance_id"], flush=True)
            output.append(row)
            write_json(destination / "dataset/instances.json", output)
        write_json(destination / "dataset/task-ids.json", sorted(selected))
        write_json(destination / "dataset/complete.json", {"tasks": len(output), "repaired": len(selected)})


if __name__ == "__main__":
    main()
