"""Freeze an unrolled Verified cohort, then run and finalize using project APIs."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path

from baseline.catalog import Catalog
from core.config import resolve
from core.io import read_json, sha256
from runtime.images.nas import NasImageStore


def publish_static(root, source, count):
    """Publish/import static NAS task assets only; rollout uses core.cli."""
    cfg = resolve(root / "config.yaml")
    catalog = Catalog(root)
    existing = catalog.read()
    seen = {existing["tasks"][r["task_key"]]["instance_id"] for r in existing["rollouts"].values()}
    rows = sorted(read_json(source), key=lambda row: row["instance_id"])
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source task IDs")
    if len(rows) != count or seen.intersection(row["instance_id"] for row in rows):
        raise ValueError("Requested cohort must contain exactly the expected number of unrolled tasks")
    # Reuse the already verified static OCI bytes, with independent NAS names.
    # Hard links on the same NAS do not duplicate large blobs; descriptors are
    # copied and validated under this project's namespace.
    stores = {}
    normalized = []
    for original in rows:
        row = copy.deepcopy(original)
        reference = row["nas_image"]
        source_store = Path(reference["store"])
        if not source_store.is_absolute():
            source_store = Path("/VM") / source_store
        destination = Path(cfg["paths"]["nas_root"]) / source_store.name
        if source_store.resolve() != destination.resolve() and str(source_store) not in stores:

            def link_or_copy(src, dst):
                try:
                    os.link(src, dst)
                except FileExistsError:
                    if sha256(src) != sha256(dst):
                        raise ValueError(f"Conflicting NAS asset: {dst}") from None
                except OSError:
                    shutil.copy2(src, dst)
                return dst

            for directory in ("blobs", "images"):
                shutil.copytree(
                    source_store / directory,
                    destination / directory,
                    copy_function=link_or_copy,
                    dirs_exist_ok=True,
                )
            stores[str(source_store)] = str(destination)
        reference["store"] = destination.name
        store = NasImageStore(cfg, reference)
        store.verify(deep=False)  # actual docker load verifies blob SHA-256 again
        reference["image_id"] = store.image_id
        env = row.get("environment_overrides") or {}
        if env.get("wheelhouse"):
            env["wheelhouse"] = str(root / "runtime/cache/nas200-wheels")
        row["environment_overrides"] = env
        normalized.append(row)
    tasks = [
        {
            "benchmark": "SWE-bench_Verified",
            "revision": "snapshot-sha256-" + sha256(source),
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "partition": "train",
            "instance": row,
        }
        for row in normalized
    ]
    keys = catalog.register_tasks(tasks)
    return {"registered_task_keys": keys}


def main():
    parser = argparse.ArgumentParser(
        description="Static task/NAS publication only. Use core.cli rollout for execution."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    print(publish_static(args.root, args.source, args.count))


if __name__ == "__main__":
    main()
