"""Checksummed, independently disposable per-site memmaps; no estimator imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.io import hash_json, read_json, sha256, write_json
from hidden.artifacts import iter_arrays_grouped
from steering.artifacts import iter_jsonl, seal, verify_local
from steering.datasets.features import FEATURE_SCHEMA, select_sites, vectors
from steering.datasets.paths import candidates, locate, publication, require_space


def specification(dataset_directory, sites=None):
    dataset_directory = Path(dataset_directory)
    manifest = read_json(dataset_directory / "dataset.json")
    selected = select_sites(manifest["sites"], sites)
    if not selected:
        raise ValueError("Dataset has no available hidden sites")
    spec = {
        "schema": FEATURE_SCHEMA,
        "dataset_receipt_sha256": sha256(dataset_directory / "receipt.json"),
        "sample_order_sha256": manifest["sample_order_sha256"],
        "sample_count": manifest["sample_count"],
        "sites": selected,
        "dtype": "float32",
        "source_space": manifest["space"],
    }
    return hash_json(spec), spec


def verify_cache(directory, expected=None, *, check_data=True):
    directory = Path(directory)
    receipt = verify_local(directory, "steering-cache", directory.name)
    manifest = read_json(directory / "cache.json")
    if receipt["identity"] != directory.name or hash_json(manifest["specification"]) != directory.name:
        raise ValueError("Feature cache identity mismatch")
    if expected is not None and manifest["specification"] != expected:
        raise ValueError("Feature cache dataset/site/order identity mismatch")
    n = manifest["specification"]["sample_count"]
    names = {f"site-{site['id']}.f32.npy" for site in manifest["specification"]["sites"]}
    if set(manifest["files"]) != names:
        raise ValueError("Feature cache is incomplete")
    for site in manifest["specification"]["sites"]:
        name = f"site-{site['id']}.f32.npy"
        if manifest["files"][name] != receipt["files"].get(name):
            raise ValueError("Cache manifest/receipt checksums disagree")
        matrix = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        try:
            if matrix.shape != (n, site["width"]) or matrix.dtype != np.float32:
                raise ValueError("Feature cache shape/dtype mismatch")
            if check_data:
                for first in range(0, n, 1024):
                    if not np.isfinite(matrix[first : first + 1024]).all():
                        raise ValueError("Feature cache contains non-finite values")
        finally:
            matrix._mmap.close()
    return manifest


def existing_cache(cfg, identity, spec, cache_root=None):
    matches = candidates(cfg, "steering_cache", identity, cache_root)
    receipts = set()
    for match in matches:
        verify_cache(match, spec)
        receipts.add(sha256(match / "receipt.json"))
    if len(receipts) > 1:
        raise ValueError("Conflicting feature cache copies across disks")
    return matches[0] if matches else None


def cache_features(
    cfg, dataset_id, *, dataset_root=None, cache_root=None, sites=None, disk_reserve_bytes=2 * 1024**3
):
    from steering.datasets.build import check_dataset

    dataset_directory = locate(cfg, "steering_dataset", dataset_id, dataset_root)
    check_dataset(dataset_directory)
    identity, spec = specification(dataset_directory, sites)
    existing = existing_cache(cfg, identity, spec, cache_root)
    if existing is not None:
        return {
            "cache_id": identity,
            "path": str(existing),
            "sample_count": spec["sample_count"],
            "sites": [site["id"] for site in spec["sites"]],
            "reused": True,
        }
    samples = list(iter_jsonl(dataset_directory / "samples.jsonl"))
    if not samples:
        raise ValueError("Dataset has no included samples")
    n = len(samples)
    if hash_json([sample["uid"] for sample in samples]) != spec["sample_order_sha256"]:
        raise ValueError("Dataset/cache sample order mismatch")
    required_bytes = sum(n * site["width"] * 4 for site in spec["sites"])
    maps = {}
    with publication(cfg, "steering_cache", identity, cache_root) as (stage, destination):
        require_space(stage, required_bytes, disk_reserve_bytes)
        try:
            maps = {
                site["id"]: np.lib.format.open_memmap(
                    stage / f"site-{site['id']}.f32.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(n, site["width"]),
                )
                for site in spec["sites"]
            }
            seen = np.zeros(n, dtype=bool)
            inputs = [(i, sample["tensor_root"], sample) for i, sample in enumerate(samples)]
            for count, (i, arrays) in enumerate(iter_arrays_grouped(inputs), 1):
                if seen[i]:
                    raise ValueError("A tensor row was returned more than once")
                for site_id, vector in vectors(arrays, spec["sites"]).items():
                    maps[site_id][i] = vector
                seen[i] = True
                if count % 1000 == 0 or count == n:
                    print(f"cache: {count}/{n} verified hidden rows", flush=True)
            if not seen.all():
                raise ValueError("Feature cache has missing rows")
            for matrix in maps.values():
                matrix.flush()
                matrix._mmap.close()
            maps.clear()
            manifest = {
                "schema_version": 1,
                "cache_id": identity,
                "dataset_id": dataset_id,
                "rebuildable": True,
                "specification": spec,
                "required_bytes": required_bytes,
                "files": {path.name: sha256(path) for path in sorted(stage.glob("*.npy"))},
            }
            write_json(stage / "cache.json", manifest)
            seal(stage, "steering-cache", identity, identity)
        finally:
            for matrix in maps.values():
                matrix._mmap.close()
    return {
        "cache_id": identity,
        "path": str(destination),
        "sample_count": n,
        "sites": [site["id"] for site in spec["sites"]],
        "reused": False,
    }
