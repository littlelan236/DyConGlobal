"""Validate and walk complete OCI/Docker image descriptor graphs."""

from __future__ import annotations

import json
from pathlib import Path

from core.io import sha256

INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}


def blob_path(root: Path, digest: str) -> Path:
    if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
        raise ValueError("Only SHA256 OCI descriptors are supported")
    int(digest[7:], 16)
    namespace = root.resolve()
    path = (namespace / digest[7:]).resolve()
    if not path.is_relative_to(namespace):
        raise ValueError("OCI blob escapes store")
    return path


def descriptor_graph(root: Path, descriptor: dict, *, deep=False, verified=None) -> list[dict]:
    """Include indexes, all platform/attestation manifests, configs and layers."""
    seen = {}
    active = set()
    verified = set() if verified is None else verified

    def visit(item, document=False):
        digest, size = item["digest"], item["size"]
        if not isinstance(size, int) or size < 0:
            raise ValueError("Invalid OCI descriptor size")
        if digest in active:
            raise ValueError("Cyclic OCI descriptor graph")
        if digest in seen:
            if seen[digest]["size"] != size:
                raise ValueError("Conflicting OCI descriptor sizes")
            return
        path = blob_path(root, digest)
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(path)
        if deep and digest not in verified:
            if "sha256:" + sha256(path) != digest:
                raise ValueError(f"Corrupt OCI blob: {path}")
            verified.add(digest)
        seen[digest] = item
        if not document:
            return
        active.add(digest)
        obj = json.loads(path.read_bytes())
        media_type = obj.get("mediaType") or item.get("mediaType")
        if item.get("mediaType") and media_type != item["mediaType"]:
            raise ValueError("OCI descriptor media type mismatch")
        if obj.get("schemaVersion") != 2:
            raise ValueError("Only OCI/Docker schema 2 is supported")
        if media_type in INDEX_TYPES:
            if not obj.get("manifests"):
                raise ValueError("Empty OCI index")
            for child in obj["manifests"]:
                visit(child, document=True)
        elif media_type in MANIFEST_TYPES:
            visit(obj["config"])
            for layer in obj["layers"]:
                visit(layer)
        else:
            raise ValueError(f"Unsupported image media type: {media_type}")
        active.remove(digest)

    visit(descriptor, document=True)
    return list(seen.values())


def platform_manifest(root: Path, descriptor: dict, platform="linux/amd64") -> tuple[dict, dict]:
    """Select runtime config/layers while keeping the complete original root graph."""
    operating_system, architecture = platform.split("/")[:2]
    current = descriptor
    for _ in range(16):
        obj = json.loads(blob_path(root, current["digest"]).read_bytes())
        if obj.get("mediaType") in MANIFEST_TYPES:
            config = json.loads(blob_path(root, obj["config"]["digest"]).read_bytes())
            if config.get("os") != operating_system or config.get("architecture") != architecture:
                raise ValueError(f"Image does not support {platform}")
            return current, obj
        candidates = [
            d
            for d in obj.get("manifests", [])
            if d.get("platform", {}).get("os") == operating_system
            and d.get("platform", {}).get("architecture") == architecture
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one manifest for {platform}")
        current = candidates[0]
    raise ValueError("OCI index nesting is too deep")


def verify_blobs(root: Path, descriptors, workers=4, progress=None):
    """Hash unique descriptors with bounded NAS read concurrency."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique = {}
    for item in descriptors:
        digest = item["digest"]
        if digest in unique and unique[digest]["size"] != item["size"]:
            raise ValueError("Conflicting OCI descriptor sizes")
        unique[digest] = item

    def check(item):
        path = blob_path(root, item["digest"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise FileNotFoundError(path)
        if "sha256:" + sha256(path) != item["digest"]:
            raise ValueError(f"Corrupt OCI blob: {path}")
        return item["digest"]

    verified = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(check, item) for item in unique.values()]):
            verified.add(future.result())
            if progress:
                progress(len(verified), len(unique))
    return verified
