"""Derive NAS OCI images with rootless-compatible tar ownership.

Source blobs remain immutable. Only unmappable UID/GID metadata changes;
file contents, modes, paths, links and test inputs are preserved.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from core.io import read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.oci import blob_path, descriptor_graph, platform_manifest


def publish_blob(path: Path, namespace: Path) -> dict:
    digest = "sha256:" + sha256(path)
    target = blob_path(namespace, digest)
    if target.exists():
        if sha256(target) != digest[7:]:
            raise ValueError("Existing derived OCI blob is corrupt")
        path.unlink()
    else:
        os.replace(path, target)
    return {"digest": digest, "size": target.stat().st_size}


def json_blob(value: dict, namespace: Path) -> dict:
    fd, filename = tempfile.mkstemp(prefix=".json-", dir=namespace)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())
    return publish_blob(Path(filename), namespace)


class DigestWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.stream.write(data)


def normalize_layer(source: Path, namespace: Path, cache: Path, max_id: int = 65535) -> dict:
    """Scan once per source layer; cache verified, content-addressed results."""
    digest = source.name
    receipt = cache / (digest + ".json")
    with file_lock(cache / (digest + ".lock")):
        if receipt.exists():
            result = read_json(receipt)
            target = blob_path(namespace, result["descriptor"]["digest"])
            if target.is_file() and target.stat().st_size == result["descriptor"]["size"]:
                return result
        if sha256(source) != digest:
            raise ValueError("Source layer checksum mismatch")
        changed = []
        count = 0
        with tarfile.open(source, "r|*") as tar:
            for item in tar:
                if item.uid < 0 or item.gid < 0:
                    raise ValueError("Negative OCI ownership is unsupported")
                if item.uid > max_id or item.gid > max_id:
                    count += 1
                    if len(changed) < 20:
                        changed.append({"path": item.name, "uid": item.uid, "gid": item.gid})
        result = {"source_digest": "sha256:" + digest, "max_id": max_id, "changed_entries": count,
                  "examples": changed, "diff_id": None}
        if not count:
            target = namespace / digest
            if not target.exists():
                os.link(source, target)
            result["descriptor"] = {"digest": "sha256:" + digest, "size": source.stat().st_size}
        else:
            fd, filename = tempfile.mkstemp(prefix=".layer-", dir=namespace)
            try:
                with os.fdopen(fd, "wb") as raw, gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0
                ) as compressed:
                    hashed = DigestWriter(compressed)
                    with tarfile.open(source, "r|*") as original, tarfile.open(
                        fileobj=hashed, mode="w|", format=tarfile.PAX_FORMAT
                    ) as repaired:
                        for item in original:
                            updated = copy.deepcopy(item)
                            for field, name in [("uid", "uname"), ("gid", "gname")]:
                                if getattr(updated, field) > max_id:
                                    setattr(updated, field, 0)
                                    setattr(updated, name, "root")
                                    updated.pax_headers.pop(field, None)
                                    updated.pax_headers.pop(name, None)
                            repaired.addfile(updated, original.extractfile(item) if item.isfile() else None)
                    result["diff_id"] = "sha256:" + hashed.digest.hexdigest()
                result["descriptor"] = publish_blob(Path(filename), namespace)
            finally:
                Path(filename).unlink(missing_ok=True)
        write_json(receipt, result)
        return result


def derive_image(source_store, destination: Path, instance_id: str, max_id: int = 65535) -> dict:
    """Create a derived OCI manifest plus exact source/metadata-change evidence."""
    namespace = destination / "blobs/sha256"
    namespace.mkdir(parents=True, exist_ok=True)
    cache = destination / "ownership-layers" / str(max_id)
    cache.mkdir(parents=True, exist_ok=True)
    source_root = {"digest": source_store.image_id, "size": source_store.record["manifest_size"],
                   "mediaType": source_store.record["root_media_type"]}
    _, manifest = platform_manifest(source_store.root / "blobs/sha256", source_root)
    config_path = source_store.blob(manifest["config"]["digest"])
    if "sha256:" + sha256(config_path) != manifest["config"]["digest"]:
        raise ValueError("Source config checksum mismatch")
    config = read_json(config_path)
    repairs = []
    layers = []
    for index, layer in enumerate(manifest["layers"]):
        if layer["mediaType"] not in {
            "application/vnd.docker.image.rootfs.diff.tar.gzip",
            "application/vnd.oci.image.layer.v1.tar+gzip",
            "application/vnd.oci.image.layer.v1.tar",
        }:
            raise ValueError("Unsupported layer media type")
        result = normalize_layer(source_store.blob(layer["digest"]), namespace, cache, max_id)
        descriptor = {**layer, **result["descriptor"], "mediaType": (
            "application/vnd.oci.image.layer.v1.tar" if layer["mediaType"].endswith(".tar")
            else "application/vnd.oci.image.layer.v1.tar+gzip"
        )}
        if result["changed_entries"]:
            descriptor["mediaType"] = "application/vnd.oci.image.layer.v1.tar+gzip"
            config["rootfs"]["diff_ids"][index] = result["diff_id"]
            repairs.append(result)
        layers.append(descriptor)
    if not repairs:
        return copy.deepcopy(source_store.reference)
    provenance = {"kind": "rootless-ownership-v1", "source": source_store.reference,
                  "max_id": max_id, "layers": repairs}
    config.setdefault("config", {}).setdefault("Labels", {})["dycon.image-repair"] = "rootless-ownership-v1"
    config_descriptor = {**manifest["config"], **json_blob(config, namespace),
                         "mediaType": "application/vnd.oci.image.config.v1+json"}
    derived = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
               "config": config_descriptor, "layers": layers}
    root = {**json_blob(derived, namespace), "mediaType": derived["mediaType"]}
    descriptors = descriptor_graph(namespace, root)
    record = {"schema_version": 2, "requested": source_store.record.get("requested"),
              "manifest_digest": root["digest"], "manifest_size": root["size"],
              "root_media_type": root["mediaType"], "runtime_manifest_digest": root["digest"],
              "config": config_descriptor, "layers": layers, "platform": "linux/amd64",
              "compressed_bytes": sum(item["size"] for item in descriptors),
              "repo_digests": [], "descriptor_count": len(descriptors), "repair": provenance}
    (destination / "images").mkdir(parents=True, exist_ok=True)
    path = destination / "images" / (instance_id + ".json")
    if path.exists() and read_json(path) != record:
        raise ValueError("Derived image already exists with different provenance")
    write_json(path, record)
    return {"store": destination.name, "manifest": str(path.relative_to(destination)),
            "manifest_sha256": sha256(path), "image_id": root["digest"]}


def validate_repair(cfg: dict, source_reference: dict, derived_reference: dict) -> dict:
    """Bind a derived graph to its original config, layer order and repair receipts."""
    from runtime.images.nas import NasImageStore

    source = NasImageStore(cfg, source_reference)
    derived = NasImageStore(cfg, derived_reference)
    source.verify()
    derived.verify()
    proof = derived.record.get("repair", {})
    if proof.get("kind") != "rootless-ownership-v1" or proof.get("source") != source_reference:
        raise ValueError("Missing or incompatible ownership repair provenance")
    def manifest(store):
        return platform_manifest(store.root / "blobs/sha256", {
            "digest": store.image_id, "size": store.record["manifest_size"],
            "mediaType": store.record["root_media_type"],
        })[1]
    original, repaired = manifest(source), manifest(derived)
    old_config = read_json(source.blob(original["config"]["digest"]))
    new_config = read_json(derived.blob(repaired["config"]["digest"]))
    for store, item in [(source, original), (derived, repaired)]:
        if "sha256:" + sha256(store.blob(item["config"]["digest"])) != item["config"]["digest"]:
            raise ValueError("Config digest changed")
    if len(original["layers"]) != len(repaired["layers"]):
        raise ValueError("Ownership repair changed layer count")
    by_source = {row["source_digest"]: row for row in proof["layers"]}
    expected_config = copy.deepcopy(old_config)
    expected_config.setdefault("config", {}).setdefault("Labels", {})[
        "dycon.image-repair"
    ] = "rootless-ownership-v1"
    for index, (old, new) in enumerate(zip(original["layers"], repaired["layers"])):
        change = by_source.get(old["digest"])
        expected = change["descriptor"] if change else old
        if any(new[key] != expected[key] for key in ["digest", "size"]):
            raise ValueError("Derived layer disagrees with ownership repair receipt")
        if change:
            expected_config["rootfs"]["diff_ids"][index] = change["diff_id"]
    if expected_config != new_config:
        raise ValueError("Ownership repair changed unrelated image configuration")
    return {"kind": proof["kind"], "source_image_id": source.image_id,
            "source_reference": source_reference, "derived_reference": derived_reference}
