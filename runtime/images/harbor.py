"""Harbor task images use the existing registry/OCI/NAS publication boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from catalog.harbor import verify_package
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.nas import NasImageStore
from runtime.images.oci import blob_path, descriptor_graph, platform_manifest
from runtime.images.registry import image_metadata, transfer_blob


def prepare(cfg, snapshot):
    snapshot = Path(snapshot).resolve()
    frozen = read_json(snapshot)
    store = Path(cfg["paths"]["nas_root"]) / identifier(cfg["harbor"]["image_store"])
    identity = {"snapshot_sha256": sha256(snapshot), "platform": cfg["environment"]["platform"]}
    seeds = [Path(cfg["paths"]["nas_root"]) / identifier(name) / "blobs/sha256"
             for name in cfg["harbor"].get("seed_stores", [])]
    images = sorted({image for task in frozen["tasks"].values() for image in task["instance"]["images"].values()})
    for task in frozen["tasks"].values():
        verify_package(task)
    references = {}
    with file_lock(store / ".prepare.lock", nonblocking=True):
        selection = {**identity, "images": images}
        selection_path = store / "selection.json"
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ValueError("Harbor image store belongs to another snapshot/platform")
        write_json(selection_path, selection)
        blobs = store / "blobs/sha256"
        blobs.mkdir(parents=True, exist_ok=True)
        for image in images:
            key = hash_json(image)
            lock_path = store / "source-locks" / (key + ".json")
            if lock_path.exists():
                lock = read_json(lock_path)
                if lock["selection"] != identity or lock["requested"] != image:
                    raise ValueError("Harbor image source lock changed")
            else:
                registry = "docker.io"
                name = image.removeprefix("docker.io/")
                if name.split("/")[0] in {"ghcr.io", "quay.io"}:
                    registry, name = name.split("/", 1)
                if "@" in name:
                    repository, tag = name.split("@", 1)
                else:
                    repository, tag = name.rsplit(":", 1)
                if "/" not in repository:
                    repository = "library/" + repository
                root, descriptors = image_metadata(repository, blobs, registry=registry, reference=tag,
                    proxy=cfg["harbor"].get("host_proxy"), platform=identity["platform"])
                runtime, manifest = platform_manifest(blobs, root, identity["platform"])
                by_digest = {r["digest"]: r for r in descriptors}
                required = [by_digest[r["digest"]] for r in (runtime, manifest["config"], *manifest["layers"])]
                lock = {"selection": identity, "requested": image, "registry": registry, "repository": repository,
                        "root": root, "runtime": runtime, "descriptors": required, "resolved_at": now()}
                write_json(lock_path, lock)
            with ThreadPoolExecutor(max_workers=min(4, cfg["images"]["load_workers"])) as pool:
                list(pool.map(lambda row: transfer_blob(row, blobs, seed_directories=seeds,
                                                       proxy=cfg["harbor"].get("host_proxy")),
                              lock["descriptors"]))
            graph = descriptor_graph(blobs, lock["runtime"], deep=True)
            manifest = read_json(blob_path(blobs, lock["runtime"]["digest"]))
            record = {"schema_version": 2, "requested": image, "manifest_digest": lock["runtime"]["digest"],
                      "manifest_size": lock["runtime"]["size"], "root_media_type": lock["runtime"]["mediaType"],
                      "runtime_manifest_digest": lock["runtime"]["digest"], "config": manifest["config"],
                      "layers": manifest["layers"], "compressed_bytes": sum(r["size"] for r in manifest["layers"]),
                      "repo_digests": [lock["registry"] + "/" + lock["repository"] + "@" + lock["root"]["digest"]],
                      "platform": identity["platform"], "descriptor_count": len(graph), "deep_verified": True,
                      "provenance": {**identity, "source_lock_sha256": sha256(lock_path)}}
            path = store / "images" / (key + ".json")
            if path.exists() and read_json(path) != record:
                raise ValueError("Immutable Harbor image publication changed")
            write_json(path, record)
            references[image] = {"store": store.name, "manifest": path.relative_to(store).as_posix(),
                                 "manifest_sha256": sha256(path), "image_id": record["manifest_digest"]}
        publication = {"schema_version": 1, **identity, "images": references}
        write_json(store / "images-manifest.json", publication)
        return publication


def references(cfg, snapshot_sha256):
    path = Path(cfg["paths"]["nas_root"]) / identifier(cfg["harbor"]["image_store"]) / "images-manifest.json"
    publication = read_json(path)
    if publication["snapshot_sha256"] != snapshot_sha256 or publication["platform"] != cfg["environment"]["platform"]:
        raise ValueError("Harbor NAS images belong to another task snapshot/platform")
    for ref in publication["images"].values():
        NasImageStore(cfg, ref).verify()
    return publication["images"]
