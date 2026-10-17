"""Four tiny synthetic OCI images: live NAS load/create/remove smoke, no benchmark data."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import resolve
from core.io import now, sha256, write_json
from runtime.images.nas import ImageLeaseManager, NasImageStore


def smoke(cfg):
    import docker

    token = "smoke-" + uuid.uuid4().hex[:16]
    nas = Path(cfg["paths"]["nas_root"]) / "images" / token
    blobs = nas / "blobs/sha256"
    blobs.mkdir(parents=True)

    def blob(data, media):
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return {"mediaType": media, "digest": "sha256:" + digest, "size": len(data)}

    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w"):
        pass
    layer = blob(empty.getvalue(), "application/vnd.oci.image.layer.v1.tar")
    refs = []
    for i in range(4):
        config = {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [layer["digest"]]},
            "config": {"Labels": {"dycon.smoke": token, "dycon.index": str(i)}},
        }
        config = blob(json.dumps(config).encode(), "application/vnd.oci.image.config.v1+json")
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config,
            "layers": [layer],
        }
        descriptor = blob(json.dumps(manifest).encode(), manifest["mediaType"])
        record = {
            "manifest_digest": descriptor["digest"],
            "manifest_size": descriptor["size"],
            "config": config,
            "layers": [layer],
            "compressed_bytes": config["size"] + layer["size"],
        }
        name = f"image-{i}.json"
        write_json(nas / name, record)
        refs.append(
            {
                "store": str(nas.relative_to(Path(cfg["paths"]["nas_root"]))),
                "manifest": name,
                "manifest_sha256": sha256(nas / name),
                "image_id": descriptor["digest"],
            }
        )
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"])
    manager = ImageLeaseManager(cfg, client)
    barrier = threading.Barrier(4, timeout=60)

    def worker(reference):
        store = NasImageStore(cfg, reference)
        barrier.wait()
        with manager.lease(store) as image_id:
            c = client.containers.create(
                image_id,
                command=["/fixture-no-executable"],
                labels={"dycon.project": cfg["project_root"], "dycon.smoke": token},
            )
            try:
                assert c.image is not None
                return {"image_id": image_id, "container_id": c.id, "created": True}
            finally:
                c.remove()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, refs))
    finally:
        cleanup = manager.cleanup()
    remaining = []
    for ref in refs:
        try:
            client.images.get(ref["image_id"])
            remaining.append(ref["image_id"])
        except docker.errors.ImageNotFound:
            pass
    report = {
        "at": now(),
        "smoke_id": token,
        "workers": 4,
        "results": results,
        "remaining_images": remaining,
        "cleanup": cleanup,
        "passed": len(results) == 4 and not remaining,
    }
    write_json(Path(cfg["project_root"]) / "runtime/reports" / token / "summary.json", report)
    if not report["passed"]:
        raise RuntimeError("Live NAS smoke failed cleanup")
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    print(json.dumps(smoke(resolve(p.parse_args().config)), indent=2))
