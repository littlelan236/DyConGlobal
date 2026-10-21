"""Resume the 258-task NAS export, preserving complete immutable OCI graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

import docker

from catalog.store import Catalog
from core.config import resolve
from core.io import now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.oci import blob_path, descriptor_graph, platform_manifest, verify_blobs


def import_archive(cfg, tags, blobs):
    """Stream only missing tasks, deduplicating blobs without staging local tar files."""
    proc = subprocess.Popen(
        [
            cfg["environment"]["docker_executable"],
            "--host",
            cfg["environment"]["docker_host"],
            "save",
            *tags,
        ],
        stdout=subprocess.PIPE,
    )
    aliases = {}
    deferred = []
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                if member.islnk() or member.issym():
                    deferred.append((member.name, member.linkname))
                    continue
                if not member.isfile():
                    continue
                source = archive.extractfile(member)
                if not member.name.startswith("blobs/sha256/"):
                    # Read via the streaming tar iterator, never extract paths to disk.
                    continue
                expected = member.name.rsplit("/", 1)[-1]
                target = blob_path(blobs, "sha256:" + expected)
                tmp = blobs / (".export-" + uuid.uuid4().hex)
                digest = hashlib.sha256()
                try:
                    with tmp.open("wb") as output:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if digest.hexdigest() != expected:
                        raise ValueError(f"Wrong archive blob hash: {member.name}")
                    if (
                        target.exists()
                        and target.stat().st_size == member.size
                        and sha256(target) == expected
                    ):
                        tmp.unlink()
                    else:
                        tmp.replace(target)
                    aliases[member.name] = target
                finally:
                    tmp.unlink(missing_ok=True)
        if proc.wait():
            raise RuntimeError("docker save failed")
        # Docker save can represent identical files as hardlinks. Only content
        # digest aliases inside the archive are accepted, never filesystem links.
        for name, link in deferred:
            if not name.startswith("blobs/sha256/"):
                continue
            source = aliases.get(link)
            if source is None:
                source = blob_path(blobs, "sha256:" + link.rsplit("/", 1)[-1])
            expected = name.rsplit("/", 1)[-1]
            if sha256(source) != expected:
                raise ValueError(f"Invalid Docker archive blob alias: {name}")
            target = blob_path(blobs, "sha256:" + expected)
            if not target.exists():
                shutil.copyfile(source, target)
    except BaseException:
        proc.kill()
        proc.wait()
        raise


def export(root: Path, store_name="legacy-258-20261017", *, dry_run=False, evidence=None):
    root = root.resolve()
    evidence = evidence or root / "state/nas-export-258-recovery-20261018"
    cfg = resolve(root / "config.yaml")
    catalog = Catalog(root)
    initial = catalog.read()
    tasks = sorted((t["instance"] for t in initial["tasks"].values()), key=lambda r: r["instance_id"])
    if len(tasks) != 258 or not all(t.get("docker_image") for t in tasks):
        raise ValueError("Expected exactly 258 catalog tasks with original Docker tags")
    store = (Path(cfg["paths"]["nas_root"]) / store_name).resolve()
    if not store.is_relative_to(Path(cfg["paths"]["nas_root"]).resolve()):
        raise ValueError("NAS store escapes configured namespace")
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
    try:
        images = client.api.images(all=True)
        listed = {tag: image["Id"] for image in images for tag in (image.get("RepoTags") or [])}
        ids = {r["instance_id"]: listed[r["docker_image"]] for r in tasks}
    finally:
        client.close()
    blobs = store / "blobs/sha256"
    records = store / "images"
    blobs.mkdir(parents=True, exist_ok=True)
    records.mkdir(exist_ok=True)

    def describe(row):
        image_id = ids[row["instance_id"]]
        path = blob_path(blobs, image_id)
        obj = read_json(path)
        descriptor = {"digest": image_id, "size": path.stat().st_size, "mediaType": obj["mediaType"]}
        graph = descriptor_graph(blobs, descriptor)
        runtime, manifest = platform_manifest(blobs, descriptor, cfg["environment"]["platform"])
        return {
            "requested": row["docker_image"],
            "manifest_digest": image_id,
            "manifest_size": descriptor["size"],
            "root_media_type": descriptor["mediaType"],
            "runtime_manifest_digest": runtime["digest"],
            "config": manifest["config"],
            "layers": manifest["layers"],
            "compressed_bytes": sum(d["size"] for d in graph),
            "repo_digests": [],
        }

    pending = []
    for task in tasks:
        try:
            describe(task)
        except (FileNotFoundError, KeyError, ValueError):
            pending.append(task["docker_image"])
    print(
        f"RESUME: {len(tasks) - len(pending)}/258 complete graphs, {len(pending)} images need export",
        flush=True,
    )
    if dry_run:
        return {"tasks": len(tasks), "store": str(store), "missing_images": pending}
    evidence.mkdir(parents=True, exist_ok=True)
    write_json(evidence / "original-identities.json", ids)
    write_json(evidence / "catalog-before.json", initial)
    if pending:
        import_archive(cfg, pending, blobs)
    # Derive runtime fields from linux/amd64, retain index + attestation digests.
    task_records = {r["instance_id"]: describe(r) for r in tasks}
    descriptors = []
    for record in task_records.values():
        descriptors.extend(
            descriptor_graph(
                blobs,
                {
                    "digest": record["manifest_digest"],
                    "size": record["manifest_size"],
                    "mediaType": record["root_media_type"],
                },
            )
        )

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"INTEGRITY {done}/{total} unique blobs verified (4 read workers)", flush=True)

    verified = verify_blobs(blobs, descriptors, workers=4, progress=progress)
    for iid, record in task_records.items():
        write_json(records / (iid + ".json"), record)
    old_cfg = read_json("/home/wangtianyi/tactRefractor/baseline/config/resolved-config.json")
    env = old_cfg["environment"]
    wheels = store / "wheels"
    shutil.copytree(env["wheelhouse"], wheels, dirs_exist_ok=True)
    write_json(store / "wheel-hashes.json", {p.name: sha256(p) for p in wheels.glob("*.whl")})
    with catalog.transaction() as data:
        if set(data["tasks"]) != set(initial["tasks"]):
            raise ValueError("Task membership changed during export")
        for key, task in data["tasks"].items():
            row = task["instance"]
            iid = row["instance_id"]
            if row["docker_image"] != initial["tasks"][key]["instance"]["docker_image"]:
                raise ValueError("Task image changed during export")
            overrides = row.get("environment_overrides") or {}
            row["environment_overrides"] = {
                **overrides,
                "wheelhouse": str(wheels),
                "constraints": list(dict.fromkeys(env["constraints"] + overrides.get("constraints", []))),
                "repo_pins": {**env["repo_pins"], **overrides.get("repo_pins", {})},
                "network_fallback": False,
            }
            row["nas_image"] = {
                "store": store_name,
                "manifest": "images/" + iid + ".json",
                "manifest_sha256": sha256(records / (iid + ".json")),
                "image_id": ids[iid],
            }
    result = {
        "tasks": len(tasks),
        "store": str(store),
        "manifest_files": len(task_records),
        "unique_verified_blobs": len(verified),
        "reexported_images": len(pending),
        "index_images": sum(
            "index" in r["root_media_type"] or "manifest.list" in r["root_media_type"]
            for r in task_records.values()
        ),
        "deep_verified": True,
        "finished_at": now(),
    }
    write_json(evidence / "receipt.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--store", default="legacy-258-20261017")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    with file_lock(args.root / "state/nas-export-258.lock"):
        print(json.dumps(export(args.root, args.store, dry_run=args.dry_run), indent=2))
