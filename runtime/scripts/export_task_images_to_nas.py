"""Export the canonical 258-task Docker cohort into a DyConGlobal OCI NAS store."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import uuid
from pathlib import Path

from baseline.catalog import Catalog
from core.io import sha256, write_json


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def export(root: Path, store_name: str, dry_run: bool = False) -> dict:
    root = root.resolve()
    catalog = Catalog(root)
    manifest = catalog.read()
    tasks = [t["instance"] for t in manifest["tasks"].values()]
    tasks.sort(key=lambda x: x["instance_id"])
    if len(tasks) != 258 or not all(t.get("docker_image") for t in tasks):
        raise ValueError("Expected 258 catalog tasks with docker_image")
    requested = {t["docker_image"]: t["instance_id"] for t in tasks}
    docker = "/home/wangtianyi/.local/bin/docker"
    host = "unix:///run/user/1002/docker.sock"
    listed = subprocess.check_output(
        [docker, "--host", host, "images", "--no-trunc", "--format", "{{.ID}} {{.Repository}}:{{.Tag}}"],
        text=True,
    )
    ids = {}
    for line in listed.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1] in requested:
            ids[requested[parts[1]]] = parts[0]
    missing = sorted(set(requested.values()) - set(ids))
    if missing:
        raise ValueError(f"Missing Docker images for {len(missing)} tasks: {missing[:10]}")
    store = Path("/VM/wangtianyi/DyConGlobal") / store_name
    if dry_run:
        return {"tasks": len(tasks), "docker_images": len(ids), "missing": missing, "store": str(store)}
    blob_root = store / "blobs/sha256"
    image_root = store / "images"
    blob_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [docker, "--host", host, "save", *[t["docker_image"] for t in tasks]], stdout=subprocess.PIPE
    )
    member_digest: dict[str, dict] = {}
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                # Docker archive metadata is small; blobs are copied once to NAS.
                if member.name == "manifest.json":
                    metadata = json.load(source)
                    member_digest[member.name] = {"metadata": metadata}
                    continue
                if member.name == "repositories":
                    continue
                token = uuid.uuid4().hex
                tmp = blob_root / (".export-" + token)
                h = hashlib.sha256()
                size = 0
                with tmp.open("wb") as out:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        h.update(block)
                        size += len(block)
                        out.write(block)
                digest = h.hexdigest()
                final = blob_root / digest
                if final.exists():
                    if final.stat().st_size != size:
                        raise ValueError(f"Existing NAS blob size mismatch: {final}")
                    tmp.unlink()
                else:
                    tmp.replace(final)
                member_digest[member.name] = {"digest": "sha256:" + digest, "size": size}
        code = proc.wait()
        if code:
            raise RuntimeError(f"docker save exited with {code}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    task_records = {}
    for task in tasks:
        iid = task["instance_id"]
        image_id = ids[iid]
        manifest_blob = blob_root / image_id[7:]
        oci = json.loads(manifest_blob.read_bytes())
        if "config" not in oci or "layers" not in oci:
            raise ValueError(f"Expected a single-platform OCI manifest: {iid}")
        descriptors = [oci["config"], *oci["layers"]]
        for d in descriptors:
            target = blob_root / d["digest"][7:]
            if not target.is_file() or target.stat().st_size != d["size"]:
                raise ValueError(f"Missing archived descriptor: {iid}/{d['digest']}")
        task_records[iid] = {
            "requested": task["docker_image"],
            "manifest_digest": image_id,
            "manifest_size": manifest_blob.stat().st_size,
            "config": oci["config"],
            "layers": oci["layers"],
            "compressed_bytes": sum(d["size"] for d in oci["layers"]),
            "repo_digests": [],
        }
        write_json(image_root / (iid + ".json"), task_records[iid])
    import shutil

    old_config = json.loads(
        Path("/home/wangtianyi/tactRefractor/baseline/config/resolved-config.json").read_text()
    )
    env = old_config["environment"]
    wheels = store / "wheels"
    shutil.copytree(env["wheelhouse"], wheels, dirs_exist_ok=True)
    write_json(store / "wheel-hashes.json", {p.name: sha256(p) for p in wheels.glob("*.whl")})
    with catalog.transaction() as data:
        for task in data["tasks"].values():
            iid = task["instance_id"]
            p = image_root / (iid + ".json")
            task["instance"]["environment_overrides"] = {
                **(task["instance"].get("environment_overrides") or {}),
                "wheelhouse": str(wheels),
                "constraints": env["constraints"],
                "repo_pins": env["repo_pins"],
                "network_fallback": False,
            }
            task["instance"]["nas_image"] = {
                "store": store_name,
                "manifest": "images/" + iid + ".json",
                "manifest_sha256": sha256(p),
                "image_id": task_records[iid]["manifest_digest"],
            }
    return {
        "tasks": len(tasks),
        "docker_images": len(ids),
        "store": str(store),
        "unique_blobs": len(list(blob_root.glob("[0-9a-f]*"))),
        "manifest_files": len(task_records),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--store", default="legacy-258-20261017")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    result = export(args.root, args.store, args.dry_run)
    if not args.dry_run:
        write_json(args.root / "state/nas-export-258/receipt.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
