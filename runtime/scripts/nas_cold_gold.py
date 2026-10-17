"""Run both cohorts from NAS with cold image starts and lifecycle evidence."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import subprocess
import tarfile
import tempfile
import traceback
from pathlib import Path

import docker

from baseline.catalog import Catalog
from core.io import now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.nas import NasImageStore
from runtime.scripts.gold_image_audit import audit, load_cfg


def wait_exit(pid):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def backup_container(client, item, out):
    """Preserve the stopped container's repository before removing its layer."""
    cid = item["Id"]
    meta = client.api.inspect_container(cid)
    if meta["State"]["Running"] or meta["State"]["Status"] != "exited":
        raise RuntimeError(f"Container is no longer stopped: {cid}")
    folder = Path("/VM/wangtianyi/DyConGlobal/container-backups/nas-cold-458-20261017") / cid
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "inspect.json", meta)
    write_json(folder / "filesystem-diff.json", client.api.diff(cid))
    (folder / "logs.txt").write_bytes(client.api.logs(cid, stdout=True, stderr=True))
    # docker archive works on a stopped container and preserves its .git and worktree.
    archive = folder / "testbed.tar"
    stream, stat = client.api.get_archive(cid, "/testbed")
    with archive.open("wb") as dest:
        for chunk in stream:
            dest.write(chunk)
    write_json(folder / "archive.json", {"stat": stat, "sha256": sha256(archive)})
    with tempfile.TemporaryDirectory(prefix="stopped-repo-", dir=out) as scratch:
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
        repo = Path(scratch) / stat["name"]
        git = "/home/wangtianyi/.local/bin/git"
        command = [git, "-c", "safe.directory=" + str(repo), "-C", str(repo)]
        for filename, arguments in [
            ("git-diff.patch", ["diff", "--binary", "HEAD"]),
            ("git-status.txt", ["status", "--porcelain=v1"]),
        ]:
            result = subprocess.run(command + arguments, capture_output=True)
            (folder / filename).write_bytes(result.stdout)
            (folder / (filename + ".stderr")).write_bytes(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Cannot preserve git diff: {folder}")
    # Recheck state immediately before the authorized deletion.
    meta = client.api.inspect_container(cid)
    if meta["State"]["Status"] != "exited":
        raise RuntimeError("Container state changed before removal")
    client.api.remove_container(cid, force=False, v=False)
    write_json(folder / "removed.json", {"at": now(), "container_id": cid})


def purge(cfg, rows, out, allowed_stopped):
    """Delete only verified cohort identities and their aliases; never prune."""
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
    try:
        ids = {NasImageStore(cfg, r["nas_image"]).image_id for r in rows}
        containers = [c for c in client.api.containers(all=True) if c["ImageID"] in ids]
        blockers = [c for c in containers if c["Id"] not in allowed_stopped or c["State"] != "exited"]
        write_json(out / "container-preflight.json", {"containers": containers, "blockers": blockers})
        if blockers:
            raise RuntimeError(f"{len(blockers)} unapproved/in-use containers block cold start")
        for item in containers:
            backup_container(client, item, out)
        images = {x["Id"]: x for x in client.api.images(all=True)}
        receipt = []
        for image_id in sorted(ids):
            info = images.get(image_id)
            if info:
                # Remove all aliases explicitly, with no force and no parent pruning.
                for tag in info.get("RepoTags") or []:
                    client.api.remove_image(tag, force=False, noprune=True)
                try:
                    client.api.remove_image(image_id, force=False, noprune=True)
                except docker.errors.ImageNotFound:
                    pass
            receipt.append(
                {"image_id": image_id, "present_before": bool(info), "aliases": (info or {}).get("RepoTags")}
            )
            write_json(out / "image-removal.json", receipt)
        remaining = ids & {x["Id"] for x in client.api.images(all=True)}
        write_json(
            out / "cold-start-preflight.json",
            {"at": now(), "images": len(ids), "remaining": sorted(remaining)},
        )
        if remaining:
            raise RuntimeError(f"{len(remaining)} task images remain before evaluation")
    finally:
        client.close()


def prepare200(root, out):
    source = Path("/home/wangtianyi/tactRefractor/data/swebench/verified200-nas/instances.json")
    rows = read_json(source)
    wheels = root / "runtime/cache/nas200-wheels"
    shutil.copytree("/home/wangtianyi/tactRefractor/assets/wheels-nas200", wheels, dirs_exist_ok=True)
    # Evaluation inputs are independent of the old project's local wheel paths.
    for row in rows:
        env = row.get("environment_overrides") or {}
        if env.get("wheelhouse"):
            env["wheelhouse"] = str(wheels)
        row["environment_overrides"] = env
    write_json(
        out / "verified200-source.json",
        {"source": str(source), "source_sha256": sha256(source), "tasks": len(rows)},
    )
    assert len(rows) == 200
    return rows


def prepare258(root):
    proof = read_json(root / "state/nas-export-258/verification.json")
    if not proof.get("verified") or proof.get("tasks") != 258:
        raise RuntimeError("258-image export verification failed")
    rows = [t["instance"] for t in Catalog(root).read()["tasks"].values()]
    assert len(rows) == 258
    # Audit uses /VM as the namespace to read both cohorts. The catalog is unchanged.
    result = []
    for row in rows:
        ref = dict(row["nas_image"])
        ref["store"] = "wangtianyi/DyConGlobal/" + ref["store"]
        result.append({**row, "nas_image": ref})
    return result


def run_cohort(root, out, name, rows, allowed_stopped):
    folder = out / name
    folder.mkdir(exist_ok=False)
    path = folder / "instances.json"
    write_json(path, sorted(rows, key=lambda x: x["instance_id"]))
    cfg = load_cfg(root, path, folder)
    # Descriptor integrity is verified before any image deletion. write_archive
    # performs full SHA256 verification before each actual Docker load.
    print(f"NAS INTEGRITY {name} ({len(rows)} tasks)", flush=True)
    verified_blobs = set()
    for row in rows:
        store = NasImageStore(cfg, row["nas_image"])
        for descriptor in store.verify(deep=False):
            digest = descriptor["digest"]
            if digest not in verified_blobs:
                if "sha256:" + sha256(store.blob(digest)) != digest:
                    raise ValueError(f"Corrupt NAS backup: {store.blob(digest)}")
                verified_blobs.add(digest)
    write_json(
        folder / "backup-integrity.json",
        {"at": now(), "unique_blobs": len(verified_blobs), "deep_verified": True},
    )
    print(f"PURGE {name}: verified {len(verified_blobs)} unique NAS blobs", flush=True)
    purge(cfg, rows, folder, allowed_stopped)
    print(f"COLD GOLD {name}: 4 workers", flush=True)
    latest = {}
    summaries = []
    pending = rows
    for attempt in range(1, 4):
        if not pending:
            break
        selected = folder / f"attempt-{attempt}-ids.json"
        write_json(selected, [r["instance_id"] for r in pending])
        aid = out.name + "-" + name + f"-attempt{attempt}"
        summary = audit(root, path, aid, 4, selected, cold_start=True)
        summaries.append(str(root / "state/gold-image-audit" / aid / "summary.json"))
        latest.update({r["instance_id"]: r for r in summary["results"]})
        # Fresh attempts retry infrastructure/lifecycle failures. Genuine test
        # failures remain visible for diagnosis rather than repeated unchanged runs.
        pending = [r for r in rows if latest[r["instance_id"]]["status"] != "completed"]
        if pending and attempt < 3:
            # cleanup after a failed import must also be verified, not bypassed.
            retry_dir = folder / f"retry-{attempt}-cleanup"
            retry_dir.mkdir()
            purge(cfg, pending, retry_dir, set())
    failed = [
        iid
        for iid, r in latest.items()
        if r.get("resolved") is not True or not r.get("cold_lifecycle_verified")
    ]
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
    ids = {NasImageStore(cfg, r["nas_image"]).image_id for r in rows}
    remaining = sorted(ids & {x["Id"] for x in client.api.images(all=True)})
    client.close()
    report = {
        "tasks": len(rows),
        "resolved": sum(r.get("resolved") is True for r in latest.values()),
        "cold_lifecycle_verified": sum(r.get("cold_lifecycle_verified") is True for r in latest.values()),
        "failed": failed,
        "remaining_images": remaining,
        "attempt_summaries": summaries,
        "passed": not failed and not remaining,
        "finished_at": now(),
        "results": latest,
    }
    write_json(folder / "summary.json", report)
    return report


def run(root, export_verifier_pid, allowed_path):
    out = root / "state/gold-image-audit/nas-cold-458-20261017"
    out.mkdir(exist_ok=False)
    allowed = set(read_json(allowed_path))
    status = {"started_at": now(), "stage": "verified200", "cohorts": {}}
    write_json(out / "status.json", status)
    # The two cohorts have disjoint image digests. 200 can run during 258 export.
    for name in ["verified200", "legacy258"]:
        try:
            status["stage"] = name
            write_json(out / "status.json", status)
            if name == "legacy258":
                status["stage"] = "waiting-for-258-export-verification"
                write_json(out / "status.json", status)
                print("WAIT for 258 export verification (pidfd event)", flush=True)
                wait_exit(export_verifier_pid)
                rows = prepare258(root)
            else:
                rows = prepare200(root, out)
            print(f"START {name} {len(rows)} tasks", flush=True)
            status["cohorts"][name] = run_cohort(root, out, name, rows, allowed)
            print(f"FINISH {name} passed={status['cohorts'][name]['passed']}", flush=True)
        except Exception:
            status["cohorts"][name] = {"passed": False, "error": traceback.format_exc()}
            print(status["cohorts"][name]["error"], flush=True)
        write_json(out / "status.json", status)
    status.update(
        stage="passed" if all(c["passed"] for c in status["cohorts"].values()) else "needs-investigation",
        finished_at=now(),
    )
    write_json(out / "summary.json", status)
    write_json(out / "status.json", status)
    return 0 if status["stage"] == "passed" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--export-verifier-pid", type=int, required=True)
    p.add_argument("--approved-stopped", type=Path, required=True)
    args = p.parse_args()
    with file_lock(args.root / "state/gold-image-audit/nas-cold-458.lock"):
        raise SystemExit(run(args.root.resolve(), args.export_verifier_pid, args.approved_stopped))
