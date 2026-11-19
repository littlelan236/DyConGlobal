"""Immutable Harbor task packages with separate agent/evaluator views."""
from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

from catalog.skillsbench import package_manifest
from core.io import hash_json, identifier, read_json, sha256, write_json

REVISIONS = {"swebench_pro_v2": "66f92766bba642462d4bbe5479e83f91f9211862",
             "terminalbench_2_1": "7131e4375048a0e408a8fb404b5f499d726b695b"}


def validate_payload(payload, instance_id, revision, benchmark):
    required = {"instance_id", "task_path", "package_sha256", "files", "metadata", "instruction", "images"}
    if set(payload) != required or payload["instance_id"] != instance_id or revision != REVISIONS[benchmark]:
        raise ValueError("Invalid pinned Harbor task identity")
    if hash_json(payload["files"]) != payload["package_sha256"]:
        raise ValueError("Harbor task package manifest hash changed")
    seen = set()
    for row in payload["files"]:
        path = PurePosixPath(row["path"])
        if (path.is_absolute() or ".." in path.parts or str(path) in seen or
                not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
            raise ValueError("Unsafe Harbor package manifest")
        seen.add(str(path))
    expected = "1.4" if benchmark == "swebench_pro_v2" else "1.1"
    if payload["metadata"].get("schema_version") != expected:
        raise ValueError("Harbor task schema does not match its benchmark revision")
    if benchmark == "swebench_pro_v2" and payload["metadata"].get("agent", {}).get("network_mode") != "no-network":
        raise ValueError("Pro V2 requires the locked offline agent phase")


def verify_package(task):
    payload = task["instance"]
    validate_payload(payload, task["instance_id"], task["revision"], task["benchmark"])
    if package_manifest(payload["task_path"]) != payload["files"]:
        raise ValueError("Pinned Harbor task package changed")


def import_tasks(cfg, snapshot_id, source, benchmark, task_ids, runtime, partition="train"):
    from catalog.store import Catalog, Task
    from runtime.backends.harbor.dependencies import probe_schemas, verify_runtime

    root, source = Path(cfg["project_root"]).resolve(), Path(source).resolve()
    snapshot_id = identifier(snapshot_id)
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Select a nonempty explicit list of distinct Harbor tasks")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != REVISIONS[benchmark]:
        raise ValueError("Harbor benchmark checkout differs from its pinned revision")
    task_root = source / ("v2/tasks" if benchmark == "swebench_pro_v2" else "tasks")
    target = root / "catalog/snapshots" / snapshot_id
    if target.exists():
        raise FileExistsError("Harbor snapshot is immutable; use a new snapshot ID")
    paths = []
    for iid in task_ids:
        path = task_root / identifier(iid)
        if path.is_symlink() or not path.is_dir() or path.parent != task_root:
            raise ValueError("Unknown/unsafe official Harbor task: " + iid)
        files = package_manifest(path)
        for row in files:
            expected = subprocess.check_output(["git", "show", revision + ":" +
                                                (path / row["path"]).relative_to(source).as_posix()], cwd=source)
            if (path / row["path"]).read_bytes() != expected:
                raise ValueError("Modified official Harbor task: " + iid)
        paths.append(path)
    probe = probe_schemas(cfg, runtime, paths)
    rows = []
    for iid, path in zip(task_ids, paths, strict=True):
        files = package_manifest(path)
        metadata = tomllib.loads((path / "task.toml").read_text())
        if metadata.get("steps"):
            raise ValueError("Multi-step Harbor trials require separate qualification")
        if list((path / "environment").glob("*compose*")):
            raise ValueError("Multi-service Harbor task environments require separate qualification")
        image = metadata.get("environment", {}).get("docker_image")
        if not image:
            raise ValueError("A published prebuilt task image is required; implicit builds are disabled")
        images = {"agent": image}
        verifier = metadata.get("verifier", {}).get("environment", {})
        if verifier:
            if not verifier.get("docker_image"):
                raise ValueError("Separate verifier requires its own published prebuilt image")
            images["verifier"] = verifier["docker_image"]
        rows.append({"benchmark": benchmark, "revision": revision, "instance_id": iid, "partition": partition,
                     "instance": {"instance_id": iid, "task_path": str(target / "packages" / iid),
                                  "package_sha256": hash_json(files), "files": files, "metadata": metadata,
                                  "instruction": (path / "instruction.md").read_text(), "images": images}})
    rows = [Task.model_validate(row).model_dump() for row in rows]
    # Catalog identity is benchmark/revision/instance, independent of which
    # explicit subset imports it. Reuse its existing immutable package location.
    registered = Catalog(root).read()["tasks"]
    for index, row in enumerate(rows):
        key = hash_json({k: row[k] for k in ("benchmark", "revision", "instance_id")})
        if key in registered:
            existing = registered[key]
            def comparable(value):
                return {**value, "instance": {k: v for k, v in value["instance"].items()
                                                if k != "task_path"}}
            if comparable(existing) != comparable(row):
                raise ValueError("Official Harbor task conflicts with an existing catalog identity")
            verify_package(existing)
            rows[index] = existing
    target.mkdir(parents=True, mode=0o700)
    for iid, path in zip(task_ids, paths, strict=True):
        shutil.copytree(path, target / "packages" / iid)
    # Upstream replay is evaluator-only; it is never uploaded to an agent env.
    tooling = source / "v2/tooling/patch_replay.py"
    if benchmark == "swebench_pro_v2":
        expected = subprocess.check_output(["git", "show", revision + ":v2/tooling/patch_replay.py"], cwd=source)
        if tooling.read_bytes() != expected:
            raise ValueError("Official Pro replay implementation changed")
        (target / "evaluator").mkdir(mode=0o700)
        shutil.copy2(tooling, target / "evaluator/patch_replay.py")
    tasks = {hash_json({k: t[k] for k in ("benchmark", "revision", "instance_id")}): t for t in rows}
    runtime_manifest = verify_runtime(runtime)
    snapshot = {"schema_version": 1, "benchmark": benchmark, "revision": revision, "tasks": tasks,
                "schema_probe": probe, "harbor_revision": runtime_manifest["revision"],
                "worker_protocol": 1, "re_evaluation": benchmark == "swebench_pro_v2"}
    if benchmark == "swebench_pro_v2":
        snapshot["evaluator"] = {"path": (target / "evaluator/patch_replay.py").relative_to(root).as_posix(),
                                 "sha256": sha256(target / "evaluator/patch_replay.py")}
    Catalog(root).register_tasks(rows)
    write_json(target / "manifest.json", snapshot)
    return {"snapshot_ref": (target / "manifest.json").relative_to(root).as_posix(),
            "snapshot_sha256": sha256(target / "manifest.json"), "task_count": len(tasks)}


def execution_context(cfg, tasks):
    from runtime.agents.openhands import OpenHandsDriverProfile
    from runtime.backends.harbor.dependencies import verify_runtime

    root = Path(cfg["project_root"]).resolve()

    def project_file(ref):
        path = (root / ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Harbor references must name existing project files")
        return path

    benchmark = cfg["benchmark"]["backend"]
    snapshot_path = project_file(cfg["benchmark"].get("snapshot_ref") or "")
    snapshot = read_json(snapshot_path)
    if snapshot.get("benchmark") != benchmark or snapshot.get("revision") != REVISIONS[benchmark]:
        raise ValueError("Invalid pinned Harbor benchmark snapshot")
    for key, task in tasks.items():
        if snapshot["tasks"].get(key) != task:
            raise ValueError("Harbor task does not belong to frozen snapshot")
        verify_package(task)
    profile_path = project_file(cfg["agent"].get("profile_ref") or "")
    profile = OpenHandsDriverProfile.model_validate(read_json(profile_path)).model_dump()
    from runtime.agents.dependencies import verify_snapshot
    if not profile["dependency_snapshot"]:
        raise ValueError("Harbor ACP requires the existing offline OpenHands publication")
    verify_snapshot(profile["dependency_snapshot"])
    dependency_path = Path(profile["dependency_snapshot"]) / "manifest.json"
    if profile["dependency_snapshot_sha256"] != sha256(dependency_path):
        raise ValueError("OpenHands dependency publication differs from the Harbor profile")
    # A SkillsBench qualification does not qualify the new framework transport.
    if profile["qualification"] and profile["qualification"].get("backend_family") != "harbor":
        raise ValueError("Harbor request-role qualification cannot inherit SkillsBench receipts")
    runtime_path = project_file(cfg["harbor"]["runtime_ref"])
    runtime = verify_runtime(runtime_path.parent)
    from runtime.images.harbor import references
    image_references = references(cfg, sha256(snapshot_path))
    evaluator = snapshot.get("evaluator")
    if evaluator and sha256(project_file(evaluator["path"])) != evaluator["sha256"]:
        raise ValueError("Official Pro replay implementation changed")
    body = {"schema_version": 1, "benchmark": benchmark, "agent": "openhands", "revision": snapshot["revision"],
            "snapshot_sha256": sha256(snapshot_path), "profile_sha256": sha256(profile_path), "profile": profile,
            "runtime_ref": cfg["harbor"]["runtime_ref"], "runtime_sha256": sha256(runtime_path),
            "worker_python": runtime["worker_python"], "worker_protocol": runtime["worker_protocol"],
            "image_references": image_references,
            "task_keys": sorted(tasks), "evaluator": evaluator,
            "generation": cfg["generation"], "decoder": cfg.get("parallel", {})}
    if cfg.get("harbor", {}).get("task_preparation") is not None:
        body["task_preparation"] = cfg["harbor"]["task_preparation"]
    from runtime.backends.harbor.policy import configured_policy
    policy = configured_policy(cfg)
    if policy:
        body["terminal_policy"] = policy
        from runtime.agents.openhands import condenser_context
        body.update(condenser_context(cfg, profile))
    return {**body, "context_sha256": hash_json(body)}
