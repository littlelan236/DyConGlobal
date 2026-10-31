"""Typed, content-addressed evaluation subjects and durable finalization stages."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from core.io import hash_json, now, owned_path, read_json, sha256, write_json

STAGES = ("agent_terminal", "subject_bound", "verifier_persisted", "artifacts_committed",
          "cleanup_confirmed", "settled")


def artifact_kinds(row):
    return ("trajectory", "subject") if row.get("evaluation_subject", {}).get("kind") == "workspace" else (
        "trajectory", "patch")


def subject_digest(row):
    key = "subject_sha256" if "subject" in artifact_kinds(row) else "patch_sha256"
    return key, row[key]


def verify_subject(root, row):
    for kind in artifact_kinds(row):
        path = owned_path(root, row[kind + "_path"])
        if sha256(path) != row[kind + "_sha256"]:
            raise ValueError("Changed " + kind)
        if kind == "subject":
            manifest = read_json(path)
            if manifest.get("schema_version") != 1 or manifest.get("kind") != "workspace":
                raise ValueError("Invalid workspace subject manifest")
            for output in manifest["outputs"]:
                file = owned_path(path.parent, output["path"])
                if file.is_symlink() or not file.is_file() or sha256(file) != output["sha256"]:
                    raise ValueError("Changed subject output")
    return True


def safe_output_path(value):
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) == "/" or ".." in path.parts
            or any(str(path) == p or str(path).startswith(p + "/")
                   for p in ("/proc", "/sys", "/dev", "/oracle", "/verifier", "/tests"))):
        raise ValueError("Unsafe submission output path: " + str(value))
    return str(path)


def journal(directory, stage, binding, **details):
    if stage not in STAGES:
        raise ValueError("Unknown finalization stage")
    target = Path(directory) / "finalization" / (f"{STAGES.index(stage):02d}-" + stage + ".json")
    value = {"schema_version": 1, "stage": stage, "binding": binding, "details": details}
    if target.exists():
        previous = read_json(target)
        if any(previous.get(k) != v for k, v in value.items()):
            raise ValueError("Conflicting finalization stage")
        return previous
    value["written_at"] = now()
    value["receipt_sha256"] = hash_json(value)
    write_json(target, value)
    return value


async def export_workspace(env, output_paths, target, binding, task_sha256, image_identity):
    import shlex
    import uuid

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "probes").mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for index, raw in enumerate(output_paths):
        source = safe_output_path(raw)
        quoted = shlex.quote(source)
        probe = await env.exec(
            "test ! -L " + quoted + " && { if test -f " + quoted + "; then printf 'file\n'; "
            "elif test -d " + quoted + "; then printf 'directory\n'; else exit 1; fi; } && stat -c '%a' " + quoted,
            timeout_sec=20, user="root")
        write_json(target / "probes" / f"{index:03d}.json", {
            "schema_version": 1, "probe_version": "workspace-kind-mode-v2",
            "source_path": source, "return_code": probe.return_code,
            "stdout": probe.stdout, "stderr": getattr(probe, "stderr", "")})
        if probe.return_code != 0:
            missing.append(source)
            continue
        lines = probe.stdout.splitlines()
        if len(lines) < 2 or lines[0] not in {"file", "directory"}:
            raise ValueError("Could not classify submission output: " + source)
        kind, mode = lines[0], lines[1].strip()
        destination = target / "outputs" / f"{index:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            file = destination / PurePosixPath(source).name
            await env.download_file(source, file)
            if file.is_symlink() or not file.is_file():
                raise ValueError("Submission export is not a regular file")
            record = {"kind": "file", "source_path": source,
                      "path": file.relative_to(target).as_posix(), "sha256": sha256(file),
                      "size": file.stat().st_size, "mode": mode}
        else:
            archive_name = PurePosixPath(source).name + ".tar.gz"
            file = destination / archive_name
            remote_archive = "/tmp/dycon-subject-" + uuid.uuid4().hex + ".tar.gz"
            parent, name = shlex.quote(str(PurePosixPath(source).parent)), shlex.quote(PurePosixPath(source).name)
            try:
                packed = await env.exec("tar -czf " + shlex.quote(remote_archive) + " -C " + parent + " " + name,
                                        timeout_sec=300, user="root")
                if packed.return_code != 0:
                    raise RuntimeError("Could not archive submission directory: " + source)
                await env.download_file(remote_archive, file)
            finally:
                await env.exec("rm -f " + shlex.quote(remote_archive), timeout_sec=20, user="root")
            if file.is_symlink() or not file.is_file():
                raise ValueError("Directory submission export is not a regular archive")
            record = {"kind": "directory_archive", "source_path": source,
                      "path": file.relative_to(target).as_posix(), "sha256": sha256(file),
                      "size": file.stat().st_size, "mode": mode}
        rows.append(record)
    manifest = {"schema_version": 1, "kind": "workspace", "binding": binding,
                "task_package_sha256": task_sha256, "image_identity": image_identity,
                "outputs": rows, "missing_outputs": missing, "complete": not missing,
                "re_evaluation": False, "re_evaluation_reason": "Live sandbox required; file export is not full state"}
    path = target / "subject.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Subject was already bound to other bytes")
    write_json(path, manifest)
    return path
