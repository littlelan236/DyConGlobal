"""Typed, content-addressed evaluation subjects and durable finalization stages."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from core.io import hash_json, now, owned_path, read_json, sha256, write_json

STAGES = ("agent_terminal", "subject_bound", "verifier_persisted", "artifacts_committed",
          "cleanup_confirmed", "settled")


def artifact_kinds(row):
    return ("trajectory", "subject") if row.get("evaluation_subject", {}).get("kind") in {"workspace", "web_trace", "live_state"} else (
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
            if (manifest.get("schema_version") != 1 or manifest.get("kind") not in {"workspace", "web_trace", "live_state"}
                    or manifest.get("kind") != row.get("evaluation_subject", {}).get("kind")):
                raise ValueError("Invalid evaluation subject manifest")
            if manifest["kind"] == "workspace" and "output_inventory" in manifest:
                if (type(manifest.get("complete")) is not bool
                        or manifest["complete"] != (not manifest.get("missing_outputs")
                                                     and not manifest.get("rejected_outputs"))):
                    raise ValueError("Workspace output inventory conflicts with completeness")
            paths = set()
            for output in manifest["outputs"]:
                if output["path"] in paths:
                    raise ValueError("Duplicate subject output")
                paths.add(output["path"])
                file = owned_path(path.parent, output["path"])
                if file.is_symlink() or not file.is_file() or sha256(file) != output["sha256"]:
                    raise ValueError("Changed subject output")
                if "size" in output and file.stat().st_size != output["size"]:
                    raise ValueError("Changed subject output size")
            if manifest["kind"] == "live_state":
                if manifest.get("re_evaluation") is not False or not manifest.get("trial_id") or not manifest["outputs"]:
                    raise ValueError("Live state proof must bind a real trial and preserved evidence; it is not a replay snapshot")
                binding = manifest.get("binding", {})
                for field in ("run_id", "attempt_id", "task_key", "sample_key"):
                    if binding.get(field) != row.get(field) and not (
                            field == "run_id" and binding.get(field) == row.get("provenance", {}).get("source_run_id")):
                        raise ValueError("Live state proof belongs to another attempt")
                if manifest.get("task_package_sha256") != row.get("task_package_sha256"):
                    raise ValueError("Live state proof belongs to another task package")
            if manifest["kind"] == "web_trace":
                if (manifest.get("task_sha256") is not None and row.get("task_definition_sha256") is not None
                        and manifest["task_sha256"] != row["task_definition_sha256"]):
                    raise ValueError("Web trace belongs to a different task definition")
                if type(manifest.get("complete")) is not bool or type(manifest.get("har_flushed")) is not bool:
                    raise ValueError("Web trace requires explicit completion and HAR flush state")
                required = [manifest.get(key) for key in ("answer_path", "har_path", "actions_path")]
                if any(value is not None and value not in paths for value in required):
                    raise ValueError("Web trace references an unbound output")
                if manifest["complete"] and (not all(required) or not manifest["har_flushed"]):
                    raise ValueError("Complete web trace requires a final answer, flushed HAR and actions")
                binding = manifest.get("binding", {})
                for field in ("run_id", "attempt_id", "task_key"):
                    allowed = {row.get(field)}
                    if field == "run_id":
                        allowed.add(row.get("provenance", {}).get("source_run_id"))
                    if field in binding and binding[field] not in allowed:
                        raise ValueError("Web trace binding differs from its rollout")
                if manifest.get("har_path") and manifest["har_flushed"]:
                    har = read_json(owned_path(path.parent, manifest["har_path"]))
                    if not isinstance(har, dict) or not isinstance(har.get("log", {}).get("entries"), list):
                        raise ValueError("Web trace HAR is not a readable HAR log")
    return True


def safe_output_path(value):
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) == "/" or ".." in path.parts
            or any(str(path) == p or str(path).startswith(p + "/")
                   for p in ("/proc", "/sys", "/dev", "/oracle", "/verifier", "/tests"))):
        raise ValueError("Unsafe submission output path: " + str(value))
    return str(path)


def read_archive_members(path, expected_sha256, wanted, *, member_limit=2*1024*1024,
                         total_limit=16*1024*1024, scan_limit=512*1024*1024):
    """Read selected regular tar members without extracting or following links.

    The caller supplies an authenticated subject output hash. All names/types
    and expanded bytes are bounded, including members the caller does not need.
    """
    import hashlib
    import tarfile

    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > scan_limit:
        raise ValueError('Invalid subject archive')
    if sha256(path) != expected_sha256:
        raise ValueError('Changed subject archive')
    rows, names, expanded, consumed = [], set(), 0, 0
    with tarfile.open(path, mode='r|*') as archive:
        for index, member in enumerate(archive):
            name = PurePosixPath(member.name)
            if index >= 50000 or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe subject archive member')
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                if wanted(name.as_posix()):
                    raise ValueError('Requested subject member is not a regular file')
                # Reject the member; never follow even a benign internal link.
                continue
            if member.name in names:
                raise ValueError('Duplicate subject archive member')
            names.add(member.name)
            expanded += member.size
            if expanded > scan_limit:
                raise ValueError('Expanded archive budget exceeded')
            if not member.isfile() or not wanted(name.as_posix()):
                continue
            if member.size > member_limit or consumed+member.size > total_limit:
                raise ValueError('Subject member read budget exceeded')
            stream = archive.extractfile(member)
            data = stream.read(member_limit+1)
            if len(data) != member.size:
                raise ValueError('Incomplete archive member')
            consumed += len(data)
            rows.append({'name':member.name,'sha256':hashlib.sha256(data).hexdigest(),
                         'archive_sha256':expected_sha256,'size':len(data),'data':data})
    if sha256(path) != expected_sha256:
        raise ValueError('Subject archive changed while reading')
    return rows


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
    rows, missing, rejected, inventory = [], [], [], []
    for index, raw in enumerate(output_paths):
        try:
            source = safe_output_path(raw)
        except ValueError as exc:
            rejected.append(str(raw))
            inventory.append({"required_path": str(raw), "status": "rejected",
                              "reason": "unsafe_path", "error_type": type(exc).__name__})
            continue
        quoted = shlex.quote(source)
        probe = await env.exec(
            "if test -L " + quoted + "; then printf 'symlink\n'; exit 2; fi; "
            "if test -f " + quoted + "; then printf 'file\n'; "
            "elif test -d " + quoted + "; then printf 'directory\n'; "
            "else printf 'missing\n'; exit 1; fi; stat -c '%a' " + quoted,
            timeout_sec=20, user="root")
        write_json(target / "probes" / f"{index:03d}.json", {
            "schema_version": 1, "probe_version": "workspace-kind-mode-v3",
            "source_path": source, "return_code": probe.return_code,
            "stdout": probe.stdout, "stderr": getattr(probe, "stderr", "")})
        if probe.return_code != 0:
            kind = (probe.stdout or "").splitlines()[0] if probe.stdout else ""
            if kind == "missing":
                missing.append(source)
                inventory.append({"required_path": source, "status": "missing",
                                  "reason": "required_output_not_found"})
            else:
                rejected.append(source)
                inventory.append({"required_path": source, "status": "rejected",
                                  "reason": "output_probe_failed", "probe_return_code": probe.return_code})
            continue
        lines = probe.stdout.splitlines()
        if len(lines) < 2 or lines[0] not in {"file", "directory"}:
            rejected.append(source)
            inventory.append({"required_path": source, "status": "rejected",
                              "reason": "output_probe_unclassified"})
            continue
        kind, mode = lines[0], lines[1].strip()
        destination = target / "outputs" / f"{index:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        try:
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
                        raise RuntimeError("Could not archive submission directory")
                    await env.download_file(remote_archive, file)
                finally:
                    await env.exec("rm -f " + shlex.quote(remote_archive), timeout_sec=20, user="root")
                if file.is_symlink() or not file.is_file():
                    raise ValueError("Directory submission export is not a regular archive")
                record = {"kind": "directory_archive", "source_path": source,
                          "path": file.relative_to(target).as_posix(), "sha256": sha256(file),
                          "size": file.stat().st_size, "mode": mode}
        except Exception as exc:
            rejected.append(source)
            inventory.append({"required_path": source, "status": "rejected",
                              "reason": "export_failed", "error_type": type(exc).__name__})
            continue
        rows.append(record)
        inventory.append({"required_path": source, "status": "exported",
                          "output_index": len(rows) - 1, "artifact_path": record["path"]})
    manifest = {"schema_version": 1, "kind": "workspace", "binding": binding,
                "task_package_sha256": task_sha256, "image_identity": image_identity,
                "outputs": rows, "missing_outputs": missing, "rejected_outputs": rejected,
                "output_inventory": inventory, "complete": not missing and not rejected,
                "re_evaluation": False, "re_evaluation_reason": "Live sandbox required; file export is not full state"}
    path = target / "subject.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Subject was already bound to other bytes")
    write_json(path, manifest)
    return path
