"""Publish and verify a pinned Harbor worker without changing the runner venv."""
from __future__ import annotations

import json
import os
import subprocess
import tomllib
import uuid
from pathlib import Path

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock

HARBOR_REVISION = "5eb713ff9500daced4e143b98765a7c732744a1b"
HARBOR_VERSION = "0.25.0"


def verify_runtime(directory, *, deep=True):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    body = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    if (manifest.get("schema_version") != 1 or manifest.get("revision") != HARBOR_REVISION or
            manifest.get("version") != HARBOR_VERSION or hash_json(body) != manifest.get("receipt_sha256")):
        raise ValueError("Invalid pinned Harbor worker publication")
    for row in manifest["files"]:
        path = directory / row["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(directory) or not path.is_file() or
                path.stat().st_size != row["size"] or (deep and sha256(path) != row["sha256"])):
            raise ValueError("Harbor dependency changed: " + row["path"])
    python = directory / "venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Harbor worker interpreter is unavailable")
    if deep and sha256(python.resolve()) != manifest["python"]["executable_sha256"]:
        raise ValueError("Harbor worker interpreter changed")
    return manifest


def prepare_runtime(cfg, publication_id, source, uv, python):
    root = Path(cfg["project_root"])
    publication_id = identifier(publication_id)
    source, uv, python = Path(source).resolve(), Path(uv).resolve(), Path(python).absolute()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != HARBOR_REVISION or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=source).strip():
        raise ValueError("Harbor source must be a clean checkout of the pinned revision")
    destination = root / "runtime/backend-snapshots" / publication_id
    with file_lock(root / "state/harbor-runtime.lock"):
        if destination.exists():
            return {"runtime_ref": str((destination / "manifest.json").relative_to(root)),
                    "manifest": verify_runtime(destination)}
        destination.mkdir(parents=True)
        work = root / "temp/backend-refactor-20261031" / publication_id
        work.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "UV_PYTHON_DOWNLOADS": "never"}

        def run(args):
            result = subprocess.run([str(uv), *map(str, args)], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
            with (work / "prepare.log").open("a") as log:
                log.write(result.stdout)
            if result.returncode:
                raise RuntimeError("Harbor dependency preparation failed; see " + str(work / "prepare.log"))
            return result.stdout

        # Only the portable ACP SDK is shared. Installing BenchFlow in this
        # worker would conflict with Harbor's different LiteLLM requirement.
        requirements = tomllib.loads((source / "pyproject.toml").read_text())["project"]["dependencies"]
        inputs = destination / "requirements.in"
        inputs.write_text("\n".join([*requirements, "agent-client-protocol==0.12.1"]) + "\n")
        run(["pip", "compile", inputs, "--python", python, "--generate-hashes",
             "--output-file", destination / "requirements.txt"])
        run(["build", "--python", python, "--wheel", "--out-dir", destination / "wheels", source])
        run(["venv", "--python", python, destination / "venv"])
        worker_python = destination / "venv/bin/python"
        run(["pip", "sync", "--python", worker_python, "--require-hashes", destination / "requirements.txt"])
        wheels = list((destination / "wheels").glob("harbor-*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one pinned Harbor wheel")
        run(["pip", "install", "--python", worker_python, "--no-deps", wheels[0]])
        run(["pip", "check", "--python", worker_python])
        probe = subprocess.check_output([str(worker_python), "-c", '''
import importlib.metadata as m, json, sys
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
assert m.version('harbor') == '0.25.0'
print(json.dumps({'version': sys.version, 'distributions': sorted((d.metadata['Name'], d.version) for d in m.distributions())}))
'''], cwd=root, env=env, text=True)
        files = [{"path": p.relative_to(destination).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(destination.rglob("*")) if p.is_file() and not p.is_symlink()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        manifest = {"schema_version": 1, "revision": revision, "version": HARBOR_VERSION,
                    "python": {**json.loads(probe), "executable_sha256": sha256(worker_python.resolve())},
                    "worker_python": str(worker_python), "files": files,
                    "uv_sha256": sha256(uv), "created_at": now(), "worker_protocol": 1}
        manifest["receipt_sha256"] = hash_json(manifest)
        write_json(destination / "manifest.json", manifest)
        return {"runtime_ref": str((destination / "manifest.json").relative_to(root)),
                "manifest": manifest}


def probe_schemas(cfg, publication, task_paths):
    root = Path(cfg["project_root"])
    manifest = verify_runtime(publication)
    output = root / "temp/backend-refactor-20261031/schema-probes" / uuid.uuid4().hex
    output.mkdir(parents=True)
    request = {"worker_protocol": 1, "operation": "schema", "task_paths": list(map(str, task_paths)),
               "response_path": str(output / "response.json")}
    write_json(output / "request.json", request)
    result = subprocess.run([manifest["worker_python"], "-m", "runtime.backends.harbor.worker",
                             str(output / "request.json")], cwd=root, capture_output=True, text=True, timeout=60)
    write_json(output / "worker-log.json", {"returncode": result.returncode, "stdout": result.stdout,
                                            "stderr": result.stderr})
    if result.returncode:
        raise ValueError("Harbor rejected task schemas; see " + str(output / "worker-log.json"))
    return read_json(output / "response.json")
