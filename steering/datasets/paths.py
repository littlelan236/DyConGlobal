"""Typed external roots. This deliberately does not relax core.io.owned_path."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from core.io import artifact_roots, identifier, sha256
from core.locking import file_lock

KINDS = {
    "steering_dataset": ("datasets", "steering-dataset"),
    "steering_cache": ("cache", "steering-cache"),
    "steering_analysis": ("outputs", "steering-analysis"),
}


def validate_root(value):
    raw = Path(value).expanduser()
    if not raw.is_absolute() or ".." in raw.parts or raw == Path(raw.anchor):
        raise ValueError(f"Steering root must be an absolute non-root directory: {value}")
    return raw.resolve()


def roots(cfg, kind, explicit=None):
    if kind not in KINDS:
        raise ValueError(f"Unknown steering artifact category: {kind}")
    values = ([explicit] if explicit is not None else []) + artifact_roots(cfg, kind)
    values.append(cfg["project_root"])
    return list(dict.fromkeys(validate_root(value) for value in values))


def directory(root, kind, artifact_id):
    base = validate_root(root)
    path = base / "steering" / KINDS[kind][0] / identifier(artifact_id)
    if not path.resolve().is_relative_to(base):
        raise ValueError("Steering category/ID escapes its registered root")
    return path


def candidates(cfg, kind, artifact_id, explicit=None):
    return [
        directory(root, kind, artifact_id)
        for root in roots(cfg, kind, explicit)
        if directory(root, kind, artifact_id).exists()
    ]


def locate(cfg, kind, artifact_id, explicit=None):
    from steering.artifacts import verify_local

    matches = candidates(cfg, kind, artifact_id, explicit)
    if not matches:
        raise FileNotFoundError(f"{kind} not found: {artifact_id}; specify its storage root")
    digests = set()
    for match in matches:
        verify_local(match, KINDS[kind][1], artifact_id)
        digests.add(sha256(match / "receipt.json"))
    if len(digests) != 1:
        raise ValueError(f"Conflicting {kind} copies across storage roots: {artifact_id}")
    return matches[0]


@contextmanager
def publication(cfg, kind, artifact_id, explicit=None):
    destination = directory(roots(cfg, kind, explicit)[0], kind, artifact_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(destination.parent / f".{artifact_id}.lock", nonblocking=True):
        if destination.exists():
            raise FileExistsError(f"Immutable steering artifact already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.partial-", dir=destination.parent))
        try:
            yield temporary, destination
            if not (temporary / "receipt.json").is_file():
                raise ValueError("Refusing publication without a sealed receipt")
            os.rename(temporary, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def require_space(path, size, reserve=2 * 1024**3):
    if size < 0 or reserve < 0:
        raise ValueError("Disk size/reserve cannot be negative")
    if shutil.disk_usage(path).free < size + reserve:
        raise ValueError(f"Insufficient disk space: need {size} bytes plus {reserve} reserve")
