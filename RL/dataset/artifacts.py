"""RL-specific external roots and publication; global owned_path is unchanged."""
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from core.io import artifact_roots, identifier, read_json, sha256, write_json
from core.locking import file_lock
from RL.artifacts import commit_receipt, read_receipt, verify_receipt

KINDS = {"rl-prepared": "prepared", "rl-reference": "references", "rl-extension": "extensions", "rl-dataset": "datasets", "rl-cache": "cache", "rl-replay": "replays", "rl-probability": "probabilities", "rl-probability-plan": "probability-plans"}
REQUIRED = {
    "rl-prepared": {"manifest.json", "schedule.json", "dataset-binding.json", "inventory.sqlite3"},
    "rl-reference": {"manifest.json"},
    "rl-extension": {"manifest.json"},
    "rl-probability": {"probability.json", "branches.jsonl", "selection.json", "targets.jsonl", "missing.jsonl", "coverage.json", "sources.json", "summary.json", "source-config.yaml", "resolved-config.json"},
    "rl-probability-plan": {"plan.json", "selection.json", "targets.jsonl", "missing.jsonl", "coverage.json", "sources.json", "source-config.yaml", "resolved-config.json"},
    "rl-dataset": {"dataset.json", "source-config.yaml", "resolved-config.json", "selection.json", "rollouts.jsonl", "groups.jsonl", "branches.jsonl", "aliases.jsonl", "exclusions.jsonl", "summary.json", "dependencies.json"},
    "rl-cache": {"cache.json", "groups.jsonl", "branches.jsonl"},
    "rl-replay": {"replay.json", "root-sampling.json", "effective-policies.json", "models.json", "selection.json", "branches.jsonl", "diagnostics.jsonl", "summary.json", "validation.json"},
}


def directory(root, kind, artifact_id):
    raw = Path(root).expanduser()
    if not raw.is_absolute() or ".." in raw.parts or raw == Path(raw.anchor):
        raise ValueError("Artifact root must be absolute and cannot be the filesystem root")
    base = raw.resolve()
    path = base / "RL" / KINDS[kind] / identifier(artifact_id)
    if path.resolve() != path:
        raise ValueError("RL category/ID crosses a symlink or escapes its root")
    return path


def verify_local(path, kind, *, verify_files=True):
    receipt = (verify_receipt if verify_files else read_receipt)(path, kind + "-v1")
    if not REQUIRED[kind] <= receipt["files"].keys():
        raise ValueError(f"Missing required {kind} files")
    return receipt


def locate(cfg, kind, artifact_id, root=None):
    roots = [root] if root is not None else artifact_roots(cfg, kind)
    matches = [directory(r, kind, artifact_id) for r in roots if directory(r, kind, artifact_id).exists()]
    if not matches:
        raise FileNotFoundError(f"{kind} not found: {artifact_id}; specify its physical root")
    for path in matches:
        verify_local(path, kind)
    if len({sha256(p / "receipt.json") for p in matches}) != 1:
        raise ValueError(f"Conflicting {kind} copies across roots")
    return matches[0]


def seal(path, kind, artifact_id, **metadata):
    names = {p.relative_to(path).as_posix() for p in Path(path).rglob("*") if p.is_file()}
    if not REQUIRED[kind] <= names:
        raise ValueError(f"Refusing incomplete {kind} receipt")
    return commit_receipt(path, kind + "-v1", artifact_id=artifact_id, **metadata)


@contextmanager
def publication(root, kind, artifact_id, *, recover=False, signature=None):
    destination = directory(root, kind, artifact_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name("." + artifact_id + ".staging")
    with file_lock(destination.with_name("." + artifact_id + ".lock"), nonblocking=True):
        if destination.exists():
            raise FileExistsError(f"Immutable artifact ID already exists: {destination}")
        if stage.is_symlink():
            raise ValueError("Staging path cannot be a symlink")
        if stage.exists():
            if not recover:
                raise FileExistsError("Interrupted staging exists; choose a new ID or explicit compatible recovery")
            if read_json(stage / "journal.json").get("signature") != signature:
                raise ValueError("Recovery input/runtime/policy signature differs")
        else:
            if recover:
                raise ValueError("No interrupted artifact exists for recovery")
            stage.mkdir()
            write_json(stage / "journal.json", {"signature": signature, "state": "building"})
        yield stage, destination
        verify_local(stage, kind)
        os.rename(stage, destination)
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def require_space(root, size, reserve=2147483648):
    path = Path(root)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < size + reserve:
        raise ValueError(f"Insufficient disk space for {size} bytes plus reserve")


def verify_dependencies(path):
    files = read_json(Path(path) / "dependencies.json")["files"]
    for name, digest in files.items():
        if not Path(name).is_absolute() or not Path(name).is_file() or sha256(name) != digest:
            raise ValueError(f"Frozen source dependency missing or changed: {name}")
    return len(files)


def execute_artifact(cfg, kind, artifact_id, root=None, *, deep=True):
    path = locate(cfg, kind, artifact_id, root)
    if kind == "rl-prepared":
        from RL.grpo.prepared import PreparedReader, summary, verify_prepared
        result = summary(verify_prepared(path) if deep else PreparedReader(path))
    elif kind == "rl-extension":
        from RL.grpo.prepared_service import read_extension
        result = read_extension(path, summary_only=True)
    elif kind == "rl-reference":
        from RL.grpo.reference_artifact import ReferenceReader
        result = ReferenceReader(path).summary()
    elif kind == "rl-dataset":
        from .build import check_dataset
        result = check_dataset(path, level="metadata")
    elif kind == "rl-probability":
        from .probability_artifacts import check_probability
        result = check_probability(path, with_values=True)
    elif kind == "rl-probability-plan":
        from .probability_artifacts import read_plan
        result = read_plan(path)["summary"]
    elif kind == "rl-replay":
        from .replay.artifacts import check_replay
        result = check_replay(path, with_values=True)
    else:
        result = {"status": "verified", "check_level": "local_artifact", "files": len(verify_local(path, kind)["files"])}
    return {**result, "scope": kind, "artifact_id": artifact_id, "path": str(path), "receipt_sha256": sha256(path / "receipt.json")}
