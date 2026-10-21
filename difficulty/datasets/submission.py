"""Consume a frozen submission selection without changing its historical split."""

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.receipts import verify_receipt


def load_selection(cfg, settings):
    # Training datasets are explicit shared input artifacts bound by the settings receipt.
    # Run scoping changes output locations; it must not reinterpret this dataset ID.
    source = owned_path(cfg["project_root"], f"difficulty/datasets/{settings.dataset_id}")
    receipt_path = source / "receipt.json"
    if sha256(receipt_path) != settings.dataset_receipt_sha256:
        raise ValueError("Frozen dataset receipt SHA mismatch")
    receipt = read_json(receipt_path)
    resolved_only = receipt.get('eligibility') == 'resolved_only'
    if receipt.get("state") != "complete" or not {"rollouts.json", "samples.jsonl"} <= receipt["files"].keys():
        raise ValueError("Selection receipt must bind complete rollouts and samples")
    for name, expected in receipt["files"].items():
        if sha256(owned_path(source, name)) != expected:
            raise ValueError(f"Frozen selection file changed: {name}")
    verify_receipt(cfg["project_root"], source, expected_sha256=settings.dataset_receipt_sha256)
    selection = read_json(source / "rollouts.json")
    samples = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines() if line]
    rollouts = selection["rollouts"]
    task_commits = defaultdict(set)
    for row in rollouts:
        task_commits[(row["repo"], row["instance_id"])].add(row["base_commit"])
    if any(len(commits) != 1 for commits in task_commits.values()):
        raise ValueError("Same logical task has conflicting base commits")
    expected_rollouts = settings.expected_rollouts or settings.expected_tasks
    if (len(task_commits) != settings.expected_tasks or len(rollouts) != expected_rollouts
            or len(samples) != settings.expected_samples):
        raise ValueError("Frozen dataset counts do not match experiment config")
    ids = {r["rollout_id"]: r for r in rollouts}
    if len(ids) != len(rollouts) or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate rollout/sample ID")
    paths = {}
    for row in rollouts:
        if resolved_only or row.get('training_eligibility') == 'resolved_only':
            if row.get('resolved') is not True or row.get('evaluation_status') != 'completed' or not row.get('selected_evaluation_id'):
                raise ValueError('Resolved selection evaluation changed')
            reports = [read_json(path) for path in row.get('source_files', {}) if Path(path).suffix == '.json']
            outcomes = [r.get('outcomes', {}).get(row['rollout_id']) for r in reports if isinstance(r, dict)]
            if not any(o and o.get('resolved') is True and o.get('status') == 'completed'
                       and o.get('patch_sha256') == row['patch_sha256'] for o in outcomes):
                raise ValueError('Resolved selection does not bind a successful exact patch')
        if cfg.get('model', {}).get('fingerprint') and row['model_fingerprint'] != cfg['model']['fingerprint']:
            raise ValueError('Training input hidden model differs from configured model fingerprint')
        if row["submission_status"] != "submitted" or row["canonical_exit_status"] != "Submitted":
            raise ValueError("Selection does not bind a submitted canonical trajectory")
        trajectory = owned_path(cfg["project_root"], row["trajectory_path"])
        if sha256(trajectory) != row["trajectory_sha256"]:
            raise ValueError("Canonical trajectory SHA mismatch")
        data = read_json(trajectory)
        if data.get("info", {}).get("exit_status") != "Submitted":
            raise ValueError("Canonical trajectory is not Submitted")
        steps = [i for i, m in enumerate(data["messages"]) if m.get("role") == "assistant"]
        if len(steps) != row["canonical_valid_assistant_steps"]:
            raise ValueError("Canonical step count changed")
        row["_message_indices"] = steps
        paths[str(owned_path(cfg["project_root"], row["patch_path"]))] = row["patch_sha256"]
        for path, digest in row["source_files"].items():
            if path in paths and paths[path] != digest:
                raise ValueError("Conflicting source hashes")
            paths[path] = digest
    for path, expected in paths.items():
        if sha256(path) != expected:
            raise ValueError(f"Bound source file changed: {path}")
    seen = set()
    used = set()
    for sample in samples:
        row = ids[sample["rollout_id"]]
        step = sample["agent_step"]
        if not 0 <= step < row["canonical_valid_assistant_steps"]:
            raise ValueError("Sample step outside canonical trajectory")
        if sample["message_index"] != row["_message_indices"][step]:
            raise ValueError("Sample message/step mismatch")
        if any(sample[k] != row[k] for k in ("repo", "instance_id", "source_kind", "trajectory_sha256")):
            raise ValueError("Sample/rollout identity mismatch")
        if sample["task_group"] != row["repo"] + "::" + row["instance_id"]:
            raise ValueError("Sample logical task mismatch")
        if (not sample.get("both_targets_available") or sample.get("censored")
                or sample["submission_status"] != "submitted" or sample["endpoint"] != "submission"
                or sample["anchor"] != "think_close" or sample["occurrence"] != 0):
            raise ValueError("Sample is not an eligible submission close")
        for key in ("remaining_steps_to_submission", "remaining_reasoning_tokens_to_submission"):
            value = sample[key]
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError("Missing/nonfinite/negative submission label")
        if sample["remaining_steps_to_submission"] != row["canonical_valid_assistant_steps"] - step:
            raise ValueError("Step label does not match canonical step count")
        identity = (sample["rollout_id"], step)
        if identity in seen:
            raise ValueError("Multiple close samples for the same action")
        if sample["hidden_root"] != row["hidden_root"]:
            raise ValueError("Sample hidden root mismatch")
        seen.add(identity)
        used.add(sample["rollout_id"])
    if used != set(ids):
        raise ValueError("A selected rollout has no usable samples")
    for row in rollouts:
        row.pop("_message_indices")
    return rollouts, samples, {"selection_id": settings.dataset_id,
                               "receipt_sha256": settings.dataset_receipt_sha256,
                               "selection_authority": "frozen dataset receipt",
                               "source_files_verified": len(paths),
                               "sample_order_sha256": hash_json([s["sample_id"] for s in samples])}


def arrays_from_bytes(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Hidden tensor SHA mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def public_features(arrays, sample):
    if "hidden" in arrays:
        x = arrays["hidden"]
        if "array_row" in sample:
            x = x[sample["array_row"]]
    else:
        names = [f"layer_{i}" for i in range(31)] + ["layer_32"]
        if set(arrays) != set(names):
            raise ValueError("Native hidden must contain sites 0..30 and 32")
        x = np.stack([arrays[name] for name in names])
    if x.shape != (32, 4096) or not np.isfinite(x).all():
        raise ValueError("Expected finite [32,4096] public hidden")
    if "array_row" in sample:
        if "step" not in arrays or arrays["step"][sample["array_row"]] != sample["agent_step"]:
            raise ValueError("Legacy NPZ row/step mismatch")
    return x


def iter_features(samples):
    """Read each legacy file/tar shard once, yielding rows in arbitrary I/O order."""
    loose = defaultdict(list)
    shards = defaultdict(list)
    for i, sample in enumerate(samples):
        root = sample["hidden_root"]
        if sample.get("tensor_locator"):
            shards[(root, sample["tensor_locator"]["shard"])].append((i, sample))
        else:
            loose[(root, sample["hidden_path"])].append((i, sample))
    for (root, relative), rows in sorted(loose.items()):
        digests = {r["hidden_sha256"] for _, r in rows}
        if len(digests) != 1:
            raise ValueError("Conflicting tensor file hashes")
        path = owned_path(root, relative)
        arrays = arrays_from_bytes(path.read_bytes(), next(iter(digests)))
        for i, sample in rows:
            yield i, public_features(arrays, sample)
    for (root, relative), rows in sorted(shards.items()):
        path = owned_path(root, relative)
        digests = {r["tensor_locator"]["shard_sha256"] for _, r in rows}
        if len(digests) != 1 or sha256(path) != next(iter(digests)):
            raise ValueError("NAS shard SHA mismatch")
        with tarfile.open(path, "r:") as archive:
            for i, sample in rows:
                loc = sample["tensor_locator"]
                if loc["member_sha256"] != sample["hidden_sha256"]:
                    raise ValueError("Conflicting tensor member hashes")
                member = archive.getmember(loc["member"])
                if not member.isfile():
                    raise ValueError("Hidden member is not a regular file")
                arrays = arrays_from_bytes(archive.extractfile(member).read(), sample["hidden_sha256"])
                yield i, public_features(arrays, sample)


def prepare_cache(cfg, settings, samples):
    """Build checksummed disposable layer memmaps, never reuse partial cache files."""
    identity = hash_json({"source": settings.dataset_receipt_sha256,
                          "sample_order": [s["sample_id"] for s in samples], "schema": 1})
    parent = Path(cfg["project_root"]) / "difficulty/cache"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identity
    with file_lock(parent / (identity + ".lock"), nonblocking=True):
        legacy = Path(cfg['project_root']) / 'state/difficulty-cache' / identity
        # Old cache is a checksummed compatibility input; new writes are domain-owned.
        if not destination.exists() and legacy.exists():
            destination = legacy
        if destination.exists():
            manifest = read_json(destination / "cache.json")
            if manifest["identity"] != identity or set(manifest["files"]) != {
                f"layer-{layer:02d}.f32.npy" for layer in range(32)
            }:
                raise ValueError("Invalid cache manifest")
            for name, digest in manifest["files"].items():
                if sha256(owned_path(destination, name)) != digest:
                    raise ValueError("Feature cache checksum mismatch; remove corrupt cache explicitly")
            return destination
        required = len(samples) * 32 * 4096 * 4 + 256 * 1024**2
        if shutil.disk_usage(parent).free < required + 2 * 1024**3:
            raise ValueError("Insufficient disk for bounded feature cache and reserve")
        temporary = Path(tempfile.mkdtemp(prefix=".ridge-cache-", dir=parent))
        maps = []
        try:
            maps = [np.lib.format.open_memmap(temporary / f"layer-{layer:02d}.f32.npy", mode="w+",
                                             dtype=np.float32, shape=(len(samples), 4096))
                    for layer in range(32)]
            count = 0
            for i, x in iter_features(samples):
                for layer, matrix in enumerate(maps):
                    matrix[i] = x[layer]
                count += 1
                if count % 1000 == 0:
                    print(f"cache: {count}/{len(samples)} checked hidden rows", flush=True)
            if count != len(samples):
                raise ValueError("Incomplete feature cache")
            for matrix in maps:
                matrix.flush()
                matrix._mmap.close()
            maps.clear()
            write_json(temporary / "cache.json", {"identity": identity, "shape": [len(samples), 4096],
                       "public_sites": list(range(32)), "dtype": "float32", "rebuildable": True,
                       "dataset_receipt_sha256": settings.dataset_receipt_sha256,
                       "sample_order_sha256": hash_json([s["sample_id"] for s in samples]),
                       "files": {p.name: sha256(p) for p in sorted(temporary.glob("*.npy"))}})
            os.rename(temporary, destination)
        finally:
            for matrix in maps:
                matrix._mmap.close()
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination
