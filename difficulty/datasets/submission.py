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

from application.execution.subject import artifact_kinds, subject_digest, verify_subject
from core.io import cache_roots, hash_json, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.datasets.identity import task_group, task_key, validate_same_task
from difficulty.receipts import verify_receipt
from difficulty.datasets.eligibility import verify_positive_report


def load_selection(cfg, settings):
    # Training datasets are explicit shared input artifacts bound by the settings receipt.
    # Run scoping changes output locations; it must not reinterpret this dataset ID.
    from difficulty.datasets.paths import dataset_directory
    source = dataset_directory(cfg, settings.dataset_id)
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
    dataset_meta = read_json(source / "dataset.json") if (source / "dataset.json").is_file() else {}
    fraction_labels = dataset_meta.get('label_normalization') == 'trajectory_fraction'
    samples = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines() if line]
    rollouts = selection["rollouts"]
    task_rows = defaultdict(list)
    for row in rollouts:
        task_rows[task_key(row)].append(row)
    for rows in task_rows.values():
        validate_same_task(rows)
    expected_rollouts = settings.expected_rollouts or settings.expected_tasks
    if (len(task_rows) != settings.expected_tasks or len(rollouts) != expected_rollouts
            or len(samples) != settings.expected_samples):
        raise ValueError("Frozen dataset counts do not match experiment config")
    ids = {r["rollout_id"]: r for r in rollouts}
    if len(ids) != len(rollouts) or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate rollout/sample ID")
    paths = {}
    for row in rollouts:
        is_workspace = row.get("evaluation_subject", {}).get("kind") == "workspace"
        if resolved_only or row.get('training_eligibility') == 'resolved_only':
            if row.get('resolved') is not True or row.get('evaluation_status') != 'completed' or not row.get('selected_evaluation_id'):
                raise ValueError('Resolved selection evaluation changed')
            reports = [read_json(path) for path in row.get('source_files', {}) if Path(path).suffix == '.json']
            if is_workspace:
                if not any(r.get('resolved') is True and r.get('evaluation_status') == 'completed'
                           and r.get('subject_sha256') == row.get('subject_sha256')
                           for r in reports if isinstance(r, dict)):
                    raise ValueError('Resolved selection does not bind a successful workspace subject')
            else:
                outcomes = [r.get('outcomes', {}).get(row['rollout_id']) for r in reports if isinstance(r, dict)]
                digest_key, digest = subject_digest(row)
                if not any(o and o.get('resolved') is True and o.get('status') == 'completed'
                           and o.get(digest_key) == digest for o in outcomes):
                    raise ValueError('Resolved selection does not bind a successful exact evaluation subject')
        if receipt.get('eligibility') == 'positive_reward' or row.get('training_eligibility') == 'positive_reward':
            if row.get('evaluation_subject', {}).get('kind') != 'workspace':
                raise ValueError('Positive reward training currently requires workspace evaluation subjects')
            reports = [read_json(path) for path in row.get('source_files', {}) if Path(path).suffix == '.json']
            if len(reports) != 1:
                raise ValueError('Positive reward selection requires one bound official evaluation report')
            verify_positive_report(row, reports[0])
        if receipt.get('eligibility') == 'accepted_terminal' or row.get('training_eligibility') == 'accepted_terminal':
            from difficulty.datasets.eligibility import verify_terminal_report
            reports = [read_json(path) for path in row.get('source_files', {}) if Path(path).suffix == '.json']
            if len(reports) != 1:
                raise ValueError('Accepted terminal selection requires one bound official evaluation report')
            verify_terminal_report(row, reports[0])
        if cfg.get('model', {}).get('fingerprint') and row['model_fingerprint'] != cfg['model']['fingerprint']:
            raise ValueError('Training input hidden model differs from configured model fingerprint')
        expected_terminal = "AgentCompleted" if is_workspace else "Submitted"
        if row["submission_status"] != "submitted" or row["canonical_exit_status"] != expected_terminal:
            raise ValueError("Selection does not bind a successful benchmark terminal trajectory" if is_workspace
                             else "Selection does not bind a submitted canonical trajectory")
        trajectory = owned_path(cfg["project_root"], row["trajectory_path"])
        if sha256(trajectory) != row["trajectory_sha256"]:
            raise ValueError("Canonical trajectory SHA mismatch")
        data = read_json(trajectory)
        if data.get("info", {}).get("exit_status") != expected_terminal:
            raise ValueError("Canonical trajectory has the wrong benchmark terminal state")
        steps = [i for i, m in enumerate(data["messages"]) if m.get("role") == "assistant"]
        if len(steps) != row["canonical_valid_assistant_steps"]:
            raise ValueError("Canonical step count changed")
        row["_message_indices"] = steps
        if 'subject' in artifact_kinds(row):
            verify_subject(cfg["project_root"], row)
            if row.get('evaluation_subject', {}).get('kind') == 'web_trace':
                manifest = read_json(owned_path(cfg['project_root'], row['subject_path']))
                if not (resolved_only or row.get('training_eligibility') == 'resolved_only') or not manifest.get('complete'):
                    raise ValueError('WebArena training requires resolved-only complete web trace subjects')
        else:
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
        if any(sample.get(k) != row.get(k) for k in ("benchmark", "revision", "repo", "instance_id",
                                                       "source_kind", "trajectory_sha256", "intent_template_id")):
            raise ValueError("Sample/rollout identity mismatch")
        if sample["task_group"] != task_group(row):
            raise ValueError("Sample logical task mismatch")
        expected_endpoint = "agent_terminal" if row.get("evaluation_subject", {}).get("kind") == "workspace" else "submission"
        expected_status = "completed" if row.get("evaluation_subject", {}).get("kind") == "workspace" else "submitted"
        if (not sample.get("both_targets_available") or sample.get("censored")
                or sample["submission_status"] != expected_status or sample["endpoint"] != expected_endpoint
                or sample["anchor"] != "think_close" or sample["occurrence"] != 0):
            raise ValueError("Sample is not an eligible successful terminal close")
        for key in ("remaining_steps_to_submission", "remaining_reasoning_tokens_to_submission"):
            value = sample[key]
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError("Missing/nonfinite/negative submission label")
        if sample["remaining_steps_to_submission"] != row["canonical_valid_assistant_steps"] - step:
            raise ValueError("Step label does not match canonical step count")
        if row.get('training_eligibility') == 'accepted_terminal' and row.get('resolved') is not True:
            sample['remaining_steps_to_success'] = None
            sample['remaining_reasoning_tokens_to_success'] = None
        else:
            sample.setdefault("remaining_steps_to_success", sample["remaining_steps_to_submission"])
            sample.setdefault("remaining_reasoning_tokens_to_success", sample["remaining_reasoning_tokens_to_submission"])
        if fraction_labels:
            for key in ("remaining_steps_fraction", "remaining_reasoning_tokens_fraction"):
                value = sample.get(key)
                if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Missing/nonfinite/out-of-range normalized submission label")
            expected = (row["canonical_valid_assistant_steps"] - step) / row["canonical_valid_assistant_steps"]
            if not np.isclose(sample["remaining_steps_fraction"], expected):
                raise ValueError("Normalized step label does not match canonical step count")
            sample.setdefault("remaining_steps_fraction_to_success", sample["remaining_steps_fraction"])
            total = row.get('total_reasoning_tokens')
            if type(total) is not int or total <= 0 or not np.isclose(
                    sample['remaining_reasoning_tokens_fraction'],
                    sample['remaining_reasoning_tokens_to_submission'] / total):
                raise ValueError('Normalized reasoning label does not match trajectory denominator')
            sample.setdefault("remaining_reasoning_tokens_fraction_to_success",
                              sample["remaining_reasoning_tokens_fraction"])
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
    contract = feature_contract(samples)
    if len({row.get('model_fingerprint') for row in rollouts}) != 1:
        raise ValueError('Dataset mixes hidden model fingerprints')
    for sample in samples:
        sample['_native_sites'], sample['_hidden_width'] = contract['native_sites'], contract['width']
    return rollouts, samples, {'feature_contract': contract, "selection_id": settings.dataset_id,
                               "receipt_sha256": settings.dataset_receipt_sha256,
                               "selection_authority": "frozen dataset receipt",
                               "source_files_verified": len(paths),
                               "sample_order_sha256": hash_json([s["sample_id"] for s in samples])}


def arrays_from_bytes(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Hidden tensor SHA mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def feature_contract(samples):
    contracts = []
    for sample in samples:
        shapes = sample.get('storage_shapes') or {}
        names = sorted((k for k in shapes if k.startswith('layer_')), key=lambda k: int(k[6:]))
        if names:
            widths = {tuple(shapes[k]) for k in names}
            if len(widths) != 1 or len(next(iter(widths))) != 1:
                raise ValueError('Hidden sites must have one common vector width')
            sites, width = [int(k[6:]) for k in names], next(iter(widths))[0]
        else:
            sites = sample.get('native_hidden_sites', list(range(31)) + [32])
            width = sample.get('hidden_width', 4096)
        if type(width) is not int or width <= 0 or not sites or len(set(sites)) != len(sites):
            raise ValueError('Invalid hidden feature contract')
        contracts.append((tuple(sites), width))
    if not contracts or len(set(contracts)) != 1:
        raise ValueError('Dataset mixes incompatible hidden feature contracts')
    sites, width = contracts[0]
    return {'native_sites': list(sites), 'public_sites': list(range(len(sites))), 'width': width}


def public_features(arrays, sample):
    if "hidden" in arrays:
        x = arrays["hidden"]
        if "array_row" in sample:
            x = x[sample["array_row"]]
    else:
        sites = sample.get('_native_sites') or sample.get('native_hidden_sites')
        names = ([f'layer_{i}' for i in sites] if sites is not None else
                 sorted(arrays, key=lambda k: int(k.removeprefix('layer_'))))
        if set(arrays) != set(names) or any(not k.startswith('layer_') for k in names):
            raise ValueError('Native hidden site set differs from dataset contract')
        x = np.stack([arrays[name] for name in names])
    sites = sample.get('_native_sites') or sample.get('native_hidden_sites')
    width = sample.get('_hidden_width', sample.get('hidden_width'))
    expected = (len(sites), width) if sites is not None and width is not None else x.shape
    if x.ndim != 2 or x.shape != expected or not np.isfinite(x).all():
        raise ValueError('Expected finite hidden matching dataset feature contract')
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
    contract = feature_contract(samples)
    sites, width = contract['native_sites'], contract['width']
    layer_count = len(sites)
    for sample in samples:
        sample['_native_sites'], sample['_hidden_width'] = sites, width
    identity = hash_json({'feature_contract': contract, "source": settings.dataset_receipt_sha256,
                          "sample_order": [s["sample_id"] for s in samples], "schema": 2})
    roots = cache_roots(cfg)
    parent = roots[0]
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identity
    with file_lock(parent / (identity + ".lock"), nonblocking=True):
        legacy = Path(cfg['project_root']) / 'state/difficulty-cache' / identity
        candidates = [root / identity for root in roots if (root / identity).exists()]
        if not candidates and legacy.exists():
            candidates = [legacy]
        if candidates:
            destination = candidates[0]
            manifest = read_json(destination / "cache.json")
            if manifest["identity"] != identity or set(manifest["files"]) != {
                f"layer-{layer:02d}.f32.npy" for layer in range(layer_count)
            }:
                raise ValueError("Invalid cache manifest")
            for name, digest in manifest["files"].items():
                if sha256(owned_path(destination, name)) != digest:
                    raise ValueError("Feature cache checksum mismatch; remove corrupt cache explicitly")
            for candidate in candidates[1:]:
                other = read_json(candidate / "cache.json")
                if other != manifest:
                    raise ValueError("Conflicting feature caches found across storage roots")
                for name, digest in other["files"].items():
                    if sha256(owned_path(candidate, name)) != digest:
                        raise ValueError("Feature cache checksum mismatch; remove corrupt cache explicitly")
            return destination
        required = len(samples) * layer_count * width * 4 + 256 * 1024**2
        if shutil.disk_usage(parent).free < required + 2 * 1024**3:
            raise ValueError("Insufficient disk for bounded feature cache and reserve")
        temporary = Path(tempfile.mkdtemp(prefix=".ridge-cache-", dir=parent))
        maps = []
        try:
            maps = [np.lib.format.open_memmap(temporary / f"layer-{layer:02d}.f32.npy", mode="w+",
                                             dtype=np.float32, shape=(len(samples), width))
                    for layer in range(layer_count)]
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
            write_json(temporary / "cache.json", {"identity": identity, "shape": [len(samples), width],
                       "public_sites": list(range(layer_count)), "native_sites": sites, "feature_contract": contract, "dtype": "float32", "rebuildable": True,
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
