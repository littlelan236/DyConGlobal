from difficulty.storage import difficulty_path
"""Immutable dataset-owned split artifacts shared by model training attempts."""

import json
import math
from collections import Counter
from pathlib import Path

from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.training.split import make_event_split, make_split, make_task_threeway_split


def _build_split(rollouts, samples, strategy, seed, validation_fraction, test_fraction,
                 validation_tasks, parent=None):
    if any(row.get('benchmark') == 'webarena_verified' for row in rollouts) and strategy != 'task_random_v1':
        raise ValueError('WebArena requires task_random_v1 grouped by intent_template_id to prevent template leakage')
    if strategy == "event_random_v1":
        return make_event_split(samples, validation_fraction, test_fraction, seed)
    if strategy == "task_random_v1":
        return make_task_threeway_split(rollouts, samples, validation_fraction, test_fraction, seed)
    if strategy == "repo_source_quota_v1":
        return make_split(rollouts, samples, validation_tasks, seed)
    if strategy == "inherit_task_split_v1":
        from difficulty.training.split import inherit_task_split
        if parent is None:
            raise ValueError("Inherited task split requires a verified parent split")
        return inherit_task_split(rollouts, samples, parent, validation_tasks, seed)
    raise ValueError(f"Unsupported new split strategy: {strategy}")


def create_split(root, dataset_id, split_id, *, strategy="task_random_v1", seed=42,
                 validation_fraction=0.2, test_fraction=0.2, validation_tasks=48,
                 parent_experiment_id=None, parent_receipt_sha256=None, task_partitions=None, inner_seed=42):
    root = Path(root).resolve()
    dataset_id, split_id = identifier(dataset_id), identifier(split_id)
    if seed < 0 or not math.isfinite(validation_fraction) or not math.isfinite(test_fraction):
        raise ValueError("Invalid split seed or fractions")
    dataset = difficulty_path(root, 'datasets', dataset_id)
    dataset_receipt_path = dataset / "receipt.json"
    dataset_receipt_sha = sha256(dataset_receipt_path)
    dataset_receipt = read_json(dataset_receipt_path)
    if dataset_receipt.get("state") != "complete":
        raise ValueError("Split source dataset is not complete")
    if not {'rollouts.json', 'samples.jsonl'} <= dataset_receipt.get('files', {}).keys():
        raise ValueError('Dataset receipt must bind training rows')
    for name, digest in dataset_receipt.get("files", {}).items():
        if sha256(owned_path(dataset, name)) != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    rollouts = read_json(dataset / "rollouts.json")["rollouts"]
    samples = [json.loads(line) for line in (dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not samples or len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Dataset samples must be nonempty and have unique IDs")
    output = dataset / "splits" / split_id
    lock_root = root / "state/difficulty-split-locks" / dataset_id
    with file_lock(lock_root / f"{split_id}.lock", nonblocking=True):
        if output.exists():
            raise FileExistsError(output)
        parent = None
        if strategy == "inherit_task_split_v1":
            if not parent_experiment_id or not parent_receipt_sha256:
                raise ValueError("Inherited split requires parent ID and receipt SHA")
            parent_experiment_id = identifier(parent_experiment_id)
            parent_dir = difficulty_path(root, 'checkpoints', parent_experiment_id)
            if not parent_dir.is_dir() or sha256(parent_dir / "receipt.json") != parent_receipt_sha256:
                raise ValueError("Parent attempt receipt SHA mismatch")
            proof = read_json(parent_dir / "receipt.json")
            if proof.get("state") != "complete":
                raise ValueError("Parent attempt is not complete")
            if (parent_dir / "split.json").is_file():
                parent = read_json(parent_dir / "split.json")
                parent_split_digest = sha256(parent_dir / "split.json")
            else:
                parent_attempt = read_json(parent_dir / "attempt.json")
                parent_dataset = difficulty_path(root, 'datasets', parent_attempt["dataset_id"])
                parent_split_dir = parent_dataset / "splits" / parent_attempt["split_id"]
                if sha256(parent_split_dir / "receipt.json") != parent_attempt["split_receipt_sha256"]:
                    raise ValueError("Parent split receipt changed")
                parent = read_json(parent_split_dir / "split.json")
                parent_split_digest = sha256(parent_split_dir / "split.json")
            if proof.get("files", {}).get("split.json") and proof["files"]["split.json"] != parent_split_digest:
                raise ValueError("Parent split is not bound to parent receipt")
        assignment_source = None
        if strategy == 'task_holdout_event_v1':
            from difficulty.training.split import make_task_holdout_split
            manifest = None
            if task_partitions is not None:
                source = Path(task_partitions).resolve()
                assignment_source = {'path': str(source), 'sha256': sha256(source)}
                manifest = read_json(source)
            split = make_task_holdout_split(rollouts, samples, test_fraction, seed,
                                            validation_fraction, inner_seed, manifest)
        elif strategy == 'task_manifest_v1':
            if task_partitions is None:
                raise ValueError('task_manifest_v1 requires --task-partitions')
            source = Path(task_partitions).resolve()
            assignment_source = {'path': str(source), 'sha256': sha256(source)}
            split = make_manifest_split(rollouts, samples, read_json(source), seed, validation_fraction)
        else:
            if task_partitions is not None:
                raise ValueError('--task-partitions only applies to task_manifest_v1')
            split = _build_split(rollouts, samples, strategy, seed, validation_fraction,
                                 test_fraction, validation_tasks, parent=parent)
        if assignment_source:
            split['task_partition_source'] = assignment_source
        split["dataset_id"] = dataset_id
        split["dataset_receipt_sha256"] = dataset_receipt_sha
        split.pop("role_by_sample", None)
        split["distributions"] = {}
        for role, ids in split["sample_ids"].items():
            selected_ids = set(ids)
            selected = [row for row in samples if row["sample_id"] in selected_ids]
            split["distributions"][role] = {
                "source_samples": dict(Counter(row.get("source_kind", "unknown") for row in selected)),
                "task_count": len({row["task_group"] for row in selected}),
                "rollout_count": len({row["rollout_id"] for row in selected}),
            }
        output.mkdir(parents=True)
        write_json(output / "split.json", split)
        write_json(output / "receipt.json", {
            "schema_version": 1, "state": "complete", "dataset_id": dataset_id,
            "dataset_receipt_sha256": dataset_receipt_sha, "split_id": split_id,
            "split_sha256": sha256(output / "split.json"), "created_at": now(),
            'source_files': {assignment_source['path']: assignment_source['sha256']} if assignment_source else {},
            "code": {"difficulty/training/split.py": sha256(Path(__file__).with_name("split.py")),
                     "difficulty/training/splits.py": sha256(Path(__file__))},
            "files": {"split.json": sha256(output / "split.json")},
        })
    return {"dataset_id": dataset_id, "split_id": split_id, "path": str(output),
            "strategy": strategy, "task_counts": split["task_counts"],
            "sample_counts": split["sample_counts"], "receipt_sha256": sha256(output / "receipt.json")}


def load_split(root, dataset_id, split_id, dataset_receipt_sha256, samples):
    root = Path(root).resolve()
    dataset_id, split_id = identifier(dataset_id), identifier(split_id)
    dataset = difficulty_path(root, 'datasets', dataset_id)
    dataset_receipt = sha256(dataset / "receipt.json")
    if dataset_receipt != dataset_receipt_sha256:
        raise ValueError("Dataset receipt differs from training configuration")
    path = dataset / "splits" / split_id
    receipt_path = path / "receipt.json"
    receipt = read_json(receipt_path)
    split_path = path / "split.json"
    if (receipt.get("state") != "complete" or receipt.get("dataset_id") != dataset_id
            or receipt.get("dataset_receipt_sha256") != dataset_receipt
            or sha256(split_path) != receipt.get("split_sha256")):
        raise ValueError("Split receipt does not bind the frozen dataset")
    for name, digest in receipt.get("files", {}).items():
        if sha256(owned_path(path, name)) != digest:
            raise ValueError(f"Split checksum mismatch: {name}")
    for source, digest in receipt.get('source_files', {}).items():
        if sha256(source) != digest:
            raise ValueError('Bound task partition source changed')
    split = read_json(split_path)
    if split.get('strategy') == 'task_holdout_event_v1':
        from difficulty.training.split import make_task_holdout_split
        rollouts = read_json(dataset / 'rollouts.json')['rollouts']
        expected = make_task_holdout_split(rollouts, samples, split['outer']['test_fraction'] or 0.2,
            split['seed'], split['inner']['validation_fraction'], split['inner']['seed'], split['outer']['manifest'])
        for field, value in expected.items():
            if split.get(field) != value:
                raise ValueError(f'Nested split protocol mismatch: {field}')
        if (split.get('dataset_id') != dataset_id or split.get('dataset_receipt_sha256') != dataset_receipt
                or receipt.get('split_id') != split_id):
            raise ValueError('Nested split identity mismatch')
        return split, receipt
    sample_ids = {row["sample_id"] for row in samples}
    assigned = [sample_id for ids in split["sample_ids"].values() for sample_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != sample_ids:
        raise ValueError("Split must assign every dataset sample exactly once")
    if split.get("dataset_id") != dataset_id or split.get("dataset_receipt_sha256") != dataset_receipt:
        raise ValueError("Split identity does not match its dataset")
    if receipt.get("split_id") != split_id:
        raise ValueError("Split receipt ID mismatch")
    if split.get("sample_order_sha256") != hash_json([row["sample_id"] for row in samples]):
        raise ValueError("Split sample order changed")
    if set(split["sample_ids"]) not in ({"train", "validation"}, {"train", "validation", "test"}):
        raise ValueError("Invalid split partition names")
    if not all(split["sample_ids"].values()):
        raise ValueError("Split has an empty partition")
    if split["sample_counts"] != {role: len(ids) for role, ids in split["sample_ids"].items()}:
        raise ValueError("Split sample counts mismatch")
    if split["strategy"] != "event_random_v1":
        roles = {sid: role for role, ids in split["sample_ids"].items() for sid in ids}
        by_task = {}
        for sample in samples:
            role = roles[sample["sample_id"]]
            if by_task.setdefault(sample["task_group"], role) != role:
                raise ValueError("Task split leaks samples across partitions")
    return split, receipt


def make_manifest_split(rollouts, samples, manifest, seed, validation_fraction):
    """Bind dataset rows to explicit task holdout; validate inside the train cohort."""
    import hashlib
    from difficulty.datasets.identity import task_descriptor, task_group
    assignments = manifest.get('task_assignments', [])
    by_key = {row['task_key']: row for row in assignments}
    if len(by_key) != len(assignments) or not by_key:
        raise ValueError('Task partition manifest has duplicate or empty identities')
    if any(row.get('partition') not in {'train', 'test', 'val', 'validation'} for row in assignments):
        raise ValueError('Unsupported task partition')
    groups = {}
    roles = {}
    for row in rollouts:
        key = row['task_key']
        assignment = by_key.get(key)
        if assignment is None or assignment['instance_id'] != row['instance_id']:
            raise ValueError('Dataset task is absent or mismatched in task partition manifest')
        group = row.get('task_group') or task_group(row)
        if group in groups and groups[group]['task_key'] != key:
            raise ValueError('Logical task group has conflicting identities')
        groups[group] = row
        role = assignment['partition']
        roles[group] = 'validation' if role == 'val' else role
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be in (0,1)')
    if 'validation' not in roles.values():
        candidates = sorted((g for g in roles if roles[g] == 'train'),
                            key=lambda g: (hashlib.sha256(f'{seed}|{g}'.encode()).hexdigest(), g))
        if len(candidates) < 2:
            raise ValueError('Need at least two training tasks for internal validation')
        count = max(1, min(len(candidates) - 1, round(len(candidates) * validation_fraction)))
        for group in candidates[:count]:
            roles[group] = 'validation'
    role_names = ['train', 'validation'] + (['test'] if 'test' in roles.values() else [])
    ids = {role: [row['sample_id'] for row in samples if roles[row.get('task_group') or task_group(row)] == role]
           for role in role_names}
    if not all(ids.values()) or sum(map(len, ids.values())) != len(samples):
        raise ValueError('Task partition leaves an empty role or unassigned samples')
    hashes = {}
    for row in rollouts:
        role = roles[row.get('task_group') or task_group(row)]
        previous = hashes.setdefault(row['trajectory_sha256'], role)
        if previous != role:
            raise ValueError('Identical trajectory crosses task partitions')
    return {'schema_version': 2, 'strategy': 'task_manifest_v1', 'seed': seed,
            'validation_fraction': validation_fraction,
            'tasks': [{**task_descriptor(row), 'task_key': row['task_key'], 'split': roles[group]}
                      for group, row in sorted(groups.items())],
            'rollouts': [{**task_descriptor(row), 'rollout_id': row['rollout_id'],
                          'trajectory_sha256': row['trajectory_sha256'], 'split': roles[row.get('task_group') or task_group(row)]}
                         for row in rollouts],
            'task_counts': dict(Counter(roles.values())),
            'sample_ids': ids, 'sample_counts': {role: len(values) for role, values in ids.items()},
            'external_test_tasks': sorted(row['instance_id'] for row in assignments
                                          if row['partition'] == 'test' and row['task_key'] not in
                                          {r['task_key'] for r in rollouts}),
            'sample_order_sha256': hash_json([row['sample_id'] for row in samples])}
