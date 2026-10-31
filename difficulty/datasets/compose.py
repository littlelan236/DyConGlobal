"""Compose immutable training datasets from multiple difficulty datasets."""

from collections import Counter
import json
from pathlib import Path

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.receipts import verify_receipt
from difficulty.datasets.identity import benchmark, task_group, task_key, validate_same_task


def compose_datasets(root, dataset_id, source_dataset_ids):
    root = Path(root).resolve()
    dataset_id = identifier(dataset_id)
    source_dataset_ids = [identifier(value) for value in source_dataset_ids]
    if len(source_dataset_ids) < 2 or len(set(source_dataset_ids)) != len(source_dataset_ids):
        raise ValueError("Composition requires at least two unique source dataset IDs")

    sources, rollouts, samples = [], [], []
    rollout_ids, sample_ids = set(), set()
    eligibilities = set()
    label_modes = set()
    for source_id in source_dataset_ids:
        source = root / "difficulty/datasets" / source_id
        receipt_path = source / "receipt.json"
        receipt, _ = verify_receipt(root, source)
        if not {"rollouts.json", "samples.jsonl"} <= receipt["files"].keys():
            raise ValueError(f"Source receipt does not bind training rows: {source_id}")
        rollout_path, sample_path = source / "rollouts.json", source / "samples.jsonl"
        if not rollout_path.is_file() or not sample_path.is_file():
            raise ValueError(f"Source is not a training dataset: {source_id}")
        source_rollouts = read_json(rollout_path).get("rollouts", [])
        source_samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line]
        metadata = read_json(source / 'dataset.json') if (source / 'dataset.json').is_file() else {}
        label_modes.add(metadata.get('label_normalization', 'raw'))
        if not source_rollouts or not source_samples:
            raise ValueError(f"Source dataset is empty: {source_id}")
        if {row["rollout_id"] for row in source_samples} != {row["rollout_id"] for row in source_rollouts}:
            raise ValueError(f"Source samples/rollouts disagree: {source_id}")
        source_dataset = read_json(source / "dataset.json") if (source / "dataset.json").is_file() else {}
        for row in source_rollouts:
            row.setdefault("benchmark", source_dataset.get("benchmark", "swebench"))
            row.setdefault("agent", source_dataset.get("agent", "unknown"))
            rid = row["rollout_id"]
            if rid in rollout_ids:
                raise ValueError(f"Duplicate rollout across source datasets: {rid}")
            rollout_ids.add(rid)
            rollouts.append({**row, "training_eligibility": row.get(
                "training_eligibility", receipt.get("eligibility", "all_eligible"))})
        for row in source_samples:
            sid = row["sample_id"]
            if sid in sample_ids:
                raise ValueError(f"Duplicate sample across source datasets: {sid}")
            sample_ids.add(sid)
            producer = next(r for r in source_rollouts if r['rollout_id'] == row['rollout_id'])
            if row.get('benchmark', producer['benchmark']) != producer['benchmark']:
                raise ValueError('Sample benchmark differs from its authoritative rollout')
            samples.append({**row, 'benchmark': producer['benchmark'],
                            'agent': row.get('agent', producer.get('agent', 'unknown'))})
        eligibilities.add(receipt.get("eligibility", "all_eligible"))
        run_ids = {row["run_id"] for row in source_rollouts if row.get("run_id")}
        run_ids.update(receipt.get("source_run_ids", []))
        if receipt.get("source_run_id"):
            run_ids.add(receipt["source_run_id"])
        sources.append({"dataset_id": source_id, "receipt_sha256": sha256(receipt_path),
                        "sample_order_sha256": hash_json([row["sample_id"] for row in source_samples]),
                        "source_run_ids": sorted(set(
                            run_ids))})
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample ID")
    if len(label_modes) != 1:
        raise ValueError('Cannot compose datasets with different label normalization modes')
    fingerprints = {row['model_fingerprint'] for row in rollouts if row.get('model_fingerprint')}
    if len(fingerprints) > 1:
        raise ValueError('Cannot compose incompatible hidden model fingerprints')
    tokenizers = {row['tokenizer_fingerprint'] for row in samples if row.get('tokenizer_fingerprint')}
    if len(tokenizers) > 1:
        raise ValueError('Cannot compose incompatible tokenizer fingerprints')
    by_task = {}
    for row in rollouts:
        task = task_key(row)
        by_task.setdefault(task, []).append(row)
    for rows in by_task.values():
        validate_same_task(rows)
    source_benchmarks = sorted({benchmark(row) for row in rollouts})
    source_agents = sorted({row.get("agent", "unknown") for row in rollouts})

    destination = root / "difficulty/datasets" / dataset_id
    with file_lock(root / "state/difficulty-dataset-locks" / f"{dataset_id}.lock", nonblocking=True):
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        write_json(destination / "rollouts.json", {"rollouts": rollouts})
        with (destination / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in samples:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(destination / "source-evidence.json", {
            "schema_version": 1, "composition": "immutable-dataset-union-v1", "sources": sources,
        })
        write_json(destination / "summary.json", {
            "dataset_id": dataset_id, "source_datasets": source_dataset_ids,
            "source_runs": sorted({run for source in sources for run in source["source_run_ids"]}),
            "rollouts": len(rollouts), "tasks": len({task_group(row) for row in rollouts}),
            "samples": len(samples), "eligibility": sorted(eligibilities),
            "source_benchmarks": source_benchmarks, "source_agents": source_agents,
            "source_kinds": dict(Counter(row.get("source_kind", "unknown") for row in samples)),
        })
        write_json(destination / "dataset.json", {
            "schema_version": 1, "dataset_id": dataset_id, "kind": "submission_training_selection",
            "benchmark": source_benchmarks[0] if len(source_benchmarks) == 1 else "mixed",
            "agent": source_agents[0] if len(source_agents) == 1 else "mixed",
            "source_benchmarks": source_benchmarks, "source_agents": source_agents,
            "sample_order_sha256": hash_json([row["sample_id"] for row in samples]),
            "feature_schema": "public-hidden-last-token-v1", "source_datasets": sources,
            "label_fields": ["remaining_steps_to_submission", "remaining_reasoning_tokens_to_submission"],
            "label_normalization": next(iter(label_modes)),
            **({'normalized_label_fields': ['remaining_steps_fraction',
                                             'remaining_reasoning_tokens_fraction']}
               if next(iter(label_modes)) == 'trajectory_fraction' else {}),
        })
        files = {path.name: sha256(path) for path in sorted(destination.iterdir()) if path.is_file()}
        receipt = {"schema_version": 1, "state": "complete", "eligibility":
                   "resolved_only" if eligibilities == {"resolved_only"} else "composed",
                   "source_dataset_ids": source_dataset_ids,
                   "source_run_ids": sorted({run for source in sources for run in source["source_run_ids"]}),
                   "parents": [{"path": f"difficulty/datasets/{source['dataset_id']}",
                                "receipt_sha256": source["receipt_sha256"]} for source in sources],
                   "created_at": now(), "files": files}
        write_json(destination / "receipt.json", receipt)
        return {"dataset_id": dataset_id, "path": str(destination), "rollouts": len(rollouts),
                "tasks": len({row["task_group"] for row in samples}), "samples": len(samples),
                "receipt_sha256": sha256(destination / "receipt.json")}
