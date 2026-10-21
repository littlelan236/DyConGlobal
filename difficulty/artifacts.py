"""Inspect and verify shared difficulty artifacts and their receipt graph."""
import json
from pathlib import Path

from core.io import identifier, owned_path, read_json, sha256
from difficulty.receipts import verify_receipt


def _check_models(directory, attempt, receipt):
    if attempt["mode"] != "fit":
        return
    import numpy as np
    from difficulty.models.ridge_grid import load_model

    for name in receipt["files"]:
        if not name.endswith('.npz') or not name.startswith('models/'):
            continue
        path = owned_path(directory, name)
        model = load_model(path)
        with np.load(path, allow_pickle=False) as archive:
            expected = {
                "dataset_id": attempt["dataset_id"], "split_id": attempt["split_id"],
                "dataset_receipt_sha256": attempt["dataset_receipt_sha256"],
                "split_receipt_sha256": attempt["split_receipt_sha256"],
                "split_sha256": attempt["split_sha256"],
                "model_kind": "ridge", "architecture_id": "linear-ridge-v1",
                "feature_schema": "public-hidden-last-token-v1",
            }
            if any(key not in archive or archive[key].item() != value for key, value in expected.items()):
                raise ValueError(f"Checkpoint identity disagrees with attempt: {name}")
            spec = json.loads(archive['input_spec_json'].item())
            if (spec['width'] != len(model['coef']) or spec['public_sites'] != [int(archive['layer'])]
                    or int(archive['input_width']) != spec['width']):
                raise ValueError(f"Checkpoint input structure mismatch: {name}")


def verify(root, scope, artifact_id, *, dataset_id=None):
    root = Path(root).resolve()
    artifact_id = identifier(artifact_id)
    if scope == "difficulty-dataset":
        directory = root / "difficulty/datasets" / artifact_id
        receipt, checked = verify_receipt(root, directory)
    elif scope == "difficulty-split":
        if dataset_id is None:
            raise ValueError("difficulty-split verification requires --dataset-id")
        dataset_id = identifier(dataset_id)
        dataset = root / "difficulty/datasets" / dataset_id
        _, checked = verify_receipt(root, dataset)
        samples = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines() if line]
        from difficulty.training.splits import load_split
        directory = dataset / "splits" / artifact_id
        _, receipt = load_split(root, dataset_id, artifact_id, sha256(dataset / 'receipt.json'), samples)
        _, split_checked = verify_receipt(root, directory)
        checked += split_checked
    elif scope == "difficulty-attempt":
        directory = root / "difficulty/checkpoints" / artifact_id
        receipt, checked = verify_receipt(root, directory)
        attempt = read_json(directory / "attempt.json")
        if attempt.get("attempt_id") != artifact_id:
            raise ValueError("Training attempt identity mismatch")
        dataset_id = identifier(attempt["dataset_id"])
        split_id = identifier(attempt["split_id"])
        dataset = root / "difficulty/datasets" / dataset_id
        _, parent_checked = verify_receipt(root, dataset, expected_sha256=attempt['dataset_receipt_sha256'])
        checked += parent_checked
        split_dir = dataset / "splits" / split_id
        _, parent_checked = verify_receipt(root, split_dir, expected_sha256=attempt['split_receipt_sha256'])
        checked += parent_checked
        if sha256(split_dir / 'split.json') != attempt['split_sha256']:
            raise ValueError('Attempt split assignment changed')
        for key in ['dataset_id', 'split_id', 'dataset_receipt_sha256', 'split_receipt_sha256', 'split_sha256', 'mode']:
            if receipt.get(key) != attempt.get(key):
                raise ValueError(f'Attempt/receipt identity differs: {key}')
        if read_json(directory / 'experiment.json') != attempt:
            raise ValueError('Compatibility experiment and attempt disagree')
        _check_models(directory, attempt, receipt)
    elif scope in {"difficulty-prediction", "difficulty-evaluation"}:
        category = "predictions" if scope == "difficulty-prediction" else "evaluations"
        directory = root / "difficulty" / category / artifact_id
        receipt, checked = verify_receipt(root, directory)
    else:
        raise ValueError(f"Unknown difficulty artifact scope: {scope}")
    return {"scope": scope, "artifact_id": artifact_id, "path": str(directory),
            "verified": True, "files_checked": checked, "receipt_sha256": sha256(directory / 'receipt.json')}


def show(root, scope, artifact_id, *, dataset_id=None):
    result = verify(root, scope, artifact_id, dataset_id=dataset_id)
    directory = Path(result["path"])
    return {**result, "files": [
        {"path": str(path.relative_to(directory)), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*")) if path.is_file()
    ]}
