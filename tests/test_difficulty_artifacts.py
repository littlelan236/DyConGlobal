import json
from pathlib import Path

import pytest

from core.io import artifact_dir, read_json, sha256, write_json
from difficulty.artifacts import verify
from difficulty.datasets.compose import compose_datasets
from difficulty.training.splits import create_split
from tests.test_ridge_probe import fixture_project  # noqa: F401


def _dataset(root, dataset_id, prefix):
    directory = root / "difficulty/datasets" / dataset_id
    directory.mkdir(parents=True)
    rollouts, samples = [], []
    for index in range(3):
        instance_id = f"{prefix}-{index}"
        task_group = f"org/repo::{instance_id}"
        rollout_id = f"rollout-{prefix}-{index}"
        rollouts.append({"rollout_id": rollout_id, "repo": "org/repo", "instance_id": instance_id,
                         "base_commit": "abc", "source_kind": "online_close",
                         "trajectory_sha256": f"{prefix}{index}".encode().hex().ljust(64, "0")})
        samples.append({"sample_id": f"sample-{prefix}-{index}", "rollout_id": rollout_id,
                        "task_group": task_group, "repo": "org/repo", "instance_id": instance_id,
                        "source_kind": "online_close"})
    write_json(directory / "rollouts.json", {"rollouts": rollouts})
    (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    write_json(directory / "source-evidence.json", {"source_run_id": f"source-{prefix}"})
    write_json(directory / "summary.json", {"samples": len(samples)})
    write_json(directory / "receipt.json", {
        "state": "complete", "eligibility": "resolved_only", "source_run_id": f"source-{prefix}",
        "files": {name: sha256(directory / name) for name in
                  ["rollouts.json", "samples.jsonl", "source-evidence.json", "summary.json"]},
    })


def test_difficulty_artifacts_are_project_owned_even_with_run_context(tmp_path):
    cfg = {"project_root": str(tmp_path), "run_id": "source-run"}
    assert artifact_dir(cfg, "difficulty", "dataset", "datasets") == tmp_path / "difficulty/datasets/dataset"
    assert artifact_dir(cfg, "difficulty", "attempt", "checkpoints") == tmp_path / "difficulty/checkpoints/attempt"
    assert artifact_dir(cfg, "action", "analysis", "outputs") == tmp_path / "runs/source-run/action/outputs/analysis"


def test_compose_dataset_split_and_receipt_chain(tmp_path):
    _dataset(tmp_path, "batch-a", "a")
    _dataset(tmp_path, "batch-b", "b")
    result = compose_datasets(tmp_path, "combined", ["batch-a", "batch-b"])
    assert result["samples"] == 6
    dataset = tmp_path / "difficulty/datasets/combined"
    source = read_json(dataset / "source-evidence.json")
    assert {row["source_run_ids"][0] for row in source["sources"]} == {"source-a", "source-b"}
    split_result = create_split(tmp_path, "combined", "task-split", strategy="task_random_v1", seed=42)
    split = read_json(Path(split_result["path"]) / "split.json")
    assert split["task_counts"] == {"train": 4, "validation": 1, "test": 1}
    assert verify(tmp_path, "difficulty-dataset", "combined")["verified"]
    assert verify(tmp_path, "difficulty-split", "task-split", dataset_id="combined")["verified"]


def test_split_binds_exact_dataset_and_duplicate_samples_are_rejected(tmp_path):
    _dataset(tmp_path, "batch-a", "a")
    _dataset(tmp_path, "batch-b", "b")
    compose_datasets(tmp_path, "combined", ["batch-a", "batch-b"])
    sample_path = tmp_path / "difficulty/datasets/combined/samples.jsonl"
    with sample_path.open("a", encoding="utf-8") as stream:
        stream.write(sample_path.read_text(encoding="utf-8").splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        create_split(tmp_path, "combined", "broken", strategy="task_random_v1")


def test_cli_composition_and_two_independent_splits(cfg, capsys):
    from core.cli import main
    root = Path(cfg['project_root'])
    _dataset(root, 'batch-a', 'a')
    _dataset(root, 'batch-b', 'b')
    main(['--config', cfg['config_source'], 'difficulty-dataset', 'compose', 'combined',
          '--source-dataset', 'batch-a', '--source-dataset', 'batch-b'])
    assert json.loads(capsys.readouterr().out)['samples'] == 6
    receipt_sha = sha256(root / 'difficulty/datasets/combined/receipt.json')
    for strategy, split_id in [('task_random_v1', 'task'), ('event_random_v1', 'event')]:
        main(['--config', cfg['config_source'], 'difficulty-split', 'combined', split_id, '--strategy', strategy])
        assert json.loads(capsys.readouterr().out)['sample_counts'] == {'train': 4, 'validation': 1, 'test': 1}
        main(['--config', cfg['config_source'], 'artifact', 'verify', split_id,
              '--scope', 'difficulty-split', '--dataset-id', 'combined'])
        assert json.loads(capsys.readouterr().out)['verified']
    assert sha256(root / 'difficulty/datasets/combined/receipt.json') == receipt_sha
    source = root / 'difficulty/datasets/batch-a/samples.jsonl'
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(root, 'difficulty-dataset', 'combined')


def test_run_free_cli_fit_verify_and_publish(fixture_project, monkeypatch, capsys):
    from core.cli import main
    from core.io import hash_json
    from runtime.serving.predictors import publish_predictor
    from runtime.serving.parallel import artifact_contract
    cfg, settings, dataset = fixture_project
    root = Path(cfg['project_root'])
    cfg['model'] = {'fingerprint': 'model'}
    cfg['difficulty']['ridge_probe'].update(layers=[3], alphas=[1.0], split_strategy='event_random_v1')
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    create_split(root, 'fixture', 'shared', strategy='event_random_v1')
    monkeypatch.setattr('core.cli.resolve', lambda *args: cfg)
    monkeypatch.setattr('difficulty.training.ridge_probe.draw_plots', lambda *args: None)
    main(['difficulty-ridge', 'small-fit', '--split-id', 'shared'])
    output_text = capsys.readouterr().out
    assert json.loads(output_text[output_text.index('{'):])['models'] == 2
    assert not (root / 'runs/small-fit').exists()
    main(['artifact', 'verify', 'small-fit', '--scope', 'difficulty-attempt'])
    assert json.loads(capsys.readouterr().out)['verified']
    output = root / 'difficulty/checkpoints/small-fit'
    result = publish_predictor(cfg, 'fixture-published', experiment=output)
    assert artifact_contract(result['manifest'])['sites'] == [3]
    metrics = read_json(output / 'metrics.json')
    target = 'submission_remaining_reasoning_tokens_log1p'
    with (output / metrics[target]['refit_model']).open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(root, 'difficulty-attempt', 'small-fit')


def test_legacy_index_requires_explicit_path(tmp_path):
    from difficulty.datasets.paths import dataset_directory, source_directory
    cfg = {'project_root': str(tmp_path), 'run_id': 'unrelated'}
    legacy = tmp_path / 'runs/source/difficulty/datasets/view'
    legacy.mkdir(parents=True)
    assert dataset_directory(cfg, 'view') == tmp_path / 'difficulty/datasets/view'
    assert dataset_directory(cfg, 'runs/source/difficulty/datasets/view') == legacy
    assert source_directory(cfg, {'dataset_id': 'source'}, legacy, 'enrichment') == tmp_path / 'runs/source/enrichment/offline'
