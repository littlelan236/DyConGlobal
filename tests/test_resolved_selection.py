from application.execution.evidence import RunEvidence
from pathlib import Path

import pytest

from catalog.store import Catalog
from core.io import read_json, sha256, write_json
from difficulty.datasets.resolved import build_resolved_selection
from difficulty.datasets.submission import load_selection
from tests.conftest import import_fixture
from tests.test_ridge_probe import fixture_project  # noqa: F401


def test_resolved_builder_excludes_nonaction_and_binds_evaluation(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=True)
    row = RunEvidence(root, 'source').read()['rollouts'][rid]
    trace = root / 'runs/source/enrichment/online'
    events = [{'sample_id': 'close', 'rollout_id': rid,
        'trajectory_sha256': row['trajectory_sha256'], 'request_id': 'native',
        'selected_evaluation_id': row['selected_evaluation_id'], 'message_index': 2,
        'labels': {'remaining_steps': 2, 'remaining_reasoning_tokens': 3},
        'remaining_steps': 2, 'censored': False, 'anchor': 'think_close',
        'alignment_status': 'exact', 'sha256': 'tensor-sha'},
        {'sample_id': 'format', 'rollout_id': rid,
        'trajectory_sha256': row['trajectory_sha256'], 'request_id': 'format',
        'selected_evaluation_id': row['selected_evaluation_id'],
        'labels': {'remaining_steps': 2, 'remaining_reasoning_tokens': 3}, 'censored': False}]
    write_json(trace / 'index.json', events)
    write_json(trace / 'requests.json', {'native': {'is_action_response': True},
                                       'format': {'is_action_response': False}})
    write_json(trace / 'receipt.json', {'files': {name: sha256(trace / name)
                                               for name in ('index.json', 'requests.json')}})
    index_path = root / 'difficulty/datasets/view/index.json'
    write_json(index_path, {'schema_version': 3, 'dataset_id': 'source',
        'source_receipt_sha256': sha256(trace / 'receipt.json'),
        'hidden_root': str(root / 'nas'), 'samples': events})
    result = build_resolved_selection(root, 'source', 'view', 'resolved')
    assert result['samples'] == 1 and result['resolved_rollouts'] == 1
    assert result['excluded'] == {'not_action_response': 1}
    samples = (root / 'difficulty/datasets/resolved/samples.jsonl').read_text()
    assert 'remaining_steps_to_submission' in samples
    assert read_json(root / 'difficulty/datasets/resolved/receipt.json')['eligibility'] == 'resolved_only'
    index = read_json(index_path)
    index['samples'][0]['selected_evaluation_id'] = 'stale'
    write_json(index_path, index)
    with pytest.raises(ValueError, match='evaluation is stale'):
        build_resolved_selection(root, 'source', 'view', 'stale')


def test_ridge_loader_rechecks_resolved_eligibility(fixture_project):  # noqa: F811
    cfg, settings, dataset = fixture_project
    receipt = read_json(dataset / 'receipt.json')
    receipt['eligibility'] = 'resolved_only'
    write_json(dataset / 'receipt.json', receipt)
    settings = settings.model_copy(update={'dataset_receipt_sha256': sha256(dataset / 'receipt.json')})
    with pytest.raises(ValueError, match='Resolved selection evaluation changed'):
        load_selection(cfg, settings)
