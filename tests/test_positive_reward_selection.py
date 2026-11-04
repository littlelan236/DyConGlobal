import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io import read_json, sha256, write_json
from difficulty.datasets.eligibility import eligible_evaluation, positive_reward, verify_positive_report
from difficulty.datasets.resolved import build_resolved_selection
from difficulty.datasets.submission import load_selection
from application.execution.evidence import RunEvidence
from tests.conftest import import_fixture


@pytest.mark.parametrize('value,expected', [(1, True), (.25, True), (0, False), (-1, False),
    (None, False), (True, False), (float('nan'), False), (float('inf'), False)])
def test_positive_reward_requires_finite_numeric_score(value, expected):
    assert positive_reward({'reward': value}) == expected


def test_partial_reward_builder_and_loader_preserve_unresolved(cfg, monkeypatch):
    import difficulty.datasets.resolved as builder
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=True)
    evidence = copy.deepcopy(RunEvidence(root, 'source').read())
    row = evidence['rollouts'][rid]
    task = evidence['tasks'][row['task_key']]
    task.update(benchmark='skillsbench', revision='f'*40, instance={'package_sha256': 'a'*64})
    row.update(benchmark='skillsbench', revision='f'*40, evaluation_subject={'kind': 'workspace'},
               task_package_sha256='a'*64, resolved=False)
    trajectory = root / row['trajectory_path']
    data = read_json(trajectory)
    data['info']['exit_status'] = 'AgentCompleted'
    write_json(trajectory, data)
    row['trajectory_sha256'] = sha256(trajectory)
    subject = root / 'subject.json'
    write_json(subject, {'schema_version': 1, 'kind': 'workspace', 'outputs': []})
    row.update(subject_path=str(subject), subject_sha256=sha256(subject))
    evaluation = evidence['evaluations'][row['selected_evaluation_id']]
    report = root / evaluation['report_path']
    report_data = {'evaluation_id': row['selected_evaluation_id'], 'attempt_id': row.get('attempt_id'),
        'evaluation_status': 'completed', 'resolved': False, 'subject_sha256': row['subject_sha256'],
        'raw_rewards': {'reward': .25}}
    write_json(report, report_data)
    evaluation.update(report_sha256=sha256(report), outcomes={rid: {
        'status': 'completed', 'resolved': False, 'subject_sha256': row['subject_sha256'],
        'rewards': {'reward': .25}}})
    assert not eligible_evaluation(row, evidence, rid, 'resolved_only')
    assert eligible_evaluation(row, evidence, rid, 'positive_reward')
    monkeypatch.setattr(builder, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(builder, 'select_run', lambda *a: {'rollout_ids': [rid]})
    trace = root / 'runs/source/enrichment/online'
    event = {'sample_id': 'close', 'rollout_id': rid, 'trajectory_sha256': row['trajectory_sha256'],
        'request_id': 'native', 'selected_evaluation_id': row['selected_evaluation_id'],
        'message_index': 2, 'labels': {'remaining_steps': None, 'remaining_reasoning_tokens': None},
        'remaining_steps': 2, 'remaining_reasoning_tokens_after': 3, 'censored': False,
        'anchor': 'think_close', 'alignment_status': 'exact', 'sha256': 'tensor-sha'}
    write_json(trace / 'index.json', [event])
    write_json(trace / 'requests.json', {'native': {'is_action_response': True}})
    write_json(trace / 'receipt.json', {'files': {name: sha256(trace / name)
                                               for name in ('index.json', 'requests.json')}})
    write_json(root / 'difficulty/datasets/view/index.json', {'schema_version': 3,
        'dataset_id': 'source', 'source_receipt_sha256': sha256(trace / 'receipt.json'),
        'hidden_root': str(root / 'nas'), 'samples': [event]})
    with pytest.raises(ValueError, match='No resolved'):
        build_resolved_selection(root, 'source', 'view', 'default')
    result = build_resolved_selection(root, 'source', 'view', 'positive', eligibility='positive_reward')
    assert result['samples'] == 1
    destination = root / 'difficulty/datasets/positive'
    assert read_json(destination / 'rollouts.json')['rollouts'][0]['resolved'] is False
    settings = SimpleNamespace(dataset_id='positive', dataset_receipt_sha256=sha256(destination/'receipt.json'),
        expected_tasks=1, expected_rollouts=1, expected_samples=1)
    rollouts, samples, _ = load_selection({'project_root': str(root)}, settings)
    assert rollouts[0]['resolved'] is False
    assert samples[0]['remaining_reasoning_tokens_to_submission'] == 3
    for key, bad in [('raw_rewards', {'reward': 0}), ('subject_sha256', 'wrong'),
                     ('evaluation_id', 'stale'), ('evaluation_status', 'infra_error')]:
        with pytest.raises(ValueError, match='positive official evaluation'):
            verify_positive_report(row, {**report_data, key: bad})
