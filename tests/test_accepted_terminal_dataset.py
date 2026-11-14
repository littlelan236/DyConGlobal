import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io import read_json, sha256, write_json
from difficulty.datasets.eligibility import eligible_evaluation, positive_reward, verify_positive_report, verify_terminal_report
from difficulty.datasets.resolved import build_resolved_selection
from difficulty.datasets.submission import load_selection
from application.execution.evidence import RunEvidence
from tests.conftest import import_fixture


def test_zero_reward_terminal_dataset_external_storage_and_loader(cfg, monkeypatch, tmp_path):
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
        'raw_rewards': {'reward': 0.0}}
    write_json(report, report_data)
    evaluation.update(report_sha256=sha256(report), outcomes={rid: {
        'status': 'completed', 'resolved': False, 'subject_sha256': row['subject_sha256'],
        'rewards': {'reward': 0.0}}})
    assert not eligible_evaluation(row, evidence, rid, 'resolved_only')
    assert not eligible_evaluation(row, evidence, rid, 'positive_reward')
    assert eligible_evaluation(row, evidence, rid, 'accepted_terminal')
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
    storage = tmp_path/'external-storage'
    result = build_resolved_selection(root, 'source', 'view', 'accepted', [rid],
                                      eligibility='accepted_terminal', storage_root=storage)
    assert result['samples'] == 1
    destination = storage/'difficulty/datasets/accepted'
    assert not (root/'difficulty/datasets/accepted').exists()
    assert read_json(destination / 'rollouts.json')['rollouts'][0]['resolved'] is False
    settings = SimpleNamespace(dataset_id='accepted', dataset_receipt_sha256=sha256(destination/'receipt.json'),
        expected_tasks=1, expected_rollouts=1, expected_samples=1)
    rollouts, samples, _ = load_selection({'project_root': str(root), 'paths': {'artifact_roots': {'difficulty_dataset': str(storage)}}}, settings)
    assert rollouts[0]['resolved'] is False
    assert samples[0]['remaining_reasoning_tokens_to_submission'] == 3
    assert samples[0]['remaining_reasoning_tokens_to_success'] is None
    assert samples[0]['remaining_steps_to_success'] is None
    for key, bad in [('subject_sha256', 'wrong'),
                     ('evaluation_id', 'stale'), ('evaluation_status', 'infra_error')]:
        with pytest.raises(ValueError, match='completed exact official evaluation'):
            verify_terminal_report(row, {**report_data, key: bad})


def test_dataset_root_is_scoped_and_rejects_ambiguous_ids(tmp_path):
    from core.io import artifact_dir, owned_path
    from difficulty.datasets.paths import dataset_directory
    project=tmp_path/'project'
    external=tmp_path/'data'
    project.mkdir()
    external.mkdir()
    cfg={'project_root': str(project), 'paths': {'artifact_roots': {'difficulty_dataset': str(external)}}}
    p=artifact_dir(cfg, 'difficulty', 'new-index', 'datasets')
    assert p == external/'difficulty/datasets/new-index'
    p.mkdir(parents=True)
    assert dataset_directory(cfg,str(p)) == p
    with pytest.raises(ValueError,match='escapes project'):
        owned_path(project,p)
    with pytest.raises(ValueError):
        dataset_directory(cfg,str(external/'unrelated'))
    (project/'difficulty/datasets/new-index').mkdir(parents=True)
    with pytest.raises(ValueError,match='Ambiguous'):
        dataset_directory(cfg,'new-index')
    # Evaluations/checkpoints retain their original project paths.
    assert artifact_dir(cfg,'difficulty','eval','evaluations') == project/'difficulty/evaluations/eval'
