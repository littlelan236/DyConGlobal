import copy
import json

import pytest

from core.io import sha256
from difficulty.evaluation.trajectory_sources import canonical_point, publication_truth


def point_fixture(binding='native_exact'):
    request = dict(prompt_token_ids=[1], output_token_ids=[248069, 2], request_id='native-r',
                   query_id='query', api_attempt_id='api', task_id='task', run_id='run', request_sha256='prompt')
    point = dict(session_id='session', task_id='task', run_id='run', rollout_id='pending-task',
                 query_id='query', api_attempt_id='api', request_id='native-r', request_sha256='prompt',
                 sample_id='sample', anchor='think_close', absolute_position=1, token_id=248069,
                 model_fingerprint='model', status='persisted', sha256='tensor-sha', tensor='raw/sample.npz',
                 recovery=dict(binding=binding, canonical_rollout_id='rollout',
                               canonical_message_index=2, trajectory_sha256='trajectory'))
    kwargs = dict(sid='session', run_id='run', rid='rollout', row=dict(model_fingerprint='model', trajectory_sha256='trajectory'),
                  task='task', actions=[dict(message_index=2)], bound={}, identities={})
    if binding == 'native_exact':
        kwargs['bound'] = {'native-r': dict(message_index=2, request=request, targets={(1, 'think_close')})}
    elif binding == 'canonical_identity_only':
        kwargs['identities'] = {('query', 'api'): [(2, {'id': 'native-r'}, {})]}
    return point, kwargs


@pytest.mark.parametrize('binding', ['native_exact', 'canonical_identity_only'])
def test_prediction_does_not_require_future_tokens_or_ground_truth(binding):
    point, kwargs = point_fixture(binding)
    sample, reason = canonical_point(point, **kwargs)
    assert reason is None
    assert sample['rollout_id'] == 'rollout'
    assert sample['agent_step'] == 0
    assert sample['prediction_binding'] == binding
    assert 'remaining_reasoning_tokens_to_submission' not in sample
    assert sample['hidden_sha256'] == 'tensor-sha'


def test_missing_native_evidence_is_not_promoted_to_native_exact():
    point, kwargs = point_fixture('canonical_identity_only')
    sample, _ = canonical_point(point, **kwargs)
    assert sample['prediction_binding'] == 'canonical_identity_only'
    assert sample['recovery']['binding'] == 'canonical_identity_only'


def test_orphans_and_nonaction_points_are_excluded():
    point, kwargs = point_fixture('canonical_identity_only')
    kwargs['identities'] = {}
    sample, reason = canonical_point(point, **kwargs)
    assert sample is None and reason == 'diagnostic_orphan'
    point, kwargs = point_fixture()
    kwargs['actions'] = [dict(message_index=7)]
    sample, reason = canonical_point(point, **kwargs)
    assert sample is None and reason == 'not_action_response'


@pytest.mark.parametrize('mutation', ['model', 'trajectory', 'position', 'classification'])
def test_identity_conflicts_are_rejected(mutation):
    point, kwargs = point_fixture()
    if mutation == 'model':
        point['model_fingerprint'] = 'other'
    elif mutation == 'trajectory':
        point['recovery']['trajectory_sha256'] = 'other'
    elif mutation == 'position':
        point['token_id'] = 5
    else:
        point['recovery']['binding'] = 'canonical_identity_only'
    with pytest.raises(ValueError):
        canonical_point(point, **kwargs)


def test_truth_is_available_without_hidden_and_censored_truth_stays_missing(tmp_path, monkeypatch):
    import difficulty.evaluation.trajectory_sources as module
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    publication = tmp_path / 'publication'
    request = dict(rollout_id='rollout', run_id='run', is_action_response=True,
                   trajectory_sha256='trajectory', agent_step_index=0, message_index=2,
                   blocks=[dict(close=10, remaining_censored=False, future_reasoning_tokens=17)])
    censored = copy.deepcopy(request)
    censored.update(agent_step_index=1, message_index=4)
    censored['blocks'][0]['remaining_censored'] = True
    write(publication / 'requests.json', {'request1': request, 'request2': censored})
    write(publication / 'receipt.json', dict(files={'requests.json': sha256(publication / 'requests.json')}))
    index = tmp_path / 'index.json'
    write(index, dict(publication_id='publication', source_receipt_sha256=sha256(publication / 'receipt.json')))
    dataset = tmp_path / 'dataset'
    write(dataset / 'source-evidence.json', dict(source_index=str(index), source_index_sha256=sha256(index)))
    monkeypatch.setattr(module, 'publication_directory', lambda *args: publication)
    catalog = {'rollouts': {'rollout': {'trajectory_sha256': 'trajectory'}}}
    actions = {'rollout': [dict(message_index=2), dict(message_index=4)]}
    sources = {}
    truth, _ = publication_truth(tmp_path, 'run', dataset, catalog, actions, sources)
    assert truth == {('rollout', 0): 17}
    assert 'requests.json' in {__import__('pathlib').Path(p).name for p in sources}
    (publication / 'requests.json').write_text('{}')
    with pytest.raises(ValueError, match='publication changed'):
        publication_truth(tmp_path, 'run', dataset, catalog, actions, sources)


def test_curve_masks_allow_prediction_only_and_truth_only_steps(tmp_path, monkeypatch):
    import numpy as np
    from matplotlib.figure import Figure
    from difficulty.evaluation.visualize import plot_trajectory
    figures = []
    monkeypatch.setattr(Figure, 'savefig', lambda figure, *args, **kwargs: figures.append(figure))
    plot_trajectory([dict(step_id=0, target=8, prediction=None, prediction_log1p=None),
                     dict(step_id=1, target=None, prediction=5, prediction_log1p=float(np.log1p(5)))],
                    tmp_path / 'independent.svg', comparison=True)
    truth, prediction = figures[0].axes[0].lines
    assert truth.get_ydata()[0] == 8
    assert np.isnan(truth.get_ydata()[1])
    assert np.isnan(prediction.get_ydata()[0])
    assert prediction.get_ydata()[1] == 5


def test_original_publication_does_not_require_a_repair_wrapper():
    point, kwargs = point_fixture()
    point.pop('recovery')
    sample, reason = canonical_point(point, require_recovery=False, **kwargs)
    assert reason is None and sample['prediction_binding'] == 'native_exact'
    assert sample['rollout_id'] == 'rollout'
    with pytest.raises(ValueError, match='canonical identity'):
        canonical_point(point, **kwargs)


def test_shared_native_suffix_annotation():
    from enrichment.online_trace import annotate_remaining_workloads
    rows = [dict(step_id=0, prompt_token_ids=[1], output_token_ids=[10, 11, 2], is_action_response=True,
                 blocks=[dict(start=0, end=2, close=2, complete=True)]),
            dict(step_id=1, prompt_token_ids=[1], output_token_ids=[12, 13, 14, 2], is_action_response=True,
                 blocks=[dict(start=0, end=3, close=3, complete=True)])]
    for row in rows:
        row['task_id'] = 'task'
    annotate_remaining_workloads({'rollout': rows})
    assert [r['remaining_steps'] for r in rows] == [2, 1]
    assert [r['blocks'][0]['future_reasoning_tokens'] for r in rows] == [3, 0]
    assert not any(r['blocks'][0]['remaining_censored'] for r in rows)
    rows[1]['blocks'][0]['complete'] = False
    annotate_remaining_workloads({'rollout': rows})
    assert all(r['blocks'][0]['remaining_censored'] for r in rows)


def test_trajectory_without_capture_metadata_is_reported_without_inventing_points(tmp_path, monkeypatch):
    import action.datasets.trajectories as readers
    import difficulty.evaluation.trajectory_sources as module
    monkeypatch.setattr(module, 'load_snapshot', lambda *args: {
        'paths': {'nas_root': str(tmp_path)}, 'capture': {'export': {'root': 'hidden'}}})
    monkeypatch.setattr(readers, 'read_messages', lambda *args: [{'role': 'assistant'}])
    monkeypatch.setattr(module, 'event_records', lambda *args: iter([]))
    result, truth, audit = module.extend_original_publications(
        {'project_root': str(tmp_path)}, 'run', {'rollouts': {'r': {}}}, {},
        {'r': [{'message_index': 0}]}, {}, None, 3, {})
    assert result == {} and truth == {}
    assert audit['exclusion_counts'] == {'missing_or_ambiguous_capture_session': 1}
