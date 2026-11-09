import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, sha256
from difficulty.evaluation import trajectories as module


@pytest.mark.parametrize('publication', [False, True])
def test_explicit_workspace_view_keeps_failed_trajectory_and_native_gaps(tmp_path, monkeypatch, publication):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path
    target = 'submission_remaining_reasoning_tokens_log1p'
    model_dir = tmp_path / 'difficulty/checkpoints/fit'
    model_path = model_dir / 'model.npz'
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b'fixture')
    write(model_dir / 'receipt.json', {'state': 'complete'})
    write(model_dir / 'attempt.json', dict(mode='fit', dataset_id='training-only'))
    write(model_dir / 'selected.json', {'targets': {target: {'layer': 3, 'model': 'model.npz'}}})
    write(model_dir / 'metrics.json', {target: {}})
    write(model_path.with_suffix('.json'), dict(checkpoint_sha256=sha256(model_path), metadata={
        'target': target, 'model_kind': 'ridge', 'model_fingerprint': 'model', 'layer': 3}))
    rows = {}
    for rid, resolved in [('good', True), ('bad', False)]:
        trajectory = write(tmp_path / f'runs/run/{rid}/trajectory.json', {'messages': [
            {'role': 'assistant', 'content': 'work'}, {'role': 'assistant', 'content': 'finish'}]})
        output = tmp_path / f'runs/run/{rid}/answer.txt'
        output.write_text(rid)
        subject = write(output.parent / 'subject.json', dict(schema_version=1, kind='workspace', outputs=[
            dict(path='answer.txt', sha256=sha256(output))]))
        report = write(output.parent / 'evaluation.json', dict(resolved=resolved))
        rows[rid] = dict(task_key='task', repetition_id='rep-'+rid, resolved=resolved,
            evaluation_status='completed', termination_reason='AgentCompleted',
            selected_evaluation_id=rid, trajectory_path=str(trajectory.relative_to(tmp_path)),
            trajectory_sha256=sha256(trajectory), evaluation_subject={'kind': 'workspace'},
            subject_path=str(subject.relative_to(tmp_path)), subject_sha256=sha256(subject))
    catalog = dict(rollouts=rows, tasks={'task': {'instance_id': 'task'}}, evaluations={
        rid: dict(report_path=f'runs/run/{rid}/evaluation.json',
                  report_sha256=sha256(tmp_path / f'runs/run/{rid}/evaluation.json')) for rid in rows})
    class Evidence:
        def __init__(self, *args):
            pass
        def read(self):
            return copy.deepcopy(catalog)
    monkeypatch.setattr(module, 'RunEvidence', Evidence)
    selection = dict(rollouts=copy.deepcopy(rows), rollout_ids=list(rows))
    monkeypatch.setattr(module, 'final_selection', lambda *args: (selection, {}, [], []))
    monkeypatch.setattr(module, 'select_run', lambda *args: selection)
    monkeypatch.setattr(module, 'artifact_dir', lambda cfg, kind, key, stage: tmp_path / 'difficulty' / stage / key)
    monkeypatch.setattr(module, 'verify_receipt', lambda *args: None)
    monkeypatch.setattr(module, 'load_model', lambda *args: object())
    def native(cfg, run, evidence, predictions, actions, truth, model, layer, sources):
        assert set(actions) == {'good', 'bad'}
        # Step 1 has observed native truth but no hidden. Do not fill its prediction.
        predictions = {(rid, 0): (dict(run_id='run', rollout_id=rid, agent_step=0,
            sample_id=rid, trajectory_sha256=rows[rid]['trajectory_sha256'], message_index=0), np.log1p(7)) for rid in rows}
        observed = {(rid, step): value for rid in rows for step, value in [(0, 5), (1, 0)]}
        return predictions, observed, {}
    monkeypatch.setattr(module, 'extend_original_publications', native)
    if publication:
        import hidden.publish_visualization as visualization
        import difficulty.evaluation.trajectory_sources as trajectory_sources
        truth=[dict(rollout_id=rid,step_id=step,remaining_reasoning_tokens=value,token_truth_available=value is not None)
               for rid in rows for step,value in [(0,None),(1,0)]]
        view={'signature':{'repair_id':'repair'},'coverage':{'sessions':[{'rollout_id':rid} for rid in rows]},
              'canonical-truth':truth}
        monkeypatch.setattr(visualization,'read_visualization_publication',lambda *args,**kw:view)
        def predictions(cfg,run,view,catalog,model,layer,**kwargs):
            actions={rid:[{'message_index':0},{'message_index':1}] for rid in rows}
            values,_,_=native(cfg,run,catalog,{},actions,{},model,layer,{})
            return values,{}, {},actions
        monkeypatch.setattr(trajectory_sources,'predict_visualization',predictions)
    for name in ('difficulty/evaluation/trajectories.py', 'difficulty/evaluation/visualize.py',
                 'difficulty/evaluation/trajectory_sources.py', 'enrichment/online_trace.py', 'core/cli.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture')
    call = getattr(module.visualize_trajectories, '__wrapped__', module.visualize_trajectories)
    result = call({'project_root': str(tmp_path), 'model': {'fingerprint': 'model'}},
        'run', 'view', attempt_id='fit', target=target, rollout_ids=['good', 'bad'],
        publication_id='visual' if publication else None)
    out = tmp_path / 'difficulty/evaluations/view'
    trajectories = json.loads((out / 'trajectories.json').read_text())
    assert {r['resolved'] for r in trajectories} == {True, False}
    assert result['plotted_rollouts'] == 2 and result['resolved_rollouts'] == 1
    with (out / 'curves.csv').open(encoding='utf-8-sig') as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 4
    assert all(r['truth_endpoint'] == 'observed_agent_terminal' for r in curves)
    assert all(r['prediction'] == '' and r['target'] == '0' for r in curves if r['step_id'] == '1')
    if publication:
        assert all(r['prediction'] and r['target']=='' for r in curves if r['step_id']=='0')
    assert all(r['used_in_fit'] == '' and r['split'] == 'not_audited' for r in curves)
    document = (out / 'report.html').read_text()
    assert document.count('<figure>') == 2 and 'resolved=False' in document
    assert 'Observed to agent termination' in (out / 'plots/bad.svg').read_text()
    receipt = json.loads((out / 'receipt.json').read_text())
    assert str(tmp_path / 'runs/run/bad/answer.txt') in receipt['source_files']
    if publication and os.environ.get('DYCON_CAPTURE_TEST_REPORT_ROOT'):
        example=Path(os.environ['DYCON_CAPTURE_TEST_REPORT_ROOT'])/'partial-plot'
        example.mkdir(parents=True,exist_ok=True)
        for source,name in ((out/'plots/bad.svg','unresolved-gap.svg'),(out/'curves.csv','curves.csv'),
                            (out/'summary.json','summary.json')):
            (example/name).write_bytes(source.read_bytes())


@pytest.mark.parametrize('options', [{}, {'rollout_ids': ['r'], 'dataset_id': 'labels'}])
def test_view_modes_reject_ambiguous_sources_before_reading_model(tmp_path, options):
    call = getattr(module.visualize_trajectories, '__wrapped__', module.visualize_trajectories)
    with pytest.raises(ValueError, match='requires --dataset-id|omit dataset/repair/cache'):
        call({'project_root': str(tmp_path)}, 'run', 'view', attempt_id='fit',
             target='submission_remaining_reasoning_tokens_log1p', **options)


def test_visualization_missing_checkpoint_is_unavailable_without_training(tmp_path):
    call=getattr(module.visualize_trajectories,'__wrapped__',module.visualize_trajectories)
    result=call({'project_root':str(tmp_path)},'run','view',attempt_id='missing',
        target='submission_remaining_reasoning_tokens_log1p',rollout_ids=['r'],publication_id='visual')
    assert result['status']=='unavailable' and result['prediction_layer']=='unavailable'
    assert not (tmp_path/'difficulty/checkpoints').exists()


def test_relocated_canonical_run_keeps_owner_boundary(tmp_path):
    from hidden.repair_points import _canonical
    root = tmp_path / 'project'
    moved = tmp_path / 'storage/runs/run'
    moved.mkdir(parents=True)
    (root / 'runs').mkdir(parents=True)
    (root / 'runs/run').symlink_to(moved, target_is_directory=True)
    base = {'project_root': str(root), 'run_storage_root': str(tmp_path / 'storage')}
    (moved / 'resolved-config.json').write_text(json.dumps({**base,
        'selection': {}, 'selection_sha256': hash_json({}), 'config_hash': hash_json(base)}))
    path = moved / 'trajectory.json'
    path.write_text(json.dumps({'messages': []}))
    row = dict(run_id='run', task_key='task', trajectory_path='runs/run/trajectory.json',
               trajectory_sha256=sha256(path))
    catalog = {'tasks': {'task': {'instance_id': 'task'}}}
    proofs = {}
    _canonical(root, 'run', 'session', [('r', row)], catalog, proofs)
    assert str(path) in proofs
    foreign = root / 'runs/foreign'
    foreign.mkdir()
    (foreign / 'trajectory.json').write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='another run'):
        _canonical(root, 'run', 'session', [('r', {**row, 'trajectory_path': 'runs/foreign/trajectory.json'})], catalog, {})


def test_compression_triggers_are_read_from_explicit_acp_events():
    events = [
        {'index': 0, 'event': {'type': 'agent_message', 'text': 'start'}},
        {'index': 1, 'event': {'type': 'tool_call', 'title': 'terminal: compact json'}},
        {'index': 2, 'event': {'type': 'tool_call', 'tool_call_id': 'compress-id',
                               'title': 'compress_context: summarize old events'}},
        {'index': 3, 'event': {'type': 'condenser_triggered', 'agent_step_index': 1,
                               'reason': 'max tokens'}},
        {'index': 4, 'event': {'type': 'context_condensed'}},
        {'index': 5, 'event': {'type': 'tool_call', 'title': 'Decompress PDF streams: $ python3'}},
        {'index': 6, 'event': {'type': 'tool_call', 'title': 'Generate compression report JSON: $ python3'}},
    ]
    actions = [{'tool_calls': [{'id': 'compress-id'}]}, {'tool_calls': [{'id': 'other-id'}]}]
    found = module.compression_triggers({'raw_acp_events': events}, actions)
    assert [row['event_index'] for row in found] == [2, 3, 4]
    assert [row['step_id'] for row in found] == [0, 1, None]


def test_empty_trajectory_renders_explicit_no_steps_panel(tmp_path):
    output = tmp_path / 'empty.svg'
    module.plot_trajectory([], output, comparison=True, title='empty task')
    assert 'No canonical assistant steps' in output.read_text(encoding='utf-8')


def test_completed_condensation_request_maps_to_next_agent_step(tmp_path):
    attempt = tmp_path / 'attempt-000'
    request_dir = attempt / 'requests'
    request_dir.mkdir(parents=True)
    source = attempt / 'openhands' / 'canonical.traj.json'
    source.parent.mkdir()
    source.write_text('{}')
    records = [
        {'request_id': 'agent-2', 'role': 'agent_decision', 'step_id': 2,
         'status': 'completed', 'started_at': '2026-01-01T00:00:02'},
        {'request_id': 'condense', 'role': 'condensation', 'status': 'completed',
         'started_at': '2026-01-01T00:00:01', 'finished_at': '2026-01-01T00:00:01.5'},
        {'request_id': 'failed', 'role': 'condensation', 'status': 'failed',
         'started_at': '2026-01-01T00:00:03'},
    ]
    for record in records:
        (request_dir / f"req-{record['request_id']}.json").write_text(json.dumps(record))
    events, files = module.condensation_requests({'provenance': {'source': str(source)}})
    assert events == [{'request_id': 'condense', 'step_id': 2,
                       'started_at': '2026-01-01T00:00:01',
                       'finished_at': '2026-01-01T00:00:01.5', 'status': 'completed'}]
    assert len(files) == 3
