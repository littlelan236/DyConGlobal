"""Append executed repetitions without discarding failures or duplicating imports."""
from pathlib import Path

import pytest

from application.execution.evidence import RunEvidence
from application.execution.artifact import verify
from application.execution.merge import merge_runs, sample_state
from application.execution.run_plan import create_run, load_plan
from application.execution.status import run_status
from catalog.store import Catalog
from core.io import artifact_reference, read_json, sha256, write_json
from rollout.retry import verify_final
from tests.conftest import task


def _runs(cfg, repeat=3):
    root = Path(cfg['project_root'])
    Catalog(root).register_tasks([task('append-task')])
    runs = []
    for rid in ('forward', 'reverse'):
        run = create_run(cfg, rid, operation='rollout', task_ids=['append-task'], repeat_count=repeat)
        loaded = load_plan(root, rid)
        write_json(run / 'status.json', {'stage': 'cancelled'})
        write_json(run / 'progress.json', {'samples': {
            i['sample_key']: {**i, 'exit_status': 'Pending', 'disposition': 'pending'}
            for i in loaded['work_items']['items']}})
        runs.append(loaded)
    return root, runs[0], runs[1]


def _executed(loaded, index, *, accepted=True, status=None, history=False):
    root, run = loaded['run'].parent.parent, loaded['run']
    item = loaded['work_items']['items'][index]
    aid = item['repetition_id'] + '-attempt-001'
    work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
    context = {**item, 'attempt_id': aid, 'seed_round': 0, 'effective_seed': item['sample_seed']}
    status = status or ('Submitted' if accepted else 'OpenHandsError')
    result = {**context, 'exit_status': status, 'agent_exit_status': status,
              'disposition': 'accepted' if accepted else 'failed'}
    out = work / 'attempts' / aid
    write_json(out / 'attempt-context.json', context)
    write_json(out / 'attempt-started.json', {'execution_attempt_id': '000'})
    write_json(out / 'agent-outcome.json', {**context, 'agent_exit_status': status})
    if history:
        old = work / 'attempts' / (item['repetition_id'] + '-attempt-000')
        old_context = {**context, 'attempt_id': old.name}
        write_json(old / 'attempt-context.json', old_context)
        write_json(old / 'agent-outcome.json', {**old_context, 'agent_exit_status': 'TimeoutError'})
        write_json(old / 'raw.traj.json', {'history': True})
    if accepted:
        rid = 'r-' + run.name + '-' + str(index)
        raw = run / 'rollout/trajectories' / rid / 'raw.traj.json'
        patch = run / 'rollout/patches' / rid / 'patch.diff'
        write_json(raw, {'instance_id': item['instance_id'], 'info': {'exit_status': status}})
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text('patch ' + rid)
        row = {**context, 'rollout_id': rid, 'benchmark': 'fixture', 'revision': 'v1',
               'agent': 'fixture', 'purposes': ['baseline'], 'model_fingerprint': 'fixture-model',
               'trajectory_path': artifact_reference(root, raw), 'trajectory_sha256': sha256(raw),
               'patch_path': artifact_reference(root, patch), 'patch_sha256': sha256(patch),
               'submission_status': 'submitted', 'termination_reason': status,
               'evaluation_status': 'not_run', 'resolved': None, 'selected_evaluation_id': None,
               'provenance': {}}
        RunEvidence(root, run.name).register_rollout(rid, row, task('append-task'))
        result['rollout_id'] = rid
    write_json(out / 'attempt.json', result)
    write_json(work / 'result.json', result)
    progress = read_json(run / 'progress.json')
    progress['samples'][item['sample_key']] = result
    write_json(run / 'progress.json', progress)
    return result


def test_append_keeps_forward_failures_caps_and_preserves_history(cfg):
    root, dst, src = _runs(cfg)
    own = _executed(dst, 0, accepted=False)
    failed = _executed(src, 0, accepted=False, status='AgentCompleted', history=True)
    accepted = _executed(src, 1)
    _executed(src, 2)
    snapshots = {p: p.read_bytes() for loaded in (dst, src)
                 for p in [loaded['run'] / 'plan.json', loaded['run'] / 'resolved-config.json',
                           loaded['run'] / 'selection.json']}
    before = (dst['run'] / 'progress.json').read_bytes()
    preview = merge_runs(root, 'reverse', 'forward', mode='append-executed', dry_run=True)
    assert preview['changed'] == 2
    assert preview['stats']['over_target'] == 1
    assert (dst['run'] / 'progress.json').read_bytes() == before
    result = merge_runs(root, 'reverse', 'forward', mode='append-executed')
    assert result['changed'] == 2
    rows = read_json(dst['run'] / 'progress.json')['samples']
    items = dst['work_items']['items']
    assert rows[items[0]['sample_key']]['attempt_id'] == own['attempt_id']
    imported_failure = rows[items[1]['sample_key']]
    assert imported_failure['exit_status'] == 'AgentCompleted'
    assert imported_failure['disposition'] == 'failed'
    assert imported_failure['source_sample_key'] == failed['sample_key']
    imported = rows[items[2]['sample_key']]
    assert imported['rollout_id'] == accepted['rollout_id']
    assert imported['source_repetition_id'] == 'rep-001'
    assert imported['repetition_id'] == 'rep-002'
    assert imported['effective_seed'] == accepted['effective_seed']
    verify_final(root, 'forward', items[2], imported, RunEvidence(root, 'forward').read())
    attempts = dst['run'] / 'rollout/tasks/append-task/repetitions/rep-001/attempts'
    assert any(attempts.glob('*-history-*/raw.traj.json'))
    assert result['progress']['failed_samples'] == 2
    assert result['progress']['accepted_samples'] == 1
    assert verify(root, 'forward')['verified']
    assert sample_state(dst['config'], imported_failure) == 'failed'
    assert run_status(root, 'forward')['sample_summary']['accepted_count'] == 1
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed')['changed'] == 0
    for path, content in snapshots.items():
        assert path.read_bytes() == content


def test_unstarted_admission_is_not_a_trajectory_and_cleanup_is_required(cfg):
    root, dst, src = _runs(cfg)
    item = src['work_items']['items'][0]
    progress = read_json(src['run'] / 'progress.json')
    progress['samples'][item['sample_key']].update(exit_status='AdmissionWaiting',
                                                 disposition='failed', attempt_id='rep-000-attempt-000')
    write_json(src['run'] / 'progress.json', progress)
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed', dry_run=True)['changed'] == 0
    _executed(src, 1, accepted=False)
    write_json(src['run'] / 'execution/latest.json', {'attempt': '000'})
    with pytest.raises(ValueError, match='exit receipt'):
        merge_runs(root, 'reverse', 'forward', mode='append-executed')
    write_json(src['run'] / 'execution/attempts/000/exit.json', {'returncode': 0})
    with pytest.raises(ValueError, match='cleanup'):
        merge_runs(root, 'reverse', 'forward', mode='append-executed')


def test_append_is_idempotent_with_unused_capacity_and_failures(cfg):
    root, dst, src = _runs(cfg, repeat=4)
    _executed(src, 0, accepted=False)
    _executed(src, 1)
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed')['changed'] == 2
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed')['changed'] == 0
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed', dry_run=True)['changed'] == 0


def test_queued_cursor_does_not_erase_an_actually_executed_repetition(cfg):
    root, dst, src = _runs(cfg)
    previous = _executed(src, 0, accepted=False)
    progress = read_json(src['run'] / 'progress.json')
    item = src['work_items']['items'][0]
    progress['samples'][item['sample_key']] = {**item, 'exit_status': 'Pending', 'disposition': 'pending'}
    write_json(src['run'] / 'progress.json', progress)
    # A later execution requeued the sample, but has not physically started it.
    write_json(src['run'] / 'execution/latest.json', {'attempt': '001'})
    write_json(src['run'] / 'execution/attempts/001/exit.json', {})
    write_json(src['run'] / 'execution/attempts/001/termination.json', {'cleanup_complete': True})
    assert merge_runs(root, 'reverse', 'forward', mode='append-executed')['changed'] == 1
    imported = next(v for v in read_json(dst['run'] / 'progress.json')['samples'].values() if v.get('merged_from'))
    assert imported['exit_status'] == previous['exit_status']
