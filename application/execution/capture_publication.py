"""Receipt-bound run-end capture publication. Services never acquire the run lock.

The runner calls this after executor/serving/backend teardown. The explicit CLI
owns the same exclusive run lock and proves producer inactivity before retrying.
No model, dataset training, or original session finalization is performed here.
"""
from collections import Counter
from pathlib import Path
import time

from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_run
from hidden.archive_online import assert_inactive
from hidden.publish_online import complete_native, has_publication_subject, publish_online_service
from hidden.repair_points import recover_points
from hidden.layer_contract import POINTWISE_CONTRACT
from rollout.retry import sample_accepted, verify_final


class PendingRecovery(RuntimeError):
    """Producers or unacknowledged spool bytes prevent safe publication."""


def freeze_selection(cfg, run_id, items, evidence):
    """Use the scheduler's durable result pointers and its retry/identity checks."""
    from application.execution.failed_retry import retry_index, superseded
    root = Path(cfg['project_root']).resolve()
    run = root / 'runs' / run_id
    retries = retry_index(run)
    rows, excluded = [], []
    selected = set()
    for item in items:
        path = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
        result = read_json(path) if path.exists() else {}
        reason = None
        if superseded(retries, item['sample_key'], result):
            reason = 'superseded_by_operator_retry'
        elif result.get('disposition') != 'accepted' or not sample_accepted(cfg, result):
            reason = 'not_final_accepted'
        if reason:
            excluded.append({'sample_key': item['sample_key'], 'reason': reason})
            continue
        rid = verify_final(root, run_id, item, result, evidence)
        row = evidence['rollouts'][rid]
        if row.get('provenance', {}).get('quarantined') or not has_publication_subject(row):
            excluded.append({'rollout_id': rid, 'reason': 'quarantined_or_not_submitted'})
            continue
        if rid in selected:
            raise ValueError('Duplicate final canonical rollout')
        selected.add(rid)
        sid = identifier('cap-' + item['sample_key'] + '-' + result['attempt_id'])
        metadata = {}
        for base in (root / cfg['capture']['export']['spool_root'] / run_id,
                     Path(cfg['paths']['nas_root']) / cfg['capture']['export']['root'] / run_id):
            for name in ('session.json', 'receipt.json', 'coverage.json', 'finalization.json'):
                source = base / 'sessions' / sid / name
                if source.exists():
                    metadata[str(source)] = sha256(source)
        evaluation = evidence['evaluations'].get(row.get('selected_evaluation_id'))
        if evaluation:
            report = owned_path(root, evaluation['report_path'])
            if sha256(report) != evaluation['report_sha256']:
                raise ValueError('Evaluation report checksum mismatch')
            metadata[str(report)] = evaluation['report_sha256']
        rows.append({'rollout_id': rid, 'sample_key': item['sample_key'],
                     'repetition_id': item['repetition_id'], 'attempt_id': result['attempt_id'],
                     'session_id': sid, 'disposition': result['disposition'],
                     'capture_status': result.get('capture_status'),
                     'evaluation_status': row.get('evaluation_status'), 'resolved': row.get('resolved'),
                     'trajectory_path': row['trajectory_path'], 'trajectory_sha256': row['trajectory_sha256'],
                     'model_fingerprint': row['model_fingerprint'],
                     'result_path': str(path), 'result_sha256': sha256(path), 'source_files': metadata})
    excluded.extend({'rollout_id': rid, 'reason': 'not_selected_canonical_result'}
                    for rid in sorted(set(evidence['rollouts']) - selected))
    return {'schema_version': 1, 'selection_rule': 'final_accepted_canonical_v1',
            'run_id': run_id, 'config_sha256': sha256(run / 'resolved-config.json'),
            'policy': 'native_exact_accepted_v1', 'rollouts': rows, 'excluded': excluded}


def recover_spool(cfg, run_id):
    """Reuse ACK/upload recovery only; do not rewrite native session receipts."""
    from runtime.serving.spool import CaptureSpool
    root = Path(cfg['project_root']) / cfg['capture']['export']['spool_root'] / run_id
    def pending():
        return [p for p in (root / 'uploads').glob('*.json')
                if not p.name.endswith('.pid.json') and read_json(p).get('state') != 'published']
    queued = any(isinstance(value := read_json(p), dict) and value.get('artifact_state') == 'queued'
                 for p in (root / 'index').glob('*.json'))
    if not pending() and not queued:
        return
    spool = CaptureSpool(cfg['project_root'], cfg['capture']['export'], run_id, cfg['paths']['nas_root'])
    try:
        spool.recover()
    finally:
        closed = spool.close(cfg['capture']['export']['upload_max_elapsed_seconds'])
    if not closed or pending():
        raise PendingRecovery('Unacknowledged spool uploads remain; use dycon capture recover')


def finalize_capture_publication(cfg, run_id, *, items=None, producers=(), no_deadline=False):
    """Caller owns the run lock; failures affect only the publication summary."""
    capture = cfg.get('capture', {})
    if not capture.get('enabled') or not capture.get('export', {}).get('enabled'):
        return None
    root = Path(cfg['project_root']).resolve()
    run_id = identifier(run_id)
    run = root / 'runs' / run_id
    summary_path = run / 'capture-publication.json'
    stamp = time.monotonic()
    summary = {'schema_version': 1, 'run_id': run_id, 'status': 'pending',
               'updated_at': now(), 'stage_seconds': {}, 'deadline_disabled': no_deadline,
               'recovery_command': f'dycon capture finalize {run_id}'}
    phase = 'selection'
    def checkpoint():
        if not no_deadline and time.monotonic() - stamp > capture.get('publication_seconds', 3600):
            raise PendingRecovery('Capture publication deadline reached; retry the same selection')
    try:
        previous = read_json(summary_path) if summary_path.exists() else {}
        if previous.get('last_success_at'):
            summary['last_success_at'] = previous['last_success_at']
        write_json(summary_path, summary)
        parallel = (cfg.get('parallel', {}).get('enabled')
                    and cfg['parallel'].get('branch_count', 1) > 1)
        if capture.get('anchors') != ['think_close']:
            summary.update(status='unsupported', reason='Requires think_close capture')
            return summary
        spool = root / capture['export']['spool_root'] / run_id
        for producer in producers:
            queue = getattr(producer, 'queue', None)
            if queue is not None and any(getattr(queue, name, None) is not None
                    and getattr(queue, name).is_alive() for name in ('_worker', '_spill_worker')):
                raise PendingRecovery('Local capture writer is still active after teardown')
        try:
            assert_inactive(run_id, spool)
        except ValueError as exc:
            raise PendingRecovery(str(exc)) from exc
        phase = 'spool_recovery'
        recover_spool(cfg, run_id)
        checkpoint()
        phase = 'selection'
        evidence = RunEvidence(root, run_id).read()
        selection = freeze_selection(cfg, run_id, items if items is not None else
                                     load_plan(root, run_id)['work_items']['items'], evidence)
        # Corrected audits get new immutable IDs; old failed repairs remain evidence.
        selection['repair_contract'] = POINTWISE_CONTRACT
        if parallel:
            from hidden.parallel_proof import decision_inventory
            nas = Path(cfg['paths']['nas_root']) / capture['export']['root'] / run_id
            selection.update(purpose='visualization',policy='parallel_visualization_v2',eligible_for_training=False)
            for row in selection['rollouts']:
                _,sources,_=decision_inventory([spool,nas],row['session_id'],row['rollout_id'],row['trajectory_sha256'])
                unique={}
                for path,digest in sources.items():
                    source=Path(path)
                    base=spool if source.is_relative_to(spool) else nas
                    unique.setdefault(str(source.relative_to(base)),(path,digest))
                row['source_files'].update(dict(unique.values()))
        digest = hash_json(selection)
        repair_id, publication_id, index_id = ('auto-' + run_id[:80] + '-' + digest[:24] + '-' + suffix
                                              for suffix in ('repair', 'publication', 'index'))
        summary.update(selection_sha256=digest, repair_id=repair_id,
                       publication_id=publication_id, index_id=index_id,
                       selected_sessions=len(selection['rollouts']),
                       excluded_rollouts=len(selection['excluded']))
        audit = run / 'hidden/capture-repairs' / repair_id
        path = audit / 'publication-selection.json'
        if path.exists():
            prior = read_json(path)
            if prior['selection'] != selection or prior['selection_sha256'] != digest:
                raise ValueError('Immutable publication selection conflict')
        else:
            write_json(path, {'selection': selection, 'selection_sha256': digest, 'created_at': now()})
        selection_receipt = audit / 'publication-selection-receipt.json'
        receipt = {'selection_sha256': digest, 'files': {path.name: sha256(path)}}
        if selection_receipt.exists() and read_json(selection_receipt) != receipt:
            raise ValueError('Publication selection receipt checksum mismatch')
        write_json(selection_receipt, receipt)
        if not selection['rollouts']:
            summary.update(status='complete', publication_id=None, index_id=None, repair_id=None,
                           reason='no_eligible_final_accepted_sessions', published_points=0)
            return summary
        ids = [r['rollout_id'] for r in selection['rollouts']]
        phase = 'repair'
        stage = time.monotonic()
        report = recover_points(cfg, run_id, rollout_ids=ids, apply=True, repair_id=repair_id,
                                selection_sha256=digest, checkpoint=checkpoint, visualization_parallel=parallel)
        summary['stage_seconds'][phase] = time.monotonic() - stage
        coverages = [r['coverage'] for r in report['sessions']]
        summary.update(expected=sum(c['expected_known'] for c in coverages),
                       observed=sum(c['observed_native_exact'] for c in coverages),
                       missing=sum(len(c['missing_known']) for c in coverages),
                       unexpected=0,
                       complete_sessions=sum(complete_native(c) for c in coverages),
                       partial_sessions=sum(not complete_native(c) for c in coverages),
                       unbound_responses=sum(len(c['unbound_responses']) for c in coverages),
                       rejected_points=report['summary']['rejected_points'])
        if report['summary']['published_sessions'] != len(selection['rollouts']):
            raise PendingRecovery('Repair uploads remain unacknowledged')
        if (not parallel and (report['input_issues'] or report['summary']['rejected_points'])) or any(c.get('canonical_error') for c in coverages):
            raise ValueError('Capture integrity audit rejected source evidence; inspect repair report')
        checkpoint()
        phase = 'publication'
        stage = time.monotonic()
        if parallel:
            from hidden.publish_visualization import publish_visualization
            publication=publish_visualization(cfg,run_id,repair_id=repair_id,publication_id=publication_id,
                selection=selection,selection_sha256=digest,checkpoint=checkpoint)
            summary['stage_seconds'][phase]=time.monotonic()-stage
            summary.update(status=publication['coverage_status'],purpose='visualization',eligible_for_training=False,
                publication_state='published',published_points=publication['published_points'],
                branch_points=publication['branch_points'],truth_available=publication['truth_available'],
                truth_censored=publication['truth_censored'],canonical_coverage=publication['canonical_coverage'],
                all_candidate_coverage=publication['all_candidate_coverage'],index_id=None,
                publication_receipt_sha256=sha256(Path(publication['path'])/'receipt.json'),last_success_at=now(),
                visualization_commands={
                    'parallel':f'dycon difficulty-plot-parallel {run_id} PLOT_ID --rollout-id ROLLOUT_ID --publication-id {publication_id}',
                    'trajectories':f'dycon difficulty-plot-trajectories {run_id} PLOT_ID --attempt-id EXISTING_FIT_ATTEMPT --rollout-id ROLLOUT_ID --publication-id {publication_id} --target submission_remaining_reasoning_tokens_log1p'})
            return summary
        policy = ('complete_native_sessions_v1' if all(complete_native(c) for c in coverages)
                  else 'available_native_suffix_v1')
        try:
            publication = publish_online_service(cfg, run_id, repair_id=repair_id, publication_id=publication_id,
                                                 coverage_policy=policy, rollout_ids=ids, named_output=True,
                                                 selection_sha256=digest)
        except ValueError as exc:
            if str(exc) == 'Publication needs nonempty qualified sessions with one marker inventory':
                summary.update(status='partial', publication_id=None, index_id=None,
                               excluded_sessions=len(selection['rollouts']), published_points=0,
                               reason='no_exact_hidden_in_qualified_suffix')
                return summary
            raise
        summary['stage_seconds'][phase] = time.monotonic() - stage
        summary.update(status=publication['coverage_status'], published_points=publication['published_points'],
                       excluded_sessions=publication['excluded_rollouts'], policy=policy,
                       publication_receipt_sha256=sha256(Path(publication['path']) / 'receipt.json'))
        # A stable index ID is recorded, with the existing official command as its
        # materialization interface. No user-named dataset/split is created here.
        summary['index_command'] = f'dycon online-index {run_id} {index_id} --publication-id {publication_id}'
        summary['index_state'] = 'available_to_materialize'
        summary['last_success_at'] = now()
        return summary
    except (PendingRecovery, TimeoutError) as exc:
        summary.update(status='pending_recovery', reason=str(exc)[:512], failed_stage=phase)
        return summary
    except Exception as exc:
        from runtime.serving.diagnostics import sanitize_error_text
        summary.update(status='failed', reason=sanitize_error_text(exc, 512),
                       reason_code='capture_' + phase + '_failed',
                       error_type=type(exc).__name__, failed_stage=phase)
        return summary
    finally:
        summary.update(updated_at=now(), elapsed_seconds=time.monotonic() - stamp)
        try:
            write_json(summary_path, summary)
        except OSError:
            pass  # Reporting storage cannot retroactively fail rollout.


@exclusive_run
def retry_capture_publication(cfg, run_id, *, no_deadline=False):
    loaded = load_plan(cfg['project_root'], run_id)
    return finalize_capture_publication(loaded['config'], run_id, items=loaded['work_items']['items'], no_deadline=no_deadline)
