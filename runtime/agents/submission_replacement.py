"""Backed-up canonical outcome replacement after authenticated submission replay."""
from __future__ import annotations

import copy
import json
import math
from contextlib import ExitStack
from pathlib import Path

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.agents.pddl_regrading import _bytes, _digest, _transaction


def eligible(record):
    rewards = record.get('new_rewards')
    return (record.get('classification') == 'ValidOutcome'
        and record.get('verifier_executed') is True
        and record.get('cleanup_confirmed') is True
        and not any(record.get(k) for k in ['verifier_error', 'agent_error', 'cleanup_error'])
        and isinstance(rewards, dict) and set(rewards) == {'reward'}
        and type(rewards['reward']) in (int, float)
        and math.isfinite(rewards['reward']) and 0 <= rewards['reward'] <= 1)


def prepare_changes(source, target, selected, records, output):
    from runtime.agents.submission_regrading import member

    run = source['run'].resolve()
    evidence, progress = read_json(run / 'evidence.json'), read_json(run / 'progress.json')
    changes, applied, skipped = {}, [], []
    mapped = {r['sample_key']: r for r in records}
    if len(mapped) != len(records) or set(mapped) != {i['source']['sample_key'] for i in selected}:
        raise ValueError('Replacement records do not match the exact replay selection')
    for item in selected:
        row = item['source']
        record = mapped[row['sample_key']]
        folder = output / 'samples' / row['sample_key']
        if read_json(folder / 'result.json') != record:
            raise ValueError('Replay result changed before replacement')
        if not eligible(record):
            skipped.append({'sample_key': row['sample_key'], 'classification': record['classification'],
                            'reason': record.get('reason'), 'historical_outcome_preserved': True})
            continue
        old = member(run, row['attempt_path'])
        raw_path, report_path = old / 'evaluation-raw.json', old / 'evaluation-report.json'
        value = read_json(old / 'attempt.json')
        if (value['sample_key'] != row['sample_key'] or value['attempt_id'] != row['attempt_id']
                or value['raw_reward'] != row['raw_reward']
                or value['subject_sha256'] != item['subject_sha256']
                or sha256(item['subject_path']) != item['subject_sha256']
                or sha256(report_path) != value['evaluation_report_sha256']):
            raise ValueError('Historical attempt changed before replacement')
        rid, eid = value['rollout_id'], value['evaluation_id']
        ev, trajectory = evidence['evaluations'][eid], evidence['rollouts'][rid]
        if (trajectory['selected_evaluation_id'] != eid
                or trajectory['subject_sha256'] != item['subject_sha256']
                or ev['report_sha256'] != sha256(report_path) or set(ev['outcomes']) != {rid}):
            raise ValueError('Canonical evaluation binding changed')
        raw = read_json(folder / 'evaluation-raw.json')
        if raw.get('completed') is not True or raw.get('rewards') != record['new_rewards']:
            raise ValueError('No matching completed replay evaluation')
        resolved = record['new_rewards']['reward'] == 1
        correction = {'schema_version': 1, 'policy': 'skillsbench-submission-replacement-v1',
            'mode': 'replace-original-with-backup', 'regraded_at': now(),
            'evaluation_run_id': target['plan']['run_id'], 'replay_scope': record['replay_scope'],
            'proof_path': str(folder / 'result.json'), 'proof_sha256': sha256(folder / 'result.json'),
            'task_package_sha256': record['task_package_sha256'],
            'dependency_snapshot_sha256': record['dependency_snapshot_sha256'],
            'evaluation_config_sha256': sha256(target['run'] / 'resolved-config.json'),
            'original_raw_sha256': sha256(raw_path), 'original_report_sha256': sha256(report_path),
            'backup_manifest': str(output / 'replacement-manifest.json')}
        # The canonical outcome binds the original authenticated subject. The
        # fresh-container subject identity remains explicit in the replay proof.
        raw.update(subject_sha256=item['subject_sha256'], regrading=correction,
                   replay_subject_sha256=raw.get('subject_sha256'),
                   verifier_evidence=str(folder / 'diagnostics/verifier-execution.json'))
        changes[raw_path] = _bytes(raw)
        audit = read_json(folder / 'verifier-audit.json')
        audit.update(evidence_root=str(folder), regrading=correction)
        audit_path = old / 'diagnostics/verifier-zero-audit.json'
        changes[audit_path] = _bytes(audit)
        report = read_json(report_path)
        report.update(raw_rewards=record['new_rewards'], resolved=resolved, evaluation_status='completed',
                      report_sha256=_digest(changes[raw_path]), verifier_zero_audit=audit,
                      evaluated_at=record['finished_at'], regrading=correction)
        changes[report_path] = _bytes(report)
        value.update(raw_reward=record['new_rewards'], resolved=resolved, evaluation_status='completed',
                     evaluation_report_sha256=_digest(changes[report_path]), verifier_zero_audit=audit,
                     regrading=correction)
        if 'raw_evaluation_sha256' in value:
            value['raw_evaluation_sha256'] = _digest(changes[raw_path])
        for ref in (value.get('termination_evidence') or {}).get('evidence_files', []):
            for path in (raw_path, audit_path):
                if ref['path'] == str(path.relative_to(old)):
                    ref['sha256'] = _digest(changes[path])
        changes[old / 'attempt.json'] = _bytes(value)
        final = old.parent.parent / 'result.json'
        if read_json(final)['attempt_id'] != row['attempt_id']:
            raise ValueError('Historical repetition was superseded')
        changes[final] = _bytes(value)
        for name in ['03-artifacts_committed.json', '05-settled.json']:
            path = old / 'finalization' / name
            document = read_json(path)
            if name.startswith('03'):
                document['details']['report_sha256'] = _digest(changes[raw_path])
            else:
                document['details'].update(raw_reward=record['new_rewards'], resolved=resolved)
            document['regrading'] = correction
            document.pop('receipt_sha256', None)
            document['receipt_sha256'] = hash_json(document)
            changes[path] = _bytes(document)
        trajectory.update(resolved=resolved, evaluation_status='completed', regrading=correction)
        ev['outcomes'][rid].update(resolved=resolved, rewards=record['new_rewards'], status='completed')
        ev['report_sha256'] = _digest(changes[report_path])
        ev.setdefault('provenance', {})['regrading'] = correction
        progress['samples'][row['sample_key']] = value
        applied.append({'sample_key': row['sample_key'], 'before': row['raw_reward'],
                        'after': record['new_rewards'], 'resolved': resolved})
    if applied:
        evidence['revision'] += 1
        progress['updated_at'] = now()
        changes[run / 'evidence.json'] = _bytes(evidence)
        changes[run / 'progress.json'] = _bytes(progress)
    return changes, applied, skipped


def replace_outcomes(root, source, target, selected, records, output, before, *, allow_active_source=False):
    from application.execution.config_revision import _offline
    from application.execution.evidence import RunEvidence
    from application.execution.run_plan import load_plan
    from application.execution.status import _sample_status_projection, refresh_run_status_summary

    root, run = Path(root).resolve(), source['run'].resolve()
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (source['plan']['run_id'] + '.lock'),
                     run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock',
                     run / 'validation/.oracle-lifecycle.lock', run / 'evidence.lock', run / 'progress.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        if not allow_active_source:
            _offline(run)
        changed = [p for p, digest in before.items() if not Path(p).is_file() or sha256(p) != digest]
        if changed:
            raise ValueError('Historical sources changed during replay; refusing replacement')
        latest = load_plan(root, source['plan']['run_id'])
        projection, _ = _sample_status_projection(latest, read_json(run / 'progress.json'))
        for item in selected:
            row = item['source']; current = projection[row['sample_key']]
            if any(current.get(k) != row[k] for k in ['instance_id','attempt_id','repetition_id','raw_reward']):
                raise ValueError('Historical sample superseded during replay')
        changes, applied, skipped = prepare_changes(latest, target, selected, records, output)
        result = {'requested': True, 'applied_count': len(applied), 'skipped_count': len(skipped),
                  'applied': applied, 'skipped': skipped, 'finished_at': now()}
        if changes:
            RunEvidence(root, source['plan']['run_id']).validate(json.loads(changes[run / 'evidence.json']))
            manifest = _transaction(run, output, changes)
            refresh_run_status_summary(root, source['plan']['run_id'])
            projection, _ = _sample_status_projection(latest, read_json(run / 'progress.json'))
            if not all(projection[r['sample_key']]['raw_reward'] == r['after']
                       and projection[r['sample_key']]['resolved'] == r['resolved'] for r in applied):
                raise RuntimeError('Canonical projection verification failed; inspect replacement journal')
            result.update(backup=str(output / 'backup'), replacement_files=len(manifest), verified=True)
        else:
            result['verified'] = True
        write_json(output / 'replacement-summary.json', result)
        return result
