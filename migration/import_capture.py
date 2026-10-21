"""One-shot, explicit migration of registered captured trajectories into a run.

Source run snapshots and NAS tensors remain immutable. Trajectory IDs and payload
hashes are retained; new run-local descriptors bind the original capture root.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from application.execution.artifact import refresh, verify
from application.execution.run_plan import load_plan
from application.execution.evidence import RunEvidence
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock


def migrate(root, source_run, target_run, selection_file):
    root = Path(root).resolve()
    source_run, target_run = identifier(source_run), identifier(target_run)
    loaded = load_plan(root, target_run)
    run = loaded['run']
    if source_run == target_run or loaded['plan']['operation'] != 'harness':
        raise ValueError('Migration requires a distinct, planned harness run')
    selection = read_json(selection_file)
    selected = selection['rollouts']
    if not selected or len({row['rollout_id'] for row in selected}) != len(selected):
        raise ValueError('Explicit nonempty, unique rollout selection required')
    selection_sha = hash_json(selection)
    with file_lock(root / 'state' / f'migration-{target_run}.lock', nonblocking=True):
        receipt_path = run / 'migration-receipt.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['selection_sha256'] != selection_sha or receipt['source_run_id'] != source_run:
                raise ValueError('Migration selection conflicts with completed import')
            verify(root, target_run)
            return receipt
        catalog = RunEvidence(root, source_run)
        before = catalog.read()
        wanted = {r['rollout_id']: r for r in selected}
        tasks = {r['task_key'] for r in loaded['tasks']['tasks']}
        originals = {}
        for rid, spec in wanted.items():
            row = before['rollouts'][rid]
            if row['run_id'] != source_run or row['task_key'] not in tasks:
                raise ValueError('Selected trajectory crosses source run or target tasks')
            for kind in ('trajectory', 'patch'):
                if row[kind + '_sha256'] != spec[kind + '_sha256']:
                    raise ValueError('Selection hash differs from source run')
                if sha256(owned_path(root, row[kind + '_path'])) != row[kind + '_sha256']:
                    raise ValueError('Selected source payload changed')
            originals[rid] = copy.deepcopy(row)
        captures = read_json(root / 'hidden/manifest.json')['online_captures']
        by_rollout = {}
        for sid, cap in captures.items():
            if cap.get('run_id') == source_run and cap.get('rollout_id') in wanted:
                rid = cap['rollout_id']
                if rid in by_rollout or cap.get('status') != 'complete' or cap.get('published') is not True:
                    raise ValueError('Ambiguous or incomplete capture for selected trajectory')
                if cap.get('trajectory_sha256') != originals[rid]['trajectory_sha256']:
                    raise ValueError('Capture does not bind the selected trajectory')
                final = read_json(cap['finalization_path'])
                if final['rollout_id'] != rid or final['trajectory_sha256'] != originals[rid]['trajectory_sha256']:
                    raise ValueError('NAS finalization differs from canonical identity')
                if final['model_fingerprint'] != originals[rid]['model_fingerprint']:
                    raise ValueError('Capture model differs from canonical identity')
                coverage = read_json(Path(cap['finalization_path']).parent / 'coverage.json')
                if coverage.get('missing'):
                    raise ValueError('Capture has missing evidence; keep outside complete migration')
                by_rollout[rid] = (sid, cap, final)
        if set(by_rollout) != set(wanted):
            raise ValueError('Every selected trajectory requires an explicit published capture')
        migration = run / 'migration'
        migration.mkdir(exist_ok=True)
        source_catalog = migration / 'source-catalog.json'
        if not source_catalog.exists():
            write_json(source_catalog, before)
        write_json(migration / 'selection.json', selection)
        local_files = {}

        def record(path):
            local_files[str(path.relative_to(run))] = sha256(path)

        record(source_catalog)
        record(migration / 'selection.json')
        for name in ('resolved-config.json', 'selection.json', 'status.json', 'stop-receipt.json',
                     'restart-provenance.json', 'process.json', 'startup-verified.json'):
            source = root / 'runs' / source_run / name
            if source.exists():
                target = migration / 'source-run' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256(target) != sha256(source):
                    raise ValueError('Source evidence changed during interrupted migration')
                if not target.exists():
                    shutil.copy2(source, target)
                record(target)
        converted, publications, mapping = {}, [], []
        for rid, row in originals.items():
            updated = copy.deepcopy(row)
            for kind, folder, name in [('trajectory', 'trajectories', 'raw.traj.json'),
                                       ('patch', 'patches', 'patch.diff')]:
                target = run / 'rollout' / folder / rid / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256(target) != row[kind + '_sha256']:
                    raise ValueError('Migration destination has different payload')
                if not target.exists():
                    shutil.copy2(owned_path(root, row[kind + '_path']), target)
                if sha256(target) != row[kind + '_sha256']:
                    raise ValueError('Copied payload hash differs')
                updated[kind + '_path'] = str(target.relative_to(root))
                record(target)
            updated['run_id'] = target_run
            updated['provenance'].update(origin='migrated', migration_run_id=target_run,
                migration_source_run_id=source_run,
                migration_source_paths={k: row[k + '_path'] for k in ('trajectory', 'patch')})
            converted[rid] = updated
            sid, cap, final = by_rollout[rid]
            refs = {}
            session_root = Path(cap['finalization_path']).parent
            for name in ('session.json', 'receipt.json', 'finalization.json', 'coverage.json'):
                source = session_root / name
                refs[name] = {'nas_locator': str(source), 'sha256': sha256(source)}
            current_final = {**final, 'trajectory_path': str(root / updated['trajectory_path']),
                             'origin': 'migrated', 'producer_run_id': target_run,
                             'source_finalization': refs['finalization.json']}
            target = run / 'hidden/sessions' / sid / 'finalization.json'
            write_json(target, current_final)
            record(target)
            publications.append({'session_id': sid, 'rollout_id': rid,
                'trajectory_sha256': row['trajectory_sha256'], 'files': refs,
                'nas_locator': cap['export_root'], 'source_run_id': source_run,
                'sample_count': cap.get('sample_count'),
                'finalization_locator': str(target.relative_to(root)),
                'accepted_native_request_count': len(final['accepted_native_request_ids'])})
            mapping.append({'rollout_id': rid, 'source_run_id': source_run, 'target_run_id': target_run,
                'attempt_id': row['attempt_id'], 'trajectory_sha256': row['trajectory_sha256'],
                'patch_sha256': row['patch_sha256'], 'session_id': sid})
        roots = {cap['export_root'] for _, cap, _ in by_rollout.values()}
        if len(roots) != 1:
            raise ValueError('Select one capture root per migration run')
        publication = {'schema_version': 1, 'run_id': target_run, 'source_run_id': source_run,
            'nas_locator': roots.pop(), 'state': 'published', 'origin': 'migrated',
            'sessions': publications, 'tensor_policy': 'reference immutable original NAS shards'}
        write_json(run / 'hidden/publication.json', publication)
        record(run / 'hidden/publication.json')
        write_json(migration / 'identity-mapping.json', mapping)
        record(migration / 'identity-mapping.json')
        excluded = [dict(session_id=sid, task_id=c.get('task_id'), status=c.get('status'),
                        published=c.get('published'), rollout_id=c.get('rollout_id'))
                    for sid, c in captures.items() if c.get('run_id') == source_run
                    and c.get('rollout_id') not in wanted]
        write_json(migration / 'excluded-sessions.json', excluded)
        record(migration / 'excluded-sessions.json')
        # Source evidence stays unchanged; the target gets its own independent index.
        target_evidence = RunEvidence(root, target_run)
        with target_evidence.transaction() as data:
            for rid, row in converted.items():
                if rid in data['rollouts'] and data['rollouts'][rid] != row:
                    raise ValueError('Target evidence conflicts with migration')
                data['rollouts'][rid] = row
                data['tasks'][row['task_key']] = before['tasks'][row['task_key']]
            for eid, evaluation in before['evaluations'].items():
                outcomes = {rid: value for rid, value in evaluation['outcomes'].items() if rid in converted}
                if not outcomes:
                    continue
                report = owned_path(root, evaluation['report_path'])
                if sha256(report) != evaluation['report_sha256']:
                    raise ValueError('Source evaluation report changed')
                target = run / 'harness/imports' / eid / 'summary.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256(target) != evaluation['report_sha256']:
                    raise ValueError('Target evaluation report conflicts')
                if not target.exists():
                    shutil.copy2(report, target)
                data['evaluations'][eid] = {**evaluation, 'outcomes': outcomes,
                    'report_path': str(target.relative_to(root))}
                record(target)
        receipt = {'schema_version': 1, 'origin': 'migrated', 'source_run_id': source_run,
            'migration_run_id': target_run, 'selection_sha256': selection_sha,
            'source_catalog_sha256': sha256(source_catalog), 'files': local_files,
            'rollouts': len(wanted), 'sessions': len(publications), 'finished_at': now(),
            'resume_policy': 'historical artifacts only; no reconstructed rollout execution'}
        write_json(receipt_path, receipt)
        refresh(root, target_run)
        verify(root, target_run)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--target-run', required=True)
    parser.add_argument('--selection', required=True)
    args = parser.parse_args()
    result = migrate(args.root, args.source_run, args.target_run, args.selection)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
