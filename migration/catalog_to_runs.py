"""Convert a v1 global manifest to tasks-only Catalog plus explicit retained runs.

Only --keep-run entries are migrated. Other rollout/evaluation registrations are
discarded. Existing payloads, frozen run plans and NAS captures are unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from application.execution.evidence import RunEvidence
from catalog.store import Manifest
from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json
from core.locking import file_lock


def migrate(root, keep_runs, *, apply=False):
    root = Path(root).resolve()
    keep_runs = [identifier(run) for run in keep_runs]
    if not keep_runs or len(keep_runs) != len(set(keep_runs)):
        raise ValueError("Provide unique explicit keep-run IDs")
    path = root / 'catalog/manifest.json'
    with file_lock(root / 'state/catalog.lock'):
        source = read_json(path)
        if source['schema_version'] == 2:
            return {'state': 'already_migrated', 'catalog': Manifest.model_validate(source).model_dump(),
                    'runs': [RunEvidence(root, run).verify() for run in keep_runs]}
        if source['schema_version'] != 1:
            raise ValueError('Unsupported source catalog schema')
        converted = {'schema_version': 2, 'catalog_revision': source['catalog_revision'] + 1,
                     'tasks': source['tasks']}
        Manifest.model_validate(converted)
        retained, copies = {}, []
        for run_id in keep_runs:
            evidence = RunEvidence(root, run_id)
            data = evidence.empty()
            data['rollouts'] = {rid: copy.deepcopy(row) for rid, row in source['rollouts'].items()
                                if row['run_id'] == run_id}
            if not data['rollouts']:
                raise ValueError(f'No registered trajectories in keep-run: {run_id}')
            data['tasks'] = {row['task_key']: source['tasks'][row['task_key']]
                             for row in data['rollouts'].values()}
            for row in data['rollouts'].values():
                for kind in ('trajectory', 'patch'):
                    payload = owned_path(root, row[kind + '_path'])
                    if not payload.is_relative_to(evidence.path.parent):
                        raise ValueError('Retained payload must already belong to its run')
                    if sha256(payload) != row[kind + '_sha256']:
                        raise ValueError('Retained payload checksum mismatch')
            for eid, evaluation in source['evaluations'].items():
                outcomes = {rid: outcome for rid, outcome in evaluation['outcomes'].items()
                            if rid in data['rollouts']}
                if not outcomes:
                    continue
                report = owned_path(root, evaluation['report_path'])
                if sha256(report) != evaluation['report_sha256']:
                    raise ValueError('Retained evaluation checksum mismatch')
                if not report.is_relative_to(evidence.path.parent):
                    destination = evidence.path.parent / 'harness/imports' / identifier(eid) / 'summary.json'
                    copies.append((report, destination, evaluation['report_sha256']))
                else:
                    destination = report
                data['evaluations'][eid] = {**copy.deepcopy(evaluation), 'outcomes': outcomes,
                                            'report_path': str(destination.relative_to(root))}
            data['revision'] = 1
            evidence.validate(data)
            if evidence.path.exists() and evidence.read() != data:
                raise ValueError('Existing run evidence differs from planned migration')
            retained[run_id] = data
        count = sum(len(data['rollouts']) for data in retained.values())
        summary = {'state': 'planned', 'keep_runs': keep_runs, 'tasks': len(source['tasks']),
                   'retained_rollouts': count, 'discarded_registrations': len(source['rollouts']) - count,
                   'retained_evaluations': {run: len(data['evaluations']) for run, data in retained.items()},
                   'source_manifest_sha256': sha256(path)}
        if not apply:
            return summary
        for report, destination, digest in copies:
            if destination.exists() and sha256(destination) != digest:
                raise ValueError('Evaluation destination conflicts')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(report, destination)
            if sha256(destination) != digest:
                raise ValueError('Copied evaluation checksum mismatch')
        for run_id, data in retained.items():
            evidence = RunEvidence(root, run_id)
            with evidence.transaction() as current:
                current.clear()
                current.update(data)
            evidence.verify()
        # Publish task-only authority after all retained run evidence has verified.
        write_json(path, converted)
        summary.update(state='complete', catalog_sha256=hash_json(converted))
        write_json(root / 'state/catalog-task-only-migration.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--keep-run', action='append', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = migrate(args.root, args.keep_run, apply=args.apply)
    if 'catalog' in result:
        result['tasks'] = len(result.pop('catalog')['tasks'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
