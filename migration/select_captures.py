"""Freeze one explicit online capture per task, preferring complete trajectories.

Ties are uniformly sampled with an auditable seed, independent of grading or
length. Capture completeness and successful/submitted rollout status remain
separate facts. Exceptional tasks are retained with explicit limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import secrets

from application.execution.evidence import RunEvidence
from core.io import now, owned_path, read_json, sha256, write_json
from hidden.artifacts import response_records


def choose(rows, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['instance_id']].append(row)
    rng = random.Random(seed)
    selected = []
    for task, candidates in sorted(grouped.items()):
        valid = [r for r in candidates if r['capture_integrity_complete']]
        if not valid:
            raise ValueError('No integrity-complete capture for ' + task)
        complete = [r for r in valid if r['rollout_complete']]
        submitted = [r for r in valid if r['submitted']]
        pool = complete or submitted or valid
        winner = rng.choice(sorted(pool, key=lambda r: r['rollout_id']))
        selected.append({**winner, 'selection_pool': [r['rollout_id'] for r in sorted(pool, key=lambda r: r['rollout_id'])],
                         'selection_rule': 'complete_then_submitted_then_archival; random_tie'})
    return selected


def select(root, scope_file, output, seed=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Selection is immutable; use the existing selection or a new output ID')
    scope = read_json(scope_file)
    requested = {t['instance_id']: t for t in scope['tasks']}
    catalogs = {}
    captures = read_json(root / 'hidden/manifest.json')['online_captures']
    by_rollout = defaultdict(list)
    for cap in captures.values():
        by_rollout[cap.get('rollout_id')].append(cap)
    rows = []
    for task in scope['tasks']:
        for candidate in task['think_close_rollouts']:
            rid = candidate['rollout_id']
            source_run = candidate.get('source_run_id') or candidate.get('run_id')
            if not source_run:
                raise ValueError('Capture candidate must specify source_run_id before rollout_id')
            if source_run not in catalogs:
                catalogs[source_run] = RunEvidence(root, source_run).read()
            catalog = catalogs[source_run]
            row = catalog['rollouts'][rid]
            definition = catalog['tasks'][row['task_key']]
            if definition['instance_id'] != task['instance_id'] or definition['base_commit'] != task['base_commit']:
                raise ValueError('Task identity differs from frozen scope')
            errors = []
            for kind in ('trajectory', 'patch'):
                if sha256(owned_path(root, row[kind + '_path'])) != row[kind + '_sha256']:
                    errors.append(kind + '_hash_mismatch')
            caps = [c for c in by_rollout[rid] if c.get('status') == 'complete'
                    and c.get('published') is True and 'think_close' in c.get('anchors', [])]
            if len(caps) != 1:
                raise ValueError('Expected one published capture for ' + rid)
            cap = caps[0]
            final_path = Path(cap['finalization_path'])
            final = read_json(final_path)
            coverage = read_json(final_path.parent / 'coverage.json')
            if final['rollout_id'] != rid or final['trajectory_sha256'] != row['trajectory_sha256']:
                errors.append('finalization_identity_mismatch')
            if final.get('model_fingerprint') != row['model_fingerprint']:
                errors.append('model_mismatch')
            if coverage.get('missing') or coverage.get('expected') != coverage.get('observed'):
                errors.append('capture_coverage_incomplete')
            trajectory = read_json(owned_path(root, row['trajectory_path']))
            responses = list(response_records(trajectory))
            ids = [r.get('dycon_native_request_id') for _, _, r in responses]
            if len(ids) != len(set(ids)) or set(ids) != set(final['accepted_native_request_ids']):
                errors.append('accepted_request_identity_mismatch')
            finishes = Counter(str(r.get('choices', [{}])[0].get('finish_reason')) for _, _, r in responses)
            submitted = row['submission_status'] == 'submitted' and trajectory.get('info', {}).get('exit_status') == 'Submitted'
            limitations = []
            if not submitted:
                limitations.append('not_submitted:' + row['termination_reason'])
            if finishes.get('length'):
                limitations.append('contains_length_truncated_response')
            if not owned_path(root, row['patch_path']).stat().st_size:
                limitations.append('empty_patch')
            rows.append({'instance_id': definition['instance_id'], 'base_commit': definition['base_commit'],
                'task_key': row['task_key'], 'rollout_id': rid, 'source_run_id': row['run_id'],
                'trajectory_sha256': row['trajectory_sha256'], 'patch_sha256': row['patch_sha256'],
                'session_id': cap['session_id'], 'capture_root': cap['export_root'],
                'capture_integrity_complete': not errors, 'integrity_errors': errors,
                'submitted': submitted, 'rollout_complete': submitted and not finishes.get('length'),
                'response_finish_reasons': dict(finishes), 'limitations': limitations,
                'patch_bytes': owned_path(root, row['patch_path']).stat().st_size,
                'source_metadata_files': {str(p): sha256(p) for p in [final_path, final_path.parent / 'coverage.json']}})
    seed = seed if seed is not None else secrets.randbits(128)
    selected = choose(rows, seed)
    if {r['instance_id'] for r in selected} != set(requested):
        raise ValueError('Not all requested tasks have a selected candidate')
    output.mkdir(parents=True)
    write_json(output / 'source-runs.json', catalogs)
    write_json(output / 'candidate-integrity.json', rows)
    write_json(output / 'selected.json', {'schema_version': 1, 'created_at': now(), 'seed': seed,
        'scope_sha256': sha256(scope_file), 'source_run_revisions': {run: data['revision'] for run, data in catalogs.items()},
        'policy': 'one per task; complete preferred; uniform random ties; no evaluation/step-length ranking',
        'tasks': selected})
    groups = defaultdict(list)
    for row in selected:
        groups[row['source_run_id']].append(row)
    runs = []
    for index, (source, chosen) in enumerate(sorted(groups.items()), 1):
        target = f'migration-thinkclose53-{index:02d}-20261021'
        selection = {'rollouts': [{k: r[k] for k in ('rollout_id', 'trajectory_sha256', 'patch_sha256')} for r in chosen]}
        write_json(output / f'{target}-selection.json', selection)
        write_json(output / f'{target}-tasks.json', sorted({r['instance_id'] for r in chosen}))
        runs.append({'source_run_id': source, 'target_run_id': target, 'tasks': len(chosen),
                     'selection': str(output / f'{target}-selection.json'),
                     'tasks_file': str(output / f'{target}-tasks.json')})
    write_json(output / 'runs.json', runs)
    write_json(output / 'selection-receipt.json', {'schema_version': 1,
        'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.glob('*.json'))}})
    return {'tasks': len(selected), 'candidates': len(rows), 'source_runs': len(runs), 'seed': seed,
            'complete_rollouts': sum(r['rollout_complete'] for r in selected),
            'submitted': sum(r['submitted'] for r in selected), 'output': str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--scope', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    print(json.dumps(select(a.root, a.scope, a.output, a.seed), indent=2))


if __name__ == '__main__':
    main()
