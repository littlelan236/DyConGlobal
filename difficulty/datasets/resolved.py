"""Freeze resolved canonical action-close samples for the existing Ridge workflow.

For a resolved trajectory the successful endpoint is its final submission. The
existing submission-remainder field names therefore retain their exact values;
eligibility is additionally bound to a completed, exact-patch resolved outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from application.execution.evidence import RunEvidence
from difficulty.datasets.selection import select_run
from core.io import identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.datasets.paths import dataset_directory


def build_resolved_selection(root, source_run, index_id, dataset_id, rollout_ids=(), rollout_list=None):
    root = Path(root).resolve()
    source_run, dataset_id = map(identifier, (source_run, dataset_id))
    run = root / 'runs' / source_run
    index_path = dataset_directory({'project_root': str(root)}, index_id) / 'index.json'
    trace = run / 'enrichment/online'
    index = read_json(index_path)
    if index['schema_version'] != 3 or index['dataset_id'] != source_run:
        raise ValueError('Requires a source-run schema 3 online index')
    if sha256(trace / 'receipt.json') != index['source_receipt_sha256']:
        raise ValueError('Online index does not bind current immutable trace receipt')
    trace_receipt = read_json(trace / 'receipt.json')
    for name in ('index.json', 'requests.json'):
        if sha256(trace / name) != trace_receipt['files'][name]:
            raise ValueError('Finalized trace checksum mismatch')
    requests = read_json(trace / 'requests.json')
    catalog = RunEvidence(root, source_run).read()
    selected_scope = select_run(root, source_run, rollout_ids, rollout_list)
    wanted = set(selected_scope["rollout_ids"])
    chosen = {rid: r for rid, r in catalog['rollouts'].items()
              if rid in wanted and r['evaluation_status'] == 'completed'
              and r['resolved'] is True and not r['provenance'].get('quarantined')}
    if not chosen:
        raise ValueError('No resolved trajectories in the specified producer run')
    destination = root / 'difficulty/datasets' / dataset_id
    with file_lock(root / 'state' / f'resolved-selection-{dataset_id}.lock', nonblocking=True):
        if destination.exists():
            raise FileExistsError(destination)
        rollouts, samples, excluded, steps_by_id = {}, [], [], {}
        source_files = {str(index_path): sha256(index_path),
                        str(trace / 'receipt.json'): sha256(trace / 'receipt.json'),
                        str(trace / 'requests.json'): sha256(trace / 'requests.json')}
        snapshot = run / 'resolved-config.json'
        if snapshot.is_file():
            source_files[str(snapshot)] = sha256(snapshot)
        for rid, row in sorted(chosen.items()):
            task = catalog['tasks'][row['task_key']]
            path = owned_path(root, row['trajectory_path'])
            if sha256(path) != row['trajectory_sha256']:
                raise ValueError('Resolved canonical trajectory checksum mismatch')
            trajectory = read_json(path)
            if trajectory.get('info', {}).get('exit_status') != 'Submitted':
                raise ValueError('Resolved source is not a submitted canonical trajectory')
            steps = [i for i, message in enumerate(trajectory['messages']) if message.get('role') == 'assistant']
            steps_by_id[rid] = steps
            evaluation = catalog['evaluations'][row['selected_evaluation_id']]
            outcome = evaluation['outcomes'][rid]
            report = owned_path(root, evaluation['report_path'])
            if (outcome['status'] != 'completed' or outcome['resolved'] is not True
                    or outcome['patch_sha256'] != row['patch_sha256']
                    or sha256(report) != evaluation['report_sha256']):
                raise ValueError('Resolved evaluation must bind the exact patch and report')
            source_files[str(report)] = evaluation['report_sha256']
            rollouts[rid] = {**row, 'rollout_id': rid, 'repo': task['repo'],
                'instance_id': task['instance_id'], 'base_commit': task['base_commit'],
                'canonical_exit_status': 'Submitted', 'canonical_valid_assistant_steps': len(steps),
                'source_kind': 'online_close', 'hidden_root': index['hidden_root'],
                'source_files': {str(report): evaluation['report_sha256']}}
        for event in index['samples']:
            rid = event['rollout_id']
            if rid not in catalog['rollouts']:
                raise ValueError('Online index contains a trajectory from another source run')
            if rid not in chosen:
                excluded.append({'sample_id': event['sample_id'], 'reason': 'not_resolved'})
                continue
            row = rollouts[rid]
            if event['selected_evaluation_id'] != row['selected_evaluation_id']:
                raise ValueError('Online index evaluation is stale; rebuild under a new index ID')
            if event['trajectory_sha256'] != row['trajectory_sha256']:
                raise ValueError('Online event/canonical identity mismatch')
            if (event.get('censored') or event['labels'].get('remaining_steps') is None
                    or event['labels'].get('remaining_reasoning_tokens') is None):
                excluded.append({'sample_id': event['sample_id'], 'reason': 'censored_or_missing_labels'})
                continue
            request = requests[event['request_id']]
            if request.get('is_action_response') is not True:
                excluded.append({'sample_id': event['sample_id'], 'reason': 'not_action_response'})
                continue
            steps = steps_by_id[rid]
            if event['message_index'] not in steps:
                raise ValueError('Action sample does not bind a canonical assistant message')
            step = steps.index(event['message_index'])
            if event['remaining_steps'] != len(steps) - step:
                raise ValueError('Remaining-step label differs from canonical action suffix')
            if event['anchor'] != 'think_close' or event['alignment_status'] != 'exact':
                raise ValueError('Requires exact action-close hidden evidence')
            samples.append({**event, 'repo': row['repo'], 'instance_id': row['instance_id'],
                'source_kind': row['source_kind'], 'task_group': row['repo'] + '::' + row['instance_id'],
                'agent_step': step, 'occurrence': 0, 'endpoint': 'submission',
                'submission_status': 'submitted', 'both_targets_available': True,
                'hidden_root': index['hidden_root'], 'hidden_sha256': event['sha256'],
                'remaining_steps_to_submission': event['labels']['remaining_steps'],
                'remaining_reasoning_tokens_to_submission': event['labels']['remaining_reasoning_tokens']})
        used = {s['rollout_id'] for s in samples}
        omitted = sorted(set(chosen) - used)
        selected = [row for rid, row in sorted(rollouts.items()) if rid in used]
        if not samples or not selected:
            raise ValueError('No usable action-close samples')
        destination.mkdir(parents=True)
        write_json(destination / 'rollouts.json', {'rollouts': selected})
        for name, values in [('samples.jsonl', samples), ('excluded.jsonl', excluded)]:
            with (destination / name).open('w', encoding='utf-8') as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        write_json(destination / 'source-evidence.json', {'source_run_id': source_run,
            'evidence_revision': catalog['revision'], 'source_files': source_files,
            'source_index': str(index_path), 'source_index_sha256': sha256(index_path),
            'resolved_rollouts': sorted(chosen), 'omitted_no_usable_samples': omitted,
            'endpoint': 'successful final submission of a harness-resolved canonical trajectory'})
        summary = {'dataset_id': dataset_id, 'resolved_rollouts': len(chosen),
            'rollouts': len(selected), 'tasks': len({r['task_key'] for r in selected}),
            'samples': len(samples), 'excluded': dict(Counter(s['reason'] for s in excluded))}
        write_json(destination / 'summary.json', summary)
        write_json(destination / 'dataset.json', {
            'schema_version': 1, 'dataset_id': dataset_id, 'kind': 'submission_training_selection',
            'source_run_ids': [source_run], 'eligibility': 'resolved_only',
            'feature_schema': 'public-hidden-last-token-v1', 'anchor': 'think_close',
            'label_fields': ['remaining_steps_to_submission', 'remaining_reasoning_tokens_to_submission'],
        })
        write_json(destination / 'receipt.json', {'schema_version': 1, 'state': 'complete',
            'eligibility': 'resolved_only', 'source_run_id': source_run,
            'created_at': now(), 'files': {p.name: sha256(p) for p in sorted(destination.iterdir())
                                        if p.is_file()}})
        return {**summary, 'path': str(destination),
                'receipt_sha256': sha256(destination / 'receipt.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--source-run', required=True)
    p.add_argument('--index-id', required=True)
    p.add_argument('--dataset-id', required=True)
    p.add_argument('--rollout-id', action='append', default=[])
    p.add_argument('--rollout-list', type=Path)
    a = p.parse_args()
    print(json.dumps(build_resolved_selection(a.root, a.source_run, a.index_id, a.dataset_id, a.rollout_id, a.rollout_list), indent=2))


if __name__ == '__main__':
    main()
