"""Freeze resolved canonical action-close samples for the existing Ridge workflow.

For a resolved trajectory the successful endpoint is its final submission. The
existing submission-remainder field names therefore retain their exact values;
eligibility is additionally bound to a completed, exact-patch resolved outcome.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from application.execution.evidence import RunEvidence
from application.execution.subject import subject_digest, verify_subject
from core.io import artifact_reference, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.datasets.identity import benchmark as row_benchmark
from difficulty.datasets.identity import task_descriptor
from difficulty.datasets.paths import dataset_directory
from difficulty.datasets.selection import select_run
from difficulty.datasets.eligibility import eligible_evaluation, validate_policy
from enrichment.online_trace import publication_directory, verify_publication


def trajectory_fraction_labels(*, total_steps, step, remaining_reasoning_tokens, total_reasoning_tokens):
    """Return causal remaining-work fractions for one close sample."""
    if type(total_steps) is not int or total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError('Invalid canonical step denominator')
    if type(total_reasoning_tokens) is not int or total_reasoning_tokens <= 0:
        raise ValueError('Reasoning denominator must be positive')
    if type(remaining_reasoning_tokens) not in (int, float) or not 0 <= remaining_reasoning_tokens <= total_reasoning_tokens:
        raise ValueError('Invalid remaining reasoning label')
    return {
        'remaining_steps_fraction': (total_steps - step) / total_steps,
        'remaining_reasoning_tokens_fraction': remaining_reasoning_tokens / total_reasoning_tokens,
    }


def build_resolved_selection(root, source_run, index_id, dataset_id, rollout_ids=(), rollout_list=None,
                             label_normalization='raw', *, eligibility='resolved_only', storage_root=None):
    validate_policy(eligibility)
    root = Path(root).resolve()
    source_run, dataset_id = map(identifier, (source_run, dataset_id))
    if label_normalization not in {'raw', 'trajectory_fraction'}:
        raise ValueError('Unknown label normalization')
    run = root / 'runs' / source_run
    dataset_cfg = {'project_root': str(root)}
    if storage_root is not None:
        dataset_cfg['paths'] = {'artifact_roots': {'difficulty_dataset': str(storage_root)}}
    if eligibility == 'accepted_terminal' and not (rollout_ids or rollout_list):
        raise ValueError('Accepted terminal datasets require an explicit accepted rollout selection')
    index_path = dataset_directory(dataset_cfg, index_id) / 'index.json'
    index = read_json(index_path)
    trace = publication_directory(root, source_run, index.get('publication_id'))
    if owned_path(root, index.get('label_root', artifact_reference(root, trace))) != trace:
        raise ValueError('Online index label root differs from its named publication')
    if index['schema_version'] != 3 or index['dataset_id'] != source_run:
        raise ValueError('Requires a source-run schema 3 online index')
    if sha256(trace / 'receipt.json') != index['source_receipt_sha256']:
        raise ValueError('Online index does not bind current immutable trace receipt')
    trace_receipt = read_json(trace / 'receipt.json')
    if index.get('publication_id'):
        verify_publication(trace, index['hidden_root'])
    for name in ('index.json', 'requests.json'):
        if sha256(trace / name) != trace_receipt['files'][name]:
            raise ValueError('Finalized trace checksum mismatch')
    requests = read_json(trace / 'requests.json')
    catalog = RunEvidence(root, source_run).read()
    selected_scope = select_run(root, source_run, rollout_ids, rollout_list)
    wanted = set(selected_scope["rollout_ids"])
    chosen = {rid: r for rid, r in catalog['rollouts'].items()
              if rid in wanted and eligible_evaluation(r, catalog, rid, eligibility)
              and not r['provenance'].get('quarantined')}
    if not chosen:
        raise ValueError('No resolved trajectories in the specified producer run')
    reasoning_totals = {}
    complete_reasoning = {}
    if label_normalization == 'trajectory_fraction':
        for request in requests.values():
            rid = request.get('rollout_id')
            if rid in chosen:
                blocks = request.get('blocks', [])
                reasoning_totals[rid] = reasoning_totals.get(rid, 0) + sum(
                    max(0, int(block['end']) - int(block['start'])) for block in blocks
                )
                if request.get('is_action_response') is True:
                    complete_reasoning.setdefault(rid, set()).add(request.get('message_index'))
    destination = dataset_directory(dataset_cfg, dataset_id)
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
            workspace_subject = row.get('evaluation_subject', {}).get('kind') == 'workspace'
            web_subject = row.get('evaluation_subject', {}).get('kind') == 'web_trace'
            verify_subject(root, row)
            if web_subject and not read_json(owned_path(root, row['subject_path'])).get('complete'):
                raise ValueError('Resolved WebArena training requires a complete web trace subject')
            expected_terminal = 'AgentCompleted' if workspace_subject else 'Submitted'
            if trajectory.get('info', {}).get('exit_status') != expected_terminal:
                raise ValueError('Resolved source does not have its benchmark terminal state')
            steps = [i for i, message in enumerate(trajectory['messages']) if message.get('role') == 'assistant']
            steps_by_id[rid] = steps
            evaluation = catalog['evaluations'][row['selected_evaluation_id']]
            outcome = evaluation['outcomes'][rid]
            report = owned_path(root, evaluation['report_path'])
            digest_key, _ = subject_digest(row)
            if (not eligible_evaluation(row, catalog, rid, eligibility)
                    or outcome.get(digest_key) != row.get(digest_key)
                    or sha256(report) != evaluation['report_sha256']):
                raise ValueError('Resolved evaluation must bind the exact subject and report')
            source_files[str(report)] = evaluation['report_sha256']
            if eligibility == 'positive_reward':
                from difficulty.datasets.eligibility import verify_positive_report
                verify_positive_report(row, read_json(report))
            elif eligibility == 'accepted_terminal':
                from difficulty.datasets.eligibility import verify_terminal_report
                verify_terminal_report(row, read_json(report))
            descriptor = task_descriptor({**task, **row,
                'benchmark_revision': task.get('revision') or row.get('revision'),
                'task_package_sha256': task.get('instance', {}).get('package_sha256')})
            rollouts[rid] = {**row, **descriptor, 'rollout_id': rid,
                'benchmark': row_benchmark(task), 'agent': row.get('agent', 'unknown'),
                'canonical_exit_status': expected_terminal, 'canonical_valid_assistant_steps': len(steps),
                'source_kind': 'online_close', 'hidden_root': index['hidden_root'],
                'source_files': {str(report): evaluation['report_sha256']},
                'training_eligibility': eligibility,
                'submission_status': 'submitted'}
            if workspace_subject:
                rollouts[rid]['task_package_sha256'] = task['instance']['package_sha256']
            if workspace_subject or web_subject:
                rollouts[rid]['evaluation_subject_sha256'] = row['subject_sha256']
            if label_normalization == 'trajectory_fraction':
                rollouts[rid]['total_reasoning_tokens'] = reasoning_totals.get(rid, 0)
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
            if eligibility in {'positive_reward', 'accepted_terminal'}:
                # Raw finalized suffix labels describe this actual endpoint;
                # success-only online labels intentionally remain null.
                event = {**event, 'labels': {
                    'remaining_steps': event.get('remaining_steps'),
                    'remaining_reasoning_tokens': event.get('remaining_reasoning_tokens_after')}}
            if (event.get('censored') or event['labels'].get('remaining_steps') is None
                    or event['labels'].get('remaining_reasoning_tokens') is None):
                excluded.append({'sample_id': event['sample_id'], 'reason': 'censored_or_missing_labels'})
                continue
            request = requests[event['request_id']]
            if request.get('is_action_response') is not True:
                excluded.append({'sample_id': event['sample_id'], 'reason': 'not_action_response'})
                continue
            steps = steps_by_id[rid]
            message_index = request.get('message_index', event['message_index'])
            if message_index not in steps:
                raise ValueError('Action sample does not bind a canonical assistant message')
            # Capture event indices refer to the generation-time prompt, which can
            # differ after OpenHands condensation. Finalization binds the stable
            # canonical index through the exact native ledger identity.
            if message_index != event['message_index']:
                trajectory = read_json(owned_path(root, row['trajectory_path']))
                response = trajectory['messages'][message_index].get('extra', {}).get('response', {})
                if (not request.get('ledger_ref') or response.get('ledger_ref') != request['ledger_ref']
                        or response.get('ledger_sha256') != request.get('ledger_sha256')
                        or response.get('dycon_native_request_id', response.get('id')) != event['request_id']):
                    raise ValueError('Finalized canonical index lacks exact native identity')
            step = steps.index(message_index)
            if event['remaining_steps'] != len(steps) - step:
                raise ValueError('Remaining-step label differs from canonical action suffix')
            if event['anchor'] != 'think_close' or event['alignment_status'] != 'exact':
                raise ValueError('Requires exact action-close hidden evidence')
            workspace_subject = row.get('evaluation_subject', {}).get('kind') == 'workspace'
            sample = {**event, 'capture_message_index': event['message_index'], 'message_index': message_index, **{k: row[k] for k in ('benchmark', 'agent', 'revision', 'repo', 'instance_id',
                                  'intent_template_id', 'source_task_id', 'task_revision', 'source_task_sha256')
                                  if k in row},
                'source_kind': row['source_kind'], 'task_group': row['task_group'],
                'agent_step': step, 'occurrence': 0,
                'endpoint': 'agent_terminal' if workspace_subject else 'submission',
                'submission_status': 'completed' if workspace_subject else 'submitted',
                'both_targets_available': True,
                'hidden_root': index['hidden_root'], 'hidden_sha256': event['sha256'],
                'remaining_steps_to_submission': event['labels']['remaining_steps'],
                'remaining_reasoning_tokens_to_submission': event['labels']['remaining_reasoning_tokens']}
            if label_normalization == 'trajectory_fraction':
                total = reasoning_totals.get(rid, 0)
                if total <= 0 or complete_reasoning.get(rid, set()) != set(
                        steps_by_id[rid]):
                    excluded.append({'sample_id': event['sample_id'],
                                     'reason': 'incomplete_reasoning_denominator' if total > 0
                                     else 'zero_reasoning_denominator'})
                    continue
                sample.update(trajectory_fraction_labels(
                    total_steps=len(steps), step=step,
                    remaining_reasoning_tokens=event['labels']['remaining_reasoning_tokens'],
                    total_reasoning_tokens=total))
            samples.append(sample)
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
            'eligible_rollouts': sorted(chosen),
            'resolved_rollouts': sorted(rid for rid, row in chosen.items() if row['resolved'] is True),
            'omitted_no_usable_samples': omitted,
            'endpoint': ('successful final submission of a harness-resolved canonical trajectory'
                         if eligibility == 'resolved_only' else ('actual terminal endpoint of an officially evaluated positive-reward canonical trajectory'
                         if eligibility == 'positive_reward' else
                         'actual terminal endpoint of an officially evaluated accepted canonical trajectory'))})
        summary = {'dataset_id': dataset_id, 'resolved_rollouts': sum(row['resolved'] is True for row in chosen.values()),
            'eligible_rollouts': len(chosen), 'eligibility': eligibility,
            'rollouts': len(selected), 'tasks': len({r['task_key'] for r in selected}),
            'samples': len(samples), 'excluded': dict(Counter(s['reason'] for s in excluded))}
        write_json(destination / 'summary.json', summary)
        dataset_meta = {
            'schema_version': 1, 'dataset_id': dataset_id, 'kind': 'submission_training_selection',
            'source_run_ids': [source_run], 'eligibility': eligibility,
            'benchmark': row_benchmark(next(iter(catalog['tasks'].values()))) if catalog['tasks'] else 'unknown',
            'agent': next((r.get('agent', 'unknown') for r in rollouts.values()), 'unknown'),
            'feature_schema': 'public-hidden-last-token-v1', 'anchor': 'think_close',
            'label_fields': ['remaining_steps_to_submission', 'remaining_reasoning_tokens_to_submission'],
        }
        if label_normalization == 'trajectory_fraction':
            dataset_meta.update(label_normalization='trajectory_fraction',
                                label_denominators={'steps': 'canonical_valid_assistant_steps',
                                                    'reasoning_tokens': 'sum(finalized_reasoning_block_lengths)'},
                                normalized_label_fields=['remaining_steps_fraction',
                                                         'remaining_reasoning_tokens_fraction'])
        write_json(destination / 'dataset.json', dataset_meta)
        write_json(destination / 'receipt.json', {'schema_version': 1, 'state': 'complete',
            'eligibility': eligibility, 'source_run_id': source_run,
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
    p.add_argument('--label-normalization', choices=['raw', 'trajectory_fraction'], default='raw')
    p.add_argument('--eligibility', choices=['resolved_only', 'positive_reward', 'accepted_terminal'], default='resolved_only')
    a = p.parse_args()
    print(json.dumps(build_resolved_selection(a.root, a.source_run, a.index_id, a.dataset_id,
                                              a.rollout_id, a.rollout_list, a.label_normalization, eligibility=a.eligibility), indent=2))


if __name__ == '__main__':
    main()
