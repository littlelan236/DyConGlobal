"""Derive deployment training sources from actual fit samples, not dataset names."""
import json
from pathlib import Path
from core.io import hash_json, read_json, run_path, sha256


def fit_sources(root, dataset_id, split_id, scope):
    directory = Path(root) / 'difficulty/datasets' / dataset_id
    split_path = directory / 'splits' / split_id / 'split.json'
    split = read_json(split_path)
    roles = ['train', 'validation'] if scope == 'train_validation' else ['train']
    selected = {sid for role in roles for sid in split['sample_ids'].get(role, [])}
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines() if line]
    rows = {row['rollout_id']: row for row in read_json(directory / 'rollouts.json')['rollouts']}
    fitted = [row for row in samples if row['sample_id'] in selected]
    tokenizers=sorted({row['tokenizer_fingerprint'] for row in fitted if row.get('tokenizer_fingerprint')})
    if len(tokenizers)>1:raise ValueError('Fit sources have incompatible tokenizer contracts')
    if {row['sample_id'] for row in fitted} != selected:
        raise ValueError('Fit provenance references unknown training samples')
    sources = [rows[rid] for rid in sorted({row['rollout_id'] for row in fitted})]
    profiles=[]
    for run in sorted({r['run_id'] for r in sources if r.get('run_id')}):
        path=run_path(root,run)/'resolved-config.json'
        snapshot=read_json(path) if path.is_file() else {}
        context=snapshot.get('execution_context') or {}
        record={'source_run_id':run,'runtime_profile_certified':bool(context.get('profile_sha256'))}
        if context.get('profile_sha256'):
            record.update(profile_sha256=context['profile_sha256'],
                config_sha256=sha256(path),generation=snapshot['generation'],
                benchmark=context['benchmark'],agent=context['agent'])
        profiles.append(record)
    return {'schema_version': 1, 'fit_scope': scope, 'dataset_id': dataset_id, 'split_id': split_id,
        'dataset_receipt_sha256': sha256(directory / 'receipt.json'), 'split_sha256': sha256(split_path),
        'fit_sample_count': len(fitted), 'fit_sample_ids_sha256': hash_json(sorted(selected)),
        'source_benchmarks': sorted({row.get('benchmark', 'swebench') for row in sources}),
        'source_agents': sorted({row.get('agent', 'unknown') for row in sources}),
        'source_run_ids': sorted({row['run_id'] for row in sources if row.get('run_id')}),
        'source_profiles':profiles,
        'source_rollout_ids': [row['rollout_id'] for row in sources],
        'model_fingerprints': sorted({row['model_fingerprint'] for row in sources}),
        'tokenizer_fingerprints':tokenizers,
        'evaluation_subjects': [{'rollout_id': row['rollout_id'],
            'subject_sha256': row.get('subject_sha256') or row.get('patch_sha256'),
            'evaluation_id': row.get('selected_evaluation_id')} for row in sources],
        'split_strategy': split['strategy']}
