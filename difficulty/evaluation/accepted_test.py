"""Evaluate a frozen Ridge checkpoint on strictly held-out accepted task trajectories."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.config import assert_frozen
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.identity import task_group
from difficulty.datasets.submission import load_selection, prepare_cache
from difficulty.evaluation.metrics import paired_regression_metrics
from difficulty.models.estimators import transform_target
from difficulty.models.ridge_grid import load_model, predict_model
from difficulty.receipts import verify_receipt
from difficulty.storage import difficulty_path, scoped_storage


def distribution(reports, percentiles):
    values=np.array([m['r2'] for m in reports if m.get('r2') is not None and np.isfinite(m['r2'])],float)
    return {'trajectory_count':len(reports),'defined_r2_count':len(values),
            'undefined_r2_count':len(reports)-len(values),
            'mean':float(values.mean()) if len(values) else None,
            'median':float(np.median(values)) if len(values) else None,
            **{f'p{p}':float(np.percentile(values,p,method='linear')) if len(values) else None for p in percentiles}}


def validate_holdout(rollouts, assignments, fitted_groups):
    if len({r['task_key'] for r in assignments})!=len(assignments):
        raise ValueError('Duplicate task assignment')
    roles={r['task_key']:r['partition'] for r in assignments}
    if set(roles.values())!={'train','test'}:
        raise ValueError('Explicit train/test task assignments required')
    if any(r['task_key'] not in roles for r in rollouts):
        raise ValueError('Evaluation dataset tasks missing from task manifest')
    group_roles={}
    for row in rollouts:
        group=task_group(row);role=roles[row['task_key']]
        if group in group_roles and group_roles[group]!=role:
            raise ValueError('Logical task assigned to train and test')
        group_roles[group]=role
    test={g for g,role in group_roles.items() if role=='test'}
    if not test or test.intersection(fitted_groups):
        raise ValueError('Task leakage: evaluation test tasks overlap checkpoint fit tasks')
    return test


@scoped_storage
@exclusive_artifact('difficulty-evaluation','evaluation_id')
def run(cfg,evaluation_id,*,attempt_id,dataset_id,task_partitions,target='submission_remaining_reasoning_tokens_log1p'):
    assert_frozen(cfg)
    root=Path(cfg['project_root'])
    for value in [evaluation_id,attempt_id,dataset_id]:identifier(value)
    output=difficulty_path(root,'evaluations',evaluation_id)
    if output.exists():raise FileExistsError(output)
    attempt_dir=difficulty_path(root,'checkpoints',attempt_id)
    from difficulty.artifacts import verify
    verify(root,'difficulty-attempt',attempt_id)
    attempt=read_json(attempt_dir/'attempt.json')
    if attempt['mode']!='fit' or target not in attempt['settings']['targets']:
        raise ValueError('Requires fitted checkpoint with requested target')
    train_dataset=difficulty_path(root,'datasets',attempt['dataset_id'])
    split_dir=train_dataset/'splits'/attempt['split_id']
    split=read_json(split_dir/'split.json')
    if split['strategy']!='task_holdout_event_v1':raise ValueError('Requires frozen outer task holdout checkpoint')
    fitted_groups={g for g,role in split['outer']['task_assignments'].items() if role=='train'}
    source=difficulty_path(root,'datasets',dataset_id)
    verify_receipt(root,source)
    rows=read_json(source/'rollouts.json')['rollouts']
    count=sum(bool(line.strip()) for line in (source/'samples.jsonl').read_text().splitlines())
    settings=SimpleNamespace(dataset_id=dataset_id,dataset_receipt_sha256=sha256(source/'receipt.json'),
        expected_tasks=len({task_group(r) for r in rows}),expected_rollouts=len(rows),expected_samples=count)
    rollouts,samples,evidence=load_selection(cfg,settings)
    manifest=Path(task_partitions).resolve()
    assignments=read_json(manifest)['task_assignments']
    test_groups=validate_holdout(rollouts,assignments,fitted_groups)
    # Compare every nominal assignment with the checkpoint's original outer task roles.
    mapped={r['task_key']:r['partition'] for r in assignments}
    train_rows=read_json(train_dataset/'rollouts.json')['rollouts']
    for row in train_rows:
        if mapped.get(row['task_key'])!=split['outer']['task_assignments'][task_group(row)]:
            raise ValueError('Requested task partition differs from checkpoint training partition')
    contract=evidence['feature_contract']
    if contract!=attempt['input']['feature_contract']:
        raise ValueError('Test hidden feature contract differs from checkpoint')
    selected=read_json(attempt_dir/'selected.json')['targets'][target]
    model_rel=read_json(attempt_dir/'metrics.json')[target]['refit_model']
    model_path=attempt_dir/model_rel
    model=load_model(model_path)
    model_digest=sha256(model_path)
    layer=selected['layer']
    rollout_lookup={r['rollout_id']:r for r in rollouts}
    test_indices=[i for i,s in enumerate(samples) if s['task_group'] in test_groups]
    if not test_indices:raise ValueError('No eligible accepted test events')
    print(f'Accepted test preflight: {len(test_groups)} held-out tasks, {len(test_indices)} events; no fit-task overlap',flush=True)
    # Cache the complete accepted dataset once, reused for every split/checkpoint.
    cache=prepare_cache(cfg,settings,samples)
    matrix=np.load(cache/f'layer-{layer:02d}.f32.npy',mmap_mode='r',allow_pickle=False)
    scores=np.concatenate([predict_model(model,np.asarray(matrix[test_indices[i:i+512]],dtype=float))
                           for i in range(0,len(test_indices),512)])
    labels=np.array([samples[i]['remaining_reasoning_tokens_to_submission'] for i in test_indices],float)
    truth=transform_target(labels,target)
    if not np.isfinite(scores).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite test pairs')
    by_rollout=defaultdict(list)
    for j,i in enumerate(test_indices):by_rollout[samples[i]['rollout_id']].append(j)
    trajectory_metrics=[]
    for rid,indices in sorted(by_rollout.items()):
        row=rollout_lookup[rid]
        trajectory_metrics.append({'rollout_id':rid,'task_key':row['task_key'],'instance_id':row['instance_id'],
            'resolved':row.get('resolved'),'metrics':paired_regression_metrics(truth[indices],scores[indices])})
    accepted=[r['metrics'] for r in trajectory_metrics]
    resolved=[r['metrics'] for r in trajectory_metrics if r['resolved'] is True]
    resolved_mask=[j for j,i in enumerate(test_indices) if rollout_lookup[samples[i]['rollout_id']].get('resolved') is True]
    summary={'evaluation_id':evaluation_id,'attempt_id':attempt_id,'dataset_id':dataset_id,
        'target':target,'space':'log1p','truth_endpoint':'observed_agent_terminal',
        'test_task_count':len(test_groups),'train_test_task_overlap':0,
        'layer':layer,'alpha':selected['alpha'],
        'accepted':{'trajectory_r2':distribution(accepted,[99,95,90,75,50,25]),
                    'pooled':paired_regression_metrics(truth,scores)},
        'resolved':{'trajectory_r2':distribution(resolved,[99,90,75,50,25]),
                    'pooled':paired_regression_metrics(truth[resolved_mask],scores[resolved_mask])}}
    output.mkdir(parents=True)
    write_json(output/'summary.json',summary)
    write_json(output/'by-rollout.json',{'trajectories':trajectory_metrics})
    write_json(output/'task-isolation.json',{'fit_task_groups':sorted(fitted_groups),'test_task_groups':sorted(test_groups),
        'overlap':sorted(fitted_groups.intersection(test_groups)),
        'task_manifest_sha256':sha256(manifest),'test_sample_order_sha256':hash_json([samples[i]['sample_id'] for i in test_indices])})
    with (output/'predictions.jsonl').open('w') as f:
        for j,i in enumerate(test_indices):
            s=samples[i]
            f.write(json.dumps({'sample_id':s['sample_id'],'rollout_id':s['rollout_id'],'task_group':s['task_group'],
                'resolved':rollout_lookup[s['rollout_id']].get('resolved'),
                'target_transformed':float(truth[j]),'prediction_transformed':float(scores[j]),
                'model_sha256':model_digest})+'\n')
    with (output/'by-rollout.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['rollout_id','task_key','instance_id','resolved','count','r2'])
        writer.writeheader()
        for r in trajectory_metrics:writer.writerow({k:r[k] for k in ['rollout_id','task_key','instance_id','resolved']}|{k:r['metrics'][k] for k in ['count','r2']})
    write_json(output/'receipt.json',{'schema_version':1,'state':'complete','kind':'ridge-accepted-test-v1',
        'created_at':now(),'evaluation_id':evaluation_id,'attempt_id':attempt_id,'dataset_id':dataset_id,
        'source_files':{str(manifest):sha256(manifest),str(model_path):sha256(model_path)},
        'parents':[{'path':str(p),'receipt_sha256':sha256(p/'receipt.json')} for p in [source,attempt_dir,split_dir]],
        'files':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='receipt.json'}})
    verify_receipt(root,output)
    return summary


def register(sub):
    p=sub.add_parser('difficulty-ridge-test',help='Evaluate frozen Ridge checkpoints on isolated accepted test tasks')
    p.add_argument('evaluation_id')
    p.add_argument('--attempt-id',required=True)
    p.add_argument('--dataset-id',required=True)
    p.add_argument('--task-partitions',required=True,type=Path)
