import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from core.io import read_json,sha256
from difficulty.artifacts import verify
from difficulty.evaluation.holdout import run,child_id
from difficulty.evaluation.accepted_test import distribution,validate_holdout
from difficulty.storage import storage_scope,difficulty_path,receipt_parent
from tests.test_ridge_holdout import config_file
from tests.test_ridge_probe import fixture_project  # noqa: F401


def test_external_holdout_fit_and_evaluation_share_original_models(fixture_project,tmp_path):
    cfg,_,dataset=fixture_project
    root=Path(cfg['project_root'])
    external=tmp_path/'data-disk'
    moved=external/'difficulty/datasets'/dataset.name
    moved.parent.mkdir(parents=True)
    shutil.move(str(dataset),str(moved))
    path=config_file(cfg,'external-workflow',layers=[0])
    from core.config import assert_frozen
    from core.io import hash_json
    cfg={**cfg,'paths':{**cfg.get('paths',{}),'artifact_roots':{
        **cfg.get('paths',{}).get('artifact_roots',{}),'difficulty':str(external),'difficulty_cache':str(external)}}}
    cfg['config_hash']=hash_json({k:v for k,v in cfg.items() if k not in {'config_hash','selection','selection_sha256'}})
    assert_frozen(cfg)
    summary=run(cfg,'external-fit',workflow_config=path)
    assert summary['status']=='complete'
    with storage_scope(root,external):
        assert verify(root,'difficulty-evaluation','external-fit')['verified']
        assert difficulty_path(root,'datasets','fixture')==moved
        assert receipt_parent(root,str(moved))==moved
        from difficulty.datasets.identity import task_group
        from difficulty.evaluation.accepted_test import run as evaluate_test
        split=read_json(moved/'splits'/summary['split_id']/'split.json')
        rows=read_json(moved/'rollouts.json')['rollouts']
        manifest=tmp_path/'all-task-roles.json'
        manifest.write_text(json.dumps({'task_assignments':[{
            'task_key':row['task_key'],'partition':split['outer']['task_assignments'][task_group(row)]
        } for row in rows]}))
        result=evaluate_test(cfg,'accepted-test',attempt_id=summary['attempt_id'],dataset_id='fixture',task_partitions=manifest)
        assert result['train_test_task_overlap']==0
        metrics=read_json(external/'difficulty/evaluations/external-fit/metrics.json')
        assert result['accepted']['pooled']==metrics[result['target']]['test_transformed']['pooled']
        assert (external/'difficulty/evaluations/accepted-test/receipt.json').exists()
    assert not (root/'difficulty/evaluations/external-fit').exists()
    assert not (root/'difficulty/checkpoints'/child_id('external-fit','ridge')).exists()
    assert difficulty_path(root,'evaluations','fresh')==root/'difficulty/evaluations/fresh'


def test_trajectory_quantiles_and_task_leakage_rejection():
    stats=distribution([{'r2':-1.0},{'r2':0.0},{'r2':1.0},{'r2':None}],[99,95,90,75,50,25])
    assert stats['mean']==0 and stats['median']==stats['p50']==0
    assert stats['p25']==-.5 and stats['p99']==pytest.approx(.98)
    assert stats['undefined_r2_count']==1
    rows=[{'task_key':'task1','benchmark':'skillsbench','revision':'v','instance_id':'a'},
          {'task_key':'task2','benchmark':'skillsbench','revision':'v','instance_id':'b'}]
    from difficulty.datasets.identity import task_group
    manifest=[{'task_key':'task1','partition':'train'},{'task_key':'task2','partition':'test'}]
    assert validate_holdout(rows,manifest,{task_group(rows[0])})=={task_group(rows[1])}
    with pytest.raises(ValueError,match='leakage'):
        validate_holdout(rows,manifest,{task_group(rows[1])})
