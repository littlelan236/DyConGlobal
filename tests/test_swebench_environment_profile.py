"""Opt-in repair identity, historical isolation, and real nose lifecycle checks."""
import hashlib
import json

import pytest

from runtime.containers.repairs import environment_overrides


def profile_cfg(tmp_path, *, requirements=None):
    profile = {'schema_version': 1, 'policy_id': 'control-v1', 'tasks': {
        'pydicom__pydicom-1031': {'requirements': requirements or ['pytest==7.4.4']}}}
    path=tmp_path/'runtime/containers/repair_profiles/control-v1.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    return {'project_root':str(tmp_path),'environment':{'repo_pins':{},'variables':{
        'DYCON_SWEBENCH_REPAIR_POLICY':'control-v1',
        'DYCON_SWEBENCH_REPAIR_POLICY_SHA256':hashlib.sha256(path.read_bytes()).hexdigest()}}},path


def test_profile_requires_exact_frozen_content(tmp_path):
    cfg,path=profile_cfg(tmp_path)
    task={'instance_id':'pydicom__pydicom-1031','repo':'pydicom/pydicom'}
    assert environment_overrides(cfg,task)['repo_pins'][task['repo']]==['pytest==7.4.4']
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):
        environment_overrides(cfg,task)


def test_legacy_and_unselected_task_policies_unchanged(tmp_path):
    cfg,_=profile_cfg(tmp_path)
    task={'instance_id':'pydicom__pydicom-1031','repo':'pydicom/pydicom'}
    cfg['environment']['variables']={}
    assert environment_overrides(cfg,task)=={}
    cfg,_=profile_cfg(tmp_path)
    task['instance_id']='pydicom__pydicom-unselected'
    assert environment_overrides(cfg,task)=={}


@pytest.mark.parametrize('missing',['DYCON_SWEBENCH_REPAIR_POLICY','DYCON_SWEBENCH_REPAIR_POLICY_SHA256'])
def test_half_bound_profile_fails_closed(tmp_path,missing):
    cfg,_=profile_cfg(tmp_path)
    del cfg['environment']['variables'][missing]
    with pytest.raises(ValueError,match='both'):
        environment_overrides(cfg,{'instance_id':'pydicom__pydicom-1031','repo':'pydicom/pydicom'})


def test_profile_retains_unrelated_task_dependencies(tmp_path):
    cfg,_=profile_cfg(tmp_path)
    cfg['environment']['repo_pins']={'pydicom/pydicom':['numpy==1.24.4','pytest==8.3.5']}
    out=environment_overrides(cfg,{'instance_id':'pydicom__pydicom-1031','repo':'pydicom/pydicom'})
    assert out['repo_pins']['pydicom/pydicom']==['numpy==1.24.4','pytest==7.4.4']
    assert cfg['environment']['repo_pins']['pydicom/pydicom'][-1]=='pytest==8.3.5'
