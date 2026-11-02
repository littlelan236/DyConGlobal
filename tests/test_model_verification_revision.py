import copy
import os
import pytest
from application.execution.capture_budget_revision import revise_model_verification, revise_capture_budget, CONCURRENCY64_VALUES
from application.execution.run_plan import load_plan
from core.io import read_json, sha256, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run

@pytest.mark.parametrize('flag', [False, True])
def test_only_identity_flag_changes_and_progress_is_preserved(cfg, flag):
    run=make_run(cfg)
    before=read_json(run/'resolved-config.json')
    protected={n:sha256(run/n) for n in ['selection.json','tasks.json','rollout/work-items.json','evidence.json'] if (run/n).exists()}
    args=dict(revision_id='identity',verify_identity=flag,reason='User requested flag change')
    receipt=revise_model_verification(cfg['project_root'],run.name,**args)
    after=load_plan(cfg['project_root'],run.name)['config']
    assert after['model']['verify_identity'] is flag
    normalized=copy.deepcopy(after)
    normalized['model']=before['model']; normalized['config_hash']=before['config_hash']
    assert normalized==before
    assert all(sha256(run/n)==h for n,h in protected.items())
    assert receipt['changed_config_fields']==['model.verify_identity']
    assert read_json(run/'config-revisions/identity/before/resolved-config.json')==before
    assert read_json(run/'events/config-revision-identity.json')['event']=='run.model_verification_revised'
    assert revise_model_verification(cfg['project_root'],run.name,**args)==receipt

@pytest.mark.parametrize('active',[True,False])
def test_active_or_unclean_execution_is_rejected(cfg,active):
    run=make_run(cfg); p=run/'execution/attempts/001'; p.mkdir()
    write_json(run/'execution/latest.json',{'attempt':'001'})
    if active: write_json(p/'process.json',identity(os.getpid()))
    else:
        write_json(p/'exit.json',{'exit_code':-15})
        write_json(p/'termination.json',{'cleanup_complete':False})
    before=sha256(run/'resolved-config.json')
    with pytest.raises(RuntimeError,match='still active|cleanup is incomplete'):
        revise_model_verification(cfg['project_root'],run.name,revision_id='identity',verify_identity=False,reason='User requested flag change')
    assert sha256(run/'resolved-config.json')==before

@pytest.mark.parametrize('flag',['false',0,1,None])
def test_non_boolean_rejected(cfg,flag):
    with pytest.raises(ValueError,match='boolean'):
        revise_model_verification(cfg['project_root'],'budget-test',revision_id='identity',verify_identity=flag,reason='User requested flag change')

def test_mixed_revision_modes_rejected(cfg):
    with pytest.raises(ValueError,match='cannot mix'):
        revise_capture_budget(cfg['project_root'],'budget-test',revision_id='identity',model_verify_identity=False,runtime_values=CONCURRENCY64_VALUES,reason='User requested flag change')
