"""State preserving general configuration revisions and single-to-two migration."""
import copy
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from core.io import hash_json, read_json, sha256
from RL.config import ConfigMigration, schedule_contract, attempt_directory
from RL.config_migration import schedule_matches, contract_matches, apply_optimizer_config


def authorize(cfg, source):
    cfg.run.start.config_migration = ConfigMigration(
        source_schedule_sha256=hash_json(source), target_schedule_sha256=hash_json(schedule_contract(cfg)),
        reason='explicit test configuration revision')


def test_general_revision_is_bound_to_both_schedules_and_preserves_backend_checks(tmp_path):
    from tests.test_grpo_prepared import config
    cfg = config(tmp_path)
    source = schedule_contract(cfg)
    cfg.training.learning_rate *= .5
    cfg.training.grad_clip = .3
    cfg.training.memory.host_budget_gib = 7.0
    cfg.grpo.kl_beta = .02
    assert not schedule_matches(cfg, source)
    authorize(cfg, source)
    assert schedule_matches(cfg, source)
    expected = {'schedule':source, 'adapter':{'rank':4}, 'attention':{'backend':'original'}}
    actual = {**expected, 'schedule':schedule_contract(cfg)}
    assert contract_matches(cfg, expected, actual)
    assert not contract_matches(cfg, expected, {**actual,'attention':{'backend':'changed'}})
    cfg.training.grad_clip = .4
    with pytest.raises(ValueError,match='hash mismatch'):
        schedule_matches(cfg, source)


def test_revision_does_not_reinterpret_frozen_cursors_or_advantages(tmp_path):
    from tests.test_grpo_prepared import config
    cfg = config(tmp_path)
    source = schedule_contract(cfg)
    cfg.grpo.group_batch_size *= 2
    authorize(cfg, source)
    with pytest.raises(ValueError, match='frozen plan'):
        schedule_matches(cfg,source)


def test_new_optimizer_settings_apply_after_moment_restore_without_reset(tmp_path):
    from tests.test_grpo_prepared import config
    from RL.training import optimizer_and_scheduler
    cfg = config(tmp_path)
    model = torch.nn.Linear(2,1)
    opt, sched = optimizer_and_scheduler(model,cfg.training)
    model(torch.ones(1,2)).sum().backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    moments = copy.deepcopy(opt.state_dict()['state'])
    old_epoch = sched.last_epoch
    source = schedule_contract(cfg)
    cfg.training.learning_rate = .001
    cfg.training.weight_decay = .2
    authorize(cfg,source)
    apply_optimizer_config(cfg,opt,sched)
    assert opt.param_groups[0]['lr'] == .001
    assert opt.param_groups[0]['weight_decay'] == .2
    assert sched.last_epoch == old_epoch
    for key, state in moments.items():
        for name, value in state.items():
            torch.testing.assert_close(opt.state_dict()['state'][key][name],value,atol=0,rtol=0)


@pytest.mark.parametrize('streaming', [False, True])
def test_single_worker_checkpoint_to_two_worker_service_preserves_training(tmp_path, streaming):
    from tests.test_grpo_prepared import config, dataset, train
    from tests.test_grpo_topology import _service_worker
    from tests.test_grpo_lifecycle import assert_tree_close
    from RL.checkpoints import verify_checkpoint
    dataset(tmp_path,'a',range(5),branch_count=3)
    cfg = config(tmp_path,updates=3)
    cfg.preflight.verify_identity = False
    cfg.preflight.audit_native_tokens = False
    cfg.training.checkpoint_every_updates = 1
    cfg.runtime.data_preparation_async = streaming
    if streaming:
        from tests.test_grpo_external_reference import make_source, select
        def native_rows(manifest, rows, path):
            for row in rows:
                row['native_request_id'] = row['branch_sample_id']
        make_source(tmp_path/'probabilities',cfg,mutate=native_rows)
        select(cfg,tmp_path/'probabilities')
    oracle,_,_,_ = train(cfg,'oracle')
    source,_,_,_ = train(cfg,'single',stop=1)
    cp = source/'checkpoints/step-000001'
    original = {str(p):sha256(p) for p in cp.rglob('*') if p.is_file()}
    target = cfg.model_copy(deep=True)
    target.run.attempt_id = 'single-to-two'
    target.run.start.mode = 'resume'
    target.run.start.checkpoint = str(cp)
    target.runtime.gpu = None
    target.runtime.gpus = [0,1]
    target.runtime.topology_policy = 'preserve_updates'
    target.validation.stop_after_update = 2
    target.training.grad_clip = .9
    target.training.memory.host_budget_gib = 7.0
    authorize(target,read_json(cp/'state.json')['contract']['schedule'])
    mp.spawn(_service_worker,args=(2,target,str(tmp_path/'single-two-store')),nprocs=2,join=True)
    intermediate = attempt_directory(target)/'checkpoints/step-000002'
    verify_checkpoint(intermediate)
    continued = target.model_copy(deep=True)
    continued.run.attempt_id = 'two-to-two-after-revision'
    continued.run.start.checkpoint = str(intermediate)
    continued.run.start.config_migration = None
    continued.validation.stop_after_update = None
    mp.spawn(_service_worker,args=(2,continued,str(tmp_path/'two-two-store')),nprocs=2,join=True)
    final = attempt_directory(continued)/'checkpoints/step-000003'
    verify_checkpoint(final)
    got = torch.load(final/'training-state.pt',weights_only=True)
    want = torch.load(oracle/'checkpoints/step-000003/training-state.pt',weights_only=True)
    assert got['metadata']['state'] == want['metadata']['state']
    assert len(got['rank_rng']) == 2
    assert got['scheduler'] == want['scheduler']
    # With this tiny input the revised clipping threshold is inactive.
    assert_tree_close(got['optimizer'],want['optimizer'])
    audit = read_json(intermediate/'config-transition.json')
    assert audit['source_receipt_sha256'] == sha256(cp/'receipt.json')
    assert audit['source_state']['global_step'] == 1
    assert {r['path'] for r in audit['schedule_changes']} >= {'training.grad_clip','training.memory.host_budget_gib'}
    assert all(sha256(p) == digest for p,digest in original.items())
