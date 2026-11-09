import os
from pathlib import Path

import pytest

from application.execution import config_revision as revision
from application.execution.config_revision import parse_updates, revise_config
from application.execution.run_plan import load_plan
from core.io import hash_json, read_json, sha256, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run


def revise(cfg, updates, **kwargs):
    return revise_config(cfg['project_root'], 'budget-test', revision_id='general',
                         updates=updates, reason='fixture', **kwargs)


def test_mixed_fields_share_one_transaction_and_preserve_results(cfg):
    run = make_run(cfg)
    write_json(run / 'evidence.json', {'rollouts': {'old': {'value': 1}}})
    before = {str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file()}
    updates = {'sandbox.memory_mb': 16384, 'generation.temperature': .4,
               'timeouts.model_seconds': 1234, 'agent.step_limit': 222,
               'model.verify_identity': False, 'execution.supervisor_max_open_files': 65536,
               'serving.max_num_seqs': 7, 'capture.receipt_workers': 8}
    preview = revise(cfg, updates, dry_run=True)
    assert preview['dry_run'] and any(d['path'] == '/sandbox/memory_mb' for d in preview['differences'])
    assert before == {str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file()}
    receipt = revise(cfg, updates, expect_config_hash=preview['old_config_hash'])
    current = load_plan(cfg['project_root'], run.name)['config']
    assert current['sandbox']['memory_mb'] == 16384
    assert current['serving']['max_num_seqs'] == 7
    assert current.get('parallel', {}).get('max_inflight_branches', 7) == 7
    assert sha256(run / 'evidence.json') == before['evidence.json']
    assert sha256(run / 'progress.json') == before['progress.json']
    assert sha256(run / 'rollout/work-items.json') == before['rollout/work-items.json']
    assert revise(cfg, updates) == receipt
    assert receipt['new_config_hash'] == preview['new_config_hash']


@pytest.mark.parametrize('updates', [
    {'sandbox.memory_mb': '16384'}, {'sandbox.memory_mb': 0},
    {'generation.not_a_field': 1}, {'execution_context.profile': {}},
    {'parallel.max_inflight_branches': 7},
    {'capture.export.spool_high_watermark': .2, 'capture.export.spool_low_watermark': .8},
    {'sandbox': {}, 'sandbox.memory_mb': 8},
])
def test_invalid_input_never_mutates_snapshots(cfg, updates):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError):
        revise(cfg, updates)
    assert sha256(run / 'resolved-config.json') == before
    assert not (run / 'config-revisions/general').exists()


def test_live_preview_is_allowed_but_apply_is_blocked(cfg):
    run = make_run(cfg)
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(run / 'execution/attempts/001/process.json', identity(os.getpid()))
    assert revise(cfg, {'sandbox.memory_mb': 16384}, dry_run=True)['dry_run']
    with pytest.raises(RuntimeError, match='still active'):
        revise(cfg, {'sandbox.memory_mb': 16384})


def test_seed_replan_archives_results_and_rebuilds_samples(cfg):
    run = make_run(cfg)
    attempt = run / 'rollout/tasks/budget-task/repetitions/rep-000/attempts/rep-000-attempt-005'
    write_json(attempt / 'attempt.json', {'exit_status': 'Submitted', 'sample_seed': 42})
    write_json(run / 'evidence.json', {'rollouts': {'old': {}}})
    old = read_json(run / 'resolved-config.json')
    with pytest.raises(ValueError, match='--replan'):
        revise(cfg, {'generation.seed': 101})
    receipt = revise(cfg, {'generation.seed': 101}, replan=True)
    loaded = load_plan(cfg['project_root'], run.name)
    assert loaded['work_items']['items'][0]['sample_seed'] == 101
    assert loaded['config']['generation']['seed'] == 101
    archive = run / 'config-revisions/general/archived-state'
    assert read_json(archive / 'evidence.json') == {'rollouts': {'old': {}}}
    assert (archive / attempt.relative_to(run)).exists()
    assert not (run / 'evidence.json').exists()
    assert read_json(run / 'rollout/replan-attempt-floors.json') == {'budget-task/rep-000': 5}
    assert read_json(run / 'config-revisions/general/before/resolved-config.json') == old
    assert receipt['identity_changes'] == ['/generation/seed']
    assert all(s['exit_status'] == 'Pending' for s in read_json(run / 'progress.json')['samples'].values())


def test_replan_partition_uses_existing_selection_service(cfg):
    from catalog.store import Catalog
    from tests.conftest import task
    run = make_run(cfg)
    Catalog(cfg['project_root']).register_tasks([task('test-task', partition='test')])
    revise(cfg, {'benchmark.partition': 'test'}, replan=True)
    loaded = load_plan(cfg['project_root'], run.name)
    assert [t['instance_id'] for t in loaded['config']['selection']['tasks'].values()] == ['test-task']


def test_failed_commit_restores_snapshot_and_archived_state(cfg, monkeypatch):
    run = make_run(cfg)
    write_json(run / 'evidence.json', {'original': True})
    before = {name: (run / name).read_bytes() for name in
              ['resolved-config.json', 'plan.json', 'config.yaml', 'progress.json', 'evidence.json']}
    original = revision._put
    injected = False
    def fail_once(path, content):
        nonlocal injected
        if path == run / 'plan.json' and not injected:
            injected = True
            raise OSError('simulated disk failure')
        return original(path, content)
    monkeypatch.setattr(revision, '_put', fail_once)
    with pytest.raises(OSError, match='simulated'):
        revise(cfg, {'generation.seed': 101}, replan=True)
    assert all((run / name).read_bytes() == content for name, content in before.items())
    load_plan(cfg['project_root'], run.name)
    assert read_json(run / 'config-revisions/general/progress.json')['phase'] == 'rolled_back'


def test_patch_supports_optional_sections_lists_maps_and_duplicate_rejection(tmp_path):
    patch = tmp_path / 'patch.yaml'
    patch.write_text('sandbox:\n  memory_mb: 16384\nserving:\n  gpus: [0, 2]\n')
    assert parse_updates(['environment.variables={X: value}'], patch) == {
        'sandbox.memory_mb': 16384, 'serving.gpus': [0, 2], 'environment.variables': {'X': 'value'}}
    with pytest.raises(ValueError, match='Conflicting'):
        parse_updates(['sandbox.memory_mb=8'], patch)
    patch.write_text('sandbox: {}\nsandbox: {}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        parse_updates(patch=patch)


def test_cli_reads_frozen_config_when_root_has_invalid_unrelated_values(cfg, capsys):
    from core.cli import main
    run = make_run(cfg)
    root_config = Path(cfg['project_root']) / 'config.yaml'
    root_config.write_text('unrelated_invalid_config: true\n')
    main(['--config', str(root_config), 'run', 'revise-config', run.name,
          '--revision-id', 'cli', '--set', 'sandbox.memory_mb=16384', '--reason', 'fixture', '--dry-run'])
    assert '"dry_run": true' in capsys.readouterr().out
    assert not (run / 'config-revisions/cli').exists()


@pytest.mark.parametrize('command', ['revise-task-memory', 'revise-capture-budget', 'revise-capture-control',
    'revise-concurrency', 'revise-model-verification', 'revise-wheelhouse',
    'revise-serving-placement', 'revise-serving-memory'])
def test_removed_cli_commands_are_not_accepted(command):
    from core.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['run', command, 'run'])
    assert exc.value.code == 2


def test_preview_hash_prevents_stale_apply(cfg):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError, match='changed since preview'):
        revise(cfg, {'sandbox.memory_mb': 16384}, expect_config_hash='stale')
    assert sha256(run / 'resolved-config.json') == before


def test_replanned_run_executes_with_fresh_seed_and_unique_attempt(cfg, monkeypatch):
    from rollout.runner import run_manifest
    from tests.test_limits_retry import configure, fake_runtime, prepare
    cfg = configure(cfg)
    run, _ = prepare(cfg, run_id='budget-test')
    factory, first_calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    original = run / 'rollout/tasks/a/repetitions/rep-000/attempts/rep-000-attempt-000/attempt.json'
    original_bytes = original.read_bytes()
    revise(cfg, {'generation.seed': 101}, replan=True)
    factory, next_calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    new_cfg = load_plan(cfg['project_root'], run.name)['config']
    result = next(iter(run_manifest(new_cfg, run.name, resume=True, factory=factory).values()))
    assert first_calls[0][1:3] == ('rep-000-attempt-000', 42)
    assert next_calls[0][1:3] == ('rep-000-attempt-001', 101)
    assert result['exit_status'] == 'Submitted'
    stored = run / 'config-revisions/general/archived-state' / original.relative_to(run)
    assert stored.read_bytes() == original_bytes
    revise_config(cfg['project_root'], run.name, revision_id='second',
                  updates={'generation.seed': 102}, replan=True, reason='fixture')
    factory, third_calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    run_manifest(load_plan(cfg['project_root'], run.name)['config'], run.name, resume=True, factory=factory)
    assert third_calls[0][1:3] == ('rep-000-attempt-002', 102)


def test_interrupted_multifile_transaction_can_recover_without_loading_partial_plan(cfg, monkeypatch):
    run = make_run(cfg)
    before = {name: (run / name).read_bytes() for name in ['plan.json', 'resolved-config.json', 'config.yaml']}
    original_put, original_rollback = revision._put, revision._rollback
    def fail_write(path, content):
        if path == run / 'plan.json':
            raise OSError('simulated interruption')
        return original_put(path, content)
    def interrupted_rollback(*args):
        raise OSError('rollback interrupted too')
    monkeypatch.setattr(revision, '_put', fail_write)
    monkeypatch.setattr(revision, '_rollback', interrupted_rollback)
    with pytest.raises(OSError):
        revise(cfg, {'sandbox.memory_mb': 16384})
    with pytest.raises((ValueError, RuntimeError)):
        load_plan(cfg['project_root'], run.name)
    monkeypatch.setattr(revision, '_put', original_put)
    monkeypatch.setattr(revision, '_rollback', original_rollback)
    receipt = revise_config(cfg['project_root'], run.name, revision_id='general',
                            recover=True, reason='recover interrupted fixture')
    assert receipt['recovered']
    assert all((run / name).read_bytes() == value for name, value in before.items())
    load_plan(cfg['project_root'], run.name)


def test_revised_paths_keep_existing_run_location(cfg, tmp_path):
    run = make_run(cfg)
    revise(cfg, {'paths.artifact_roots.run': str(tmp_path / 'new-store')})
    loaded = load_plan(cfg['project_root'], run.name)
    from core.io import run_path
    assert run_path(loaded['config'], run.name) == run.resolve()


def test_optional_section_can_be_added_to_legacy_snapshot(cfg):
    run = make_run(cfg)
    config = read_json(run / 'resolved-config.json')
    config.pop('sandbox', None)
    config['config_hash'] = hash_json({k: v for k, v in config.items()
                                      if k not in {'config_hash', 'selection', 'selection_sha256'}})
    plan = read_json(run / 'plan.json')
    plan['config_hash'] = config['config_hash']
    plan['plan_sha256'] = hash_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
    write_json(run / 'resolved-config.json', config)
    write_json(run / 'plan.json', plan)
    revise(cfg, {'sandbox.memory_mb': 16384})
    assert load_plan(cfg['project_root'], run.name)['config']['sandbox']['memory_mb'] == 16384


@pytest.mark.parametrize('field,value', [('model.served_name', 'new-model'),
    ('model.fingerprint', 'new-fingerprint'), ('parallel.branch_count', 2)])
def test_identity_changes_are_allowed_with_explicit_replanning(cfg, field, value):
    run = make_run(cfg)
    with pytest.raises(ValueError, match='--replan'):
        revise(cfg, {field: value})
    revise(cfg, {field: value}, replan=True)
    loaded = load_plan(cfg['project_root'], run.name)
    section, name = field.split('.')
    assert loaded['config'][section][name] == value


def test_incomplete_cleanup_blocks_recover_and_apply(cfg):
    run = make_run(cfg)
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(run / 'execution/attempts/001/exit.json', {'exit_code': 0})
    write_json(run / 'execution/attempts/001/termination.json', {'cleanup_complete': False})
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        revise(cfg, {'sandbox.memory_mb': 16384})
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        revise_config(cfg['project_root'], run.name, revision_id='general', recover=True, reason='fixture')


def test_incomplete_revision_blocks_load_even_with_valid_hashes(cfg):
    run = make_run(cfg)
    write_json(run / 'config-revisions/interrupted/progress.json', {'phase': 'committing'})
    with pytest.raises(RuntimeError, match='Incomplete configuration revision'):
        load_plan(cfg['project_root'], run.name)


def test_receipt_written_before_interruption_can_finish_recovery(cfg):
    run = make_run(cfg)
    receipt = revise(cfg, {'sandbox.memory_mb': 16384})
    journal = run / 'config-revisions/general/progress.json'
    write_json(journal, {**read_json(journal), 'phase': 'committing'})
    result = revise_config(cfg['project_root'], run.name, revision_id='general', recover=True, reason='fixture')
    assert result['committed']
    assert load_plan(cfg['project_root'], run.name)['config']['config_hash'] == receipt['new_config_hash']


def test_preparation_interruption_does_not_leave_an_unrecoverable_revision(cfg):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    (run / 'config-revisions/general/before').mkdir(parents=True)
    assert revise_config(cfg['project_root'], run.name, revision_id='general',
                         recover=True, reason='fixture')['recovered']
    assert sha256(run / 'resolved-config.json') == before
